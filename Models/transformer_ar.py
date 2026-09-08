"""
Autoregressive Transformer for Discrete Token Modeling (Stage 2)

Based on SDFormer: Similarity-driven Discrete Transformer (NeurIPS 2024).

Decoder-only GPT-style transformer that autoregressively models
the sequence of VQ-VAE discrete tokens.

Features:
- BOS token (index K, distinct from codebook indices {0, ..., K-1})
- Causal self-attention (no peeking at future tokens)
- Random replacement data augmentation during training
- Cross-entropy loss over codebook vocabulary
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - pure PyTorch, no flash_attn."""
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, dim/2)
        return torch.cat([freqs, freqs], dim=-1)  # (T, dim)


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, freqs):
    """Apply rotary embeddings to x. x: (B, nh, T, hd), freqs: (T, hd)"""
    cos = freqs.cos()[None, None, :, :]  # (1, 1, T, hd)
    sin = freqs.sin()[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional RoPE."""
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1,
                 max_len=256, lookback=0, **kwargs):
        super().__init__()
        assert n_embd % n_head == 0

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

        # Causal mask with optional lookback window
        causal = torch.tril(torch.ones(max_len, max_len))
        if lookback > 0:
            band = torch.triu(torch.ones(max_len, max_len), -(lookback - 1))
            causal = causal * band
        self.register_buffer("mask", causal.view(1, 1, max_len, max_len))

        self.rotary = RotaryEmbedding(n_embd // n_head)

    def forward(self, x):
        B, T, C = x.size()
        nh = self.n_head
        hd = C // nh

        k = self.key(x).view(B, T, nh, hd).transpose(1, 2)
        q = self.query(x).view(B, T, nh, hd).transpose(1, 2)
        v = self.value(x).view(B, T, nh, hd).transpose(1, 2)

        freqs = self.rotary(T, x.device)
        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class TransformerBlock(nn.Module):
    """Transformer decoder block with causal self-attention + MLP."""
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1,
                 mlp_hidden_times=4, max_len=256, lookback=0, **kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            max_len=max_len,
            lookback=lookback,
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            nn.GELU(),
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class AutoregressiveTransformer(nn.Module):
    """
    Decoder-only GPT-style Transformer for autoregressive token modeling.

    Given a sequence of VQ-VAE token indices [y1, y2, ..., yL],
    trains to predict each token given preceding tokens:
        Input:  [BOS, y1, y2, ..., yL-1]
        Target: [y1,  y2, y3, ..., yL  ]

    Args:
        codebook_size: K, number of codebook entries
        token_length: L, length of token sequences
        n_embd: embedding dimension
        n_head: number of attention heads
        n_layer: number of transformer blocks
        attn_pdrop: attention dropout
        resid_pdrop: residual dropout
        mlp_hidden_times: MLP expansion ratio
        random_replace_prob: probability of random replacement during training
    """
    def __init__(
        self,
        codebook_size=512,
        token_length=6,
        n_embd=256,
        n_head=8,
        n_layer=6,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        random_replace_prob=0.1,
        lookback=0,
        **kwargs,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_length = token_length
        self.n_embd = n_embd
        self.random_replace_prob = random_replace_prob
        self.lookback = lookback

        # Vocabulary: K codebook indices + 1 BOS token
        # BOS token index = K
        self.bos_token_id = codebook_size
        vocab_size = codebook_size + 1

        # Token embedding (vocab_size includes BOS)
        self.tok_emb = nn.Embedding(vocab_size, n_embd)

        self.drop = nn.Dropout(resid_pdrop)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                max_len=token_length,
                lookback=lookback,
            )
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        # Output head: predict codebook index (K classes, no BOS in output)
        self.head = nn.Linear(n_embd, codebook_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _random_replace(self, tokens):
        """
        Random replacement augmentation during training.
        Each token is independently replaced with a random token with probability
        ``random_replace_prob``.

        Args:
            tokens: (B, L) token indices
        Returns:
            tokens: (B, L) augmented token indices
        """
        if not self.training or self.random_replace_prob <= 0:
            return tokens

        mask = torch.rand_like(tokens.float()) < self.random_replace_prob
        random_tokens = torch.randint(0, self.codebook_size, tokens.shape, device=tokens.device)
        tokens = torch.where(mask, random_tokens, tokens)
        return tokens

    def forward(self, indices, **kwargs):
        """
        Training forward pass.

        Args:
            indices: (B, L) ground-truth token indices from VQ-VAE encoder

        Returns:
            loss: cross-entropy loss (scalar)
        """
        B, L = indices.shape

        # Create shifted input: [BOS, y1, y2, ..., yL-1]  (shift first, then replace)
        bos = torch.full((B, 1), self.bos_token_id, dtype=torch.long, device=indices.device)
        input_tokens = torch.cat([bos, indices[:, :-1]], dim=1)  # (B, L)

        # Apply random replacement AFTER shifting (paper Algorithm 2, including BOS)
        input_tokens = self._random_replace(input_tokens)

        # Token embeddings (RoPE handles position in attention)
        tok_emb = self.tok_emb(input_tokens)  # (B, L, n_embd)
        x = self.drop(tok_emb)

        # Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)

        # Predict logits
        logits = self.head(x)  # (B, L, K)

        # Cross-entropy loss
        loss = F.cross_entropy(logits.reshape(-1, self.codebook_size), indices.reshape(-1))

        return loss

    @torch.no_grad()
    def generate(self, batch_size=1, temperature=1.0, top_k=None,
                 watermark_mask=None, watermark_delta=0.0,
                 alt_context=False, alternating_partition=False):
        """
        Autoregressive generation of token sequences.

        Args:
            batch_size: number of sequences to generate
            temperature: sampling temperature (higher = more diverse)
            top_k: if set, only sample from top-k logits
            watermark_mask: None, a (K,) green-list mask, or a (K+1, K) table indexed
                by the previous token. In this repo that table is always K+1 identical
                copies of ONE fixed green list
                (Utils.greenlist.create_fixed_watermark_table, seeded by
                --watermark_seed), so the per-row indexing below is machinery, not a
                context rule.
            watermark_delta: logit bias added to green list codes. Either a scalar or a
                per-position list of length token_length — the unbiased warm-up passes
                [0.0]*m + [delta]*(L-m), so the first m positions get no bias
                (Utils.greenlist.warmup_delta_schedule).
            alt_context: if True, flip the green list each time the same prev_token
                repeats. NOT the paper's rule and not exposed by any entry point:
                generate.py and train.py both leave it at False, so the live path is the
                `masks = watermark_mask[prev_tokens]` branch below.
            alternating_partition: the paper's *alternating partition* (arXiv:2608.19727
                p.6, Eq. 9) — if True, flip the green/red partition on odd token
                positions. LVQMark: True.

        Returns:
            tokens: (batch_size, L) generated token indices
        """
        device = self.tok_emb.weight.device
        L = self.token_length

        # Determine watermark mode
        context_wm = watermark_mask is not None and watermark_mask.dim() == 2
        fixed_wm = watermark_mask is not None and watermark_mask.dim() == 1

        # Per-batch prev_token counters for alt_context (never taken: see the docstring)
        if alt_context and context_wm:
            prev_counts = [{} for _ in range(batch_size)]

        # Start with BOS token
        tokens = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)

        for step in range(L):
            # Get embeddings for current sequence
            tok_emb = self.tok_emb(tokens)  # (B, t, n_embd)
            x = self.drop(tok_emb)

            # Forward through transformer
            x = self.blocks(x)
            x = self.ln_f(x)

            # Get logits for last position only
            logits = self.head(x[:, -1, :])  # (B, K)
            logits = logits / temperature

            # Watermark: add bias to green list codes
            cur_delta = watermark_delta[step] if isinstance(watermark_delta, (list, tuple)) else watermark_delta
            if cur_delta > 0:
                if context_wm:
                    prev_tokens = tokens[:, -1]  # (B,)
                    if alt_context:
                        for b in range(batch_size):
                            pt = prev_tokens[b].item()
                            cnt = prev_counts[b].get(pt, 0)
                            mask = watermark_mask[pt]
                            if cnt % 2 == 1:
                                mask = ~mask
                            if alternating_partition and step % 2 == 1:
                                mask = ~mask
                            logits[b, mask] += cur_delta
                            prev_counts[b][pt] = cnt + 1
                    else:
                        masks = watermark_mask[prev_tokens]  # (B, K)
                        if alternating_partition and step % 2 == 1:
                            masks = ~masks
                        logits[masks] += cur_delta
                elif fixed_wm:
                    mask = watermark_mask
                    if alternating_partition and step % 2 == 1:
                        mask = ~mask
                    logits[:, mask] += cur_delta

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append
            tokens = torch.cat([tokens, next_token], dim=1)

        # Remove BOS token
        return tokens[:, 1:]  # (B, L)

    @torch.no_grad()
    def generate_mts(self, batch_size=16, vqvae=None, temperature=1.0, top_k=None,
                     watermark_mask=None, watermark_delta=0.0,
                     alt_context=False, alternating_partition=False, **kwargs):
        """
        Generate multivariate time series using the full pipeline.
        """
        tokens = self.generate(
            batch_size=batch_size, temperature=temperature, top_k=top_k,
            watermark_mask=watermark_mask, watermark_delta=watermark_delta,
            alt_context=alt_context, alternating_partition=alternating_partition,
        )

        # Decode tokens to time series
        if vqvae is not None:
            samples = vqvae.decode_from_indices(tokens)
        else:
            raise ValueError("vqvae model must be provided for generation")

        return samples


@torch.no_grad()
def sample_mts(transformer, vqvae, num, size_every, shape,
               temperature=1.0, top_k=None, watermark_mask=None,
               watermark_delta=0.0, alt_context=False, alternating_partition=False):
    """Generate `num` multivariate time series in batches of `size_every`.

    Thin batching wrapper around AutoregressiveTransformer.generate_mts:
    keeps peak memory bounded and returns a single (num, T, D) array.

    shape: [T, D]. watermark_delta: scalar or per-position list (see .generate).
    alternating_partition: bool, the paper's alternating partition (Eq. 9).
    Returns (num, T, D) float64 ndarray.
    """
    samples = np.empty([0, shape[0], shape[1]])
    num_cycle = int(num // size_every) + 1

    for _ in range(num_cycle):
        sample = transformer.generate_mts(
            batch_size=size_every,
            vqvae=vqvae,
            temperature=temperature,
            top_k=top_k,
            watermark_mask=watermark_mask,
            watermark_delta=watermark_delta,
            alt_context=alt_context,
            alternating_partition=alternating_partition,
        )
        samples = np.vstack([samples, sample.detach().cpu().numpy()])
        torch.cuda.empty_cache()

    return samples[:num]


if __name__ == '__main__':
    # Quick test
    model = AutoregressiveTransformer(
        codebook_size=512,
        token_length=6,
        n_embd=256,
        n_head=8,
        n_layer=4,
    )

    # Simulate VQ-VAE token output
    indices = torch.randint(0, 512, (4, 6))
    loss = model(indices)
    print(f"Training loss: {loss.item():.4f}")

    # Generate
    tokens = model.generate(batch_size=4)
    print(f"Generated tokens shape: {tokens.shape}")  # (4, 6)
    print(f"Generated tokens range: [{tokens.min().item()}, {tokens.max().item()}]")
