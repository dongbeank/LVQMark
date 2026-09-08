# From https://github.com/soizhiwen/TimeWak
#
# The SDformer AR transformer: learned pos_emb, full causal mask. `alt_position` is the
# alternating partition and `alt_context` is a context rule no entry point enables; the
# translation from this repo's argument names happens once, in
# Models/sdformer/solver.sample_mts.

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


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with causal mask."""
    def __init__(self, n_embd, n_head, attn_pdrop=0.1, resid_pdrop=0.1, max_len=256):
        super().__init__()
        assert n_embd % n_head == 0

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

        # Causal mask
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len))
                             .view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.size()
        nh = self.n_head

        k = self.key(x).view(B, T, nh, C // nh).transpose(1, 2)
        q = self.query(x).view(B, T, nh, C // nh).transpose(1, 2)
        v = self.value(x).view(B, T, nh, C // nh).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
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
                 mlp_hidden_times=4, max_len=256):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            max_len=max_len,
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
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.token_length = token_length
        self.n_embd = n_embd
        self.random_replace_prob = random_replace_prob

        # Vocabulary: K codebook indices + 1 BOS token
        # BOS token index = K
        self.bos_token_id = codebook_size
        vocab_size = codebook_size + 1

        # Token embedding (vocab_size includes BOS)
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, token_length, n_embd))
        nn.init.normal_(self.pos_emb, std=0.02)

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
        Each token is independently replaced with a random token with probability pi.

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

        # Token + position embeddings
        tok_emb = self.tok_emb(input_tokens)  # (B, L, n_embd)
        x = self.drop(tok_emb + self.pos_emb[:, :L, :])

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
                 alt_context=False, alt_position=False):
        """
        Autoregressive generation of token sequences.

        Args:
            batch_size: number of sequences to generate
            temperature: sampling temperature (higher = more diverse)
            top_k: if set, only sample from top-k logits
            watermark_mask: (K,) fixed bool tensor, (K+1,K) context-dependent table, or None
            watermark_delta: scalar or list of per-position logit biases
            alt_context: if True, flip green list when same prev_token repeats
            alt_position: if True, flip green list on odd positions

        Returns:
            tokens: (batch_size, L) generated token indices
        """
        device = self.tok_emb.weight.device
        L = self.token_length

        # Determine watermark mode
        context_wm = watermark_mask is not None and watermark_mask.dim() == 2
        fixed_wm = watermark_mask is not None and watermark_mask.dim() == 1

        # Per-batch prev_token counters for alt_context
        if alt_context and context_wm:
            prev_counts = [{} for _ in range(batch_size)]

        # Start with BOS token
        tokens = torch.full((batch_size, 1), self.bos_token_id, dtype=torch.long, device=device)

        for step in range(L):
            # Get embeddings for current sequence
            tok_emb = self.tok_emb(tokens)  # (B, t, n_embd)
            t = tokens.shape[1]
            x = tok_emb + self.pos_emb[:, :t, :]
            x = self.drop(x)

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
                            if alt_position and step % 2 == 1:
                                mask = ~mask
                            logits[b, mask] += cur_delta
                            prev_counts[b][pt] = cnt + 1
                    else:
                        masks = watermark_mask[prev_tokens]  # (B, K)
                        if alt_position and step % 2 == 1:
                            masks = ~masks
                        logits[masks] += cur_delta
                elif fixed_wm:
                    mask = watermark_mask
                    if alt_position and step % 2 == 1:
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
                     alt_context=False, alt_position=False, **kwargs):
        """
        Generate multivariate time series using the full pipeline:
        AR Transformer generates tokens -> VQ-VAE decoder reconstructs time series.

        Args:
            batch_size: number of time series to generate
            vqvae: trained VQ-VAE model
            temperature: sampling temperature
            top_k: top-k sampling parameter
            watermark_mask: (K,) bool tensor for green list watermarking
            watermark_delta: scalar or list of per-position logit biases
            alt_context: if True, flip green list when same prev_token repeats
            alt_position: if True, flip green list on odd positions

        Returns:
            samples: (batch_size, T, D) generated time series tensor
        """
        # Generate token sequences
        tokens = self.generate(
            batch_size=batch_size, temperature=temperature, top_k=top_k,
            watermark_mask=watermark_mask, watermark_delta=watermark_delta,
            alt_context=alt_context, alt_position=alt_position,
        )

        # Decode tokens to time series
        if vqvae is not None:
            samples = vqvae.decode_from_indices(tokens)
        else:
            raise ValueError("vqvae model must be provided for generation")

        return samples


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
