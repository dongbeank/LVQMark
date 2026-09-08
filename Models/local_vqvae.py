"""
Local VQ-VAE for Time Series Watermarking.

Components:
  LocalEncoder:   windowed cross-attention (RF=4, stride=2), shared across lengths
  LocalDecoder:   per-window reconstruction (for encoder pretrain)
  GlobalDecoder:  N tokens → T timesteps cross-attention (length-specific)
  L2VectorQuantizer: EMA + code reset
  LocalVQVAE:     full model combining above
  RobustEncoder:  robust detection encoder (1024 dim, 16 heads)

Training strategy:
  Stage 1: LocalEncoder + LocalDecoder pretrain → good codebook
  Stage 2: Freeze encoder+quantizer, train GlobalDecoder only
  Stage 3: AR Transformer
  Stage 4: Robust Encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ── Building blocks ──

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, kv):
        out, _ = self.cross_attn(query, kv, kv)
        query = self.norm1(query + out)
        query = self.norm2(query + self.ff(query))
        return query


class L2VectorQuantizer(nn.Module):
    """L2 Vector Quantizer with EMA update + code reset."""
    def __init__(self, codebook_size, codebook_dim, commitment_weight=0.25,
                 ema_decay=0.8, reset_threshold=100):
        super().__init__()
        self.K = codebook_size
        self.dc = codebook_dim
        self.commitment_weight = commitment_weight
        self.ema_decay = ema_decay
        self.reset_threshold = reset_threshold

        embedding = torch.randn(codebook_size, codebook_dim)
        nn.init.kaiming_uniform_(embedding)
        self.register_buffer('embedding', embedding)
        self.register_buffer('ema_count', torch.ones(codebook_size))
        self.register_buffer('ema_weight', embedding.clone())
        self.register_buffer('steps_since_used', torch.zeros(codebook_size))

    def encode(self, z_e):
        B, L, D = z_e.shape
        flat = z_e.reshape(-1, D)
        dist = -(flat.pow(2).sum(-1, keepdim=True)
                 + self.embedding.pow(2).sum(-1, keepdim=True).t()
                 - 2 * flat @ self.embedding.t())
        return dist.argmax(dim=-1).reshape(B, L)

    def forward(self, z_e):
        B, L, dc = z_e.shape
        flat_z = z_e.reshape(-1, dc)
        dist = -(flat_z.pow(2).sum(-1, keepdim=True)
                 + self.embedding.pow(2).sum(-1, keepdim=True).t()
                 - 2 * flat_z @ self.embedding.t())
        indices = dist.argmax(dim=-1)
        z_q_flat = self.embedding[indices]

        commit_loss = self.commitment_weight * F.mse_loss(flat_z, z_q_flat.detach())
        codebook_loss = F.mse_loss(z_q_flat, flat_z.detach())

        if self.training:
            with torch.no_grad():
                self._ema_update(flat_z.detach(), indices.detach())
                if self.reset_threshold > 0:
                    self._code_reset(flat_z.detach())

        z_q_flat = flat_z + (z_q_flat - flat_z).detach()
        z_q = z_q_flat.reshape(B, L, dc)
        indices = indices.reshape(B, L)
        return z_q, indices, commit_loss, codebook_loss

    def _ema_update(self, flat_z, indices):
        encodings = F.one_hot(indices, self.K).float()
        code_count = encodings.sum(dim=0)
        code_sum = encodings.t() @ flat_z
        self.ema_count.mul_(self.ema_decay).add_(code_count, alpha=1 - self.ema_decay)
        self.ema_weight.mul_(self.ema_decay).add_(code_sum, alpha=1 - self.ema_decay)
        n = self.ema_count.sum()
        count_smoothed = (self.ema_count + 1e-5) / (n + self.K * 1e-5) * n
        self.embedding.copy_(self.ema_weight / count_smoothed.unsqueeze(-1))
        self.steps_since_used.add_(1)
        self.steps_since_used[code_count > 0] = 0

    def _code_reset(self, flat_z):
        inactive = self.steps_since_used >= self.reset_threshold
        n_inactive = inactive.sum().item()
        if n_inactive > 0 and flat_z.shape[0] > 0:
            replace_idx = torch.randint(0, flat_z.shape[0], (n_inactive,), device=flat_z.device)
            self.embedding[inactive] = flat_z[replace_idx]
            self.ema_weight[inactive] = flat_z[replace_idx]
            self.ema_count[inactive] = 1.0
            self.steps_since_used[inactive] = 0


# ── Encoder ──

class LocalEncoder(nn.Module):
    """Windowed cross-attention encoder. RF=4, stride=2.
    Shared across sequence lengths.
    """
    def __init__(self, feature_size, d_model=512, n_heads=8, n_layers=3,
                 rf=4, stride=2, codebook_dim=None, dropout=0.0):
        super().__init__()
        self.rf = rf
        self.stride = stride
        self.d_model = d_model
        codebook_dim = codebook_dim or d_model

        self.input_proj = nn.Linear(feature_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, rf, d_model) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.proj = nn.Linear(d_model, codebook_dim) if codebook_dim != d_model else nn.Identity()

    def forward(self, x):
        """(B, T, D) → (B, N, codebook_dim)"""
        B, T, D = x.shape
        n_tokens = (T - self.rf) // self.stride + 1

        windows = []
        for i in range(n_tokens):
            start = i * self.stride
            windows.append(x[:, start:start + self.rf, :])
        windows = torch.stack(windows, dim=1)  # (B, N, rf, D)
        w = windows.reshape(B * n_tokens, self.rf, D)

        h = self.input_proj(w) + self.pos_emb
        q = self.query.expand(B * n_tokens, -1, -1)
        for layer in self.layers:
            q = layer(q, h)

        z_e = q.squeeze(1).reshape(B, n_tokens, -1)
        return self.proj(z_e)


# ── Decoders ──

class LocalDecoder(nn.Module):
    """Per-window local decoder. For encoder pretrain."""
    def __init__(self, feature_size, d_model=512, n_heads=8, n_layers=3,
                 rf=4, stride=2, codebook_dim=None, dropout=0.0):
        super().__init__()
        self.rf = rf
        self.stride = stride
        self.feature_size = feature_size
        codebook_dim = codebook_dim or d_model

        self.input_proj = nn.Linear(codebook_dim, d_model) if codebook_dim != d_model else nn.Identity()
        self.queries = nn.Parameter(torch.randn(1, rf, d_model) * 0.02)
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, feature_size)

    def forward(self, z_q):
        """(B, N, codebook_dim) → (B, N, rf, feature_size)"""
        B, N, _ = z_q.shape
        h = self.input_proj(z_q).reshape(B * N, 1, -1)

        q = self.queries.expand(B * N, -1, -1)
        for layer in self.layers:
            q = layer(q, h)

        out = self.output_proj(q)
        return out.reshape(B, N, self.rf, self.feature_size)

    def reconstruct_full(self, windows_out):
        """Overlap-average windows to full sequence."""
        B, N, rf, D = windows_out.shape
        T = (N - 1) * self.stride + rf
        out = torch.zeros(B, T, D, device=windows_out.device)
        counts = torch.zeros(B, T, 1, device=windows_out.device)
        for i in range(N):
            start = i * self.stride
            out[:, start:start + rf, :] += windows_out[:, i, :, :]
            counts[:, start:start + rf, :] += 1
        return out / counts.clamp(min=1)


class GlobalDecoder(nn.Module):
    """Global cross-attention decoder. N tokens → T timesteps.
    Length-specific (trained per window size).
    """
    def __init__(self, feature_size, seq_length, n_tokens,
                 d_model=512, n_heads=8, n_layers=5,
                 codebook_dim=None, dropout=0.0):
        super().__init__()
        self.seq_length = seq_length
        self.feature_size = feature_size
        codebook_dim = codebook_dim or d_model

        self.token_proj = nn.Linear(codebook_dim, d_model) if codebook_dim != d_model else nn.Identity()
        self.token_pos_emb = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)
        self.query_pos_emb = nn.Parameter(torch.randn(1, seq_length, d_model) * 0.02)
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, feature_size)

    def forward(self, z_q):
        """(B, N, codebook_dim) → (B, T, feature_size)"""
        B, N, _ = z_q.shape
        T = self.seq_length

        kv = self.token_proj(z_q) + self.token_pos_emb[:, :N, :]
        q = self.query_pos_emb[:, :T, :].expand(B, -1, -1)

        for layer in self.layers:
            q = layer(q, kv)

        return self.output_proj(q)


# ── Full Model ──

class LocalVQVAE(nn.Module):
    """Local VQ-VAE with pretrain strategy.

    Training modes:
      pretrain: LocalEncoder + LocalDecoder (window-level loss)
      finetune: Freeze encoder+quantizer, train GlobalDecoder (full sequence loss)
    """
    def __init__(self, feature_size, seq_length=64, d_model=512, n_heads=8,
                 enc_layers=3, local_dec_layers=3, global_dec_layers=5,
                 rf=4, stride=2,
                 codebook_size=512, codebook_dim=None, commitment_weight=0.25,
                 dropout=0.0):
        super().__init__()
        self.feature_size = feature_size
        self.seq_length = seq_length
        self.rf = rf
        self.stride = stride
        codebook_dim = codebook_dim or d_model
        self.n_tokens = (seq_length - rf) // stride + 1

        # Encoder
        self.encoder = LocalEncoder(
            feature_size, d_model, n_heads, enc_layers,
            rf, stride, codebook_dim, dropout)

        # Quantizer
        self.quantizer = L2VectorQuantizer(codebook_size, codebook_dim, commitment_weight)

        # Local decoder (for pretrain)
        self.local_decoder = LocalDecoder(
            feature_size, d_model, n_heads, local_dec_layers,
            rf, stride, codebook_dim, dropout)

        # Global decoder (for finetune)
        self.global_decoder = GlobalDecoder(
            feature_size, seq_length, self.n_tokens,
            d_model, n_heads, global_dec_layers,
            codebook_dim, dropout)

    @property
    def token_length(self):
        return self.n_tokens

    def _create_windows(self, x):
        B, T, D = x.shape
        n_tokens = (T - self.rf) // self.stride + 1
        windows = []
        for i in range(n_tokens):
            start = i * self.stride
            windows.append(x[:, start:start + self.rf, :])
        return torch.stack(windows, dim=1)

    def forward_pretrain(self, x):
        """Stage 1: encoder + local decoder. Window-level loss."""
        z_e = self.encoder(x)
        z_q, indices, commit_loss, codebook_loss = self.quantizer(z_e)
        recon_windows = self.local_decoder(z_q)
        gt_windows = self._create_windows(x)
        recon_loss = F.mse_loss(recon_windows, gt_windows)
        return recon_loss + commit_loss + codebook_loss

    def forward_finetune(self, x):
        """Stage 2: encoder+quantizer held fixed (no grad), train global decoder."""
        with torch.no_grad():
            z_e = self.encoder(x)
            z_q, indices, _, _ = self.quantizer(z_e)
        z_q_ste = z_e + (z_q - z_e).detach()
        recon = self.global_decoder(z_q_ste)
        return F.mse_loss(recon, x)

    def forward(self, x, mode='pretrain', **kwargs):
        if mode == 'pretrain':
            return self.forward_pretrain(x)
        elif mode == 'finetune':
            return self.forward_finetune(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @torch.no_grad()
    def encode(self, x):
        z_e = self.encoder(x)
        return self.quantizer.encode(z_e)

    @torch.no_grad()
    def decode_from_indices(self, indices):
        z_q = F.embedding(indices, self.quantizer.embedding)
        return self.global_decoder(z_q)


# ── Robust Encoder ──

class RobustEncoder(nn.Module):
    """Robust encoder for attack-resilient detection.
    Outputs logits over codebook (classification head).
    """
    def __init__(self, feature_size, d_model=1024, n_heads=16, n_layers=3,
                 rf=4, stride=2, dropout=0.0, n_classes=512):
        super().__init__()
        self.rf = rf
        self.stride = stride
        self.d_model = d_model

        self.input_proj = nn.Linear(feature_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, rf, d_model) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        """(B, T, D) → (B, N, n_classes) logits."""
        B, T, D = x.shape
        n_tokens = (T - self.rf) // self.stride + 1

        windows = []
        for i in range(n_tokens):
            start = i * self.stride
            windows.append(x[:, start:start + self.rf, :])
        windows = torch.stack(windows, dim=1)
        w = windows.reshape(B * n_tokens, self.rf, D)

        h = self.input_proj(w) + self.pos_emb
        q = self.query.expand(B * n_tokens, -1, -1)
        for layer in self.layers:
            q = layer(q, h)

        out = self.head(q.squeeze(1))
        return out.reshape(B, n_tokens, -1)


# Defaults used when a config predates the `robust_encoder:` section.
ROBUST_DEFAULTS = {'d_model': 1024, 'n_layers': 3, 'n_heads': None}


def resolve_robust_cfg(config, d_model=None, n_layers=None, n_heads=None):
    """Resolve the robust-encoder architecture from a config, with optional
    CLI overrides. Returns (d_model, n_layers, n_heads).

    Precedence: explicit override > config['robust_encoder'] > ROBUST_DEFAULTS.
    n_heads falls back to max(4, d_model // 64) when left unset.
    """
    rcfg = config.get('robust_encoder') or {}
    d = d_model if d_model is not None else rcfg.get('d_model', ROBUST_DEFAULTS['d_model'])
    nl = n_layers if n_layers is not None else rcfg.get('n_layers', ROBUST_DEFAULTS['n_layers'])
    nh = n_heads if n_heads is not None else rcfg.get('n_heads', ROBUST_DEFAULTS['n_heads'])
    if nh is None:
        nh = max(4, d // 64)
    return d, nl, nh


def infer_robust_cfg(state_dict):
    """Recover (d_model, n_layers) from a saved RobustEncoder state_dict, so
    inference never has to trust a config that may have drifted.
    """
    d_model = state_dict['head.weight'].shape[1]
    n_layers = 1 + max(int(k.split('.')[1]) for k in state_dict
                       if k.startswith('layers.'))
    return d_model, n_layers
