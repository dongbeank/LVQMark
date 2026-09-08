# From https://github.com/soizhiwen/TimeWak

"""
VQ-VAE for Time Series Tokenization (Stage 1)

Based on SDFormer: Similarity-driven Discrete Transformer (NeurIPS 2024).

Key features:
- 1D Conv Encoder for temporal downsampling
- Similarity-driven Vector Quantization (cosine similarity)
- EMA codebook updates + Code Reset for preventing codebook collapse
- 1D Conv Decoder for reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """1D Residual convolutional block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.block(x) + self.shortcut(x))


class Encoder(nn.Module):
    """
    1D Convolutional Encoder for time series.
    Input:  (B, T, D)  where T=seq_length, D=feature_dim
    Output: (B, L, dc) where L=T/downsample_rate, dc=codebook_dim

    Optionally L2-normalizes output for cosine-similarity VQ.
    """
    def __init__(self, feature_dim, hidden_dim, codebook_dim, downsample_rate=4, n_resblocks=2, normalize=True):
        super().__init__()
        self.normalize = normalize
        assert downsample_rate in [2, 4, 8], "downsample_rate must be 2, 4, or 8"

        layers = []
        # Initial projection
        layers.append(nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.GELU())

        # Downsampling layers
        current_dim = hidden_dim
        if downsample_rate >= 2:
            layers.append(nn.Conv1d(current_dim, hidden_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GELU())
            for _ in range(n_resblocks):
                layers.append(ResidualBlock1D(hidden_dim, hidden_dim))

        if downsample_rate >= 4:
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GELU())
            for _ in range(n_resblocks):
                layers.append(ResidualBlock1D(hidden_dim, hidden_dim))

        if downsample_rate >= 8:
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GELU())
            for _ in range(n_resblocks):
                layers.append(ResidualBlock1D(hidden_dim, hidden_dim))

        # Final projection to codebook dim
        layers.append(nn.Conv1d(hidden_dim, codebook_dim, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, T, D) time series input
        Returns:
            h: (B, L, dc) encoded latent vectors
        """
        # (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        # (B, D, T) -> (B, dc, L)
        h = self.net(x)
        # (B, dc, L) -> (B, L, dc)
        h = h.transpose(1, 2)
        # L2 normalize for cosine similarity
        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)
        return h


class Decoder(nn.Module):
    """
    1D Convolutional Decoder for time series.
    Input:  (B, L, dc) quantized latent vectors
    Output: (B, T, D) reconstructed time series
    """
    def __init__(self, feature_dim, hidden_dim, codebook_dim, upsample_rate=4, n_resblocks=2):
        super().__init__()

        layers = []
        # Initial projection from codebook dim
        layers.append(nn.Conv1d(codebook_dim, hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.GELU())

        # Upsampling layers
        if upsample_rate >= 8:
            layers.append(nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GELU())
            for _ in range(n_resblocks):
                layers.append(ResidualBlock1D(hidden_dim, hidden_dim))

        if upsample_rate >= 4:
            layers.append(nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GELU())
            for _ in range(n_resblocks):
                layers.append(ResidualBlock1D(hidden_dim, hidden_dim))

        if upsample_rate >= 2:
            layers.append(nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GELU())
            for _ in range(n_resblocks):
                layers.append(ResidualBlock1D(hidden_dim, hidden_dim))

        # Final projection to feature dim
        layers.append(nn.Conv1d(hidden_dim, feature_dim, kernel_size=3, padding=1))

        self.net = nn.Sequential(*layers)

    def forward(self, z_q):
        """
        Args:
            z_q: (B, L, dc) quantized vectors
        Returns:
            x_recon: (B, T, D) reconstructed time series
        """
        # (B, L, dc) -> (B, dc, L)
        z_q = z_q.transpose(1, 2)
        # (B, dc, L) -> (B, D, T)
        x_recon = self.net(z_q)
        # (B, D, T) -> (B, T, D)
        x_recon = x_recon.transpose(1, 2)
        return x_recon


class SimilarityVectorQuantizer(nn.Module):
    """
    Similarity-driven Vector Quantization (SDFormer style).

    Uses cosine similarity instead of L2 distance to find nearest codebook entry.
    Features:
    - EMA codebook updates (no gradient through codebook)
    - Code Reset for inactive codes
    - L2-normalized codebook entries

    Args:
        codebook_size: K, number of codebook entries
        codebook_dim: dc, dimension of each entry
        ema_decay: decay rate for EMA codebook updates (default: 0.99)
        reset_threshold: steps of inactivity before resetting a code (default: 100)
    """
    def __init__(self, codebook_size, codebook_dim, ema_decay=0.99, reset_threshold=100):
        super().__init__()
        self.K = codebook_size
        self.dc = codebook_dim
        self.ema_decay = ema_decay
        self.reset_threshold = reset_threshold

        # Codebook: K entries of dimension dc, L2-normalized
        embedding = torch.randn(codebook_size, codebook_dim)
        embedding = F.normalize(embedding, p=2, dim=-1)
        self.register_buffer('embedding', embedding)

        # EMA tracking
        self.register_buffer('ema_count', torch.ones(codebook_size))
        self.register_buffer('ema_weight', embedding.clone())

        # Inactivity tracking for code reset
        self.register_buffer('usage_count', torch.zeros(codebook_size))
        self.register_buffer('steps_since_used', torch.zeros(codebook_size))

    def forward(self, z_e):
        """
        Args:
            z_e: (B, L, dc) L2-normalized encoder outputs
        Returns:
            z_q: (B, L, dc) quantized vectors (with straight-through gradient)
            indices: (B, L) codebook indices
            vq_loss: scalar embedding loss
            perplexity: scalar codebook usage metric
        """
        B, L, dc = z_e.shape

        # Flatten for quantization: (B*L, dc)
        flat_z = z_e.reshape(-1, dc)

        # Cosine similarity: (B*L, K) = flat_z @ codebook.T
        # Both are already L2-normalized
        similarity = flat_z @ self.embedding.t()  # (B*L, K)

        # Find most similar code
        indices = similarity.argmax(dim=-1)  # (B*L,)

        # Dequantize: lookup codebook entries
        z_q_flat = self.embedding[indices]  # (B*L, dc)

        # Compute embedding loss: 1 - cos_sim(z_e, sg(z_q))
        # Since both are normalized, cos_sim = dot product
        embed_loss = (1.0 - (flat_z * z_q_flat.detach()).sum(dim=-1)).mean()

        # EMA codebook update (only during training, no gradient tracking)
        if self.training:
            with torch.no_grad():
                self._ema_update(flat_z.detach(), indices.detach())
                self._code_reset(flat_z.detach())

        # Straight-through estimator: gradient flows through z_e
        z_q_flat = flat_z + (z_q_flat - flat_z).detach()

        # Reshape back
        z_q = z_q_flat.reshape(B, L, dc)
        indices = indices.reshape(B, L)

        # Compute perplexity (codebook usage metric)
        encodings = F.one_hot(indices.reshape(-1), self.K).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, indices, embed_loss, perplexity

    def _ema_update(self, flat_z, indices):
        """Exponential Moving Average codebook update."""
        encodings = F.one_hot(indices, self.K).float()  # (B*L, K)

        # Count of assignments per code
        code_count = encodings.sum(dim=0)  # (K,)

        # Sum of assigned encoder outputs per code
        code_sum = encodings.t() @ flat_z  # (K, dc)

        # EMA update
        self.ema_count.mul_(self.ema_decay).add_(code_count, alpha=1 - self.ema_decay)
        self.ema_weight.mul_(self.ema_decay).add_(code_sum, alpha=1 - self.ema_decay)

        # Laplace smoothing for counts
        n = self.ema_count.sum()
        count_smoothed = (self.ema_count + 1e-5) / (n + self.K * 1e-5) * n

        # Update codebook
        new_embedding = self.ema_weight / count_smoothed.unsqueeze(-1)
        # Re-normalize to unit length
        new_embedding = F.normalize(new_embedding, p=2, dim=-1)
        self.embedding.copy_(new_embedding)

        # Track usage
        self.usage_count.add_(code_count)
        self.steps_since_used.add_(1)
        self.steps_since_used[code_count > 0] = 0

    def _code_reset(self, flat_z):
        """Reset inactive codes by replacing them with random encoder outputs."""
        inactive = self.steps_since_used >= self.reset_threshold
        n_inactive = inactive.sum().item()

        if n_inactive > 0:
            # Sample random encoder outputs to replace inactive codes
            n_samples = flat_z.shape[0]
            if n_samples > 0:
                replace_idx = torch.randint(0, n_samples, (n_inactive,), device=flat_z.device)
                new_codes = flat_z[replace_idx].detach()
                new_codes = F.normalize(new_codes, p=2, dim=-1)

                self.embedding[inactive] = new_codes
                self.ema_weight[inactive] = new_codes
                self.ema_count[inactive] = 1.0
                self.steps_since_used[inactive] = 0
                self.usage_count[inactive] = 0

    def encode(self, z_e):
        """Encode latent vectors to indices without gradient.
        Args:
            z_e: (B, L, dc) L2-normalized encoder outputs
        Returns:
            indices: (B, L) codebook indices
        """
        B, L, dc = z_e.shape
        flat_z = z_e.reshape(-1, dc)
        similarity = flat_z @ self.embedding.t()
        indices = similarity.argmax(dim=-1)
        return indices.reshape(B, L)

    def encode_topk(self, z_e, k=3):
        """Encode latent vectors to top-K candidate indices.
        Args:
            z_e: (B, L, dc) L2-normalized encoder outputs
            k: number of top candidates
        Returns:
            topk_indices: (B, L, k) top-K codebook indices
            topk_sims: (B, L, k) corresponding cosine similarities
        """
        B, L, dc = z_e.shape
        flat_z = z_e.reshape(-1, dc)
        similarity = flat_z @ self.embedding.t()  # (B*L, K)
        topk_sims, topk_indices = torch.topk(similarity, k, dim=-1)  # (B*L, k)
        return topk_indices.reshape(B, L, k), topk_sims.reshape(B, L, k)

    def decode(self, indices):
        """Decode indices to codebook vectors.
        Args:
            indices: (B, L) or (B*L,) codebook indices
        Returns:
            z_q: (..., dc) codebook vectors
        """
        return self.embedding[indices]


class L2VectorQuantizer(nn.Module):
    """
    L2 distance-based Vector Quantization.

    Uses ||z_e - e_k||^2 to find nearest codebook entry.
    Features:
    - EMA codebook updates
    - Code Reset for inactive codes (optional)
    - Commitment loss instead of cosine embedding loss

    Args:
        codebook_size: K, number of codebook entries
        codebook_dim: dc, dimension of each entry
        ema_decay: decay rate for EMA codebook updates (default: 0.8)
        reset_threshold: steps of inactivity before resetting a code (default: 100, 0=disabled)
        commitment_weight: weight for commitment loss (default: 0.25)
    """
    def __init__(self, codebook_size, codebook_dim, ema_decay=0.8,
                 reset_threshold=100, commitment_weight=0.25):
        super().__init__()
        self.K = codebook_size
        self.dc = codebook_dim
        self.ema_decay = ema_decay
        self.reset_threshold = reset_threshold
        self.commitment_weight = commitment_weight

        # Codebook: K entries of dimension dc (Kaiming init, no normalization)
        embedding = torch.randn(codebook_size, codebook_dim)
        nn.init.kaiming_uniform_(embedding)
        self.register_buffer('embedding', embedding)

        # EMA tracking
        self.register_buffer('ema_count', torch.ones(codebook_size))
        self.register_buffer('ema_weight', embedding.clone())

        # Inactivity tracking for code reset
        self.register_buffer('usage_count', torch.zeros(codebook_size))
        self.register_buffer('steps_since_used', torch.zeros(codebook_size))

    def forward(self, z_e):
        """
        Args:
            z_e: (B, L, dc) encoder outputs (NOT L2-normalized)
        Returns:
            z_q: (B, L, dc) quantized vectors (with straight-through gradient)
            indices: (B, L) codebook indices
            vq_loss: scalar commitment loss
            perplexity: scalar codebook usage metric
        """
        B, L, dc = z_e.shape

        # Flatten: (B*L, dc)
        flat_z = z_e.reshape(-1, dc)

        # L2 distance: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z@e.T
        # Negate for argmax (closest = largest negative distance)
        dist = -(
            flat_z.pow(2).sum(dim=-1, keepdim=True)
            + self.embedding.pow(2).sum(dim=-1, keepdim=True).t()
            - 2 * flat_z @ self.embedding.t()
        )  # (B*L, K)

        # Find nearest code
        indices = dist.argmax(dim=-1)  # (B*L,)

        # Dequantize: lookup codebook entries
        z_q_flat = self.embedding[indices]  # (B*L, dc)

        # Commitment loss: commitment_weight * ||z_e - sg(z_q)||^2
        embed_loss = self.commitment_weight * F.mse_loss(flat_z, z_q_flat.detach())

        # EMA codebook update (only during training)
        if self.training:
            with torch.no_grad():
                self._ema_update(flat_z.detach(), indices.detach())
                if self.reset_threshold > 0:
                    self._code_reset(flat_z.detach())

        # Straight-through estimator
        z_q_flat = flat_z + (z_q_flat - flat_z).detach()

        # Reshape back
        z_q = z_q_flat.reshape(B, L, dc)
        indices = indices.reshape(B, L)

        # Compute perplexity (codebook usage metric)
        encodings = F.one_hot(indices.reshape(-1), self.K).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, indices, embed_loss, perplexity

    def _ema_update(self, flat_z, indices):
        """Exponential Moving Average codebook update."""
        encodings = F.one_hot(indices, self.K).float()  # (B*L, K)
        code_count = encodings.sum(dim=0)  # (K,)
        code_sum = encodings.t() @ flat_z  # (K, dc)

        # EMA update
        self.ema_count.mul_(self.ema_decay).add_(code_count, alpha=1 - self.ema_decay)
        self.ema_weight.mul_(self.ema_decay).add_(code_sum, alpha=1 - self.ema_decay)

        # Laplace smoothing
        n = self.ema_count.sum()
        count_smoothed = (self.ema_count + 1e-5) / (n + self.K * 1e-5) * n

        # Update codebook (no L2 normalize for L2 distance)
        new_embedding = self.ema_weight / count_smoothed.unsqueeze(-1)
        self.embedding.copy_(new_embedding)

        # Track usage
        self.usage_count.add_(code_count)
        self.steps_since_used.add_(1)
        self.steps_since_used[code_count > 0] = 0

    def _code_reset(self, flat_z):
        """Reset inactive codes by replacing them with random encoder outputs."""
        inactive = self.steps_since_used >= self.reset_threshold
        n_inactive = inactive.sum().item()

        if n_inactive > 0:
            n_samples = flat_z.shape[0]
            if n_samples > 0:
                replace_idx = torch.randint(0, n_samples, (n_inactive,), device=flat_z.device)
                new_codes = flat_z[replace_idx].detach()

                self.embedding[inactive] = new_codes
                self.ema_weight[inactive] = new_codes
                self.ema_count[inactive] = 1.0
                self.steps_since_used[inactive] = 0
                self.usage_count[inactive] = 0

    def encode(self, z_e):
        """Encode latent vectors to indices without gradient."""
        B, L, dc = z_e.shape
        flat_z = z_e.reshape(-1, dc)
        dist = -(
            flat_z.pow(2).sum(dim=-1, keepdim=True)
            + self.embedding.pow(2).sum(dim=-1, keepdim=True).t()
            - 2 * flat_z @ self.embedding.t()
        )
        indices = dist.argmax(dim=-1)
        return indices.reshape(B, L)

    def encode_topk(self, z_e, k=3):
        """Encode latent vectors to top-K candidate indices (by L2 distance).
        Args:
            z_e: (B, L, dc) encoder outputs
            k: number of top candidates
        Returns:
            topk_indices: (B, L, k) top-K codebook indices
            topk_dists: (B, L, k) corresponding negative L2 distances
        """
        B, L, dc = z_e.shape
        flat_z = z_e.reshape(-1, dc)
        dist = -(
            flat_z.pow(2).sum(dim=-1, keepdim=True)
            + self.embedding.pow(2).sum(dim=-1, keepdim=True).t()
            - 2 * flat_z @ self.embedding.t()
        )  # (B*L, K)
        topk_dists, topk_indices = torch.topk(dist, k, dim=-1)  # (B*L, k)
        return topk_indices.reshape(B, L, k), topk_dists.reshape(B, L, k)

    def decode(self, indices):
        """Decode indices to codebook vectors."""
        return self.embedding[indices]


class VQVAE(nn.Module):
    """
    VQ-VAE for Time Series (SDFormer Stage 1).

    Architecture:
        Encoder (1D Conv) -> VQ (cosine or L2) -> Decoder (1D Conv)

    Args:
        seq_length: T, length of input time series
        feature_size: D, number of variables
        hidden_dim: hidden dimension for conv layers
        codebook_size: K, number of codebook entries
        codebook_dim: dc, dimension of each codebook entry
        downsample_rate: r, temporal downsampling factor (2, 4, or 8)
        n_resblocks: number of residual blocks per scale
        ema_decay: decay for EMA codebook updates
        embed_loss_weight: lambda, weight for embedding loss
        quantizer_type: 'cosine' or 'l2' (default: 'cosine')
        commitment_weight: weight for L2 commitment loss (only used when quantizer_type='l2')
    """
    def __init__(
        self,
        seq_length,
        feature_size,
        hidden_dim=128,
        codebook_size=512,
        codebook_dim=64,
        downsample_rate=4,
        n_resblocks=2,
        ema_decay=0.99,
        embed_loss_weight=1.0,
        reset_threshold=100,
        quantizer_type='cosine',
        commitment_weight=0.25,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.downsample_rate = downsample_rate
        self.token_length = seq_length // downsample_rate  # L
        self.embed_loss_weight = embed_loss_weight
        self.quantizer_type = quantizer_type

        use_l2_normalize = (quantizer_type == 'cosine')

        self.encoder = Encoder(
            feature_dim=feature_size,
            hidden_dim=hidden_dim,
            codebook_dim=codebook_dim,
            downsample_rate=downsample_rate,
            n_resblocks=n_resblocks,
            normalize=use_l2_normalize,
        )

        if quantizer_type == 'l2':
            self.quantizer = L2VectorQuantizer(
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                ema_decay=ema_decay,
                reset_threshold=reset_threshold,
                commitment_weight=commitment_weight,
            )
        else:
            self.quantizer = SimilarityVectorQuantizer(
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                ema_decay=ema_decay,
                reset_threshold=reset_threshold,
            )

        self.decoder = Decoder(
            feature_dim=feature_size,
            hidden_dim=hidden_dim,
            codebook_dim=codebook_dim,
            upsample_rate=downsample_rate,
            n_resblocks=n_resblocks,
        )

    def forward(self, x, **kwargs):
        """
        Training forward pass.

        Args:
            x: (B, T, D) input time series
        Returns:
            total_loss: reconstruction + lambda * embedding loss
        """
        # Encode
        z_e = self.encoder(x)  # (B, L, dc)

        # Vector Quantize
        z_q, indices, embed_loss, perplexity = self.quantizer(z_e)  # (B, L, dc)

        # Decode
        x_recon = self.decoder(z_q)  # (B, T', D)

        # Handle potential length mismatch from conv padding
        T = x.shape[1]
        if x_recon.shape[1] != T:
            x_recon = x_recon[:, :T, :]

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # Total loss
        total_loss = recon_loss + self.embed_loss_weight * embed_loss

        return total_loss

    @torch.no_grad()
    def encode(self, x):
        """
        Encode time series to discrete token indices.

        Args:
            x: (B, T, D) input time series
        Returns:
            indices: (B, L) codebook indices
        """
        z_e = self.encoder(x)
        indices = self.quantizer.encode(z_e)
        return indices

    @torch.no_grad()
    def encode_topk(self, x, k=3):
        """
        Encode time series to top-K candidate token indices.

        Args:
            x: (B, T, D) input time series
            k: number of top candidates
        Returns:
            topk_indices: (B, L, k) top-K codebook indices
            topk_sims: (B, L, k) corresponding similarities
        """
        z_e = self.encoder(x)
        return self.quantizer.encode_topk(z_e, k)

    @torch.no_grad()
    def decode_from_indices(self, indices):
        """
        Decode from discrete token indices to time series.

        Args:
            indices: (B, L) codebook indices
        Returns:
            x_recon: (B, T, D) reconstructed time series
        """
        z_q = self.quantizer.decode(indices)
        x_recon = self.decoder(z_q)
        return x_recon[:, :self.seq_length, :]

    @torch.no_grad()
    def reconstruct(self, x):
        """
        Full encode-quantize-decode reconstruction.

        Args:
            x: (B, T, D) input time series
        Returns:
            x_recon: (B, T, D) reconstructed time series
            indices: (B, L) codebook indices
        """
        z_e = self.encoder(x)
        z_q, indices, _, perplexity = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        T = x.shape[1]
        if x_recon.shape[1] != T:
            x_recon = x_recon[:, :T, :]
        return x_recon, indices


class RobustConvEncoder(nn.Module):
    """
    Robust encoder for VQ-VAE: conv-based (same architecture as Encoder)
    but with a classification head that outputs codebook logits.

    Input:  (B, T, D)
    Output: (B, L, n_classes) logits
    """
    def __init__(self, feature_dim, hidden_dim=512, n_classes=512,
                 downsample_rate=4, n_resblocks=2):
        super().__init__()
        self.downsample_rate = downsample_rate

        layers = []
        layers.append(nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1))
        layers.append(nn.GELU())

        if downsample_rate >= 2:
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GELU())
            for _ in range(n_resblocks):
                layers.append(ResidualBlock1D(hidden_dim, hidden_dim))

        if downsample_rate >= 4:
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GELU())
            for _ in range(n_resblocks):
                layers.append(ResidualBlock1D(hidden_dim, hidden_dim))

        if downsample_rate >= 8:
            layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1))
            layers.append(nn.GELU())
            for _ in range(n_resblocks):
                layers.append(ResidualBlock1D(hidden_dim, hidden_dim))

        # Classification head instead of codebook projection
        layers.append(nn.Conv1d(hidden_dim, n_classes, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(B, T, D) → (B, L, n_classes) logits."""
        x = x.transpose(1, 2)      # (B, D, T)
        h = self.net(x)             # (B, n_classes, L)
        return h.transpose(1, 2)    # (B, L, n_classes)


if __name__ == '__main__':
    # Quick test
    model = VQVAE(
        seq_length=24,
        feature_size=7,
        hidden_dim=128,
        codebook_size=512,
        codebook_dim=64,
        downsample_rate=4,
    )
    x = torch.randn(4, 24, 7)
    loss = model(x)
    print(f"Loss: {loss.item():.4f}")

    indices = model.encode(x)
    print(f"Token shape: {indices.shape}")  # (4, 6)

    x_recon = model.decode_from_indices(indices)
    print(f"Reconstruction shape: {x_recon.shape}")  # (4, 24, 7)
