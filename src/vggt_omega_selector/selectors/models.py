"""Fixed-budget selector models for Stage 2."""

from __future__ import annotations

import torch
from torch import nn


class FixedKSetSelector(nn.Module):
    """Contextual per-frame scorer for fixed-ratio/fixed-K subset selection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_frames: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames

        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.frame_embedding = nn.Embedding(max_frames, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected features [B,N,C], got {tuple(features.shape)}")
        if frame_mask.ndim != 2:
            raise ValueError(f"Expected frame_mask [B,N], got {tuple(frame_mask.shape)}")
        batch_size, frame_count, _ = features.shape
        if frame_count > self.max_frames:
            raise ValueError(f"frame_count={frame_count} exceeds max_frames={self.max_frames}")

        hidden = self.projector(features.float())
        frame_ids = torch.arange(frame_count, device=features.device)
        hidden = hidden + self.frame_embedding(frame_ids)[None, :, :]
        padding_mask = ~frame_mask.to(dtype=torch.bool, device=features.device)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        scores = self.score_head(hidden).squeeze(-1)
        return scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)


def soft_topk_mask(scores: torch.Tensor, frame_mask: torch.Tensor, k: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return a bounded differentiable mask whose row sum is approximately K."""

    if scores.ndim != 2:
        raise ValueError(f"Expected scores [B,N], got {tuple(scores.shape)}")
    valid = frame_mask.to(dtype=torch.bool, device=scores.device)
    tau = max(temperature, 1e-6)
    target = k.to(dtype=scores.dtype, device=scores.device).unsqueeze(1)
    masked_scores = scores.masked_fill(~valid, 0.0)

    low = masked_scores.amin(dim=1, keepdim=True) - 20.0 * tau
    high = masked_scores.amax(dim=1, keepdim=True) + 20.0 * tau
    for _ in range(32):
        mid = (low + high) * 0.5
        estimate = torch.sigmoid((masked_scores - mid) / tau).masked_fill(~valid, 0.0).sum(dim=1, keepdim=True)
        low = torch.where(estimate > target, mid, low)
        high = torch.where(estimate > target, high, mid)
    threshold = (low + high) * 0.5
    return torch.sigmoid((masked_scores - threshold) / tau).masked_fill(~valid, 0.0)


def hard_topk_mask(scores: torch.Tensor, frame_mask: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    valid = frame_mask.to(dtype=torch.bool, device=scores.device)
    out = torch.zeros_like(scores, dtype=torch.float32)
    for row in range(scores.shape[0]):
        count = int(k[row].item())
        valid_indices = torch.nonzero(valid[row], as_tuple=False).flatten()
        count = max(1, min(count, int(valid_indices.numel())))
        row_scores = scores[row, valid_indices]
        selected = valid_indices[torch.topk(row_scores, k=count).indices]
        out[row, selected] = 1.0
    return out
