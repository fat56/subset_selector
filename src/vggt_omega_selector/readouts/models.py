"""Readout heads for cached VGGT-OMEGA camera/register tokens."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PooledReadout(nn.Module):
    """A compact scene readout from selected camera/register tokens.

    Input tokens are expected as ``[B, N, T, C]`` with camera token at slot 0
    and register tokens in slots 1:.
    """

    def __init__(
        self,
        token_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        summary_dim = token_dim * 4
        self.net = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tokens: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError(f"Expected tokens [B,N,T,C], got {tuple(tokens.shape)}")
        if frame_mask.ndim != 2:
            raise ValueError(f"Expected frame_mask [B,N], got {tuple(frame_mask.shape)}")
        frame_mask = frame_mask.to(dtype=tokens.dtype, device=tokens.device)
        frame_mask = frame_mask.clamp(min=0, max=1)
        denom_frames = frame_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        camera = tokens[:, :, 0, :]
        register = tokens[:, :, 1:, :]

        camera_mean = (camera * frame_mask[..., None]).sum(dim=1) / denom_frames

        token_mask = frame_mask[:, :, None, None]
        denom_register = (denom_frames * register.shape[2]).clamp_min(1.0)
        register_mean = (register * token_mask).sum(dim=(1, 2)) / denom_register

        centered = (register - register_mean[:, None, None, :]) * token_mask
        register_std = torch.sqrt((centered.square().sum(dim=(1, 2)) / denom_register).clamp_min(1e-8))

        neg_inf = torch.finfo(tokens.dtype).min
        masked_register = register.masked_fill(token_mask <= 0, neg_inf)
        register_max = masked_register.amax(dim=(1, 2))
        register_max = torch.where(torch.isfinite(register_max), register_max, register_mean)

        summary = torch.cat([camera_mean, register_mean, register_max, register_std], dim=-1)
        return F.normalize(self.net(summary.float()), dim=-1)
