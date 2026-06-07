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


class CrossAttentionReadout(nn.Module):
    """A token-structure-aware readout using learned queries over VGGT tokens."""

    def __init__(
        self,
        token_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 256,
        num_metrics: int = 3,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_frames: int = 256,
        max_token_slots: int = 32,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_metrics = num_metrics
        self.max_frames = max_frames
        self.max_token_slots = max_token_slots

        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.frame_embedding = nn.Embedding(max_frames, hidden_dim)
        self.slot_embedding = nn.Embedding(max_token_slots, hidden_dim)
        self.query_embedding = nn.Parameter(torch.randn(1, 1 + num_metrics, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [CrossAttentionReadoutBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.scene_norm = nn.LayerNorm(hidden_dim)
        self.scene_proj = nn.Linear(hidden_dim, output_dim)
        self.metric_norm = nn.LayerNorm(hidden_dim)
        self.metric_score = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        frame_mask: torch.Tensor,
        *,
        return_scores: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 4:
            raise ValueError(f"Expected tokens [B,N,T,C], got {tuple(tokens.shape)}")
        if frame_mask.ndim != 2:
            raise ValueError(f"Expected frame_mask [B,N], got {tuple(frame_mask.shape)}")
        batch_size, frame_count, token_slots, _token_dim = tokens.shape
        if frame_count > self.max_frames:
            raise ValueError(f"frame_count={frame_count} exceeds max_frames={self.max_frames}")
        if token_slots > self.max_token_slots:
            raise ValueError(f"token_slots={token_slots} exceeds max_token_slots={self.max_token_slots}")

        device = tokens.device
        frame_ids = torch.arange(frame_count, device=device)
        slot_ids = torch.arange(token_slots, device=device)
        token_features = self.token_proj(tokens.float())
        token_features = token_features + self.frame_embedding(frame_ids)[None, :, None, :]
        token_features = token_features + self.slot_embedding(slot_ids)[None, None, :, :]
        token_features = token_features.reshape(batch_size, frame_count * token_slots, self.hidden_dim)

        frame_mask = frame_mask.to(dtype=torch.bool, device=device)
        token_padding_mask = ~frame_mask[:, :, None].expand(batch_size, frame_count, token_slots)
        token_padding_mask = token_padding_mask.reshape(batch_size, frame_count * token_slots)

        queries = self.query_embedding.expand(batch_size, -1, -1)
        for block in self.blocks:
            queries = block(queries, token_features, token_padding_mask)

        scene_query = self.scene_norm(queries[:, 0])
        embedding = F.normalize(self.scene_proj(scene_query), dim=-1)
        if not return_scores:
            return embedding

        metric_queries = self.metric_norm(queries[:, 1 : 1 + self.num_metrics])
        metric_scores = self.metric_score(metric_queries).squeeze(-1)
        return embedding, metric_scores


class CrossAttentionReadoutBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        tokens: torch.Tensor,
        token_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        norm_queries = self.query_norm(queries)
        self_out, _ = self.self_attn(norm_queries, norm_queries, norm_queries, need_weights=False)
        queries = queries + self_out

        norm_queries = self.query_norm(queries)
        norm_tokens = self.token_norm(tokens)
        cross_out, _ = self.cross_attn(
            norm_queries,
            norm_tokens,
            norm_tokens,
            key_padding_mask=token_padding_mask,
            need_weights=False,
        )
        queries = queries + cross_out
        return queries + self.ffn(queries)
