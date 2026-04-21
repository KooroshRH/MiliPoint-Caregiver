"""
3DInAction–style clip modeling for point cloud video.

Registry keys: ``3dinaction``, ``3dinaction_aux`` (see ``model_map``).

Inspired by 3DInAction (temporal clip classification on point sequences). This
implementation uses a per-frame mini PointNet embedding plus a temporal
Transformer over frame tokens (no vendored 3DInAction code).
"""

import torch
import torch.nn as nn


class _MiniPointNetEmbed(nn.Module):
    """(B*T, N, C) -> (B*T, E) via 1x1 conv + max over N."""

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.net(x)
        return x.max(dim=-1).values


class _ThreeDInActionCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 256,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: int = 4,
        max_t: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = _MiniPointNetEmbed(in_channels, embed_dim)
        self.time_embed = nn.Embedding(max_t, embed_dim)
        dim_ff = embed_dim * mlp_ratio
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, feats: torch.Tensor, _xyz: torch.Tensor) -> torch.Tensor:
        B, T, N, C = feats.shape
        x = feats.reshape(B * T, N, C)
        x = self.embed(x)
        x = x.view(B, T, -1)
        t_idx = torch.arange(T, device=x.device).clamp_max(self.time_embed.num_embeddings - 1)
        x = x + self.time_embed(t_idx).view(1, T, -1)
        x = self.temporal_encoder(x)
        feat = x.mean(dim=1)
        return self.head(feat)


class ThreeDInAction(nn.Module):
    """Regular: XYZ-only. Use ``--model '3dinaction'``."""

    def __init__(
        self,
        info=None,
        embed_dim: int = 256,
        depth: int = 2,
        heads: int = 4,
        max_t: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = info["num_classes"]
        self.core = _ThreeDInActionCore(
            in_channels=3,
            num_classes=self.num_classes,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            max_t=max_t,
            dropout=dropout,
        )

    def forward(self, data):
        point_cloud, _frame_signals = data
        xyz = point_cloud[..., :3]
        feats = xyz
        return self.core(feats=feats, xyz=xyz)


class ThreeDInAction_Aux(nn.Module):
    """Aux: 15D per point. Use ``--model '3dinaction_aux'``."""

    def __init__(
        self,
        info=None,
        embed_dim: int = 256,
        depth: int = 2,
        heads: int = 4,
        max_t: int = 64,
        dropout: float = 0.1,
        point_aux_dim: int = 3,
        frame_aux_dim: int = 9,
    ):
        super().__init__()
        self.num_classes = info["num_classes"]
        self.point_aux_dim = point_aux_dim
        self.frame_aux_dim = frame_aux_dim
        in_channels = 3 + point_aux_dim + frame_aux_dim
        self.core = _ThreeDInActionCore(
            in_channels=in_channels,
            num_classes=self.num_classes,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            max_t=max_t,
            dropout=dropout,
        )

    def forward(self, data):
        point_cloud, frame_signals = data
        xyz = point_cloud[..., :3]
        point_aux = point_cloud[..., 3 : 3 + self.point_aux_dim]
        fs = frame_signals.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        feats = torch.cat([xyz, point_aux, fs], dim=-1)
        return self.core(feats=feats, xyz=xyz)
