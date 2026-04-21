"""
PST-Transformer–style decoupled spatial–temporal modeling for 4D point video.

Inspired by PST-Transformer (global attention with decoupled spatio-temporal
encoding). This repo uses shared P4DConvLite tokenization (`p4transformer.py`)
then: (1) Transformer over points within each frame, (2) pooling to frame tokens,
(3) Transformer over time — instead of a single sequence over T×N.
"""

import torch
import torch.nn as nn

from .p4transformer import P4DConvLite


class _PSTTransformerCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 256,
        spatial_depth: int = 2,
        temporal_depth: int = 2,
        heads: int = 4,
        mlp_ratio: int = 4,
        k: int = 8,
        temporal_window: int = 1,
        max_t: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.p4d = P4DConvLite(
            in_channels=in_channels,
            out_channels=embed_dim,
            k=k,
            temporal_window=temporal_window,
            hidden=max(128, embed_dim // 2),
        )

        self.xyz_proj = nn.Linear(3, embed_dim)
        self.time_embed = nn.Embedding(max_t, embed_dim)

        dim_ff = embed_dim * mlp_ratio
        sp_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(sp_layer, num_layers=spatial_depth)

        tp_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(tp_layer, num_layers=temporal_depth)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, feats: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = feats.shape
        tok = self.p4d(feats, xyz)

        pos = self.xyz_proj(xyz)
        t_idx = torch.arange(T, device=tok.device).clamp_max(self.time_embed.num_embeddings - 1)
        t_emb = self.time_embed(t_idx).view(1, T, 1, -1)
        x = tok + pos + t_emb

        # Spatial: attend over N for each frame
        seq = x.reshape(B * T, N, -1)
        seq = self.spatial_encoder(seq)
        frame_tokens = seq.mean(dim=1)  # (B*T, E)
        frame_tokens = frame_tokens.view(B, T, -1)

        # Temporal: attend over T
        frame_tokens = self.temporal_encoder(frame_tokens)
        feat = frame_tokens.mean(dim=1)
        return self.head(feat)


class PSTTransformer(nn.Module):
    """Regular: XYZ-only; ignores frame_signals."""

    def __init__(
        self,
        info=None,
        embed_dim: int = 256,
        spatial_depth: int = 2,
        temporal_depth: int = 2,
        heads: int = 4,
        k: int = 8,
        temporal_window: int = 1,
        max_t: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = info["num_classes"]
        self.core = _PSTTransformerCore(
            in_channels=3,
            num_classes=self.num_classes,
            embed_dim=embed_dim,
            spatial_depth=spatial_depth,
            temporal_depth=temporal_depth,
            heads=heads,
            k=k,
            temporal_window=temporal_window,
            max_t=max_t,
            dropout=dropout,
        )

    def forward(self, data):
        point_cloud, _frame_signals = data
        xyz = point_cloud[..., :3]
        feats = xyz
        return self.core(feats=feats, xyz=xyz)


class PSTTransformer_Aux(nn.Module):
    """Aux: point Doppler/SNR/Density + broadcast frame IMU/BLE (15D)."""

    def __init__(
        self,
        info=None,
        embed_dim: int = 256,
        spatial_depth: int = 2,
        temporal_depth: int = 2,
        heads: int = 4,
        k: int = 8,
        temporal_window: int = 1,
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
        self.core = _PSTTransformerCore(
            in_channels=in_channels,
            num_classes=self.num_classes,
            embed_dim=embed_dim,
            spatial_depth=spatial_depth,
            temporal_depth=temporal_depth,
            heads=heads,
            k=k,
            temporal_window=temporal_window,
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
