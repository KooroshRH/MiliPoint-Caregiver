"""
UST-SSM–style unified spatio-temporal state-space modeling for 4D point video.

Inspired by UST-SSM (unified ST sequence + SSM). This implementation uses
P4DConvLite for local 4D neighborhoods, then flattens tokens to a single
sequence (B, T*N, D) and stacks Mamba SSM blocks (mamba-ssm), matching other
models in this repo.
"""

import torch
import torch.nn as nn

from .p4transformer import P4DConvLite

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class USTSSMBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_mamba: bool = True,
    ):
        super().__init__()
        self.use_mamba = use_mamba and MAMBA_AVAILABLE
        self.norm = nn.LayerNorm(dim)
        if self.use_mamba:
            self.seq = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.seq = nn.GRU(dim, dim, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        h = self.norm(x)
        if self.use_mamba:
            out = self.seq(h)
        else:
            out, _ = self.seq(h)
        return x + self.dropout(out)


class _USTSSMCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 256,
        depth: int = 4,
        k: int = 8,
        temporal_window: int = 1,
        max_t: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        use_mamba = MAMBA_AVAILABLE

        self.p4d = P4DConvLite(
            in_channels=in_channels,
            out_channels=embed_dim,
            k=k,
            temporal_window=temporal_window,
            hidden=max(128, embed_dim // 2),
        )

        self.xyz_proj = nn.Linear(3, embed_dim)
        self.time_embed = nn.Embedding(max_t, embed_dim)

        self.blocks = nn.ModuleList(
            [
                USTSSMBlock(
                    embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    use_mamba=use_mamba,
                )
                for _ in range(depth)
            ]
        )

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

        seq = x.reshape(B, T * N, -1)
        for blk in self.blocks:
            seq = blk(seq)

        feat = seq.mean(dim=1)
        return self.head(feat)


class USTSSM(nn.Module):
    """
    Regular UST-SSM–style baseline: XYZ-only; ignores frame_signals.
    """

    def __init__(
        self,
        info=None,
        embed_dim: int = 256,
        depth: int = 4,
        k: int = 8,
        temporal_window: int = 1,
        max_t: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = info["num_classes"]
        self.core = _USTSSMCore(
            in_channels=3,
            num_classes=self.num_classes,
            embed_dim=embed_dim,
            depth=depth,
            k=k,
            temporal_window=temporal_window,
            max_t=max_t,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

    def forward(self, data):
        point_cloud, _frame_signals = data
        xyz = point_cloud[..., :3]
        feats = xyz
        return self.core(feats=feats, xyz=xyz)


class USTSSM_Aux(nn.Module):
    """
    Aux version: point-level aux + broadcast frame IMU/BLE.
    """

    def __init__(
        self,
        info=None,
        embed_dim: int = 256,
        depth: int = 4,
        k: int = 8,
        temporal_window: int = 1,
        max_t: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        point_aux_dim: int = 3,
        frame_aux_dim: int = 9,
    ):
        super().__init__()
        self.num_classes = info["num_classes"]
        self.point_aux_dim = point_aux_dim
        self.frame_aux_dim = frame_aux_dim
        in_channels = 3 + point_aux_dim + frame_aux_dim
        self.core = _USTSSMCore(
            in_channels=in_channels,
            num_classes=self.num_classes,
            embed_dim=embed_dim,
            depth=depth,
            k=k,
            temporal_window=temporal_window,
            max_t=max_t,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

    def forward(self, data):
        point_cloud, frame_signals = data
        xyz = point_cloud[..., :3]
        point_aux = point_cloud[..., 3 : 3 + self.point_aux_dim]
        fs = frame_signals.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        feats = torch.cat([xyz, point_aux, fs], dim=-1)
        return self.core(feats=feats, xyz=xyz)
