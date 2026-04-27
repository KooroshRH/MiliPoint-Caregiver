"""
Kinet-inspired static + dynamic branches for point cloud sequences.

Inspired by Kinet (space-time surfaces in feature space with static/dynamic
fusion). Pure PyTorch: mini PointNet per frame, temporal mean as static
representation, 1D temporal conv as dynamic trajectory, then fusion + head.
"""

import torch
import torch.nn as nn


class _KinetCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 256,
        dynamic_kernel: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.point_stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
        )
        pad = dynamic_kernel // 2
        self.dynamic_branch = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=dynamic_kernel, padding=pad),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
        )
        self.fuse = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, feats: torch.Tensor, xyz: torch.Tensor = None) -> torch.Tensor:
        """
        feats: (B,T,N,Cin) — uses full channels (xyz or 15D aux).
        xyz: unused; kept for API parity with other temporal models.
        """
        B, T, N, C = feats.shape
        x = feats.reshape(B * T, N, C).transpose(1, 2)  # (B*T, C, N)
        x = self.point_stem(x)
        x = x.max(dim=-1).values  # (B*T, E)
        x = x.view(B, T, -1)

        static = x.mean(dim=1)
        dyn = self.dynamic_branch(x.transpose(1, 2)).transpose(1, 2).mean(dim=1)

        h = self.fuse(torch.cat([static, dyn], dim=-1))
        return self.head(h)


class Kinet(nn.Module):
    """Regular: XYZ-only."""

    def __init__(
        self,
        info=None,
        embed_dim: int = 256,
        dynamic_kernel: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = info["num_classes"]
        self.core = _KinetCore(
            in_channels=3,
            num_classes=self.num_classes,
            embed_dim=embed_dim,
            dynamic_kernel=dynamic_kernel,
            dropout=dropout,
        )

    def forward(self, data):
        point_cloud, _frame_signals = data
        xyz = point_cloud[..., :3]
        feats = xyz
        return self.core(feats=feats, xyz=xyz)


class Kinet_Aux(nn.Module):
    """Aux: 15D per point (point + broadcast frame signals)."""

    def __init__(
        self,
        info=None,
        embed_dim: int = 256,
        dynamic_kernel: int = 5,
        dropout: float = 0.1,
        point_aux_dim: int = 3,
        frame_aux_dim: int = 9,
    ):
        super().__init__()
        self.num_classes = info["num_classes"]
        self.point_aux_dim = point_aux_dim
        self.frame_aux_dim = frame_aux_dim
        in_channels = 3 + point_aux_dim + frame_aux_dim
        self.core = _KinetCore(
            in_channels=in_channels,
            num_classes=self.num_classes,
            embed_dim=embed_dim,
            dynamic_kernel=dynamic_kernel,
            dropout=dropout,
        )

    def forward(self, data):
        point_cloud, frame_signals = data
        xyz = point_cloud[..., :3]
        point_aux = point_cloud[..., 3 : 3 + self.point_aux_dim]
        fs = frame_signals.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        feats = torch.cat([xyz, point_aux, fs], dim=-1)
        return self.core(feats=feats, xyz=xyz)
