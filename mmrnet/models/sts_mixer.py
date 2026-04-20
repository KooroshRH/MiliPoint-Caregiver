"""
STS-Mixer–style spatio-temporal mixing for 4D point cloud video.

Inspired by STS-Mixer (spectral / mixer blocks for 4D point tubes). This repo
uses a lightweight pure-PyTorch stack: P4DConvLite tokenization (see
`p4transformer.py`) followed by alternating spatial / temporal / channel
mixing (MLP-Mixer style) on (B, T, N, D) tensors.
"""

import torch
import torch.nn as nn

from .p4transformer import P4DConvLite


class STSMixerBlock(nn.Module):
    """
    One block: spatial mix (across N), temporal mix (across T), channel MLP.
    """

    def __init__(self, dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_s = nn.LayerNorm(dim)
        self.spatial_mix = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm_t = nn.LayerNorm(dim)
        self.temporal_mix = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm_c = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        # Spatial mix across points (per frame)
        h = self.norm_s(x)
        h = h.reshape(B * T, N, D).transpose(1, 2)  # (B*T, D, N)
        h = self.spatial_mix(h).transpose(1, 2).reshape(B, T, N, D)
        x = x + h
        # Temporal mix across frames (per point index)
        h = self.norm_t(x)
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, D).transpose(1, 2)  # (B*N, D, T)
        h = self.temporal_mix(h).transpose(1, 2).reshape(B, N, T, D).permute(0, 2, 1, 3)
        x = x + h
        # Channel mixing
        h = self.norm_c(x)
        x = x + self.channel_mlp(h)
        return x


class _STSMixerCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 256,
        depth: int = 2,
        k: int = 8,
        temporal_window: int = 1,
        max_t: int = 64,
        mlp_ratio: int = 4,
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

        self.blocks = nn.ModuleList(
            [STSMixerBlock(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, feats: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        feats: (B,T,N,Cin)
        xyz: (B,T,N,3)
        returns: (B,num_classes)
        """
        B, T, N, _ = feats.shape
        tok = self.p4d(feats, xyz)  # (B,T,N,E)

        pos = self.xyz_proj(xyz)
        t_idx = torch.arange(T, device=tok.device).clamp_max(self.time_embed.num_embeddings - 1)
        t_emb = self.time_embed(t_idx).view(1, T, 1, -1)
        x = tok + pos + t_emb

        for blk in self.blocks:
            x = blk(x)

        feat = x.mean(dim=(1, 2))
        return self.head(feat)


class STSMixer(nn.Module):
    """
    Regular STS-Mixer–style baseline: XYZ-only; ignores frame_signals.
    """

    def __init__(
        self,
        info=None,
        embed_dim: int = 256,
        depth: int = 2,
        k: int = 8,
        temporal_window: int = 1,
        max_t: int = 64,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = info["num_classes"]
        self.core = _STSMixerCore(
            in_channels=3,
            num_classes=self.num_classes,
            embed_dim=embed_dim,
            depth=depth,
            k=k,
            temporal_window=temporal_window,
            max_t=max_t,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, data):
        point_cloud, _frame_signals = data
        xyz = point_cloud[..., :3]
        feats = xyz
        return self.core(feats=feats, xyz=xyz)


class STSMixer_Aux(nn.Module):
    """
    Aux version: point-level Doppler/SNR/Density + broadcast frame IMU/BLE.
    """

    def __init__(
        self,
        info=None,
        embed_dim: int = 256,
        depth: int = 2,
        k: int = 8,
        temporal_window: int = 1,
        max_t: int = 64,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        point_aux_dim: int = 3,
        frame_aux_dim: int = 9,
    ):
        super().__init__()
        self.num_classes = info["num_classes"]
        self.point_aux_dim = point_aux_dim
        self.frame_aux_dim = frame_aux_dim
        in_channels = 3 + point_aux_dim + frame_aux_dim
        self.core = _STSMixerCore(
            in_channels=in_channels,
            num_classes=self.num_classes,
            embed_dim=embed_dim,
            depth=depth,
            k=k,
            temporal_window=temporal_window,
            max_t=max_t,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, data):
        point_cloud, frame_signals = data
        xyz = point_cloud[..., :3]
        point_aux = point_cloud[..., 3 : 3 + self.point_aux_dim]
        fs = frame_signals.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)
        feats = torch.cat([xyz, point_aux, fs], dim=-1)
        return self.core(feats=feats, xyz=xyz)
