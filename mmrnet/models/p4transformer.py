import torch
import torch.nn as nn


def _knn_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """
    xyz: (B, M, 3)
    returns: (B, M, k) indices into dim=1 (self included)
    """
    # Pairwise distances (B, M, M). M is tiny in our setting (<= 66).
    d = torch.cdist(xyz, xyz)  # (B, M, M)
    return d.topk(k=k, dim=-1, largest=False).indices


class P4DConvLite(nn.Module):
    """
    A lightweight, pure-PyTorch approximation of P4DConv suitable for very small N.

    It forms a spatio-temporal neighborhood for each point at time t from a window
    of frames [t-w, ..., t+w] and aggregates neighbor features with an MLP.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 8,
        temporal_window: int = 1,
        hidden: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.temporal_window = temporal_window

        # edge input: [center_feat, neigh_feat, rel(dx,dy,dz,dt)]
        edge_in = 2 * in_channels + 4
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        feats: (B, T, N, Cin)
        xyz:   (B, T, N, 3)
        returns token features: (B, T, N, Cout)
        """
        B, T, N, Cin = feats.shape
        w = self.temporal_window
        device = feats.device

        outs = []
        for t in range(T):
            t0 = max(0, t - w)
            t1 = min(T - 1, t + w)
            frames = list(range(t0, t1 + 1))
            # Window tensors (B, M, *)
            xyz_w = xyz[:, frames].reshape(B, len(frames) * N, 3)
            feats_w = feats[:, frames].reshape(B, len(frames) * N, Cin)

            # KNN in the window per window-point
            knn_w = _knn_indices(xyz_w, k=min(self.k, xyz_w.size(1)))  # (B, M, k)

            # Map centers: we only output for the current frame points.
            # They are located at offset (t - t0) * N .. +N
            center_offset = (t - t0) * N
            center_ids = torch.arange(center_offset, center_offset + N, device=device)  # (N,)

            # Gather center xyz/feats: (B, N, *)
            c_xyz = xyz_w[:, center_ids]
            c_feats = feats_w[:, center_ids]

            # Neighbor ids for each center: (B, N, k)
            n_ids = knn_w[:, center_ids]

            # Gather neighbor xyz/feats: (B, N, k, *)
            n_xyz = torch.gather(
                xyz_w.unsqueeze(1).expand(-1, N, -1, -1),
                2,
                n_ids.unsqueeze(-1).expand(-1, -1, -1, 3),
            )
            n_feats = torch.gather(
                feats_w.unsqueeze(1).expand(-1, N, -1, -1),
                2,
                n_ids.unsqueeze(-1).expand(-1, -1, -1, Cin),
            )

            # Relative position: (B, N, k, 4) where dt is based on which frame the neighbor came from.
            rel_xyz = n_xyz - c_xyz.unsqueeze(2)
            # frame index of each neighbor in the window: (B, N, k)
            n_frame = (n_ids // N).to(rel_xyz.dtype)  # 0..len(frames)-1
            c_frame = torch.full((B, N, 1), float(t - t0), device=device, dtype=rel_xyz.dtype)
            rel_t = (n_frame - c_frame) / max(1.0, float(w))  # normalize
            rel = torch.cat([rel_xyz, rel_t.unsqueeze(-1)], dim=-1)

            # Edge features: (B, N, k, edge_in)
            edge = torch.cat(
                [
                    c_feats.unsqueeze(2).expand(-1, -1, n_feats.size(2), -1),
                    n_feats,
                    rel,
                ],
                dim=-1,
            )
            e = self.edge_mlp(edge)  # (B, N, k, Cout)
            out = e.max(dim=2).values  # (B, N, Cout)
            outs.append(out)

        return torch.stack(outs, dim=1)  # (B, T, N, Cout)


class _P4TransformerCore(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embed_dim: int = 256,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: int = 4,
        k: int = 8,
        temporal_window: int = 1,
        max_t: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
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

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

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

        # Add simple (xyz + t) positional signal
        pos = self.xyz_proj(xyz)  # (B,T,N,E)
        t_idx = torch.arange(T, device=tok.device).clamp_max(self.time_embed.num_embeddings - 1)
        t_emb = self.time_embed(t_idx).view(1, T, 1, -1)  # (1,T,1,E)
        tok = tok + pos + t_emb

        seq = tok.view(B, T * N, -1)  # (B, L, E)
        seq = self.encoder(seq)
        feat = seq.mean(dim=1)  # global average pool over tokens
        return self.head(feat)


class P4Transformer(nn.Module):
    """
    Regular P4Transformer-style baseline: XYZ-only temporal input.

    Expects `data == (point_cloud, frame_signals)` but ignores frame_signals and
    uses only XYZ from point_cloud.
    """

    def __init__(
        self,
        info=None,
        embed_dim: int = 256,
        depth: int = 2,
        heads: int = 4,
        k: int = 8,
        temporal_window: int = 1,
        max_t: int = 64,
    ):
        super().__init__()
        self.num_classes = info["num_classes"]
        self.core = _P4TransformerCore(
            in_channels=3,
            num_classes=self.num_classes,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            k=k,
            temporal_window=temporal_window,
            max_t=max_t,
        )

    def forward(self, data):
        point_cloud, _frame_signals = data
        xyz = point_cloud[..., :3]
        feats = xyz
        return self.core(feats=feats, xyz=xyz)


class P4Transformer_Aux(nn.Module):
    """
    Aux version: uses ALL auxiliary signals.

    - Per-point aux: Doppler/SNR/Density (3) from point_cloud[..., 3:6]
    - Per-frame aux: IMU+BLE (9) from frame_signals
    """

    def __init__(
        self,
        info=None,
        embed_dim: int = 256,
        depth: int = 2,
        heads: int = 4,
        k: int = 8,
        temporal_window: int = 1,
        max_t: int = 64,
        point_aux_dim: int = 3,
        frame_aux_dim: int = 9,
    ):
        super().__init__()
        self.num_classes = info["num_classes"]
        self.point_aux_dim = point_aux_dim
        self.frame_aux_dim = frame_aux_dim
        in_channels = 3 + point_aux_dim + frame_aux_dim
        self.core = _P4TransformerCore(
            in_channels=in_channels,
            num_classes=self.num_classes,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            k=k,
            temporal_window=temporal_window,
            max_t=max_t,
        )

    def forward(self, data):
        point_cloud, frame_signals = data
        xyz = point_cloud[..., :3]
        point_aux = point_cloud[..., 3 : 3 + self.point_aux_dim]  # (B,T,N,3)
        fs = frame_signals.unsqueeze(2).expand(-1, -1, xyz.size(2), -1)  # (B,T,N,9)
        feats = torch.cat([xyz, point_aux, fs], dim=-1)
        return self.core(feats=feats, xyz=xyz)

