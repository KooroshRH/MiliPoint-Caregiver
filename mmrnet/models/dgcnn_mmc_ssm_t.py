"""
dgcnn_mmc_ssm_t.py — DGCNN-MMC with SSM (Mamba) temporal modeling

This is an aux-only model (by design) that mirrors `dgcnn_mmc_t` up to the
frame-level conditioning step, then replaces the temporal Transformer with a
stacked Mamba/SSM sequence model over frame tokens.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MLP, knn_graph, global_max_pool
from torch_geometric.utils import scatter


try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class EdgeConvAuxLayer(nn.Module):
    """
    Same EdgeConv+FiLM design as dgcnn_mmc_t (kept local here to avoid imports).
    """

    def __init__(self, in_geom_dim, out_dim, aux_dim, k=20, use_film_modulation=True):
        super().__init__()
        self.k = k
        self.out_dim = out_dim
        self.aux_dim = aux_dim
        self.use_film_modulation = use_film_modulation

        self.edge_mlp = MLP([2 * in_geom_dim, out_dim, out_dim], plain_last=False)

        if use_film_modulation and aux_dim > 0:
            self.aux_mlp = MLP([2 * aux_dim, 64, 2 * out_dim], plain_last=True, norm=None)
        else:
            self.aux_mlp = None

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, geom, aux, batch):
        edge_index = knn_graph(geom, k=self.k, batch=batch)
        idx_i = edge_index[0]
        idx_j = edge_index[1]

        xi = geom[idx_i]
        xj = geom[idx_j]
        edge_geom = torch.cat([xi, xj - xi], dim=1)
        edge_feat = self.edge_mlp(edge_geom)

        if self.use_film_modulation and self.aux_mlp is not None:
            aux_i = aux[idx_i]
            aux_j = aux[idx_j]
            edge_aux = torch.cat([aux_i, aux_j], dim=1)
            gb = self.aux_mlp(edge_aux)
            d = gb.shape[-1] // 2
            gamma = torch.sigmoid(gb[:, :d] + 1.0)
            beta = gb[:, d:]
            mod_edge = gamma * edge_feat + beta
        else:
            mod_edge = edge_feat

        out = scatter(mod_edge, idx_i, dim=0, dim_size=geom.size(0), reduce="max")
        out = self.norm(out)
        out = torch.relu(out)
        return out


class FrameCrossAttn(nn.Module):
    """
    Same lightweight per-frame cross-modal gating attention as dgcnn_mmc_t.
    """

    def __init__(self, radar_dim, frame_aux_dim=9, d_ca=64, modality_dims=(3, 3, 3)):
        super().__init__()
        self.d_ca = d_ca
        self.scale = d_ca**-0.5
        self.modality_dims = modality_dims
        assert sum(modality_dims) == frame_aux_dim

        self.modality_norms = nn.ModuleList([nn.LayerNorm(d) for d in modality_dims])

        self.W_q = nn.Linear(radar_dim, d_ca, bias=False)
        self.W_k = nn.Linear(frame_aux_dim, d_ca, bias=False)
        self.W_v = nn.Linear(frame_aux_dim, d_ca, bias=False)
        self.W_o = nn.Linear(d_ca, radar_dim, bias=False)
        nn.init.zeros_(self.W_o.weight)

    def forward(self, E, frame_signals):
        chunks = torch.split(frame_signals, self.modality_dims, dim=-1)
        normed = [norm(chunk) for norm, chunk in zip(self.modality_norms, chunks)]
        s = torch.cat(normed, dim=-1)

        q = self.W_q(E)
        k = self.W_k(s)
        v = self.W_v(s)

        attn = torch.sigmoid(q * k * self.scale)
        ctx = attn * v
        return E + self.W_o(ctx)


class _TemporalSSMBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.use_mamba = MAMBA_AVAILABLE
        if self.use_mamba:
            self.seq = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.seq = nn.GRU(dim, dim, batch_first=True, bidirectional=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.use_mamba:
            out = self.seq(h)
        else:
            out, _ = self.seq(h)
        return x + self.dropout(out)


class DGCNNMultiModalCondSSMT(nn.Module):
    """
    DGCNN-MMC-SSM-T:
    - Same point-level FiLM (Doppler/SNR/Density) and frame-level conditioning (IMU/BLE)
    - Temporal modeling: Mamba/SSM blocks over (B, T, D) frame tokens

    This model is aux-only by nature (expects both point aux + frame aux).
    """

    def __init__(
        self,
        info,
        k: int = 20,
        conv_layers=(32, 32, 32),
        # Keep at 256-dim scale by default for fair baseline comparison.
        dense_layers=(256, 256, 128, 64),
        point_aux_dim: int = 3,
        frame_aux_dim: int = 9,
        frame_modality_dims=(3, 3, 3),
        geom_dim: int = 3,
        d_ca: int = 64,
        ssm_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_film_modulation: bool = True,
        use_frame_conditioning: bool = True,
        use_temporal_pos_embed: bool = True,
        max_t: int = 64,
    ):
        super().__init__()

        self.geom_dim = geom_dim
        self.point_aux_dim = point_aux_dim
        self.frame_aux_dim = frame_aux_dim
        self.frame_modality_dims = frame_modality_dims
        self.use_film_modulation = use_film_modulation
        self.use_frame_conditioning = use_frame_conditioning
        self.use_temporal_pos_embed = use_temporal_pos_embed

        self.num_classes = info.get("num_classes", None)

        # 1) Spatial backbone
        self.edge_layers = nn.ModuleList()
        in_feat = geom_dim
        for out_feat in conv_layers:
            self.edge_layers.append(
                EdgeConvAuxLayer(
                    in_geom_dim=in_feat,
                    out_dim=out_feat,
                    aux_dim=point_aux_dim,
                    k=k,
                    use_film_modulation=use_film_modulation,
                )
            )
            in_feat = out_feat

        sum_conv = sum(conv_layers)
        self.lin1 = MLP([sum_conv, dense_layers[0]], plain_last=False)
        self.temporal_dim = dense_layers[0]

        # 2) Frame-level conditioning
        if use_frame_conditioning and frame_aux_dim > 0:
            self.frame_cross_attn = FrameCrossAttn(
                radar_dim=self.temporal_dim,
                frame_aux_dim=frame_aux_dim,
                d_ca=d_ca,
                modality_dims=frame_modality_dims,
            )
        else:
            self.frame_cross_attn = None

        # 3) Temporal SSM
        self.ssm_blocks = nn.ModuleList(
            [
                _TemporalSSMBlock(
                    dim=self.temporal_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(ssm_layers)
            ]
        )
        if use_temporal_pos_embed:
            self.time_pos_embed = nn.Parameter(torch.randn(1, max_t, self.temporal_dim) * 0.02)
        else:
            self.time_pos_embed = None

        # 4) Head
        self.output = MLP([*dense_layers, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        point_cloud, frame_signals = data
        B, T, N, _ = point_cloud.shape
        device = point_cloud.device

        geom = point_cloud[..., : self.geom_dim]
        aux = point_cloud[..., self.geom_dim : self.geom_dim + self.point_aux_dim]

        geom_flat = geom.reshape(B * T * N, self.geom_dim)
        aux_flat = aux.reshape(B * T * N, self.point_aux_dim)
        frame_idx = torch.arange(B * T, device=device).repeat_interleave(N)

        x = geom_flat
        xs = []
        for layer in self.edge_layers:
            x = layer(x, aux_flat, batch=frame_idx)
            xs.append(x)

        x_cat = torch.cat(xs, dim=1)
        x_lin = self.lin1(x_cat)

        E = global_max_pool(x_lin, frame_idx).view(B, T, -1)

        if self.frame_cross_attn is not None:
            fs = frame_signals[..., : self.frame_aux_dim]
            E = self.frame_cross_attn(E, fs)

        if self.use_temporal_pos_embed and self.time_pos_embed is not None:
            max_pos = self.time_pos_embed.size(1)
            if T <= max_pos:
                pos = self.time_pos_embed[:, :T, :].expand(B, -1, -1)
            else:
                reps = (T + max_pos - 1) // max_pos
                pos = self.time_pos_embed.repeat(1, reps, 1)[:, :T, :].expand(B, -1, -1)
            seq = E + pos
        else:
            seq = E

        for blk in self.ssm_blocks:
            seq = blk(seq)

        feat = seq.mean(dim=1)
        return self.output(feat)

