# dgcnn_mmc_t.py — DGCNN with Multi-Modal Conditioning + Temporal modeling
#
# Two-level conditioning hierarchy:
#   Point-level : FiLM modulation inside EdgeConv using per-point radar metadata
#                 (Doppler, SNR, Density) — signals are co-located with each point,
#                 relationship is direct and local → FiLM is sufficient.
#   Frame-level : Cross-modal attention using frame-level wearable/RF signals
#                 (IMU acc+gyro, BLE RSSI) — signals are foreign-modality global context,
#                 content-dependent selection is needed → cross-attention is appropriate.
#
# Ablation flags (independent, enabling 2x2 ablation):
#   use_film_modulation   : disable point-level FiLM (EdgeConv becomes plain DGCNN)
#   use_frame_conditioning: disable frame-level cross-modal attention

import torch
import torch.nn as nn
from torch.nn import Linear as Lin
from torch_geometric.nn import MLP
from torch_geometric.nn import knn_graph, global_max_pool
from torch_geometric.utils import scatter


class EdgeConvAuxLayer(nn.Module):
    """
    EdgeConv with optional FiLM modulation from per-point auxiliary features.

    For each edge (i <- j):
      edge_geom = [geom_i, geom_j - geom_i]          (2 * in_geom_dim)
      edge_feat = edge_mlp(edge_geom)                 (out_dim)

      if use_film_modulation:
        edge_aux       = [aux_i, aux_j]               (2 * aux_dim)
        [gamma, beta]  = aux_mlp(edge_aux)
        gamma          = sigmoid(gamma + 1)            stabilised around 1
        mod_edge       = gamma * edge_feat + beta
      else:
        mod_edge = edge_feat

      node_out = max_pool(mod_edge, index=i)
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
        """
        geom  : (P, in_geom_dim)   P = B*T*N flattened points
        aux   : (P, aux_dim)
        batch : (P,)               frame index per point
        returns (P, out_dim)
        """
        edge_index = knn_graph(geom, k=self.k, batch=batch)
        idx_i = edge_index[0]   # target (center)
        idx_j = edge_index[1]   # neighbour

        xi = geom[idx_i]
        xj = geom[idx_j]
        edge_geom = torch.cat([xi, xj - xi], dim=1)    # (E, 2*in_geom_dim)

        edge_feat = self.edge_mlp(edge_geom)            # (E, out_dim)

        if self.use_film_modulation and self.aux_mlp is not None:
            aux_i = aux[idx_i]
            aux_j = aux[idx_j]
            edge_aux = torch.cat([aux_i, aux_j], dim=1)    # (E, 2*aux_dim)
            gb = self.aux_mlp(edge_aux)                     # (E, 2*out_dim)
            d = gb.shape[-1] // 2
            gamma = torch.sigmoid(gb[:, :d] + 1.0)         # stabilised ~1
            beta  = gb[:, d:]
            mod_edge = gamma * edge_feat + beta
        else:
            mod_edge = edge_feat

        out = scatter(mod_edge, idx_i, dim=0, dim_size=geom.size(0), reduce='max')
        out = self.norm(out)
        out = torch.relu(out)
        return out


class FrameCrossAttn(nn.Module):
    """
    Lightweight single-head cross-modal attention for frame-level conditioning.

    The radar frame embedding queries the auxiliary (IMU+BLE) signals to
    selectively retrieve relevant context.

    Scale mismatch is severe across modalities (acc ~±4, gyro ~±2000, BLE ~-120..0),
    so each modality is normalised independently before projection. A single
    LayerNorm(9) would let gyro variance dominate and suppress acc and BLE.

    Operates independently per frame (no cross-time interaction — the temporal
    Transformer handles that).

      acc_n, gyro_n, ble_n = LayerNorm(acc), LayerNorm(gyro), LayerNorm(ble)
      s   = concat(acc_n, gyro_n, ble_n)      (B, T, frame_aux_dim)
      q   = W_q(E)                             (B, T, d_ca)
      k   = W_k(s)                             (B, T, d_ca)
      v   = W_v(s)                             (B, T, d_ca)
      ctx = sigmoid(q*k / sqrt(d_ca)) * v      (B, T, d_ca)  element-wise gate
      E'  = E + W_o(ctx)                       (B, T, D)     residual
    """
    def __init__(self, radar_dim, frame_aux_dim=9, d_ca=64, modality_dims=(3, 3, 3)):
        """
        modality_dims: tuple of ints, one per modality, must sum to frame_aux_dim.
                       Each modality gets its own LayerNorm.
                       Default (3,3,3) = acc, gyro, ble.
                       Use (3,3) for gyro+ble only (frame_aux_dim=6).
        """
        super().__init__()
        self.d_ca = d_ca
        self.scale = d_ca ** -0.5
        self.modality_dims = modality_dims
        assert sum(modality_dims) == frame_aux_dim, \
            f"modality_dims {modality_dims} must sum to frame_aux_dim {frame_aux_dim}"

        # One LayerNorm per modality — independent scale normalisation
        self.modality_norms = nn.ModuleList([nn.LayerNorm(d) for d in modality_dims])

        self.W_q = nn.Linear(radar_dim, d_ca, bias=False)
        self.W_k = nn.Linear(frame_aux_dim, d_ca, bias=False)
        self.W_v = nn.Linear(frame_aux_dim, d_ca, bias=False)
        self.W_o = nn.Linear(d_ca, radar_dim, bias=False)

        # zero-init output projection so conditioning starts as identity
        nn.init.zeros_(self.W_o.weight)

    def forward(self, E, frame_signals):
        """
        E             : (B, T, radar_dim)
        frame_signals : (B, T, frame_aux_dim)
        returns       : (B, T, radar_dim)
        """
        # Normalise each modality independently to equalise scales
        chunks = torch.split(frame_signals, self.modality_dims, dim=-1)
        normed = [norm(chunk) for norm, chunk in zip(self.modality_norms, chunks)]
        s = torch.cat(normed, dim=-1)     # (B, T, frame_aux_dim)

        q = self.W_q(E)                     # (B, T, d_ca)
        k = self.W_k(s)                     # (B, T, d_ca)
        v = self.W_v(s)                     # (B, T, d_ca)

        # element-wise scaled dot-product (per frame, single head)
        attn = torch.sigmoid(q * k * self.scale)    # (B, T, d_ca)
        ctx  = attn * v                              # (B, T, d_ca)

        return E + self.W_o(ctx)            # (B, T, radar_dim)  residual


class DGCNNMultiModalCondT(nn.Module):
    """
    DGCNN-MMC-T: DGCNN with Multi-Modal Conditioning and Temporal modeling.

    Architecture:
      1. Spatial backbone  — 3 EdgeConv layers with point-level FiLM conditioning
                             (Doppler, SNR, Density modulate edge features)
      2. Global max-pool   — per frame embedding (B, T, D)
      3. Frame conditioning — cross-modal attention using IMU+BLE signals
                             (frame-level context selectively modulates radar embedding)
      4. Temporal encoder  — TransformerEncoder across T frames
      5. Classification    — MLP head

    Ablation flags (independent):
      use_film_modulation   : if False, point-level FiLM is disabled
      use_frame_conditioning: if False, frame-level cross-attention is disabled
      use_temporal_pos_embed: if False, learnable temporal positional embeddings disabled
      temporal_layers=0     : disables temporal transformer (mean-pool instead)
      point_aux_dim=0       : no point-level auxiliary features at all
      frame_aux_dim=0       : no frame-level auxiliary features at all
    """
    def __init__(self,
                 info,
                 k=20,
                 conv_layers=(32, 32, 32),
                 dense_layers=(1024, 1024, 256, 128),
                 point_aux_dim=3,        # Doppler, SNR, Density
                 frame_aux_dim=9,        # acc(3) + gyro(3) + BLE(3)
                 frame_modality_dims=(3, 3, 3),  # one entry per modality in frame_signals
                 geom_dim=3,             # XYZ
                 d_ca=64,                # cross-attention projection dim
                 temporal_layers=1,
                 temporal_heads=4,
                 # Ablation flags
                 use_film_modulation=True,
                 use_frame_conditioning=True,
                 use_temporal_pos_embed=True):
        super().__init__()

        self.geom_dim = geom_dim
        self.point_aux_dim = point_aux_dim
        self.frame_aux_dim = frame_aux_dim
        self.frame_modality_dims = frame_modality_dims
        self.use_film_modulation = use_film_modulation
        self.use_frame_conditioning = use_frame_conditioning
        self.use_temporal_pos_embed = use_temporal_pos_embed

        self.num_classes = info.get('num_classes', None)

        # --- 1. Spatial backbone: EdgeConv stack with point-level FiLM ---
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
                ))
            in_feat = out_feat

        sum_conv = sum(conv_layers)
        self.lin1 = MLP([sum_conv, dense_layers[0]], plain_last=False)

        # --- 2. Frame-level cross-modal attention ---
        self.temporal_dim = dense_layers[0]
        if use_frame_conditioning and frame_aux_dim > 0:
            self.frame_cross_attn = FrameCrossAttn(
                radar_dim=self.temporal_dim,
                frame_aux_dim=frame_aux_dim,
                d_ca=d_ca,
                modality_dims=frame_modality_dims,
            )
        else:
            self.frame_cross_attn = None

        # --- 3. Temporal encoder ---
        if temporal_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.temporal_dim,
                nhead=temporal_heads,
                dim_feedforward=max(512, self.temporal_dim * 2),
                dropout=0.1,
                batch_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)
            if use_temporal_pos_embed:
                self.time_pos_embed = nn.Parameter(torch.randn(1, 64, self.temporal_dim) * 0.02)
            else:
                self.time_pos_embed = None
        else:
            self.temporal_encoder = None
            self.time_pos_embed = None

        # --- 4. Classification head ---
        self.output = MLP([*dense_layers, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        """
        data: tuple (point_cloud, frame_signals)
          point_cloud   : (B, T, N, 6)  [X, Y, Z, Doppler, SNR, Density]
          frame_signals : (B, T, 9)     [acc(3), gyro(3), BLE(3)]
        returns logits  : (B, num_classes)
        """
        point_cloud, frame_signals = data
        assert point_cloud.dim() == 4, "Expected point_cloud shape (B, T, N, C)"
        B, T, N, C = point_cloud.shape
        device = point_cloud.device

        # Split point cloud into geometry and point-level auxiliary
        geom = point_cloud[..., :self.geom_dim]                              # (B, T, N, 3)
        if self.point_aux_dim > 0:
            aux = point_cloud[..., self.geom_dim:self.geom_dim + self.point_aux_dim]  # (B, T, N, 3)
        else:
            aux = torch.zeros(B, T, N, 0, device=device)

        # Flatten to B*T frames for spatial processing
        geom_flat = geom.reshape(B * T * N, self.geom_dim)
        aux_flat  = aux.reshape(B * T * N, self.point_aux_dim)
        frame_idx = torch.arange(B * T, device=device).repeat_interleave(N)  # (B*T*N,)

        # EdgeConv stack
        x = geom_flat
        xs = []
        for layer in self.edge_layers:
            x = layer(x, aux_flat, batch=frame_idx)
            xs.append(x)

        x_cat = torch.cat(xs, dim=1)        # (B*T*N, sum_conv)
        x_lin = self.lin1(x_cat)            # (B*T*N, temporal_dim)

        # Global max-pool → per-frame embeddings
        E = global_max_pool(x_lin, frame_idx)   # (B*T, temporal_dim)
        E = E.view(B, T, -1)                    # (B, T, temporal_dim)

        # Frame-level cross-modal conditioning
        if self.frame_cross_attn is not None:
            fs = frame_signals[..., :self.frame_aux_dim]    # (B, T, frame_aux_dim)
            E = self.frame_cross_attn(E, fs)                # (B, T, temporal_dim)

        # Temporal positional embeddings + Transformer
        if self.temporal_encoder is not None:
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

            seq_out = self.temporal_encoder(seq)    # (B, T, temporal_dim)
            feat = seq_out.mean(dim=1)              # (B, temporal_dim)
        else:
            feat = E.mean(dim=1)                    # (B, temporal_dim)

        return self.output(feat)                    # (B, num_classes)
