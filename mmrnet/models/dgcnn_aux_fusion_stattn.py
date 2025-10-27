# dgcnn_auxfusion_stattn.py
import torch
import torch.nn as nn
from torch.nn import Linear as Lin
from torch_geometric.nn import MLP
from torch_geometric.nn import knn_graph, global_max_pool
from torch_geometric.utils import scatter


class EdgeConvAuxLayer(nn.Module):
    """
    EdgeConv-like layer with auxiliary-based FiLM modulation.

    For each edge (i <- j):
      - build edge_geom = concat(x_i, x_j - x_i)
      - build edge_aux  = concat(aux_i, aux_j)
      - edge_feat = edge_mlp(edge_geom)
      - [gamma, beta] = aux_mlp(edge_aux)  # shapes -> 2 * out_dim
      - mod_edge = gamma * edge_feat + beta
      - node_out = max_pool(mod_edge, index=i)
    """
    def __init__(self, in_geom_dim, out_dim, aux_dim, k=20, aggr='max'):
        super().__init__()
        self.k = k
        self.out_dim = out_dim
        self.aux_dim = aux_dim
        self.aggr = aggr

        # Edge MLP: input = [geom_i, geom_j - geom_i] -> 2*in_geom_dim
        self.edge_mlp = MLP([2 * in_geom_dim, out_dim, out_dim], plain_last=False)

        # Aux MLP: input = [aux_i, aux_j] -> 2*aux_dim -> produce gamma & beta (2*out_dim)
        self.aux_mlp = MLP([2 * aux_dim, 64, 2 * out_dim], plain_last=True, norm=None)

        # Optional normalization on output
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, geom, aux, batch):
        """
        geom: (num_points, in_geom_dim)
        aux:  (num_points, aux_dim)
        batch: (num_points,) batch idx of each point (here batch encodes frame index if pre-flattened)
        returns: (num_points, out_dim)
        """
        # Build kNN edge index per batch
        # knn_graph returns edges where first row = target index (center), second row = neighbor index
        edge_index = knn_graph(geom, k=self.k, batch=batch)  # (2, E)
        idx_target = edge_index[0]  # centers (i)
        idx_neighbor = edge_index[1]  # neighbors (j)

        # Gather features
        xi = geom[idx_target]        # (E, in_geom_dim)
        xj = geom[idx_neighbor]      # (E, in_geom_dim)
        edge_geom = torch.cat([xi, xj - xi], dim=1)  # (E, 2*in_geom_dim)

        # Aux pair features
        if aux is not None and aux.shape[1] > 0:
            aux_i = aux[idx_target]
            aux_j = aux[idx_neighbor]
            edge_aux = torch.cat([aux_i, aux_j], dim=1)
        else:
            edge_aux = torch.zeros(edge_geom.size(0), 2 * self.aux_dim, device=edge_geom.device)

        # Edge MLP -> raw edge features
        edge_feat = self.edge_mlp(edge_geom)  # (E, out_dim)

        # Aux modulation -> produce gamma & beta
        gb = self.aux_mlp(edge_aux)  # (E, 2*out_dim)
        d = gb.shape[-1] // 2
        gamma = gb[:, :d]
        beta = gb[:, d:]

        # Stabilize gamma: around 1.0
        gamma = torch.sigmoid(gamma + 1.0)  # values roughly (0.5..~1)
        # Apply FiLM modulation per-edge
        mod_edge = gamma * edge_feat + beta  # (E, out_dim)

        # Aggregate per target node (center)
        out = scatter(mod_edge, idx_target, dim=0, dim_size=geom.size(0), reduce='max')
        out = self.norm(out)
        out = torch.relu(out)
        return out


class DGCNNAuxFusion_STAttn(nn.Module):
    """
    DGCNN variant with Aux-Fusion inside EdgeConv layers and improved temporal aggregation:
      - attention-weighted temporal pooling (learnable),
      - SNR bias on attention (optional),
      - short-term temporal-difference residual added to pooled representation.

    Name: DGCNNAuxFusion-STAttn
    """
    def __init__(self,
                 info,
                 k=30,
                 conv_layers=(32, 32, 32),
                 dense_layers=(1024, 1024, 256, 128),
                 aux_dim=4,
                 geom_dim=3,
                 temporal_layers=0,
                 temporal_heads=4,
                 use_snr_bias=True,
                 use_diff_residual=True):
        """
        info: dict with 'num_classes' or 'num_keypoints'
        conv_layers: tuple of feature dims for edge conv stacks (same length as original)
        dense_layers: tuple for dense head (same as original)
        aux_dim: number of per-point auxiliary channels (zone, doppler, snr, density)
        geom_dim: usually 3 (x,y,z)
        temporal_layers: *not used here for deep temporal Transformer* (kept for API parity) — we use light attention pooling
        temporal_heads: kept for API parity
        use_snr_bias: whether to bias temporal attention with mean SNR of frames
        use_diff_residual: whether to include Δ-frame residual added to attended pooled vector
        """
        super().__init__()
        self.k = k
        self.aux_dim = aux_dim
        self.geom_dim = geom_dim
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.use_snr_bias = use_snr_bias
        self.use_diff_residual = use_diff_residual

        self.num_classes = info.get('num_classes', None)

        # Build EdgeConvAux stack (keep same depth and dims)
        self.edge_layers = nn.ModuleList()
        in_feat = geom_dim
        for out_feat in conv_layers:
            self.edge_layers.append(EdgeConvAuxLayer(in_geom_dim=in_feat,
                                                     out_dim=out_feat,
                                                     aux_dim=aux_dim,
                                                     k=k))
            in_feat = out_feat

        # Linear to combine stacked conv outputs per-point (like original DGCNN)
        sum_conv = sum(conv_layers)
        self.lin1 = MLP([sum_conv, dense_layers[0]], plain_last=False)

        # Temporal attention pooling components (lightweight):
        # - a linear to compute per-frame logit (d -> 1)
        # - optional learnable scale for SNR bias
        self.temporal_dim = dense_layers[0]
        self.attn_fc = Lin(self.temporal_dim, 1)
        if use_snr_bias:
            # scalar learnable multiplier for SNR term
            self.snr_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.snr_scale = None

        # Classification / output head: mirror original dense layers as MLP
        if self.num_classes is None:
            num_points = info['num_keypoints']
            self.num_points = num_points
            point_branches = {}
            for i in range(num_points):
                point_branches[f'branch_{i}'] = MLP([dense_layers[-1], 64, 3], norm=None)
            self.output = nn.ModuleDict(point_branches)
        else:
            # build a classifier MLP: after temporal aggregation, features are dense_layers[0]
            # we keep the same dense MLP stack to preserve architecture
            self.output = MLP([*dense_layers, self.num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        """
        data: shape (B, T, N, C) with channels:
              [x,y,z, zone, doppler, snr, local_density, ...optional]
        returns: logits (B, num_classes)
        """
        assert data.dim() == 4, "Expect input shape (B, T, N, C). For single frame use T=1."
        B, T, N, C = data.shape
        device = data.device

        # split geom and aux
        geom = data[..., :self.geom_dim]  # (B, T, N, geom_dim)
        if self.aux_dim > 0:
            aux = data[..., self.geom_dim:self.geom_dim + self.aux_dim]  # (B, T, N, aux_dim)
        else:
            aux = torch.zeros(B, T, N, 0, device=device)

        # flatten frames into batch of frames: (B*T, N, ...)
        frames = B * T
        geom_flat = geom.reshape(B * T * N, self.geom_dim)    # (B*T*N, geom_dim)
        aux_flat = aux.reshape(B * T * N, self.aux_dim)       # (B*T*N, aux_dim)

        # build frame index per point: 0..B*T-1
        frame_idx = torch.arange(B * T, device=device).repeat_interleave(N)  # (B*T*N,)

        # Process EdgeConv stack:
        x = geom_flat  # use coordinates as initial geometry features
        xs = []
        for layer in self.edge_layers:
            x = layer(x, aux_flat, batch=frame_idx)  # (B*T*N, out_dim)
            xs.append(x)

        # concatenate features across conv layers (per-point)
        x_cat = torch.cat(xs, dim=1)  # (B*T*N, sum_conv)
        x_lin = self.lin1(x_cat)      # (B*T*N, dense0)

        # Per-frame global pooling (max)
        pooled_frames = global_max_pool(x_lin, frame_idx)  # (B*T, dense0)
        pooled_frames = pooled_frames.view(B, T, -1)      # (B, T, F) where F = dense0

        # Compute per-frame SNR mean (if available) to bias temporal attention
        if self.aux_dim > 0:
            # assume snr is at index 2 within aux channel ordering: [zone, doppler, snr, density]
            # if your ordering differs, change the index accordingly
            snr_idx = 2
            # aux: (B, T, N, aux_dim)
            if snr_idx < aux.shape[-1]:
                snr_per_point = aux[..., snr_idx]  # (B, T, N)
                # Mean SNR per frame
                snr_mean = snr_per_point.mean(dim=2)  # (B, T)
            else:
                snr_mean = torch.zeros(B, T, device=device)
        else:
            snr_mean = torch.zeros(B, T, device=device)

        # Temporal attention pooling:
        # Compute raw logits per frame from features
        attn_logits = self.attn_fc(pooled_frames).squeeze(-1)  # (B, T)
        if self.snr_scale is not None:
            attn_logits = attn_logits + (self.snr_scale * snr_mean)

        attn_weights = torch.softmax(attn_logits, dim=1)  # (B, T)
        attended = (attn_weights.unsqueeze(-1) * pooled_frames).sum(dim=1)  # (B, F)

        # Short-term temporal-difference residual: mean(|F_{t+1} - F_t|) over t
        if self.use_diff_residual and T > 1:
            diffs = pooled_frames[:, 1:, :] - pooled_frames[:, :-1, :]  # (B, T-1, F)
            diff_stat = diffs.abs().mean(dim=1)                         # (B, F)
            feat = attended + diff_stat
        else:
            feat = attended

        # Classifier / output head
        # feat is (B, dense0) and we kept classifier MLP expecting dense_layers[0] as input
        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.output[f'branch_{i}'](feat))
            y = torch.stack(y, dim=1)
            return y

        logits = self.output(feat)  # (B, num_classes)
        return logits