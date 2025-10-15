# point_transformer_with_cross_attn.py
import torch
import torch.nn as nn
from torch.nn import Linear as Lin
from torch_geometric.nn import MLP, PointTransformerConv, fps, global_mean_pool, knn, knn_graph
from torch_geometric.utils import scatter


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)
        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None, plain_last=False)
        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    """
    Downsample via FPS + kNN pooling. Also allow optional aux pooling so aux tokens align with features.
    """
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch, aux=None):
        """
        x: (num_points, Cx)
        pos: (num_points, 3)
        batch: (num_points,)
        aux: (num_points, Caux) or None
        returns: out_features (num_clusters, Cout), sub_pos (num_clusters,3), sub_batch (num_clusters,),
                 aux_out (num_clusters, Caux) if aux provided else None, id_clusters (indices into original)
        """
        # FPS sampling (per-batch)
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # get batch indices for sampled
        sub_batch = batch[id_clusters] if batch is not None else None

        # knn from all points -> sampled centers (edges: (2, E) with [center_idx, neighbor_idx])
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch, batch_y=sub_batch)

        # transform features
        x_trans = self.mlp(x)

        # pool (max) neighbor features into cluster centers
        x_out = scatter(x_trans[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                        dim_size=id_clusters.size(0), reduce='max')

        aux_out = None
        if aux is not None:
            # For aux, we can either take the aux of sampled point directly, or pool neighbors similarly.
            # We'll pool aux values (mean) across neighbors for stability.
            aux_neighbor_vals = aux[id_k_neighbor[1]]  # (E, Caux)
            aux_out = scatter(aux_neighbor_vals, id_k_neighbor[0], dim=0,
                              dim_size=id_clusters.size(0), reduce='mean')

        sub_pos = pos[id_clusters]
        return x_out, sub_pos, sub_batch, aux_out, id_clusters


class AuxEncoder(nn.Module):
    """
    Encodes per-point aux features into per-stage aux tokens (one token projection per stage/dim).
    We create a small shared trunk, then per-stage projection heads to produce aux tokens that match
    geometry dims at each stage.
    """
    def __init__(self, aux_dim, stage_dims):
        """
        aux_dim: number of aux channels per point
        stage_dims: list of ints; token dim for each stage (e.g. [d0, d1, d2,...])
        """
        super().__init__()
        self.aux_dim = aux_dim
        self.trunk = MLP([aux_dim, 64, 64], plain_last=False)
        self.heads = nn.ModuleList([MLP([64, d], plain_last=True, norm=None) for d in stage_dims])

    def forward(self, aux):
        """
        aux: (num_points, aux_dim)
        returns: list of aux tokens per stage: [ (num_points, d0), (num_points, d1), ... ]
        """
        h = self.trunk(aux)
        outs = [head(h) for head in self.heads]
        return outs


class PointCrossAttention(nn.Module):
    """
    Lightweight point-wise cross-attention (cheap & effective).
    For each point i, compute:
      q = Wq * geom_i
      k = Wk * aux_i
      v = Wv * aux_i
      gate = sigmoid( (q * k).sum(dim=-1) / sqrt(dim) )
      out_i = Wout( v * gate.unsqueeze(-1) )
    This is O(N) and keeps point alignment; it's more expressive than simple concat/FiLM while being cheap.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.wq = Lin(dim, dim)
        self.wk = Lin(dim, dim)
        self.wv = Lin(dim, dim)
        self.out = Lin(dim, dim)
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)

    def forward(self, geom, aux):
        """
        geom: (num_tokens, dim)
        aux:  (num_tokens, dim)  (tokens aligned 1:1)
        returns: fused (num_tokens, dim)
        """
        q = self.wq(geom)          # (N, d)
        k = self.wk(aux)           # (N, d)
        v = self.wv(aux)           # (N, d)

        # per-point scalar gate
        gate = torch.sigmoid((q * k).sum(dim=-1) * self.scale)  # (N,)
        out = v * gate.unsqueeze(-1)  # (N, d)
        out = self.out(out)           # (N, d)
        # residual + norm
        fused = self.norm(geom + out)
        return fused


def weighted_global_mean_pool(x, weight, batch, eps=1e-6):
    """
    x: (num_points, C)
    weight: (num_points,) non-negative
    batch: (num_points,)
    returns: (batch_size, C)
    """
    if weight.dim() == 2:
        weight = weight.squeeze(-1)
    num = scatter(x * weight.unsqueeze(-1), batch, dim=0, reduce='sum')
    den = scatter(weight, batch, dim=0, reduce='sum').unsqueeze(-1).clamp(min=eps)
    return num / den


class AuxFormer(nn.Module):
    """
    Full model that integrates geometry backbone with per-stage aux projections and point-wise cross-attention fusion.
    Input data shape: (B, N, C) where C >= 7 and the aux slice is at indices:
       aux = [zone, doppler, snr, local_density]  (so aux_dim=4)
    Configurable options:
      - apply_cross_attn: whether to fuse aux via cross-attn
      - use_snr_pooling: whether to use snr as weighting for final pooling
    """
    def __init__(self,
                 info,
                 dim_model=[32, 64, 128, 256, 512],
                 k=16,
                 aux_dim=4,
                 apply_cross_attn=True,
                 use_snr_pooling=True,
                 fps_ratio=0.25):
        """
        info: dict with 'num_classes' or 'num_keypoints'
        dim_model: list of dims for each stage (len = num_stages)
        aux_dim: number of aux channels per point
        """
        super().__init__()
        self.k = k
        self.aux_dim = aux_dim
        self.apply_cross_attn = apply_cross_attn
        self.use_snr_pooling = use_snr_pooling
        self.fps_ratio = fps_ratio

        in_channels = 3
        self.num_classes = info.get('num_classes', None)
        out_channels = self.num_classes if self.num_classes is not None else info.get('num_keypoints')

        # Input MLP
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)
        self.transformer_input = TransformerBlock(in_channels=dim_model[0], out_channels=dim_model[0])

        # Build downsampling + transformer stacks
        self.transition_down = nn.ModuleList()
        self.transformers_down = nn.ModuleList()
        for i in range(len(dim_model) - 1):
            self.transition_down.append(TransitionDown(in_channels=dim_model[i],
                                                       out_channels=dim_model[i + 1],
                                                       ratio=self.fps_ratio,
                                                       k=self.k))
            self.transformers_down.append(TransformerBlock(in_channels=dim_model[i + 1],
                                                           out_channels=dim_model[i + 1]))

        # Aux encoder: produce per-stage aux tokens that match geometry dims.
        if aux_dim > 0 and apply_cross_attn:
            stage_dims = [dim_model[0]] + [dim_model[i + 1] for i in range(len(dim_model) - 1)]
            self.aux_encoder = AuxEncoder(aux_dim=aux_dim, stage_dims=stage_dims)
            # create a PointCrossAttention module per stage
            self.cross_attn_blocks = nn.ModuleList([PointCrossAttention(d) for d in stage_dims])
        else:
            self.aux_encoder = None
            self.cross_attn_blocks = None

        # Output head
        if self.num_classes is None:
            # keypoint mode
            num_points = info['num_keypoints']
            self.num_points = num_points
            point_branches = {}
            for i in range(num_points):
                point_branches[f'branch_{i}'] = MLP([dim_model[-1], 64, 3], norm=None)
            self.mlp_output = nn.ModuleDict(point_branches)
        else:
            self.mlp_output = MLP([dim_model[-1], 64, out_channels], norm=None)

    def forward(self, data):
        """
        data: (B, N, C) where C >= 7 with order:
              [x,y,z, zone, doppler, snr, local_density]
        returns: logits (B, num_classes)
        """
        assert data.dim() == 3
        B, N, C = data.shape
        device = data.device

        # Extract coords and aux
        coords = data[:, :, :3]  # (B, N, 3)
        # aux ordering assumption: zone, doppler, snr, local_density
        if self.aux_dim > 0:
            aux = data[:, :, 3:3 + self.aux_dim]  # (B, N, aux_dim)
            aux_flat = aux.reshape(B * N, self.aux_dim)  # (B*N, aux_dim)
        else:
            aux = None
            aux_flat = None

        # Flatten coords to (B*N, 3) and build batch vector for PyG ops
        pos = coords.reshape(B * N, 3)
        batch = torch.arange(B, device=device).repeat_interleave(N)  # (B*N,)

        # Build initial features (dummy features)
        x = pos  # using coords as initial features is ok; you may want to pass zeros or learned features
        x_feat = self.mlp_input(x)  # (B*N, dim0)

        # initial graph & transformer
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x_feat = self.transformer_input(x_feat, pos, edge_index)

        # Prepare aux tokens per stage if available
        if self.aux_encoder is not None and aux_flat is not None:
            aux_stage_tokens = self.aux_encoder(aux_flat)  # list of (B*N, d_stage)
            # aux_stage_tokens[i] corresponds to geometry before or after stage i pooling (we'll downsample them)
        else:
            aux_stage_tokens = None

        # If cross-attn is used, we will maintain aux_stage_tokens downsampled alongside features.
        # Stage 0 fusion: before any TransitionDown
        if self.apply_cross_attn and aux_stage_tokens is not None:
            # stage 0 dims correspond to x_feat channel dim
            aux_tokens_stage0 = aux_stage_tokens[0]  # (B*N, d0)
            # ensure dims match by linear projection if needed (aux encoder already sized to match)
            fused = self.cross_attn_blocks[0](x_feat, aux_tokens_stage0)
            x_feat = fused

        # Backbone: loop through TransitionDown + Transformer
        curr_pos = pos
        curr_batch = batch
        # Track current aux tokens (downsampled progressively)
        curr_aux_tokens = aux_stage_tokens[0] if aux_stage_tokens is not None else None  # stage 0: (B*N, d0)
        # Track cumulative indices mapping current positions back to original positions
        cumulative_indices = torch.arange(B * N, device=device)

        # For later stages we will compute aux_down for each stage using the TransitionDown pooling procedure
        for stage_idx, (td, tr) in enumerate(zip(self.transition_down, self.transformers_down)):
            # td will pool from current x_feat / curr_pos
            # Pass current aux tokens (aligned with curr_pos) to be downsampled
            x_out, sub_pos, sub_batch, aux_out, id_clusters = td(x_feat, curr_pos, curr_batch,
                                                                aux=curr_aux_tokens)
            # x_out: (num_clusters_total, dim_next)
            # aux_out: (num_clusters_total, d_next) or None
            # Note: id_clusters are indices into curr_pos (so alignment preserved)
            # Update feature & pos & batch for next stage
            x_feat = x_out
            curr_pos = sub_pos
            curr_batch = sub_batch

            # local graph and transformer on downsampled points
            edge_index = knn_graph(curr_pos, k=self.k, batch=curr_batch)
            x_feat = tr(x_feat, curr_pos, edge_index)

            # Update cumulative indices: map from new downsampled positions back to original
            cumulative_indices = cumulative_indices[id_clusters]

            # cross-attn fusion for this stage if configured
            if self.apply_cross_attn and self.cross_attn_blocks is not None:
                if aux_stage_tokens is not None and stage_idx + 1 < len(self.cross_attn_blocks):
                    # Use cumulative_indices to map from current downsampled positions to original positions
                    # Then get the next stage aux tokens for those original positions
                    aux_for_fusion = aux_stage_tokens[stage_idx + 1][cumulative_indices]
                    x_feat = self.cross_attn_blocks[stage_idx + 1](x_feat, aux_for_fusion)

            # Update curr_aux_tokens for next iteration: use next stage's aux tokens mapped via cumulative indices
            if aux_stage_tokens is not None and stage_idx + 1 < len(aux_stage_tokens):
                curr_aux_tokens = aux_stage_tokens[stage_idx + 1][cumulative_indices]
            else:
                curr_aux_tokens = None

        # Final pooling. Optionally use SNR as weight. We need weights aligned to current x positions.
        if self.use_snr_pooling and (aux_flat is not None):
            # Use cumulative_indices to map SNR from original positions to current downsampled positions
            snr_orig = aux_flat[:, 2]  # (B*N,) - SNR is at index 2 in aux features
            mapped_weights = snr_orig[cumulative_indices].clamp(min=0.0)  # (M_final,)
            # Normalize weights to [0, 1]
            mapped_weights = torch.sigmoid(mapped_weights)
            pooled = weighted_global_mean_pool(x_feat, mapped_weights, curr_batch)
        else:
            pooled = global_mean_pool(x_feat, curr_batch)

        # outputs
        if self.num_classes is None:
            y = []
            for i in range(self.num_points):
                y.append(self.mlp_output[f'branch_{i}'](pooled))
            y = torch.stack(y, dim=1)
            return y

        out = self.mlp_output(pooled)  # (B, num_classes)
        return out
