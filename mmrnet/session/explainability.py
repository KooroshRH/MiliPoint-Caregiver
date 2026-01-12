"""
Explainability module for DGCNN-AFTNet and other point cloud models.

Provides:
1. Point Saliency Maps - gradient-based importance scores per point
2. FiLM Weight Visualization - shows how auxiliary features modulate spatial features
3. Temporal Attention Visualization - shows which frames are most important
4. Critical Points Extraction - points that survive max-pooling

Generates visualizations for True Positives and False Negatives.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from tqdm import tqdm
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import TSNE
import seaborn as sns


class PointCloudExplainer:
    """
    Explainability tools for point cloud classification models.
    """

    def __init__(self, model, device='cuda'):
        """
        Args:
            model: The trained model (e.g., DGCNNAuxFusionT)
            device: Device to run computations on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        # Storage for intermediate activations and gradients
        self.activations = {}
        self.gradients = {}
        self.film_params = {}
        self.embeddings = None  # For storing feature embeddings

    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients."""
        self.hooks = []

        # Hook for capturing FiLM parameters if available
        def get_film_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'aux_mlp') and module.aux_mlp is not None:
                    # Store the FiLM gamma/beta values
                    self.film_params[name] = output.detach()
            return hook

        # Register hooks on EdgeConvAuxLayer if present
        if hasattr(self.model, 'edge_layers'):
            for i, layer in enumerate(self.model.edge_layers):
                hook = layer.register_forward_hook(get_film_hook(f'edge_layer_{i}'))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_point_saliency(self, x, target_class=None):
        """
        Compute saliency scores for each point using gradient-based method.

        Points are shifted toward centroid (differentiable approximation of point dropping)
        and gradients are computed w.r.t. the input.

        Args:
            x: Input tensor (B, T, N, C) or (B, N, C)
            target_class: Target class for gradient computation. If None, uses predicted class.

        Returns:
            saliency: (B, T, N) or (B, N) saliency scores per point
        """
        x = x.to(self.device)
        x.requires_grad_(True)

        self.model.eval()

        # Forward pass
        output = self.model(x)

        if target_class is None:
            target_class = output.argmax(dim=1)

        # Ensure target_class is on the same device as output
        target_class = target_class.to(output.device)

        # Compute gradient of target class score w.r.t. input
        self.model.zero_grad()

        # One-hot encode target
        one_hot = torch.zeros_like(output).to(output.device)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1)

        # Backward pass
        output.backward(gradient=one_hot)

        # Get gradients
        gradients = x.grad.detach()

        # Compute saliency as L2 norm of gradients across feature dimension
        if x.dim() == 4:  # (B, T, N, C)
            saliency = gradients.norm(dim=-1)  # (B, T, N)
        else:  # (B, N, C)
            saliency = gradients.norm(dim=-1)  # (B, N)

        # Normalize to [0, 1]
        saliency_min = saliency.view(saliency.size(0), -1).min(dim=1, keepdim=True)[0]
        saliency_max = saliency.view(saliency.size(0), -1).max(dim=1, keepdim=True)[0]

        if x.dim() == 4:
            saliency_min = saliency_min.unsqueeze(-1)
            saliency_max = saliency_max.unsqueeze(-1)

        saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)

        return saliency.detach().cpu().numpy()

    def compute_point_saliency_by_dropping(self, x, target_class=None, num_iterations=10):
        """
        Compute saliency by iteratively shifting points toward centroid.
        More accurate but slower than gradient-based method.

        Args:
            x: Input tensor (B, T, N, C) or (B, N, C)
            target_class: Target class
            num_iterations: Number of shift iterations

        Returns:
            saliency: Saliency scores per point
        """
        x = x.to(self.device)
        original_x = x.clone()

        self.model.eval()

        with torch.no_grad():
            output = self.model(x)
            if target_class is None:
                target_class = output.argmax(dim=1)
            original_scores = output.gather(1, target_class.unsqueeze(1)).squeeze()

        # Get XYZ coordinates
        if x.dim() == 4:  # (B, T, N, C)
            B, T, N, C = x.shape
            xyz = x[:, :, :, :3]
            centroid = xyz.mean(dim=2, keepdim=True)  # (B, T, 1, 3)
        else:  # (B, N, C)
            B, N, C = x.shape
            T = 1
            xyz = x[:, :, :3]
            centroid = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)

        saliency = torch.zeros(B, T, N).to(self.device) if x.dim() == 4 else torch.zeros(B, N).to(self.device)

        for iteration in range(num_iterations):
            alpha = (iteration + 1) / num_iterations

            for point_idx in range(N):
                # Create perturbed input
                x_perturbed = original_x.clone()

                if x.dim() == 4:
                    # Shift point toward centroid
                    x_perturbed[:, :, point_idx, :3] = (
                        (1 - alpha) * original_x[:, :, point_idx, :3] +
                        alpha * centroid.squeeze(2)
                    )
                else:
                    x_perturbed[:, point_idx, :3] = (
                        (1 - alpha) * original_x[:, point_idx, :3] +
                        alpha * centroid.squeeze(1)
                    )

                with torch.no_grad():
                    perturbed_output = self.model(x_perturbed)
                    perturbed_scores = perturbed_output.gather(1, target_class.unsqueeze(1)).squeeze()

                # Score drop indicates point importance
                score_drop = (original_scores - perturbed_scores).clamp(min=0)

                if x.dim() == 4:
                    saliency[:, :, point_idx] += score_drop.unsqueeze(1).expand(-1, T) / num_iterations
                else:
                    saliency[:, point_idx] += score_drop / num_iterations

        # Normalize
        saliency_min = saliency.view(B, -1).min(dim=1, keepdim=True)[0]
        saliency_max = saliency.view(B, -1).max(dim=1, keepdim=True)[0]

        if x.dim() == 4:
            saliency_min = saliency_min.unsqueeze(-1)
            saliency_max = saliency_max.unsqueeze(-1)

        saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)

        return saliency.detach().cpu().numpy()

    def extract_film_weights(self, x):
        """
        Extract FiLM modulation weights (gamma, beta) from the model.

        Args:
            x: Input tensor

        Returns:
            dict with gamma and beta values per layer
        """
        self.film_params = {}
        self._register_hooks()

        x = x.to(self.device)

        with torch.no_grad():
            _ = self.model(x)

        self._remove_hooks()

        # Process FiLM parameters
        film_weights = {}
        for name, params in self.film_params.items():
            if params is not None:
                d = params.shape[-1] // 2
                gamma = params[..., :d]
                beta = params[..., d:]
                film_weights[name] = {
                    'gamma': gamma.cpu().numpy(),
                    'beta': beta.cpu().numpy()
                }

        return film_weights

    def extract_temporal_attention(self, x):
        """
        Extract temporal attention weights if model has temporal transformer.

        Args:
            x: Input tensor (B, T, N, C)

        Returns:
            attention_weights: (B, T) importance per frame
        """
        if not hasattr(self.model, 'temporal_encoder') or self.model.temporal_encoder is None:
            logging.warning("Model does not have temporal encoder")
            return None

        x = x.to(self.device)

        # We need to capture intermediate outputs
        # This is model-specific - for DGCNNAuxFusionT
        temporal_importance = []

        def hook_fn(module, input, output):
            # Capture attention output
            temporal_importance.append(output.detach())

        # Register hook on temporal encoder
        hook = self.model.temporal_encoder.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self.model(x)

        hook.remove()

        if temporal_importance:
            # Compute frame importance as mean activation magnitude
            seq_out = temporal_importance[0]  # (B, T, D)
            frame_importance = seq_out.norm(dim=-1)  # (B, T)

            # Normalize
            frame_importance = frame_importance / frame_importance.sum(dim=1, keepdim=True)

            return frame_importance.cpu().numpy()

        return None

    def extract_embeddings(self, x):
        """
        Extract feature embeddings from the model before final classification layer.

        Args:
            x: Input tensor (B, T, N, C) or (B, N, C)

        Returns:
            embeddings: (B, D) feature embeddings
        """
        x = x.to(self.device)

        # Store embeddings using a hook
        embeddings = []

        def hook_fn(module, input):
            # Pre-hook receives (module, input) only - no output
            # For MLP, the input is a tuple, get first element
            if isinstance(input, tuple):
                embeddings.append(input[0].detach())
            else:
                embeddings.append(input.detach())

        # Find the output layer and hook before it
        hook = None
        if hasattr(self.model, 'output'):
            # For DGCNNAuxFusionT and similar models with self.output
            hook = self.model.output.register_forward_pre_hook(hook_fn)
        elif hasattr(self.model, 'linear'):
            # Hook before final linear layer
            hook = self.model.linear.register_forward_pre_hook(hook_fn)
        elif hasattr(self.model, 'fc'):
            hook = self.model.fc.register_forward_pre_hook(hook_fn)
        elif hasattr(self.model, 'classifier'):
            hook = self.model.classifier.register_forward_pre_hook(hook_fn)

        with torch.no_grad():
            _ = self.model(x)

        if hook is not None:
            hook.remove()

        if embeddings:
            emb = embeddings[0]
            # Handle case where embedding might have extra dimensions
            if emb.dim() > 2:
                emb = emb.view(emb.size(0), -1)
            return emb.cpu().numpy()
        else:
            # Fallback: use model output as embeddings
            logging.warning("Could not find embedding layer, using model output")
            with torch.no_grad():
                output = self.model(x)
            return output.cpu().numpy()


def visualize_point_saliency_3d(points, saliency, title="Point Saliency", save_path=None,
                                  frame_idx=None, view_angles=(30, 45), radar_height=2.20):
    """
    Visualize point cloud with saliency coloring in 3D from radar's perspective.

    Args:
        points: (N, 3) or (T, N, 3) point coordinates in radar frame
        saliency: (N,) or (T, N) saliency scores
        title: Plot title
        save_path: Path to save figure
        frame_idx: If temporal data, which frame to visualize (None = average)
        view_angles: (elevation, azimuth) for 3D view
        radar_height: Height of radar above ground (default 2.20m)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    if points.ndim == 3:  # Temporal data (T, N, 3)
        if frame_idx is not None:
            pts = points[frame_idx].copy()
            sal = saliency[frame_idx]
            title += f" (Frame {frame_idx})"
        else:
            # Average across frames
            pts = points.reshape(-1, 3).copy()
            sal = saliency.flatten()
            title += " (All Frames)"
    else:
        pts = points.copy()
        sal = saliency

    # Adjust Z coordinates: radar is at height 2.20m, so add radar_height to z
    # This converts from radar frame (z=0 at radar) to ground frame (z=0 at ground)
    pts[:, 2] = pts[:, 2] + radar_height

    # Create colormap
    norm = Normalize(vmin=sal.min(), vmax=sal.max())
    colors = cm.hot(norm(sal))

    scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=sal, cmap='hot', s=50, alpha=0.8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m) - Height from Ground')
    ax.set_title(title)

    # Set viewing angle from radar's perspective
    # Radar is at (0, 0, radar_height) looking down and forward
    # elevation: angle above horizontal (negative = looking down)
    # azimuth: rotation around z-axis
    ax.view_init(elev=view_angles[0], azim=view_angles[1])

    # Add radar position marker
    ax.scatter([0], [0], [radar_height], c='red', marker='^', s=200,
              label=f'Radar Position (0, 0, {radar_height}m)', edgecolors='black', linewidths=2)
    ax.legend(loc='upper right')

    plt.colorbar(scatter, ax=ax, label='Saliency Score')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_point_saliency_2d(points, saliency, title="Point Saliency", save_path=None,
                                  frame_idx=None, radar_height=2.20):
    """
    Visualize point cloud saliency in 2D projections (XY, XZ, YZ) from radar's perspective.

    Args:
        points: (N, 3) or (T, N, 3) point coordinates in radar frame
        saliency: (N,) or (T, N) saliency scores
        title: Plot title
        save_path: Path to save figure
        frame_idx: If temporal data, which frame to visualize
        radar_height: Height of radar above ground (default 2.20m)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if points.ndim == 3:
        if frame_idx is not None:
            pts = points[frame_idx].copy()
            sal = saliency[frame_idx]
            title += f" (Frame {frame_idx})"
        else:
            pts = points.reshape(-1, 3).copy()
            sal = saliency.flatten()
            title += " (All Frames)"
    else:
        pts = points.copy()
        sal = saliency

    # Adjust Z coordinates: radar is at height 2.20m
    pts[:, 2] = pts[:, 2] + radar_height

    projections = [
        (pts[:, 0], pts[:, 1], 'X (m)', 'Y (m)', 'XY Projection (Top View)'),
        (pts[:, 0], pts[:, 2], 'X (m)', 'Z (m) - Height', 'XZ Projection (Side View)'),
        (pts[:, 1], pts[:, 2], 'Y (m)', 'Z (m) - Height', 'YZ Projection (Front View)')
    ]

    for ax, (x, y, xlabel, ylabel, proj_title) in zip(axes, projections):
        scatter = ax.scatter(x, y, c=sal, cmap='hot', s=30, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(proj_title)
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Saliency')

        # Add radar position marker (x=0, y=0, z=radar_height)
        if 'X' in xlabel and 'Y' in ylabel:  # XY projection
            ax.scatter([0], [0], c='red', marker='^', s=100, edgecolors='black',
                      linewidths=1.5, label='Radar', zorder=10)
            ax.legend(loc='upper right', fontsize=8)
        elif 'X' in xlabel and 'Z' in ylabel:  # XZ projection
            ax.axhline(y=radar_height, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Radar Height')
            ax.legend(loc='upper right', fontsize=8)
        elif 'Y' in xlabel and 'Z' in ylabel:  # YZ projection
            ax.axhline(y=radar_height, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Radar Height')
            ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_temporal_importance(frame_importance, title="Temporal Importance", save_path=None):
    """
    Visualize frame importance over time.

    Args:
        frame_importance: (T,) importance scores per frame
        title: Plot title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    frames = np.arange(len(frame_importance))

    ax.bar(frames, frame_importance, color='steelblue', alpha=0.8)
    ax.plot(frames, frame_importance, 'r-', linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Importance Score')
    ax.set_title(title)
    ax.set_xlim(-0.5, len(frame_importance) - 0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_film_modulation(gamma, beta, aux_names=None, title="FiLM Modulation", save_path=None):
    """
    Visualize FiLM gamma (scale) and beta (shift) parameters.

    Args:
        gamma: (N, D) or mean gamma values
        beta: (N, D) or mean beta values
        aux_names: Names of auxiliary features
        title: Plot title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Average across points if needed
    if gamma.ndim > 1:
        gamma_mean = gamma.mean(axis=0)
        gamma_std = gamma.std(axis=0)
        beta_mean = beta.mean(axis=0)
        beta_std = beta.std(axis=0)
    else:
        gamma_mean = gamma
        gamma_std = None
        beta_mean = beta
        beta_std = None

    x = np.arange(len(gamma_mean))

    # Gamma plot
    axes[0].bar(x, gamma_mean, color='coral', alpha=0.8)
    if gamma_std is not None:
        axes[0].errorbar(x, gamma_mean, yerr=gamma_std, fmt='none', color='black', capsize=2)
    axes[0].axhline(y=1.0, color='gray', linestyle='--', label='Neutral (1.0)')
    axes[0].set_xlabel('Feature Dimension')
    axes[0].set_ylabel('Gamma (Scale)')
    axes[0].set_title('FiLM Gamma Values')
    axes[0].legend()

    # Beta plot
    axes[1].bar(x, beta_mean, color='steelblue', alpha=0.8)
    if beta_std is not None:
        axes[1].errorbar(x, beta_mean, yerr=beta_std, fmt='none', color='black', capsize=2)
    axes[1].axhline(y=0.0, color='gray', linestyle='--', label='Neutral (0.0)')
    axes[1].set_xlabel('Feature Dimension')
    axes[1].set_ylabel('Beta (Shift)')
    axes[1].set_title('FiLM Beta Values')
    axes[1].legend()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_doppler_saliency_correlation(points, saliency, aux_features, title="Doppler-Saliency Correlation", save_path=None):
    """
    Visualize correlation between auxiliary features (Doppler, SNR, Density) and saliency scores.

    Args:
        points: (N, 3) or (T, N, 3) XYZ coordinates
        saliency: (N,) or (T, N) saliency scores
        aux_features: (N, K) or (T, N, K) auxiliary features [doppler, snr, density, ...]
        title: Plot title
        save_path: Path to save figure
    """
    # Flatten temporal dimension if present
    if points.ndim == 3:
        points = points.reshape(-1, 3)
        saliency = saliency.flatten()
        if aux_features.ndim == 3:
            aux_features = aux_features.reshape(-1, aux_features.shape[-1])

    # Extract auxiliary features
    if aux_features.shape[1] >= 3:
        doppler = aux_features[:, 0]
        snr = aux_features[:, 1]
        density = aux_features[:, 2]
        feature_names = ['Doppler Velocity', 'SNR', 'Density']
        features = [doppler, snr, density]
    elif aux_features.shape[1] == 1:
        doppler = aux_features[:, 0]
        feature_names = ['Doppler Velocity']
        features = [doppler]
    else:
        logging.warning("Unexpected auxiliary feature dimension")
        return

    num_features = len(features)
    fig, axes = plt.subplots(2, num_features, figsize=(6 * num_features, 10))
    if num_features == 1:
        axes = axes.reshape(-1, 1)

    for i, (feature, feat_name) in enumerate(zip(features, feature_names)):
        # Remove NaN/Inf values for correlation
        valid_mask = ~(np.isnan(feature) | np.isinf(feature) | np.isnan(saliency) | np.isinf(saliency))
        feat_valid = feature[valid_mask]
        sal_valid = saliency[valid_mask]

        if len(feat_valid) < 2:
            continue

        # Top row: Scatter plot
        axes[0, i].scatter(feat_valid, sal_valid, alpha=0.5, s=10, c=sal_valid, cmap='hot')
        axes[0, i].set_xlabel(feat_name)
        axes[0, i].set_ylabel('Saliency Score')
        axes[0, i].set_title(f'{feat_name} vs Saliency')

        # Add trend line
        z = np.polyfit(feat_valid, sal_valid, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(feat_valid.min(), feat_valid.max(), 100)
        axes[0, i].plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'y={z[0]:.3f}x+{z[1]:.3f}')
        axes[0, i].legend()

        # Compute correlations
        try:
            pearson_r, pearson_p = pearsonr(feat_valid, sal_valid)
            spearman_r, spearman_p = spearmanr(feat_valid, sal_valid)
            corr_text = f'Pearson: r={pearson_r:.3f}, p={pearson_p:.4f}\nSpearman: ρ={spearman_r:.3f}, p={spearman_p:.4f}'
            axes[0, i].text(0.05, 0.95, corr_text, transform=axes[0, i].transAxes,
                          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        except:
            pass

        # Bottom row: 2D histogram (heatmap)
        h, xedges, yedges = np.histogram2d(feat_valid, sal_valid, bins=30)
        h = h.T  # Transpose for correct orientation
        im = axes[1, i].imshow(h, origin='lower', aspect='auto', cmap='YlOrRd',
                              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        axes[1, i].set_xlabel(feat_name)
        axes[1, i].set_ylabel('Saliency Score')
        axes[1, i].set_title(f'Density Heatmap: {feat_name} vs Saliency')
        plt.colorbar(im, ax=axes[1, i], label='Count')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_tsne_risk_categorization(embeddings, labels, predictions, risk_map, class_names=None,
                                        title="t-SNE: Risk-Based Categorization", save_path=None):
    """
    Visualize t-SNE embeddings colored by risk categories.

    Args:
        embeddings: (N, D) feature embeddings
        labels: (N,) true labels
        predictions: (N,) predicted labels
        risk_map: dict mapping class labels to risk levels (0, 1, 2, 3)
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
    """
    logging.info("Computing t-SNE (this may take a few minutes)...")

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    embeddings_2d = tsne.fit_transform(embeddings)

    # Map labels to risk categories
    risk_labels = np.array([risk_map.get(int(label), -1) for label in labels])
    risk_preds = np.array([risk_map.get(int(pred), -1) for pred in predictions])

    # Determine correct/incorrect predictions
    correct_mask = labels == predictions

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Define risk colors
    risk_colors = {0: '#1f77b4', 1: '#2ca02c', 2: '#ff7f0e', 3: '#d62728'}  # Blue, Green, Orange, Red
    risk_labels_names = {0: 'Risk 0 (Start-Walking)', 1: 'Risk 1 (Low)', 2: 'Risk 2 (Medium)', 3: 'Risk 3 (High)'}

    # Left plot: True risk categories
    for risk_level in sorted(risk_colors.keys()):
        mask = risk_labels == risk_level
        if np.any(mask):
            axes[0].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                          c=risk_colors[risk_level], label=risk_labels_names[risk_level],
                          alpha=0.6, s=50, edgecolors='black', linewidths=0.5)

    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0].set_title('True Risk Categories', fontsize=14)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(alpha=0.3)

    # Right plot: Correct vs Incorrect predictions
    # Correct predictions
    axes[1].scatter(embeddings_2d[correct_mask, 0], embeddings_2d[correct_mask, 1],
                   c=[risk_colors[r] for r in risk_labels[correct_mask]],
                   marker='o', alpha=0.6, s=50, edgecolors='black', linewidths=0.5, label='Correct')

    # Incorrect predictions
    axes[1].scatter(embeddings_2d[~correct_mask, 0], embeddings_2d[~correct_mask, 1],
                   c=[risk_colors[r] for r in risk_labels[~correct_mask]],
                   marker='X', alpha=0.8, s=100, edgecolors='red', linewidths=2, label='Incorrect')

    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[1].set_title('Correct vs Incorrect Predictions', fontsize=14)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    logging.info("✓ t-SNE visualization complete")

    # Additional analysis: Confusion between risk categories
    fig2, ax = plt.subplots(figsize=(8, 6))

    # Create confusion matrix for risk categories
    from sklearn.metrics import confusion_matrix
    risk_cm = confusion_matrix(risk_labels, risk_preds, labels=[0, 1, 2, 3])

    sns.heatmap(risk_cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Risk 0', 'Risk 1', 'Risk 2', 'Risk 3'],
                yticklabels=['Risk 0', 'Risk 1', 'Risk 2', 'Risk 3'])
    ax.set_xlabel('Predicted Risk Category')
    ax.set_ylabel('True Risk Category')
    ax.set_title('Confusion Matrix: Risk-Based Categories')

    plt.tight_layout()

    if save_path:
        risk_cm_path = save_path.replace('.png', '_risk_confusion.png')
        plt.savefig(risk_cm_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def run_explainability_analysis(model, test_loader, class_names, output_dir,
                                  num_samples=5, device='cuda', test_dataset=None):
    """
    Run complete explainability analysis on test data.
    Generates visualizations for True Positives and False Negatives.

    Args:
        model: Trained model
        test_loader: Test data loader
        class_names: List of class names
        output_dir: Directory to save visualizations
        num_samples: Number of samples per category (TP, FN) to visualize
        device: Device to use
        test_dataset: Optional dataset object to access metadata (subject_id, scenario_id)
    """
    logging.info("="*70)
    logging.info("EXPLAINABILITY ANALYSIS STARTED")
    logging.info("="*70)
    logging.info(f"Model type: {type(model).__name__}")
    logging.info(f"Device: {device}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Number of samples: {num_samples}")
    logging.info(f"Test loader batches: {len(test_loader)}")
    logging.info(f"Class names provided: {class_names is not None}")
    logging.info(f"Test dataset provided: {test_dataset is not None}")

    logging.info("\nCreating output directories...")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'true_positives'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'false_negatives'), exist_ok=True)
    logging.info("✓ Directories created")

    logging.info("\nInitializing PointCloudExplainer...")
    explainer = PointCloudExplainer(model, device)
    logging.info("✓ Explainer initialized")

    # Collect predictions and metadata
    all_data = []
    all_labels = []
    all_preds = []
    all_metadata = []  # Store subject_id, scenario_id for each sample

    logging.info("\n" + "-"*70)
    logging.info("STEP 1: Collecting predictions from test data")
    logging.info("-"*70)
    model.eval()
    logging.info("Model set to eval mode")

    sample_idx = 0
    batch_count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            x, y = batch[0].to(device), batch[1].to(device)
            output = model(x)
            preds = output.argmax(dim=1)

            all_data.append(x.cpu())
            all_labels.append(y.cpu())
            all_preds.append(preds.cpu())

            batch_count += 1
            if batch_count % 10 == 0:
                logging.info(f"  Processed {batch_count} batches, {sample_idx} samples so far...")

            # Collect metadata if dataset is available
            batch_size = x.shape[0]
            for i in range(batch_size):
                if test_dataset is not None and hasattr(test_dataset, 'data'):
                    data_item = test_dataset.data[sample_idx]
                    metadata = {
                        'subject_id': data_item.get('subject_id', 'unknown'),
                        'scenario_id': data_item.get('scenario_id', 'unknown'),
                        'sample_idx': sample_idx
                    }
                else:
                    metadata = {
                        'subject_id': 'unknown',
                        'scenario_id': 'unknown',
                        'sample_idx': sample_idx
                    }
                all_metadata.append(metadata)
                sample_idx += 1

    logging.info(f"✓ Collected {len(all_data)} batches")

    all_data = torch.cat(all_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)

    logging.info(f"Total samples: {len(all_data)}")
    logging.info(f"Data shape: {all_data.shape}")

    # Identify True Positives and False Negatives
    logging.info("\n" + "-"*70)
    logging.info("STEP 2: Identifying True Positives and False Negatives")
    logging.info("-"*70)
    correct_mask = all_preds == all_labels
    tp_indices = torch.where(correct_mask)[0]
    fn_indices = torch.where(~correct_mask)[0]

    logging.info(f"✓ Found {len(tp_indices)} True Positives")
    logging.info(f"✓ Found {len(fn_indices)} False Negatives")
    logging.info(f"Accuracy: {len(tp_indices) / len(all_labels) * 100:.2f}%")

    # Sample indices - Use deterministic sampling for fair model comparison
    logging.info(f"\nSampling {num_samples} True Positives and {num_samples} False Negatives...")
    logging.info("Using deterministic sampling (fixed seed=42) for reproducibility across models")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Sample with fixed seed - this ensures same samples are selected across different models
    tp_sample_idx = tp_indices[torch.randperm(len(tp_indices), generator=torch.Generator().manual_seed(42))[:num_samples]].tolist()
    fn_sample_idx = fn_indices[torch.randperm(len(fn_indices), generator=torch.Generator().manual_seed(42))[:num_samples]].tolist()

    logging.info(f"✓ Sampled {len(tp_sample_idx)} TP and {len(fn_sample_idx)} FN (deterministic)")
    logging.info(f"  TP sample indices: {tp_sample_idx}")
    logging.info(f"  FN sample indices: {fn_sample_idx}")

    # ========== NEW: t-SNE Risk-Based Categorization Visualization ==========
    logging.info("\n" + "-"*70)
    logging.info("STEP 2.5: Generating t-SNE Risk-Based Categorization")
    logging.info("-"*70)

    # Define risk mapping (from wrapper.py)
    risk_map = {
        0: 1, 1: 3, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2,
        8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 3, 15: 1,
        16: 3, 17: 1, 18: 1, 19: 3, 20: 1, 21: 2, 22: 1, 23: 3,
        24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 0
    }

    try:
        # Extract embeddings for all samples
        logging.info("Extracting feature embeddings from all test samples...")
        all_embeddings = []
        batch_size = 32  # Process in batches to avoid memory issues
        for i in tqdm(range(0, len(all_data), batch_size), desc="Extracting embeddings"):
            batch = all_data[i:i+batch_size].to(device)
            emb = explainer.extract_embeddings(batch)
            all_embeddings.append(emb)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        logging.info(f"✓ Extracted embeddings with shape: {all_embeddings.shape}")

        # Generate t-SNE visualization
        visualize_tsne_risk_categorization(
            embeddings=all_embeddings,
            labels=all_labels.numpy(),
            predictions=all_preds.numpy(),
            risk_map=risk_map,
            class_names=class_names,
            title="t-SNE: Risk-Based Categorization (All Test Samples)",
            save_path=os.path.join(output_dir, 'tsne_risk_categorization.png')
        )
        logging.info("✓ t-SNE visualization saved")

    except Exception as e:
        logging.error(f"✗ t-SNE visualization failed: {e}")
        import traceback
        logging.error(traceback.format_exc())

    # ========================================================================

    # Process True Positives
    logging.info("\n" + "-"*70)
    logging.info("STEP 3: Generating True Positive visualizations")
    logging.info("-"*70)
    for i, idx in enumerate(tqdm(tp_sample_idx, desc="True Positives")):
        logging.info(f"Processing TP {i+1}/{len(tp_sample_idx)}: sample index {idx}")
        x = all_data[idx:idx+1].to(device)  # Ensure data is on correct device
        label = all_labels[idx].item()
        pred = all_preds[idx].item()
        metadata = all_metadata[idx]

        # Compute saliency - create target_class tensor on correct device
        saliency = explainer.compute_point_saliency(x, target_class=torch.tensor([pred], device=device))

        # Get points (XYZ) - detach and move to CPU first
        if x.dim() == 4:  # (1, T, N, C)
            points = x[0, :, :, :3].detach().cpu().numpy()
            sal = saliency[0]
        else:  # (1, N, C)
            points = x[0, :, :3].detach().cpu().numpy()
            sal = saliency[0]

        class_name = class_names[label] if class_names else str(label)
        subject_id = metadata['subject_id']
        scenario_id = metadata['scenario_id']
        sample_idx_str = metadata['sample_idx']

        # Create info string for titles
        info_str = f"Subject: {subject_id}, Scenario: {scenario_id}"

        # 3D visualization
        visualize_point_saliency_3d(
            points, sal,
            title=f"TP: {class_name}\n{info_str}",
            save_path=os.path.join(output_dir, 'true_positives', f'tp_{i}_3d_subj{subject_id}_scen{scenario_id}_{class_name}.png')
        )

        # 2D projections
        visualize_point_saliency_2d(
            points, sal,
            title=f"TP: {class_name}\n{info_str}",
            save_path=os.path.join(output_dir, 'true_positives', f'tp_{i}_2d_subj{subject_id}_scen{scenario_id}_{class_name}.png')
        )

        # Temporal importance if available
        if x.dim() == 4:
            temp_imp = explainer.extract_temporal_attention(x)
            if temp_imp is not None:
                visualize_temporal_importance(
                    temp_imp[0],
                    title=f"TP: {class_name} - Temporal Importance\n{info_str}",
                    save_path=os.path.join(output_dir, 'true_positives', f'tp_{i}_temporal_subj{subject_id}_scen{scenario_id}_{class_name}.png')
                )

        # ========== NEW: Doppler-Saliency Correlation Visualization ==========
        # Extract auxiliary features if available
        if x.dim() == 4:  # Temporal: (1, T, N, C)
            if x.shape[-1] >= 7:  # Has [x, y, z, zone, doppler, snr, density]
                aux_features = x[0, :, :, 4:7].detach().cpu().numpy()  # (T, N, 3)
                visualize_doppler_saliency_correlation(
                    points, sal, aux_features,
                    title=f"TP: {class_name} - Doppler-Saliency Correlation\n{info_str}",
                    save_path=os.path.join(output_dir, 'true_positives', f'tp_{i}_doppler_corr_subj{subject_id}_scen{scenario_id}_{class_name}.png')
                )
            elif x.shape[-1] >= 5:  # Has [x, y, z, zone, doppler]
                aux_features = x[0, :, :, 4:5].detach().cpu().numpy()  # (T, N, 1)
                visualize_doppler_saliency_correlation(
                    points, sal, aux_features,
                    title=f"TP: {class_name} - Doppler-Saliency Correlation\n{info_str}",
                    save_path=os.path.join(output_dir, 'true_positives', f'tp_{i}_doppler_corr_subj{subject_id}_scen{scenario_id}_{class_name}.png')
                )
        elif x.dim() == 3:  # Non-temporal: (1, N, C)
            if x.shape[-1] >= 7:  # Has [x, y, z, zone, doppler, snr, density]
                aux_features = x[0, :, 4:7].detach().cpu().numpy()  # (N, 3)
                visualize_doppler_saliency_correlation(
                    points, sal, aux_features,
                    title=f"TP: {class_name} - Doppler-Saliency Correlation\n{info_str}",
                    save_path=os.path.join(output_dir, 'true_positives', f'tp_{i}_doppler_corr_subj{subject_id}_scen{scenario_id}_{class_name}.png')
                )
            elif x.shape[-1] >= 5:  # Has [x, y, z, zone, doppler]
                aux_features = x[0, :, 4:5].detach().cpu().numpy()  # (N, 1)
                visualize_doppler_saliency_correlation(
                    points, sal, aux_features,
                    title=f"TP: {class_name} - Doppler-Saliency Correlation\n{info_str}",
                    save_path=os.path.join(output_dir, 'true_positives', f'tp_{i}_doppler_corr_subj{subject_id}_scen{scenario_id}_{class_name}.png')
                )
        # ====================================================================

    # Process False Negatives
    logging.info("\n" + "-"*70)
    logging.info("STEP 4: Generating False Negative visualizations")
    logging.info("-"*70)
    for i, idx in enumerate(tqdm(fn_sample_idx, desc="False Negatives")):
        logging.info(f"Processing FN {i+1}/{len(fn_sample_idx)}: sample index {idx}")
        x = all_data[idx:idx+1].to(device)  # Ensure data is on correct device
        label = all_labels[idx].item()
        pred = all_preds[idx].item()
        metadata = all_metadata[idx]

        # Compute saliency for both true and predicted class - create tensors on correct device
        saliency_true = explainer.compute_point_saliency(x, target_class=torch.tensor([label], device=device))
        saliency_pred = explainer.compute_point_saliency(x, target_class=torch.tensor([pred], device=device))

        # Get points - detach and move to CPU first
        if x.dim() == 4:
            points = x[0, :, :, :3].detach().cpu().numpy()
            sal_true = saliency_true[0]
            sal_pred = saliency_pred[0]
        else:
            points = x[0, :, :3].detach().cpu().numpy()
            sal_true = saliency_true[0]
            sal_pred = saliency_pred[0]

        true_class_name = class_names[label] if class_names else str(label)
        pred_class_name = class_names[pred] if class_names else str(pred)
        subject_id = metadata['subject_id']
        scenario_id = metadata['scenario_id']

        # Create info string for titles
        info_str = f"Subject: {subject_id}, Scenario: {scenario_id}"

        # Saliency for true class
        visualize_point_saliency_3d(
            points, sal_true,
            title=f"FN: True={true_class_name}, Pred={pred_class_name}\n{info_str}\n(Saliency for True Class)",
            save_path=os.path.join(output_dir, 'false_negatives', f'fn_{i}_3d_true_subj{subject_id}_scen{scenario_id}_{true_class_name}.png')
        )

        # Saliency for predicted class
        visualize_point_saliency_3d(
            points, sal_pred,
            title=f"FN: True={true_class_name}, Pred={pred_class_name}\n{info_str}\n(Saliency for Predicted Class)",
            save_path=os.path.join(output_dir, 'false_negatives', f'fn_{i}_3d_pred_subj{subject_id}_scen{scenario_id}_{pred_class_name}.png')
        )

        # 2D comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        if points.ndim == 3:
            pts = points.reshape(-1, 3)
            sal_t = sal_true.flatten()
            sal_p = sal_pred.flatten()
        else:
            pts = points
            sal_t = sal_true
            sal_p = sal_pred

        projections = [(0, 1, 'X', 'Y'), (0, 2, 'X', 'Z'), (1, 2, 'Y', 'Z')]

        for j, (xi, yi, xlabel, ylabel) in enumerate(projections):
            # True class saliency
            sc1 = axes[0, j].scatter(pts[:, xi], pts[:, yi], c=sal_t, cmap='hot', s=20, alpha=0.8)
            axes[0, j].set_xlabel(xlabel)
            axes[0, j].set_ylabel(ylabel)
            axes[0, j].set_title(f'True: {true_class_name}')
            plt.colorbar(sc1, ax=axes[0, j])

            # Predicted class saliency
            sc2 = axes[1, j].scatter(pts[:, xi], pts[:, yi], c=sal_p, cmap='hot', s=20, alpha=0.8)
            axes[1, j].set_xlabel(xlabel)
            axes[1, j].set_ylabel(ylabel)
            axes[1, j].set_title(f'Pred: {pred_class_name}')
            plt.colorbar(sc2, ax=axes[1, j])

        fig.suptitle(f'False Negative: True={true_class_name}, Predicted={pred_class_name}\n{info_str}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'false_negatives', f'fn_{i}_comparison_subj{subject_id}_scen{scenario_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Temporal importance
        if x.dim() == 4:
            temp_imp = explainer.extract_temporal_attention(x)
            if temp_imp is not None:
                visualize_temporal_importance(
                    temp_imp[0],
                    title=f"FN: True={true_class_name}, Pred={pred_class_name} - Temporal\n{info_str}",
                    save_path=os.path.join(output_dir, 'false_negatives', f'fn_{i}_temporal_subj{subject_id}_scen{scenario_id}.png')
                )

        # ========== NEW: Doppler-Saliency Correlation for False Negatives ==========
        # Extract auxiliary features and create comparison visualizations for both true and pred saliency
        if x.dim() == 4:  # Temporal: (1, T, N, C)
            if x.shape[-1] >= 7:  # Has [x, y, z, zone, doppler, snr, density]
                aux_features = x[0, :, :, 4:7].detach().cpu().numpy()  # (T, N, 3)

                # Doppler correlation with TRUE class saliency
                visualize_doppler_saliency_correlation(
                    points, sal_true, aux_features,
                    title=f"FN: True={true_class_name} - Doppler-Saliency Correlation (True Class)\n{info_str}",
                    save_path=os.path.join(output_dir, 'false_negatives', f'fn_{i}_doppler_corr_true_subj{subject_id}_scen{scenario_id}_{true_class_name}.png')
                )

                # Doppler correlation with PREDICTED class saliency
                visualize_doppler_saliency_correlation(
                    points, sal_pred, aux_features,
                    title=f"FN: Pred={pred_class_name} - Doppler-Saliency Correlation (Pred Class)\n{info_str}",
                    save_path=os.path.join(output_dir, 'false_negatives', f'fn_{i}_doppler_corr_pred_subj{subject_id}_scen{scenario_id}_{pred_class_name}.png')
                )
            elif x.shape[-1] >= 5:  # Has [x, y, z, zone, doppler]
                aux_features = x[0, :, :, 4:5].detach().cpu().numpy()  # (T, N, 1)

                # Doppler correlation with TRUE class saliency
                visualize_doppler_saliency_correlation(
                    points, sal_true, aux_features,
                    title=f"FN: True={true_class_name} - Doppler-Saliency Correlation (True Class)\n{info_str}",
                    save_path=os.path.join(output_dir, 'false_negatives', f'fn_{i}_doppler_corr_true_subj{subject_id}_scen{scenario_id}_{true_class_name}.png')
                )

                # Doppler correlation with PREDICTED class saliency
                visualize_doppler_saliency_correlation(
                    points, sal_pred, aux_features,
                    title=f"FN: Pred={pred_class_name} - Doppler-Saliency Correlation (Pred Class)\n{info_str}",
                    save_path=os.path.join(output_dir, 'false_negatives', f'fn_{i}_doppler_corr_pred_subj{subject_id}_scen{scenario_id}_{pred_class_name}.png')
                )
        elif x.dim() == 3:  # Non-temporal: (1, N, C)
            if x.shape[-1] >= 7:  # Has [x, y, z, zone, doppler, snr, density]
                aux_features = x[0, :, 4:7].detach().cpu().numpy()  # (N, 3)

                # Doppler correlation with TRUE class saliency
                visualize_doppler_saliency_correlation(
                    points, sal_true, aux_features,
                    title=f"FN: True={true_class_name} - Doppler-Saliency Correlation (True Class)\n{info_str}",
                    save_path=os.path.join(output_dir, 'false_negatives', f'fn_{i}_doppler_corr_true_subj{subject_id}_scen{scenario_id}_{true_class_name}.png')
                )

                # Doppler correlation with PREDICTED class saliency
                visualize_doppler_saliency_correlation(
                    points, sal_pred, aux_features,
                    title=f"FN: Pred={pred_class_name} - Doppler-Saliency Correlation (Pred Class)\n{info_str}",
                    save_path=os.path.join(output_dir, 'false_negatives', f'fn_{i}_doppler_corr_pred_subj{subject_id}_scen{scenario_id}_{pred_class_name}.png')
                )
            elif x.shape[-1] >= 5:  # Has [x, y, z, zone, doppler]
                aux_features = x[0, :, 4:5].detach().cpu().numpy()  # (N, 1)

                # Doppler correlation with TRUE class saliency
                visualize_doppler_saliency_correlation(
                    points, sal_true, aux_features,
                    title=f"FN: True={true_class_name} - Doppler-Saliency Correlation (True Class)\n{info_str}",
                    save_path=os.path.join(output_dir, 'false_negatives', f'fn_{i}_doppler_corr_true_subj{subject_id}_scen{scenario_id}_{true_class_name}.png')
                )

                # Doppler correlation with PREDICTED class saliency
                visualize_doppler_saliency_correlation(
                    points, sal_pred, aux_features,
                    title=f"FN: Pred={pred_class_name} - Doppler-Saliency Correlation (Pred Class)\n{info_str}",
                    save_path=os.path.join(output_dir, 'false_negatives', f'fn_{i}_doppler_corr_pred_subj{subject_id}_scen{scenario_id}_{pred_class_name}.png')
                )
        # ============================================================================

    # Generate summary statistics
    logging.info("Generating summary statistics...")

    # FiLM weights analysis (sample a few batches)
    if hasattr(model, 'edge_layers'):
        sample_data = next(iter(test_loader))[0][:4].to(device)
        film_weights = explainer.extract_film_weights(sample_data)

        for layer_name, weights in film_weights.items():
            gamma = weights['gamma'].mean(axis=(0, 1)) if weights['gamma'].ndim > 2 else weights['gamma'].mean(axis=0)
            beta = weights['beta'].mean(axis=(0, 1)) if weights['beta'].ndim > 2 else weights['beta'].mean(axis=0)

            visualize_film_modulation(
                gamma, beta,
                title=f"FiLM Modulation - {layer_name}",
                save_path=os.path.join(output_dir, f'film_{layer_name}.png')
            )

    logging.info("\n" + "="*70)
    logging.info("EXPLAINABILITY ANALYSIS COMPLETE!")
    logging.info("="*70)
    logging.info(f"Total visualizations generated:")
    logging.info(f"  - True Positives: {len(tp_sample_idx)} samples")
    logging.info(f"  - False Negatives: {len(fn_sample_idx)} samples")
    logging.info(f"Output saved to: {output_dir}")
    logging.info("="*70)

    return {
        'num_true_positives': len(tp_indices),
        'num_false_negatives': len(fn_indices),
        'accuracy': len(tp_indices) / len(all_labels)
    }
