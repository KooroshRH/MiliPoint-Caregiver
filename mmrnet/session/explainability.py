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


def visualize_point_saliency_3d(points, saliency, title="Point Saliency", save_path=None,
                                  frame_idx=None, view_angles=(30, 45)):
    """
    Visualize point cloud with saliency coloring in 3D.

    Args:
        points: (N, 3) or (T, N, 3) point coordinates
        saliency: (N,) or (T, N) saliency scores
        title: Plot title
        save_path: Path to save figure
        frame_idx: If temporal data, which frame to visualize (None = average)
        view_angles: (elevation, azimuth) for 3D view
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    if points.ndim == 3:  # Temporal data (T, N, 3)
        if frame_idx is not None:
            pts = points[frame_idx]
            sal = saliency[frame_idx]
            title += f" (Frame {frame_idx})"
        else:
            # Average across frames
            pts = points.reshape(-1, 3)
            sal = saliency.flatten()
            title += " (All Frames)"
    else:
        pts = points
        sal = saliency

    # Create colormap
    norm = Normalize(vmin=sal.min(), vmax=sal.max())
    colors = cm.hot(norm(sal))

    scatter = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=sal, cmap='hot', s=50, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(elev=view_angles[0], azim=view_angles[1])

    plt.colorbar(scatter, ax=ax, label='Saliency Score')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_point_saliency_2d(points, saliency, title="Point Saliency", save_path=None,
                                  frame_idx=None):
    """
    Visualize point cloud saliency in 2D projections (XY, XZ, YZ).

    Args:
        points: (N, 3) or (T, N, 3) point coordinates
        saliency: (N,) or (T, N) saliency scores
        title: Plot title
        save_path: Path to save figure
        frame_idx: If temporal data, which frame to visualize
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if points.ndim == 3:
        if frame_idx is not None:
            pts = points[frame_idx]
            sal = saliency[frame_idx]
            title += f" (Frame {frame_idx})"
        else:
            pts = points.reshape(-1, 3)
            sal = saliency.flatten()
            title += " (All Frames)"
    else:
        pts = points
        sal = saliency

    projections = [
        (pts[:, 0], pts[:, 1], 'X', 'Y', 'XY Projection'),
        (pts[:, 0], pts[:, 2], 'X', 'Z', 'XZ Projection'),
        (pts[:, 1], pts[:, 2], 'Y', 'Z', 'YZ Projection')
    ]

    for ax, (x, y, xlabel, ylabel, proj_title) in zip(axes, projections):
        scatter = ax.scatter(x, y, c=sal, cmap='hot', s=30, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(proj_title)
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Saliency')

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

    # Sample indices
    logging.info(f"\nSampling {num_samples} True Positives and {num_samples} False Negatives...")
    tp_sample_idx = tp_indices[torch.randperm(len(tp_indices))[:num_samples]].tolist()
    fn_sample_idx = fn_indices[torch.randperm(len(fn_indices))[:num_samples]].tolist()
    logging.info(f"✓ Sampled {len(tp_sample_idx)} TP and {len(fn_sample_idx)} FN")

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
