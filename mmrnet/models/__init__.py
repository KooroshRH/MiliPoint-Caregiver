
from .dgcnn import DGCNN
from .mlp import MLP
from .pointnet import PointNet
from .point_transformer import PointTransformer
from .pointmlp import PointMLP
from .attdgcnn import AttDGCNN
from .hybrid_pointnet import HybridPointNet
from .ensemble_model import EnsembleModel, load_ensemble_from_checkpoint

# Import the new PointNet-FiLM model
try:
    from .pointnet_film import PointNetWithFiLM
    POINTNET_FILM_AVAILABLE = True
except ImportError:
    POINTNET_FILM_AVAILABLE = False

model_map = {
    'dgcnn': DGCNN,
    'mlp': MLP,
    'pointnet': PointNet,
    'pointtransformer': PointTransformer,
    'pointmlp': PointMLP,
    'attdgcnn': AttDGCNN,
    'hybrid': HybridPointNet,
    'ensemble': EnsembleModel,
}

# Add PointNet-FiLM to model_map if available
if POINTNET_FILM_AVAILABLE:
    model_map['pointnet-film'] = PointNetWithFiLM
