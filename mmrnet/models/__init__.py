
from .dgcnn import DGCNN
from .mlp import MLP
from .pointnet import PointNet
from .point_transformer import PointTransformer
from .pointmlp import PointMLP
from .attdgcnn import AttDGCNN
from .hybrid_pointnet import HybridPointNet
from .ensemble_model import EnsembleModel, load_ensemble_from_checkpoint
from .pointnet_film import PointNetWithFiLM
from .pointnet_film_density import PointNetWithFiLMDensity
from .aux_former import AuxFormer

model_map = {
    'dgcnn': DGCNN,
    'mlp': MLP,
    'pointnet': PointNet,
    'pointtransformer': PointTransformer,
    'pointmlp': PointMLP,
    'attdgcnn': AttDGCNN,
    'hybrid': HybridPointNet,
    'ensemble': EnsembleModel,
    'pointnet-film': PointNetWithFiLM,
    'pointnet-film-density': PointNetWithFiLMDensity,
    'aux_former': AuxFormer
}
