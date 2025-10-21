
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
from .dgcnn_aux import DGCNN_Aux
from .point_transformer_aux import PointTransformer_Aux
from .pointnet_aux import PointNet_Aux
from .dgcnn_aux_fusion_t import DGCNNAuxFusionT

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
    'aux_former': AuxFormer,
    'dgcnn_aux': DGCNN_Aux,
    'pointtransformer_aux': PointTransformer_Aux,
    'pointnet_aux': PointNet_Aux,
    'dgcnn_aux_fusion_t': DGCNNAuxFusionT,
}
