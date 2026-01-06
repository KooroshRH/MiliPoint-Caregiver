
from .dgcnn import DGCNN
from .mlp import MLP
from .pointnet import PointNet
from .point_transformer import PointTransformer
from .pointmlp import PointMLP
from .pointmlp_aux import PointMLP_Aux
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
from .dgcnn_aux_fusion_stattn import DGCNNAuxFusion_STAttn
from .pointnext import PointNext
from .pointnext_aux import PointNext_Aux
from .deepgcn import DeepGCN
from .deepgcn_aux import DeepGCN_Aux
from .point_transformer_v3 import PointTransformerV3
from .point_transformer_v3_aux import PointTransformerV3_Aux
from .pointmamba import PointMamba
from .pointmamba_aux import PointMamba_Aux
from .mamba4d import Mamba4D
from .mamba4d_aux import Mamba4D_Aux
from .mamba4d_aux_film import Mamba4D_Aux_FiLM

model_map = {
    'dgcnn': DGCNN,
    'mlp': MLP,
    'pointnet': PointNet,
    'pointtransformer': PointTransformer,
    'pointmlp': PointMLP,
    'pointmlp_aux': PointMLP_Aux,
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
    'dgcnn_aux_fusion_stattn': DGCNNAuxFusion_STAttn,
    'pointnext': PointNext,
    'pointnext_aux': PointNext_Aux,
    'deepgcn': DeepGCN,
    'deepgcn_aux': DeepGCN_Aux,
    'pointtransformerv3': PointTransformerV3,
    'pointtransformerv3_aux': PointTransformerV3_Aux,
    'pointmamba': PointMamba,
    'pointmamba_aux': PointMamba_Aux,
    'mamba4d': Mamba4D,
    'mamba4d_aux': Mamba4D_Aux,
    'mamba4d_aux_film': Mamba4D_Aux_FiLM,
}
