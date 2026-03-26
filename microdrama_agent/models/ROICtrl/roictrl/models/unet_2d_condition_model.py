from ..models.stable_diffusion_models.unet_2d_condition import UNet2DConditionModel
from ..utils.registry import MODEL_REGISTRY

MODEL_REGISTRY.register(UNet2DConditionModel)
