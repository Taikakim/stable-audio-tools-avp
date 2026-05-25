from . import rocm_env as _rocm_env  # sets HIP/MIOpen/TunableOp env before torch
_rocm_env.apply_profile("inference")
from .models.factory import create_model_from_config, create_model_from_config_path
from .models.pretrained import get_pretrained_model