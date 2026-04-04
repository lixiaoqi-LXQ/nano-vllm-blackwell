"""Model weight loader supporting BF16 and FP8 weights."""

import os
from glob import glob
from typing import Optional
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    loaded_scale: Optional[torch.Tensor] = None,
):
    """Default weight loader that copies weight to parameter.

    Args:
        param: Parameter to load weight into
        loaded_weight: Weight tensor to load
        loaded_scale: Optional scale factor for FP8 weights
    """
    if param.data.dtype != loaded_weight.dtype:
        param.data = param.data.to(dtype=loaded_weight.dtype)
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """Load model weights from safetensors files.

    Supports both regular BF16 weights and FP8 weights with scale factors.
    For FP8 models, scale factors are stored as 'weight_name_scale_inv'.

    Args:
        model: Model to load weights into
        path: Path to directory containing safetensors files
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # First pass: detect FP8 and preload all scale factors
    has_fp8 = False
    scale_factors = {}

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            keys = f.keys()
            # Check for FP8 scale factors (suffix '_scale_inv')
            if any(k.endswith('_scale_inv') for k in keys):
                has_fp8 = True
                # Preload all scale factors
                for key in keys:
                    if key.endswith('_scale_inv'):
                        scale_factors[key] = f.get_tensor(key)

    # Set FP8 flag on model
    setattr(model, "has_fp8_weights", has_fp8)

    # Print backend info if FP8
    if has_fp8:
        from nanovllm.utils.fp8_utils import get_fp8_info
        print(get_fp8_info())

    # Second pass: load weights
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Skip scale factors - they were already loaded
                if weight_name.endswith('_scale_inv'):
                    continue

                # Check if this is a packed module (qkv_proj, gate_up_proj)
                is_packed = False
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")

                        loaded_weight = f.get_tensor(weight_name)
                        scale_key = weight_name + '_scale_inv'
                        loaded_scale = scale_factors.get(scale_key)

                        weight_loader(param, loaded_weight, shard_id, loaded_scale)
                        is_packed = True
                        break

                if is_packed:
                    continue

                # Regular weight (not packed)
                param = model.get_parameter(weight_name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                loaded_weight = f.get_tensor(weight_name)
                scale_key = weight_name + '_scale_inv'
                loaded_scale = scale_factors.get(scale_key)

                if loaded_scale is not None:
                    weight_loader(param, loaded_weight, loaded_scale)
                else:
                    weight_loader(param, loaded_weight)
