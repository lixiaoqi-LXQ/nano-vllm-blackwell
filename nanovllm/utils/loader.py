"""Model weight loader supporting BF16, FP8, and FP4 weights."""

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
        loaded_scale: Optional scale factor for FP8/FP4 weights
    """
    if param.data.dtype != loaded_weight.dtype:
        param.data = param.data.to(dtype=loaded_weight.dtype)
    param.data.copy_(loaded_weight)


def _detect_quantization(keys):
    """Detect quantization type from checkpoint key names.

    Returns:
        "fp8" if FP8 scale factors found
        "fp4" if FP4 scale factors found
        None for unquantized BF16
    """
    has_fp4 = False
    has_fp8 = False
    for key in keys:
        if key.endswith("_scale_inv"):
            has_fp8 = True
        if key.endswith("_scale_2") or key.endswith(".input_scale"):
            # FP4 models have weight_scale + weight_scale_2 + input_scale
            # But be careful: k_scale/v_scale also end without _scale_inv
            pass
        if key.endswith("_scale") and "weight_scale" in key:
            # weight_scale (block scale) is present in both FP8 and FP4
            # FP4 also has weight_scale_2
            pass

    # Check for FP4 indicators: weight_scale_2 is unique to FP4
    for key in keys:
        if key.endswith("_scale_2"):
            return "fp4"

    if has_fp8:
        return "fp8"

    return None


def load_model(model: nn.Module, path: str):
    """Load model weights from safetensors files.

    Supports regular BF16, FP8, and FP4 weights.
    - FP8: scale factors stored as 'weight_name_scale_inv'
    - FP4: scale factors stored as 'weight_name_weight_scale', 'weight_name_weight_scale_2',
           and 'weight_name_input_scale'

    Args:
        model: Model to load weights into
        path: Path to directory containing safetensors files
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # First pass: detect quantization and preload all scale factors
    quant_type = None
    fp8_scale_factors = {}
    fp4_scale_factors = {}

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            keys = f.keys()

            # Detect quantization type from this file
            file_quant = _detect_quantization(keys)
            if file_quant == "fp4":
                quant_type = "fp4"
            elif file_quant == "fp8" and quant_type is None:
                quant_type = "fp8"

            if quant_type == "fp4" or file_quant == "fp4":
                quant_type = "fp4"
                # Preload FP4 scale factors
                for key in keys:
                    if key.endswith("_scale") and "weight_scale" in key:
                        fp4_scale_factors[key] = f.get_tensor(key)
                    elif key.endswith("_scale_2"):
                        fp4_scale_factors[key] = f.get_tensor(key)
                    elif key.endswith("input_scale"):
                        fp4_scale_factors[key] = f.get_tensor(key)
                    elif key.endswith("k_scale") or key.endswith("v_scale"):
                        fp4_scale_factors[key] = f.get_tensor(key)
            elif quant_type == "fp8" or file_quant == "fp8":
                # Preload FP8 scale factors
                for key in keys:
                    if key.endswith("_scale_inv"):
                        fp8_scale_factors[key] = f.get_tensor(key)

    # Set quantization flags on model
    model.has_fp8_weights = quant_type == "fp8"
    model.has_fp4_weights = quant_type == "fp4"

    # Print backend info
    if quant_type == "fp4":
        from nanovllm.utils.fp4_utils import get_fp4_info
        print(get_fp4_info())
    elif quant_type == "fp8":
        from nanovllm.utils.fp8_utils import get_fp8_info
        print(get_fp8_info())

    # Set k_scale/v_scale on attention layers for FP4 models
    if quant_type == "fp4":
        for key, value in fp4_scale_factors.items():
            if key.endswith(".k_scale") or key.endswith(".v_scale"):
                # e.g. model.layers.N.self_attn.k_proj.k_scale
                parts = key.split('.')
                layer_idx = int(parts[2])
                if "k_proj" in key:
                    attn = model.model.layers[layer_idx].self_attn.attn
                    attn.k_scale = value.cuda().float()
                elif "v_proj" in key:
                    attn = model.model.layers[layer_idx].self_attn.attn
                    attn.v_scale = value.cuda().float()

    # Second pass: load weights
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # Skip scale factors - already loaded
                if weight_name.endswith("_scale_inv"):
                    continue
                if weight_name in fp4_scale_factors:
                    continue
                # Also skip FP4 scale factors
                if quant_type == "fp4" and (
                    weight_name.endswith("_scale")
                    or weight_name.endswith("_scale_2")
                    or weight_name.endswith("input_scale")
                    or weight_name.endswith("k_scale")
                    or weight_name.endswith("v_scale")
                ):
                    continue

                loaded_weight = f.get_tensor(weight_name)

                # Check if this is a packed module (qkv_proj, gate_up_proj)
                is_packed = False
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)

                        # Check if this weight exists as a parameter
                        try:
                            param = model.get_parameter(param_name)
                        except AttributeError:
                            continue

                        weight_loader = getattr(param, "weight_loader", None)
                        if weight_loader is None:
                            continue

                        if quant_type == "fp4":
                            # FP4: get associated scales
                            scale_key = weight_name + "_scale"
                            scale_2_key = weight_name + "_scale_2"
                            input_scale_key = weight_name.replace(
                                ".weight", ".input_scale"
                            )
                            loaded_scale = fp4_scale_factors.get(scale_key)
                            loaded_scale_2 = fp4_scale_factors.get(scale_2_key)
                            loaded_input_scale = fp4_scale_factors.get(input_scale_key)

                            # FP4 packed modules need scale_2 and input_scale
                            weight_loader(
                                param, loaded_weight, shard_id,
                                loaded_scale, loaded_scale_2, loaded_input_scale
                            )
                        elif quant_type == "fp8":
                            scale_key = weight_name + "_scale_inv"
                            loaded_scale = fp8_scale_factors.get(scale_key)
                            weight_loader(param, loaded_weight, shard_id, loaded_scale)
                        else:
                            weight_loader(param, loaded_weight, shard_id)

                        is_packed = True
                        break

                if is_packed:
                    continue

                # Regular weight (not packed)
                try:
                    param = model.get_parameter(weight_name)
                except AttributeError:
                    continue

                weight_loader = getattr(param, "weight_loader", default_weight_loader)

                if quant_type == "fp4":
                    # FP4: get associated scales
                    scale_key = weight_name + "_scale"
                    scale_2_key = weight_name + "_scale_2"
                    input_scale_key = weight_name.replace(".weight", ".input_scale")
                    loaded_scale = fp4_scale_factors.get(scale_key)
                    loaded_scale_2 = fp4_scale_factors.get(scale_2_key)
                    loaded_input_scale = fp4_scale_factors.get(input_scale_key)

                    if loaded_scale is not None and loaded_scale_2 is not None:
                        weight_loader(
                            param, loaded_weight,
                            loaded_scale, loaded_scale_2, loaded_input_scale
                        )
                    else:
                        weight_loader(param, loaded_weight)
                elif quant_type == "fp8":
                    scale_key = weight_name + '_scale_inv'
                    loaded_scale = fp8_scale_factors.get(scale_key)
                    if loaded_scale is not None:
                        weight_loader(param, loaded_weight, loaded_scale)
                    else:
                        weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight)
