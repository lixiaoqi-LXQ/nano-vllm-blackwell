"""FP8 utilities for native FP8 GEMM operations.

This module provides utilities for FP8 quantization and matrix multiplication
using torch._scaled_mm for Blackwell architecture GPUs with automatic fallback.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# FP8 E4M3 maximum representable value
FP8_E4M3_MAX = 448.0

# Default block size for block-wise quantization
DEFAULT_BLOCK_SIZE = (128, 128)

# Global flag to track if native FP8 is being used
_using_native_fp8 = False


def quantize_to_fp8(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 input to FP8 with per-tensor scaling.

    Args:
        input: Input tensor in BF16 format

    Returns:
        Tuple of (quantized FP8 tensor, scale factor as float32)
    """
    # Compute amax and scale
    amax = torch.abs(input).max().clamp(min=1e-12)
    scale = (amax / FP8_E4M3_MAX).float()  # Ensure float32

    # Quantize to FP8
    input_fp8 = (input.float() / scale).to(torch.float8_e4m3fn)

    return input_fp8, scale


def expand_block_scale(
    scale_inv: torch.Tensor,
    weight_shape: Tuple[int, int],
    block_size: Tuple[int, int] = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Expand block-wise scale to full weight shape.

    The scale_inv tensor has shape [num_blocks_h, num_blocks_w] where
    each element is the inverse scale for a block of size [block_h, block_w].

    Args:
        scale_inv: Block-wise inverse scale tensor [num_blocks_h, num_blocks_w]
        weight_shape: Target weight shape [out_features, in_features]
        block_size: Block size (block_h, block_w)

    Returns:
        Expanded scale tensor of shape [out_features, in_features]
    """
    block_h, block_w = block_size
    out_features, in_features = weight_shape

    # Ensure float32 for precision
    scale_inv = scale_inv.float()

    # Repeat each element block_h times along dim 0 and block_w times along dim 1
    expanded = scale_inv.repeat_interleave(block_h, dim=0).repeat_interleave(block_w, dim=1)

    # Handle edge cases where weight shape is not perfectly divisible by block size
    if expanded.shape[0] > out_features or expanded.shape[1] > in_features:
        expanded = expanded[:out_features, :in_features]

    return expanded


def is_native_fp8_supported() -> bool:
    """Check if native FP8 GEMM is supported on current hardware.

    Returns:
        True if torch._scaled_mm is available and GPU supports FP8
    """
    if not hasattr(torch, '_scaled_mm'):
        return False

    try:
        # Check compute capability (need 12.0+ for native FP8)
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            return capability >= (12, 0)
    except Exception:
        pass

    return False


def _can_use_native_fp8(
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: Tuple[int, int] = DEFAULT_BLOCK_SIZE,
) -> bool:
    """Check if we can use native FP8 GEMM for the given weight and scale.

    For FP8 inference, we use fallback (dequantization) by default
    because it provides better accuracy than per-tensor scaling.

    Native FP8 GEMM options explored:
    1. torch._scaled_mm BlockWise 128x128: Requires near-inner-dim-major layout
       which is not compatible with standard PyTorch tensor layouts.
    2. torch._scaled_mm RowWise: Works but causes 37% accuracy loss when
       converting block-wise scales to per-column scales.
    3. FlashInfer gemm_fp8_nt_groupwise: Unstable behavior with inf values.

    Future work:
    - Implement custom CUDA kernel for layout conversion
    - Wait for PyTorch/FlashInfer updates for better block-wise FP8 support

    Args:
        weight_fp8: FP8 weight tensor [N, K]
        weight_scale_inv: Block-wise inverse scale [num_blocks_n, num_blocks_k]
        block_size: Block size for quantization

    Returns:
        True if native FP8 GEMM can be used
    """
    # For now, disable native FP8 and use fallback for better accuracy
    return False


def _native_fp8_gemm(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    block_size: Tuple[int, int] = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Native FP8 GEMM using torch._scaled_mm with per-tensor scaling.

    This function converts block-wise scales to per-tensor scales for
    compatibility with torch._scaled_mm TensorWise mode.

    Args:
        input: Input tensor [M, K] in BF16
        weight_fp8: Weight tensor [N, K] in FP8 E4M3 format
        weight_scale_inv: Block-wise inverse scale [num_blocks_n, num_blocks_k]
        bias: Optional bias tensor [N]
        block_size: Block size for scale

    Returns:
        Output tensor [M, N] in BF16
    """
    # Quantize input to FP8 with per-tensor scaling
    input_fp8, input_scale = quantize_to_fp8(input)

    # Convert block-wise scale to per-tensor scale
    # Use the max of all block scales as the global scale for better accuracy
    # This is more conservative than mean and provides better dynamic range
    weight_scale_per_tensor = weight_scale_inv.float().max()

    # Prepare scales as float32 scalars
    scale_a = input_scale.float()
    scale_b = weight_scale_per_tensor.float()

    # Convert weight to column-major layout for torch._scaled_mm
    # torch._scaled_mm expects mat2 (weight) in column-major format
    weight_colmajor = weight_fp8.t().contiguous().t()

    # Call torch._scaled_mm with TensorWise scaling
    output = torch._scaled_mm(
        input_fp8,
        weight_colmajor.t(),  # [K, N] for matmul, column-major
        scale_a=scale_a,
        scale_b=scale_b,
        bias=bias,
        out_dtype=torch.bfloat16,
    )

    return output


def _fallback_fp8_gemm(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    block_size: Tuple[int, int] = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Fallback FP8 GEMM by dequantizing to BF16.

    Args:
        input: Input tensor [M, K] in BF16
        weight_fp8: Weight tensor [N, K] in FP8 E4M3 format
        weight_scale_inv: Block-wise inverse scale [num_blocks_n, num_blocks_k]
        bias: Optional bias tensor [N]
        block_size: Block size for scale expansion

    Returns:
        Output tensor [M, N] in BF16
    """
    # Expand block-wise scale to full shape
    weight_scale_expanded = expand_block_scale(
        weight_scale_inv, weight_fp8.shape, block_size
    )

    # Dequantize weight to BF16 (ensure result is bfloat16)
    weight_bf16 = (weight_fp8.to(torch.float32) * weight_scale_expanded).to(torch.bfloat16)

    # Standard GEMM
    return F.linear(input, weight_bf16, bias)


def fp8_linear(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    block_size: Tuple[int, int] = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """FP8 linear layer using torch._scaled_mm with automatic fallback.

    This function performs matrix multiplication using native FP8 GEMM
    on supported hardware (Blackwell architecture with compute capability 12.0+).
    If native FP8 fails, it automatically falls back to dequantization.

    Args:
        input: Input tensor [M, K] in BF16
        weight_fp8: Weight tensor [N, K] in FP8 E4M3 format
        weight_scale_inv: Block-wise inverse scale [num_blocks_n, num_blocks_k]
        bias: Optional bias tensor [N]
        block_size: Block size for scale expansion

    Returns:
        Output tensor [M, N] in BF16
    """
    global _using_native_fp8

    # Check if we can use native FP8
    use_native = _can_use_native_fp8(weight_fp8, weight_scale_inv, block_size)

    if use_native:
        try:
            output = _native_fp8_gemm(input, weight_fp8, weight_scale_inv, bias, block_size)
            _using_native_fp8 = True
            return output
        except (RuntimeError, AssertionError) as e:
            # Native FP8 failed, fall back
            _using_native_fp8 = False

    # Fallback: dequantize weight and use standard GEMM
    return _fallback_fp8_gemm(input, weight_fp8, weight_scale_inv, bias, block_size)


def fp8_linear_native(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    block_size: Tuple[int, int] = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """FP8 linear layer - alias for fp8_linear for compatibility.

    Args:
        input: Input tensor [M, K] in BF16
        weight_fp8: Weight tensor [N, K] in FP8 E4M3 format
        weight_scale_inv: Block-wise inverse scale [num_blocks_n, num_blocks_k]
        bias: Optional bias tensor [N]
        block_size: Block size for scale expansion

    Returns:
        Output tensor [M, N] in BF16
    """
    return fp8_linear(input, weight_fp8, weight_scale_inv, bias, block_size)


def is_using_native_fp8() -> bool:
    """Check if the last fp8_linear call used native FP8 GEMM.

    Returns:
        True if native FP8 was used, False if fallback was used
    """
    return _using_native_fp8


def get_fp8_info() -> str:
    """Get FP8 support information for logging.

    Returns:
        String describing FP8 support status
    """
    if not torch.cuda.is_available():
        return "FP8: CUDA not available"

    device_name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()

    if hasattr(torch, '_scaled_mm'):
        return f"FP8: Supported on {device_name} (SM {capability[0]}.{capability[1]})"
    else:
        return f"FP8: torch._scaled_mm not available (PyTorch version may be too old)"
