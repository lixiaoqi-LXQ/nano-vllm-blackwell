"""FP8 utilities for FP8 GEMM operations.

This module provides utilities for FP8 quantization and matrix multiplication
with multi-backend support: FlashInfer (SM90) and Fallback (dequantization).

Backend selection:
- FlashInfer: SM90+ (Hopper), native block-scaled FP8 GEMM
- Fallback: All GPUs, dequantize to BF16 then standard GEMM
"""

from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from flashinfer.gemm import gemm_fp8_nt_groupwise


class FP8Backend(Enum):
    """Available FP8 GEMM backends."""
    FALLBACK = "fallback"           # Dequantize to BF16, standard GEMM
    FLASHINFER = "flashinfer"       # FlashInfer block-scaled GEMM (SM90)


# FP8 E4M3 maximum representable value
FP8_E4M3_MAX = 448.0

# Default block size for block-wise quantization
DEFAULT_BLOCK_SIZE = (128, 128)

# Activation quantization block size (per-token-block)
ACT_BLOCK_SIZE = 128

# Supported runtime dtypes
DEFAULT_FP8_DTYPE = torch.float8_e4m3fn
DEFAULT_SCALE_DTYPE = torch.float32

# Global flag to track current backend
_current_backend: Optional[FP8Backend] = None


@triton.jit
def _quantize_activation_fp8_kernel(
    x_ptr,
    x_stride_m,
    x_stride_k,
    out_ptr,
    out_stride_m,
    out_stride_k,
    scale_inv_ptr,
    scale_inv_stride_m,
    scale_inv_stride_k,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """Triton kernel for FP8 per-token-block quantization.

    Each program instance handles one (token, block) pair.

    Quantization formula:
        scale_inv = amax / FP8_MAX
        x_fp8 = x / scale_inv
        x = x_fp8 * scale_inv  (dequantization)
    """
    # Grid: (num_tokens, num_blocks)
    pid_m = tl.program_id(0)  # token index
    pid_k = tl.program_id(1)  # block index

    # Compute block range
    k_start = pid_k * BLOCK_SIZE
    k_offsets = k_start + tl.arange(0, BLOCK_SIZE)

    # Load input block [BLOCK_SIZE]
    x_ptrs = x_ptr + pid_m * x_stride_m + k_offsets * x_stride_k
    mask = k_offsets < K
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute amax and scale_inv
    x_abs = tl.abs(x)
    amax = tl.max(x_abs)
    amax = tl.maximum(amax, 1e-4)
    scale_inv = amax / FP8_MAX
    log2_scale = tl.log2(scale_inv)
    log2_scale = tl.math.ceil(log2_scale)
    scale_inv = tl.exp2(log2_scale)

    # Quantize: x / scale_inv -> FP8
    x_scaled = x / scale_inv
    out = x_scaled.to(tl.float8e4nv)

    # Store output
    out_ptrs = out_ptr + pid_m * out_stride_m + k_offsets * out_stride_k
    tl.store(out_ptrs, out, mask=mask)

    # Store scale_inv
    scale_inv = scale_inv.to(tl.bfloat16)
    scale_inv_ptrs = scale_inv_ptr + pid_m * \
        scale_inv_stride_m + pid_k * scale_inv_stride_k
    tl.store(scale_inv_ptrs, scale_inv)


def quantize_activation_fp8(
    activation: torch.Tensor,
    block_size: int = ACT_BLOCK_SIZE,
    fp8_dtype: torch.dtype = DEFAULT_FP8_DTYPE,
    scale_dtype: torch.dtype = DEFAULT_SCALE_DTYPE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 activation to FP8 with per-token-block scaling.

    Args:
        activation: Input tensor [M, K] in BF16
        block_size: Block size for per-token quantization (default: 128)

    Returns:
        act_fp8: [M, K] float8_e4m3fn
        act_scale_inv: [M, K//128] BF16 (dequant scale factor, multiply to dequantize)
    """
    M, K = activation.shape
    num_blocks = (K + block_size - 1) // block_size
    activation = activation.contiguous()

    act_fp8 = torch.empty(M, K, dtype=fp8_dtype, device=activation.device)
    act_scale_inv = torch.zeros(
        M, num_blocks, dtype=scale_dtype, device=activation.device)

    grid = (M, num_blocks)
    _quantize_activation_fp8_kernel[grid](
        activation,
        activation.stride(0), activation.stride(1),
        act_fp8,
        act_fp8.stride(0), act_fp8.stride(1),
        act_scale_inv,
        act_scale_inv.stride(0), act_scale_inv.stride(1),
        K=K,
        BLOCK_SIZE=block_size,
        FP8_MAX=FP8_E4M3_MAX,
    )
    return act_fp8, act_scale_inv


def _fp8_linear_flashinfer(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    block_size: Tuple[int, int] = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """FlashInfer native FP8 GEMM using gemm_fp8_nt_groupwise."""
    input = input.contiguous()
    M, K = input.shape

    # Quantize activation to FP8
    act_fp8, act_scale_inv = quantize_activation_fp8(input)

    # Convert scales to float32 for gemm_fp8_nt_groupwise
    if act_scale_inv.dtype != torch.float32:
        act_scale_inv = act_scale_inv.float()
    if weight_scale_inv.dtype != torch.float32:
        weight_scale_inv = weight_scale_inv.float()

    output = gemm_fp8_nt_groupwise(
        a=act_fp8,
        b=weight_fp8,
        a_scale=act_scale_inv,
        b_scale=weight_scale_inv,
        scale_granularity_mnk=(1, block_size[0], block_size[1]),
        scale_major_mode="K",
        out_dtype=torch.bfloat16,
    )

    if bias is not None:
        output = output + bias
    return output


def get_fp8_backend() -> FP8Backend:
    """Auto-detect best available FP8 backend.

    Backend selection:
    - FlashInfer: SM90+ (Hopper) and SM120+ (Blackwell), native block-scaled FP8 GEMM
    - Fallback: All other GPUs, dequantize to BF16 then standard GEMM

    Returns:
        Best available FP8Backend
    """
    global _current_backend

    if _current_backend is not None:
        return _current_backend

    if not torch.cuda.is_available():
        _current_backend = FP8Backend.FALLBACK
        return _current_backend

    capability = torch.cuda.get_device_capability()

    # FlashInfer supports SM90+ (Hopper) and SM120+ (Blackwell)
    if capability >= (9, 0):
        _current_backend = FP8Backend.FLASHINFER
        return _current_backend

    _current_backend = FP8Backend.FALLBACK
    return _current_backend


def _fp8_linear_fallback(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    block_size: Tuple[int, int] = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Fallback FP8 GEMM by dequantizing to BF16."""
    block_h, block_w = block_size
    # Expand block-wise scale to full weight shape
    scale_expanded = weight_scale_inv.repeat_interleave(
        block_h, dim=0).repeat_interleave(block_w, dim=1)
    n, k = weight_fp8.shape
    if scale_expanded.shape[0] > n or scale_expanded.shape[1] > k:
        scale_expanded = scale_expanded[:n, :k]
    # Dequantize to BF16 and use standard GEMM
    weight_bf16 = weight_fp8.to(input.dtype) * scale_expanded
    return F.linear(input, weight_bf16, bias)


def fp8_linear(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    block_size: Tuple[int, int] = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Unified FP8 linear layer entry point.

    Automatically selects the best available backend.
    """
    backend = get_fp8_backend()

    if backend == FP8Backend.FLASHINFER:
        return _fp8_linear_flashinfer(
            input, weight_fp8, weight_scale_inv, bias, block_size
        )

    return _fp8_linear_fallback(input, weight_fp8, weight_scale_inv, bias, block_size)


def fp8_linear_chunked(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    block_size: Tuple[int, int] = DEFAULT_BLOCK_SIZE,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """FP8 linear with M-dimension chunking for large batches."""
    M = input.size(0)
    if M <= chunk_size:
        return fp8_linear(input, weight_fp8, weight_scale_inv, bias, block_size)
    chunks = []
    for i in range(0, M, chunk_size):
        out_i = fp8_linear(
            input[i:i + chunk_size], weight_fp8, weight_scale_inv, None, block_size)
        chunks.append(out_i)
    result = torch.cat(chunks, dim=0)
    if bias is not None:
        result = result + bias
    return result


def get_fp8_info() -> str:
    """Get FP8 support information for logging.

    Returns:
        String describing FP8 support status
    """
    if not torch.cuda.is_available():
        return "FP8: CUDA not available"

    device_name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    backend = get_fp8_backend()

    if backend == FP8Backend.FLASHINFER:
        return f"FP8: Native FlashInfer GEMM on {device_name} (SM {capability[0]}.{capability[1]})"

    return f"FP8: Fallback (dequantize) on {device_name} (SM {capability[0]}.{capability[1]})"
