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
from flashinfer.testing.utils import quantize_fp8


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
DEFAULT_SCALE_DTYPE = torch.bfloat16
_SUPPORTED_SCALE_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_SUPPORTED_OUTPUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

# Global flag to track current backend
_current_backend: Optional[FP8Backend] = None


def _validate_block_size_2d(block_size: Tuple[int, int], name: str) -> Tuple[int, int]:
    if len(block_size) != 2:
        raise ValueError(f"{name} must be a 2D tuple, got {block_size}")
    block_n, block_k = int(block_size[0]), int(block_size[1])
    if block_n <= 0 or block_k <= 0:
        raise ValueError(f"{name} values must be positive, got {block_size}")
    return block_n, block_k


def _validate_scale_shape(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: Tuple[int, int],
) -> None:
    block_n, block_k = _validate_block_size_2d(block_size, "block_size")
    n, k = weight.shape
    expected = (
        (n + block_n - 1) // block_n,
        (k + block_k - 1) // block_k,
    )
    if weight_scale_inv.shape != expected:
        raise ValueError(
            "weight_scale_inv shape mismatch: "
            f"expected {expected}, got {tuple(weight_scale_inv.shape)}"
        )


def _to_runtime_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if x.dtype == dtype:
        return x
    return x.to(dtype=dtype)


@triton.jit
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :]
                    < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None]
                    < K - k * BLOCK_SIZE_K, other=0.0)

        k_start = k * BLOCK_SIZE_K
        offs_ks = k_start // group_k
        a_s = tl.load(As_ptrs + offs_ks * stride_As_k)
        b_s = tl.load(Bs_ptrs + offs_ks * stride_Bs_k)

        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    out: torch.Tensor,
    block_size: Tuple[int, int] = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    M = A.shape[0]
    N, K = B.shape
    block_n, block_k = _validate_block_size_2d(block_size, "block_size")

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 32,
        "num_warps": 4,
        "num_stages": 3,
    }

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) *
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    _w8a8_block_fp8_matmul[grid](
        A,
        B,
        out,
        As,
        Bs,
        M,
        N,
        K,
        block_n,
        block_k,
        A.stride(-2),
        A.stride(-1),
        B.stride(1),
        B.stride(0),
        out.stride(-2),
        out.stride(-1),
        As.stride(-2),
        As.stride(-1),
        Bs.stride(1),
        Bs.stride(0),
        **config,  # type: ignore[arg-type]
    )

    return out


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
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if fp8_dtype != DEFAULT_FP8_DTYPE:
        raise ValueError(
            f"Only {DEFAULT_FP8_DTYPE} is currently supported, got {fp8_dtype}"
        )
    if scale_dtype not in _SUPPORTED_SCALE_DTYPES:
        raise TypeError(
            f"Unsupported scale_dtype {scale_dtype}. Supported: {_SUPPORTED_SCALE_DTYPES}"
        )

    M, K = activation.shape
    num_blocks = (K + block_size - 1) // block_size

    # Ensure input is contiguous and in BF16
    activation = activation.contiguous()
    if activation.dtype != torch.bfloat16:
        activation = activation.to(torch.bfloat16)

    # Allocate output tensors
    act_fp8 = torch.empty(M, K, dtype=fp8_dtype, device=activation.device)
    act_scale_inv = torch.zeros(
        M, num_blocks, dtype=scale_dtype, device=activation.device)

    # Launch kernel
    grid = (M, num_blocks)
    _quantize_activation_fp8_kernel[grid](
        activation,
        activation.stride(0),
        activation.stride(1),
        act_fp8,
        act_fp8.stride(0),
        act_fp8.stride(1),
        act_scale_inv,
        act_scale_inv.stride(0),
        act_scale_inv.stride(1),
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
    """FlashInfer native FP8 GEMM using gemm_fp8_nt_groupwise.

    Requirements:
    - SM90 (Hopper) or SM120+ (Blackwell)
    - N % 64 == 0, K % 128 == 0

    Args:
        input: Input tensor [M, K] in BF16
        weight_fp8: Weight tensor [N, K] in FP8 E4M3 format
        weight_scale_inv: Block-wise inverse scale [num_blocks_n, num_blocks_k]
        bias: Optional bias tensor [N]
        block_size: Block size for scale

    Returns:
        Output tensor [M, N] in BF16
    """

    if weight_fp8.dtype != DEFAULT_FP8_DTYPE:
        raise TypeError(
            f"Expected FP8 weight dtype {DEFAULT_FP8_DTYPE}, got {weight_fp8.dtype}"
        )
    if weight_scale_inv.dtype not in _SUPPORTED_SCALE_DTYPES:
        raise TypeError(
            "Unsupported weight_scale_inv dtype "
            f"{weight_scale_inv.dtype}. Supported: {_SUPPORTED_SCALE_DTYPES}"
        )

    block_n, block_k = _validate_block_size_2d(block_size, "block_size")
    _validate_scale_shape(weight_fp8, weight_scale_inv, (block_n, block_k))

    # Keep runtime tensors colocated and contiguous for Triton kernel launch.
    input = input.contiguous()
    weight_fp8 = weight_fp8.contiguous()
    weight_scale_inv = weight_scale_inv.contiguous()

    output_dtype = torch.bfloat16

    M, K = input.shape
    N = weight_fp8.shape[0]

    # Quantize activation to FP8 with per-token-block scaling
    act_fp8, act_scale_inv = quantize_activation_fp8(
        input, block_size=ACT_BLOCK_SIZE)
    # FlashInfer Testing API
    # act_fp8, act_scale_inv = quantize_fp8(
    #     input,
    #     scale_shape=(M, K // 128),
    #     tile_shape=(1, 128),
    #     scale_major_mode="K",
    # )

    output = torch.empty(M, N, dtype=output_dtype, device=input.device)
    triton_w8a8_block_fp8_matmul(
        act_fp8, weight_fp8, act_scale_inv, weight_scale_inv, output)

    # # FlashInfer GEMM
    # output = gemm_fp8_nt_groupwise(
    #     a=act_fp8,
    #     b=weight_fp8,
    #     a_scale=act_scale_inv,
    #     b_scale=weight_scale_inv,
    #     scale_granularity_mnk=(1, 128, 128),
    #     scale_major_mode="K",
    #     backend="cutlass",
    #     out_dtype=torch.bfloat16,
    # )

    # Add bias (FlashInfer doesn't support bias directly)
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


def get_fp8_backend_name() -> str:
    """Get human-readable backend name."""
    return get_fp8_backend().value


def expand_block_scale(
    scale_inv: torch.Tensor,
    weight_shape: Tuple[int, int],
    block_size: Tuple[int, int] = DEFAULT_BLOCK_SIZE,
    out_dtype: Optional[torch.dtype] = None,
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
    block_h, block_w = _validate_block_size_2d(block_size, "block_size")
    out_features, in_features = weight_shape

    if out_dtype is None:
        out_dtype = scale_inv.dtype
    scale_inv = scale_inv.to(out_dtype)

    # Repeat each element block_h times along dim 0 and block_w times along dim 1
    expanded = scale_inv.repeat_interleave(
        block_h, dim=0).repeat_interleave(block_w, dim=1)

    # Handle edge cases where weight shape is not perfectly divisible by block size
    if expanded.shape[0] > out_features or expanded.shape[1] > in_features:
        expanded = expanded[:out_features, :in_features]

    return expanded


def _fp8_linear_fallback(
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
    block_size = _validate_block_size_2d(block_size, "block_size")
    _validate_scale_shape(weight_fp8, weight_scale_inv, block_size)

    compute_dtype = input.dtype if input.dtype in _SUPPORTED_OUTPUT_DTYPES else DEFAULT_SCALE_DTYPE
    input = _to_runtime_dtype(input, compute_dtype)

    # Expand block-wise scale to full shape
    weight_shape_2d = (weight_fp8.shape[0], weight_fp8.shape[1])
    weight_scale_expanded = expand_block_scale(
        weight_scale_inv, weight_shape_2d, block_size, out_dtype=compute_dtype
    )

    # Dequantize weight to BF16 (ensure result is bfloat16)
    weight_bf16 = weight_fp8.to(compute_dtype) * weight_scale_expanded

    if bias is not None:
        bias = _to_runtime_dtype(bias, compute_dtype)

    # Standard GEMM
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

    Args:
        input: Input tensor [M, K] in BF16
        weight_fp8: Weight tensor [N, K] in FP8 E4M3 format
        weight_scale_inv: Block-wise inverse scale [num_blocks_n, num_blocks_k]
        bias: Optional bias tensor [N]
        block_size: Block size for scale expansion

    Returns:
        Output tensor [M, N] in BF16
    """
    backend = get_fp8_backend()

    if backend == FP8Backend.FLASHINFER:
        return _fp8_linear_flashinfer(
            input, weight_fp8, weight_scale_inv, bias, block_size
        )

    return _fp8_linear_fallback(input, weight_fp8, weight_scale_inv, bias, block_size)


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
