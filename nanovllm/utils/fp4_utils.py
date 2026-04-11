"""NVFP4 utilities for FP4 quantization and matrix multiplication.

Provides:
- fp4_linear: Unified FP4 linear layer (auto-selects backend)
- FP4 weight dequantization (fallback path)
- swizzle_blockscale: Block scale swizzle for vllm-like backend

NVFP4 format uses a two-level scaling scheme:
- Block scales: FP8 E4M3FN, one per 16 FP4 elements
- Global scale: FP32 scalar per tensor (weight_scale_2)
- Input scale: FP32 scalar per tensor
- FP4 values: E2M1 format (2 values packed per uint8)

Backend priority: vllm-like → FlashInfer → fallback (dequantize)
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# FP4 dequantization (used by fallback path)
# ---------------------------------------------------------------------------

# E2M1 FP4 lookup table: index by 4-bit nibble
# Bit3=sign, Bit[2:0]=magnitude
# 0000->0, 0001->0.5, 0010->1, 0011->1.5, 0100->2, 0101->3, 0110->4, 0111->6
_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _unpack_fp4_to_float32(packed_uint8: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 packed FP4 E2M1 values to float32.

    Each uint8 contains two FP4 values: [upper_nibble | lower_nibble].
    Returns float32 tensor with shape [M, K] where K = packed_uint8.shape[1] * 2.
    """
    m, packed_k = packed_uint8.shape
    flat = packed_uint8.flatten()

    high = (flat >> 4) & 0xF
    low = flat & 0xF
    combined = torch.stack((low, high), dim=1).flatten()

    lut = _E2M1_LUT.to(device=packed_uint8.device)
    values = lut[combined.to(torch.long)]
    return values.reshape(m, packed_k * 2)


def _dequantize_nvfp4_weight(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    """Dequantize NVFP4 weight to BF16.

    Formula: dequant = fp4_val * block_scale * global_scale
    """
    fp4_values = _unpack_fp4_to_float32(weight_packed)  # [N, K]
    block_scales_f32 = weight_scale.view(torch.float8_e4m3fn).to(torch.float32)
    block_scales_expanded = block_scales_f32.repeat_interleave(
        16, dim=1)  # [N, K]
    global_scale = weight_scale_2.to(torch.float32).item()
    dequant = fp4_values * block_scales_expanded * global_scale
    return dequant.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Backend 1: vllm-like (JIT-compiled CUTLASS FP4 GEMM + C++ quant)
# ---------------------------------------------------------------------------


def swizzle_blockscale(scale: torch.Tensor) -> torch.Tensor:
    """Pad and block-interleave FP4 block-scales for CUTLASS layout."""
    assert scale.dtype == torch.float8_e4m3fn
    scale_ndim = scale.ndim
    if scale_ndim == 2:
        scale = scale.unsqueeze(0)
    assert scale.ndim == 3

    B, M, K = scale.shape
    M_padded = (M + 127) // 128 * 128
    K_padded = (K + 3) // 4 * 4

    padded = torch.zeros(
        (B, M_padded, K_padded), dtype=scale.dtype, device=scale.device
    )
    padded[:B, :M, :K] = scale
    padded = padded.reshape(B, M_padded // 128, 4, 32, K_padded // 4, 4)
    swizzled = padded.permute(0, 1, 4, 3, 2, 5).contiguous()

    if scale_ndim == 2:
        return swizzled.reshape(M_padded, K_padded)
    return swizzled.reshape(B, M_padded, K_padded)


_cutlass_fp4_gemm_mod = None
_nvfp4_quant_mod = None
_cutlass_loaded = False
_cutlass_available = None


def _get_cutlass_include_dirs():
    """Find CUTLASS include dirs from FlashInfer package."""
    try:
        import flashinfer
        fi_base = os.path.dirname(flashinfer.__file__)
        cutlass_base = os.path.join(fi_base, 'data', 'cutlass')
        inc = os.path.join(cutlass_base, 'include')
        util_inc = os.path.join(cutlass_base, 'tools', 'util', 'include')
        if os.path.isdir(inc) and os.path.isdir(util_inc):
            return [inc, util_inc]
    except ImportError:
        pass
    return None


def _get_cuda_arch_flags():
    """Get CUDA arch flags for current GPU."""
    major, minor = torch.cuda.get_device_capability()
    return [f"-gencode=arch=compute_{major * 10 + minor}f,code=sm_{major * 10 + minor}f"]


def _load_cutlass():
    """Try to JIT-compile and load custom CUTLASS FP4 kernels."""
    global _cutlass_fp4_gemm_mod, _nvfp4_quant_mod
    global _cutlass_loaded, _cutlass_available

    if _cutlass_loaded:
        return _cutlass_available

    _cutlass_loaded = True

    cutlass_inc = _get_cutlass_include_dirs()
    if cutlass_inc is None:
        _cutlass_available = False
        return False

    csrc_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'csrc')
    gemm_src = os.path.join(csrc_dir, 'cutlass_fp4_sm120.cu')
    quant_src = os.path.join(csrc_dir, 'nvfp4_quant_kernel.cu')

    if not os.path.isfile(gemm_src) or not os.path.isfile(quant_src):
        _cutlass_available = False
        return False

    try:
        from torch.utils.cpp_extension import load

        arch_flags = _get_cuda_arch_flags()

        _cutlass_fp4_gemm_mod = load(
            name='cutlass_fp4_gemm',
            sources=[gemm_src],
            extra_include_paths=cutlass_inc,
            extra_cuda_cflags=arch_flags + [
                '-DCUTLASS_ARCH_MMA_SM120_SUPPORTED',
                '-O3',
            ],
        )

        _nvfp4_quant_mod = load(
            name='nvfp4_quant',
            sources=[quant_src],
            extra_cuda_cflags=arch_flags + ['-O3'],
        )

        _cutlass_available = True
    except Exception:
        _cutlass_available = False

    return _cutlass_available


def _fp4_linear_cutlass(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    input_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP4 linear using custom CUTLASS GEMM + C++ quant kernel."""
    _load_cutlass()

    global_sf_a = 1.0 / input_scale
    a_fp4, a_sf = _nvfp4_quant_mod.scaled_fp4_quant(x, global_sf_a)

    w_sf = swizzle_blockscale(weight_scale.view(torch.float8_e4m3fn))

    alpha = input_scale * weight_scale_2

    out = _cutlass_fp4_gemm_mod.cutlass_fp4_mm(
        a_fp4, weight_fp4,
        a_sf,
        w_sf,
        alpha,
    )

    if bias is not None:
        out = out + bias
    return out


# ---------------------------------------------------------------------------
# Backend 2: FlashInfer (mm_fp4 cutlass backend)
# ---------------------------------------------------------------------------
_mm_fp4 = None
_nvfp4_quantize = None
_block_scale_interleave = None
_SfLayout = None
_flashinfer_loaded = False
_flashinfer_available = None


def _load_flashinfer():
    """Try to load FlashInfer native FP4 functions."""
    global _mm_fp4, _nvfp4_quantize, _block_scale_interleave, _SfLayout
    global _flashinfer_loaded, _flashinfer_available

    if _flashinfer_loaded:
        return _flashinfer_available

    _flashinfer_loaded = True
    try:
        from flashinfer import mm_fp4, nvfp4_quantize, SfLayout
        from flashinfer.quantization import block_scale_interleave

        _mm_fp4 = mm_fp4
        _nvfp4_quantize = nvfp4_quantize
        _block_scale_interleave = block_scale_interleave
        _SfLayout = SfLayout
        _flashinfer_available = True
    except ImportError:
        _flashinfer_available = False

    return _flashinfer_available


def _fp4_linear_flashinfer(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    input_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP4 linear using FlashInfer mm_fp4 cutlass backend (SM100+)."""
    _load_flashinfer()

    global_sf_a = 1.0 / input_scale
    x_fp4, x_sf = _nvfp4_quantize(
        x, global_sf_a,
        sfLayout=_SfLayout.layout_128x4,
        do_shuffle=False,
    )

    w_sf = _block_scale_interleave(weight_scale.view(torch.uint8))

    alpha = input_scale * weight_scale_2

    out = _mm_fp4(
        a=x_fp4,
        b=weight_fp4.T,
        a_descale=x_sf,
        b_descale=w_sf,
        alpha=alpha,
        out_dtype=torch.bfloat16,
        block_size=16,
        use_nvfp4=True,
        backend="cutlass",
    )

    if bias is not None:
        out = out + bias
    return out


# ---------------------------------------------------------------------------
# Backend 3: Fallback (dequantize to BF16)
# ---------------------------------------------------------------------------

def _fp4_linear_fallback(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fallback FP4 linear: dequantize weight to BF16, then F.linear."""
    weight_bf16 = _dequantize_nvfp4_weight(
        weight_packed, weight_scale, weight_scale_2)
    return F.linear(x, weight_bf16.to(x.device), bias)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def _resolve_backend():
    """Resolve the best available FP4 backend."""
    if _load_cutlass():
        return 'vllm-like'
    elif _load_flashinfer():
        return 'flashinfer'
    return 'fallback'


def fp4_linear(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unified FP4 linear layer entry point.

    Automatically selects the best available backend:
    1. vllm-like (custom CUTLASS GEMM + C++ quant) - preferred
    2. FlashInfer (mm_fp4 cutlass) - second choice
    3. Fallback (dequantize to BF16) - last resort

    Args:
        x: Input tensor [M, K] in BF16
        weight_fp4: [N, K/2] uint8 packed FP4 values
        weight_scale: [N, K/16] float8_e4m3fn block scales
        weight_scale_2: scalar float32 global scale
        input_scale: scalar float32 activation scale
        bias: Optional bias [N]

    Returns:
        Output tensor [M, N] in BF16
    """
    backend = _resolve_backend()

    if backend == 'vllm-like':
        return _fp4_linear_cutlass(
            x, weight_fp4, weight_scale, weight_scale_2, input_scale, bias
        )
    elif backend == 'flashinfer':
        return _fp4_linear_flashinfer(
            x, weight_fp4, weight_scale, weight_scale_2, input_scale, bias
        )
    else:
        return _fp4_linear_fallback(
            x, weight_fp4, weight_scale, weight_scale_2, input_scale, bias
        )


def fp4_linear_chunked(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """FP4 linear with M-dimension chunking for large batches.

    Splits the input along dim 0 into chunks of `chunk_size`, runs fp4_linear
    on each chunk, and concatenates the results. This avoids CUTLASS tile
    scheduling inefficiencies with very large M values.

    Args:
        x: Input tensor [M, K] in BF16
        weight_fp4: [N, K/2] uint8 packed FP4 values
        weight_scale: [N, K/16] float8_e4m3fn block scales
        weight_scale_2: scalar float32 global scale
        input_scale: scalar float32 activation scale
        bias: Optional bias [N]
        chunk_size: Maximum rows per chunk

    Returns:
        Output tensor [M, N] in BF16
    """
    M = x.size(0)
    if M <= chunk_size:
        return fp4_linear(x, weight_fp4, weight_scale,
                          weight_scale_2, input_scale, bias)
    chunks = []
    for i in range(0, M, chunk_size):
        out_i = fp4_linear(
            x[i:i + chunk_size], weight_fp4, weight_scale,
            weight_scale_2, input_scale, None)
        chunks.append(out_i)
    result = torch.cat(chunks, dim=0)
    if bias is not None:
        result = result + bias
    return result


def get_fp4_info() -> str:
    """Get FP4 support information for logging."""
    if not torch.cuda.is_available():
        return "FP4: CUDA not available"
    device_name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    backend = _resolve_backend()
    return f"FP4: {backend} on {device_name} (SM {capability[0]}.{capability[1]})"
