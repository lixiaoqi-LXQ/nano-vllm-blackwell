"""NVFP4 utilities for FP4 quantization and matrix multiplication.

Provides:
- fp4_linear: Unified FP4 linear layer (auto-selects native or fallback backend)
- FP4 weight dequantization (fallback path)

NVFP4 format uses a two-level scaling scheme:
- Block scales: FP8 E4M3FN, one per 16 FP4 elements
- Global scale: FP32 scalar per tensor (weight_scale_2)
- Input scale: FP32 scalar per tensor
- FP4 values: E2M1 format (2 values packed per uint8)
"""

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
# Native FP4 path (FlashInfer mm_fp4 cutlass backend, SM100+)
# ---------------------------------------------------------------------------

# Lazy imports - only loaded when needed
_mm_fp4 = None
_nvfp4_quantize = None
_block_scale_interleave = None
_SfLayout = None
_native_loaded = False
_native_available = None


def _load_native_fp4():
    """Try to load FlashInfer native FP4 functions."""
    global _mm_fp4, _nvfp4_quantize, _block_scale_interleave, _SfLayout
    global _native_loaded, _native_available

    if _native_loaded:
        return _native_available

    _native_loaded = True
    try:
        from flashinfer import mm_fp4, nvfp4_quantize, SfLayout
        from flashinfer.quantization import block_scale_interleave

        _mm_fp4 = mm_fp4
        _nvfp4_quantize = nvfp4_quantize
        _block_scale_interleave = block_scale_interleave
        _SfLayout = SfLayout
        _native_available = True
    except ImportError:
        _native_available = False

    return _native_available


def _fp4_linear_native(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    input_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Native FP4 linear using FlashInfer mm_fp4 cutlass backend (SM100+)."""
    _load_native_fp4()

    # Quantize activation to FP4 at runtime.
    # The model's input_scale is the dequantization factor (= 1 / global_sf).
    # nvfp4_quantize expects global_sf, so we pass 1 / input_scale.
    global_sf_a = 1.0 / input_scale
    x_fp4, x_sf = _nvfp4_quantize(
        x, global_sf_a,
        sfLayout=_SfLayout.layout_128x4,
        do_shuffle=False,
    )

    # Swizzle weight block scales (checkpoint stores linear layout)
    w_sf = _block_scale_interleave(weight_scale.view(torch.uint8))

    # alpha = input_scale * weight_scale_2
    alpha = input_scale * weight_scale_2

    # mm_fp4 expects b shape (k, n), pass weight.T
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
# Unified entry point
# ---------------------------------------------------------------------------

def fp4_linear(
    x: torch.Tensor,
    weight_fp4: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Unified FP4 linear layer entry point.

    Automatically selects native (FlashInfer mm_fp4 cutlass) or fallback
    (dequantize to BF16) backend based on availability.

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
    if _load_native_fp4():
        return _fp4_linear_native(
            x, weight_fp4, weight_scale, weight_scale_2, input_scale, bias
        )
    return _fp4_linear_fallback(
        x, weight_fp4, weight_scale, weight_scale_2, input_scale, bias
    )


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


def get_fp4_info() -> str:
    """Get FP4 support information for logging."""
    if not torch.cuda.is_available():
        return "FP4: CUDA not available"
    device_name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    if _load_native_fp4():
        return f"FP4: Native (FlashInfer mm_fp4 cutlass) on {device_name} (SM {capability[0]}.{capability[1]})"
    return f"FP4: Fallback (dequantize) on {device_name} (SM {capability[0]}.{capability[1]})"
