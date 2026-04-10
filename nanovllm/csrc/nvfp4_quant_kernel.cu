/*
 * FP4 Quantization Kernel with Swizzled Scale Factor Layout.
 *
 * Ported from vLLM's nvfp4_quant_kernels.cu and nvfp4_utils.cuh,
 * with vLLM dependencies inlined. Only supports BF16 input and
 * swizzled SF layout.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_runtime.h>
#include <cuda_fp8.h>

// We use 8 elements per thread (not 16) since SM120 does not support pack16.
#define ELTS_PER_THREAD 8
constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr bool CVT_FP4_PACK16 = false;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;

// --- Inline utility functions ---

template <typename Int>
__host__ __device__ inline Int round_up(Int x, Int y) {
  return ((x + y - 1) / y) * y;
}

template <typename Int>
__host__ __device__ inline Int div_round_up(Int x, Int y) {
  return (x + y - 1) / y;
}

inline int computeEffectiveRows(int m) {
  constexpr int ROW_TILE = 128;
  return round_up(m, ROW_TILE);
}

// 16-byte packed vector type for BF16 (4 x bfloat162 = 8 elements).
template <class Type>
struct alignas(16) PackedVec {
  typename Type::type2 elts[4];
};

template <>
struct alignas(16) PackedVec<__nv_bfloat16> {
  __nv_bfloat162 elts[4];
};

// --- FP4 conversion intrinsics ---

// Convert 4 float2 values into 8 e2m1 values (one uint32_t).
inline __device__ uint32_t fp32_vec8_to_e2m1(float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}\n"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
}

__device__ __forceinline__ uint32_t pack_fp4(float2 (&v)[4]) {
  return fp32_vec8_to_e2m1(v);
}

// Fast reciprocal.
__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

// Conditional 128-bit load (4 x uint32).
template <class Type>
__device__ __forceinline__ void ld128_or_zero_cg_u32(PackedVec<Type>& out,
                                                     const void* ptr,
                                                     bool pred) {
  uint32_t r0, r1, r2, r3;

  asm volatile(
      "{\n"
      "  .reg .pred pr;\n"
      "  setp.ne.u32 pr, %4, 0;\n"
      "  mov.u32 %0, 0;\n"
      "  mov.u32 %1, 0;\n"
      "  mov.u32 %2, 0;\n"
      "  mov.u32 %3, 0;\n"
      "  @pr ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%5];\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"((int)pred), "l"(ptr));

  *reinterpret_cast<uint4*>(&out) = uint4{r0, r1, r2, r3};
}

// Compute SF output offset for swizzled tensor core layout.
// SF layout: [numMTiles, numKTiles, 32, 4, 4]
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ __forceinline__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(
    int rowIdx, int colIdx, int32_t numKTiles, SFType* SFout) {
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 ||
                CVT_FP4_NUM_THREADS_PER_SF == 2);

  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF != 0) {
    return nullptr;
  }

  int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
  int32_t mIdx = rowIdx;

  int32_t mTileIdx = mIdx >> 7;
  int32_t outerMIdx = mIdx & 31;
  int32_t innerMIdx = (mIdx >> 5) & 3;
  int32_t kTileIdx = kIdx >> 2;
  int32_t innerKIdx = kIdx & 3;

  int64_t SFOffset = (static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx)
                         << 9 |
                     (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx;

  return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
}

// Quantize a PackedVec of BF16 into uint32_t FP4 output + write SF.
template <class Type, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ __forceinline__ uint32_t
cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
  auto localMax = __habs2(vec.elts[0]);

#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    localMax = __hmax2(__shfl_xor_sync(0xffffffffu, localMax, 1), localMax);
  }

  float vecMax = __bfloat162float(__hmax(localMax.x, localMax.y));
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));

  // Use UE4M3 SF format (same as E4M3 for positive values).
  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
  uint8_t fp8SFVal;
  reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
  SFValue = float(tmp);

  if (SFout) *SFout = fp8SFVal;

  float outputScale =
      SFValue != 0.0f ? reciprocal_approximate_ftz(
                            SFValue * reciprocal_approximate_ftz(SFScaleVal))
                      : 0.0f;

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  return pack_fp4(fp2Vals);
}

// --- Main quantization kernel ---

template <class Type>
__global__ void __launch_bounds__(512, 3)
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, int32_t num_padded_cols,
                    Type const* __restrict__ in,
                    float const* __restrict__ SFScale,
                    uint32_t* __restrict__ out, uint32_t* __restrict__ SFout) {
  using PackedVecT = PackedVec<Type>;

  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);

  int32_t const numKTiles = (numCols + 63) / 64;
  int sf_m = round_up<int>(numRows, 128);
  int32_t const colIdx = blockDim.x * blockIdx.y + threadIdx.x;
  int elem_idx = colIdx * CVT_FP4_ELTS_PER_THREAD;

  float const global_scale = (SFScale == nullptr) ? 1.0f : SFScale[0];

  for (int rowIdx = blockIdx.x; rowIdx < sf_m; rowIdx += gridDim.x) {
    if (colIdx < num_padded_cols) {
      PackedVecT in_vec;
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;

      bool valid = (rowIdx < numRows) && (elem_idx < numCols);
      ld128_or_zero_cg_u32<Type>(
          in_vec, &reinterpret_cast<const uint32_t*>(in)[inOffset * 4],
          valid);

      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, numKTiles, SFout);

      auto out_val =
          cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF>(
              in_vec, global_scale, sf_out);

      if (valid) {
        out[inOffset] = out_val;
      }
    }
  }
}

// --- Python binding ---

std::tuple<torch::Tensor, torch::Tensor>
scaled_fp4_quant(torch::Tensor const& input,
                 torch::Tensor const& input_global_scale) {
  int32_t m = input.size(0);
  int32_t n = input.size(1);

  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::BFloat16,
              "Only BF16 input is supported.");
  TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
              "Input must be a contiguous CUDA tensor.");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  // Output FP4 tensor: [m, n/2] uint8 (each byte holds 2 fp4 values)
  // Stored as [m, n/8] uint32 for alignment
  auto output = torch::empty({m, n / 8}, torch::dtype(torch::kInt32).device(input.device()));

  // Swizzled SF output dimensions
  auto round_up_fn = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up_fn(m, 128);
  int rounded_n = round_up_fn(n, 128);
  int rounded_k = round_up_fn(n / 16, 4);  // n/16 = number of SF in K dim

  auto output_sf = torch::empty({rounded_m, rounded_k},
                                torch::dtype(torch::kFloat8_e4m3fn).device(input.device()));

  int sf_n_unpadded = int(n / CVT_FP4_SF_VEC_SIZE);

  // Grid, Block size. Each thread converts 8 values.
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));

  // Compute blocks per SM for SM120 (1536 threads/SM / 512 = 3, clamped to cap 4)
  int max_threads_per_sm;
  cudaDeviceGetAttribute(&max_threads_per_sm,
                         cudaDevAttrMaxThreadsPerMultiProcessor,
                         input.get_device());
  int numBlocksPerSM = std::min(4, std::max(1, max_threads_per_sm / static_cast<int>(block.x)));

  int multiProcessorCount;
  cudaDeviceGetAttribute(&multiProcessorCount,
                         cudaDevAttrMultiProcessorCount,
                         input.get_device());

  int sf_n_int = int(round_up<int>(sf_n_unpadded, 4) / 4);
  int32_t num_padded_cols =
      sf_n_int * 4 * CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD;

  int grid_y = div_round_up(num_padded_cols, static_cast<int>(block.x));
  int grid_x =
      std::min(computeEffectiveRows(m),
               std::max(1, (multiProcessorCount * numBlocksPerSM) / grid_y));
  dim3 grid(grid_x, grid_y);

  auto input_ptr = static_cast<__nv_bfloat16 const*>(input.data_ptr());
  auto input_sf_ptr = static_cast<float const*>(input_global_scale.data_ptr());

  cvt_fp16_to_fp4<__nv_bfloat16><<<grid, block, 0, stream>>>(
      m, n, num_padded_cols, input_ptr, input_sf_ptr,
      reinterpret_cast<uint32_t*>(output.data_ptr()),
      reinterpret_cast<uint32_t*>(output_sf.data_ptr()));

  // output: [m, n/8] int32 -> view as [m, n/2] uint8
  auto output_uint8 = output.view(torch::kUInt8);

  return std::make_tuple(output_uint8, output_sf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("scaled_fp4_quant", &scaled_fp4_quant,
          "FP4 Quantization Kernel with Swizzled SF Layout (BF16 only)");
}
