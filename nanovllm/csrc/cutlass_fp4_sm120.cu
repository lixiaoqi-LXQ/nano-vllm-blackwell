/*
 * CUTLASS FP4 Block-Scaled GEMM for SM120.
 *
 * Ported from vLLM's nvfp4_scaled_mm_sm120_kernels.cu, with vLLM
 * dependencies inlined.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// Inline replacements for vLLM dependencies
#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                cutlassGetStatusString(error));     \
  }

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

// Tile configs
struct sm120_fp4_config_M256 {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_128, _128, _128>;
  using PerSmTileShape_MNK = Shape<_128, _128, _128>;
};

struct sm120_fp4_config_default {
  using ClusterShape = Shape<_1, _1, _1>;
  using MmaTileShape = Shape<_256, _128, _128>;
  using PerSmTileShape_MNK = Shape<_256, _128, _128>;
};

template <typename Config, typename OutType>
struct Fp4GemmSm120 {
  using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutATag = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 32;

  using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 32;

  using ElementD = OutType;
  using ElementC = OutType;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using MmaTileShape = typename Config::MmaTileShape;
  using ClusterShape = typename Config::ClusterShape;
  using PerSmTileShape_MNK = typename Config::PerSmTileShape_MNK;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, PerSmTileShape_MNK, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
          ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD,
          LayoutDTag, AlignmentD,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB,
          LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm>
typename Gemm::Arguments args_from_options(at::Tensor& D, at::Tensor const& A,
                                           at::Tensor const& B,
                                           at::Tensor const& A_sf,
                                           at::Tensor const& B_sf,
                                           torch::Tensor const& alpha, int M,
                                           int N, int K) {
  using ElementSFA = cutlass::float_ue4m3_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  using ElementCompute = float;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementD = typename Gemm::ElementD;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {static_cast<ElementA const*>(A.data_ptr()), stride_A,
       static_cast<ElementB const*>(B.data_ptr()), stride_B,
       static_cast<ElementSFA const*>(A_sf.data_ptr()), layout_SFA,
       static_cast<ElementSFB const*>(B_sf.data_ptr()), layout_SFB},
      {{},
       static_cast<ElementD const*>(D.data_ptr()),
       stride_D,
       static_cast<ElementD*>(D.data_ptr()),
       stride_D}};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha_ptr = static_cast<ElementCompute const*>(alpha.data_ptr());

  return arguments;
}

template <typename Gemm>
void runGemm(at::Tensor& D, at::Tensor const& A, at::Tensor const& B,
             at::Tensor const& A_sf, at::Tensor const& B_sf,
             torch::Tensor const& alpha, int M, int N, int K,
             cudaStream_t stream) {
  Gemm gemm;

  auto arguments = args_from_options<Gemm>(D, A, B, A_sf, B_sf, alpha, M, N, K);

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto const workspace_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(A.device());
  auto workspace = torch::empty(workspace_size, workspace_options);

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.run(arguments, workspace.data_ptr(), stream));
}

// Public API: cutlass_fp4_mm(A, B, A_sf, B_sf, alpha) -> Tensor
torch::Tensor cutlass_fp4_mm(torch::Tensor const& A,   // [M, K/2] uint8 (fp4 packed)
                             torch::Tensor const& B,   // [N, K/2] uint8 (fp4 packed)
                             torch::Tensor const& A_sf, // swizzled SF
                             torch::Tensor const& B_sf, // swizzled SF
                             torch::Tensor const& alpha) { // scalar float
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
  TORCH_CHECK(A.is_cuda() && A.is_contiguous(), "A must be a contiguous CUDA tensor");
  TORCH_CHECK(B.is_cuda() && B.is_contiguous(), "B must be a contiguous CUDA tensor");
  TORCH_CHECK(A_sf.is_cuda() && A_sf.is_contiguous(), "A_sf must be a contiguous CUDA tensor");
  TORCH_CHECK(B_sf.is_cuda() && B_sf.is_contiguous(), "B_sf must be a contiguous CUDA tensor");
  TORCH_CHECK(alpha.is_cuda(), "alpha must be a CUDA tensor");

  TORCH_CHECK(A.dim() == 2, "A must be a matrix");
  TORCH_CHECK(B.dim() == 2, "B must be a matrix");
  TORCH_CHECK(A.sizes()[1] == B.sizes()[1],
              "A and B shapes cannot be multiplied");

  auto const m = A.sizes()[0];
  auto const n = B.sizes()[0];
  auto const k = A.sizes()[1] * 2;  // 2 fp4 values per byte

  constexpr int alignment = 32;
  TORCH_CHECK(k % alignment == 0, "Expected k to be divisible by ", alignment);
  TORCH_CHECK(n % alignment == 0, "Expected n to be divisible by ", alignment);

  auto round_up_fn = [](int x, int y) { return (x + y - 1) / y * y; };
  int rounded_m = round_up_fn(m, 128);
  int rounded_n = round_up_fn(n, 128);
  int rounded_k = round_up_fn(k / 16, 4);

  TORCH_CHECK(A_sf.sizes()[0] == rounded_m && A_sf.sizes()[1] == rounded_k,
              "A_sf must be padded and swizzled to shape (", rounded_m, "x", rounded_k,
              "), but got (", A_sf.sizes()[0], "x", A_sf.sizes()[1], ")");
  TORCH_CHECK(B_sf.sizes()[0] == rounded_n && B_sf.sizes()[1] == rounded_k,
              "B_sf must be padded and swizzled to shape (", rounded_n, "x", rounded_k,
              "), but got (", B_sf.sizes()[0], "x", B_sf.sizes()[1], ")");

  auto D = torch::empty({m, n}, torch::dtype(torch::kBFloat16).device(A.device()));

  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.get_device());

  uint32_t const mp2 = std::max(static_cast<uint32_t>(16), next_pow_2(m));
  if (mp2 <= 256) {
    runGemm<Fp4GemmSm120<sm120_fp4_config_M256, cutlass::bfloat16_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  } else {
    runGemm<Fp4GemmSm120<sm120_fp4_config_default, cutlass::bfloat16_t>::Gemm>(
        D, A, B, A_sf, B_sf, alpha, m, n, k, stream);
  }

  return D;
#else
  TORCH_CHECK(false, "CUTLASS FP4 GEMM requires CUTLASS_ARCH_MMA_SM120_SUPPORTED");
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_fp4_mm", &cutlass_fp4_mm,
          "CUTLASS FP4 Block-Scaled GEMM for SM120");
}
