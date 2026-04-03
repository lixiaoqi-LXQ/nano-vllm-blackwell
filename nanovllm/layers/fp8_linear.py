"""Independent FP8 Linear Layers for nano-vllm.

This module provides FP8-specific linear layers that are completely independent
from BF16 layers. These layers store weights in FP8 format and use native FP8
GEMM operations with automatic fallback to dequantization.

Key features:
- Native FP8 GEMM using torch._scaled_mm on supported hardware
- Automatic fallback to dequantization if native FP8 fails
- Tensor parallelism support
- Block-wise scale handling (128x128 blocks)
"""

from typing import Optional

import torch
from torch import nn
import torch.distributed as dist


def divide(numerator: int, denominator: int) -> int:
    """Integer division with assertion."""
    assert numerator % denominator == 0
    return numerator // denominator


class FP8LinearBase(nn.Module):
    """Base class for FP8 linear layers.

    This is independent from BF16 layers and specifically designed for FP8
    weights with block-wise scaling.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()

        # FP8 weight - stored in FP8 E4M3 format
        self.weight = nn.Parameter(
            torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn)
        )
        self.weight.weight_loader = self.weight_loader

        # FP8 scale (inverse) - will be set by weight_loader
        self.weight_scale_inv: Optional[torch.Tensor] = None

        # Track if native FP8 is being used (set after first forward)
        self._use_native_fp8: Optional[bool] = None

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size, dtype=torch.bfloat16))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using FP8 GEMM with automatic fallback."""
        from nanovllm.utils.fp8_utils import fp8_linear_native

        return fp8_linear_native(x, self.weight, self.weight_scale_inv, self.bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_scale: Optional[torch.Tensor] = None,
    ):
        """Load FP8 weight and scale.

        Args:
            param: Parameter to load into
            loaded_weight: FP8 weight tensor
            loaded_scale: Block-wise inverse scale (must be provided for FP8)
        """
        param.data.copy_(loaded_weight)
        if loaded_scale is not None:
            # Convert to float32 for precision and move to device
            self.weight_scale_inv = loaded_scale.float().to(param.data.device)


class FP8ReplicatedLinear(FP8LinearBase):
    """FP8 replicated linear layer (no tensor parallelism)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)


class FP8ColumnParallelLinear(FP8LinearBase):
    """FP8 column-parallel linear layer.

    The weight is sharded along the output dimension (dim 0).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_scale: Optional[torch.Tensor] = None,
    ):
        """Load sharded FP8 weight and scale.

        For column parallel, we shard along dimension 0 of the weight.
        The scale is also sharded along dimension 0.
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size

        # Shard weight
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

        if loaded_scale is not None:
            # Shard scale along dimension 0
            scale_shard_h = loaded_scale.size(0) // self.tp_size
            scale_start = self.tp_rank * scale_shard_h
            self.weight_scale_inv = loaded_scale[
                scale_start : scale_start + scale_shard_h
            ].float().to(param_data.device)


class FP8RowParallelLinear(FP8LinearBase):
    """FP8 row-parallel linear layer.

    The weight is sharded along the input dimension (dim 1).
    Requires all_reduce after the matmul.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with all_reduce for tensor parallelism."""
        from nanovllm.utils.fp8_utils import fp8_linear_native

        # Only rank 0 applies bias during forward
        y = fp8_linear_native(
            x, self.weight, self.weight_scale_inv,
            self.bias if self.tp_rank == 0 else None
        )

        # All-reduce across tensor parallel ranks
        if self.tp_size > 1:
            dist.all_reduce(y)

        return y

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_scale: Optional[torch.Tensor] = None,
    ):
        """Load sharded FP8 weight and scale.

        For row parallel, we shard along dimension 1 of the weight.
        The scale is sharded along dimension 1.
        """
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size

        # Shard weight
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

        if loaded_scale is not None:
            # Shard scale along dimension 1
            scale_shard_w = loaded_scale.size(1) // self.tp_size
            scale_start = self.tp_rank * scale_shard_w
            self.weight_scale_inv = loaded_scale[
                :, scale_start : scale_start + scale_shard_w
            ].float().to(param_data.device)


class FP8QKVParallelLinear(FP8ColumnParallelLinear):
    """FP8 QKV parallel linear layer.

    Combines Q, K, V projections into a single linear layer with separate
    scales for each component.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads

        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        self.q_size = self.num_heads * self.head_size
        self.kv_size = self.num_kv_heads * self.head_size

        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

        # Store scales separately for Q, K, V
        self.weight_scale_inv_dict: dict[str, torch.Tensor] = {}

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
        loaded_scale: Optional[torch.Tensor] = None,
    ):
        """Load Q, K, or V shard with its corresponding scale."""
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]

        # Calculate shard position
        if loaded_shard_id == "q":
            shard_size = self.q_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.kv_size
            shard_offset = self.q_size
        else:  # v
            shard_size = self.kv_size
            shard_offset = self.q_size + self.kv_size

        # Get the target slice and load sharded weight
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

        if loaded_scale is not None:
            # Shard scale and store separately
            scale_shard = loaded_scale.chunk(self.tp_size, 0)[self.tp_rank]
            self.weight_scale_inv_dict[loaded_shard_id] = scale_shard.float().to(param_data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with separate FP8 GEMM for Q, K, V."""
        from nanovllm.utils.fp8_utils import fp8_linear_native

        # Split weights for Q, K, V
        q_weight = self.weight[: self.q_size]
        k_weight = self.weight[self.q_size : self.q_size + self.kv_size]
        v_weight = self.weight[self.q_size + self.kv_size :]

        # Execute separate FP8 GEMM for each
        q = fp8_linear_native(x, q_weight, self.weight_scale_inv_dict["q"])
        k = fp8_linear_native(x, k_weight, self.weight_scale_inv_dict["k"])
        v = fp8_linear_native(x, v_weight, self.weight_scale_inv_dict["v"])

        return torch.cat([q, k, v], dim=-1)


class FP8MergedColumnParallelLinear(FP8ColumnParallelLinear):
    """FP8 merged column-parallel linear layer.

    Combines two column-parallel layers (e.g., gate_proj and up_proj) into
    a single layer with separate scales.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

        # Store scales separately for each output
        self.weight_scale_inv_list: list[Optional[torch.Tensor]] = [None, None]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
        loaded_scale: Optional[torch.Tensor] = None,
    ):
        """Load gate or up shard with its corresponding scale."""
        param_data = param.data

        # Calculate shard position
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size

        # Get the target slice and load sharded weight
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

        if loaded_scale is not None:
            # Shard scale and store separately
            scale_shard = loaded_scale.chunk(self.tp_size, 0)[self.tp_rank]
            self.weight_scale_inv_list[loaded_shard_id] = scale_shard.float().to(param_data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with separate FP8 GEMM for gate and up."""
        from nanovllm.utils.fp8_utils import fp8_linear_native

        # Split weights
        gate_size = self.output_sizes[0] // self.tp_size
        gate_weight = self.weight[:gate_size]
        up_weight = self.weight[gate_size:]

        # Execute separate FP8 GEMM for each
        gate = fp8_linear_native(x, gate_weight, self.weight_scale_inv_list[0])
        up = fp8_linear_native(x, up_weight, self.weight_scale_inv_list[1])

        return torch.cat([gate, up], dim=-1)
