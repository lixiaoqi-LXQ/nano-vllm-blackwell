from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.fp8_utils import fp8_linear


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

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
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader

        # FP8 scale (inverse) - set by weight_loader if FP8 weights are loaded
        self.weight_scale_inv: Optional[torch.Tensor] = None

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _ensure_param_dtype(param: nn.Parameter, target_dtype: torch.dtype) -> None:
        """Ensure parameter storage dtype matches loaded checkpoint dtype."""
        if param.data.dtype != target_dtype:
            param.data = param.data.to(dtype=target_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_scale: Optional[torch.Tensor] = None,
    ):
        self._ensure_param_dtype(param, loaded_weight.dtype)
        param.data.copy_(loaded_weight)
        if loaded_scale is not None:
            self.weight_scale_inv = loaded_scale.to(param.data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_scale_inv is not None:
            return fp8_linear(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_scale: Optional[torch.Tensor] = None,
    ):
        tp_dim = self.tp_dim
        assert tp_dim is not None
        self._ensure_param_dtype(param, loaded_weight.dtype)
        param_data = param.data
        shard_size = param_data.size(tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

        if loaded_scale is not None:
            # Shard scale along dimension 0
            scale_shard_h = loaded_scale.size(0) // self.tp_size
            scale_start = self.tp_rank * scale_shard_h
            self.weight_scale_inv = loaded_scale[
                scale_start: scale_start + scale_shard_h
            ].to(param_data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_scale_inv is not None:
            return fp8_linear(x, self.weight, self.weight_scale_inv, self.bias)
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)
        self._gate_size = self.output_sizes[0] // self.tp_size

        # Store scales separately for each output (for FP8)
        self.weight_scale_inv_list: list[Optional[torch.Tensor]] = [None, None]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
        loaded_scale: Optional[torch.Tensor] = None,
    ):
        tp_dim = self.tp_dim
        self._ensure_param_dtype(param, loaded_weight.dtype)
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

        if loaded_scale is not None:
            # Shard scale and store separately
            scale_shard = loaded_scale.chunk(self.tp_size, 0)[self.tp_rank]
            self.weight_scale_inv_list[loaded_shard_id] = scale_shard.to(
                param_data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_scale_inv_list[0] is not None:
            # FP8 path: separate GEMM for gate and up
            gate_scale, up_scale = self.weight_scale_inv_list
            gate_weight = self.weight[: self._gate_size]
            up_weight = self.weight[self._gate_size:]

            gate = fp8_linear(x, gate_weight, gate_scale)
            up = fp8_linear(x, up_weight, up_scale)
            return torch.cat([gate, up], dim=-1)

        return F.linear(x, self.weight, self.bias)


class QKVParallelLinear(ColumnParallelLinear):
    _SHARD_ORDER = ("q", "k", "v")
    _SHARD_TO_INDEX = {"q": 0, "k": 1, "v": 2}

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
        output_size = (total_num_heads + 2 *
                       total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)
        self._q_slice = slice(0, self.q_size)
        self._k_slice = slice(self.q_size, self.q_size + self.kv_size)
        self._v_slice = slice(self.q_size + self.kv_size,
                              self.q_size + 2 * self.kv_size)

        # Store scales separately for Q, K, V (for FP8)
        self.weight_scale_inv_list: list[Optional[torch.Tensor]] = [
            None, None, None]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
        loaded_scale: Optional[torch.Tensor] = None,
    ):
        assert loaded_shard_id in self._SHARD_TO_INDEX
        tp_dim = self.tp_dim
        self._ensure_param_dtype(param, loaded_weight.dtype)
        param_data = param.data

        if loaded_shard_id == "q":
            shard_size = self.q_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.kv_size
            shard_offset = self.q_size
        else:
            shard_size = self.kv_size
            shard_offset = self.q_size + self.kv_size

        param_data = param_data.narrow(tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

        if loaded_scale is not None:
            # Shard scale and store separately
            scale_shard = loaded_scale.chunk(self.tp_size, 0)[self.tp_rank]
            self.weight_scale_inv_list[self._SHARD_TO_INDEX[loaded_shard_id]] = scale_shard.to(
                param_data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_scale_inv_list[0] is not None:
            # FP8 path: separate GEMM for Q, K, V
            q_scale, k_scale, v_scale = self.weight_scale_inv_list
            q_weight = self.weight[self._q_slice]
            k_weight = self.weight[self._k_slice]
            v_weight = self.weight[self._v_slice]

            q = fp8_linear(x, q_weight, q_scale)
            k = fp8_linear(x, k_weight, k_scale)
            v = fp8_linear(x, v_weight, v_scale)
            return torch.cat([q, k, v], dim=-1)

        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_scale: Optional[torch.Tensor] = None,
    ):
        tp_dim = self.tp_dim
        self._ensure_param_dtype(param, loaded_weight.dtype)
        param_data = param.data
        shard_size = param_data.size(tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

        if loaded_scale is not None:
            # Shard scale along dimension 1
            scale_shard_w = loaded_scale.size(1) // self.tp_size
            scale_start = self.tp_rank * scale_shard_w
            self.weight_scale_inv = loaded_scale[
                :, scale_start: scale_start + scale_shard_w
            ].to(param_data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_scale_inv is not None:
            y = fp8_linear(
                x, self.weight, self.weight_scale_inv,
                self.bias if self.tp_rank == 0 else None
            )
        else:
            y = F.linear(x, self.weight,
                         self.bias if self.tp_rank == 0 else None)

        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
