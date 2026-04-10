from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.fp8_utils import fp8_linear
from nanovllm.utils.fp4_utils import fp4_linear


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

        # FP4 scales - set by weight_loader if FP4 weights are loaded
        # [N, K/16] float8_e4m3fn
        self.weight_scale_fp4: Optional[torch.Tensor] = None
        # scalar float32
        self.weight_scale_2_fp4: Optional[torch.Tensor] = None
        self.input_scale_fp4: Optional[torch.Tensor] = None  # scalar float32

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

    def _is_fp4(self) -> bool:
        return self.weight_scale_fp4 is not None

    def _fp4_forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = getattr(self, "weight_fp4", self.weight)
        return fp4_linear(
            x, weight, self.weight_scale_fp4,
            self.weight_scale_2_fp4, self.input_scale_fp4, self.bias
        )

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
        loaded_scale_2: Optional[torch.Tensor] = None,
        loaded_input_scale: Optional[torch.Tensor] = None,
    ):
        if loaded_weight.dtype == torch.uint8 and loaded_scale_2 is not None:
            # FP4 path: store weight as buffer (uint8 can't be nn.Parameter)
            self.register_buffer("weight_fp4", loaded_weight.cuda())
            self.weight_scale_fp4 = loaded_scale.cuda()
            self.weight_scale_2_fp4 = loaded_scale_2.cuda()
            self.input_scale_fp4 = (
                loaded_input_scale.cuda()
                if loaded_input_scale is not None else None
            )
            # Free the original bf16 weight to save GPU memory
            self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
            return
        self._ensure_param_dtype(param, loaded_weight.dtype)
        param.data.copy_(loaded_weight)
        if loaded_scale is not None:
            self.weight_scale_inv = loaded_scale.to(param.data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fp4():
            return self._fp4_forward(x)
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
        loaded_scale_2: Optional[torch.Tensor] = None,
        loaded_input_scale: Optional[torch.Tensor] = None,
    ):
        tp_dim = self.tp_dim
        assert tp_dim is not None

        if loaded_weight.dtype == torch.uint8 and loaded_scale_2 is not None:
            # FP4 path: shard and store as buffer
            shard_size = loaded_weight.size(tp_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size
            weight_shard = loaded_weight.narrow(tp_dim, start_idx, shard_size)
            self.register_buffer("weight_fp4", weight_shard.cuda())

            # Shard scale along dimension 0
            scale_shard_h = loaded_scale.size(0) // self.tp_size
            scale_start = self.tp_rank * scale_shard_h
            self.weight_scale_fp4 = loaded_scale[
                scale_start: scale_start + scale_shard_h
            ].cuda()
            self.weight_scale_2_fp4 = loaded_scale_2.cuda()
            self.input_scale_fp4 = (
                loaded_input_scale.cuda()
                if loaded_input_scale is not None else None
            )
            # Free the original bf16 weight to save GPU memory
            self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
            return

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
        if self._is_fp4():
            return self._fp4_forward(x)
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

        # Store scales separately for each output (for FP8/FP4)
        self.weight_scale_inv_list: list[Optional[torch.Tensor]] = [None, None]
        # FP4 separate weights and scales per shard
        self.weight_fp4_list: list[Optional[torch.Tensor]] = [None, None]
        self.weight_scale_fp4_list: list[Optional[torch.Tensor]] = [None, None]
        self.weight_scale_2_fp4_list: list[Optional[torch.Tensor]] = [
            None, None]
        self.input_scale_fp4_list: list[Optional[torch.Tensor]] = [None, None]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
        loaded_scale: Optional[torch.Tensor] = None,
        loaded_weight_scale_2: Optional[torch.Tensor] = None,
        loaded_input_scale: Optional[torch.Tensor] = None,
    ):
        tp_dim = self.tp_dim

        if loaded_weight.dtype == torch.uint8 and loaded_weight_scale_2 is not None:
            # FP4 path: shard weight and store as separate buffer per shard
            loaded_shard = loaded_weight.chunk(
                self.tp_size, tp_dim)[self.tp_rank]
            self.weight_fp4_list[loaded_shard_id] = loaded_shard.cuda()

            scale_shard = loaded_scale.chunk(self.tp_size, 0)[self.tp_rank]
            self.weight_scale_fp4_list[loaded_shard_id] = scale_shard.cuda()
            self.weight_scale_2_fp4_list[loaded_shard_id] = (
                loaded_weight_scale_2.cuda()
            )
            self.input_scale_fp4_list[loaded_shard_id] = (
                loaded_input_scale.cuda()
                if loaded_input_scale is not None else None
            )
            # Merge FP4 weights once all shards are loaded
            if all(w is not None for w in self.weight_fp4_list):
                assert all(
                    torch.equal(self.weight_scale_2_fp4_list[0], s)
                    for s in self.weight_scale_2_fp4_list[1:]
                ), "weight_scale_2 must be identical across shards"
                assert all(
                    torch.equal(self.input_scale_fp4_list[0], s)
                    for s in self.input_scale_fp4_list[1:]
                ), "input_scale must be identical across shards"
                self.register_buffer("weight_fp4",
                                     torch.cat(self.weight_fp4_list, dim=0))
                self.weight_scale_fp4 = torch.cat(
                    self.weight_scale_fp4_list, dim=0)
                self.weight_scale_2_fp4 = self.weight_scale_2_fp4_list[0]
                self.input_scale_fp4 = self.input_scale_fp4_list[0]
                self.weight_fp4_list = [None, None]
                self.weight_scale_fp4_list = [None, None]
                self.weight_scale_2_fp4_list = [None, None]
                self.input_scale_fp4_list = [None, None]
                self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
            return

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
            scale_shard = scale_shard.to(param_data.device)
            # FP8 path
            self.weight_scale_inv_list[loaded_shard_id] = scale_shard
            # Pre-merge scales once all shards are loaded
            if all(s is not None for s in self.weight_scale_inv_list):
                self.weight_scale_inv = torch.cat(
                    self.weight_scale_inv_list, dim=0)
                self.weight_scale_inv_list = [None, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fp4():
            return self._fp4_forward(x)

        if self.weight_scale_inv is not None:
            return fp8_linear(x, self.weight, self.weight_scale_inv)

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

        # Store scales separately for Q, K, V (for FP8/FP4)
        self.weight_scale_inv_list: list[Optional[torch.Tensor]] = [
            None, None, None]
        # FP4 separate weights and scales per shard
        self.weight_fp4_list: list[Optional[torch.Tensor]] = [
            None, None, None]
        self.weight_scale_fp4_list: list[Optional[torch.Tensor]] = [
            None, None, None]
        self.weight_scale_2_fp4_list: list[Optional[torch.Tensor]] = [
            None, None, None]
        self.input_scale_fp4_list: list[Optional[torch.Tensor]] = [
            None, None, None]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
        loaded_scale: Optional[torch.Tensor] = None,
        loaded_weight_scale_2: Optional[torch.Tensor] = None,
        loaded_input_scale: Optional[torch.Tensor] = None,
    ):
        assert loaded_shard_id in self._SHARD_TO_INDEX
        tp_dim = self.tp_dim
        idx = self._SHARD_TO_INDEX[loaded_shard_id]

        if loaded_weight.dtype == torch.uint8 and loaded_weight_scale_2 is not None:
            # FP4 path: shard weight and store as separate buffer per shard
            loaded_shard = loaded_weight.chunk(
                self.tp_size, tp_dim)[self.tp_rank]
            self.weight_fp4_list[idx] = loaded_shard.cuda()

            scale_shard = loaded_scale.chunk(self.tp_size, 0)[self.tp_rank]
            self.weight_scale_fp4_list[idx] = scale_shard.cuda()
            self.weight_scale_2_fp4_list[idx] = (
                loaded_weight_scale_2.cuda()
            )
            self.input_scale_fp4_list[idx] = (
                loaded_input_scale.cuda()
                if loaded_input_scale is not None else None
            )
            # Merge FP4 weights once all shards are loaded
            if all(w is not None for w in self.weight_fp4_list):
                assert all(
                    torch.equal(self.weight_scale_2_fp4_list[0], s)
                    for s in self.weight_scale_2_fp4_list[1:]
                ), "weight_scale_2 must be identical across shards"
                assert all(
                    torch.equal(self.input_scale_fp4_list[0], s)
                    for s in self.input_scale_fp4_list[1:]
                ), "input_scale must be identical across shards"
                self.register_buffer("weight_fp4",
                                     torch.cat(self.weight_fp4_list, dim=0))
                self.weight_scale_fp4 = torch.cat(
                    self.weight_scale_fp4_list, dim=0)
                self.weight_scale_2_fp4 = self.weight_scale_2_fp4_list[0]
                self.input_scale_fp4 = self.input_scale_fp4_list[0]
                self.weight_fp4_list = [None, None, None]
                self.weight_scale_fp4_list = [None, None, None]
                self.weight_scale_2_fp4_list = [None, None, None]
                self.input_scale_fp4_list = [None, None, None]
                self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
            return

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
            scale_shard = loaded_scale.chunk(self.tp_size, 0)[self.tp_rank]
            scale_shard = scale_shard.to(param_data.device)
            # FP8 path
            self.weight_scale_inv_list[idx] = scale_shard
            # Pre-merge scales once all shards are loaded
            if all(s is not None for s in self.weight_scale_inv_list):
                self.weight_scale_inv = torch.cat(
                    self.weight_scale_inv_list, dim=0)
                self.weight_scale_inv_list = [None, None, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fp4():
            return self._fp4_forward(x)

        if self.weight_scale_inv is not None:
            return fp8_linear(x, self.weight, self.weight_scale_inv)

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
        loaded_scale_2: Optional[torch.Tensor] = None,
        loaded_input_scale: Optional[torch.Tensor] = None,
    ):
        tp_dim = self.tp_dim

        if loaded_weight.dtype == torch.uint8 and loaded_scale_2 is not None:
            # FP4 path: shard along dim 1 and store as buffer
            shard_size = loaded_weight.size(tp_dim) // self.tp_size
            start_idx = self.tp_rank * shard_size
            weight_shard = loaded_weight.narrow(tp_dim, start_idx, shard_size)
            self.register_buffer("weight_fp4", weight_shard.cuda())

            # Shard scale along dimension 1
            scale_shard_w = loaded_scale.size(1) // self.tp_size
            scale_start = self.tp_rank * scale_shard_w
            self.weight_scale_fp4 = loaded_scale[
                :, scale_start: scale_start + scale_shard_w
            ].cuda()
            self.weight_scale_2_fp4 = loaded_scale_2.cuda()
            self.input_scale_fp4 = (
                loaded_input_scale.cuda()
                if loaded_input_scale is not None else None
            )
            # Free the original bf16 weight to save GPU memory
            self.weight = nn.Parameter(torch.empty(0), requires_grad=False)
            return

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
            scale_sharded = loaded_scale[
                :, scale_start: scale_start + scale_shard_w
            ].to(param_data.device)
            self.weight_scale_inv = scale_sharded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_fp4():
            y = self._fp4_forward(x)
        elif self.weight_scale_inv is not None:
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
