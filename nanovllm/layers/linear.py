import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional


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

        # FP8 specific attributes
        self.is_fp8 = False
        self.weight_scale_inv: Optional[torch.Tensor] = None

        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

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
        param.data.copy_(loaded_weight)
        if loaded_scale is not None:
            self.is_fp8 = True
            self.weight_scale_inv = loaded_scale.to(param.data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_fp8 and self.weight_scale_inv is not None:
            from nanovllm.utils.fp8_utils import fp8_linear
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
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

        if loaded_scale is not None:
            self.is_fp8 = True
            # For column parallel, scale is sharded along dimension 0
            scale_shard_h = loaded_scale.size(0) // self.tp_size
            scale_start = self.tp_rank * scale_shard_h
            self.weight_scale_inv = loaded_scale[
                scale_start : scale_start + scale_shard_h
            ].to(param_data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_fp8 and self.weight_scale_inv is not None:
            from nanovllm.utils.fp8_utils import fp8_linear
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
        # For merged columns, we need to store scale for each shard separately
        self.weight_scale_inv_list: list[Optional[torch.Tensor]] = [None, None]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
        loaded_scale: Optional[torch.Tensor] = None,
    ):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

        if loaded_scale is not None:
            self.is_fp8 = True
            # Shard the scale along dimension 0
            scale_shard = loaded_scale.chunk(self.tp_size, 0)[self.tp_rank]
            self.weight_scale_inv_list[loaded_shard_id] = scale_shard.to(param_data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_fp8 and all(s is not None for s in self.weight_scale_inv_list):
            from nanovllm.utils.fp8_utils import fp8_linear

            # Split weights and apply separate FP8 GEMM
            gate_size = self.output_sizes[0] // self.tp_size
            gate_weight = self.weight[:gate_size]
            up_weight = self.weight[gate_size:]

            gate = fp8_linear(x, gate_weight, self.weight_scale_inv_list[0])
            up = fp8_linear(x, up_weight, self.weight_scale_inv_list[1])

            return torch.cat([gate, up], dim=-1)

        return F.linear(x, self.weight, self.bias)


class QKVParallelLinear(ColumnParallelLinear):

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
        # For QKV, store scales separately for q, k, v
        self.weight_scale_inv_dict: dict[str, Optional[torch.Tensor]] = {}

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
        loaded_scale: Optional[torch.Tensor] = None,
    ):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]

        if loaded_shard_id == "q":
            shard_size = self.q_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.kv_size
            shard_offset = self.q_size
        else:  # v
            shard_size = self.kv_size
            shard_offset = self.q_size + self.kv_size

        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

        if loaded_scale is not None:
            self.is_fp8 = True
            # Shard the scale along dimension 0
            scale_shard = loaded_scale.chunk(self.tp_size, 0)[self.tp_rank]
            self.weight_scale_inv_dict[loaded_shard_id] = scale_shard.to(param_data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_fp8 and len(self.weight_scale_inv_dict) == 3:
            from nanovllm.utils.fp8_utils import fp8_linear

            # Split weights for Q, K, V
            q_weight = self.weight[: self.q_size]
            k_weight = self.weight[self.q_size : self.q_size + self.kv_size]
            v_weight = self.weight[self.q_size + self.kv_size :]

            # Execute separate FP8 GEMM for each
            q = fp8_linear(x, q_weight, self.weight_scale_inv_dict["q"])
            k = fp8_linear(x, k_weight, self.weight_scale_inv_dict["k"])
            v = fp8_linear(x, v_weight, self.weight_scale_inv_dict["v"])

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
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

        if loaded_scale is not None:
            self.is_fp8 = True
            # For row parallel, scale is sharded along dimension 1
            scale_shard_w = loaded_scale.size(1) // self.tp_size
            scale_start = self.tp_rank * scale_shard_w
            self.weight_scale_inv = loaded_scale[
                :, scale_start : scale_start + scale_shard_w
            ].to(param_data.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_fp8 and self.weight_scale_inv is not None:
            from nanovllm.utils.fp8_utils import fp8_linear
            y = fp8_linear(
                x, self.weight, self.weight_scale_inv,
                self.bias if self.tp_rank == 0 else None
            )
        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)

        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
