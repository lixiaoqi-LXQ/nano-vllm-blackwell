# RMS归一化层模块
import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMS归一化层，支持融合residual连接"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """
        初始化RMS归一化层

        Args:
            hidden_size: 隐藏层维度
            eps: 数值稳定性常数
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        RMS归一化前向计算

        Args:
            x: 输入张量

        Returns:
            归一化后的张量
        """
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        融合residual连接的RMS归一化

        Args:
            x: 输入张量
            residual: residual张量

        Returns:
            tuple: (归一化后的张量, residual)
        """
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向计算

        Args:
            x: 输入张量
            residual: residual张量（可选）

        Returns:
            归一化结果，或(归一化结果, residual)
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
