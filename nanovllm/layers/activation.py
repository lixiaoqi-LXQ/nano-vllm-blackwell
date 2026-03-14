# 激活函数模块
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SiLU激活函数与乘法融合，用于SwiGLU MLP"""

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算：将输入拆分为两半，对前一半应用SiLU，然后与后一半相乘

        Args:
            x: 输入张量，最后一维被分为gate和up两部分

        Returns:
            激活后的张量
        """
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
