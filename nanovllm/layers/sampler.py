# 采样器模块
import torch
from torch import nn


class Sampler(nn.Module):
    """采样器：使用Gumbel-max trick进行采样"""

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        前向计算：使用温度缩放和Gumbel-max trick进行采样

        Args:
            logits: 模型输出的logits [batch_size, vocab_size]
            temperatures: 采样温度 [batch_size]

        Returns:
            采样的token ids [batch_size]
        """
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-max trick: -G/log(p) 等价于从p中采样
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
