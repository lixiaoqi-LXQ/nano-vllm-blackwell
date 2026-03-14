# 旋转位置编码（RoPE）模块
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    应用旋转位置编码

    Args:
        x: 输入张量
        cos: 余弦值
        sin: 正弦值

    Returns:
        应用RoPE后的张量
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """旋转位置编码（RoPE）"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        """
        初始化旋转位置编码

        Args:
            head_size: 头维度
            rotary_dim: 旋转维度（通常等于head_size）
            max_position_embeddings: 最大位置编码长度
            base: 旋转角度基数
        """
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # 计算逆频率
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向计算：对query和key应用旋转位置编码

        Args:
            positions: 位置索引
            query: Query张量
            key: Key张量

        Returns:
            tuple: (应用RoPE后的query, 应用RoPE后的key)
        """
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """
    获取旋转位置编码实例（缓存）

    Args:
        head_size: 头维度
        rotary_dim: 旋转维度
        max_position: 最大位置
        base: 旋转基数
        rope_scaling: RoPE扩展配置

    Returns:
        RotaryEmbedding实例
    """
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
