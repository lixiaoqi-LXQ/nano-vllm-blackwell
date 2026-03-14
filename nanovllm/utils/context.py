# 全局上下文模块，用于传递prefill/decode阶段的上下文信息
from dataclasses import dataclass
import torch


@dataclass
class Context:
    """推理上下文数据类"""
    is_prefill: bool = False                         # 是否为prefill阶段
    cu_seqlens_q: torch.Tensor | None = None        # 累积序列长度（query）
    cu_seqlens_k: torch.Tensor | None = None        # 累积序列长度（key）
    max_seqlen_q: int = 0                           # 最大序列长度（query）
    max_seqlen_k: int = 0                           # 最大序列长度（key）
    slot_mapping: torch.Tensor | None = None        # token到KV缓存slot的映射
    context_lens: torch.Tensor | None = None        # 各序列的上下文长度
    block_tables: torch.Tensor | None = None        # 块表

# 全局上下文实例
_CONTEXT = Context()

def get_context():
    """获取当前全局上下文"""
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """设置全局上下文"""
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """重置全局上下文"""
    global _CONTEXT
    _CONTEXT = Context()
