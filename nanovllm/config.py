# nano-vllm 配置模块
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """推理引擎配置类"""
    model: str                              # 模型路径
    max_num_batched_tokens: int = 16384    # 批处理的最大token数
    max_num_seqs: int = 512                # 同时处理的最大序列数
    max_model_len: int = 4096              # 模型最大序列长度
    gpu_memory_utilization: float = 0.9     # GPU显存使用率
    tensor_parallel_size: int = 1          # 张量并行大小
    enforce_eager: bool = False            # 是否强制使用eager模式（不使用CUDA Graph）
    hf_config: AutoConfig | None = None   # HuggingFace配置
    eos: int = -1                          # 结束符token id
    kvcache_block_size: int = 256          # KV缓存块大小
    num_kvcache_blocks: int = -1          # KV缓存块数量（自动计算）

    def __post_init__(self):
        """配置初始化后的验证和设置"""
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
