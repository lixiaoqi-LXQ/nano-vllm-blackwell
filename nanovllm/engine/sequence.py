# 序列管理模块，处理单个生成请求的状态
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列状态枚举"""
    WAITING = auto()    # 等待处理
    RUNNING = auto()    # 正在推理
    FINISHED = auto()   # 已完成


class Sequence:
    """序列类，表示一个待生成的请求"""
    block_size = 256   # KV缓存块大小
    counter = count()  # 序列ID计数器

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        初始化序列

        Args:
            token_ids: 输入prompt的token id列表
            sampling_params: 采样参数
        """
        self.seq_id = next(Sequence.counter)          # 唯一序列ID
        self.status = SequenceStatus.WAITING          # 初始状态为等待
        self.token_ids = copy(token_ids)               # 所有token ids
        self.last_token = token_ids[-1]                # 最后一个token
        self.num_tokens = len(self.token_ids)          # 总token数
        self.num_prompt_tokens = len(token_ids)        # prompt token数
        self.num_cached_tokens = 0                     # 已缓存的token数（用于前缀缓存）
        self.block_table = []                          # KV缓存块表
        self.temperature = sampling_params.temperature # 采样温度
        self.max_tokens = sampling_params.max_tokens   # 最大生成token数
        self.ignore_eos = sampling_params.ignore_eos   # 是否忽略EOS

    def __len__(self):
        """返回序列总长度"""
        return self.num_tokens

    def __getitem__(self, key):
        """索引访问token_ids"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """判断序列是否已完成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成的completion token数"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """获取prompt部分的token ids"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """获取已生成的completion token ids"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """已缓存的块数"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """序列需要的总块数"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """最后一个块中的token数"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """获取第i个块的token ids"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """追加一个生成的token"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """序列化，用于进程间通信"""
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """反序列化"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
