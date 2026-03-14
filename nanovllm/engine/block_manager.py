# KV缓存块管理模块，管理前缀缓存和块分配
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """KV缓存块，用于前缀缓存共享"""

    def __init__(self, block_id):
        """
        初始化块

        Args:
            block_id: 块的唯一标识符
        """
        self.block_id = block_id    # 块ID
        self.ref_count = 0          # 引用计数，用于共享缓存
        self.hash = -1              # 块内容的哈希值
        self.token_ids = []         # 块中存储的token ids

    def update(self, hash: int, token_ids: list[int]):
        """更新块的内容和哈希值"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """块管理器，负责KV缓存块的分配、回收和前缀缓存"""

    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化块管理器

        Args:
            num_blocks: 总块数
            block_size: 每个块的token大小
        """
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()     # 哈希到块ID的映射（前缀缓存）
        self.free_block_ids: deque[int] = deque(range(num_blocks))  # 可用块ID队列
        self.used_block_ids: set[int] = set()              # 已使用块ID集合

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算token ids的哈希值

        Args:
            token_ids: 要哈希的token列表
            prefix: 前缀块的哈希值（用于计算前缀+当前块的组合哈希）

        Returns:
            int: 哈希值
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """分配一个块"""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """释放一个块"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否可以为序列分配足够的块"""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为序列分配块，支持前缀缓存

        相同内容的块会被复用，避免重复计算KV
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中，增加引用计数
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有块"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查是否可以向序列追加新token（需要新块时）"""
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """允许序列追加新token，必要时分配新块"""
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # 需要分配新块
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 刚好填满最后一个块，计算其哈希用于缓存
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 还在填充当前块中
            assert last_block.hash == -1
