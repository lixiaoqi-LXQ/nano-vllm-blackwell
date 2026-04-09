import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_cache = None  # FlashInfer combined kv cache [num_blocks, 2, block_size, num_kv_heads, head_dim]
        self.k_scale = None
        self.v_scale = None
        # FlashInfer wrappers (lazy init)
        self._prefill_wrapper = None
        self._decode_wrapper = None
        self._flashinfer_workspace = None

    @property
    def use_fp8_kv(self):
        return self.k_scale is not None

    def _ensure_flashinfer_wrappers(self):
        if self._prefill_wrapper is None:
            from flashinfer import BatchPrefillWithPagedKVCacheWrapper
            workspace = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda")
            self._flashinfer_workspace = workspace
            self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
        if self._decode_wrapper is None:
            from flashinfer import BatchDecodeWithPagedKVCacheWrapper
            workspace = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda")
            self._decode_workspace = workspace
            self._decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                workspace, "NHD", use_tensor_cores=True,
            )

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # Store to cache
        if k_cache.numel() and v_cache.numel():
            if self.use_fp8_kv:
                k = (k / self.k_scale).to(torch.float8_e4m3fn)
                v = (v / self.v_scale).to(torch.float8_e4m3fn)
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if self.use_fp8_kv and self.kv_cache is not None:
            return self._flashinfer_forward(q, k_cache, v_cache, context)
        else:
            return self._flashattn_forward(q, k, v, k_cache, v_cache, context)

    def _flashattn_forward(self, q, k, v, k_cache, v_cache, context):
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o

    def _flashinfer_forward(self, q, k_cache, v_cache, context):
        self._ensure_flashinfer_wrappers()

        kv_cache = self.kv_cache

        block_size = k_cache.size(1)
        q_dtype = q.dtype
        kv_dtype = k_cache.dtype

        if context.is_prefill:
            wrapper = self._prefill_wrapper
            wrapper.plan(
                context.cu_seqlens_q,
                context.flashinfer_kv_indptr,
                context.flashinfer_kv_indices,
                context.flashinfer_kv_last_page_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                block_size,
                causal=True,
                sm_scale=self.scale,
                q_data_type=q_dtype,
                kv_data_type=kv_dtype,
            )
            o = wrapper.run(q, kv_cache, k_scale=self.k_scale.item(), v_scale=self.v_scale.item())
        else:
            wrapper = self._decode_wrapper
            wrapper.plan(
                context.flashinfer_kv_indptr,
                context.flashinfer_kv_indices,
                context.flashinfer_kv_last_page_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                block_size,
                "NONE",
                sm_scale=self.scale,
                q_data_type=q_dtype,
                kv_data_type=kv_dtype,
            )
            o = wrapper.run(q, kv_cache, k_scale=self.k_scale.item(), v_scale=self.v_scale.item())

        return o
