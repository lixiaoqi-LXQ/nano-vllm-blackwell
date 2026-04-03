# 性能基准测试脚本
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    """运行性能基准测试"""
    seed(0)
    num_seqs = 256        # 测试序列数
    max_input_len = 1024  # 最大输入长度
    max_ouput_len = 1024  # 最大输出长度

    path = os.path.expanduser("/data/models/Qwen3-0.6B")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 生成随机输入token ids
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    # 为每个序列生成随机采样参数
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_ouput_len) for _ in range(num_seqs)]
    # 取消下面的注释以使用vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # 预热
    llm.generate(["Benchmark: "], SamplingParams())
    # 正式测试
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
