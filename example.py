# nano-vllm 使用示例
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    """示例：使用nano-vllm进行文本生成"""
    path = os.path.expanduser("/data/models/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    # 初始化LLM
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # 设置采样参数
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    # 准备prompt列表
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # 应用chat template
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    # 批量生成
    outputs = llm.generate(prompts, sampling_params)

    # 打印结果
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
