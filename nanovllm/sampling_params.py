# 采样参数模块
from dataclasses import dataclass


@dataclass
class SamplingParams:
    """采样参数类，控制生成过程的行为"""
    temperature: float = 1.0    # 采样温度，值越高越随机
    max_tokens: int = 64        # 最大生成的token数
    ignore_eos: bool = False    # 是否忽略结束符

    def __post_init__(self):
        """参数验证：不支持贪婪采样（temperature不能为0）"""
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
