"""针对深层（7B）模型的动态 token 预算辅助函数。

思路是让快速模型的置信度／不确定性来决定深层模型被允许花费多少 token。
高置信度的情形给更短的预算；不确定的情形给更长的预算。
"""

from __future__ import annotations


def compute_deep_budget(
    confidence: float,
    base: int = 1024,
    min_tokens: int = 256,
    max_tokens: int = 2048,
) -> int:
    """把 0-10 的置信度分数映射为 7B 模型的 ``num_predict`` 预算。

    Parameters
    ----------
    confidence : float
        取值范围 [0, 10] 的置信度；越高表示快速模型越确定。
    base : int
        置信度为 5 时的中性 token 预算。
    min_tokens, max_tokens : int
        硬下界与硬上界。

    Returns
    -------
    int
        用于深层模型调用的 token 预算。
    """
    # 把置信度归一化到 [0, 1]，并构造一个奖励确定性的缩放因子。
    # 置信度 0 -> base 的 1.5 倍；置信度 10 -> base 的 0.3 倍。
    c = max(0.0, min(10.0, float(confidence))) / 10.0
    scale = 1.5 - 1.2 * c
    budget = int(base * scale)
    return max(min_tokens, min(max_tokens, budget))


def compute_uncertainty_budget(
    top_two_gap: float,
    base: int = 1024,
    min_tokens: int = 256,
    max_tokens: int = 2048,
) -> int:
    """把 top-2 分数差映射为深层模型的 token 预算。

    差值大意味着决策明确（更短预算）；差值小意味着决策不确定（更长预算）。
    """
    # 把差值夹到合理范围再取反：差值小 -> 缩放因子大。
    gap = max(0.0, float(top_two_gap))
    scale = 1.5 - 0.12 * gap  # gap 为 0 -> 1.5 倍，gap 为 10 -> 0.3 倍
    scale = max(0.3, min(1.5, scale))
    budget = int(base * scale)
    return max(min_tokens, min(max_tokens, budget))
