"""不调用 LLM 的分支剪枝廉价启发式打分。

目标是避免暴力 LLM 推理：用确定性的几何启发式对走法排序／过滤，只在启发式
结果模糊时才调用模型。
"""

from __future__ import annotations

from typing import Dict, Optional

from kakeya_ai.grains import GrainsDecomposition


def heuristic_move_scores(
    grains: Optional[GrainsDecomposition],
    rho: float,
    delta: float,
    local_error: float,
    volume_ratio: float,
) -> Dict[int, float]:
    """为每个 Move 返回快速启发式分数。

    分数在 [0, 10]；越高表示越有希望。

    逻辑：
      - 当 grains 相对目标太短时，Move #1 是好的。
      - 当 rho 很大（grains "方正"）时，Move #2 是好的。
      - 当局部凸 Wolff 误差高时，Move #3 是好的。
    """
    scores = {1: 5.0, 2: 5.0, 3: 5.0}

    # Move #1：若 c 相对 delta/rho 之比偏小则鼓励。
    if grains is not None and grains.prisms:
        avg_c = sum(p.c for p in grains.prisms) / len(grains.prisms)
        target_c = (delta ** 0.01) * (rho / delta) * 0.5
        if avg_c < target_c:
            scores[1] += 3.0
        else:
            scores[1] -= 2.0

    # Move #2：若 rho 接近 1（方正的 grains）则鼓励。
    if rho > 0.5:
        scores[2] += 2.0 + 3.0 * min(rho, 1.0)
    else:
        scores[2] -= 1.0

    # Move #3：若局部误差高则鼓励。
    if local_error > 1.0:
        scores[3] += 2.0 + 3.0 * min(local_error / 10.0, 1.0)
    else:
        scores[3] -= 1.5

    # 体积比反馈：若已经超出上界，则所有走法都不那么紧急。
    if volume_ratio > 1.0:
        for k in scores:
            scores[k] -= 1.0

    # 夹到 [0, 10]。
    return {k: max(0.0, min(10.0, v)) for k, v in scores.items()}


def should_call_llm(heuristic_scores: Dict[int, float], margin: float = 2.0) -> bool:
    """若启发式分数太接近、无法脱离 LLM 做决定，则返回 True。

    Parameters
    ----------
    heuristic_scores : dict move_index -> score
    margin : float
        若最高分 - 次高分 < margin，则调用 LLM。
    """
    values = sorted(heuristic_scores.values(), reverse=True)
    if len(values) < 2:
        return False
    return values[0] - values[1] < margin
