"""拆核并发规划（Q4 #14，2026-11-13）。

**问题**：可并行的算力任务此前只会"各自满核串行"——任务 A 用全部核
跑完，任务 B 再用全部核。但在 P+E 异构拓扑上，"9+9 串行"常常输给
"4+4 并发"：E 核效率低，第二个任务全核跑时大量时间花在低效核上。

**做法**：枚举核数分配（n1, n2, …, Σni ≤ 总核数），对每种分配用
``chip_model.predict_schedule`` 的模型预测各任务墙钟，并发完成时间取
``max(T_i)``，与"各自满核串行"基线（``Σ T_i(满核)``）对比，选最优。

模型级、确定性、可测试；真实墙钟测量受系统噪声影响（PROJECT.md 局限：
约 1.7× 未定位噪声），故基准以模型预测为准、实测脚本另列。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..model.chip_model import predict_schedule


@dataclass(frozen=True)
class SplitPlan:
    """一次拆核并发的规划结果。"""

    allocation: tuple[int, ...]      # 各任务分到的核数
    predicted_each_ms: tuple[float, ...]  # 各任务在分得核数下的预测墙钟
    predicted_makespan_ms: float      # 并发完成时间 = max(each)
    serial_baseline_ms: float         # 各自满核串行的预测总时长
    speedup: float                    # serial / makespan（>1 即拆核更优）

    @property
    def worth_it(self) -> bool:
        """拆核是否值得（严格优于串行满核）。"""
        return self.speedup > 1.0


def _task_time(task_ms: float, n_cores: int, *, p_cores: int, e_cores: int,
               e_eff: float, p_core_all_ratio: float) -> float:
    pred = predict_schedule(
        task_ms, p_cores=p_cores, e_cores=e_cores,
        e_core_efficiency=e_eff, max_threads=n_cores,
        p_core_all_ratio=p_core_all_ratio,
    )
    return float(pred.predicted_ms[n_cores])


def _allocations(total: int, k: int) -> list[tuple[int, ...]]:
    """枚举 k 个任务的正整数核数分配，Σ ≤ total（每任务至少 1 核）。"""
    if k == 1:
        return [(n,) for n in range(1, total + 1)]
    out = []
    for n1 in range(1, total - k + 2):
        for rest in _allocations(total - n1, k - 1):
            out.append((n1, *rest))
    return out


def plan_core_split(
    tasks_ms: Sequence[float],
    *,
    p_cores: int,
    e_cores: int,
    e_core_efficiency: float,
    p_core_all_ratio: float = 0.0,
) -> SplitPlan:
    """为 k 个可并行任务规划核数拆分，返回最优分配。

    基线：各自满核（p+e）串行的预测总时长。最优：并发完成时间
    （max 各任务墙钟）最小的分配。
    """
    tasks = [float(t) for t in tasks_ms]
    if not tasks:
        raise ValueError("tasks_ms 不能为空")
    if any(t <= 0 for t in tasks):
        raise ValueError(f"任务时长必须为正：{tasks}")
    total = p_cores + e_cores
    k = len(tasks)
    if k > total:
        raise ValueError(f"任务数 {k} 超过总核数 {total}，无法每任务至少 1 核")

    serial = sum(
        _task_time(t, total, p_cores=p_cores, e_cores=e_cores,
                   e_eff=e_core_efficiency, p_core_all_ratio=p_core_all_ratio)
        for t in tasks
    )

    best: SplitPlan | None = None
    for alloc in _allocations(total, k):
        each = tuple(
            _task_time(t, n, p_cores=p_cores, e_cores=e_cores,
                       e_eff=e_core_efficiency,
                       p_core_all_ratio=p_core_all_ratio)
            # alloc 由 _allocations(total, k=len(tasks)) 产出，长度必然相等；
            # strict=True 把"长度不等"从静默截断变成显式报错。
            for t, n in zip(tasks, alloc, strict=True)
        )
        makespan = max(each)
        if best is None or makespan < best.predicted_makespan_ms:
            best = SplitPlan(
                allocation=tuple(alloc),
                predicted_each_ms=tuple(round(x, 3) for x in each),
                predicted_makespan_ms=round(makespan, 3),
                serial_baseline_ms=round(serial, 3),
                speedup=round(serial / makespan, 3),
            )
    assert best is not None  # k ≤ total 时枚举非空
    return best
