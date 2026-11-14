"""证明搜索的可观测性：进度快照、心跳与"卡住"判定。

**为什么需要它**：:class:`~kakeya_ai.proof_orchestrator.ProofOrchestrator` 的
``run()`` 是长跑组件（本项目实测完整跑一次 4 轮迭代约 **81.8 s**），但它此前
是个**黑盒**：调用方拿到 ``ProofResult`` 之前，对"跑到第几步、每步花了多久、
是不是卡住了"一无所知。失败时只看到"没出结果"，无法区分：

- 在**正常推进**（第 3/10 步，每步 2 s）——等着就行；
- **卡在某一步**（第 3 步已经跑了 300 s）——该中断或调参；
- **在空转**（连续多步指标不变）——加 ``max_iterations`` 也救不了。

**与既有纪律的关系**：本项目反复强调"**未测**与**测了为否**必须可区分"
（``wolff_axioms`` 的零覆盖事件、``moves`` 的哨兵事件）。可观测性同理——
"不知道是否卡住"与"确认在推进"必须报不同结论。因此本模块的核心 API 返回
**三态**（见 :attr:`HealthVerdict.verdict`）：``"progressing"`` /
``"stalled"`` / ``"unknown"``——**样本不足以判定时如实报 unknown，不猜**。

设计取舍
--------

1. **不反向依赖编排器**：本模块**不 import** ``proof_orchestrator``，只吃
   :class:`ProgressSnapshot` 这类结构化数据。否则会形成循环导入，也会把
   观测逻辑与主循环耦合。快照由编排器在每步后构造。
2. **回调只做通知，不影响主循环**：``on_step`` 抛出的异常被编排器捕获并
   记入诊断，**不中断证明搜索**——观测设施不该成为新的失败源。
3. **"卡住"用双重判据**：既看**耗时**（某步远超历史中位数），也看**进展**
   （连续多步指标无变化）。只看耗时会在"慢机器上正常推进"时误报；只看进展
   会在"第一步就慢"时漏报（还没有基线可比）。
4. **判据可复现**：阈值都在构造函数里显式给出，不读环境变量、不看墙钟——
   同一份快照序列在任何机器上得出同一结论。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from itertools import pairwise
from typing import Callable, List, Optional, Protocol

__all__ = [
    "ProgressSnapshot",
    "HealthVerdict",
    "ProofMonitor",
    "NullMonitor",
    "CallableMonitor",
    "on_step_protocol",
]


@dataclass(frozen=True)
class ProgressSnapshot:
    """单步的结构化快照——观测层与编排器之间的**唯一契约**。

    刻意用普通字段而非引用 ``ProofStep``：观测模块不依赖编排器，
    编排器也不把内部对象交给外部（避免外部误改）。

    字段里的 ``volume_ratio`` 与 ``n_grains`` 是"进展"的判据来源：
    前者是结论性指标（``>= 1.0`` 即成功），后者反映合并是否还在发生。
    """

    #: 第几步（0-based，与 ``ProofStep.iteration`` 一致）。
    iteration: int
    #: 动作名（``ProofAction`` 的 ``.name``，如 ``"MOVE1_LONGER_GRAINS"``）。
    action: str
    #: 本步耗时（秒）。用 ``time.perf_counter`` 测量。
    elapsed: float
    #: 累计耗时（秒），含本步。
    total_elapsed: float
    #: 本步之后的 ``volume_ratio``（判定成功的量）。
    volume_ratio: float
    #: 本步之后的 grain 数（合并进展的迹象）。
    n_grains: int
    #: 本步之后的管数。
    n_tubes: int
    #: 本步是否终结（``TERMINAL_SUCCESS`` / ``TERMINAL_FAILURE``）。
    terminal: bool
    #: 备注（自由文本，给人看）。
    notes: tuple[str, ...] = ()


class HealthVerdict:
    """对一段快照序列的**三态**健康判定。

    分为三种字符串取值（不是布尔！）：见 :attr:`verdict`。这里用类而非
    ``Enum``，是为了让取值可以直接出现在 JSON/日志里而不必再转一次。
    """

    #: 在推进：耗时有合理基线且指标在变化。**可以继续等**。
    PROGRESSING = "progressing"
    #: 疑似卡住：耗时远超基线，或连续多步无任何进展。**建议中断或调参**。
    STALLED = "stalled"
    #: **无法判定**：样本不足（还没跑够步数建立基线）。如实报"不知道"，
    #: 不要猜——猜成 progressing 会让人干等，猜成 stalled 会误杀正常任务。
    UNKNOWN = "unknown"


# ``on_step`` 回调的签名。协议不要求实现者继承任何东西（结构化子类型）。
class on_step_protocol(Protocol):
    """``on_step`` 回调需要满足的最小接口（结构化类型）。"""

    def __call__(self, snapshot: ProgressSnapshot) -> None:  # pragma: no cover
        ...


class NullMonitor:
    """什么都不做的监视器（默认）。

    用**空对象**而不是 ``None``：主循环里就不必每处写 ``if monitor is not None``，
    少一个分支就少一处忘记加分支的机会。
    """

    def on_step(self, snapshot: ProgressSnapshot) -> None:  # pragma: no cover
        return None


class CallableMonitor:
    """把裸函数适配成监视器。"""

    def __init__(self, fn: Callable[[ProgressSnapshot], None]) -> None:
        self._fn = fn

    def on_step(self, snapshot: ProgressSnapshot) -> None:
        self._fn(snapshot)


class ProofMonitor:
    """收集进度快照，并判定"是否卡住"。

    Parameters
    ----------
    min_samples_for_verdict : int
        给出 ``progressing`` / ``stalled`` 判定所需的最少步数。少于它就报
        ``unknown``。**默认 3**：两步不足以区分"正常波动"与"真卡住"。
    slow_factor : float
        某步耗时超过"历史步耗时中位数"的多少倍即视为异常慢。默认 5.0。
    stall_window : int
        连续多少步"指标无进展"即判 ``stalled``。默认 3。
    log : callable or None
        可选日志函数（收一个字符串）。给 ``None`` 则不输出。

    Notes
    -----
    **第一个快照不参与耗时判定**：它包含 ``initialize()`` 的开销，与后续
    步不同质（实测首步常显著偏慢），拿它当基线会系统性低估"慢"的阈值。
    这一点在 :meth:`verdict` 里显式处理。
    """

    def __init__(
        self,
        *,
        min_samples_for_verdict: int = 3,
        slow_factor: float = 5.0,
        stall_window: int = 3,
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.min_samples_for_verdict = int(min_samples_for_verdict)
        self.slow_factor = float(slow_factor)
        self.stall_window = int(stall_window)
        self._log = log
        self.snapshots: List[ProgressSnapshot] = []

    # -- 收集 ---------------------------------------------------------------

    def on_step(self, snapshot: ProgressSnapshot) -> None:
        """收到一个快照。这是 ``on_step`` 回调协议的实现。"""
        self.snapshots.append(snapshot)
        if self._log is not None:
            self._log(
                f"[proof] step {snapshot.iteration} {snapshot.action} "
                f"{snapshot.elapsed:.2f}s (total {snapshot.total_elapsed:.2f}s) "
                f"vol_ratio={snapshot.volume_ratio:.4g} n_grains={snapshot.n_grains}"
            )

    # -- 判定 ---------------------------------------------------------------

    def verdict(self) -> str:
        """当前的三态判定（取值见 :class:`HealthVerdict`）。"""
        n = len(self.snapshots)
        if n < self.min_samples_for_verdict:
            return HealthVerdict.UNKNOWN

        # 1) 进展判据：连续 stall_window 步 volume_ratio 与 n_grains 都不变。
        #    用"两个指标都不动"而非只看一个：volume_ratio 在同一 grain 数下
        #    也可能因重算而微动，反之亦然；两者同时冻结才是真停滞。
        if n >= self.stall_window + 1:
            tail = self.snapshots[-(self.stall_window + 1):]
            frozen = all(
                a.volume_ratio == b.volume_ratio and a.n_grains == b.n_grains
                for a, b in pairwise(tail)
            )
            # 若末尾一步是终态，那"不动"是正常的（已结束），不算卡住。
            if frozen and not tail[-1].terminal:
                return HealthVerdict.STALLED

        # 2) 耗时判据：某步耗时 > 历史中位数 × slow_factor。
        #    基线取**除首步外**的历史步（首步含 initialize 开销，不同质）。
        history = [s.elapsed for s in self.snapshots[:-1]]
        latest = self.snapshots[-1].elapsed
        # 至少要有 2 个历史样本才谈得上"中位数基线"；否则这条判据不生效
        # （由 min_samples_for_verdict 保证样本总量，但历史长度可能仍不足）。
        if len(history) >= 2:
            baseline = statistics.median(history)
            # baseline 为 0（极快步）时不判——除以 0 无意义，且 0.0001→0.001
            # 这种绝对差极小却比值极大，判"卡住"是假阳性。
            if baseline > 0 and latest > baseline * self.slow_factor:
                return HealthVerdict.STALLED

        return HealthVerdict.PROGRESSING

    def is_stalled(self) -> bool:
        """是否**确认**卡住。``unknown`` 时返回 ``False``（不误杀）。"""
        return self.verdict() == HealthVerdict.STALLED

    def summary(self) -> str:
        """一行人类可读摘要（含判定与依据）。"""
        v = self.verdict()
        if not self.snapshots:
            return f"monitor: {v}（尚无快照）"
        last = self.snapshots[-1]
        return (
            f"monitor: {v} | steps={len(self.snapshots)} "
            f"last={last.action} total={last.total_elapsed:.2f}s "
            f"vol_ratio={last.volume_ratio:.4g}"
        )
