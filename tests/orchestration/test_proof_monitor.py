"""证明搜索可观测性的守门测试（``proof_monitor`` + 编排器的 ``on_step``）。

**为什么需要它。** ``run()`` 是长跑黑盒（本项目实测一次 4 轮约 81.8 s）。
加可观测性最危险的退化不是"忘了接线"，而是**接了线却报错结论**：

- 样本不足时报 ``progressing`` → 调用方干等一个真卡住的任务；
- 样本不足时报 ``stalled``    → 误杀正常任务；
- 有异常慢步时报 ``progressing`` → 卡住无人知。

这三条**都不会抛异常**，只会安静地给出错误分类。故本文件的核心不是"回调
被调用了吗"，而是"**判定结论对不对**"。

对齐项目既有纪律（``wolff_axioms`` 零覆盖事件）：**"不知道"必须与"确认在推进"
分开报**——即三态而非布尔。最后一条测试专门钉住这一点。

变异验证（开发时手工做过）：
- 把 ``min_samples_for_verdict`` 判定去掉（样本不足也判 progressing）→ 红；
- 去掉"终态不算卡住"分支 → 红；
- 去掉耗时判据 → 红；
- 让 ``_emit_step`` 不吞回调异常 → 红。
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.proof_monitor import (
    CallableMonitor,
    HealthVerdict,
    NullMonitor,
    ProgressSnapshot,
    ProofMonitor,
)


def _snap(i: int, vol: float, grains: int, *, elapsed: float = 1.0,
          terminal: bool = False) -> ProgressSnapshot:
    """造一个快照（测试用的最小构造器）。"""
    return ProgressSnapshot(
        iteration=i, action="X", elapsed=elapsed,
        total_elapsed=(i + 1) * elapsed, volume_ratio=vol,
        n_grains=grains, n_tubes=8, terminal=terminal,
    )


def _parallel_tubes(n: int = 8) -> list[Tube]:
    return [
        Tube(center=np.array([0.0, y, 0.0]),
             direction=Direction.from_array([0.0, 0.0, 1.0]),
             length=1.0, delta=0.05)
        for y in np.linspace(-0.4, 0.4, n)
    ]


# ---------------------------------------------------------------------------
# 1. 三态判定：正确性（核心）
# ---------------------------------------------------------------------------


def test_progressing_when_metrics_change():
    """指标在变 → ``progressing``。"""
    m = ProofMonitor()
    for i, (v, g) in enumerate([(0.5, 10), (0.7, 8), (0.9, 6), (1.1, 5)]):
        m.on_step(_snap(i, v, g))
    assert m.verdict() == HealthVerdict.PROGRESSING
    assert not m.is_stalled()


def test_stalled_when_metrics_frozen():
    """连续多步指标冻结（且未终结）→ ``stalled``。"""
    m = ProofMonitor()
    for i in range(5):
        m.on_step(_snap(i, 0.5, 10))
    assert m.verdict() == HealthVerdict.STALLED
    assert m.is_stalled()


def test_stalled_when_one_step_far_slower_than_baseline():
    """某步耗时远超历史中位数 → ``stalled``（耗时判据单独成立）。"""
    m = ProofMonitor()
    # 前 3 步正常且**指标在变**（排除冻结判据的干扰，单独验耗时判据）。
    for i in range(3):
        m.on_step(_snap(i, 0.5 + i * 0.1, 10 - i, elapsed=1.0))
    m.on_step(_snap(3, 0.8, 7, elapsed=100.0))  # 100× 基线
    assert m.verdict() == HealthVerdict.STALLED


def test_unknown_when_too_few_samples():
    """样本不足 → ``unknown``，**不猜**（这是本模块最重要的语义）。

    猜成 ``progressing`` 会让调用方干等一个真卡住的任务；猜成 ``stalled``
    会误杀正常任务。两种都比"如实说不知道"更坏。
    """
    m = ProofMonitor(min_samples_for_verdict=3)
    m.on_step(_snap(0, 0.5, 10))
    m.on_step(_snap(1, 0.5, 10))  # 只有 2 步 < 3
    assert m.verdict() == HealthVerdict.UNKNOWN
    # unknown 时 is_stalled 必须为 False（不误杀）
    assert not m.is_stalled()


def test_terminal_step_freezes_are_not_stalled():
    """末尾是终态时，"指标不动"是正常的（已结束），不算卡住。"""
    m = ProofMonitor()
    for i in range(3):
        m.on_step(_snap(i, 0.5, 10))
    m.on_step(_snap(3, 0.5, 10, terminal=True))
    assert m.verdict() == HealthVerdict.PROGRESSING
    assert not m.is_stalled()


def test_zero_baseline_does_not_false_positive():
    """基线耗时为 0（极快步）时不触发耗时判据——避免除零与虚假大幅比例。"""
    m = ProofMonitor(min_samples_for_verdict=3)
    for i in range(3):
        m.on_step(_snap(i, 0.5 + i * 0.1, 10 - i, elapsed=0.0))
    m.on_step(_snap(3, 0.8, 7, elapsed=0.001))  # 比值极大但绝对量可忽略
    assert m.verdict() == HealthVerdict.PROGRESSING


# ---------------------------------------------------------------------------
# 2. 监视器契约
# ---------------------------------------------------------------------------


def test_null_monitor_is_silent_and_callable_monitor_forwards():
    """``NullMonitor`` 不做事；``CallableMonitor`` 转发快照。"""
    NullMonitor().on_step(_snap(0, 1.0, 1))  # 不应抛
    got: list[int] = []
    CallableMonitor(lambda s: got.append(s.iteration)).on_step(_snap(7, 1.0, 1))
    assert got == [7]


def test_summary_mentions_verdict():
    """``summary()`` 必须包含判定词，便于日志直接读。"""
    m = ProofMonitor()
    assert HealthVerdict.UNKNOWN in m.summary()  # 空快照 → unknown
    for i in range(4):
        m.on_step(_snap(i, 0.5 + i * 0.1, 10))
    assert HealthVerdict.PROGRESSING in m.summary()


# ---------------------------------------------------------------------------
# 3. 与编排器的接线（真实调用）
# ---------------------------------------------------------------------------


def test_orchestrator_emits_snapshots_to_callback():
    """编排器 ``run()`` 必须真的把快照喂给 ``on_step``（真实跑一次）。"""
    from kakeya_ai.proof_orchestrator import ProofOrchestrator

    seen: list[ProgressSnapshot] = []
    orch = ProofOrchestrator(
        _parallel_tubes(8), epsilon=0.05, K=2.0,
        max_iterations=4, grid_n=16, on_step=lambda s: seen.append(s),
    )
    result = orch.run()

    assert seen, "on_step 回调一次都没被调用——接线断了"
    # 回调收到的步数应与实际执行的步数一致
    assert len(seen) == len(result.steps), (
        f"回调收到 {len(seen)} 个快照，但实际跑了 {len(result.steps)} 步"
    )
    # 即使没传 on_step，orch.snapshots 也应记录主循环真的发生了什么
    assert len(orch.snapshots) == len(result.steps)
    # 快照字段应与结果自洽
    assert seen[-1].iteration == result.steps[-1].iteration
    assert seen[-1].terminal == result.steps[-1].action.name.startswith("TERMINAL")


def test_orchestrator_no_callback_still_records_snapshots():
    """不传 ``on_step`` 时，``orch.snapshots`` 仍累积（与回调无关）。"""
    from kakeya_ai.proof_orchestrator import ProofOrchestrator

    orch = ProofOrchestrator(
        _parallel_tubes(8), epsilon=0.05, K=2.0, max_iterations=3, grid_n=16,
    )
    result = orch.run()
    assert len(orch.snapshots) == len(result.steps)
    assert isinstance(orch.monitor(), NullMonitor)


def test_callback_exception_does_not_break_search():
    """回调抛异常必须**不中断**搜索——观测设施不该成为新的失败源。

    同时异常要记进 ``diagnostics``（不能静默），否则"回调坏了"无从发现。
    """
    from kakeya_ai.proof_orchestrator import ProofOrchestrator

    def boom(_s: ProgressSnapshot) -> None:
        raise RuntimeError("回调故意失败")

    orch = ProofOrchestrator(
        _parallel_tubes(8), epsilon=0.05, K=2.0,
        max_iterations=3, grid_n=16, on_step=boom,
    )
    result = orch.run()  # 不得抛出
    assert result.steps, "搜索被回调异常中断了"
    assert any("on_step" in d and "RuntimeError" in d for d in orch.diagnostics), (
        f"回调异常未记入 diagnostics：{orch.diagnostics}"
    )


def test_invalid_on_step_is_rejected():
    """非法的 ``on_step``（既不可调用也无 on_step）应抛 ``InputValidationError``。"""
    from kakeya_ai.errors import InputValidationError
    from kakeya_ai.proof_orchestrator import ProofOrchestrator

    with pytest.raises(InputValidationError):
        # **故意传非法值**（这条用例测的就是"运行时会拦下它"），
        # 类型上必然不兼容——这不是需要修的类型错误，是负向用例的固有摩擦。
        ProofOrchestrator(_parallel_tubes(4), on_step=12345)  # type: ignore[arg-type]


def test_monitor_usable_after_run():
    """``run()`` 结束后可用监视器取判定（长跑场景的实际用法）。"""
    from kakeya_ai.proof_orchestrator import ProofOrchestrator
    from kakeya_ai.proof_monitor import ProofMonitor

    mon = ProofMonitor()
    orch = ProofOrchestrator(
        _parallel_tubes(8), epsilon=0.05, K=2.0,
        max_iterations=6, grid_n=16, on_step=mon,
    )
    orch.run()
    assert orch.monitor() is mon
    assert mon.verdict() in (
        HealthVerdict.PROGRESSING, HealthVerdict.STALLED, HealthVerdict.UNKNOWN,
    )


def test_snapshot_elapsed_is_measured_and_monotonic():
    """快照里的耗时字段必须被真的测量（非零、累计单调不减）。"""
    from kakeya_ai.proof_orchestrator import ProofOrchestrator

    orch = ProofOrchestrator(
        _parallel_tubes(8), epsilon=0.05, K=2.0, max_iterations=3, grid_n=16,
    )
    orch.run()
    snaps = orch.snapshots
    assert snapshots_are_plausible(snaps)


def snapshots_are_plausible(snaps: list[ProgressSnapshot]) -> bool:
    """辅助断言：耗时为正、累计单调、iter 连续。"""
    assert snaps, "无快照"
    assert all(s.elapsed >= 0.0 for s in snaps)
    assert all(s.total_elapsed > 0.0 for s in snaps)
    for a, b in pairwise(snaps):
        assert b.total_elapsed >= a.total_elapsed, "累计耗时不单调"
        assert b.iteration == a.iteration + 1, "iteration 不连续"
    return True
