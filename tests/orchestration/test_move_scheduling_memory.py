"""Move 调度记忆（状态机化）的守门测试。

**背景**：`_select_move` 此前是**无状态**的——每轮重新算 `m1_ok/m2_ok/m3_ok`，
不记得"刚才试过哪个、效果如何"。实测后果（12 根随机构型，`max_iterations=12`）：

    step 2..11: MOVE2_REPLACE_SQUARE_GRAINS   ← 连续 10 次
      rho=0.4112  local_error=0.2109  n_grains=94  vol=1.3939   ← 逐位不变

即**纯空转**：Move 对当前状态完全无效，调度器却一直重试到预算耗尽；而归因是
``BUDGET_EXHAUSTED``——**误导人去加 ``max_iterations``**（加预算只会多空转）。

本文件钉住四条：

1. **记忆生效**：同状态下试过且无效的 Move 不再被选中；
2. **不误伤**：状态变了之后，"曾无效"的结论**不适用**（Move 可能在新状态有效）
   ——记忆是"有条件的"，不是永久拉黑；
3. **归因正确**：因空转而停时必须报 ``NO_PROGRESS``，**不是** ``BUDGET_EXHAUSTED``；
4. **可复现**：记忆不得引入非确定性（同输入同结论）。

变异验证（开发时手工做过）：把 ``_is_known_futile`` 改成恒返回 False → 红；
把 ``NO_PROGRESS`` 归因去掉 → 红。
"""

from __future__ import annotations

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.proof_orchestrator import (
    FailureReason,
    ProofAction,
    ProofOrchestrator,
)


def _random_tubes(seed: int, n: int, delta: float = 0.1) -> list[Tube]:
    """随机构型（记忆里实测能触发 Move 空转的那一类）。"""
    rng = np.random.default_rng(seed)
    tubes = []
    for _ in range(n):
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d)
        tubes.append(Tube(center=rng.uniform(-0.3, 0.3, 3),
                          direction=Direction.from_array(d.tolist()),
                          length=1.0, delta=delta))
    return tubes


def _orch(tubes: list[Tube], max_iterations: int = 12) -> ProofOrchestrator:
    """造编排器，并**禁用 surrogate**。

    必须禁用：编排器会 lazy 探测 `data/surrogate_scorer_weights.json` 并自动
    加载，加载成功时 surrogate 在启发式之前接管 Move 选择——那样测的就是
    surrogate 路径而非本文件的启发式记忆（开发时踩到过）。
    """
    orch = ProofOrchestrator(tubes, epsilon=0.05, K=2.0,
                             max_iterations=max_iterations, grid_n=16)
    orch.surrogate = None
    return orch


def _move_actions(result) -> list[str]:
    return [s.action.name for s in result.steps if s.action.name.startswith("MOVE")]


# ---------------------------------------------------------------------------
# 1. 记忆生效：不再重复无效 Move
# ---------------------------------------------------------------------------


def test_stalling_move_is_not_retried_many_times():
    """**核心回归**：无效的 Move 不得被连续重试到预算耗尽。

    这是本改动要修的原始症状：旧实现里 MOVE2 连续 10 次、指标逐位不变。
    断言"任何单个 Move 的连续重复不超过 3 次"——留一点余量是因为"试一次发现
    有效、再试一次"是合理的（第一次会改变指标，第二次才可能发现无效）。
    """
    from itertools import groupby

    result = _orch(_random_tubes(7, 12)).run()
    actions = [s.action.name for s in result.steps]
    runs = [(k, len(list(g))) for k, g in groupby(actions)]
    worst = max((n for _, n in runs), default=0)
    assert worst <= 3, (
        f"出现 {worst} 次连续重复动作：{runs}。"
        "说明调度记忆没生效（旧行为里 MOVE2 会重复 10 次）"
    )


def test_notes_explain_the_skip():
    """跳过无效 Move 时必须在 notes 里**说清理由**（否则无法诊断）。"""
    result = _orch(_random_tubes(7, 12)).run()
    all_notes = " ".join(n for s in result.steps for n in s.notes)
    assert "skipped" in all_notes or "未改变指标" in all_notes, (
        f"没有任何一步说明'为什么跳过/为什么无效'：{all_notes[:300]}"
    )


def test_futility_recorded_on_attempt():
    """尝试历史必须被真的填写（不是恒空）。"""
    orch = _orch(_random_tubes(7, 12))
    orch.run()
    assert orch._move_attempts, (
        "_move_attempts 为空——说明 _record_attempt 从未被调用，记忆是空壳"
    )
    # 记录值形如 (指纹, 是否有变化)
    for name, rec in orch._move_attempts.items():
        assert isinstance(rec, tuple) and len(rec) == 2
        assert isinstance(rec[1], bool), f"{name} 的记录第二项应是 bool"
        assert name.startswith("MOVE")


# ---------------------------------------------------------------------------
# 2. 不误伤：状态变了则记忆失效
# ---------------------------------------------------------------------------


def test_futility_is_conditional_on_state():
    """"曾无效"只在**相同状态**下成立；状态一变就应重新允许尝试。

    这是"记忆"与"永久拉黑"的分界。判据：若记忆无条件是"试过就不许再试"，
    那么在一个 Move 先无效、后（因别的原因）状态改变的场景里，它会被错误
    跳过——本项目已有多起"把'未测'当成'不行'"的先例，此处同理。
    """
    orch = _orch(_random_tubes(7, 12))

    # 造两个不同的指标指纹
    #
    # **有意的鸭子类型桩**：``_state_fingerprint`` / ``_record_attempt`` 只读
    # 这五个属性，不需要真的 ``ProofMetrics``。参数声明是具体类型，故不兼容
    # —— ignore 如实记录"这是刻意的轻量替身"，不是掩盖缺陷。
    class _M:
        def __init__(self, rho):
            self.rho = rho
            self.local_error = 1.0
            self.volume_ratio = 1.0
            self.n_grains = 10
            self.delta = 0.1

    fp_a = orch._state_fingerprint(_M(0.4))  # type: ignore[arg-type]
    fp_b = orch._state_fingerprint(_M(0.5))  # type: ignore[arg-type]
    assert fp_a != fp_b

    # 在 fp_a 下记录"无效"
    orch._move_attempts[ProofAction.MOVE2_REPLACE_SQUARE_GRAINS.name] = (fp_a, False)
    assert orch._is_known_futile(ProofAction.MOVE2_REPLACE_SQUARE_GRAINS, fp_a)
    assert not orch._is_known_futile(
        ProofAction.MOVE2_REPLACE_SQUARE_GRAINS, fp_b
    ), "状态已变，旧的'无效'结论不应继续生效（那是永久拉黑，不是记忆）"


def test_effective_move_is_not_marked_futile():
    """带来变化的尝试不得被记为无效（否则会误跳过有效的 Move）。"""
    orch = _orch(_random_tubes(7, 12))

    # 同上：有意的鸭子类型桩（只提供指纹判据真正读的五个属性）。
    class _M:
        def __init__(self, rho):
            self.rho = rho
            self.local_error = 1.0
            self.volume_ratio = 1.0
            self.n_grains = 10
            self.delta = 0.1

    before = orch._state_fingerprint(_M(0.4))  # type: ignore[arg-type]
    after = _M(0.9)  # 变了
    orch._record_attempt(
        ProofAction.MOVE1_LONGER_GRAINS, before, after  # type: ignore[arg-type]
    )
    assert not orch._is_known_futile(ProofAction.MOVE1_LONGER_GRAINS, before), (
        "该 Move 明显改变了指标（rho 0.4→0.9），却被记为无效"
    )


# ---------------------------------------------------------------------------
# 3. 归因正确
# ---------------------------------------------------------------------------


def test_no_progress_reason_is_registered():
    """``NO_PROGRESS`` 必须在枚举与文本映射里都注册（否则打印会崩）。"""
    from kakeya_ai.proof_orchestrator import FAILURE_REASON_TEXT

    assert FailureReason.NO_PROGRESS in FAILURE_REASON_TEXT
    assert FAILURE_REASON_TEXT[FailureReason.NO_PROGRESS]


def test_all_candidates_futile_reports_no_progress():
    """所有候选 Move 都试过且无效时：报 ``NO_PROGRESS``（不是预算耗尽）。

    **必须显式禁用 surrogate**：编排器会 lazy 探测 `data/surrogate_scorer_weights.json`
    并自动加载，若加载成功则 surrogate 会在启发式之前接管选择——那样测的是
    surrogate 路径，而不是本测试要验证的**启发式记忆**。
    （开发时踩到：不设 `surrogate=None` 时 `_select_move` 直接返回 surrogate
    的结论，`notes` 里是 "move selected by surrogate"。）
    """
    orch = _orch(_random_tubes(7, 12))
    orch.surrogate = None  # 强制走启发式路径
    orch._last_failure_reason = None

    # 用真实状态：state 必须已带 grains，否则 _select_move 会提前返回
    # GRAINS_DECOMPOSITION（那测的就不是本逻辑了）。
    state = orch.initialize()
    metrics = orch._compute_metrics(state)
    state = orch._apply_move(state, ProofAction.GRAINS_DECOMPOSITION) \
        if state.grains is None else state
    assert state.grains is not None and state.grains.prisms, (
        "未能造出带 grains 的状态——本测试的前提不成立"
    )

    # 让三个条件都为"不满足"，从而三个都是候选
    import kakeya_ai.proof_orchestrator as po

    orig = (po.check_move1_condition, po.check_move2_condition,
            po.check_move3_condition)
    po.check_move1_condition = lambda *a, **k: False
    po.check_move2_condition = lambda *a, **k: False
    po.check_move3_condition = lambda *a, **k: False
    try:
        # MOVE3 需要 local_error > 1.0 才是候选
        metrics.local_error = 2.0
        fp2 = orch._state_fingerprint(metrics)
        for a in (ProofAction.MOVE1_LONGER_GRAINS,
                  ProofAction.MOVE2_REPLACE_SQUARE_GRAINS,
                  ProofAction.MOVE3_WIDER_GRAINS):
            orch._move_attempts[a.name] = (fp2, False)
        action = orch._select_move(state, metrics)
    finally:
        (po.check_move1_condition, po.check_move2_condition,
         po.check_move3_condition) = orig

    assert action == ProofAction.LOWER_BOUND_VERIFY, (
        f"全部候选无效时应去验下界，实际返回 {action}；"
        f"notes={orch._selection_notes}"
    )
    assert orch._last_failure_reason == FailureReason.NO_PROGRESS, (
        f"归因应是 NO_PROGRESS（空转），实际 {orch._last_failure_reason}"
        "——BUDGET_EXHAUSTED 会误导人去加 max_iterations"
    )


def test_no_progress_is_not_budget_exhausted():
    """两个归因必须可区分（语义不同：加预算有用 vs 没用）。"""
    assert FailureReason.NO_PROGRESS is not FailureReason.BUDGET_EXHAUSTED
    from kakeya_ai.proof_orchestrator import FAILURE_REASON_TEXT

    assert (FAILURE_REASON_TEXT[FailureReason.NO_PROGRESS]
            != FAILURE_REASON_TEXT[FailureReason.BUDGET_EXHAUSTED])


# ---------------------------------------------------------------------------
# 4. 可复现
# ---------------------------------------------------------------------------


def test_memory_does_not_break_reproducibility():
    """引入记忆后，同一输入仍必须给出**逐位相同**的动作序列。"""
    a = _orch(_random_tubes(7, 12)).run()
    b = _orch(_random_tubes(7, 12)).run()
    assert [s.action.name for s in a.steps] == [s.action.name for s in b.steps]
    assert [s.after.volume_ratio for s in a.steps] == [
        s.after.volume_ratio for s in b.steps
    ], "记忆引入了非确定性（同一输入两次运行的指标不同）"


def test_memory_is_per_orchestrator_instance():
    """记忆必须**实例级**，不跨实例共享（否则一次运行会污染下一次）。"""
    o1 = _orch(_random_tubes(7, 12))
    o1.run()
    o2 = _orch(_random_tubes(7, 12))
    assert not o2._move_attempts, (
        "_move_attempts 在新实例上非空——记忆被跨实例共享了"
    )
