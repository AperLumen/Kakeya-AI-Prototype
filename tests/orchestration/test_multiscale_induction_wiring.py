"""多尺度归纳接入编排器的守门测试。

**背景**：`sticky_kakeya.multiscale_induction`（Prop 1.7 的 ω 压缩循环）此前
**被 import 却从不调用**——`RECURSION` 分支是"无脑把 δ 减半"，没有任何
"是否需要更细尺度"的判断（内部设计文档第 II 篇登记为已知局限）。

本文件钉住四件事：

1. **接入真的生效**：`_recursion_plan` 会调 `multiscale_induction`，且其结论
   影响到 `_try_recurse` 是否递归；
2. **定量边界**（最重要）：归纳的输出**只进 notes（自由文本），不进 `.kpt`
   的定量字段**——`multiscale_induction` 的 docstring 明确警告其自定常数
   （`c_0=0.1`、上限 `ω/2`）使数值不可作定量结论；
3. **停滞即不递归**：归纳判定 ω 压不动时应停止递归，而不是撞 δ 硬下界；
4. **失败可归因**：因归纳停滞而停下时，归因是 `RECURSION_EXHAUSTED`
   ——而非 `BUDGET_EXHAUSTED`（后者会误导人去加 `max_iterations`）。

变异验证（开发时手工做过）：
- 把 `_recursion_plan` 改成恒返回 `should_recurse=True` → 红；
- 把 plan 的 reason 从 notes 里去掉 → 红；
- 不传 `multiscale_induction` 的 delta → 红（形态检查）。
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


def _fan_tubes(n_theta: int = 4, n_phi: int = 4) -> list[Tube]:
    """方向分散的 fan 构型（非 sticky，会走到递归分支）。"""
    tubes = []
    for th in np.linspace(0.1, np.pi - 0.1, n_theta):
        for ph in np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False):
            tubes.append(
                Tube(center=np.zeros(3),
                     direction=Direction.from_spherical(th, ph),
                     length=1.0, delta=0.05)
            )
    return tubes


def _parallel_tubes(n: int = 8) -> list[Tube]:
    return [
        Tube(center=np.array([0.0, y, 0.0]),
             direction=Direction.from_array([0.0, 0.0, 1.0]),
             length=1.0, delta=0.05)
        for y in np.linspace(-0.4, 0.4, n)
    ]


def _orch(tubes: list[Tube], **kw) -> ProofOrchestrator:
    return ProofOrchestrator(tubes, epsilon=0.05, K=2.0,
                             max_iterations=kw.pop("max_iterations", 6),
                             grid_n=16, **kw)


# ---------------------------------------------------------------------------
# 1. 接入生效
# ---------------------------------------------------------------------------


def test_recursion_plan_returns_structural_verdict():
    """``_recursion_plan`` 必须给出结构性判定（而非恒真/恒假）。"""
    orch = _orch(_parallel_tubes(8))
    plan = orch._recursion_plan(orch.initialize())
    assert set(plan) >= {"should_recurse", "reason", "n_steps", "stalled", "termination"}
    assert isinstance(plan["should_recurse"], bool)
    assert plan["reason"], "判定必须给出理由（否则没法诊断）"
    # n_steps > 0 说明 multiscale_induction 真的被调了（不是空壳）
    assert plan["n_steps"] > 0, (
        "归纳循环一步没跑——说明 _recursion_plan 没真正调用 multiscale_induction"
    )


def test_recursion_plan_uses_shared_termination_classifier(monkeypatch):
    """终止判定必须**复用** ``sticky_kakeya.classify_induction_termination``。

    A4 收口点：此前 ``_recursion_plan`` 自己写了一份"尾部压缩量 ≈ 0 / 是否还在
    下降"的启发式，而 L2 核验（``verify_induction_l2``）另有一份——两处各判一次
    会漂移。现在编排器**只调用那一处**，本用例用 monkeypatch 证明它确实被调过。
    """
    from kakeya_ai import proof_orchestrator as po
    from kakeya_ai.sticky_kakeya import InductionTermination

    called: list[int] = []
    real = po.classify_induction_termination

    def spy(steps, **kw):
        called.append(1)
        return real(steps, **kw)

    monkeypatch.setattr(po, "classify_induction_termination", spy)
    orch = _orch(_parallel_tubes(8))
    plan = orch._recursion_plan(orch.initialize())
    assert called, (
        "未复用 classify_induction_termination——终止判定又有了第二份实现"
    )
    assert plan["termination"] in {t.name for t in InductionTermination}


def test_recursion_plan_uses_multiscale_induction(monkeypatch):
    """接入必须是**真调用**（用 monkeypatch 证明它被调过）。"""
    from kakeya_ai import proof_orchestrator as po

    called: list[int] = []
    real = po.multiscale_induction

    def spy(*a, **kw):
        called.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(po, "multiscale_induction", spy)
    orch = _orch(_parallel_tubes(8))
    orch._recursion_plan(orch.initialize())
    assert called, (
        "multiscale_induction 未被调用——这正是本改动要修的缺口"
        "（此前它被 import 却从不被调用）"
    )


def test_recursion_plan_forwards_delta(monkeypatch):
    """必须把当前 δ 透传给归纳（否则归纳在错误的尺度上评估）。"""
    from kakeya_ai import proof_orchestrator as po

    seen: dict = {}
    real = po.multiscale_induction

    def spy(tubes, **kw):
        seen.update(kw)
        return real(tubes, **kw)

    monkeypatch.setattr(po, "multiscale_induction", spy)
    orch = _orch(_parallel_tubes(8))
    state = orch.initialize()
    orch._recursion_plan(state)
    assert seen.get("delta") == pytest.approx(state.delta), (
        f"δ 未透传（收到 {seen.get('delta')}，状态是 {state.delta}）"
    )
    assert "omega_0" in seen, "ω 未透传"


# ---------------------------------------------------------------------------
# 2. 定量边界：归纳输出只进 notes
# ---------------------------------------------------------------------------


def test_plan_reason_goes_to_notes_not_quantitative_fields():
    """归纳的判定依据只进 ``notes``（自由文本），不进定量字段。

    **为什么这条最重要**：``multiscale_induction`` 的 docstring 明确写着
    "不可作 'N 步后 ω ≈ 某值' 这类定量结论"——其压缩量公式的自定常数
    （``c_0=0.1``、上限 ``ω/2``）决定了整条曲线的下降速度。若把 ω 的数值
    写进 ``ProofMetrics`` 或 trace 的定量字段，就会让启发式数字伪装成有依据
    的结论（本项目已因同类问题修过 ``induction_step_g`` 的 docstring）。

    **注意判定依据挂在哪一步**：归纳判定"停滞"时``_try_recurse`` 返回 None，
    该步动作会**降级为 TERMINAL_***（而非 RECURSION）。所以不能只找
    ``action == RECURSION`` 的步——那样在"判定停滞"的构型上会误 skip，
    而"判定停滞"恰恰是最常见的情形。
    """
    orch = _orch(_fan_tubes())
    result = orch.run()

    planned = [s for s in result.steps
               if any("recursion plan" in n for n in s.notes)]
    assert planned, (
        f"没有任何一步带着归纳判定依据（notes 里应有 'recursion plan'）。"
        f"实际动作序列：{[s.action.name for s in result.steps]}"
    )
    # 反向：不该出现新的"定量 ω 字段"。
    # ProofMetrics 是定量结论的载体，它不该因为本次接入而多出 ω 字段。
    import dataclasses

    from kakeya_ai.proof_orchestrator import ProofMetrics

    field_names = {f.name for f in dataclasses.fields(ProofMetrics)}
    assert not any("omega" in f.lower() for f in field_names), (
        f"ProofMetrics 里出现了 ω 相关字段 {field_names}——"
        "归纳的启发式数值不得进入定量结论载体"
    )


# ---------------------------------------------------------------------------
# 3. 停滞即不递归（调度优化）
# ---------------------------------------------------------------------------


def test_stalled_induction_stops_recursion(monkeypatch):
    """归纳判定停滞时，``_try_recurse`` 必须**不**返回新状态。"""
    orch = _orch(_fan_tubes())
    state = orch.initialize()

    # 伪造"停滞"的判定
    monkeypatch.setattr(orch, "_recursion_plan",
                        lambda s: {"should_recurse": False,
                                   "reason": "测试：判定停滞",
                                   "n_steps": 5, "stalled": True})
    new_state, plan = orch._try_recurse(state)
    assert new_state is None, "判定停滞却仍然递归了"
    assert plan["stalled"] is True


def test_should_recurse_true_does_recurse():
    """判定可递归时必须真的递归，且 δ 确实变小。"""
    orch = _orch(_fan_tubes())
    state = orch.initialize()
    # 用真实构型：若不判定停滞则应递归
    plan = orch._recursion_plan(state)
    new_state, plan2 = orch._try_recurse(state)
    if plan["should_recurse"] and state.delta > 1e-4:
        assert new_state is not None, "判定可递归却没有递归"
        assert new_state.delta == pytest.approx(state.delta / 2.0)
    else:
        assert new_state is None, "判定不可递归却递归了"


def test_recursion_avoids_wasted_iterations():
    """**调度优化**：归纳判定停滞时不白跑迭代。

    实测对比（fan(4×4)，``max_iterations=6``）：

    ======  =====================================================  ==================
    版本    动作序列                                                终止归因
    ======  =====================================================  ==================
    旧      NSP → GRAINS → **RECURSION** → NSP → GRAINS → MOVE2      `BUDGET_EXHAUSTED`
    新      NSP → GRAINS → **TERMINAL_FAILURE**                      `RECURSION_EXHAUSTED`
    ======  =====================================================  ==================

    差别不只是"少几步"：旧行为递归后**重走昂贵的 NSP + grains**，最后仍以
    "预算耗尽"结束——而 `BUDGET_EXHAUSTED` 会误导人去加 `max_iterations`
    （真正原因不在预算）。
    """
    orch = _orch(_fan_tubes(), max_iterations=6)
    result = orch.run()
    actions = [s.action.name for s in result.steps]

    assert len(result.steps) <= 6, f"超出预算上限：{len(result.steps)}"
    # 本次接入的核心效果：不再无脑递归。
    assert ProofAction.RECURSION.name not in actions, (
        f"仍在无脑递归（{actions}）——归纳判定未阻止它"
    )
    # 且归因说清真实原因（不是"预算耗尽"）。
    assert result.failure_reason == FailureReason.RECURSION_EXHAUSTED, (
        f"终止归因是 {result.failure_reason}，期望 RECURSION_EXHAUSTED"
        "（BUDGET_EXHAUSTED 会误导人去加 max_iterations）"
    )


# ---------------------------------------------------------------------------
# 4. 健壮性
# ---------------------------------------------------------------------------


def test_induction_failure_does_not_break_search(monkeypatch):
    """归纳抛异常时不得中断证明搜索（观测/决策失败应降级）。"""
    from kakeya_ai import proof_orchestrator as po

    def boom(*a, **kw):
        raise RuntimeError("归纳故意失败")

    monkeypatch.setattr(po, "multiscale_induction", boom)
    orch = _orch(_parallel_tubes(8))
    plan = orch._recursion_plan(orch.initialize())
    # 降级：允许递归（保守），并把异常记进诊断
    assert plan["should_recurse"] is True
    assert any("multiscale_induction" in d for d in orch.diagnostics), (
        f"归纳异常未记入 diagnostics：{orch.diagnostics}"
    )


def test_empty_tubes_plan_is_safe():
    """空管族时不得抛异常（归纳循环对空输入应安全）。"""
    orch = _orch(_parallel_tubes(8))
    from kakeya_ai.proof_orchestrator import ProofState

    empty = ProofState(tubes=[], shading=None, grains=None,
                       delta=0.05, rho=0.5)
    plan = orch._recursion_plan(empty)
    assert plan["should_recurse"] is False
    assert "空" in plan["reason"]
