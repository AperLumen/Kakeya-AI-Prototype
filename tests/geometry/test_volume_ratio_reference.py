"""`volume_ratio` 的**参照系**守门（2026-09-30，第 5 步诊断后新增）。

**背景**：`volume_ratio = union_vol / lower_bound` 是终局判据
（`>= 1.0` ⇒ `TERMINAL_SUCCESS`）。修复前它的**分子与分母都可以被前置步骤
改变**，因而在两个方向上**虚高**：

| 前置步骤 | 管数变化 | 后果 | 实测 |
| --- | --- | --- | --- |
| `BROADNESS_REDUCTION` | 9 → 2 | 分母骤降（`lower_bound` 与管数成正比） | 0.343 → **1.545** |
| `NSP_FACTORIZATION` | 2 → 100 | 分子暴增（50 份刚体副本） | → **17.2** |

两者都是**换了判据**：论文 Theorem 1.2 断言的是**原输入** T 的体积下界，
而不是"被裁过/被复制过的管集"的。

**本文件钉住的不变量**：`_compute_metrics()` 的 `union_vol` 与 `lower_bound`
必须取自**同一族管**（`self._reference_tubes`，即本次运行的输入）。

**为什么必须测这个**（而不是只靠人工核对）：这个缺陷的表现是"数值变大、
看起来更像成功"——**没有任何异常、没有任何日志**，只有把两次运行的数并排
才看得出来。属于"静默给出错误结论"那一类，正是本项目反复强调要机器守的。
"""

from __future__ import annotations

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.proof_orchestrator import ProofOrchestrator


def _fan(k: int, delta: float = 0.1) -> list[Tube]:
    """`k×k` 扇形：方向集中 ⇒ 会触发 broadness 剥离（本组的前提）。"""
    return [
        Tube(
            center=np.zeros(3),
            direction=Direction.from_spherical(np.pi / 2 * 0.3 + 0.01 * i, 2 * np.pi * j / k),
            length=1.0,
            delta=delta,
        )
        for i in range(k)
        for j in range(k)
    ]


class TestReferenceTubeFamily:
    """参照系必须是**原输入**，且不随状态演化而变。"""

    def test_reference_tubes_defaults_to_input(self):
        tubes = _fan(3)
        orch = ProofOrchestrator(
            tubes, grid_n=8, surrogate_weights_path="/nonexistent/w.json"
        )
        assert len(orch._reference_tubes) == len(tubes), (
            "参照系默认应等于本次运行的输入"
        )

    def test_ratio_is_fully_independent_of_current_tubes(self):
        """**核心不变量**：`volume_ratio` 与当前管集**完全无关**。

        做法：构造两个"当前状态"（管集不同、参照系相同），断言比值**相同**。

        **这是比"分母不变"更强的条件**（2026-09-30 二次修正后）：
        第一版只固定了分母，于是 NSP 一扩管（16 → 400 根）分子就暴增
        ⇒ 比值从 1.60 变成 **32.7**——仍然是"换了判据"，只是换在分子上。
        实测该构型随后两个 Move **没改任何指标**却报 `TERMINAL_SUCCESS`。

        故现在分子分母**都**取参照系；本测试钉住这个更强的性质。
        """
        tubes = _fan(3)
        orch = ProofOrchestrator(
            tubes, grid_n=8, surrogate_weights_path="/nonexistent/w.json"
        )
        base = orch.initialize()
        from kakeya_ai.proof_orchestrator import ProofState

        full = ProofState(tubes=base.tubes, shading=base.shading, grains=None,
                          delta=base.delta, rho=base.rho)
        shrunk = ProofState(tubes=base.tubes[:2], shading=None, grains=None,
                            delta=base.delta, rho=base.rho)

        m_full = orch._compute_metrics(full)
        m_shrunk = orch._compute_metrics(shrunk)

        assert m_full.volume_ratio == pytest.approx(m_shrunk.volume_ratio), (
            f"比值随当前管集变了（{m_full.volume_ratio:.6f} vs "
            f"{m_shrunk.volume_ratio:.6f}）——分子与分母都必须取参照系"
        )

    def test_shrinking_current_set_does_not_inflate_ratio(self):
        """**回归钉**：直接把"剥离后"的比值与"未剥离"比，不得虚高。

        这是修复前**实测到的那次跳变**（0.343 → 1.545）的可执行版本。
        """
        tubes = _fan(3)
        orch = ProofOrchestrator(
            tubes, grid_n=8, surrogate_weights_path="/nonexistent/w.json"
        )
        base = orch.initialize()
        from kakeya_ai.proof_orchestrator import ProofState

        m_before = orch._compute_metrics(
            ProofState(tubes=base.tubes, shading=base.shading, grains=None,
                       delta=base.delta, rho=base.rho)
        )
        m_after = orch._compute_metrics(
            ProofState(tubes=base.tubes[:2], shading=None, grains=None,
                       delta=base.delta, rho=base.rho)
        )
        assert m_after.volume_ratio <= m_before.volume_ratio + 1e-9, (
            f"管集变小后比值反而变大了（{m_before.volume_ratio:.4f} → "
            f"{m_after.volume_ratio:.4f}）——这正是修复前的虚高"
        )

    def test_run_ratio_is_not_driven_above_one_by_preprocessing(self):
        """端到端：前置步骤**不得**单独把比值推过 1.0。

        用真实 `run()`：第 0 步是 broadness 剥离（前提），剥离后比值不该
        突变到 ≥ 1.0。修复前实测 0.343 → 1.545（**一步就越过成功线**）。
        """
        result = ProofOrchestrator(
            _fan(3), max_iterations=6, grid_n=8,
            surrogate_weights_path="/nonexistent/w.json",
        ).run()

        assert result.steps, "应产生步"
        first = result.steps[0]
        if first.action.name != "BROADNESS_REDUCTION":
            pytest.skip(f"前提未满足：第 0 步是 {first.action.name}，未触发剥离")

        ratio_after = first.after.volume_ratio
        assert ratio_after < 1.0, (
            f"仅靠 broadness 剥离就把 volume_ratio 推到 {ratio_after:.4f} ≥ 1.0 "
            "——那会让 TERMINAL_SUCCESS 在'没做任何 Move'时达成，正是修复前的缺陷"
        )


class TestNspDoesNotInflateRatio:
    """NSP 扩管**不得**抬高判据比值（2026-09-30 实测踩到的第二个来源）。

    修分母后仍不够：NSP 把管从 16 扩到 400，**分子**（并集体积）随之暴增，
    比值从 1.60 变成 **32.7**，随后两个 Move 没改任何指标却报
    `TERMINAL_SUCCESS`。

    ⇒ 分子分母**都**必须取参照系。本类用端到端跑钉住这一点。
    """

    def test_nsp_step_does_not_change_ratio(self):
        """NSP 那一步的 `volume_ratio` 不得变化（判据与管数无关）。"""
        import numpy as np

        from kakeya_ai.primitives import Direction, Tube

        # 该构型实测会走 NSP（管数 16 < δ⁻²/2），且会走到 Move。
        tubes = [
            Tube(center=np.zeros(3),
                 direction=Direction.from_spherical(np.pi / 2 + 0.05 * i, 2 * np.pi * i / 16),
                 length=1.0, delta=0.05)
            for i in range(16)
        ]
        result = ProofOrchestrator(
            tubes, max_iterations=8, grid_n=8,
            surrogate_weights_path="/nonexistent/w.json",
        ).run()

        nsp_steps = [s for s in result.steps if s.action.name == "NSP_FACTORIZATION"]
        assert nsp_steps, (
            "前提失效：该构型应触发 NSP（若不再触发，请换构型，不要删断言）"
        )
        for s in nsp_steps:
            assert s.after.n_tubes != s.before.n_tubes, (
                "NSP 步应改变管数（否则它没干活）"
            )
            assert s.after.volume_ratio == pytest.approx(s.before.volume_ratio), (
                f"NSP 把管数从 {s.before.n_tubes} 改到 {s.after.n_tubes}，"
                f"比值随之从 {s.before.volume_ratio:.4f} 变到 "
                f"{s.after.volume_ratio:.4f} —— 判据不该随管数变"
            )
