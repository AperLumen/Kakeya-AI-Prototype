"""主证明编排器骨架的测试。"""

from itertools import pairwise

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Shading, Tube
from kakeya_ai.proof_orchestrator import (
    FAILURE_REASON_TEXT,
    FailureReason,
    ProofAction,
    ProofOrchestrator,
    ProofResult,
    ProofState,
    run_proof_skeleton,
)


def make_fan(n_theta=4, n_phi=4, delta=0.1):
    tubes = []
    for theta in np.linspace(0.2, np.pi - 0.2, n_theta):
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False):
            d = Direction.from_spherical(theta, phi)
            tubes.append(Tube(center=np.zeros(3), direction=d, length=1.0, delta=delta))
    return tubes


class TestProofOrchestratorBasics:
    def test_can_initialize(self):
        tubes = make_fan(3, 3)
        orch = ProofOrchestrator(tubes, max_iterations=2, grid_n=16)
        assert orch.tubes == tubes
        assert orch.max_iterations == 2

    def test_run_returns_result(self):
        tubes = make_fan(3, 3, delta=0.1)
        result = run_proof_skeleton(tubes, max_iterations=2, grid_n=16)
        assert result is not None
        assert len(result.steps) >= 1
        assert result.final_metrics.n_tubes == len(tubes)

    def test_empty_tubes_returns_failure(self):
        result = run_proof_skeleton([], max_iterations=2)
        assert not result.success
        assert result.failure_reason is FailureReason.EMPTY_INPUT

    def test_initialize(self):
        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)
        state = orch.initialize()
        assert state.tubes == tubes
        assert state.delta == 0.1
        assert state.grains is None

    def test_compute_metrics_shape(self):
        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)
        state = ProofState(
            tubes=tubes,
            shading=None,
            grains=None,
            delta=0.1,
            rho=0.1,
        )
        metrics = orch._compute_metrics(state)
        assert metrics.n_tubes == len(tubes)
        assert metrics.delta == 0.1
        assert metrics.lower_bound >= 0

    def test_build_grains(self):
        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)
        state = ProofState(
            tubes=tubes,
            shading=None,
            grains=None,
            delta=0.1,
            rho=0.1,
        )
        grains = orch._build_grains(state)
        assert grains.count > 0

    def test_select_move_returns_valid_action(self):
        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)
        state = ProofState(
            tubes=tubes,
            shading=None,
            grains=None,
            delta=0.1,
            rho=0.1,
        )
        grains = orch._build_grains(state)
        state = ProofState(
            tubes=tubes,
            shading=None,
            grains=grains,
            delta=0.1,
            rho=0.1,
        )
        metrics = orch._compute_metrics(state)
        action = orch._select_move(state, metrics)
        assert action in ProofAction

    def test_apply_move_returns_state(self):
        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)
        state = ProofState(
            tubes=tubes,
            shading=None,
            grains=None,
            delta=0.1,
            rho=0.1,
        )
        grains = orch._build_grains(state)
        state = ProofState(
            tubes=tubes,
            shading=None,
            grains=grains,
            delta=0.1,
            rho=0.1,
        )
        new_state = orch._apply_move(state, ProofAction.MOVE1_LONGER_GRAINS)
        assert new_state is not None
        assert new_state.grains is not None

    def test_trace_summary_is_non_empty(self):
        tubes = make_fan(3, 3, delta=0.1)
        result = run_proof_skeleton(tubes, max_iterations=2, grid_n=16)
        summary = result.trace_summary()
        assert "Proof trace" in summary
        assert len(summary) > 0


class TestProofOrchestratorWithNano:
    def test_run_with_nano_flag(self):
        from kakeya_ai.nano_kernel import is_cpp_available
        tubes = make_fan(3, 3, delta=0.1)
        result = run_proof_skeleton(tubes, max_iterations=2, grid_n=16, use_nano=True)
        assert result is not None
        # 若 nano 内核未编译，则优雅回落。
        assert result.final_metrics.n_tubes == len(tubes)


def make_mixed_tubes(n_cluster=6, n_spread=6, delta=0.1, seed=0):
    """一个窄方向簇加一个穿过原点的均匀扇形。"""
    rng = np.random.default_rng(seed)
    tubes = []
    for i in range(n_cluster):
        d = Direction.from_spherical(np.pi / 2 + 0.01 * i, 0.0)
        tubes.append(
            Tube(center=rng.uniform(-0.4, 0.4, 3), direction=d, length=1.0, delta=delta)
        )
    for i in range(n_spread):
        d = Direction.from_spherical(np.pi / 2, 2 * np.pi * i / n_spread)
        tubes.append(
            Tube(center=rng.uniform(-0.4, 0.4, 3), direction=d, length=1.0, delta=delta)
        )
    return tubes


class TestBroadnessFilterIntegration:
    """宽泛性预过滤器在 grains 分解之前运行（Wang–Zahl §7）。"""

    def _state(self, tubes, delta=0.1):
        return ProofState(
            tubes=tubes,
            shading=None,
            grains=None,
            delta=delta,
            rho=delta,
        )

    def test_filter_defaults_on(self):
        orch = ProofOrchestrator(make_fan(3, 3), max_iterations=1, grid_n=16)
        assert orch.broadness_filter is True
        assert orch.broadness_K == 10.0
        assert orch.broadness_beta == 1.0

    def test_filter_disabled_returns_none(self):
        orch = ProofOrchestrator(
            make_mixed_tubes(), max_iterations=1, grid_n=16, broadness_filter=False
        )
        assert orch._apply_broadness_filter(self._state(make_mixed_tubes())) is None

    def test_filter_strips_narrow_cluster(self):
        tubes = make_mixed_tubes()
        orch = ProofOrchestrator(
            tubes, max_iterations=2, grid_n=16, broadness_K=1.0, broadness_points=25
        )
        state = self._state(tubes)
        new_state = orch._apply_broadness_filter(state)

        assert new_state is not None
        assert len(new_state.tubes) < len(state.tubes)
        assert new_state.grains is None  # 管集变了：grains 必须重建
        assert new_state.broadness_checked is True
        assert new_state.narrow_batches, "stripped batches should be recorded"
        assert orch.last_broadness is not None
        assert orch.last_broadness.n_removed > 0

    def test_filter_short_circuits_when_already_checked(self):
        tubes = make_mixed_tubes()
        orch = ProofOrchestrator(
            tubes, max_iterations=2, grid_n=16, broadness_K=1.0, broadness_points=25
        )
        first = orch._apply_broadness_filter(self._state(tubes))
        assert first is not None
        assert orch._apply_broadness_filter(first) is None

    def test_filter_marks_checked_when_broad(self):
        tubes = make_fan(8, delta=0.1)
        orch = ProofOrchestrator(
            tubes, max_iterations=2, grid_n=16, broadness_K=20.0, broadness_points=15
        )
        result = orch._apply_broadness_filter(self._state(tubes))

        # 没有剥离任何东西，但检查被记录，以免重复采样。
        assert result is not None
        assert len(result.tubes) == len(tubes)
        assert result.broadness_checked is True

    def test_filter_keeps_original_tubes_when_too_few_remain(self):
        tubes = make_mixed_tubes(n_cluster=1, n_spread=1)
        orch = ProofOrchestrator(
            tubes, max_iterations=2, grid_n=16, broadness_K=1.0, broadness_points=15
        )
        state = self._state(tubes)
        result = orch._apply_broadness_filter(state)
        assert result is not None
        # 太小，不足以做有意义的归约：保留原始管集。
        assert len(result.tubes) == len(state.tubes)

    def test_narrow_batches_are_original_indices(self):
        from kakeya_ai.grains import NarrowBatch

        tubes = make_mixed_tubes()
        orch = ProofOrchestrator(
            tubes, max_iterations=2, grid_n=16, broadness_K=1.0, broadness_points=25
        )
        state = self._state(tubes)
        new_state = orch._apply_broadness_filter(state)
        # ``_apply_broadness_filter`` 返回 ``Optional[ProofState]``（None =
        # 无需过滤）。本用例的构型会发生剥离，故必非 None。
        assert new_state is not None
        for batch in new_state.narrow_batches:
            # 2026-09-30：元素由 ``tuple[int, ...]`` 改为 ``NarrowBatch``
            # （因为承接需要"小帽多大"的信息）。索引在 ``.indices`` 下。
            assert isinstance(batch, NarrowBatch)
            for idx in batch.indices:
                assert 0 <= idx < len(tubes), (
                    f"剥离记录的是**原始**管的下标，越界即与输入脱节：{idx}"
                )
            # cap 必须被记录（承接方靠它判断"帽子多大"）。
            # ``0`` 只在检测失败时出现，正常路径应 > 0。
            assert batch.cap_radius >= 0.0

    def test_run_with_filter_is_stable(self):
        tubes = make_mixed_tubes()
        result = run_proof_skeleton(
            tubes, max_iterations=3, grid_n=16, broadness_K=1.0, broadness_points=20
        )
        assert result is not None
        assert len(result.steps) >= 1
        # 过滤器可以缩小管集，但绝不会增大。
        assert result.final_metrics.n_tubes <= len(tubes)
        assert result.final_state.broadness_checked is True

    def test_broadness_action_exists_and_is_reachable(self):
        assert ProofAction.BROADNESS_REDUCTION in ProofAction

    def test_shading_is_subset_after_filter(self):
        tubes = make_mixed_tubes()
        shading = Shading(np.linspace(0.2, 1.0, len(tubes)))
        orch = ProofOrchestrator(
            tubes, max_iterations=2, grid_n=16, broadness_K=1.0, broadness_points=25
        )
        state = ProofState(
            tubes=tubes, shading=shading, grains=None, delta=0.1, rho=0.1
        )
        new_state = orch._apply_broadness_filter(state)
        assert new_state is not None
        if new_state.shading is not None and len(new_state.tubes) < len(tubes):
            assert len(new_state.shading.densities) == len(new_state.tubes)

    def test_state_defaults_are_backward_compatible(self):
        """已有的构造 ProofState 的调用点仍然可用。"""
        state = ProofState(
            tubes=make_fan(2, 2), shading=None, grains=None, delta=0.1, rho=0.1
        )
        assert state.narrow_batches == []
        assert state.broadness_checked is False

    def test_invalid_K_does_not_crash_run(self):
        # 病态的 K 不得破坏主循环。
        tubes = make_mixed_tubes()
        result = run_proof_skeleton(
            tubes, max_iterations=2, grid_n=16, broadness_K=0.0, broadness_points=10
        )
        assert result is not None


class TestMove3UsesCurrentTubeSet:
    """actual 模式下 MOVE3 必须索引*当前*的管集。

    回归：``_apply_move`` 把 ``self.tubes``（原始集合）传给 Move #3 实现，而
    ``grains.tube_indices`` 索引的是 ``state.tubes`` 里保存的过滤后集合。宽泛性
    归约剥离管之后两者不一致，actual 局部 KT 误差就被算在错误的（索引错位的）
    几何上——而且是静默的，因为 proxy 模式不受影响、也没有任何异常抛出。
    """

    def test_move3_receives_state_tubes_not_self_tubes(self, monkeypatch):
        import kakeya_ai.proof_orchestrator as po

        tubes = make_mixed_tubes()
        orch = ProofOrchestrator(
            tubes,
            max_iterations=2,
            grid_n=16,
            broadness_K=1.0,
            broadness_points=25,
            move_params={"error_mode": "actual"},
        )
        # 先过滤，使 state.tubes（宽泛子集）与 self.tubes 不同。
        state = ProofState(
            tubes=list(tubes), shading=None, grains=None, delta=0.1, rho=0.1
        )
        filtered = orch._apply_broadness_filter(state)
        # 返回 ``Optional``；本用例依赖"确实发生了剥离"。断言同时把
        # 这条测试前提与"API 返回类型"两件事都讲清楚。
        assert filtered is not None
        assert len(filtered.tubes) < len(orch.tubes), "filter must strip something"

        state = ProofState(
            tubes=filtered.tubes,
            shading=None,
            grains=orch._build_grains(filtered),
            delta=0.1,
            rho=0.1,
            broadness_checked=True,
        )

        captured = {}
        original = po._move3_impl

        def spy(grains, **kwargs):
            captured["tubes"] = kwargs.get("tubes")
            return original(grains, **kwargs)

        monkeypatch.setattr(po, "_move3_impl", spy)
        orch._apply_move(state, ProofAction.MOVE3_WIDER_GRAINS)

        assert captured["tubes"] is state.tubes
        assert len(captured["tubes"]) == len(filtered.tubes)
        assert len(captured["tubes"]) != len(orch.tubes)

    def test_actual_mode_error_is_computed_on_filtered_set(self):
        """actual 模式的误差必须有限，且与状态一致。"""
        tubes = make_mixed_tubes()
        orch = ProofOrchestrator(
            tubes,
            max_iterations=2,
            grid_n=16,
            broadness_K=1.0,
            broadness_points=25,
            move_params={"error_mode": "actual"},
        )
        state = ProofState(
            tubes=list(tubes), shading=None, grains=None, delta=0.1, rho=0.1
        )
        filtered = orch._apply_broadness_filter(state)
        assert filtered is not None
        state = ProofState(
            tubes=filtered.tubes,
            shading=None,
            grains=orch._build_grains(filtered),
            delta=0.1,
            rho=0.1,
            broadness_checked=True,
        )
        new_state = orch._apply_move(state, ProofAction.MOVE3_WIDER_GRAINS)
        assert new_state.grains is not None
        for prism in new_state.grains.prisms:
            assert np.isfinite(prism.local_kt_error)
            assert prism.local_kt_error >= 0.0


class TestRunTraceFidelity:
    """每一步都必须记录其自身输入状态真实的 *before* 度量。

    回归：``run()`` 写成 ``before = steps[-1].after if steps else after``，
    这使第一步的 ``before`` 与它的 ``after`` 字面上是同一个对象——一个
    改变了状态的步骤仍然读作“没有变化”。
    """

    def test_first_step_before_equals_initial_metrics(self):
        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(tubes, max_iterations=2, grid_n=16)

        expected = orch._compute_metrics(orch.initialize())
        result = orch.run()

        assert result.steps, "run must produce at least one step"
        first = result.steps[0]
        assert first.before is not first.after
        assert first.before.volume_ratio == pytest.approx(expected.volume_ratio)
        assert first.before.n_tubes == expected.n_tubes
        assert first.before.delta == expected.delta

    def test_step_boundaries_are_consistent(self):
        """第 k 步的 after 必须等于第 k+1 步的 before（无缺口、无伪造数据）。"""
        tubes = make_fan(3, 3, delta=0.1)
        result = run_proof_skeleton(tubes, max_iterations=3, grid_n=16)
        # ``pairwise``（相邻配对）而不是 ``zip(steps, steps[1:])``：后者两序列
        # 天然不等长，与 B905 要求的 strict 直接冲突。
        for prev, nxt in pairwise(result.steps):
            assert nxt.before.volume_ratio == pytest.approx(prev.after.volume_ratio)
            assert nxt.before.n_tubes == prev.after.n_tubes
            assert nxt.before.delta == pytest.approx(prev.after.delta)

    def test_final_metrics_match_last_step_after(self):
        tubes = make_fan(3, 3, delta=0.1)
        result = run_proof_skeleton(tubes, max_iterations=3, grid_n=16)
        last = result.steps[-1]
        assert result.final_metrics.volume_ratio == pytest.approx(last.after.volume_ratio)
        assert result.final_metrics.n_tubes == last.after.n_tubes

    def test_explicit_before_is_reused_by_step(self):
        """传入 ``before`` 时必须跳过重算（性能契约）。"""
        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)
        state = orch.initialize()
        cached = orch._compute_metrics(state)

        calls = []
        original = orch._compute_metrics

        # **参数名必须与真实签名一致**（``state``）：pyright 按**名称**校验
        # 函数兼容性，写成 ``s`` 会被判为不兼容。
        def counting(state):
            calls.append(state)
            return original(state)

        orch._compute_metrics = counting
        orch.step(state, before=cached)
        # 第一个宽泛性微步骤可以重算；关键在于
        # step() 不重算*它自己的* ``before``。
        assert len(calls) <= 2


class TestVerificationFailuresAreVisible:
    """失败的检查必须被记录——而不是静默地等同于否定裁决。

    回归：``_compute_metrics``、``_apply_broadness_filter``、
    ``_try_nsp_factorization`` 以及代理路径都用裸 ``except`` 吞掉异常，
    于是验证机制崩溃的编排器与合法地判定“否”的编排器无法区分。
    """

    def test_sticky_failure_is_noted(self, monkeypatch):
        import kakeya_ai.proof_orchestrator as po

        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)

        def boom(*args, **kwargs):
            raise RuntimeError("scale chain exploded")

        monkeypatch.setattr(po, "build_scale_chain", boom)
        state = orch.initialize()
        metrics = orch._compute_metrics(state)

        assert metrics.is_sticky is False  # 仍然优雅降级
        assert any("sticky check failed" in d for d in orch._diagnostics)
        assert any("scale chain exploded" in d for d in orch._diagnostics)

    def test_diagnostics_reach_step_notes(self, monkeypatch):
        import kakeya_ai.proof_orchestrator as po

        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)

        def boom(*args, **kwargs):
            raise RuntimeError("wolff unavailable")

        monkeypatch.setattr(po, "build_scale_chain", boom)
        state = orch.initialize()
        _new_state, _action, notes = orch.step(state)
        assert any("sticky check failed" in n for n in notes)

    def test_broadness_failure_is_noted(self, monkeypatch):
        import kakeya_ai.proof_orchestrator as po

        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)

        def boom(*args, **kwargs):
            raise RuntimeError("no points sampled")

        monkeypatch.setattr(po, "broadness_reduction", boom)
        state = orch.initialize()
        assert orch._apply_broadness_filter(state) is None
        assert any("broadness filter failed" in d for d in orch._diagnostics)

    def test_nsp_failure_is_noted(self, monkeypatch):
        import kakeya_ai.proof_orchestrator as po

        # delta=0.1 -> target_count = 100, need < 50 tubes to attempt NSP.
        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)

        def boom(*args, **kwargs):
            raise RuntimeError("nsp unsupported")

        monkeypatch.setattr(po, "nsp_factorization", boom)
        state = orch.initialize()
        assert orch._try_nsp_factorization(state) is None
        assert any("NSP factorization failed" in d for d in orch._diagnostics)

    def test_surrogate_failure_is_noted(self):
        tubes = make_fan(3, 3, delta=0.1)

        class BrokenSurrogate:
            def predict_scores(self, state_dict):
                raise RuntimeError("weights corrupted")

        orch = ProofOrchestrator(
            tubes, max_iterations=1, grid_n=16,
            surrogate=BrokenSurrogate(),  # type: ignore[arg-type]
        )
        state = ProofState(
            tubes=tubes,
            shading=None,
            grains=orch._build_grains(
                ProofState(tubes=tubes, shading=None, grains=None, delta=0.1, rho=0.1)
            ),
            delta=0.1,
            rho=0.1,
        )
        metrics = orch._compute_metrics(state)
        action = orch._select_move(state, metrics)

        assert action in ProofAction  # 启发式兜底仍然可用
        assert any("surrogate prediction failed" in d for d in orch._diagnostics)

    def test_diagnostics_reset_between_steps(self, monkeypatch):
        import kakeya_ai.proof_orchestrator as po

        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)

        def boom(*args, **kwargs):
            raise RuntimeError("first call only")

        monkeypatch.setattr(po, "build_scale_chain", boom)
        state = orch.initialize()
        _s1, _a1, notes1 = orch.step(state)
        assert any("sticky check failed" in n for n in notes1)

        # 修复依赖；第二步不得继承过期的诊断信息。
        monkeypatch.undo()
        _s2, _a2, notes2 = orch.step(orch.initialize())
        assert not any("sticky check failed" in n for n in notes2)


class TestTerminationSemantics:
    """``exhausted`` 区分“预算用尽”与“已得出结论”。"""

    def test_exhausted_true_when_max_iterations_hit(self):
        tubes = make_fan(3, 3, delta=0.1)
        result = run_proof_skeleton(tubes, max_iterations=1, grid_n=16)
        # 它要么在第一步就终止，要么耗尽预算；
        # 两者都必须被明确报告。
        if result.exhausted:
            assert "max_iterations" in result.message
            assert result.steps, "exhausted runs still record their steps"
            assert result.failure_reason is FailureReason.BUDGET_EXHAUSTED
        else:
            # ``message`` 现在统一取自 ``FAILURE_REASON_TEXT``，不再手写文案。
            assert result.message == FAILURE_REASON_TEXT[result.failure_reason]

    def test_max_iterations_zero_does_not_crash(self):
        tubes = make_fan(2, 2, delta=0.1)
        result = run_proof_skeleton(tubes, max_iterations=0, grid_n=16)
        assert result is not None
        assert result.exhausted is True
        assert result.steps == []
        assert result.final_metrics.n_tubes == len(tubes)

    def test_exhausted_defaults_false_for_manual_construction(self):
        from kakeya_ai.proof_orchestrator import ProofMetrics, ProofResult, ProofScope

        metrics = ProofMetrics(
            delta=0.1, rho=0.1, a=0.1, b=0.1, c=0.1, local_error=0.0,
            volume_ratio=1.0, union_volume=1.0, lower_bound=1.0,
            n_tubes=1, n_grains=0,
        )
        result = ProofResult(
            success=True, steps=[], final_state=None,  # type: ignore[arg-type]
            final_metrics=metrics, message="manual",
            # 2026-09-30 L1：`success=True` **必须**显式声明 scope=FULL，
            # 否则无法区分"真的证明了原输入"与"只在 broad 子集上成立"。
            scope=ProofScope.FULL,
        )
        assert result.exhausted is False
        assert result.scope is ProofScope.FULL

    def test_success_true_without_full_scope_is_rejected(self):
        """**L1 的核心不变量**：`success=True` 必须配 `scope=FULL`。

        这条测试的存在理由就是它要防的那个 bug：修复前 `success=True` 会
        与 `uncovered_narrow_tubes>0`（即"下界只管 broad 子集"）**共存**，
        而这个布尔会被下游读作"证明成功"。现在它直接构造失败。

        构造时抛错（而不是自动纠正）是**有意的摩擦**：自动纠正会让调用方
        写错而不自知，它仍然以为自己设了 `success`。
        """
        from kakeya_ai.errors import ContractViolationError
        from kakeya_ai.proof_orchestrator import (
            ProofMetrics,
            ProofResult,
            ProofScope,
        )

        metrics = ProofMetrics(
            delta=0.1, rho=0.1, a=0.1, b=0.1, c=0.1, local_error=0.0,
            volume_ratio=1.0, union_volume=1.0, lower_bound=1.0,
            n_tubes=1, n_grains=0,
        )
        # C408 建议改字面量。**保留 ``dict()``**：这里的 ``# type: ignore``
        # 与 key 绑定（``final_state=None`` 类型不符）；改成字面量会把 ignore
        # 的位置与作用范围一起扰动，而改动收益仅风格。
        base = dict(  # noqa: C408
            success=True, steps=[], final_state=None,  # type: ignore[arg-type]
            final_metrics=metrics, message="manual",
        )
        # ① 默认 scope（NOT_APPLICABLE）下说成功 → 拒
        with pytest.raises(ContractViolationError, match="scope"):
            ProofResult(**base)  # type: ignore[arg-type]
        # ② 声明 BROAD_ONLY 却说成功 → 拒
        with pytest.raises(ContractViolationError, match="BROAD_ONLY"):
            ProofResult(
                **base,  # type: ignore[arg-type]
                scope=ProofScope.BROAD_ONLY,
                uncovered_narrow_tubes=3,
            )
        # ③ 有未覆盖窄簇却不标 BROAD_ONLY → 拒（两处记录同一事实）
        with pytest.raises(ContractViolationError, match="uncovered_narrow_tubes"):
            ProofResult(
                success=False, steps=[], final_state=None,  # type: ignore[arg-type]
                final_metrics=metrics, message="manual",
                scope=ProofScope.FULL,
                uncovered_narrow_tubes=3,
            )

    def test_trace_summary_mentions_exhaustion(self):
        tubes = make_fan(2, 2, delta=0.1)
        result = run_proof_skeleton(tubes, max_iterations=0, grid_n=16)
        assert "exhausted" in result.trace_summary()

    def test_exhausted_run_reports_final_metrics_consistently(self):
        """final_metrics 必须与最后记录的步骤一致（不做额外重算）。"""
        tubes = make_fan(3, 3, delta=0.1)
        result = run_proof_skeleton(tubes, max_iterations=1, grid_n=16)
        if result.steps:
            assert result.final_metrics.volume_ratio == pytest.approx(
                result.steps[-1].after.volume_ratio
            )
            assert result.final_metrics.n_tubes == result.steps[-1].after.n_tubes


class TestFailureAttribution:
    """``failure_reason`` 把 "没成功" 拆成可处置的几类。

    此前 ``success=False`` 只有一个含义，调用方分不清"加预算就行"还是"这条路
    根本走不通"。这些测试把每类的**触发路径**钉住。
    """

    def test_empty_input_is_attributed(self):
        result = run_proof_skeleton([], max_iterations=2)
        assert result.failure_reason is FailureReason.EMPTY_INPUT
        assert result.message == FAILURE_REASON_TEXT[FailureReason.EMPTY_INPUT]

    def test_budget_exhausted_is_attributed(self):
        tubes = make_fan(3, 3, delta=0.1)
        # max_iterations=0 必然走"预算用尽"分支。
        result = run_proof_skeleton(tubes, max_iterations=0, grid_n=16)
        assert result.exhausted is True
        assert result.failure_reason is FailureReason.BUDGET_EXHAUSTED

    def test_success_reports_none(self):
        """成功时归因必须是 ``NONE``，不能退化成某个失败类别。"""
        tubes = make_fan(3, 3, delta=0.1)
        result = run_proof_skeleton(tubes, max_iterations=4, grid_n=16)
        if result.success and not result.exhausted:
            assert result.failure_reason is FailureReason.NONE

    def test_every_reason_has_human_text(self):
        """每个枚举成员都要有文案——否则 ``message`` 会退化成空串。"""
        for reason in FailureReason:
            assert reason in FAILURE_REASON_TEXT
            assert FAILURE_REASON_TEXT[reason]

    def test_terminal_failure_message_matches_reason(self):
        """终止失败时，``message`` 必须与 ``failure_reason`` 自洽。"""
        tubes = make_fan(3, 3, delta=0.1)
        result = run_proof_skeleton(tubes, max_iterations=1, grid_n=16)
        if not result.success and not result.exhausted:
            assert result.message == FAILURE_REASON_TEXT[result.failure_reason]
            assert result.failure_reason in (
                FailureReason.STICKY_LOWER_BOUND_UNMET,
                FailureReason.RECURSION_EXHAUSTED,
            )

    def test_manual_construction_defaults_are_consistent(self):
        """手工构造的 ``ProofResult`` 也不能拿到 ``failure_reason=None``。"""
        from kakeya_ai.proof_orchestrator import ProofMetrics, ProofScope

        metrics = ProofMetrics(
            delta=0.1, rho=0.1, a=0.1, b=0.1, c=0.1, local_error=0.0,
            volume_ratio=1.0, union_volume=1.0, lower_bound=1.0,
            n_tubes=1, n_grains=0,
        )
        ok = ProofResult(
            success=True, steps=[], final_state=None,  # type: ignore[arg-type]
            final_metrics=metrics, message="manual",
            scope=ProofScope.FULL,  # 见 L1 不变量
        )
        assert ok.failure_reason is FailureReason.NONE

        bad = ProofResult(
            success=False, steps=[], final_state=None,  # type: ignore[arg-type]
            final_metrics=metrics, message="manual",
        )
        # 未显式归因的失败必须如实报 UNKNOWN，**不能**猜成某个具体类别。
        # 归成"预算用尽"会让调用方去加 max_iterations，而真实原因可能是
        # 黏性/递归到底/输入退化——那是编造归因。
        assert bad.failure_reason is FailureReason.UNKNOWN

    def test_unknown_is_distinct_from_budget_exhausted(self):
        """``UNKNOWN`` 与 ``BUDGET_EXHAUSTED`` 必须是两个不同的值。

        合并它们会让"没归因"被读成"预算用尽"，正是本项要防的编造。
        """
        assert FailureReason.UNKNOWN is not FailureReason.BUDGET_EXHAUSTED
        assert FAILURE_REASON_TEXT[FailureReason.UNKNOWN]

    def test_explicit_attribution_survives_post_init(self):
        """显式给出的归因不能被 ``__post_init__`` 覆盖掉。"""
        from kakeya_ai.proof_orchestrator import ProofMetrics

        metrics = ProofMetrics(
            delta=0.1, rho=0.1, a=0.1, b=0.1, c=0.1, local_error=0.0,
            volume_ratio=1.0, union_volume=1.0, lower_bound=1.0,
            n_tubes=1, n_grains=0,
        )
        r = ProofResult(
            success=False, steps=[], final_state=None,  # type: ignore[arg-type]
            final_metrics=metrics, message="m",
            failure_reason=FailureReason.STICKY_LOWER_BOUND_UNMET,
        )
        assert r.failure_reason is FailureReason.STICKY_LOWER_BOUND_UNMET

    def test_reason_does_not_leak_across_steps(self):
        """上一步的归因不能残留到下一步。

        ``step()`` 每步开头清空槽位；若不清空，一步"黏性失败"之后的所有步骤
        都会被归因成黏性失败，即使它们其实是别的路径。
        """
        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)
        state = orch.initialize()
        before = orch._compute_metrics(state)
        orch.step(state, before=before)
        first = orch._last_failure_reason

        state2 = orch.initialize()
        before2 = orch._compute_metrics(state2)
        orch.step(state2, before=before2)
        # 同一输入 → 同一归因（不是残留导致的巧合）。
        assert orch._last_failure_reason is first

    def test_trace_summary_includes_reason(self):
        tubes = make_fan(3, 3, delta=0.1)
        result = run_proof_skeleton(tubes, max_iterations=0, grid_n=16)
        summary = result.trace_summary()
        assert "Failure reason" in summary
        assert "BUDGET_EXHAUSTED" in summary


class TestPerMoveGainIsRecorded:
    """缺口二：Move 步骤的 notes 必须写明"本次是否达成定量增益"。

    **为什么需要**：论文要求"单步至少放大 δ^{-ζ}"，实现是"迭代合并直到条件
    满足"，无单步放大保证。实测（内部分批测量脚本）Move #2
    空操作率 100%、Move #1 空操作率 13/16。若不记录，trace 里"应用了但没推进"
    （`NO_PROGRESS` 的微观来源）与"应用了并推进了"分不开。

    判据由 ``moves._move_target_delta`` **单点定义**（Move #1 看 max_c 应增，
    Move #2/#3 看量应减），编排器只做薄封装——不另写一份方向判断。
    """

    def test_move_step_records_target_and_gain(self):
        from kakeya_ai.proof_orchestrator import _target_improved

        # 轻量对象：``_move_target_delta`` 只读这三个属性。
        def _g(c, rho, err):
            return type("G", (), {"max_c": c, "max_rho": rho,
                                  "max_local_kt_error": err})()

        before = _g(1.0, 0.5, 3.0)
        after = _g(2.0, 0.2, 1.0)
        # 目标量真变 ⇒ True（方向因 Move 而异，由 moves 单点定义）。
        assert _target_improved("max_c", before, after) is True
        assert _target_improved("max_rho", before, after) is True
        assert _target_improved("max_local_kt_error", before, after) is True
        # 反方向对 max_c 而言不是改进。
        assert _target_improved("max_c", after, before) is False
        # None（grains 尚未构建）按"未改进"处理，不抛。
        assert _target_improved("max_c", None, after) is False
        assert _target_improved("max_c", before, None) is False

    def test_run_notes_include_achieved_gain_for_move_steps(self):
        """整体跑一遍，凡是 Move 步都应带 ``achieved_gain=`` 备注。

        用 ``move-reaching`` 构型：实测它会走到 ``MOVE3`` 与 ``MOVE2``
        （``make_mixed_tubes`` 判 sticky、一步终局、走不到 Move）。
        """
        import numpy as np
        from kakeya_ai.primitives import Direction, Tube

        # 与 scripts/demo_proof_orchestrator.make_move_reaching_tubes 同源
        # （参数实测能走到 Move 分支）。
        n, delta, spread = 16, 0.05, 0.05
        tubes = [
            Tube(
                center=np.zeros(3),
                direction=Direction.from_spherical(
                    np.pi / 2 + spread * i, 2 * np.pi * i / n
                ),
                length=1.0,
                delta=delta,
            )
            for i in range(n)
        ]
        result = run_proof_skeleton(
            tubes, seed=42, max_iterations=12, grid_n=16,
            move_params={"error_mode": "actual"},
        )
        move_steps = [
            s for s in result.steps
            if s.action.name.startswith("MOVE")
        ]
        # 前提守卫：本构型必须真的走到 Move，否则断言是空转的。
        assert move_steps, "该构型未走到任何 Move，本用例无判别力"
        for s in move_steps:
            joined = " ".join(s.notes)
            assert "achieved_gain=" in joined, (
                f"{s.action.name} 步未记录 achieved_gain；notes={s.notes}"
            )
            assert "target=" in joined, (
                f"{s.action.name} 步未记录目标量；notes={s.notes}"
            )
            # 缺口二补强：本步 Move **实际执行的合并轮数**（精确值）必须既进
            # notes、又落到结构化字段 ``ProofStep.iters_used``（供机器判定）。
            assert "iters_used=" in joined, (
                f"{s.action.name} 步未记录 iters_used；notes={s.notes}"
            )
            assert isinstance(s.iters_used, int) and s.iters_used >= 0, (
                f"{s.action.name} 步的 iters_used 应为非负整数，实得 {s.iters_used!r}"
            )
        # 反向：非 Move 步没有"轮数"语义，必须是 ``None``（而非误填 0）。
        for s in result.steps:
            if not s.action.name.startswith("MOVE"):
                assert s.iters_used is None, (
                    f"{s.action.name} 非 Move 步，iters_used 应为 None，实得 {s.iters_used!r}"
                )
