"""``beam_scorer`` 注入的接线测试。

**为什么需要。** 此前 ``beam_search_score``（LLM beam search 打分器）只有
「``__init__`` 导出 + 演示脚本 + 测试」三类调用点，**没有一条接进主循环**：
``_select_move`` 只认启发式与 surrogate。也就是说文档里"AI 快思考推理层"
在闭环上是断开的。这组测试把新接的第三条路径钉住。

**测试全部用桩，绝不调真 LLM。** 这是刻意的：本项目吃过"测试真发 HTTP 请求"
的亏（单个用例 28 秒、内存压到 0.89 GB）。所以 ``beam_scorer`` 设计成
**必须显式注入**、默认不启用，本就与"可打桩"是一体的。
"""

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.proof_orchestrator import (
    ProofAction,
    ProofOrchestrator,
    ProofState,
    run_proof_skeleton,
)


def make_fan(n_theta=3, n_phi=3, delta=0.1):
    return [
        Tube(center=np.zeros(3),
             direction=Direction.from_spherical(theta, phi),
             length=1.0, delta=delta)
        for theta in np.linspace(0.2, np.pi - 0.2, n_theta)
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    ]


# **2026-09-30 起：打分器的结论也要过 Move 条件守卫**
#
# ``ProofOrchestrator._scored_action_allowed`` 会拦下"条件已满足"的 Move
# ——实测那种 Move 必然空转（surrogate 侧空转率曾达 86%）。所以构造"能被采纳的
# 打分结果"时，**必须让目标 Move 的条件确实不满足**。
#
# ``check_move2_condition`` 只依赖 ``omega`` 与 ``delta``
# （``threshold = delta**(omega/100)``）：默认 ``omega=0.1`` 时阈值 ≈ 0.9977，
# 几乎恒满足；取 ``omega=50`` 则阈值降到 ≈ 0.316 < prisms 的最大 rho，
# 条件为 False。故本文件的编排器统一传 ``move_params={"omega": 50.0}``。
#
# 这不是为迁就测试——守卫让无效 Move 减少 82%（步数 108→76，成功率不变）。


def _state_with_grains(orch):
    """构造一个已建好 grains 的状态——否则 ``_select_move`` 会先返回
    ``GRAINS_DECOMPOSITION``，走不到打分器。"""
    base = orch.initialize()
    grains = orch._build_grains(base)
    return ProofState(
        tubes=base.tubes, shading=base.shading, grains=grains,
        delta=base.delta, rho=base.rho,
    )


class _StubScorer:
    """记录调用并可返回任意结果的桩打分器。"""

    def __init__(self, scores, source="stub", raises=None):
        self.scores = scores
        self.source = source
        self.raises = raises
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, prompt, *, state=None, **kwargs):
        self.calls.append((prompt, state or {}))
        if self.raises is not None:
            raise self.raises
        return self.scores, self.source


class TestDefaultIsDisabled:
    """默认**不启用**——这是安全属性，不是风格偏好。

    若默认启用，一次普通的 ``run()`` 就会隐式依赖外部服务：离线环境下行为
    改变、测试变慢变脆、且"结果可复现"这条前提被外部状态破坏。
    """

    def test_default_scorer_is_none(self):
        orch = ProofOrchestrator(make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0})
        assert orch.llm_scorer is None

    def test_run_without_scorer_never_touches_llm(self):
        """不注入时，整条 run() 不应有任何 LLM 调用机会。"""
        r = run_proof_skeleton(make_fan(), max_iterations=2, grid_n=16)
        assert r is not None
        assert r.steps


class TestSelection:
    def test_stub_scorer_drives_move_selection(self):
        stub = _StubScorer({1: 1.0, 2: 9.0, 3: 2.0}, source="stub")
        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0}, llm_scorer=stub
        )
        state = _state_with_grains(orch)
        metrics = orch._compute_metrics(state)

        action = orch._select_move(state, metrics)

        assert len(stub.calls) == 1, "打分器应被调用一次"
        assert action is ProofAction.MOVE2_REPLACE_SQUARE_GRAINS, (
            "应选分数最高的 move_2"
        )
        assert any("LLM scorer" in n for n in orch._selection_notes)

    def test_note_records_source_and_gap(self):
        """备注必须写明**来源与 gap**——否则事后无法判断决策来自哪条路径。"""
        stub = _StubScorer({1: 0.0, 2: 9.0, 3: 0.0}, source="my_source")
        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0}, llm_scorer=stub
        )
        state = _state_with_grains(orch)
        orch._select_move(state, orch._compute_metrics(state))

        note = " ".join(orch._selection_notes)
        assert "my_source" in note
        assert "gap=" in note

    def test_low_confidence_falls_back_to_heuristic(self):
        """gap 低于阈值时不采用打分器结果，退回启发式。"""
        # 三个 move 分数接近 → gap 小
        stub = _StubScorer({1: 5.0, 2: 5.1, 3: 5.0}, source="stub")
        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0},
            llm_scorer=stub, llm_threshold=2.0,
        )
        state = _state_with_grains(orch)
        action = orch._select_move(state, orch._compute_metrics(state))

        assert action in ProofAction
        assert not any("LLM scorer" in n for n in orch._selection_notes), (
            "低置信度不应记为 beam 命中"
        )

    def test_threshold_boundary(self):
        """gap 恰好等于阈值时应采用（条件是 >=，不是 >）。"""
        stub = _StubScorer({1: 3.0, 2: 5.0, 3: 3.0}, source="stub")  # gap = 2.0
        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0},
            llm_scorer=stub, llm_threshold=2.0,
        )
        state = _state_with_grains(orch)
        orch._select_move(state, orch._compute_metrics(state))
        assert any("LLM scorer" in n for n in orch._selection_notes)


class TestFailureIsVisible:
    """打分器失败必须**留痕并降级**，而不是抛出或静默。"""

    def test_scorer_exception_is_recorded(self):
        stub = _StubScorer({}, raises=RuntimeError("ollama down"))
        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0}, llm_scorer=stub
        )
        state = _state_with_grains(orch)
        action = orch._select_move(state, orch._compute_metrics(state))

        assert action in ProofAction, "失败后仍应给出可用动作"
        assert any("LLM scorer" in d for d in orch._diagnostics)
        assert any("ollama down" in d for d in orch._diagnostics), (
            "诊断要带原始异常信息，否则无法排查"
        )

    def test_unusable_scores_are_recorded(self):
        stub = _StubScorer(None, source="stub")
        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0},
            # ``_StubScorer`` 是**鸭子类型桩**（结构上满足 ``LLMScorerLike``，
            # 但不是它的名义子类）。协议本就为注入而生，此处不兼容仅因类未声明。
            llm_scorer=stub,  # type: ignore[arg-type]
        )
        state = _state_with_grains(orch)
        action = orch._select_move(state, orch._compute_metrics(state))

        assert action in ProofAction
        assert any("unusable scores" in d for d in orch._diagnostics)

    def test_empty_scores_are_recorded(self):
        stub = _StubScorer({}, source="stub")
        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0}, llm_scorer=stub
        )
        state = _state_with_grains(orch)
        orch._select_move(state, orch._compute_metrics(state))
        assert any("unusable scores" in d for d in orch._diagnostics)

    def test_bare_scores_without_source_accepted(self):
        """容忍只返回 scores 的桩（测试常这样写），但 source 如实标 unknown。"""

        class _Bare:
            def __call__(self, prompt, *, state=None, **kw):
                return {1: 0.0, 2: 9.0, 3: 0.0}

        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0},
            # 同上：``_Bare`` 是鸭子类型桩（只提供 ``__call__``）。
            llm_scorer=_Bare(),  # type: ignore[arg-type]
        )
        state = _state_with_grains(orch)
        action = orch._select_move(state, orch._compute_metrics(state))

        assert action is ProofAction.MOVE2_REPLACE_SQUARE_GRAINS
        assert any("unknown" in n for n in orch._selection_notes)


class TestPrecedence:
    """beam scorer 优先于 surrogate——两者都是解决同一问题的独立路径。"""

    def test_beam_wins_over_surrogate_when_both_present(self):
        from kakeya_ai.ai.surrogate_scorer import SurrogateScorer
        from kakeya_ai.ai.training_data import FEATURE_NAMES

        # surrogate 恒输出 move_1 最高（bias 偏向第 1 个输出）
        surrogate = SurrogateScorer(
            input_dim=len(FEATURE_NAMES), hidden_dims=[4], output_dim=3, seed=0
        )
        n_layers = len(surrogate.params) // 2
        out_w_pid = 2 * (n_layers - 1)
        surrogate.params[out_w_pid] = np.zeros_like(surrogate.params[out_w_pid])
        surrogate.params[out_w_pid + 1] = np.array([9.0, 1.0, 1.0])

        stub = _StubScorer({1: 1.0, 2: 9.0, 3: 1.0}, source="stub")
        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0},
            surrogate=surrogate, llm_scorer=stub,
        )
        state = _state_with_grains(orch)
        action = orch._select_move(state, orch._compute_metrics(state))

        assert len(stub.calls) == 1, "beam scorer 应先被调用"
        assert action is ProofAction.MOVE2_REPLACE_SQUARE_GRAINS, (
            "beam 的结果应胜过 surrogate 的 move_1"
        )

    def test_surrogate_used_when_beam_absent(self):
        from kakeya_ai.ai.surrogate_scorer import SurrogateScorer
        from kakeya_ai.ai.training_data import FEATURE_NAMES

        surrogate = SurrogateScorer(
            input_dim=len(FEATURE_NAMES), hidden_dims=[4], output_dim=3, seed=0
        )
        n_layers = len(surrogate.params) // 2
        out_w_pid = 2 * (n_layers - 1)
        surrogate.params[out_w_pid] = np.zeros_like(surrogate.params[out_w_pid])
        surrogate.params[out_w_pid + 1] = np.array([9.0, 1.0, 1.0])

        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0}, surrogate=surrogate,
            surrogate_threshold=1.0,
        )
        state = _state_with_grains(orch)
        action = orch._select_move(state, orch._compute_metrics(state))

        assert action is ProofAction.MOVE1_LONGER_GRAINS
        assert any("surrogate" in n for n in orch._selection_notes)


class TestRunSkeletonPassthrough:
    def test_run_proof_skeleton_accepts_llm_scorer(self):
        """便捷入口必须能透传——否则用户只能手动构造 orchestrator。"""
        stub = _StubScorer({1: 9.0, 2: 1.0, 3: 1.0}, source="stub")
        r = run_proof_skeleton(
            make_fan(), max_iterations=2, grid_n=16, llm_scorer=stub
        )
        assert r is not None

    def test_prompt_is_well_formed(self):
        """喂给打分器的 prompt 应包含状态信息（否则 LLM 无从判断）。"""
        stub = _StubScorer({1: 9.0, 2: 1.0, 3: 1.0})
        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0}, llm_scorer=stub
        )
        state = _state_with_grains(orch)
        orch._select_move(state, orch._compute_metrics(state))

        prompt, state_dict = stub.calls[0]
        assert "Kakeya" in prompt
        assert "move_1" in prompt
        # state dict 必须带全部特征键（供 LLM 打分器与缓存使用）
        from kakeya_ai.ai.training_data import FEATURE_NAMES

        assert set(state_dict) == set(FEATURE_NAMES)


class TestLLMStrategy:
    """``llm_strategy`` 选择 beam / cascaded，两者共用同一段接线逻辑。"""

    def test_valid_strategies_accepted(self):
        for strategy in ("beam", "cascaded"):
            orch = ProofOrchestrator(
                make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0},
                llm_scorer=_StubScorer({1: 9.0, 2: 1.0, 3: 1.0}),
                llm_strategy=strategy,
            )
            assert orch.llm_strategy == strategy

    def test_unknown_strategy_rejected_at_construction(self):
        """非法策略必须**构造时**报错，而不是等到选 Move 时静默走错路径。"""
        with pytest.raises(ValueError, match="llm_strategy"):
            ProofOrchestrator(
                make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0},
                llm_scorer=_StubScorer({1: 9.0, 2: 1.0, 3: 1.0}),
                llm_strategy="bogus",
            )

    def test_strategy_recorded_in_note(self):
        """备注必须带策略名——否则事后分不清用的是哪条 LLM 路径。"""
        stub = _StubScorer({1: 1.0, 2: 9.0, 3: 1.0}, source="cascaded")
        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0},
            llm_scorer=stub, llm_strategy="cascaded",
        )
        state = _state_with_grains(orch)
        orch._select_move(state, orch._compute_metrics(state))

        note = " ".join(orch._selection_notes)
        assert "strategy=cascaded" in note
        assert "source=cascaded" in note

    def test_diagnostics_include_strategy_on_failure(self):
        """失败诊断要带策略名，便于判断是哪条 LLM 路径出的问题。"""
        stub = _StubScorer({}, raises=RuntimeError("deep model OOM"))
        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0},
            llm_scorer=stub, llm_strategy="cascaded",
        )
        state = _state_with_grains(orch)
        orch._select_move(state, orch._compute_metrics(state))

        diag = " ".join(orch._diagnostics)
        assert "cascaded" in diag
        assert "deep model OOM" in diag


class TestMakeLLMScorer:
    """``make_llm_scorer`` 让调用方不必手写转发 lambda。"""

    def test_unknown_strategy_raises(self):
        from kakeya_ai.proof_orchestrator import make_llm_scorer

        with pytest.raises(ValueError, match="未知的 LLM 策略"):
            make_llm_scorer("nonsense")

    def test_returns_callable_for_both_strategies(self):
        from kakeya_ai.proof_orchestrator import make_llm_scorer

        for strategy in ("beam", "cascaded"):
            scorer = make_llm_scorer(strategy)
            assert callable(scorer), f"{strategy} 应返回可调用对象"

    def test_kwargs_forwarded_and_overridable(self, monkeypatch):
        """构造时的 kwargs 要转发给实现，且**调用时**可覆盖。"""
        from kakeya_ai.proof_orchestrator import make_llm_scorer

        captured = {}

        def fake_impl(prompt, *, state=None, **kwargs):
            captured.update(kwargs)
            return {1: 9.0, 2: 1.0, 3: 1.0}, "fake"

        import kakeya_ai.ai.beam_search as bs

        monkeypatch.setattr(bs, "beam_search_score", fake_impl)

        scorer = make_llm_scorer("beam", beam_width=8)
        scorer("p", state={})
        assert captured["beam_width"] == 8, "构造时的 kwargs 应转发"

        captured.clear()
        scorer("p", state={}, beam_width=2)
        assert captured["beam_width"] == 2, "调用时的 kwargs 应覆盖构造值"

    def test_scorer_usable_end_to_end(self, monkeypatch):
        """整条链路：make_llm_scorer → 注入 → 选出 Move（用假实现，不调 LLM）。"""
        from kakeya_ai.proof_orchestrator import make_llm_scorer

        def fake_impl(prompt, *, state=None, **kwargs):
            return {1: 1.0, 2: 9.0, 3: 1.0}, "fake"

        import kakeya_ai.ai.cascaded_reasoning as cr

        monkeypatch.setattr(cr, "cascaded_score", fake_impl)

        scorer = make_llm_scorer("cascaded")
        orch = ProofOrchestrator(
            make_fan(), max_iterations=1, grid_n=16, move_params={"omega": 50.0},
            llm_scorer=scorer, llm_strategy="cascaded",
        )
        state = _state_with_grains(orch)
        action = orch._select_move(state, orch._compute_metrics(state))

        assert action is ProofAction.MOVE2_REPLACE_SQUARE_GRAINS
        assert any("strategy=cascaded" in n for n in orch._selection_notes)
