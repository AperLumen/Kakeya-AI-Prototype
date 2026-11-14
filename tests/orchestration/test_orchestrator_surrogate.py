"""证明编排器中代理模型集成的测试。"""

import json

import numpy as np
import pytest

from kakeya_ai.ai.surrogate_scorer import SurrogateScorer
from kakeya_ai.ai.training_data import (
    FEATURE_NAMES,
    _DEFAULT_FEATURE,
    proof_steps_to_samples,
)
from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.proof_orchestrator import (
    ProofAction,
    ProofMetrics,
    ProofOrchestrator,
    ProofState,
    ProofStep,
)


def make_fan(n_theta=4, n_phi=4, delta=0.1):
    tubes = []
    for theta in np.linspace(0.2, np.pi - 0.2, n_theta):
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False):
            d = Direction.from_spherical(theta, phi)
            tubes.append(Tube(center=np.zeros(3), direction=d, length=1.0, delta=delta))
    return tubes


def make_constant_surrogate(bias):
    """对每个 state 输出都等于 ``bias`` 的代理（零权重）。"""
    model = SurrogateScorer(
        input_dim=len(FEATURE_NAMES), hidden_dims=[4], output_dim=3, seed=0
    )
    n_layers = len(model.params) // 2
    out_w_pid = 2 * (n_layers - 1)
    model.params[out_w_pid] = np.zeros_like(model.params[out_w_pid])
    model.params[out_w_pid + 1] = np.asarray(bias, dtype=float)
    return model


def make_state_with_grains(orch):
    state = orch.initialize()
    grains = orch._build_grains(state)
    return ProofState(
        tubes=state.tubes,
        shading=state.shading,
        grains=grains,
        delta=state.delta,
        rho=state.rho,
    )


def fixed_metrics(grains):
    """构造一组固定的指标（**条件检查会真的跑它**）。

    ⚠️ **2026-09-30 起条件检查先于打分器执行**（见
    ``ProofOrchestrator._scored_action_allowed``）。此前打分器的结论会**无
    条件返回**，所以这些测试用"随便什么指标"都能通过——它测的其实是
    ``_scores_to_action`` 这个纯映射，而不是"surrogate 选出的 Move 被采纳"。

    现在要让"surrogate 生效"，必须构造**目标 Move 的条件确实未满足**的状态。
    下面的取值经实测：``delta=0.1, rho=0.2`` 时
    ``check_move2_condition`` 为 False（即 MOVE2 值得做），
    ``check_move1_condition`` 为 True。
    """
    return ProofMetrics(
        delta=0.1,
        rho=0.2,
        a=0.1,
        b=0.1,
        c=0.5,
        local_error=2.0,
        volume_ratio=1.2,
        union_volume=1.0,
        lower_bound=1.0,
        n_tubes=9,
        n_grains=grains.count,
    )


class TestSurrogateMoveSelection:
    def test_high_confidence_selects_surrogate_argmax(self):
        tubes = make_fan(3, 3, delta=0.1)
        # move 2 clearly highest: scores (1, 9, 2), gap = 7 >= threshold.
        surrogate = make_constant_surrogate([1.0, 9.0, 2.0])
        # **必须让 MOVE2 真的值得做**（``check_move2_condition`` 为 False），
        # 否则守卫（``_scored_action_allowed``）会拦下它。
        #
        # 该条件只依赖 ``omega`` 与 ``delta``：``threshold = delta**(omega/100)``，
        # 默认 ``omega=0.1`` 时阈值 ≈ 0.998，几乎所有 prism 都满足 → 条件为 True。
        # 取 ``omega=50`` 时阈值 ≈ 0.316 < 实测 prisms 的最大 rho(0.328) → 不满足。
        #
        # 注：这不是为了迁就测试——守卫是对的（实测 surrogate 空转率
        # 86%→25%），而是原测试靠“打分器绕过条件”才能通过，如今前提变了。
        orch = ProofOrchestrator(
            tubes, max_iterations=1, grid_n=16, surrogate=surrogate,
            move_params={"omega": 50.0},
        )
        state = make_state_with_grains(orch)
        metrics = fixed_metrics(state.grains)
        action = orch._select_move(state, metrics)
        assert action == ProofAction.MOVE2_REPLACE_SQUARE_GRAINS
        assert any("move selected by surrogate" in n for n in orch._selection_notes)

    def test_step_surfaces_surrogate_note(self, monkeypatch):
        tubes = make_fan(3, 3, delta=0.1)
        surrogate = make_constant_surrogate([1.0, 9.0, 2.0])
        # 同上：``omega=50`` 使 MOVE2 的条件不满足（见前一测试的说明）。
        orch = ProofOrchestrator(
            tubes, max_iterations=1, grid_n=16, surrogate=surrogate,
            move_params={"omega": 50.0},
        )
        state = make_state_with_grains(orch)
        metrics = fixed_metrics(state.grains)
        monkeypatch.setattr(orch, "_compute_metrics", lambda s: metrics)
        monkeypatch.setattr(orch, "_try_nsp_factorization", lambda s: None)

        new_state, action, notes = orch.step(state)
        assert action == ProofAction.MOVE2_REPLACE_SQUARE_GRAINS
        assert any("move selected by surrogate (gap=" in n for n in notes)

    def test_low_confidence_falls_back_to_heuristic(self, monkeypatch):
        tubes = make_fan(3, 3, delta=0.1)
        # All scores equal: gap = 0 < threshold.
        low_conf = make_constant_surrogate([5.0, 5.0, 5.0])
        orch_low = ProofOrchestrator(
            tubes, max_iterations=1, grid_n=16, surrogate=low_conf
        )
        monkeypatch.setattr(
            ProofOrchestrator, "_load_surrogate", staticmethod(lambda path: None)
        )
        orch_plain = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)

        state = make_state_with_grains(orch_low)
        metrics = fixed_metrics(state.grains)
        action_low = orch_low._select_move(state, metrics)
        action_plain = orch_plain._select_move(state, metrics)
        assert action_low == action_plain
        assert not orch_low._selection_notes

    def test_missing_weights_file_falls_back_silently(self, tmp_path):
        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(
            tubes,
            max_iterations=1,
            grid_n=16,
            surrogate_weights_path=str(tmp_path / "does_not_exist.json"),
        )
        assert orch.surrogate is None
        state = make_state_with_grains(orch)
        metrics = fixed_metrics(state.grains)
        assert orch._select_move(state, metrics) in ProofAction

    def test_corrupt_weights_file_falls_back_silently(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        tubes = make_fan(3, 3, delta=0.1)
        orch = ProofOrchestrator(
            tubes,
            max_iterations=1,
            grid_n=16,
            surrogate_weights_path=str(bad),
        )
        assert orch.surrogate is None


class TestNormalizationRoundtrip:
    def test_save_load_preserves_normalization(self, tmp_path):
        n = len(FEATURE_NAMES)
        model = SurrogateScorer(input_dim=n, hidden_dims=[4], output_dim=3, seed=1)
        model.feature_mean = np.linspace(0.5, 2.0, n)
        model.feature_std = np.full(n, 0.5)
        path = str(tmp_path / "model.json")
        model.save(path)

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        assert "feature_mean" in payload and "feature_std" in payload

        loaded = SurrogateScorer.load(path)
        # ``feature_mean`` / ``feature_std`` 声明为 ``Optional``（未启用归一化
        # 时为 None）。本用例刚刚设过它们并落盘，故读回必非 None——断言把这层
        # 前提写明（也顺带验证"确实被 store 了"）。
        assert loaded.feature_mean is not None
        assert loaded.feature_std is not None
        np.testing.assert_allclose(loaded.feature_mean, model.feature_mean)
        np.testing.assert_allclose(loaded.feature_std, model.feature_std)

    def test_normalization_changes_predictions(self):
        n = len(FEATURE_NAMES)
        model = SurrogateScorer(input_dim=n, hidden_dims=[4], output_dim=3, seed=3)
        state = {"rho": 0.3, "delta": 0.02, "local_error": 1.5, "volume_ratio": 1.1}
        raw_scores = model.predict_scores(state)
        model.feature_mean = np.full(n, 1.0)
        model.feature_std = np.full(n, 0.2)
        norm_scores = model.predict_scores(state)
        assert norm_scores != raw_scores

    def test_old_format_loads_without_normalization(self, tmp_path):
        n = len(FEATURE_NAMES)
        model = SurrogateScorer(input_dim=n, hidden_dims=[4], output_dim=3, seed=2)
        path = str(tmp_path / "legacy.json")
        model.save(path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload.pop("feature_mean", None)
        payload.pop("feature_std", None)
        # 旧格式同样没有 feature_names——加载时必须补回历史前 n 个特征名，
        # 而不是报错或按行数猜。
        payload.pop("feature_names", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        loaded = SurrogateScorer.load(path)
        assert loaded.feature_mean is None
        assert loaded.feature_std is None
        assert list(loaded.feature_names) == list(FEATURE_NAMES[:n])
        state = {"rho": 0.3, "delta": 0.02, "local_error": 1.5, "volume_ratio": 1.1}
        np.testing.assert_allclose(
            list(loaded.predict_scores(state).values()),
            list(model.predict_scores(state).values()),
        )

    def test_inconsistent_feature_names_rejected(self, tmp_path):
        """权重文件自相矛盾（input_dim 与 feature_names 长度不符）必须拒绝。

        静默按行数截断会让推理读到错位的特征，且不会有任何报错——这正是
        扩维最容易造成的“静默错误”，所以要在加载处拦住。
        """
        model = SurrogateScorer(
            input_dim=len(FEATURE_NAMES), hidden_dims=[4], output_dim=3, seed=2
        )
        path = str(tmp_path / "bad_names.json")
        model.save(path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["feature_names"] = payload["feature_names"][:2]  # 与 input_dim 不符
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

        with pytest.raises(ValueError, match="自相矛盾"):
            SurrogateScorer.load(path)


class TestProofStepsToSamples:
    def _metrics(self):
        return ProofMetrics(
            delta=0.05, rho=0.2, a=0.1, b=0.1, c=0.6,
            local_error=1.5, volume_ratio=1.3, union_volume=1.0,
            lower_bound=1.0, n_tubes=9, n_grains=3,
        )

    def test_labels_and_shape(self):
        m = self._metrics()
        steps = [
            ProofStep(0, ProofAction.MOVE2_REPLACE_SQUARE_GRAINS, m, m, []),
            ProofStep(1, ProofAction.RECURSION, m, m, []),
            ProofStep(2, ProofAction.MOVE3_WIDER_GRAINS, m, m, []),
        ]
        samples = proof_steps_to_samples(steps)
        assert len(samples) == 2

        state, target = samples[0]
        # state 由 ``state_from_metrics`` 构造，含全部 FEATURE_NAMES 键（
        # 扩维后硬编码 4 个键的断言会失效）。
        assert set(state.keys()) == set(FEATURE_NAMES)
        assert state["rho"] == pytest.approx(0.2)
        # 扩展特征确实从 metrics 取到了值，不是默认值：
        assert state["log_n_tubes"] == pytest.approx(np.log1p(9))
        assert state["n_grains_norm"] == pytest.approx(3 / 9)
        assert target == {1: 5.0, 2: 10.0, 3: 5.0}

        _, target3 = samples[1]
        assert target3 == {1: 5.0, 2: 5.0, 3: 10.0}

    def test_no_move_steps_gives_empty(self):
        m = self._metrics()
        steps = [ProofStep(0, ProofAction.GRAINS_DECOMPOSITION, m, m, [])]
        assert proof_steps_to_samples(steps) == []


class TestTraceStepsToSamples:
    """**从 .kpt 回流训练样本**——本条链路此前静默失效。

    两个失效点（都不报错，只是"训练照跑、样本为空或特征全默认"）：

    1. ``step.get("action").name`` —— ``read_trace`` 的 step 是 **dict**，
       ``action`` 已经是字符串，取 ``.name`` 抛 ``AttributeError`` 被
       ``getattr`` 的默认值吞掉 ⇒ move_id 恒为 None ⇒ **0 个样本**。
    2. ``state_from_metrics(dict)`` —— 逐字段 ``getattr(dict, "rho", 0.5)``
       一律落到默认值 ⇒ 八个特征全变成常量 ⇒ 模型学到一条水平线。
       而且 ``.kpt`` 的浮点是 ``float.hex()`` 文本，需要显式解码。

    下面每条都同时验证"该抓到的一定要抓到"（第一组）与"合法的必须放行"
    （第二组）。
    """

    def _trace_before(self):
        """一个与 ``iter_steps`` 产物同构的 step（含十六进制浮点）。"""
        return {
            "action": "MOVE3_WIDER_GRAINS",
            "before": {
                "delta": "0x1.47ae147ae147bp-5",       # 0.04
                "rho": "0x1.e7d7f1bd21146p-2",         # ~0.4764
                "a": "0x1.f6acb4c41a47dp-1",
                "c": "0x1.16cc1803aeecbp+1",
                "local_error": "0x1.4e4905c916446p+0", # ~1.3058
                "volume_ratio": "0x1.55b9ecfdc2213p+0",# ~1.3349
                "n_tubes": 630,
                "n_grains": 307,
                "is_sticky": False,
            },
        }

    # ---- 第一组：必须抓到的错误输入 ----

    def test_hex_strings_are_decoded_not_defaulted(self):
        """**注入验证**：十六进制文本若不解码，特征会退化成默认值。

        判据取 ``delta``：默认值是 0.01，真值是 0.04。只断言"不为默认值"
        还不够——那样把默认值改成别的数也能骗过测试；这里直接对真值。
        """
        samples = proof_steps_to_samples([self._trace_before()], strict=True)
        assert len(samples) == 1, "dict 载体被静默丢弃（action 取不到名字）"

        state, target = samples[0]
        assert state["delta"] == pytest.approx(0.04)
        assert state["rho"] == pytest.approx(float.fromhex("0x1.e7d7f1bd21146p-2"))
        assert state["local_error"] == pytest.approx(float.fromhex("0x1.4e4905c916446p+0"))
        assert state["log_n_tubes"] == pytest.approx(np.log1p(630))
        assert state["n_grains_norm"] == pytest.approx(307 / 630)
        assert target == {1: 5.0, 2: 5.0, 3: 10.0}

    def test_state_is_not_all_defaults(self):
        """特征**整体**不能等于默认表 ⇒ 训练集无信息，必须视为失败。

        这条是上一轮的**根因断言**：故障时八个特征恰好全都是
        ``_DEFAULT_FEATURE`` 的值，所以拿默认表对一遍最灵敏。

        但**不能要求逐个都不同**：``sticky=0.0`` 在 ``is_sticky=False`` 时
        既等于默认值也是真值，单看它分不清"解码成功"与"取了默认"。所以
        判据取"等于默认值的特征数 < 总数"——故障时是 8/8，正常时只有
        ``sticky`` 这一个可能撞上。
        """
        samples = proof_steps_to_samples([self._trace_before()], strict=True)
        state = samples[0][0]
        same_as_default = [
            name for name in FEATURE_NAMES
            if state[name] == pytest.approx(_DEFAULT_FEATURE[name])
        ]
        assert len(same_as_default) < len(FEATURE_NAMES), (
            f"全部 {len(FEATURE_NAMES)} 个特征都是默认值（载体没被识别）"
        )
        # 且这些"撞上默认值"的特征必须本来就可能等于默认值——
        # 出现 delta/rho 之类的就说明解码真的退化了。
        assert set(same_as_default) <= {"sticky"}, (
            f"不该等于默认值的特征退化了: {sorted(set(same_as_default) - {'sticky'})}"
        )

    def test_strict_rejects_wrong_carrier(self):
        """载体传错（如空 dict / 字段名写错）在 strict 下必须抛错。"""
        with pytest.raises(ValueError, match="缺少指标字段"):
            proof_steps_to_samples(
                [{"action": "MOVE1_LONGER_GRAINS", "before": {}}], strict=True
            )

    def test_strict_rejects_before_with_wrong_names(self):
        """只有部分字段名对：strict 仍须报错，不能"部分默认"地放行。"""
        with pytest.raises(ValueError, match="缺少指标字段"):
            proof_steps_to_samples(
                [{"action": "MOVE1_LONGER_GRAINS", "before": {"delta": 0.04}}],
                strict=True,
            )

    # ---- 第二组：合法的必须放行 ----

    def test_non_move_steps_still_skipped(self):
        """非 Move 步骤照旧跳过（不改既有语义）。"""
        step = self._trace_before()
        step["action"] = "GRAINS_DECOMPOSITION"
        assert proof_steps_to_samples([step], strict=True) == []

    def test_loose_mode_tolerates_missing_fields(self):
        """非 strict（热路径）保持宽松：缺字段给中性值而非抛错。

        与 ``_select_move`` 的取舍一致——指标不全时"给中性值继续跑"比
        中断整轮搜索更可接受。
        """
        samples = proof_steps_to_samples(
            [{"action": "MOVE1_LONGER_GRAINS", "before": {}}], strict=False
        )
        assert len(samples) == 1
        assert samples[0][0]["delta"] == pytest.approx(_DEFAULT_FEATURE["delta"])

    def test_object_carrier_unaffected(self):
        """对象载体（正常主循环产物）行为不变——回归守门。"""
        m = ProofMetrics(
            delta=0.05, rho=0.2, a=0.1, b=0.1, c=0.6,
            local_error=1.5, volume_ratio=1.3, union_volume=1.0,
            lower_bound=1.0, n_tubes=9, n_grains=3,
        )
        steps = [ProofStep(0, ProofAction.MOVE2_REPLACE_SQUARE_GRAINS, m, m, [])]
        samples = proof_steps_to_samples(steps, strict=True)
        assert len(samples) == 1
        assert samples[0][0]["rho"] == pytest.approx(0.2)
        assert samples[0][1] == {1: 5.0, 2: 10.0, 3: 5.0}
