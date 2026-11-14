"""``scope`` 作用域（L1 数学闭合）的端到端验收。

**这组测试要证明的事**

修复前 ``success=True`` 混合了两种含义：
(a) 下界对**原输入管族**成立； (b) 只对 broadness 剥离**之后**的子集成立。
两者都返回 ``True``，而下游只能看到那个布尔 ⇒ 把 (b) 读成 (a)。

本组测试用**真实的 ``run()``**（不是手工构造）钉住三件事：

1. **有剥离时必须不成功**：宽簇+窄簇混合 → ``scope=BROAD_ONLY``、``success=False``，
   且 ``uncovered_narrow_tubes > 0`` 与之一致；
2. **无剥离时才可以成功**：``scope=FULL`` ⇒ 这是唯一可读作"证明成功"的取值；
3. **作用域随 trace 落盘**：``peek_trace`` 读得到 ``scope``——否则复核方
   打开文件只看到 ``success``，会得出错误结论。

**为什么必须用真实 ``run()`` 而不是构造 ``ProofResult``**：手工构造只能验证
"不变量检查生效"，验证不了"``run()`` 有没有正确计算 scope"。后者才是这次要
修的东西（``_scope_fields`` / ``_effective_success`` 的接线）。
"""

from __future__ import annotations

import numpy as np
import pytest

from kakeya_ai.errors import ContractViolationError
from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.proof_orchestrator import (
    ProofOrchestrator,
    ProofResult,
    ProofScope,
)
from kakeya_ai.trace_format import peek_trace, write_trace

#: 与项目其它性能相关测试一致：grid_n=8 是"够用且快"的档位。
GRID_N = 8


def _make_orchestrator(tubes, **kw) -> ProofOrchestrator:
    """构造不含 surrogate 的编排器（隔离变量：本组测的是 scope，不是 AI）。"""
    kw.setdefault("surrogate_weights_path", "/nonexistent/weights.json")
    return ProofOrchestrator(tubes, grid_n=GRID_N, **kw)


def _fan(k: int, delta: float = 0.1) -> list[Tube]:
    """``k×k`` 扇形：方向集中，**会触发 broadness 剥离**。"""
    return [
        Tube(
            center=np.array([0.0, 0.0, 0.0]),
            direction=Direction.from_spherical(np.pi / 2 * 0.3 + 0.01 * i, 2 * np.pi * j / k),
            length=1.0,
            delta=delta,
        )
        for i in range(k)
        for j in range(k)
    ]


def _parallel(n: int, delta: float = 0.1) -> list[Tube]:
    """``n`` 根平行扇：方向均匀铺开，**不触发剥离**。"""
    return [
        Tube(
            center=np.array([0.0, 0.0, 0.0]),
            direction=Direction.from_spherical(np.pi / 2, 2 * np.pi * i / n),
            length=1.0,
            delta=delta,
        )
        for i in range(n)
    ]


class TestScopeFromRealRun:
    """用真实 ``run()`` 验证 scope 的取值。"""

    def test_induction_actually_covers_this_configuration(self):
        """**承接必须真的起效**——`fan(3×3)` 的窄簇要被全部接住。

        **为什么单列一条**（2026-09-30）：先写的是上面那条"两种情形都允许"
        的测试，结果**变异验证没能捕获** bug —— 把 `ok` 恒置 `False`
        （假装承接永远失败），测试**照样全绿**。因为双分支断言对
        "承接有没有起作用"**没有判别力**。

        这条补上判别力：本构型实测 3/3 批都能承接，故**要求**
        `uncovered == 0` + `scope is FULL` + 承接报告写明"均已被承接"。

        **不要断言 `success`**（2026-09-30 修正）：`success` 由
        `volume_ratio >= 1.0` 决定，而那个判据的参照系刚被修正（分子分母
        统一取原输入），此后该构型的比值不再虚高（0.343 → 0.558，
        **不再 ≥1**）。所以"承接成功"与"整体 success"是**两件事**：
        前者说"窄簇各自都证明了"，后者说"broad 子集也达标了"。
        把关卡压在 `success` 上会让这条测试变成"依赖判据口径"的脆弱断言。

        代价：若构型或判据演进导致该构型不再能承接，本测试需**重新评估
        目标值**，而不是删掉它（删掉就等于把判别力又丢了）。
        """
        result = _make_orchestrator(_fan(3), max_iterations=8).run()

        assert result.steps and result.steps[0].action.name == "BROADNESS_REDUCTION", (
            "前提：该构型第一步应是剥离"
        )
        assert result.uncovered_narrow_tubes == 0, (
            f"承接未起效：仍有 {result.uncovered_narrow_tubes} 根未覆盖。"
            f"承接报告：{result.narrow_coverage}"
        )
        assert result.scope is ProofScope.FULL, (
            f"全部承接后 scope 应为 FULL（作用域回到全部输入），"
            f"实际 {result.scope.name}"
        )
        assert "均已被承接" in result.narrow_coverage

    def test_stripped_input_is_never_silently_reported_as_success(self):
        """**本组的核心断言**：有剥离时，**必须**说明它到底被承接了没有。

        2026-09-30 实现"小帽上的归纳"后，这里拆成**两条互补的验收**——
        因为"有剥离"现在对应**两种**合法结果，而**非法的是第三种**
        （有未被承接的窄簇、却报 ``FULL``/``success=True``）：

        | 情形 | 期望 |
        | --- | --- |
        | 剥离的批次**全部承接** | ``scope=FULL``，可以 ``success=True`` |
        | 剥离的批次**未承接**（禁用承接构造） | ``scope=BROAD_ONLY``，``success=False`` |
        | ← **非法** | 未承接却报 ``FULL`` |

        修复前没有第三种以外的可能（承接不存在，一律 BROAD_ONLY），
        故原断言是"必须 BROAD_ONLY"；现在那条断言会**误判改进为回归**。
        """
        tubes = _fan(3)

        # —— 情形 1：承接启用（默认）——
        run1 = _make_orchestrator(tubes, max_iterations=8).run()
        assert run1.steps and run1.steps[0].action.name == "BROADNESS_REDUCTION", (
            "本用例的前提：该构型必须触发剥离"
        )
        if run1.uncovered_narrow_tubes == 0:
            # 全部承接 ⇒ 下界确实覆盖原输入。
            assert run1.scope is ProofScope.FULL
            assert "均已被承接" in run1.narrow_coverage
        else:
            # 未全部承接 ⇒ 不得声称覆盖原输入。
            assert run1.scope is ProofScope.BROAD_ONLY
            assert run1.success is False

        # —— 情形 2：**禁用承接**——必须落到 BROAD_ONLY。 ——
        from kakeya_ai.proof_orchestrator import MAX_INDUCTION_DEPTH

        run2 = _make_orchestrator(
            tubes, max_iterations=8, _induction_depth=MAX_INDUCTION_DEPTH
        ).run()
        assert run2.uncovered_narrow_tubes > 0, (
            "禁用承接后，剥离的管必须记为未覆盖（否则说明剥离本身没发生）"
        )
        assert run2.scope is ProofScope.BROAD_ONLY, (
            f"有 {run2.uncovered_narrow_tubes} 根未承接，"
            f"却报了 scope={run2.scope.name}——这正是 L1 要防的"
        )
        assert run2.success is False

    def test_scope_and_uncovered_field_are_consistent(self):
        """``scope`` 与 ``uncovered_narrow_tubes`` 必须说同一件事。

        两处记录同一事实却不一致，是"文档自相矛盾"在代码里的形态。
        """
        result = _make_orchestrator(_fan(4), max_iterations=8).run()
        has_gap = result.uncovered_narrow_tubes > 0
        assert has_gap == (result.scope is ProofScope.BROAD_ONLY), (
            f"uncovered={result.uncovered_narrow_tubes} 而 scope={result.scope.name}"
        )

    def test_no_gap_means_full_scope_allows_success(self):
        """反向验证：**没有**剥离时 scope 是 FULL，success 才可能为真。

        这条防"把 success 一律改成 False 就完事"的偷懒修法——那样虽然
        不会误报成功，但也永远证明不了任何东西。
        """
        result = _make_orchestrator(_parallel(8), max_iterations=8).run()
        assert result.uncovered_narrow_tubes == 0, (
            "本用例的前提：该构型不该触发剥离"
        )
        assert result.scope in (ProofScope.FULL, ProofScope.NOT_APPLICABLE), (
            f"无剥离时 scope 不该是 {result.scope.name}"
        )
        if result.success:
            # 只有 FULL 才配 success —— 不变量本身。
            assert result.scope is ProofScope.FULL

    def test_empty_input_is_not_applicable_not_full(self):
        """空输入是 ``NOT_APPLICABLE``：字面上"覆盖了全部管"，但没什么可证。"""
        result = _make_orchestrator([], max_iterations=4).run()
        assert result.scope is ProofScope.NOT_APPLICABLE
        assert result.success is False


class TestScopeSurvivesTraceRoundTrip:
    """作用域必须**随 trace 落盘**——否则文件里的结论丢掉限定条件。"""

    def test_scope_is_written_and_readable(self, tmp_path):
        # 用**禁用承接**的构型：此时 scope 必为 BROAD_ONLY，
        # 能同时验证 "scope 落盘" 与 "uncovered 落盘" 两个字段。
        from kakeya_ai.proof_orchestrator import MAX_INDUCTION_DEPTH

        result = _make_orchestrator(
            _fan(3), max_iterations=8, _induction_depth=MAX_INDUCTION_DEPTH
        ).run()
        path = tmp_path / "t.kpt"
        write_trace(result, str(path))

        header = peek_trace(str(path))
        assert header.scope == ProofScope.BROAD_ONLY.name
        assert header.success is False
        assert header.uncovered_narrow_tubes == result.uncovered_narrow_tubes

    def test_scope_appears_in_raw_header_text(self, tmp_path):
        """**原文里** ``scope`` 必须紧跟 ``success``。

        接口层读得到还不够：复核方可能直接 ``head`` 文件。
        若 ``scope`` 被埋在十几行之后，等于没声明。
        """
        result = _make_orchestrator(_fan(3), max_iterations=8).run()
        path = tmp_path / "t.kpt"
        write_trace(result, str(path))

        text = path.read_text(encoding="utf-8")
        # 头部 JSON 在第一行（``KPT1`` 之后的第二行）。
        lines = text.splitlines()
        json_line = next(ln for ln in lines if ln.strip().startswith("{"))
        i_success = json_line.index('"success"')
        i_scope = json_line.index('"scope"')
        assert i_scope > i_success, "scope 必须在 success 之后"
        # 中间不许隔太多字段——紧邻才有效。
        between = json_line[i_success:i_scope]
        assert between.count(",") <= 2, (
            f"scope 与 success 相隔太远（中间 {between.count(',')} 个字段）："
            "读者扫一眼头部看不出来"
        )

    def test_old_style_object_without_scope_writes_empty(self):
        """缺 ``scope`` 属性的对象（旧鸭子类型）写 trace 时不得抛错。

        写 trace 失败会让一次真实的证明搜索白跑——所以这里必须降级为空串，
        与 ``failure_reason`` 的兼容口径一致。
        """

        class _Duck:
            steps: list = []
            success = False
            exhausted = False
            message = ""
            initial_tubes: list = []
            final_metrics = None

        from kakeya_ai.trace_format import _proof_scope_name

        assert _proof_scope_name(_Duck()) == ""


class TestScopeInvariantsOnManualConstruction:
    """手工构造也要遵守不变量（把"写错"变成"构造失败"）。"""

    def _metrics(self):
        from kakeya_ai.proof_orchestrator import ProofMetrics

        return ProofMetrics(
            delta=0.1, rho=0.1, a=0.1, b=0.1, c=0.1, local_error=0.0,
            volume_ratio=1.0, union_volume=1.0, lower_bound=1.0,
            n_tubes=1, n_grains=0,
        )

    def test_success_requires_full_scope(self):
        with pytest.raises(ContractViolationError, match="scope"):
            ProofResult(
                success=True, steps=[], final_state=None,  # type: ignore[arg-type]
                final_metrics=self._metrics(), message="manual",
            )

    def test_broad_only_requires_a_gap(self):
        """``BROAD_ONLY`` 却没有未覆盖窄簇 = 自相矛盾。"""
        with pytest.raises(ContractViolationError, match="uncovered_narrow_tubes"):
            ProofResult(
                success=False, steps=[], final_state=None,  # type: ignore[arg-type]
                final_metrics=self._metrics(), message="manual",
                scope=ProofScope.FULL,
                uncovered_narrow_tubes=2,
            )

    def test_consistent_construction_is_accepted(self):
        """合法组合必须放行——否则不变量就成了"谁都过不去"。"""
        ok_full = ProofResult(
            success=True, steps=[], final_state=None,  # type: ignore[arg-type]
            final_metrics=self._metrics(), message="m",
            scope=ProofScope.FULL,
        )
        assert ok_full.success is True

        ok_broad = ProofResult(
            success=False, steps=[], final_state=None,  # type: ignore[arg-type]
            final_metrics=self._metrics(), message="m",
            scope=ProofScope.BROAD_ONLY,
            uncovered_narrow_tubes=5,
        )
        assert ok_broad.success is False


class TestInductionSeedHandling:
    """承接窄簇时的 ``seed`` 处置（2026-09-30 报错后补的守门）。

    **这个 bug 为什么能潜伏**：``_induct_on_narrow_batches`` 里写了
    ``seed=self.seed + 1000 + i``，而 ``self.seed`` 是 ``Optional[int]``
    ——``None``（**刻意的非确定性探索模式**）走这条路径直接报

        TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'

    此前一直没暴露，因为承接循环**只在真有窄簇被剥离时**才执行：
    测过的构型要么无剥离（``coverage.total == 0`` 提前返回），要么在
    深度上限处返回。**测试全绿而代码是坏的。**

    ⇒ 教训：**"只有特定构型才走到"的代码路径，测试必须显式构造那个构型。**
    """

    def test_induction_with_none_seed_does_not_crash(self):
        """``seed=None`` 时承接窄簇不得崩，且承接**真的被执行**。

        **必须断言"承接真的跑了"**（而不只是"没抛异常"）：若构型无剥离，
        承接循环一次都不进——那样这条测试就退化成"什么都没测"，
        正是这个 bug 潜伏至今的原因。
        """
        result = _make_orchestrator(
            _fan(3), max_iterations=6, seed=None
        ).run()

        # 前提守卫：确有剥离，否则测不到承接路径。
        assert result.steps and result.steps[0].action.name == "BROADNESS_REDUCTION", (
            "前提失效：该构型应触发 broadness 剥离（否则承接循环不执行，"
            "本测试测不到 seed 处置）"
        )
        assert result.narrow_coverage, (
            "承接报告为空 —— 说明承接循环没跑，本测试没测到目标路径"
        )
        # 不断言 success：非确定模式本就不保证路径（且判据参照系启用后，
        # 该构型的比值不再够线）。这里只钉"不崩 + 路径真的走到"。
        assert result.scope in (ProofScope.FULL, ProofScope.BROAD_ONLY)

    def test_induction_with_int_seed_still_works(self):
        """整数 seed 仍是确定性的（回归：别为了修 ``None`` 把正常路径改坏）。"""
        run1 = _make_orchestrator(_fan(3), max_iterations=6, seed=0).run()
        run2 = _make_orchestrator(_fan(3), max_iterations=6, seed=0).run()
        assert run1.success == run2.success
        assert len(run1.steps) == len(run2.steps)
        assert [s.action.name for s in run1.steps] == [
            s.action.name for s in run2.steps
        ], "同 seed 同输入的动作序列必须逐位一致"
