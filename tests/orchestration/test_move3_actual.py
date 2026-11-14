"""Move #3 'actual' 误差模式与管-盒裁剪的测试。"""

import numpy as np
import pytest

from kakeya_ai.grains import Prism, guth_grains_skeleton
from kakeya_ai.moves import (
    actual_local_kt_errors,
    check_move3_condition,
    move3_wider_grains,
    tube_segment_in_box,
)
from kakeya_ai.primitives import Direction, Shading, Tube
from kakeya_ai.proof_orchestrator import ProofOrchestrator, run_proof_skeleton


def make_tube_set(n=48, delta=0.1, seed=7):
    rng = np.random.default_rng(seed)
    tubes = []
    for _ in range(n):
        theta = rng.uniform(0.4, np.pi - 0.4)
        phi = rng.uniform(0, 2 * np.pi)
        d = Direction.from_spherical(theta, phi)
        center = np.array([rng.uniform(-0.3, 0.3) for _ in range(3)])
        tubes.append(Tube(center=center, direction=d, length=1.0, delta=delta))
    return tubes


def make_grains(tubes):
    shading = Shading(np.ones(len(tubes)))
    return guth_grains_skeleton(
        tubes, shading, grid_n=2, adaptive_rho=True, min_tubes_per_bin=4
    )


def axis_aligned_prisms(grains):
    """把 ``grains.prisms`` 收窄成 ``list[Prism]``（只保留轴对齐 grain）。

    ``prisms`` 的元素类型是 ``Prism | OrientedPrism``——同一个容器既装轴对齐也装
    定向 grain（见 ``grains.py``）。下面几个度量函数只吃 ``Sequence[Prism]``，故这里
    显式收窄；顺带断言"确实是轴对齐"，免得定向 grain 被**悄悄丢掉**后测出一组
    貌似正常的数字（那比报错更糟）。
    """
    assert grains.is_axis_aligned, "本用例假定 grain 全部轴对齐"
    return [p for p in grains.prisms if isinstance(p, Prism)]


class TestTubeSegmentInBox:
    box = np.array([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]])
    tube = Tube(
        center=np.zeros(3),
        direction=Direction.from_array([0.0, 0.0, 1.0]),
        length=1.0,
        delta=0.1,
    )

    def test_fully_inside_returns_equal_tube(self):
        clipped = tube_segment_in_box(self.tube, self.box)
        assert clipped is not None
        assert clipped.length == pytest.approx(1.0)
        assert clipped.delta == pytest.approx(self.tube.delta)
        np.testing.assert_allclose(clipped.direction.v, self.tube.direction.v)
        np.testing.assert_allclose(clipped.center, self.tube.center)

    def test_partially_inside_returns_shorter_tube(self):
        # Only the half 0 <= z <= 0.5 is inside the box.
        box = np.array([[-2.0, 2.0], [-2.0, 2.0], [0.0, 0.5]])
        clipped = tube_segment_in_box(self.tube, box)
        assert clipped is not None
        assert clipped.length == pytest.approx(0.5)
        np.testing.assert_allclose(clipped.center, [0.0, 0.0, 0.25])

    def test_fully_outside_returns_none(self):
        box = np.array([[-2.0, 2.0], [-2.0, 2.0], [2.0, 3.0]])
        assert tube_segment_in_box(self.tube, box) is None

    def test_tiny_clip_returns_none(self):
        # 短于 delta 的交集被丢弃。
        box = np.array([[-2.0, 2.0], [-2.0, 2.0], [0.5, 0.52]])
        assert tube_segment_in_box(self.tube, box) is None

    def test_parallel_outside_returns_none(self):
        # 方向沿 x、盒子在 y 上平移：与平行板错过。
        tube = Tube(
            center=np.zeros(3),
            direction=Direction.from_array([1.0, 0.0, 0.0]),
            length=1.0,
            delta=0.1,
        )
        box = np.array([[-2.0, 2.0], [1.0, 2.0], [-2.0, 2.0]])
        assert tube_segment_in_box(tube, box) is None


class TestActualLocalKtErrors:
    def test_errors_match_tube_counts(self):
        tubes = make_tube_set(n=16, seed=3)
        grains = make_grains(tubes)
        errors = actual_local_kt_errors(tubes, axis_aligned_prisms(grains), grid_n=16)
        assert len(errors) == grains.count
        # 每个误差都是有限且非负的。
        assert all(np.isfinite(e) and e >= 0.0 for e in errors)
        # 所有管都与盒错过的 grain 报告 0。
        for p, e in zip(grains.prisms, errors, strict=True):
            if len(p.tube_indices) == 0:
                assert e == 0.0


class TestMove3ActualMode:
    def test_invalid_error_mode_raises(self):
        tubes = make_tube_set(n=8, seed=1)
        grains = make_grains(tubes)
        with pytest.raises(ValueError):
            move3_wider_grains(grains, zeta=0.01, delta=0.1, error_mode="bogus")

    def test_actual_mode_requires_tubes(self):
        tubes = make_tube_set(n=8, seed=1)
        grains = make_grains(tubes)
        with pytest.raises(ValueError):
            move3_wider_grains(
                grains, zeta=0.01, delta=0.1, error_mode="actual", tubes=None
            )

    def test_actual_mode_converges_and_beats_proxy(self):
        tubes = make_tube_set(n=48, seed=7)
        grains = make_grains(tubes)
        # delta**(-zeta) = 0.1**(-0.01) ~ 1.023 低于 grid_n=16 体素估计的
        # 离散化下限（~1.12），故用一个显式且略大的上界来验证收敛。
        target = 1.5

        actual_out = move3_wider_grains(
            grains, zeta=0.01, delta=0.1, max_local_error=target,
            tubes=tubes, error_mode="actual", grid_n=16,
        )
        # 存储的误差就是重新计算得到的 actual 统计量。
        recomputed = actual_local_kt_errors(
            tubes, axis_aligned_prisms(actual_out), grid_n=16,
        )
        for p, e in zip(actual_out.prisms, recomputed, strict=True):
            assert p.local_kt_error == pytest.approx(e, rel=1e-6)
        assert check_move3_condition(actual_out, target)

        # 同一配置下的 proxy 模式达不到该目标：其合并停滞，最大误差
        # ~4.7 > 1.5。actual 模式不差于它。
        proxy_out = move3_wider_grains(
            grains, zeta=0.01, delta=0.1, max_local_error=target
        )
        assert not check_move3_condition(proxy_out, target)
        assert actual_out.max_local_kt_error < proxy_out.max_local_kt_error

    def test_proxy_mode_default_unchanged(self):
        # 显式 proxy 模式必须复现历史签名行为。
        tubes = make_tube_set(n=24, seed=5)
        grains = make_grains(tubes)
        out = move3_wider_grains(grains, zeta=0.01, delta=0.1, max_local_error=100.0)
        assert out.count == grains.count
        assert out.max_local_kt_error == pytest.approx(grains.max_local_kt_error)


class TestOrchestratorErrorModeDefault:
    """缺口一（2026-10-01）：编排器的 Move #3 后条件口径默认是 ``actual``。

    为什么必须有这条守门：``actual`` 与 ``proxy`` 实测**每次都给出相反结论**
    （内部分批测量脚本，8/8 构型；proxy 偏悲观）。
    默认值一改动，所有 ``run()`` 的结论含义就变了——而它是**纯默认值**，
    没有任何测试断言过，极易被无意改回。故用结构断言钉住。
    """

    def test_orchestrator_default_error_mode_is_actual(self):
        tubes = make_tube_set(n=16, seed=3)
        orch = ProofOrchestrator(tubes)
        assert orch.error_mode == "actual", (
            "编排器 Move #3 后条件口径默认应为 actual（缺口一：proxy 不是"
            "保守上界，而是系统性偏悲观）。若要改回，请先更新已知局限登记（#21）"
            "并说明理由。"
        )

    def test_explicit_proxy_override_still_works(self):
        # 逃生阀：显式传 proxy 必须生效（向后兼容旧结果）。
        tubes = make_tube_set(n=16, seed=3)
        orch = ProofOrchestrator(tubes, move_params={"error_mode": "proxy"})
        assert orch.error_mode == "proxy"

    def test_apply_move_propagates_orchestrator_grid_n(self):
        """``_apply_move`` 必须把 ``self.grid_n`` 透传给 Move #3 实现。

        不传会静默用函数默认值 32，而 ``self.grid_n`` 被域抬升改到 64 时两者
        不符——actual 误差的数值依赖体素分辨率，故不能各用各的。
        用 AST 判据（而非子串），因为"名字出现"与"真的作为关键字传参"不同。
        """
        import ast
        import inspect
        import textwrap

        src = textwrap.dedent(inspect.getsource(ProofOrchestrator._apply_move))
        tree = ast.parse(src)
        calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "_move3_impl"
        ]
        assert calls, "_apply_move 里找不到 _move3_impl 调用"
        kwarg_names = {kw.arg for c in calls for kw in c.keywords}
        assert "grid_n" in kwarg_names, (
            "move3_wider_grains 必须显式传 grid_n=self.grid_n；"
            "否则 actual 模式静默用默认分辨率 32，与运行时 grid_n 不符。"
        )

    def test_move3_step_records_error_mode_caliber(self):
        """Move #3 步骤的 notes 必须写明口径——否则 trace 无从分辨 satisfied
        是对着哪个量判的（缺口一的核心）。"""
        tubes = make_tube_set(n=16, seed=7)
        result = run_proof_skeleton(
            tubes, seed=42, max_iterations=8, grid_n=16,
            move_params={"error_mode": "actual"},
        )
        m3_steps = [
            s for s in result.steps
            if s.action.name == "MOVE3_WIDER_GRAINS"
        ]
        # 前提守卫：本构型必须真的走到了 Move #3，否则断言是空转的。
        assert m3_steps, "该构型未走到 Move #3，本用例无判别力（换构型或改参数）"
        for s in m3_steps:
            joined = " ".join(s.notes)
            assert "error_mode=actual" in joined, (
                f"Move #3 步未记录口径；notes={s.notes}"
            )

