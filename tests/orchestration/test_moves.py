"""moves.py 的测试。"""

import numpy as np
import pytest

from kakeya_ai.grains import GrainsDecomposition, Prism, guth_grains_skeleton
from kakeya_ai.moves import (
    _long_axis,
    _transverse_axes,
    apply_moves,
    check_move1_condition,
    check_move1_condition_detailed,
    check_move2_condition,
    check_move3_condition,
    move1_longer_grains,
    move2_replace_square_grains,
    move3_target_rho,
    move3_wider_grains,
    regularize_shading,
)
from kakeya_ai.primitives import Direction, Tube


def make_parallel_tubes(n: int = 8):
    tubes = []
    for y in np.linspace(-0.4, 0.4, n):
        tubes.append(
            Tube(
                center=np.array([0.0, y, 0.0]),
                direction=Direction.from_array([0.0, 0.0, 1.0]),
                length=1.0,
                delta=0.05,
            )
        )
    return tubes


def test_move1_increases_c():
    tubes = make_parallel_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    old_max_c = max(p.c for p in grains.prisms)
    new_grains = move1_longer_grains(grains, rho=0.1, delta=0.05, zeta=0.01, tube_ratio=0.5)
    new_max_c = max(p.c for p in new_grains.prisms)
    assert new_max_c >= old_max_c


def test_move2_reduces_rho():
    tubes = make_parallel_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    old_max_rho = grains.max_rho
    new_grains = move2_replace_square_grains(grains, omega=0.1, delta=0.05)
    new_max_rho = new_grains.max_rho
    assert new_max_rho <= old_max_rho


def test_move2_post_condition():
    tubes = make_parallel_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    omega = 0.1
    delta = 0.05
    new_grains = move2_replace_square_grains(grains, omega=omega, delta=delta)
    assert check_move2_condition(new_grains, omega=omega, delta=delta)


def test_move3_widens():
    tubes = make_parallel_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    new_grains = move3_wider_grains(grains, zeta=0.01, delta=0.05)
    # Move 3 横向合并；grain 计数不应增加。
    assert new_grains.count <= grains.count


def test_move3_reduces_local_error():
    tubes = make_parallel_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    new_grains = move3_wider_grains(grains, zeta=0.01, delta=0.05)
    assert new_grains.max_local_kt_error <= grains.max_local_kt_error


def test_move3_post_condition():
    tubes = make_parallel_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    target_error = 1000.0  # 给骨架用的宽松目标
    new_grains = move3_wider_grains(grains, zeta=0.01, max_local_error=target_error)
    assert check_move3_condition(new_grains, target_error)


def test_move1_condition():
    tubes = make_parallel_tubes(4)
    grains = guth_grains_skeleton(tubes, None, grid_n=2)
    # 非常小的目标应当被满足。
    assert check_move1_condition(grains, rho=0.1, delta=0.05, zeta=0.01, tube_ratio=0.01)


def test_move1_satisfies_post_condition_for_parallel_tubes():
    """Move #1 对现实的平行管配置应当满足命题 7.5 第 (iv) 项的界，
    这得益于方向对齐的 grain。
    """
    tubes = make_parallel_tubes(16)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    rho, delta, zeta, tube_ratio = 0.1, 0.05, 0.01, 0.5
    new_grains = move1_longer_grains(grains, rho, delta, zeta, tube_ratio, max_iterations=20)
    assert check_move1_condition(new_grains, rho, delta, zeta, tube_ratio)


def test_move2_condition():
    tubes = make_parallel_tubes(4)
    grains = guth_grains_skeleton(tubes, None, grid_n=2)
    # 小 omega 时阈值接近 1；check 返回 bool。
    assert isinstance(check_move2_condition(grains, omega=0.1, delta=0.05), bool)


def test_apply_moves_returns_three_traces():
    tubes = make_parallel_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    final, traces = apply_moves(
        grains,
        rho=0.1,
        delta=0.05,
        zeta=0.01,
        omega=0.1,
        tube_ratio=0.5,
        max_iterations=5,
    )
    assert len(traces) == 3
    assert all(hasattr(t, "move_name") for t in traces)
    assert final.count <= grains.count


def test_apply_moves_trace_quantitative_fields():
    tubes = make_parallel_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    _, traces = apply_moves(
        grains,
        rho=0.1,
        delta=0.05,
        zeta=0.01,
        omega=0.1,
        tube_ratio=0.5,
        max_iterations=5,
    )
    for tr in traces:
        assert tr.input_count >= 0
        assert tr.output_count >= 0
        assert tr.input_max_c >= 0
        assert tr.output_max_c >= 0
        assert tr.input_max_rho >= 0
        assert tr.output_max_rho >= 0
        assert tr.input_max_local_kt_error >= 0
        assert tr.output_max_local_kt_error >= 0
        assert isinstance(tr.satisfied, bool)
def test_apply_moves_monotonicity():
    """Move #1/#2 不应使 max_c 变小；Move #2 不应增大 max_rho；
    Move #3 不应增大 max_local_kt_error。
    """
    tubes = make_parallel_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    _, traces = apply_moves(
        grains,
        rho=0.1,
        delta=0.05,
        zeta=0.01,
        omega=0.1,
        tube_ratio=0.5,
        max_iterations=5,
    )
    m1, m2, m3 = traces
    assert m1.output_max_c >= m1.input_max_c
    assert m2.output_max_c >= m2.input_max_c
    assert m2.output_max_rho <= m2.input_max_rho
    assert m3.output_max_local_kt_error <= m3.input_max_local_kt_error


def make_fan_tubes(n: int = 8):
    """方向分散的管，构成一个球面扇形。"""
    tubes = []
    for theta in np.linspace(0.1, np.pi - 0.1, n):
        for phi in np.linspace(0.0, 2 * np.pi, n, endpoint=False):
            tubes.append(
                Tube(
                    center=np.zeros(3),
                    direction=Direction.from_spherical(theta, phi),
                    length=1.0,
                    delta=0.05,
                )
            )
    return tubes


def test_move1_satisfies_post_condition_for_fan_tubes():
    """Move #1 在方向分散的 fan 构型上**尝试**满足 per-ρ-bin 界。

    ⚠️ **本测试曾是一个空断言。** 它原本写成
    ``assert check_move1_condition(new_grains, ..., tube_ratio=None)``，而
    当时**每次都能通过**——不是因为 Move #1 真的满足了后条件，而是因为
    ``_effective_tube_ratio`` 在元数据缺失时返回哨兵 ``1e-12``，把目标压到
    ≈0，于是任何 ``c`` 都"满足"。实测旧代码里 575/575 个 grain 全部走哨兵。

    2026-09-28 修掉哨兵后发现**真实结论**：该构型下 ``max_c`` 第 2 轮后
    就卡在 1.081（合并 stall），而目标是 4.852——**Move #1 迭代多少轮都
    满足不了**。这不是 bug，是 Move #1（只做沿长轴合并）的能力边界。

    所以本测试改为断言**真实行为**：判定为 False，且**不是**因为缺元数据
    （``missing == 0`` 才算真的算过）。"缺元数据"与"算过但不够"必须分开——
    前者是数据缺陷，后者是算法边界，处置完全不同。
    """
    tubes = make_fan_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    rho, delta, zeta = 0.25, 0.05, 0.01
    new_grains = move1_longer_grains(
        grains, rho, delta, zeta, tube_ratio=None, max_iterations=20
    )
    ok, missing = check_move1_condition_detailed(
        new_grains, rho, delta, zeta, tube_ratio=None
    )
    assert missing == 0, (
        "有 grain 取不到 #T_ρ/#T——合并时丢了 bin 元数据（这是缺陷，不是边界）"
    )
    assert not ok, (
        "若这里变成 True，说明 Move #1 的实现或 ρ 分区规则变了；"
        "请复核后更新本测试的注释（原结论是 max_c 卡在 1.081、目标 4.852）"
    )


def test_move1_bin_metadata_survives_merging():
    """合并必须**继承** ρ-bin 元数据（修复：过去被丢掉，导致目标≈0）。

    这是上一条测试暴露的根因：``_merge_along_long_axis`` 只拷贝几何量，
    把 ``bin_index`` / ``bin_tube_count`` / ``total_tube_count`` 丢了 ⇒
    合并后的 grain 取不到 ``#T_ρ/#T`` ⇒ 旧代码用哨兵 ``1e-12`` 让条件
    "轻松通过"。判据直接盯住"合并后还剩多少 grain 有元数据"。
    """
    tubes = make_fan_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    assert all(p.total_tube_count > 0 for p in grains.prisms), "初始 grains 应有元数据"

    merged = move1_longer_grains(
        grains, rho=0.25, delta=0.05, zeta=0.01, tube_ratio=None, max_iterations=2
    )
    with_meta = [p for p in merged.prisms if p.bin_tube_count > 0]
    assert with_meta, "合并后**全部** grain 都丢了 bin 元数据"


def test_move1_raises_when_ratio_unavailable():
    """取不到 #T_ρ/#T 时必须报错，**不得**悄悄用哨兵让条件通过。

    判定函数（``check_move1_condition``）在缺元数据时返回 ``False``
    （"不得宣称满足"）；而修复函数（``move1_longer_grains``）必须知道目标
    是多少才能动手，所以只能抛错。两者处置不同，故分别钉住。
    """
    tubes = make_fan_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    # 把所有 bin 元数据抹掉，模拟"合并后丢失"的坏数据。
    stripped = GrainsDecomposition(
        prisms=[
            Prism(bounds=p.bounds, tube_indices=p.tube_indices, density=p.density,
                  local_kt_error=p.local_kt_error)
            for p in grains.prisms
        ]
    )
    ok, missing = check_move1_condition_detailed(
        stripped, rho=0.25, delta=0.05, zeta=0.01, tube_ratio=None
    )
    assert not ok and missing > 0, "缺元数据时必须报'无法判定'，而非'满足'"

    with pytest.raises(ValueError, match="取不到 #T_ρ/#T"):
        move1_longer_grains(
            stripped, rho=0.25, delta=0.05, zeta=0.01, tube_ratio=None,
            max_iterations=2,
        )


def test_apply_moves_summary_is_readable():
    tubes = make_parallel_tubes(4)
    grains = guth_grains_skeleton(tubes, None, grid_n=2)
    _, traces = apply_moves(
        grains,
        rho=0.1,
        delta=0.05,
        zeta=0.01,
        omega=0.1,
        tube_ratio=0.5,
        max_iterations=2,
    )
    for tr in traces:
        s = tr.summary()
        assert tr.move_name in s
        # ``iters=``（**实际执行轮数**）而非 ``iterations=``（预算）。
        # 2026-10-01 改：把"预算"当"实际用量"会掩盖"一轮都没动"。
        assert "iters=" in s
        # 缺口二：summary 要显示"是否达成增益"，否则读日志分不清
        # "应用了 Move"与"应用了并推进了"。
        assert "gain=" in s


class TestConditionKindIsRecorded:
    """``satisfied`` 必须自证"对着哪个量判的"。

    背景：Move #3 的后条件是论文 Lemma 8.3 的 ``C_KT-CW^loc(G) <= δ^{-ζ}``，
    而快速代理是 ``n|T|/|G|``——**两个不同的量**。``apply_moves`` 过去既不收
    ``tubes`` 也不收 ``error_mode``，于是 Move #3 永远走 proxy，而
    ``satisfied=True`` 写进 trace 后与"真后条件满足"长得一模一样。

    这组测试钉住：口径必须被记录，且"声称 actual 却没给 tubes"必须报错
    （否则会静默退回 proxy，等于口径说谎）。
    """

    def test_move3_defaults_to_proxy_and_says_so(self):
        tubes = make_parallel_tubes(8)
        grains = guth_grains_skeleton(tubes, None, grid_n=4)
        _, traces = apply_moves(
            grains, rho=0.1, delta=0.05, zeta=0.01, omega=0.1,
            tube_ratio=0.5, max_iterations=2,
        )
        m3 = [t for t in traces if "Move #3" in t.move_name]
        assert len(m3) == 1
        assert m3[0].condition_kind == "proxy"

    def test_move1_move2_marked_exact(self):
        """Move #1/#2 的条件是对着论文的表达式直接判的（非代理量）。"""
        tubes = make_parallel_tubes(8)
        grains = guth_grains_skeleton(tubes, None, grid_n=4)
        _, traces = apply_moves(
            grains, rho=0.1, delta=0.05, zeta=0.01, omega=0.1,
            tube_ratio=0.5, max_iterations=2,
        )
        kinds = {t.move_name.split(" (")[0]: t.condition_kind for t in traces}
        assert kinds["Move #1"] == "exact"
        assert kinds["Move #2"] == "exact"

    def test_actual_without_tubes_is_rejected(self):
        """声称 actual 却不给 tubes ⇒ 抛错（否则口径说谎）。"""
        tubes = make_parallel_tubes(8)
        grains = guth_grains_skeleton(tubes, None, grid_n=4)
        with pytest.raises(ValueError, match="需要 tubes"):
            apply_moves(
                grains, rho=0.1, delta=0.05, zeta=0.01, omega=0.1,
                tube_ratio=0.5, max_iterations=2, error_mode="actual",
            )

    def test_actual_mode_recorded_when_tubes_given(self):
        """给了 tubes 时口径应记为 actual（且真的传到了 Move #3）。"""
        tubes = make_parallel_tubes(8)
        grains = guth_grains_skeleton(tubes, None, grid_n=4)
        _, traces = apply_moves(
            grains, rho=0.1, delta=0.05, zeta=0.01, omega=0.1,
            tube_ratio=0.5, max_iterations=2, tubes=tubes, error_mode="actual",
        )
        m3 = [t for t in traces if "Move #3" in t.move_name][0]
        assert m3.condition_kind == "actual"

    def test_summary_shows_condition_kind(self):
        """summary() 要显示口径，否则读日志的人看不出语义。"""
        tubes = make_parallel_tubes(4)
        grains = guth_grains_skeleton(tubes, None, grid_n=2)
        _, traces = apply_moves(
            grains, rho=0.1, delta=0.05, zeta=0.01, omega=0.1,
            tube_ratio=0.5, max_iterations=2,
        )
        for tr in traces:
            assert f"[{tr.condition_kind}]" in tr.summary()


class TestAchievedGain:
    """缺口二：Move 单次执行必须**自知**是否达成定量增益。

    **为什么需要**：论文要求"单步至少放大 δ^{-ζ}"，实现是"迭代合并直到条件
    满足"，**无单步放大保证**。实测（内部分批测量脚本，16
    构型）Move #2 **100% 空操作**、Move #1 空操作 13/16。没有这个字段时，
    "应用了 Move 但什么都没变"与"应用了并推进了"在结果里长得一样。
    """

    def test_gain_true_when_target_improves(self):
        """目标量真的变化时 ``achieved_gain`` 必须为 True。

        用 parallel 构型 + ``tube_ratio=0.5``：实测 Move #1 把 ``max_c``
        0.55 → 1.1（+100%）。**前提守卫**：先断言目标量确实变了，否则这条断言
        是空转的。

        **同时钉住 ``iters_used`` 的精确性**（缺口二补强，2026-10-09）：
        它由 ``_moveN_impl`` 在合并循环里**直接计数**，不再是"从 grain 数变化
        推的下界估计（0 或 1）"。本构型 256→64→(Move #2 空转)→64→1，实测
        Move #1 ``iters_used == 1``、Move #2 ``== 0``、Move #3 ``== 2``——
        其中 Move #3 的 **2** 正是旧实现（至多 1）报不出来的多轮推进。
        """
        tubes = make_parallel_tubes(8)
        grains = guth_grains_skeleton(tubes, None, grid_n=4)
        _, traces = apply_moves(
            grains, rho=0.1, delta=0.05, zeta=0.01, omega=0.1,
            tube_ratio=0.5, max_iterations=2,
        )
        m1 = [t for t in traces if "Move #1" in t.move_name][0]
        assert m1.output_max_c > m1.input_max_c, "前提：该构型下 Move #1 应推大 max_c"
        assert m1.achieved_gain is True
        assert m1.relative_gain is not None and m1.relative_gain > 0

        # 逐 Move 的精确轮数 + "iters_used==0 ⟺ grain 数不变"不变量。
        by_name = {t.move_name.split()[1]: t for t in traces}
        assert by_name["#1"].iters_used == 1, "Move #1 实测 1 轮（256→64）"
        assert by_name["#2"].iters_used == 0, "Move #2 实测空操作（64→64）"
        # 多轮推进：这条是"精确值 ≠ 旧的下界估计"的直接证据。
        assert by_name["#3"].iters_used == 2, "Move #3 实测 2 轮（64→1）"
        for t in traces:
            if t.iters_used == 0:
                assert t.input_count == t.output_count, (
                    f"{t.move_name}：它报 0 轮，grain 数就必须不变"
                )
            else:
                assert t.output_count < t.input_count, (
                    f"{t.move_name}：报 {t.iters_used} 轮，grain 数就必须下降"
                )

    def test_gain_false_when_target_unchanged(self):
        """目标量逐位不变时 ``achieved_gain`` 必须为 False。

        **构造一个必经的空操作**：给一个**已满足** Move #1 条件的 grains（
        ``max_c`` 极大 ⇒ 条件成立 ⇒ 循环首轮就 break）。这样"应用 Move"确实
        什么都没做——正是 `NO_PROGRESS` 的微观形态。
        """
        import numpy as np
        # 单根长 grain：c 极大，Move #1 条件几乎必然已满足。
        p = Prism(bounds=np.array([[0.0, 0.05], [0.0, 0.05], [0.0, 0.95]]))
        grains = GrainsDecomposition([p])
        _, traces = apply_moves(
            grains, rho=0.1, delta=0.05, zeta=0.01, omega=0.1,
            tube_ratio=0.5, max_iterations=4,
        )
        m1 = [t for t in traces if "Move #1" in t.move_name][0]
        # 前提守卫：必须真的没变，否则本用例测的不是"空操作"。
        assert m1.output_max_c == m1.input_max_c
        assert m1.achieved_gain is False
        assert m1.iters_used == 0

    def test_iters_used_is_not_the_budget(self):
        """``iters_used`` 是**实际执行轮数**，不是 ``max_iterations`` 预算。

        此前 ``MoveTrace.iterations`` 直接传预算，于是"还想合并但被预算截断"
        与"一轮就没得合并"混成同一个数。用上面那个"条件已满足"的构型验证：
        预算给 9，实际必须是 0。
        """
        import numpy as np
        p = Prism(bounds=np.array([[0.0, 0.05], [0.0, 0.05], [0.0, 0.95]]))
        grains = GrainsDecomposition([p])
        _, traces = apply_moves(
            grains, rho=0.1, delta=0.05, zeta=0.01, omega=0.1,
            tube_ratio=0.5, max_iterations=9,
        )
        m1 = [t for t in traces if "Move #1" in t.move_name][0]
        assert m1.iterations == 9        # 预算如实记录
        assert m1.iters_used == 0        # 实际用量是另一回事
        assert m1.iters_used != m1.iterations

    def test_target_direction_differs_per_move(self):
        """方向因 Move 而异：Move #1 要 max_c **增**，Move #2/#3 要量**减**。

        用同一对 before/after 直接调 ``_move_target_delta``（单元级），钉住方向
        不靠符号约定。
        """
        from kakeya_ai.moves import _move_target_delta

        # **有意的鸭子类型桩**：``_move_target_delta`` 只读这三个属性
        # （``max_c`` / ``max_rho`` / ``max_local_kt_error``），不需要一个真的
        # ``GrainsDecomposition``（构造它要跑完整的 grains 分解，而本用例测的
        # 只是"判据的方向"）。参数声明是具体类型，故此处类型不兼容——ignore
        # 是如实的，不是掩盖缺陷。
        class _Fake:
            def __init__(self, c, rho, err):
                self.max_c = c
                self.max_rho = rho
                self.max_local_kt_error = err

        before = _Fake(c=1.0, rho=0.5, err=3.0)
        after = _Fake(c=2.0, rho=0.2, err=1.0)  # c 增、rho/err 减
        assert _move_target_delta("max_c", before, after).improved is True  # type: ignore[arg-type]
        assert _move_target_delta("max_rho", before, after).improved is True  # type: ignore[arg-type]
        assert _move_target_delta("max_local_kt_error", before, after).improved is True  # type: ignore[arg-type]
        # 反方向：对 max_c 而言，变小不是改进。
        assert _move_target_delta("max_c", after, before).improved is False  # type: ignore[arg-type]
        # 不变不算改进。
        assert _move_target_delta("max_c", before, before).improved is False  # type: ignore[arg-type]


def test_transverse_axes_deterministic():
    """棱柱的两个横向轴必须排除长轴。"""
    from kakeya_ai.grains import Prism
    import numpy as np

    # Prism with dimensions 0.1 x 0.2 x 0.5 (long axis = z).
    bounds = np.array([[0.0, 0.1], [0.0, 0.2], [0.0, 0.5]])
    p = Prism(bounds=bounds)
    axes = _transverse_axes(p)
    assert _long_axis(p) == 2
    assert set(axes) == {0, 1}
    assert p.dimensions[axes[0]] <= p.dimensions[axes[1]]


def test_move3_preserves_bin_metadata():
    """Move #3 应把 ρ-bin 元数据保留在拓宽后的 grain 上。"""
    tubes = make_parallel_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    # guth_grains_skeleton 会设置 bin 元数据；验证前置条件。
    assert any(p.bin_index is not None for p in grains.prisms)
    new_grains = move3_wider_grains(grains, zeta=0.01, delta=0.05)
    for p in new_grains.prisms:
        assert p.bin_index is not None
        assert p.bin_tube_count > 0
        assert p.total_tube_count == len(tubes)
        assert p.bin_direction is not None


def test_move3_preserves_total_tubes_accounted():
    """拓宽会合并 grain，但不应丢失管索引。"""
    tubes = make_parallel_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    old_accounted = grains.total_tubes_accounted
    new_grains = move3_wider_grains(grains, zeta=0.01, delta=0.05)
    assert new_grains.total_tubes_accounted == old_accounted


def test_move3_regularize_shading():
    """遮蔽正则化抬高密度下限而不改变数量。"""
    tubes = make_parallel_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    # 通过手动重建制造一个低密度 grain（move3 默认已经做了正则化）。
    low_density = min(p.density for p in grains.prisms)
    regular = regularize_shading(grains, min_density=0.5)
    assert regular.count == grains.count
    assert all(p.density >= min(low_density, 0.5) - 1e-12 for p in regular.prisms)


def test_move3_target_rho():
    """Move #3 之后的目标 rho 按 delta^{-zeta/1000} 增长。"""
    rho, delta, zeta = 0.2, 0.05, 0.01
    target = move3_target_rho(rho, delta, zeta)
    expected = rho * (delta ** (-zeta / 1000.0))
    assert target == pytest.approx(expected)
    assert target > rho


def test_move3_total_volume_does_not_shrink():
    """拓宽会合并棱柱，故总体积不应减少。"""
    tubes = make_parallel_tubes(8)
    grains = guth_grains_skeleton(tubes, None, grid_n=4)
    old_volume = grains.total_prism_volume
    new_grains = move3_wider_grains(grains, zeta=0.01, delta=0.05)
    new_volume = new_grains.total_prism_volume
    assert new_volume >= old_volume - 1e-12


class TestMove2ConditionIsStructurallyUnreachable:
    """局限 #14 的守门：默认 ``omega`` 下 Move #2 的前置条件几乎恒已满足。

    **为什么钉住这条**：内部设计文档（局限 #14、§7.0.2）记着
    "Move #2 近乎缺席"，根因是**判据不可达**而非采样不足。但"不可达"是
    **参数与阈值共同**造成的结论——一旦有人改了 `omega` 默认值或阈值公式，
    Move #2 就可能开始真的做事，上面那句文档结论就失效了。这条守门让改动
    **立刻可见**（改坏即红），逼人回来同步文档。

    判据：默认 `omega=0.1, delta=0.05` 时阈值 = ``0.05**0.001 ≈ 0.997``，
    而 `max_rho <= 1` 恒成立 ⇒ 条件恒满足。用一个真实构型验证。
    """

    def test_default_omega_threshold_above_any_valid_rho(self):
        # **必须走被测代码**（`move2_rho_threshold` / `check_move2_condition`），
        # 不能自己重算阈值公式——否则公式被改（如去掉 `/100`）时自算的阈值
        # 跟着改，变异测不出来（本项目已踩过：守门自己重算 = 装饰品）。
        from kakeya_ai.moves import move2_rho_threshold

        # 阈值公式本身：默认 omega 下应贴近 1（对 ρ≤1 平凡成立）。
        thr = move2_rho_threshold(0.1, 0.05)
        assert thr > 0.99, (
            f"默认阈值 {thr} 不再是'贴近 1'的量级（局限 #14 的前提）。"
            "若这是有意改动，请一并更新已知局限登记（局限 #14 与 Move2 阈值口径）。"
        )
        # **判别性输入**：ρ 落在两个候选阈值之间（δ^{0.001}=0.997 与
        # δ^{0.1}=0.741 之间）⇒ "阈值公式是否含 /100"会**翻转到不同结论**。
        # 没有它，改阈值公式的变异测不出来。
        import numpy as np
        box = Prism(bounds=np.array([[0.0, 0.9], [0.0, 0.9], [0.0, 1.0]]))
        g09 = GrainsDecomposition([box])
        assert g09.max_rho == pytest.approx(0.9)
        assert check_move2_condition(g09, omega=0.1, delta=0.05) is True, (
            "ρ=0.9 在默认阈值 0.05^{0.001}≈0.997 下应满足。若不满足，"
            "说明阈值公式变了（例如丢了 /100）——请同步文档局限 #14。"
        )
        # 真实构型亦应满足。
        tubes = make_parallel_tubes(8)
        grains = guth_grains_skeleton(tubes, None, grid_n=4)
        assert check_move2_condition(grains, omega=0.1, delta=0.05) is True

    def test_move2_is_noop_on_a_real_configuration(self):
        """真实构型上 Move #2 不改变 grain 数与 max_rho（= 零增益）。"""
        tubes = make_parallel_tubes(8)
        grains = guth_grains_skeleton(tubes, None, grid_n=4)
        # 前提守卫：条件已满足（不自己重算阈值——由被测代码判定）。
        assert check_move2_condition(grains, omega=0.1, delta=0.05) is True

        after = move2_replace_square_grains(grains, omega=0.1, delta=0.05)
        assert after.count == grains.count
        assert after.max_rho == grains.max_rho

    def test_move2_actually_triggers_at_large_omega(self):
        """反面：把 omega 抬到阈值咬得住的量级，Move #2 必须**不再**平凡满足。

        这是"守门有判别力"的反向验证——若把 omega 写死、阈值变成常数，
        这条会失败。
        """
        tubes = make_parallel_tubes(8)
        grains = guth_grains_skeleton(tubes, None, grid_n=4)
        # omega=100 ⇒ 阈值 = delta^1 = 0.05 < max_rho(~0.5) ⇒ 条件不满足。
        assert check_move2_condition(grains, omega=100.0, delta=0.05) is False


class TestMovesRejectOrientedGrains:
    """定向 grain 的 Move 行为契约。

    **2026-11-13（Q4 #12）契约变更**：三个 Move 已支持定向 grain（局部
    正交基合并，``moves._merge_oriented``），不再抛
    ``NotImplementedYetError``；仍拒绝的是 ① 混合分解（轴对齐+定向）、
    ② Move #3 的 ``error_mode='actual'``（重建依赖轴对齐 Prism 构造）。
    """

    def _oriented_grains(self):
        d = Direction(np.array([1.0, 1.0, 1.0]))
        tubes = [
            Tube(center=np.array([0.02 * i, 0.01 * i, 0.0]), direction=d,
                 length=1.0, delta=0.05)
            for i in range(8)
        ]
        return guth_grains_skeleton(tubes, None, grid_n=4, frame="oriented"), tubes

    def test_move1_move2_move3_execute_on_oriented(self):
        from kakeya_ai.grains import OrientedPrism

        grains, _ = self._oriented_grains()
        assert not grains.is_axis_aligned
        for out in (
            move1_longer_grains(grains, rho=0.5, delta=0.05, zeta=0.1),
            move2_replace_square_grains(grains, omega=100.0, delta=0.05),
            move3_wider_grains(grains, zeta=0.1, delta=0.05, max_local_error=1e9),
        ):
            assert all(isinstance(p, OrientedPrism) for p in out.prisms)

    def test_apply_moves_raises(self):
        from kakeya_ai.errors import NotImplementedYetError

        grains, tubes = self._oriented_grains()
        with pytest.raises(NotImplementedYetError):
            apply_moves(grains, rho=0.5, delta=0.05, zeta=0.1, omega=100.0,
                        tubes=tubes, error_mode="actual")

    def test_axis_grains_still_work(self):
        """回归：轴对齐 grain 的既有路径不受影响。"""
        grains, tubes = self._oriented_grains()
        ax = guth_grains_skeleton(tubes, None, grid_n=4, frame="axis")
        assert ax.is_axis_aligned
        # 不应抛错。
        move1_longer_grains(ax, rho=0.5, delta=0.05, zeta=0.1)
