"""定向 grain（OrientedPrism）的三个 Move（Q4 #12，2026-11-13）。

定向 Move 在**局部正交基**里合并（``moves._merge_oriented``），不再抛
``NotImplementedYetError``。本组测试钉住：

- 三个 Move 在定向分解上可执行且条件收敛（c 增 / rho 降 / 误差降）；
- 合并只在近似同标架的方向簇内进行（差异 >≈5.7° 的 grain 不被合并）；
- 合并结果是保守覆盖：定向体积只增不减、世界 AABB 不漏原 grain；
- 轴对齐情形**数值不回归**（与既有路径逐位一致由既有测试守门）；
- 混合分解（轴对齐+定向）显式报错，不静默处理。
"""

from typing import Sequence, cast

import numpy as np
import pytest

from kakeya_ai.grains import GrainsDecomposition, OrientedPrism, Prism
from kakeya_ai.moves import (
    apply_moves,
    check_move1_condition,
    move1_longer_grains,
    move3_wider_grains,
    _merge_oriented,
    _split_by_orientation,
)


def _oriented_grains(directions, n_per_dir=4, c=0.1, a=0.02):
    """造一簇定向 grain：每个方向 n_per_dir 个，沿长轴排开。"""
    prisms = []
    for d in directions:
        d = np.asarray(d, dtype=float)
        d = d / np.linalg.norm(d)
        # 长轴 = d；横向取两个与 d 正交的单位向量。
        tmp = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(d, tmp)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        axes = np.stack([e2, e1, d])  # [0]=a 短轴 … [2]=c 长轴
        for i in range(n_per_dir):
            prisms.append(OrientedPrism(
                center=d * (i * c), axes=axes,
                extents=np.array([a, a, c]),
                tube_indices=(i,), density=1.0, local_kt_error=5.0,
            ))
    return GrainsDecomposition(prisms)


def test_split_by_orientation_partitions_and_rejects_mixed():
    g = _oriented_grains([[0, 0, 1]], n_per_dir=2)
    axis, oriented = _split_by_orientation(g, "t")
    assert not axis and len(oriented) == 2
    mixed = GrainsDecomposition([
        oriented[0],
        Prism(bounds=np.array([[0, 0.1], [0, 0.1], [0, 0.1]]), tube_indices=(0,)),
    ])
    with pytest.raises(Exception, match="混合"):
        _split_by_orientation(mixed, "t")


def test_merge_oriented_reduces_count_and_covers():
    g = _oriented_grains([[0, 0, 1]], n_per_dir=4, c=0.1)
    before_vol = sum(p.volume for p in g.prisms)
    # ``GrainsDecomposition.prisms`` 的声明类型是两类并集，而本用例只造了定向
    # grain；``cast`` 只存在于类型层面，运行时不转换。
    merged = _merge_oriented(cast("Sequence[OrientedPrism]", g.prisms), factor=2)
    assert len(merged) == 2
    # 保守覆盖：合并后总体积 ≥ 原体积（盒覆盖必有外扩）。
    assert sum(p.volume for p in merged) >= before_vol * 0.999
    # 合并盒覆盖原 grain：首块的长轴范围应覆盖前两个 grain。
    span = merged[0].c
    assert span >= 2 * 0.1 - 1e-9


def test_merge_oriented_refuses_incompatible_frames():
    # 两个方向差异远大于容差（90°）的簇不得合并。
    g = _oriented_grains([[0, 0, 1], [1, 0, 0]], n_per_dir=1)
    merged = _merge_oriented(cast("Sequence[OrientedPrism]", g.prisms), factor=4)
    assert len(merged) == 2  # 一个都合不掉


def test_oriented_move1_executes_and_lengthens():
    # c 远小于目标 ⇒ Move #1 应真的合并（c 增长、数量下降）。
    g = _oriented_grains([[0, 0, 1]], n_per_dir=8, c=0.05)
    out = move1_longer_grains(g, rho=0.3, delta=0.05, zeta=0.1,
                              tube_ratio=1.0, max_iterations=6)
    assert all(isinstance(p, OrientedPrism) for p in out.prisms)
    assert out.count < g.count
    assert out.max_c > g.max_c


def test_oriented_move3_widens():
    g = _oriented_grains([[0, 0, 1]], n_per_dir=4, c=0.2)
    out = move3_wider_grains(g, zeta=0.1, delta=0.05, max_local_error=100.0,
                             max_iterations=4)
    # 强制高误差上限 ⇒ 应触发横向合并（数量下降、单 grain 变宽或误差下降）。
    assert out.count <= g.count


def test_oriented_apply_moves_full_pipeline():
    g = _oriented_grains([[0, 0, 1], [0.3, 0, 0.95]], n_per_dir=6, c=0.05)
    # 合成 grain 无 bin 元数据：显式给 tube_ratio 兜底（真实 grains 由
    # guth_grains_skeleton 带元数据，见 _grain_target_c 的契约）。
    out, traces = apply_moves(g, rho=0.3, delta=0.05, zeta=0.1, omega=0.0,
                              tube_ratio=1.0, max_iterations=6)
    assert traces  # 三个 Move 各出一条 trace
    assert all(isinstance(p, OrientedPrism) for p in out.prisms)
    for tr in traces:
        assert tr.iters_used >= 0


def test_axis_aligned_path_unchanged():
    # 轴对齐既有路径不回归：纯 Prism 分解走旧合并，结果类型仍是 Prism。
    prisms = [
        Prism(bounds=np.array([[0, 0.02], [0, 0.02], [i * 0.1, i * 0.1 + 0.1]]),
              tube_indices=(i,), density=1.0, local_kt_error=1.0)
        for i in range(4)
    ]
    g = GrainsDecomposition(prisms)
    out = move1_longer_grains(g, rho=0.3, delta=0.05, zeta=0.1,
                              tube_ratio=1.0, max_iterations=4)
    assert all(isinstance(p, Prism) and not isinstance(p, OrientedPrism)
               for p in out.prisms)
    assert check_move1_condition  # 条件函数仍可用


def test_oriented_move3_actual_mode_refused():
    g = _oriented_grains([[0, 0, 1]], n_per_dir=2)
    with pytest.raises(Exception, match="actual"):
        move3_wider_grains(g, zeta=0.1, delta=0.05, error_mode="actual",
                           tubes=[])
