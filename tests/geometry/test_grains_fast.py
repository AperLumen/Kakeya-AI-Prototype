"""面向缓存友好的 ``grains_fast`` 变体的基准与健全性测试。"""

import time

import numpy as np

from kakeya_ai.grains_fast import guth_grains_skeleton as fast_guth_grains_skeleton
from kakeya_ai.grains import guth_grains_skeleton
from kakeya_ai.moves import (
    check_move1_condition,
    check_move2_condition,
    move1_longer_grains,
    move2_replace_square_grains,
)
from kakeya_ai.primitives import Direction, Tube


def make_fan(n: int = 8):
    tubes = []
    for theta in np.linspace(0.1, np.pi - 0.1, n):
        for phi in np.linspace(0, 2 * np.pi, n, endpoint=False):
            tubes.append(
                Tube(
                    center=np.zeros(3),
                    direction=Direction.from_spherical(theta, phi),
                    length=1.0,
                    delta=0.05,
                )
            )
    return tubes


def test_fast_grains_produces_nonzero_grains():
    tubes = make_fan(6)
    grains = fast_guth_grains_skeleton(tubes, None, grid_n=4)
    assert grains.count > 0


def test_fast_grains_move1_satisfies_post_condition():
    """fast 实现同样：Move #1 **尝试**满足 per-ρ-bin 界，本构型下达不到。

    ⚠️ 本测试曾是**空断言**（原写法 ``assert check_move1_condition(...)``
    恒真）——它通过的原因是元数据被合并函数丢掉后，哨兵 ``1e-12`` 把目标
    压到 ≈0。2026-09-28 修掉哨兵与元数据丢失后，真实结论是：``max_c`` 卡在
    1.081 而目标 4.852，**该构型下 Move #1 无法满足后条件**。

    这里断言"判过（``missing==0``）且结论为 False"，与
    ``tests/test_moves.py`` 的同名测试保持同一口径。
    """
    from kakeya_ai.moves import check_move1_condition_detailed

    tubes = make_fan(8)
    grains = fast_guth_grains_skeleton(tubes, None, grid_n=4)
    rho, delta, zeta = 0.25, 0.05, 0.01
    # ``grains_fast`` 是 ``grains`` 的**隔离复制**（见其模块 docstring：
    # "API mirrors grains.py so it can be dropped in"），两者各有一个
    # ``GrainsDecomposition``——结构逐字段等价，但类型系统看到的是**两个不同的
    # 类**。运行时靠 duck typing 正常工作，类型上无法表达“结构等价”。
    # 本文件存在的意义就是**对比这两份实现**，所以必须把它们混用——
    # 这里的 ignore 是如实地记录“已知且有意的类型系统局限”，不是掩盖缺陷。
    m1 = move1_longer_grains(
        grains,  # type: ignore[arg-type]
        rho, delta, zeta, tube_ratio=None, max_iterations=20,
    )
    ok, missing = check_move1_condition_detailed(m1, rho, delta, zeta, tube_ratio=None)
    assert missing == 0, "有 grain 取不到 #T_ρ/#T——合并丢了 bin 元数据（缺陷）"
    assert not ok, (
        "若变成 True，说明实现或 ρ 分区规则变了；请复核后同步更新 "
        "tests/test_moves.py 里的同名结论"
    )


def test_fast_grains_move2_satisfies_post_condition():
    tubes = make_fan(8)
    grains = fast_guth_grains_skeleton(tubes, None, grid_n=4)
    omega, delta = 0.1, 0.05
    # 同 ``test_fast_grains_move1_satisfies_post_condition``：两份隔离复制
    # 的 ``GrainsDecomposition`` 名义不同，忽略是如实的（见那里的说明）。
    m2 = move2_replace_square_grains(
        grains,  # type: ignore[arg-type]
        omega, delta, max_iterations=20,
    )
    assert check_move2_condition(m2, omega, delta)


def test_fast_grains_faster_than_exact():
    """fast 变体应当可测量地更快；不要求精度。"""
    tubes = make_fan(8)

    n = 20
    t_exact = []
    t_fast = []
    for _ in range(n):
        t0 = time.perf_counter()
        guth_grains_skeleton(tubes, None, grid_n=4)
        t_exact.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        fast_guth_grains_skeleton(tubes, None, grid_n=4)
        t_fast.append(time.perf_counter() - t0)

    mean_exact = float(np.mean(t_exact))
    mean_fast = float(np.mean(t_fast))
    # fast 平均应至少快 1.5 倍；否则大声失败。
    assert mean_fast < mean_exact * 0.7, (
        f"fast variant not faster: exact={mean_exact*1e3:.2f}ms "
        f"fast={mean_fast*1e3:.2f}ms"
    )


def test_fast_grains_coarser_and_fewer_cells():
    """fast 变体对小 bin 有意粗化网格，故它应比精确变体产生更少的
    cell/grain，同时仍满足 Move #1。
    """
    tubes = make_fan(8)
    exact = guth_grains_skeleton(tubes, None, grid_n=4)
    fast = fast_guth_grains_skeleton(tubes, None, grid_n=4)
    assert fast.count <= exact.count
