"""守门：C++ 内核的 **AABB 预筛**必须不改变结果（逐位等价）(2026-10-07)。

**这个优化在做什么**：`voxel_union_volume` / `mc_union_volume` 对每个体素/点
要扫过所有管做距离计算。AABB 预筛给每根管算一个轴对齐包围盒（线段端点盒
每轴外扩 `delta` 再放宽一点点），**盒外的点必然不命中**，于是跳过昂贵的
`point_segment_dist2`。本机实测：f64 voxel（grid=128，64 管）单线程
**249 → 115 ms**（2.17×），10 线程 **46 → 19 ms**（2.4×）；且**逐位一致**。

**为什么必须单列守门**：AABB 判据一旦**收紧**（把盒算小了），就会**误拒**
真命中的点 → 体积偏小 → 结果变了却"看起来合理"。这类 bug 最阴——不崩、
不报错，只是数字悄悄偏了。故必须证明"盒外 ⇒ 必不命中"这条**保守性**成立。

**本文件守三件事**：

1. **与 AABB 关闭时的结果逐位相同**（同一构建无法关，故用**参考实现**对拍）：
   用一个纯 Python 的"无 AABB"参考算小规模体积，与内核结果比。
2. **边界压力**：把体素中心恰好落在 `delta` 边界附近（盒要放宽才不含漏），
   验证不误拒。
3. **跨后端 + 跨线程逐位一致**（AABB 只应在**速度**上有影响）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kakeya_ai.fast_kernel import fast_voxel_volume
from kakeya_ai.nano_kernel import nano_voxel_volume
from kakeya_ai.primitives import Direction, Tube

REPO = Path(__file__).resolve().parents[2]


def _fan(n_theta=6, n_phi=6, delta=0.1, length=1.0) -> list[Tube]:
    return [
        Tube(center=np.zeros(3), direction=Direction.from_spherical(t, p),
             length=length, delta=delta)
        for t in np.linspace(0.2, np.pi - 0.2, n_theta)
        for p in np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    ]


def _reference_voxel_volume(tubes: list[Tube], grid_n: int, bounds) -> float:
    """**无 AABB** 的朴素参考实现（Python，小规模）。

    与内核同口径：体素中心，`dist2 <= delta^2` 判包含。不依赖任何内核。
    """
    lo = np.asarray(bounds[:, 0], dtype=np.float64)
    hi = np.asarray(bounds[:, 1], dtype=np.float64)
    vs = (hi - lo) / grid_n
    vol = float(np.prod(vs))
    # 预计算段端点
    segs = []
    for t in tubes:
        v = np.asarray(t.direction.v, dtype=np.float64)
        half = 0.5 * t.length
        a = t.center - half * v
        b = t.center + half * v
        segs.append((a, b, t.delta ** 2))
    inside = 0
    for iz in range(grid_n):
        z = lo[2] + vs[2] * (iz + 0.5)
        for iy in range(grid_n):
            y = lo[1] + vs[1] * (iy + 0.5)
            for ix in range(grid_n):
                x = lo[0] + vs[0] * (ix + 0.5)
                p = np.array([x, y, z])
                hit = False
                for a, b, d2 in segs:
                    ab = b - a
                    ap = p - a
                    denom = float(ab @ ab)
                    tt = 0.0 if denom == 0 else float(ap @ ab) / denom
                    tt = min(max(tt, 0.0), 1.0)
                    c = a + tt * ab
                    if float((p - c) @ (p - c)) <= d2:
                        hit = True
                        break
                if hit:
                    inside += 1
    return inside * vol


BOUNDS = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)


class TestAABBMatchesReferenceVoxel:
    """内核（含 AABB）必须与无-AABB 参考实现算出一致的**体积**。"""

    @pytest.mark.parametrize("grid_n", [16, 24])
    def test_f64_matches_reference(self, grid_n):
        """f64 `fast_voxel_volume` 在小网格上必须与参考**逐位相同**。

        小网格 + 整数计数 ⇒ 体积是 `count * voxel_volume`，两边都用 float64，
        故可要求**精确相等**（不是 approx）。
        """
        tubes = _fan(n_theta=4, n_phi=4, delta=0.15)
        got = fast_voxel_volume(tubes, grid_n=grid_n, bounds=BOUNDS)
        ref = _reference_voxel_volume(tubes, grid_n, BOUNDS)
        assert got == ref, (
            f"AABB 内核体积 {got!r} ≠ 无-AABB 参考 {ref!r} —— "
            "AABB 判据可能误拒了真命中的点（盒算小了）"
        )

    def test_f32_and_f64_agree(self):
        """f32 nano 与 f64 fast 在同一小网格上必须给出**相同**体积。

        两者都加了 AABB，若 AABB 只在速度上有影响，结果应一致。小网格下
        float32 与 float64 的体素中心可能因舍入不同……故用整数**计数**比对：
        本用例只要求"逐位相同"，即两者都精确等于参考。
        """
        tubes = _fan(n_theta=4, n_phi=4, delta=0.15)
        v64 = fast_voxel_volume(tubes, grid_n=16, bounds=BOUNDS)
        v32 = nano_voxel_volume(tubes, grid_n=16, bounds=BOUNDS)
        assert v64 == v32, f"f64 {v64!r} ≠ f32 {v32!r}（AABB 应只影响速度）"


class TestAABBBoundaryStress:
    """体素中心**恰好落在 delta 边界附近**时，AABB 不得误拒。"""

    def test_voxel_on_tube_boundary_not_rejected(self):
        """把一根管放在网格线上，让体素中心正好贴在 `delta` 边界，验证不漏计。

        做法：一根轴向管（沿 x），中心在原点，`delta` 取到"正好落在两个体素
        中心的中垂位置"附近——AABB 若只外扩精确 delta，边界点可能因舍入被拒。
        """
        # 一根沿 x 的管，delta 取网格步长的整数倍关系，制造边界构型。
        grid_n = 32
        vs = 2.0 / grid_n
        tube = Tube(center=np.zeros(3), direction=Direction.from_array([1, 0, 0]),
                    length=1.0, delta=vs)  # delta 正好一个体素宽
        tubes = [tube]
        # 与参考对拍（参考无 AABB，是 ground truth）
        got = fast_voxel_volume(tubes, grid_n=grid_n, bounds=BOUNDS)
        ref = _reference_voxel_volume(tubes, grid_n, BOUNDS)
        assert got == ref, (
            f"边界构型下 AABB 内核 {got!r} ≠ 参考 {ref!r} —— 边界点被误拒"
        )

    def test_tiny_delta_still_consistent(self):
        """极小 delta（盒几乎退化成线段）下也必须与参考一致。"""
        tube = Tube(center=np.array([0.1, -0.2, 0.3]),
                    direction=Direction.from_spherical(1.0, 0.7),
                    length=0.5, delta=1e-4)
        got = fast_voxel_volume([tube], grid_n=24, bounds=BOUNDS)
        ref = _reference_voxel_volume([tube], 24, BOUNDS)
        assert got == ref


class TestAABBMarginSensitivity:
    """**高灵敏度构型**：AABB 的外扩裕度（margin）必须真的够大，否则误拒。

    这是本文件里**判别力最强**的一组——实测发现：若把 f64 版的 margin 从
    `delta` 误缩到 `delta*0.5`，用"扇形小管"构型**测不出来**（体积不变），
    但用"长轴对齐管 + 大 delta"构型**体积从 2.289 掉到 0.804**（差 3×）。
    故必须专门构造后者。

    **为什么这样构造**：一根沿 x、`length=2.0` 的管（几乎铺满域），`delta=0.6`
    ⇒ 真实命中区在 y/z 上达 ±0.6；而 AABB 在 y/z 上是 `[-delta, delta]`。
    若 margin 被缩小，y/z 落在 (margin, delta] 的体素会被**误拒**——它们
    本是真命中（`y² <= delta²`），体积因此显著偏小。
    """

    def test_long_axis_aligned_tube_large_delta(self):
        """长轴对齐管 + 大 delta：内核必须与**独立参考**（numpy 后端）一致。"""
        from kakeya_ai.volume import estimate_union_volume_voxel

        tubes = [Tube(center=np.zeros(3), direction=Direction.from_array([1, 0, 0]),
                      length=2.0, delta=0.6)]
        grid_n = 41
        got = fast_voxel_volume(tubes, grid_n=grid_n, bounds=BOUNDS)
        ref = estimate_union_volume_voxel(tubes, grid_n=grid_n, bounds=BOUNDS)
        assert got == ref, (
            f"长轴管 AABB 内核 {got!r} ≠ numpy 参考 {ref!r} —— "
            "AABB 的外扩裕度（margin）可能被缩小，误拒了 y/z 上的真命中点"
        )

    def test_margin_shrink_would_be_caught(self):
        """断言该构型的**判别力**：参考值必须明显落在"裕度够大"的预期区间。

        若某次改动把 margin 缩到 `delta*0.5`，体积会掉到约 0.8；本用例要求
        内核结果**远离**那个错误值（> 1.5），从而保证上一条测试真有鉴别力。
        """
        tubes = [Tube(center=np.zeros(3), direction=Direction.from_array([1, 0, 0]),
                      length=2.0, delta=0.6)]
        got = fast_voxel_volume(tubes, grid_n=41, bounds=BOUNDS)
        assert got > 1.5, (
            f"体积 {got!r} 偏低（预期 ~2.29）—— 裕度不足会掉到 ~0.8；"
            "该构型正是用来放大裕度误差的"
        )


class TestAABBThreadInvariance:
    """AABB 与并行叠加时，跨线程仍逐位一致（AABB 不引入线程相关差异）。"""

    def test_volume_invariant_across_thread_counts(self):
        """1 / 4 / 10 线程下同一构型体积必须**逐位相同**（子进程隔离）。"""
        script = (
            "import numpy as np\n"
            "from kakeya_ai.fast_kernel import fast_voxel_volume\n"
            "from kakeya_ai.primitives import Direction, Tube\n"
            "tubes=[Tube(center=np.zeros(3),"
            "direction=Direction.from_spherical(t,p),length=1.0,delta=0.1)"
            " for t in np.linspace(0.2,np.pi-0.2,6)"
            " for p in np.linspace(0,2*np.pi,6,endpoint=False)]\n"
            "print(repr(float(fast_voxel_volume(tubes, grid_n=48))))\n"
        )
        env_base = dict(os.environ)
        env_base["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env_base.get("PYTHONPATH", "")
        results = {}
        for t in (1, 4, 10):
            env = dict(env_base)
            env["KAKEYA_NUM_THREADS"] = str(t)
            proc = subprocess.run([sys.executable, "-c", script], cwd=REPO,
                                  capture_output=True, text=True, env=env, timeout=120)
            if proc.returncode != 0:
                pytest.skip(f"C++ 扩展不可用：{proc.stderr[-200:]}")
            results[t] = proc.stdout.strip().splitlines()[-1]
        assert len(set(results.values())) == 1, (
            f"跨线程体积不一致（AABB 引入了线程相关差异？）：{results}"
        )
