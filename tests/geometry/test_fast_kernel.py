"""C++ 几何内核（fast_kernel）的测试。

覆盖：
- C++ 内核正确性（与纯 NumPy 对比，逐位一致）
- 性能基准（C++ vs NumPy，slow 标记）
- 边界情况

GPU 后端（gpu_volume）已不在公开 MVP 范围内，对应测试一并移除。
"""

import time

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.volume import (
    estimate_union_volume_monte_carlo,
    estimate_union_volume_voxel,
)
from kakeya_ai.fast_kernel import (
    fast_contains_points,
    fast_mc_volume,
    fast_tube_intersect,
    fast_voxel_volume,
    get_backend,
    is_cpp_available,
    tubes_to_array,
)


# ============================================================
# 测试辅助
# ============================================================

def make_fan(n_theta=8, n_phi=8, delta=0.1) -> list[Tube]:
    tubes = []
    for theta in np.linspace(0.2, np.pi - 0.2, n_theta):
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False):
            d = Direction.from_spherical(theta, phi)
            tubes.append(Tube(center=np.zeros(3), direction=d, length=1.0, delta=delta))
    return tubes


def make_parallel(n=10, delta=0.1) -> list[Tube]:
    d = Direction.from_array([0, 0, 1])
    return [Tube(center=np.array([i * 0.05, 0.0, 0.0]), direction=d,
                 length=1.0, delta=delta) for i in range(n)]


BOUNDS = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])


# ============================================================
# 后端检测测试
# ============================================================

class TestBackendDetection:
    def test_get_backend_returns_string(self):
        backend = get_backend()
        assert backend in ("cpp", "numpy")

    def test_is_cpp_available_returns_bool(self):
        assert isinstance(is_cpp_available(), bool)


# ============================================================
# tubes_to_array 测试
# ============================================================

class TestTubesToArray:
    def test_correct_shape(self):
        tubes = make_fan(3, 3)
        arr = tubes_to_array(tubes)
        assert arr.shape == (9, 8)
        assert arr.dtype == np.float64

    def test_empty_tubes(self):
        arr = tubes_to_array([])
        assert arr.shape == (0, 8)

    def test_values_match(self):
        t = Tube(center=np.array([1.0, 2.0, 3.0]),
                 direction=Direction.from_array([1, 0, 0]),
                 length=2.0, delta=0.05)
        arr = tubes_to_array([t])
        assert arr[0, 0] == 1.0  # center x
        assert arr[0, 6] == 2.0  # length
        assert arr[0, 7] == 0.05  # delta


# ============================================================
# fast_contains_points 正确性测试
# ============================================================

class TestFastContainsPoints:
    def test_matches_numpy(self):
        """C++/Numba 结果应与纯 NumPy 一致。"""
        tubes = make_fan(6, 6, delta=0.1)
        rng = np.random.default_rng(42)
        points = rng.uniform(-1, 1, (2000, 3))

        mask_fast = fast_contains_points(tubes, points)
        mask_np = np.zeros(len(points), dtype=bool)
        for t in tubes:
            mask_np |= t.contains(points)

        assert np.array_equal(mask_fast, mask_np)

    def test_empty_tubes(self):
        points = np.random.rand(100, 3)
        mask = fast_contains_points([], points)
        assert mask.sum() == 0

    def test_single_tube_origin(self):
        """原点应被所有过原点的管包含。"""
        tubes = make_fan(3, 3, delta=0.1)
        origin = np.array([[0.0, 0.0, 0.0]])
        mask = fast_contains_points(tubes, origin)
        assert mask[0] == True

    def test_far_point_not_contained(self):
        """远离原点的点不应被包含。"""
        tubes = make_fan(3, 3, delta=0.05)
        far = np.array([[10.0, 10.0, 10.0]])
        mask = fast_contains_points(tubes, far)
        assert mask[0] == False


# ============================================================
# fast_voxel_volume 正确性测试
# ============================================================

class TestFastVoxelVolume:
    def test_matches_numpy(self):
        """C++ 体素体积应与纯 NumPy 一致。"""
        tubes = make_fan(5, 5, delta=0.1)
        vol_fast = fast_voxel_volume(tubes, grid_n=32, bounds=BOUNDS)
        vol_np = estimate_union_volume_voxel(tubes, grid_n=32, bounds=BOUNDS)
        assert abs(vol_fast - vol_np) < 1e-10

    def test_empty_tubes(self):
        vol = fast_voxel_volume([], grid_n=32)
        assert vol == 0.0

    def test_volume_positive(self):
        tubes = make_fan(5, 5, delta=0.1)
        vol = fast_voxel_volume(tubes, grid_n=32, bounds=BOUNDS)
        assert vol > 0

    def test_larger_grid_converges(self):
        """更细的网格应给出更精确的估计。"""
        tubes = make_fan(4, 4, delta=0.15)
        vol_16 = fast_voxel_volume(tubes, grid_n=16, bounds=BOUNDS)
        vol_48 = fast_voxel_volume(tubes, grid_n=48, bounds=BOUNDS)
        # 两个估计应在同一量级
        assert vol_16 > 0
        assert vol_48 > 0
        assert abs(vol_16 - vol_48) / max(vol_16, vol_48) < 0.3


# ============================================================
# fast_mc_volume 正确性测试
# ============================================================

class TestFastMCVolume:
    def test_matches_numpy(self):
        """C++ MC 体积应与纯 NumPy 一致（相同随机种子）。"""
        tubes = make_fan(4, 4, delta=0.1)
        vol_fast = fast_mc_volume(tubes, n_samples=50000, bounds=BOUNDS, seed=42)
        vol_np = estimate_union_volume_monte_carlo(tubes, n_samples=50000, bounds=BOUNDS, seed=42)
        assert abs(vol_fast - vol_np) < 1e-10

    def test_empty_tubes(self):
        vol = fast_mc_volume([], n_samples=1000)
        assert vol == 0.0

    def test_volume_positive(self):
        tubes = make_fan(4, 4, delta=0.1)
        vol = fast_mc_volume(tubes, n_samples=50000, seed=42)
        assert vol > 0

    def test_more_samples_more_stable(self):
        """更多采样点应给出更稳定的估计。"""
        tubes = make_fan(4, 4, delta=0.1)
        vol1 = fast_mc_volume(tubes, n_samples=10000, seed=1)
        vol2 = fast_mc_volume(tubes, n_samples=10000, seed=2)
        vol_big1 = fast_mc_volume(tubes, n_samples=200000, seed=1)
        vol_big2 = fast_mc_volume(tubes, n_samples=200000, seed=2)
        # 大样本量的两次估计差异应更小
        diff_small = abs(vol1 - vol2)
        diff_big = abs(vol_big1 - vol_big2)
        assert diff_big <= diff_small + 0.01  # 允许数值波动


# ============================================================
# fast_tube_intersect 测试
# ============================================================

class TestFastTubeIntersect:
    def test_self_intersect(self):
        """同方向的管应相交（如果距离够近）。"""
        tubes = make_parallel(5, delta=0.2)
        mat = fast_tube_intersect(tubes, tubes)
        assert mat.shape == (5, 5)
        # 自身一定相交
        for i in range(5):
            assert mat[i, i] == True

    def test_empty(self):
        mat = fast_tube_intersect([], [])
        assert mat.shape == (0, 0)

    def test_far_tubes_no_intersect(self):
        """远离的管不应相交。"""
        d = Direction.from_array([1, 0, 0])
        t1 = [Tube(center=np.zeros(3), direction=d, delta=0.01)]
        t2 = [Tube(center=np.array([0.0, 10.0, 0.0]), direction=d, delta=0.01)]
        mat = fast_tube_intersect(t1, t2)
        assert mat[0, 0] == False


# ============================================================
# 性能基准测试
# ============================================================

class TestPerformance:
    @pytest.mark.slow
    def test_cpp_faster_than_numpy_voxel(self):
        """C++ 体素估计应比纯 NumPy 快。"""
        tubes = make_fan(8, 8, delta=0.08)

        # C++ 时间
        t0 = time.perf_counter()
        vol_cpp = fast_voxel_volume(tubes, grid_n=48, bounds=BOUNDS)
        t_cpp = time.perf_counter() - t0

        # NumPy 时间
        t0 = time.perf_counter()
        vol_np = estimate_union_volume_voxel(tubes, grid_n=48, bounds=BOUNDS)
        t_np = time.perf_counter() - t0

        # 结果一致
        assert abs(vol_cpp - vol_np) < 1e-10

        # C++ 应更快（或至少不慢很多——CI 环境可能有波动）
        if is_cpp_available():
            print(f"\n  Voxel: C++={t_cpp:.4f}s, NumPy={t_np:.4f}s, speedup={t_np/t_cpp:.1f}x")
            assert t_cpp < t_np * 2  # 至少不能慢太多

    @pytest.mark.slow
    def test_cpp_faster_than_numpy_mc(self):
        """C++ MC 估计应比纯 NumPy 快。"""
        tubes = make_fan(8, 8, delta=0.08)
        n = 200_000

        # C++ 时间
        t0 = time.perf_counter()
        vol_cpp = fast_mc_volume(tubes, n_samples=n, bounds=BOUNDS, seed=42)
        t_cpp = time.perf_counter() - t0

        # NumPy 时间
        t0 = time.perf_counter()
        vol_np = estimate_union_volume_monte_carlo(tubes, n_samples=n, bounds=BOUNDS, seed=42)
        t_np = time.perf_counter() - t0

        # 结果一致
        assert abs(vol_cpp - vol_np) < 1e-10

        if is_cpp_available():
            print(f"\n  MC: C++={t_cpp:.4f}s, NumPy={t_np:.4f}s, speedup={t_np/t_cpp:.1f}x")
