"""float32 纳秒级 nano_kernel 封装的测试。"""

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.nano_kernel import (
    nano_contains_points,
    nano_mc_volume,
    nano_voxel_volume,
    get_backend,
    is_cpp_available,
    tubes_to_array_f32,
)
from kakeya_ai.fast_kernel import (
    fast_contains_points,
    fast_mc_volume,
    fast_voxel_volume,
)


def make_fan(n_theta=4, n_phi=4, delta=0.1) -> list[Tube]:
    tubes = []
    for theta in np.linspace(0.2, np.pi - 0.2, n_theta):
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False):
            d = Direction.from_spherical(theta, phi)
            tubes.append(Tube(center=np.zeros(3), direction=d, length=1.0, delta=delta))
    return tubes


BOUNDS = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)


class TestBackend:
    def test_backend_returns_string(self):
        assert get_backend() in ("nano", "fast_fallback")

    def test_is_cpp_available_returns_bool(self):
        assert isinstance(is_cpp_available(), bool)


class TestTubesToArrayF32:
    def test_shape_and_dtype(self):
        tubes = make_fan(3, 3)
        arr = tubes_to_array_f32(tubes)
        assert arr.shape == (9, 8)
        assert arr.dtype == np.float32


class TestNanoContainsPoints:
    def test_matches_fast_kernel(self):
        tubes = make_fan(4, 4, delta=0.1)
        rng = np.random.default_rng(42)
        points = rng.uniform(-1, 1, (2000, 3)).astype(np.float32)

        mask_fast = fast_contains_points(tubes, points)
        mask_nano = nano_contains_points(tubes, points)

        assert mask_fast.shape == mask_nano.shape
        assert np.mean(mask_fast == mask_nano) > 0.95


class TestNanoMCVolume:
    def test_matches_fast_kernel(self):
        tubes = make_fan(4, 4, delta=0.1)
        # 2026-10-30 起 nano 走融合路径（片内 splitmix64 采样），与 fast 的
        # numpy 随机流**相互独立**——容差内是双采样噪声，须用样本量把它压小：
        # p≈0.07、N=500k 时双采样 σ≈0.004 ≪ 0.01（N=50k 时 σ≈0.013 会抖动）。
        vol_fast = fast_mc_volume(tubes, n_samples=500_000, seed=42)
        vol_nano = nano_mc_volume(tubes, n_samples=500_000, seed=42)
        assert abs(vol_fast - vol_nano) < 0.01

    def test_fused_deterministic_same_seed(self):
        """融合路径逐样本 RNG 只依赖 (seed, j) ⇒ 同 seed 必须**逐位相等**
        （与并行切分/线程调度无关——动态调度下也确定）。"""
        tubes = make_fan(4, 4, delta=0.1)
        a = nano_mc_volume(tubes, n_samples=200_000, seed=42)
        b = nano_mc_volume(tubes, n_samples=200_000, seed=42)
        assert a == b

    def test_fused_different_seed_differs(self):
        """不同 seed ⇒ 不同随机流 ⇒ 结果应可区分（统计意义，不作点估计）。"""
        tubes = make_fan(4, 4, delta=0.1)
        a = nano_mc_volume(tubes, n_samples=200_000, seed=42)
        b = nano_mc_volume(tubes, n_samples=200_000, seed=43)
        assert abs(a - b) > 1e-6

    def test_seed_none_falls_back(self):
        """seed=None 保留原"每次随机"语义（退回 numpy 点数组路径），不报错。"""
        tubes = make_fan(4, 4, delta=0.1)
        v = nano_mc_volume(tubes, n_samples=20_000, seed=None)
        assert v > 0


class TestNanoVoxelVolume:
    def test_matches_fast_kernel(self):
        tubes = make_fan(4, 4, delta=0.1)
        vol_fast = fast_voxel_volume(tubes, grid_n=32, bounds=BOUNDS.astype(np.float64))
        vol_nano = nano_voxel_volume(tubes, grid_n=32, bounds=BOUNDS)
        assert abs(vol_fast - vol_nano) < 0.01


class TestNanoPerformance:
    @pytest.mark.slow
    def test_nano_mc_faster_than_fast(self):
        tubes = make_fan(8, 8, delta=0.08)
        n = 100_000

        import time
        t0 = time.perf_counter()
        fast_mc_volume(tubes, n_samples=n, seed=42)
        t_fast = time.perf_counter() - t0

        t0 = time.perf_counter()
        nano_mc_volume(tubes, n_samples=n, seed=42)
        t_nano = time.perf_counter() - t0

        if is_cpp_available():
            print(f"\n  MC: fast={t_fast*1e3:.2f}ms, nano={t_nano*1e3:.2f}ms, speedup={t_fast/t_nano:.2f}x")
            assert t_nano < t_fast * 1.5
