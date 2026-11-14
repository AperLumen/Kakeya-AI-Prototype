"""统一加速调度器的测试。"""

import numpy as np
import pytest

from kakeya_ai import accel
from kakeya_ai.primitives import Direction, Tube


BOUNDS = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
BOUNDS_F32 = BOUNDS.astype(np.float32)


def make_fan(n_theta=4, n_phi=4, delta=0.1) -> list[Tube]:
    tubes = []
    for theta in np.linspace(0.2, np.pi - 0.2, n_theta):
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False):
            d = Direction.from_spherical(theta, phi)
            tubes.append(Tube(center=np.zeros(3), direction=d, length=1.0, delta=delta))
    return tubes


class TestBackendSelection:
    def test_get_backend_returns_valid_name(self):
        assert accel.get_backend() in ("nano", "fast", "numpy")


class TestAccelVolumeVoxel:
    def test_all_backends_agree(self):
        tubes = make_fan(4, 4, delta=0.1)
        vols = {}
        for backend in ("nano", "fast", "numpy"):
            try:
                vols[backend] = accel.volume_voxel(
                    tubes, grid_n=32, bounds=BOUNDS_F32 if backend == "nano" else BOUNDS, backend=backend
                )
            except Exception as exc:
                pytest.skip(f"{backend} backend unavailable: {exc}")

        if "numpy" in vols:
            for name, vol in vols.items():
                assert vol > 0
                if name != "numpy":
                    assert abs(vol - vols["numpy"]) < 0.01

    def test_empty_tubes(self):
        assert accel.volume_voxel([], grid_n=32, backend="numpy") == 0.0


class TestAccelVolumeMC:
    def test_all_backends_agree(self):
        tubes = make_fan(4, 4, delta=0.1)
        vols = {}
        # n_samples=500k：2026-10-30 起 nano 走融合路径（片内独立随机流），
        # 与 numpy/fast 的双采样噪声须压进 0.01 容差（N=50k 时 σ≈0.013 会抖动）。
        for backend in ("nano", "fast", "numpy"):
            try:
                vols[backend] = accel.volume_mc(
                    tubes, n_samples=500_000, bounds=BOUNDS_F32 if backend == "nano" else BOUNDS, seed=42, backend=backend
                )
            except Exception as exc:
                pytest.skip(f"{backend} backend unavailable: {exc}")

        if "numpy" in vols:
            for name, vol in vols.items():
                assert vol > 0
                if name != "numpy":
                    assert abs(vol - vols["numpy"]) < 0.01

    def test_empty_tubes(self):
        assert accel.volume_mc([], n_samples=1000, backend="numpy") == 0.0


class TestAccelContainsPoints:
    def test_all_backends_agree(self):
        tubes = make_fan(4, 4, delta=0.1)
        rng = np.random.default_rng(42)
        points = rng.uniform(-1, 1, (500, 3))
        masks = {}
        for backend in ("nano", "fast", "numpy"):
            try:
                masks[backend] = accel.contains_points(tubes, points, backend=backend)
            except Exception as exc:
                pytest.skip(f"{backend} backend unavailable: {exc}")

        if "numpy" in masks:
            for name, mask in masks.items():
                assert mask.shape == (500,)
                assert mask.dtype == bool
                if name != "numpy":
                    agreement = np.mean(mask == masks["numpy"])
                    assert agreement > 0.95

    def test_empty_tubes(self):
        points = np.random.rand(10, 3)
        mask = accel.contains_points([], points, backend="numpy")
        assert mask.sum() == 0
