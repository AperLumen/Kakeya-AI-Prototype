"""对既有 C++ tube 内核做微基准，建立 **ns/op 基线**。"""
from __future__ import annotations

import time
import numpy as np

from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.fast_kernel import fast_contains_points, fast_mc_volume, fast_voxel_volume, is_cpp_available, get_backend


def make_tubes(n_theta=4, n_phi=4, delta=0.1):
    tubes = []
    for theta in np.linspace(0.2, np.pi - 0.2, n_theta):
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False):
            d = Direction.from_spherical(theta, phi)
            tubes.append(Tube(center=np.zeros(3), direction=d, length=1.0, delta=delta))
    return tubes


def bench_contains_points():
    tubes = make_tubes(4, 4, delta=0.1)
    points = np.random.randn(1000, 3).astype(np.float64)
    points /= np.linalg.norm(points, axis=1, keepdims=True) * 2

    # Warmup
    for _ in range(3):
        fast_contains_points(tubes, points)

    n_iter = 100
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fast_contains_points(tubes, points)
    t = (time.perf_counter() - t0) / n_iter

    n_ops = len(tubes) * len(points)
    ns_per_op = t * 1e9 / n_ops
    print(f"contains_points: {t*1e6:.1f} us per call, {ns_per_op:.1f} ns per tube-point pair")


def bench_mc_volume():
    tubes = make_tubes(4, 4, delta=0.1)
    n_samples = 100_000

    # Warmup
    for _ in range(3):
        fast_mc_volume(tubes, n_samples=n_samples, seed=42)

    n_iter = 20
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fast_mc_volume(tubes, n_samples=n_samples, seed=42)
    t = (time.perf_counter() - t0) / n_iter

    n_ops = len(tubes) * n_samples
    ns_per_op = t * 1e9 / n_ops
    print(f"mc_volume: {t*1e3:.1f} ms per call, {ns_per_op:.1f} ns per tube-point pair")


def bench_voxel_volume():
    tubes = make_tubes(4, 4, delta=0.1)
    grid_n = 32

    # Warmup
    for _ in range(3):
        fast_voxel_volume(tubes, grid_n=grid_n)

    n_iter = 10
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fast_voxel_volume(tubes, grid_n=grid_n)
    t = (time.perf_counter() - t0) / n_iter

    n_ops = len(tubes) * (grid_n ** 3)
    ns_per_op = t * 1e9 / n_ops
    print(f"voxel_volume: {t*1e3:.1f} ms per call, {ns_per_op:.1f} ns per tube-voxel pair")


if __name__ == "__main__":
    print(f"Backend: {get_backend()}, C++ available: {is_cpp_available()}")
    bench_contains_points()
    bench_mc_volume()
    bench_voxel_volume()
