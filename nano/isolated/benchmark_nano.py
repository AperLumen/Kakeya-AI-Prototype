"""新 float32 ``nano_kernel`` 的正确性验证与微基准。"""
from __future__ import annotations

import time
import numpy as np
import sys
import os

# 从 cpp/ 目录加载 nano_kernel.so
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cpp"))
import nano_kernel as nk

from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.fast_kernel import (
    fast_contains_points,
    fast_mc_volume,
    fast_voxel_volume,
)


def make_tubes(n_theta=4, n_phi=4, delta=0.1):
    tubes = []
    for theta in np.linspace(0.2, np.pi - 0.2, n_theta):
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False):
            d = Direction.from_spherical(theta, phi)
            tubes.append(Tube(center=np.zeros(3), direction=d, length=1.0, delta=delta))
    return tubes


def tubes_to_array_f32(tubes):
    arr = np.empty((len(tubes), 8), dtype=np.float32)
    for i, t in enumerate(tubes):
        arr[i, 0:3] = t.center.astype(np.float32)
        arr[i, 3:6] = t.direction.v.astype(np.float32)
        arr[i, 6] = np.float32(t.length)
        arr[i, 7] = np.float32(t.delta)
    return arr


def test_contains_points():
    tubes = make_tubes(4, 4, delta=0.1)
    rng = np.random.default_rng(42)
    points = rng.uniform(-1, 1, (2000, 3)).astype(np.float32)

    mask_fast = fast_contains_points(tubes, points)

    tube_arr = tubes_to_array_f32(tubes)
    mask_nano = nk.tube_contains_points_batch_f32(tube_arr, points)

    agree = np.mean(mask_fast == mask_nano)
    print(f"contains_points agreement: {agree*100:.2f}%")
    assert agree > 0.95, f"too low agreement: {agree}"


def test_mc_volume():
    tubes = make_tubes(4, 4, delta=0.1)
    n = 50_000
    bounds = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=np.float32)
    rng = np.random.default_rng(42)
    points = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    box_volume = float(np.prod(bounds[:, 1] - bounds[:, 0]))

    vol_fast = fast_mc_volume(tubes, n_samples=n, seed=42)
    vol_nano = nk.mc_union_volume_f32(tubes_to_array_f32(tubes), points, box_volume)
    print(f"MC volume: fast={vol_fast:.6f}, nano={vol_nano:.6f}, diff={abs(vol_fast-vol_nano):.6f}")
    assert abs(vol_fast - vol_nano) < 0.05


def test_voxel_volume():
    tubes = make_tubes(4, 4, delta=0.1)
    bounds = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=np.float32)

    vol_fast = fast_voxel_volume(tubes, grid_n=32, bounds=bounds.astype(np.float64))
    vol_nano = nk.voxel_union_volume_f32(tubes_to_array_f32(tubes), bounds, 32)
    print(f"Voxel volume: fast={vol_fast:.6f}, nano={vol_nano:.6f}, diff={abs(vol_fast-vol_nano):.6f}")
    assert abs(vol_fast - vol_nano) < 0.05


def benchmark_contains_points():
    tubes = make_tubes(8, 8, delta=0.08)
    points = np.random.randn(10000, 3).astype(np.float32)
    points /= np.linalg.norm(points, axis=1, keepdims=True) * 2
    tube_arr = tubes_to_array_f32(tubes)

    # Warmup
    for _ in range(3):
        nk.tube_contains_points_batch_f32(tube_arr, points)

    n_iter = 200
    t0 = time.perf_counter()
    for _ in range(n_iter):
        nk.tube_contains_points_batch_f32(tube_arr, points)
    t = (time.perf_counter() - t0) / n_iter

    n_ops = len(tubes) * len(points)
    ns_per_op = t * 1e9 / n_ops
    print(f"[NANO] contains_points: {t*1e6:.1f} us/call, {ns_per_op:.2f} ns/tube-point pair")


def benchmark_mc_volume():
    tubes = make_tubes(8, 8, delta=0.08)
    n = 200_000
    bounds = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=np.float32)
    rng = np.random.default_rng(42)
    points = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    box_volume = float(np.prod(bounds[:, 1] - bounds[:, 0]))
    tube_arr = tubes_to_array_f32(tubes)

    for _ in range(3):
        nk.mc_union_volume_f32(tube_arr, points, box_volume)

    n_iter = 50
    t0 = time.perf_counter()
    for _ in range(n_iter):
        nk.mc_union_volume_f32(tube_arr, points, box_volume)
    t = (time.perf_counter() - t0) / n_iter

    n_ops = len(tubes) * n
    ns_per_op = t * 1e9 / n_ops
    print(f"[NANO] mc_volume: {t*1e3:.1f} ms/call, {ns_per_op:.2f} ns/tube-point pair")


def benchmark_voxel_volume():
    tubes = make_tubes(8, 8, delta=0.08)
    grid_n = 48
    bounds = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=np.float32)
    tube_arr = tubes_to_array_f32(tubes)

    for _ in range(3):
        nk.voxel_union_volume_f32(tube_arr, bounds, grid_n)

    n_iter = 20
    t0 = time.perf_counter()
    for _ in range(n_iter):
        nk.voxel_union_volume_f32(tube_arr, bounds, grid_n)
    t = (time.perf_counter() - t0) / n_iter

    n_ops = len(tubes) * (grid_n ** 3)
    ns_per_op = t * 1e9 / n_ops
    print(f"[NANO] voxel_volume: {t*1e3:.1f} ms/call, {ns_per_op:.2f} ns/tube-voxel pair")


def benchmark_single_dist():
    # 测单次标量调用的延迟（此时开销主要由 pybind11 绑定层主导）。
    px, py, pz = 0.1, 0.2, 0.3
    cx, cy, cz = 0.0, 0.0, 0.0
    dx, dy, dz = 1.0, 0.0, 0.0
    length = 1.0

    # Warmup
    for _ in range(100):
        nk.single_tube_point_dist2_f32(px, py, pz, cx, cy, cz, dx, dy, dz, length)

    n_iter = 10_000
    t0 = time.perf_counter()
    for _ in range(n_iter):
        nk.single_tube_point_dist2_f32(px, py, pz, cx, cy, cz, dx, dy, dz, length)
    t = (time.perf_counter() - t0) / n_iter
    print(f"[NANO] single scalar dist2: {t*1e9:.1f} ns/call")


if __name__ == "__main__":
    print("Correctness tests")
    test_contains_points()
    test_mc_volume()
    test_voxel_volume()
    print()
    print("Nano benchmarks")
    benchmark_contains_points()
    benchmark_mc_volume()
    benchmark_voxel_volume()
    benchmark_single_dist()
