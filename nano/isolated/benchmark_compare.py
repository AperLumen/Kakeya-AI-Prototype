"""并排对比：既有 C++ f64 内核 vs 新的 float32 nano 内核。"""
from __future__ import annotations

import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cpp"))
import nano_kernel as nk

from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.fast_kernel import (
    fast_contains_points,
    fast_mc_volume,
    fast_voxel_volume,
)


def make_tubes(n_theta=8, n_phi=8, delta=0.08):
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


def benchmark_contains_points(tubes, n_points=10000):
    points_f64 = np.random.randn(n_points, 3)
    points_f64 /= np.linalg.norm(points_f64, axis=1, keepdims=True) * 2
    points_f32 = points_f64.astype(np.float32)
    tube_arr = tubes_to_array_f32(tubes)

    for _ in range(3):
        fast_contains_points(tubes, points_f64)
        nk.tube_contains_points_batch_f32(tube_arr, points_f32)

    n_iter = 200
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fast_contains_points(tubes, points_f64)
    t_fast = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        nk.tube_contains_points_batch_f32(tube_arr, points_f32)
    t_nano = (time.perf_counter() - t0) / n_iter

    n_ops = len(tubes) * n_points
    print(f"contains_points ({len(tubes)} tubes x {n_points} points)")
    print(f"  f64 fast_kernel: {t_fast*1e6:7.1f} us/call, {t_fast*1e9/n_ops:5.2f} ns/pair")
    print(f"  f32 nano_kernel: {t_nano*1e6:7.1f} us/call, {t_nano*1e9/n_ops:5.2f} ns/pair")
    print(f"  speedup: {t_fast/t_nano:.2f}x")


def benchmark_mc_volume(tubes, n_samples=200_000):
    bounds = np.array([[-1, 1], [-1, 1], [-1, 1]])
    bounds_f32 = bounds.astype(np.float32)
    rng = np.random.default_rng(42)
    points_f32 = rng.uniform(-1, 1, (n_samples, 3)).astype(np.float32)
    box_volume = float(np.prod(bounds[:, 1] - bounds[:, 0]))
    tube_arr = tubes_to_array_f32(tubes)

    for _ in range(3):
        fast_mc_volume(tubes, n_samples=n_samples, seed=42)
        nk.mc_union_volume_f32(tube_arr, points_f32, box_volume)

    n_iter = 50
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fast_mc_volume(tubes, n_samples=n_samples, seed=42)
    t_fast = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        nk.mc_union_volume_f32(tube_arr, points_f32, box_volume)
    t_nano = (time.perf_counter() - t0) / n_iter

    n_ops = len(tubes) * n_samples
    print(f"mc_volume ({len(tubes)} tubes x {n_samples} samples)")
    print(f"  f64 fast_kernel: {t_fast*1e3:7.1f} ms/call, {t_fast*1e9/n_ops:5.2f} ns/pair")
    print(f"  f32 nano_kernel: {t_nano*1e3:7.1f} ms/call, {t_nano*1e9/n_ops:5.2f} ns/pair")
    print(f"  speedup: {t_fast/t_nano:.2f}x")


def benchmark_voxel_volume(tubes, grid_n=48):
    bounds = np.array([[-1, 1], [-1, 1], [-1, 1]])
    bounds_f32 = bounds.astype(np.float32)
    tube_arr = tubes_to_array_f32(tubes)

    for _ in range(3):
        fast_voxel_volume(tubes, grid_n=grid_n, bounds=bounds)
        nk.voxel_union_volume_f32(tube_arr, bounds_f32, grid_n)

    n_iter = 20
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fast_voxel_volume(tubes, grid_n=grid_n, bounds=bounds)
    t_fast = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        nk.voxel_union_volume_f32(tube_arr, bounds_f32, grid_n)
    t_nano = (time.perf_counter() - t0) / n_iter

    n_ops = len(tubes) * (grid_n ** 3)
    print(f"voxel_volume ({len(tubes)} tubes x {grid_n}^3 voxels)")
    print(f"  f64 fast_kernel: {t_fast*1e3:7.1f} ms/call, {t_fast*1e9/n_ops:5.2f} ns/pair")
    print(f"  f32 nano_kernel: {t_nano*1e3:7.1f} ms/call, {t_nano*1e9/n_ops:5.2f} ns/pair")
    print(f"  speedup: {t_fast/t_nano:.2f}x")


if __name__ == "__main__":
    tubes = make_tubes(8, 8, delta=0.08)
    print(f"Tube count: {len(tubes)}")
    print()
    benchmark_contains_points(tubes)
    print()
    benchmark_mc_volume(tubes)
    print()
    benchmark_voxel_volume(tubes)
