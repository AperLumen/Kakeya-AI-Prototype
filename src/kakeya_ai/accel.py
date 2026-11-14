"""几何内核的统一高性能后端调度器。

本模块自动挑选当前可用的最快后端：

1. ``nano`` —— float32 的 C++ 内核（``kakeya_ai._nano_kernel``）
2. ``fast`` —— float64 的 C++ 内核（``kakeya_ai._tube_kernel``）
3. ``numpy`` —— ``kakeya_ai.volume`` 里的参考 NumPy 实现

这个调度器刻意保持很薄：它不动现有的 ``fast_kernel`` 与 ``nano_kernel``
两个包装器，只为调用方提供一个统一入口，并在 C++ 扩展未编译时优雅降级。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import fast_kernel, nano_kernel, volume
from .primitives import Tube


def _resolve_backend(backend: str | None) -> str:
    """返回具体的后端名。"""
    if backend is not None:
        return backend
    if nano_kernel.is_cpp_available():
        return "nano"
    if fast_kernel.is_cpp_available():
        return "fast"
    return "numpy"


def get_backend() -> str:
    """返回当前选定的默认后端名。"""
    return _resolve_backend(None)


def volume_voxel(
    tubes: Sequence[Tube],
    grid_n: int = 128,
    bounds: np.ndarray | None = None,
    backend: str | None = None,
) -> float:
    """用所选后端做基于体素的并集体积估计。"""
    backend = _resolve_backend(backend)
    if backend == "nano":
        return nano_kernel.nano_voxel_volume(tubes, grid_n=grid_n, bounds=bounds)
    if backend == "fast":
        return fast_kernel.fast_voxel_volume(tubes, grid_n=grid_n, bounds=bounds)
    return volume.estimate_union_volume_voxel(tubes, grid_n=grid_n, bounds=bounds)


def volume_mc(
    tubes: Sequence[Tube],
    n_samples: int = 200_000,
    bounds: np.ndarray | None = None,
    seed: int | None = None,
    backend: str | None = None,
) -> float:
    """用所选后端做蒙特卡洛并集体积估计。"""
    backend = _resolve_backend(backend)
    if backend == "nano":
        return nano_kernel.nano_mc_volume(
            tubes, n_samples=n_samples, bounds=bounds, seed=seed
        )
    if backend == "fast":
        return fast_kernel.fast_mc_volume(
            tubes, n_samples=n_samples, bounds=bounds, seed=seed
        )
    return volume.estimate_union_volume_monte_carlo(
        tubes, n_samples=n_samples, bounds=bounds, seed=seed
    )


def contains_points(
    tubes: Sequence[Tube],
    points: np.ndarray,
    backend: str | None = None,
) -> np.ndarray:
    """用所选后端批量判断点是否落在管内。"""
    backend = _resolve_backend(backend)
    if backend == "nano":
        return nano_kernel.nano_contains_points(tubes, points)
    if backend == "fast":
        return fast_kernel.fast_contains_points(tubes, points)

    points = np.atleast_2d(np.asarray(points, dtype=float))
    if points.shape[1] != 3:
        points = points.reshape(-1, 3)
    mask = np.zeros(points.shape[0], dtype=bool)
    for t in tubes:
        mask |= t.contains(points)
    return mask
