"""C++ 高性能几何内核的 Python 绑定层。

通过 pybind11 编译的 C++ 模块提供以下热点路径的加速：

- ``tube_contains_points_batch`` — 批量管-点包含测试
- ``voxel_union_volume`` — 体素网格并集体积估计
- ``mc_union_volume`` — 蒙特卡洛并集体积估计
- ``tube_pair_intersect`` — 管对相交测试

当 C++ 扩展不可用时（未编译），自动回退到纯 NumPy 实现。

使用方式
--------
    from kakeya_ai.fast_kernel import fast_voxel_volume, fast_mc_volume
    vol = fast_voxel_volume(tubes, grid_n=128)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import cpp_loader
from .primitives import Tube


# ---------------------------------------------------------------------------
# 加载 C++ 扩展
# ---------------------------------------------------------------------------

#: 已加载的 C++ 扩展模块；未编译时为 None。
#:
#: 标注为 ``Any`` 而不是 ``ModuleType``。``ModuleType`` 听起来更准确，但
#: 类型系统**无法**从一个 ``ModuleType`` 上解析出 C++ 侧 ``m.def`` 导出的
#: 名字——它只知道模块对象的标准属性（``__name__`` 等）。于是
#: ``_CPP_MODULE.foo(...)`` 依旧报"nonexistent attribute"（本文件曾有 4 处）。
#: C++ 扩展的接口无法从源码推导，只能如实承认"这里我们不掌握类型信息"。
#: 接口的真实签名记录在 ``_tube_kernel.pyi`` 存根里，供人查阅。
_CPP_MODULE: Any = None
_CPP_AVAILABLE = False


def _load_cpp_module() -> Any:
    """加载 ``_tube_kernel`` 扩展；**三级回退，接口在 :mod:`cpp_loader`**。

    本函数是 :func:`kakeya_ai.cpp_loader.load_cpp_extension` 的薄封装，只负责把
    结果落进本模块的 ``_CPP_MODULE`` / ``_CPP_AVAILABLE``（其余两个加载点——
    ``nano_kernel`` / ``grains_cpp``——同样如此）。三级回退（包内 → legacy
    ``cpp/`` → 顶层导入）与"失败返回 ``None``、绝不抛异常"的语义**只有一份
    实现**，见 ``cpp_loader``；这正是局限 #22 要消掉的那份重复。
    """
    global _CPP_MODULE, _CPP_AVAILABLE

    module = cpp_loader.load_cpp_extension("_tube_kernel")
    _CPP_MODULE = module
    _CPP_AVAILABLE = module is not None
    return module


def is_cpp_available() -> bool:
    """C++ 加速内核是否可用。"""
    if _CPP_MODULE is None:
        _load_cpp_module()
    return _CPP_AVAILABLE


def get_backend() -> str:
    """返回当前使用的后端名称。"""
    return "cpp" if is_cpp_available() else "numpy"


# ---------------------------------------------------------------------------
# 管族 → 数组转换
# ---------------------------------------------------------------------------

def tubes_to_array(tubes: Sequence[Tube]) -> np.ndarray:
    """把 Tube 列表转换为 (n_tubes, 8) 的 float64 数组。

    每行: [center_x, center_y, center_z, dir_x, dir_y, dir_z, length, delta]
    """
    if not tubes:
        return np.empty((0, 8), dtype=np.float64)
    arr = np.empty((len(tubes), 8), dtype=np.float64)
    for i, t in enumerate(tubes):
        arr[i, 0] = t.center[0]
        arr[i, 1] = t.center[1]
        arr[i, 2] = t.center[2]
        arr[i, 3] = t.direction.v[0]
        arr[i, 4] = t.direction.v[1]
        arr[i, 5] = t.direction.v[2]
        arr[i, 6] = t.length
        arr[i, 7] = t.delta
    return arr


# ---------------------------------------------------------------------------
# 加速 API
# ---------------------------------------------------------------------------

def fast_contains_points(
    tubes: Sequence[Tube],
    points: np.ndarray,
) -> np.ndarray:
    """批量判断点集是否被管族包含。

    Parameters
    ----------
    tubes : sequence of Tube
    points : (n_points, 3) array

    Returns
    -------
    mask : (n_points,) bool array
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if points.shape[1] != 3:
        points = points.reshape(-1, 3)

    if not tubes:
        return np.zeros(points.shape[0], dtype=bool)

    if is_cpp_available():
        tube_arr = tubes_to_array(tubes)
        return _CPP_MODULE.tube_contains_points_batch(tube_arr, points)

    # NumPy 回退
    mask = np.zeros(points.shape[0], dtype=bool)
    for t in tubes:
        mask |= t.contains(points)
    return mask


def fast_voxel_volume(
    tubes: Sequence[Tube],
    grid_n: int = 128,
    bounds: np.ndarray | None = None,
) -> float:
    """体素网格法估计 |∪T|（C++ 加速）。

    Parameters
    ----------
    tubes : sequence of Tube
    grid_n : int
        每轴体素数。
    bounds : (3, 2) array, optional
        包围盒。

    Returns
    -------
    volume : float
    """
    if not tubes:
        return 0.0
    if bounds is None:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    bounds = np.asarray(bounds, dtype=np.float64).reshape(3, 2)

    if is_cpp_available():
        tube_arr = tubes_to_array(tubes)
        return float(_CPP_MODULE.voxel_union_volume(tube_arr, bounds, grid_n))

    # NumPy 回退
    from .volume import estimate_union_volume_voxel
    return estimate_union_volume_voxel(tubes, grid_n=grid_n, bounds=bounds)


def fast_mc_volume(
    tubes: Sequence[Tube],
    n_samples: int = 200_000,
    bounds: np.ndarray | None = None,
    seed: int | None = None,
) -> float:
    """蒙特卡洛法估计 |∪T|（C++ 加速）。

    Parameters
    ----------
    tubes : sequence of Tube
    n_samples : int
    bounds : (3, 2) array, optional
    seed : int, optional

    Returns
    -------
    volume : float
    """
    if not tubes:
        return 0.0
    if bounds is None:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    bounds = np.asarray(bounds, dtype=np.float64).reshape(3, 2)

    rng = np.random.default_rng(seed)
    points = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, 3))
    box_volume = float(np.prod(bounds[:, 1] - bounds[:, 0]))

    if is_cpp_available():
        tube_arr = tubes_to_array(tubes)
        return float(_CPP_MODULE.mc_union_volume(tube_arr, points, box_volume))

    # NumPy 回退
    inside = np.zeros(n_samples, dtype=bool)
    for t in tubes:
        inside |= t.contains(points)
    return float(np.count_nonzero(inside)) / n_samples * box_volume


def fast_tube_intersect(
    tubes_a: Sequence[Tube],
    tubes_b: Sequence[Tube],
) -> np.ndarray:
    """管对相交测试（C++ 加速）。

    Returns
    -------
    intersect_matrix : (n_a, n_b) bool array
    """
    if not tubes_a or not tubes_b:
        return np.zeros((len(tubes_a), len(tubes_b)), dtype=bool)

    if is_cpp_available():
        arr_a = tubes_to_array(tubes_a)
        arr_b = tubes_to_array(tubes_b)
        return _CPP_MODULE.tube_pair_intersect(arr_a, arr_b)

    # NumPy 回退（简化判定）
    n_a, n_b = len(tubes_a), len(tubes_b)
    result = np.zeros((n_a, n_b), dtype=bool)
    for i, ta in enumerate(tubes_a):
        a1, a2 = ta.endpoints
        for j, tb in enumerate(tubes_b):
            b1, b2 = tb.endpoints
            # 简化：检查端点到另一管中线的距离
            for p in [b1, b2]:
                ab = a2 - a1
                ab_len2 = float(np.dot(ab, ab))
                if ab_len2 < 1e-24:
                    dist = np.linalg.norm(p - a1)
                else:
                    t = np.clip(np.dot(p - a1, ab) / ab_len2, 0, 1)
                    closest = a1 + t * ab
                    dist = np.linalg.norm(p - closest)
                if dist <= ta.delta + tb.delta:
                    result[i, j] = True
                    break
    return result
