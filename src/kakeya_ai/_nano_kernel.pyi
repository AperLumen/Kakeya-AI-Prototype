"""类型存根：``kakeya_ai._nano_kernel``（C++ float32 几何内核）。

与 ``_tube_kernel.pyi`` 同理：pybind11 编译产物，源码树里没有 Python 实现，
静态分析必须靠存根才能知道它有哪些函数。

**精度差异是这层的重点**：本模块全部走 float32。之所以能用 f32 而仍与
float64 的 numpy 后端**逐位一致**，是因为域半径取 2 的幂、grid_n 也取 2 的幂，
使两者的体素网格在两种精度下精确对齐（见 ``domain.py`` 的三条不变量）。
存根无法表达"调用方必须传 f32 数组"这类精度约束——那是运行期检查，
用文档说明。

签名与 ``cpp/kernel/nano_kernel.cpp`` 的 ``PYBIND11_MODULE`` 段一一对应。
"""

from __future__ import annotations

import numpy as np

def tube_contains_points_batch_f32(
    tubes: np.ndarray, points: np.ndarray
) -> np.ndarray: ...

def voxel_union_volume_f32(
    tubes: np.ndarray, bounds: np.ndarray, grid_n: int
) -> float: ...

def mc_union_volume_f32(
    tubes: np.ndarray, points: np.ndarray, box_volume: float
) -> float: ...

def single_tube_point_dist2_f32(
    tube: np.ndarray, point: np.ndarray
) -> float: ...
