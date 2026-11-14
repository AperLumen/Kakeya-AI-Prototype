"""类型存根：``kakeya_ai._tube_kernel``（C++ float64 几何内核）。

**为什么需要这个文件。** 该模块是 pybind11 编译产物（``cpp/kernel/tube_kernel.cpp``），
只在 ``bash cpp/build_nano.sh`` 之后才存在。源代码树里没有它的 Python 实现，
因此 Pylance **必然**报"无法解析导入 ``_tube_kernel``"，且 ``_CPP_MODULE``
的属性访问全部报"不是 None 的已知属性"——静态分析看不到 C++ 侧 ``m.def``
导出的名字。

这不是代码缺陷，而是"可选编译产物"这一事实在静态分析上的表现。但让编辑器
常驻一屏红字没有意义：它掩盖真问题（本项目此前有 1164 条 → 12 条的教训）。
正确做法是用存根**如实描述**扩展接口——类型系统据此知道模块与函数存在，
而运行时行为完全不受影响（``.pyi`` 不参与执行）。

**签名与 C++ 侧一一对应**，见 ``cpp/kernel/tube_kernel.cpp`` 的 ``PYBIND11_MODULE``
段。改动 C++ 导出时必须同步本文件，否则类型信息会与实现漂移。

**注意数组精度**：本模块是 float64 版；float32 版见 ``_nano_kernel``。
传错精度的数组不会报类型错（``numpy`` 的 dtype 不在类型系统里），
而是可能在 C++ 侧触发隐式转换——这是运行期约束，用文档说明。
"""

from __future__ import annotations

import numpy as np

def tube_contains_points_batch(
    tubes: np.ndarray, points: np.ndarray
) -> np.ndarray: ...

def voxel_union_volume(
    tubes: np.ndarray, bounds: np.ndarray, grid_n: int
) -> float: ...

def mc_union_volume(
    tubes: np.ndarray, points: np.ndarray, box_volume: float
) -> float: ...

def tube_pair_intersect(
    tubes_a: np.ndarray, tubes_b: np.ndarray
) -> np.ndarray: ...

def assign_tubes_to_grid_cells(
    tube_bboxes: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray]: ...
