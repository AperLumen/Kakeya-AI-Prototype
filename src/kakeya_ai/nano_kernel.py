"""C++ 纳秒级 float32 几何内核的 Python 绑定层。

这是 ``fast_kernel.py`` 的低精度/高吞吐版本：

- 输入/输出全部使用 float32，减少内存带宽并加倍 SIMD 宽度。
- 内层预计算 ``1 / |AB|^2``，把除法改成乘法。
- 使用无分支 ``fmin/fmax`` 做投影参数 clamp。
- 在 Apple M4 上经 -O3 -mcpu=apple-m4 -ffast-math 编译后，
  batch MC / voxel 可达到约 1 ns / tube-point pair。

当 C++ 扩展不可用时，回退到 fast_kernel（会做一次 float32 -> float64
转换，所以性能会下降，但结果一致）。

.. note::
   **本包装层是"运行时代码"，故留 `src/kakeya_ai/`**（被 ``accel.py`` 与
   包 ``__init__`` import）。与纳秒内核相关的**研发资产**——芯片预算模型
   ``chip_model.py`` 与隔离微基准——在仓库根的 ``nano/`` 目录（**刻意不在
   ``src/`` 内**，因为它们是判据/基准、不是运行时组件）。见 ``nano/README.md``。

使用方式
--------
    from kakeya_ai.nano_kernel import nano_mc_volume, nano_voxel_volume
    vol = nano_mc_volume(tubes, n_samples=200_000)
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import cpp_loader
from .primitives import Tube


#: 已加载的 C++ 扩展模块；未编译时为 None。
#:
#: 标注为 ``Any`` 而不是 ``ModuleType``：类型系统无法从 ``ModuleType`` 上
#: 解析出 C++ 侧 ``m.def`` 导出的名字，于是 ``_CPP_MODULE.foo(...)`` 依旧报
#: "不是 None 的已知属性"（本文件曾有 4 处）。接口签名见
#: ``_nano_kernel.pyi`` 存根。
_CPP_MODULE: Any = None
_CPP_AVAILABLE = False


def _load_cpp_module() -> Any:
    """加载 ``_nano_kernel`` 扩展；**三级回退，接口在 :mod:`cpp_loader`**。

    本函数是 :func:`kakeya_ai.cpp_loader.load_cpp_extension` 的薄封装，只负责把
    结果落进本模块的 ``_CPP_MODULE`` / ``_CPP_AVAILABLE``。三级回退（包内 →
    legacy ``cpp/`` → 顶层导入）与"失败返回 ``None``、绝不抛异常"的语义**只有
    一份实现**（``cpp_loader``），这正是局限 #22 要消掉的那份重复。
    """
    global _CPP_MODULE, _CPP_AVAILABLE

    module = cpp_loader.load_cpp_extension("_nano_kernel")
    _CPP_MODULE = module
    _CPP_AVAILABLE = module is not None
    return module


def is_cpp_available() -> bool:
    if _CPP_MODULE is None:
        _load_cpp_module()
    return _CPP_AVAILABLE


def get_backend() -> str:
    return "nano" if is_cpp_available() else "fast_fallback"


def cpu_topology() -> dict:
    """本机 CPU 拓扑与内核实际使用的并行线程数（芯片调度层的可观测入口）。

    返回 C++ 侧探测的字典：``p_cores`` / ``e_cores`` / ``total_cores`` /
    ``heterogeneous`` / ``num_threads``。P/E 核由 ``sysctl``（macOS）探测；
    异构芯片上 ``total_cores == p_cores + e_cores``。``num_threads`` 受
    ``KAKEYA_NUM_THREADS`` 覆盖（=1 时内核完全串行）。

    C++ 扩展未构建时返回空字典（不抛异常，与"可选加速缺失静默回退"一致）。
    """
    if not is_cpp_available():
        return {}
    return dict(_CPP_MODULE.topology())



def tubes_to_array_f32(tubes: Sequence[Tube]) -> np.ndarray:
    """把 Tube 列表转换为 (n_tubes, 8) 的 float32 数组。"""
    if not tubes:
        return np.empty((0, 8), dtype=np.float32)
    arr = np.empty((len(tubes), 8), dtype=np.float32)
    for i, t in enumerate(tubes):
        arr[i, 0:3] = t.center.astype(np.float32)
        arr[i, 3:6] = t.direction.v.astype(np.float32)
        arr[i, 6] = np.float32(t.length)
        arr[i, 7] = np.float32(t.delta)
    return arr


def nano_contains_points(
    tubes: Sequence[Tube],
    points: np.ndarray,
) -> np.ndarray:
    """批量判断点集是否被管族包含（float32 纳秒内核）。"""
    points = np.atleast_2d(np.asarray(points, dtype=np.float32))
    if points.shape[1] != 3:
        points = points.reshape(-1, 3)

    if not tubes:
        return np.zeros(points.shape[0], dtype=bool)

    if is_cpp_available():
        tube_arr = tubes_to_array_f32(tubes)
        return _CPP_MODULE.tube_contains_points_batch_f32(tube_arr, points)

    from .fast_kernel import fast_contains_points
    return fast_contains_points(tubes, points.astype(np.float64))


def nano_voxel_volume(
    tubes: Sequence[Tube],
    grid_n: int = 128,
    bounds: np.ndarray | None = None,
) -> float:
    """体素网格法估计 |∪T|（float32 纳秒内核）。"""
    if not tubes:
        return 0.0
    if bounds is None:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    bounds = np.asarray(bounds, dtype=np.float32).reshape(3, 2)

    if is_cpp_available():
        tube_arr = tubes_to_array_f32(tubes)
        return float(_CPP_MODULE.voxel_union_volume_f32(tube_arr, bounds, grid_n))

    from .fast_kernel import fast_voxel_volume
    return fast_voxel_volume(tubes, grid_n=grid_n, bounds=bounds.astype(np.float64))


def nano_mc_count_inside(
    tubes: Sequence[Tube],
    n_samples: int = 200_000,
    bounds: np.ndarray | None = None,
    seed: int = 0,
) -> int:
    """融合 MC 的**命中计数**（n_inside）——`with_ci` 等需要计数而非体积的调用方用。

    只走融合路径（片内 splitmix64 采样，点数组不进内存，能耗动机见
    PROJECT.md 第 III 篇 §6）。**要求显式 seed**：逐样本 RNG 只依赖
    (seed, j)，同 seed 结果逐位确定（与并行调度无关）。
    """
    if not tubes:
        return 0
    if bounds is None:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    bounds = np.asarray(bounds, dtype=np.float32).reshape(3, 2)
    if not is_cpp_available():
        raise RuntimeError(
            "nano 后端不可用（C++ 扩展未加载），无法做融合采样计数"
        )
    fused = getattr(_CPP_MODULE, "mc_union_volume_fused_f32", None)
    if fused is None:
        raise RuntimeError(
            "C++ 扩展缺少 mc_union_volume_fused_f32（.so 过旧？请重跑 "
            "bash cpp/build_nano.sh 重建）"
        )
    tube_arr = tubes_to_array_f32(tubes)
    return int(fused(tube_arr, bounds, int(n_samples), int(seed) % (1 << 63)))


def nano_mc_volume(
    tubes: Sequence[Tube],
    n_samples: int = 200_000,
    bounds: np.ndarray | None = None,
    seed: int | None = None,
) -> float:
    """蒙特卡洛法估计 |∪T|（float32 纳秒内核）。"""
    if not tubes:
        return 0.0
    if bounds is None:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
    bounds = np.asarray(bounds, dtype=np.float32).reshape(3, 2)

    box_volume = float(np.prod(bounds[:, 1] - bounds[:, 0]))

    if seed is not None and is_cpp_available():
        # 融合路径（2026-10-30）：点数组不进内存——采样在片内完成，
        # 逐样本 RNG 只依赖 (seed, j) ⇒ 同 seed 同结果（确定性）。
        return nano_mc_count_inside(tubes, n_samples, bounds, seed) / int(n_samples) * box_volume

    if is_cpp_available():
        tube_arr = tubes_to_array_f32(tubes)
        # 旧路径：numpy 生成点数组、跨 pybind 传入（seed=None 时保留
        # 原"每次随机"语义，也覆盖 .so 缺融合入口的回退）。
        rng = np.random.default_rng(seed)
        points = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, 3)).astype(np.float32)
        return float(_CPP_MODULE.mc_union_volume_f32(tube_arr, points, box_volume))

    from .fast_kernel import fast_mc_volume
    return fast_mc_volume(tubes, n_samples=n_samples, bounds=bounds.astype(np.float64), seed=seed)


def nano_tube_point_dist2(
    tube: Tube,
    point: np.ndarray,
) -> float:
    """单点距离平方（pybind11 开销会占主导，适合验证用）。"""
    p = np.asarray(point, dtype=np.float32).reshape(3)
    c = tube.center.astype(np.float32)
    d = tube.direction.v.astype(np.float32)
    if is_cpp_available():
        return float(_CPP_MODULE.single_tube_point_dist2_f32(
            p[0], p[1], p[2],
            c[0], c[1], c[2],
            d[0], d[1], d[2],
            np.float32(tube.length)
        ))
    return float(np.sum((p - c) ** 2))  # 粗略兜底
