"""grains 分解机制（骨架版本）。

本模块提供 Guth grains 分解，以及 Wang–Zahl 第 7 节所述两尺度 grains
分解的数据结构与一个简化实现。

它刻意只是一个"骨架"：完整算法需要 broadness 归约、正则遮蔽与多尺度
平衡。 这里我们只提供之后构建这些所需的基础原语。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .primitives import Shading, Tube
from .errors import InputValidationError
from .wolff_axioms import Box

#: 一个 grain 的几何接口——轴对齐（:class:`Prism`）或定向（:class:`OrientedPrism`）。
#: 二者都提供 ``a``/``b``/``c``/``rho``/``volume``/``dimensions``/``bounds`` 与
#: 管索引、密度、``local_kt_error``、ρ-bin 元数据。用联合类型而非 Protocol，
#: 因为下游（``moves`` / 绘图 / 度量）需要**实例判别**（``isinstance``）来区分
#: "能否走世界轴对齐的合并路径"。
Grain = "Prism | OrientedPrism"


@dataclass(frozen=True)
class Prism(Box):
    """长方体的 grain 棱柱，附带一组与之关联的管以及遮蔽。

    维度为 a × b × c，且 a ≤ b ≤ c。 该棱柱保存它所要组织的那些 δ-管的
    下标。
    """

    tube_indices: Tuple[int, ...] = field(default_factory=tuple)
    density: float = 1.0
    local_kt_error: float = 0.0
    # 可选的元数据，把该 grain 关联回它的父 ρ-管。 Move #1 用它们来
    # 评估逐 bin 的目标
    #   c ≥ δ^ζ (ρ/δ) (#T_ρ / #T)。
    bin_index: int | None = None
    bin_tube_count: int = 0
    total_tube_count: int = 0
    bin_direction: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        if not (self.dimensions[0] <= self.dimensions[1] <= self.dimensions[2]):
            # 重排坐标轴，使 a <= b <= c。
            object.__setattr__(self, "bounds", self.bounds[np.argsort(self.dimensions)])

    @property
    def a(self) -> float:
        return float(self.dimensions[0])

    @property
    def b(self) -> float:
        return float(self.dimensions[1])

    @property
    def c(self) -> float:
        return float(self.dimensions[2])

    @property
    def rho(self) -> float:
        """ρ = b / c，即 grain 的偏心率参数。"""
        return self.b / self.c if self.c > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"Prism(a={self.a:.4g}, b={self.b:.4g}, c={self.c:.4g}, "
            f"rho={self.rho:.4g}, tubes={len(self.tube_indices)})"
        )


@dataclass(frozen=True)
class OrientedPrism:
    """**定向**棱柱 grain：沿自身正交基的盒子，**按定向体积汇报**。

    **为什么需要它**（这是兑现"更贴近论文几何"的关键，2026-10-09）。grain 的
    几何在论文里是沿管方向拉长的盒子；用世界轴对齐盒（:class:`Prism`）表示斜向
    grain 只能靠**放很大**才装得下（斜向盒的世界 AABB 比盒本身大得多）。先前记录
    的负结果（总体积被放大 ~30 倍）正是"把 cell 建在定向坐标系、却按世界 AABB
    汇报体积"造成的——**问题在于汇报口径，不在于坐标系**。

    本类把两者解耦：几何用 ``center`` / ``axes`` / ``extents`` 描述，
    :attr:`volume` 返回**定向盒自身的体积** ``a·b·c``（精确，与世界朝向无关），
    而 :attr:`bounds` 仍给出**世界 AABB**（保守外扩），供只接受轴对齐框的既有
    相交测试使用。于是 ``total_prism_volume`` / ``local_kt_error`` 用真实体积，
    而相交判定不会漏管。

    ``axes`` 的行是正交归一基（``axes[2]`` 为长轴 e_c，``axes[0]`` 为短轴 e_a）；
    ``extents`` 按非降序、与 ``axes`` 的三行一一对应。
    """

    center: np.ndarray                      # (3,) 世界坐标
    axes: np.ndarray                        # (3, 3)，行 = 正交归一基（[2] = 长轴）
    extents: np.ndarray                     # (3,)，沿三轴的长度，非降序
    tube_indices: Tuple[int, ...] = field(default_factory=tuple)
    density: float = 1.0
    local_kt_error: float = 0.0
    bin_index: int | None = None
    bin_tube_count: int = 0
    total_tube_count: int = 0
    bin_direction: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float).reshape(3))
        ax = np.asarray(self.axes, dtype=float).reshape(3, 3)
        e = np.asarray(self.extents, dtype=float).reshape(3)
        if np.any(e <= 0):
            raise InputValidationError("OrientedPrism extents must be positive.")
        order = np.argsort(e)                 # a ≤ b ≤ c，与 axes 行同步重排
        object.__setattr__(self, "extents", e[order])
        object.__setattr__(self, "axes", ax[order])

    # ---- 形状 / 度量 ---------------------------------------------------
    @property
    def a(self) -> float:
        return float(self.extents[0])

    @property
    def b(self) -> float:
        return float(self.extents[1])

    @property
    def c(self) -> float:
        return float(self.extents[2])

    @property
    def rho(self) -> float:
        """ρ = b / c（与 :class:`Prism` 同口径）。"""
        return self.b / self.c if self.c > 0 else 0.0

    @property
    def volume(self) -> float:
        """**定向盒自身的体积** ``a·b·c``（精确；不取世界 AABB）。"""
        return float(np.prod(self.extents))

    @property
    def dimensions(self) -> np.ndarray:
        """沿三轴的边长（非降序）——与 :class:`Prism` 的 ``dimensions`` 语义对应，
        但**不是**世界 AABB 边长。"""
        return self.extents

    @property
    def bounds(self) -> np.ndarray:
        """世界 AABB 的 ``(3, 2)`` [[lo],[hi]]（保守外扩，供轴对齐相交测试用）。"""
        half = np.abs(self.axes.T) @ (0.5 * self.extents)
        return np.column_stack([self.center - half, self.center + half])

    def to_local(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        return (p - self.center) @ self.axes.T

    def to_world(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        return p @ self.axes + self.center

    def contains_segment(self, a: np.ndarray, b: np.ndarray, delta: float) -> bool:
        """δ-邻域是否完全落在**定向盒**内（精确，用局部坐标）。"""
        pts = self.to_local(np.vstack([np.asarray(a), np.asarray(b)]))
        lo = pts.min(axis=0) - delta
        hi = pts.max(axis=0) + delta
        return bool(np.all(lo >= -0.5 * self.extents - 1e-12)
                    and np.all(hi <= 0.5 * self.extents + 1e-12))

    def intersects_segment(self, a: np.ndarray, b: np.ndarray) -> bool:
        """线段是否与定向盒相交（局部坐标下的 Liang–Barsky）。"""
        pts = self.to_local(np.vstack([np.asarray(a), np.asarray(b)]))
        half = 0.5 * self.extents
        t_min, t_max = 0.0, 1.0
        for i in range(3):
            d = pts[1, i] - pts[0, i]
            if abs(d) < 1e-15:
                if pts[0, i] < -half[i] or pts[0, i] > half[i]:
                    return False
            else:
                t1 = (-half[i] - pts[0, i]) / d
                t2 = (half[i] - pts[0, i]) / d
                te, tx = (t1, t2) if t1 < t2 else (t2, t1)
                t_min, t_max = max(t_min, te), min(t_max, tx)
                if t_min > t_max:
                    return False
        return True

    def corners(self) -> np.ndarray:
        """8 个角点（世界坐标）。"""
        signs = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ], dtype=float)
        return self.to_world(signs * (0.5 * self.extents))

    def __repr__(self) -> str:
        return (
            f"OrientedPrism(a={self.a:.4g}, b={self.b:.4g}, c={self.c:.4g}, "
            f"rho={self.rho:.4g}, tubes={len(self.tube_indices)})"
        )


@dataclass
class GrainsDecomposition:
    """A collection of grains (``Prism`` 或 ``OrientedPrism``) 组成一次分解。

    ``prisms`` 字段是 ``list[Prism | OrientedPrism]``（可变容器，需 ``append``）。
    构造参数改成 **``Iterable``**（协变）：这样 ``list[Prism]`` 能**直接**传入，
    不必先转成联合会话——``list`` 是不变的，若构造参数也声明成
    ``list[Prism | OrientedPrism]``，``GrainsDecomposition([<Prism>])`` 会被
    类型检查器判为非法（实测踩到）。``dataclass`` 不会覆盖类里已定义的
    ``__init__``（``_set_new_attribute`` 见名即让路），故这里手写一个即可。
    """

    prisms: List[Prism | OrientedPrism] = field(default_factory=list)

    def __init__(
        self, prisms: Iterable[Prism | OrientedPrism] | None = None
    ) -> None:
        self.prisms = list(prisms) if prisms is not None else []

    def add(self, prism: Prism | OrientedPrism) -> None:
        self.prisms.append(prism)

    @property
    def is_axis_aligned(self) -> bool:
        """全部 grain 是否都是轴对齐 :class:`Prism`。

        ``False`` 表示含 :class:`OrientedPrism`——``moves`` 的世界轴合并路径
        不适用于它们（见 ``moves.py`` 的 ``_require_axis_aligned``）。
        """
        return all(isinstance(p, Prism) for p in self.prisms)

    @property
    def count(self) -> int:
        return len(self.prisms)

    @property
    def total_prism_volume(self) -> float:
        return sum(p.volume for p in self.prisms)

    @property
    def min_rho(self) -> float:
        return min((p.rho for p in self.prisms), default=0.0)

    @property
    def max_rho(self) -> float:
        return max((p.rho for p in self.prisms), default=0.0)

    @property
    def min_c(self) -> float:
        return min((p.c for p in self.prisms), default=0.0)

    @property
    def max_c(self) -> float:
        return max((p.c for p in self.prisms), default=0.0)

    @property
    def max_local_kt_error(self) -> float:
        return max((p.local_kt_error for p in self.prisms), default=0.0)

    @property
    def avg_tubes_per_grain(self) -> float:
        if self.count == 0:
            return 0.0
        return sum(len(p.tube_indices) for p in self.prisms) / self.count

    @property
    def total_tubes_accounted(self) -> int:
        return len({idx for p in self.prisms for idx in p.tube_indices})

    @property
    def avg_density(self) -> float:
        if self.count == 0:
            return 0.0
        return float(np.mean([p.density for p in self.prisms]))

    def invariants(self) -> dict:
        """返回一个包含可测量的 grain 不变量的 dict。"""
        return {
            "count": self.count,
            "total_volume": self.total_prism_volume,
            "min_rho": self.min_rho,
            "max_rho": self.max_rho,
            "min_c": self.min_c,
            "max_c": self.max_c,
            "max_local_kt_error": self.max_local_kt_error,
            "avg_tubes_per_grain": self.avg_tubes_per_grain,
            "total_tubes_accounted": self.total_tubes_accounted,
            "avg_density": self.avg_density,
        }

    def __repr__(self) -> str:
        return f"GrainsDecomposition(count={self.count}, total_volume={self.total_prism_volume:.6g})"


def _axis_aligned_bounding_box(tubes: Sequence[Tube]) -> Box:
    """返回包含所有管线段的最小轴对齐盒（含 δ 余量）。"""
    if not tubes:
        return Box.from_center_size([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for t in tubes:
        a, b = t.endpoints
        lo = np.minimum(lo, np.minimum(a, b) - t.delta)
        hi = np.maximum(hi, np.maximum(a, b) + t.delta)
    return Box(np.column_stack([lo, hi]))


@dataclass(frozen=True)
class OrientedBoundingBox:
    """沿**主方向**的定向包围盒（旋转包围盒）。

    ``axes`` 的行是正交归一基，``axes[0]`` 即主方向；点从世界坐标进入本盒坐标的
    变换是 ``x_local = axes @ (x_world - center)``。半长 ``half_extents`` 沿这三轴。

    **为什么需要它**（局限 #4 / 路线图"旋转包围盒 grains"）：轴对齐盒（`Box`）对
    斜向管族包得太松——一根沿对角线的管，其轴对齐包围盒体积可达**紧致盒的 3 倍**。
    定向盒沿管方向贴合，正是论文里 grain 的方向性几何。
    """

    center: np.ndarray          # (3,)
    axes: np.ndarray            # (3, 3)，行=正交归一基，axes[0]=主方向
    half_extents: np.ndarray    # (3,)，>= 0

    def __post_init__(self):
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float).reshape(3))
        object.__setattr__(self, "axes", np.asarray(self.axes, dtype=float).reshape(3, 3))
        he = np.asarray(self.half_extents, dtype=float).reshape(3)
        if np.any(he < 0):
            raise InputValidationError("OrientedBoundingBox half_extents must be >= 0.")
        object.__setattr__(self, "half_extents", he)

    @property
    def extents(self) -> np.ndarray:
        """全长（= 2 × 半长），沿三轴。"""
        return 2.0 * self.half_extents

    @property
    def volume(self) -> float:
        return float(np.prod(self.extents))

    @property
    def principal_direction(self) -> np.ndarray:
        return self.axes[0]

    def to_local(self, points: np.ndarray) -> np.ndarray:
        """世界坐标 → 本盒坐标（可以是 (3,) 或 (N, 3)）。"""
        p = np.asarray(points, dtype=float)
        return (p - self.center) @ self.axes.T

    def to_world(self, points: np.ndarray) -> np.ndarray:
        """本盒坐标 → 世界坐标。"""
        p = np.asarray(points, dtype=float)
        return p @ self.axes + self.center

    def contains(self, points: np.ndarray, margin: float = 0.0) -> bool:
        """所有点在本盒（含 ``margin`` 外扩）内则为真。"""
        p = np.asarray(points, dtype=float)
        local = self.to_local(p)
        return bool(np.all(np.abs(local) <= self.half_extents + margin + 1e-12))

    def corners(self) -> np.ndarray:
        """8 个角点（世界坐标）。"""
        signs = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ], dtype=float)
        return self.to_world(signs * self.half_extents)


def _orthonormal_frame(direction: np.ndarray) -> np.ndarray:
    """由主方向造正交归一基（``axes[0] = 单位主方向``）。"""
    e0 = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(e0))
    e0 = e0 / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
    # 取与 e0 最不平行的世界轴做叉积，避免退化。
    ref = np.eye(3)[int(np.argmin(np.abs(e0)))]
    e1 = np.cross(e0, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(e0, e1)
    return np.vstack([e0, e1, e2])


def oriented_bounding_box(
    tubes: Sequence[Tube], direction: np.ndarray | None = None
) -> OrientedBoundingBox:
    """沿（给定或主）方向的**定向包围盒**，含每根管的 δ 外扩。

    ``direction`` 为 None 时取 :func:`_principal_direction`（管方向的均值）。返回的
    盒**保证包含全部管的 δ-邻域**（``contains`` 对端点 ±δ 为真），且对斜向管族比
    轴对齐盒**体积更小**（这正是"更贴近论文几何"的量化判据，见
    ``tests/test_grains.py``）。

    注意（负结果 → 已修复）：本原语最初**未接入**网格构造，因为曾把 grain 网格
    换到该定向坐标系、再按"世界轴对齐盒"汇报体积，导致总体积被**放大 ~30 倍**
    （0.12 → 3.66）。**病因是汇报口径，不是坐标系**：斜向 cell 的世界 AABB 远大
    于 cell 本身。2026-10-09 已配好"按定向盒自身汇报体积"的 grain 形体
    :class:`OrientedPrism`，并接入 :func:`guth_grains_skeleton(frame="oriented")`
    与 :func:`_build_oriented_grains_for_tube_subset`——斜向管族的 grain 总体积
    回落到**小于**轴对齐路径（不再虚胖），而轴对齐管族下两条路径**逐位一致**。
    """
    if not tubes:
        return OrientedBoundingBox(
            center=np.zeros(3), axes=np.eye(3), half_extents=np.full(3, 0.5)
        )
    e0 = np.asarray(direction, dtype=float) if direction is not None else _principal_direction(tubes)
    axes = _orthonormal_frame(e0)
    pts = []
    margin = 0.0
    for t in tubes:
        a, b = t.endpoints
        pts.append(a)
        pts.append(b)
        margin = max(margin, float(t.delta))
    pts = np.asarray(pts, dtype=float)
    local = (pts - pts.mean(axis=0)) @ axes.T
    lo = local.min(axis=0)
    hi = local.max(axis=0)
    center_local = 0.5 * (lo + hi)
    world_center = pts.mean(axis=0) + center_local @ axes
    half = 0.5 * (hi - lo) + margin
    return OrientedBoundingBox(center=world_center, axes=axes, half_extents=half)


def _long_axis_grid(dims: np.ndarray, grid_n: int, long_axis: int | None = None) -> tuple[np.ndarray, int]:
    """返回网格计数 [nx, ny, nz] 以及长轴下标。

    选定的长轴维度会被划分得更少，使 grain 沿主导管方向被拉长；两个横向
    方向则划分得更多。
    """
    dims = np.asarray(dims)
    if long_axis is None:
        long_axis = int(np.argmax(dims))
    long_axis = int(long_axis) % 3
    grid = np.full(3, max(1, 2 * grid_n), dtype=int)
    grid[long_axis] = max(1, grid_n)
    return grid, long_axis


def _principal_direction(tubes: Sequence[Tube]) -> np.ndarray:
    """管集合的平均方向，已归一化。"""
    if not tubes:
        return np.array([0.0, 0.0, 1.0])
    dirs = np.array([t.direction.v for t in tubes], dtype=float)
    mean = dirs.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


def _world_axis_closest_to_direction(direction: np.ndarray) -> int:
    """返回世界坐标轴（0=x，1=y，2=z）中其单位向量与 ``direction`` 内积绝对值最大的那一个。
    """
    direction = np.asarray(direction)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return 2
    unit = direction / norm
    dots = np.abs(np.eye(3) @ unit)
    return int(np.argmax(dots))


def _angular_distance_matrix(directions: np.ndarray) -> np.ndarray:
    """单位方向两两之间在 S² 上的角距离。

    Parameters
    ----------
    directions : np.ndarray, shape (N, 3)
        单位方向向量。

    Returns
    -------
    dist : np.ndarray, shape (N, N)
        逐元素的角距离，取值于 [0, π]。
    """
    cos = np.clip(directions @ directions.T, -1.0, 1.0)
    # arccos 在远离 ±1 处数值稳定；对轻微越界做钳制。
    return np.arccos(cos)


def partition_into_rho_tubes(
    tubes: Sequence[Tube],
    rho: float,
) -> Dict[int, List[int]]:
    """按球冠距离把 δ-管下标分组为粗粒度的 ρ-管。

    这里的 ρ-管表示为 S² 上角半径为 ρ/2 的一个球冠。 每个球冠的中心取
    第一根尚未分配的 δ-管的方向；所有方向落在中心 ρ/2 以内的 δ-管都被
    归到该 ρ-管。 这保证了同一 ρ-管内的任意两根 δ-管的角间隔至多为 ρ。

    Parameters
    ----------
    tubes : sequence of Tube
    rho : float
        ρ-管的角直径阈值。

    Returns
    -------
    bins : dict
        ``bins[k]`` 是分配到第 k 个 ρ-管的 δ-管下标列表。
        每根输入的管都被恰好分配到一个 bin。
    """
    if not tubes or rho <= 0:
        return {}

    directions = np.array([t.direction.v for t in tubes], dtype=float)
    n = len(tubes)
    assigned = np.zeros(n, dtype=bool)
    bins: Dict[int, List[int]] = {}
    bin_idx = 0
    half_rho = rho / 2.0

    while not np.all(assigned):
        # 取第一根尚未分配的管的方向作为球冠中心。
        centre_idx = int(np.argmax(~assigned))
        centre = directions[centre_idx]

        # 每个方向到球冠中心的角距离。
        # 抑制 BLAS 的数值告警：输入都是合法的单位向量，
        # 结果会立即被钳制到 [-1, 1]。
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            cos = np.clip(directions @ centre, -1.0, 1.0)
        dist = np.arccos(cos)

        mask = (~assigned) & (dist <= half_rho)
        indices = np.where(mask)[0].tolist()
        if not indices:
            # 防御性：不应发生，因为 centre_idx 尚未分配，
            # 且它到自身的距离为 0 ≤ half_rho。
            indices = [centre_idx]

        assigned[indices] = True
        bins[bin_idx] = indices
        bin_idx += 1

    return bins


def adaptive_partition_into_rho_tubes(
    tubes: Sequence[Tube],
    rho: float | None = None,
    min_tubes_per_bin: int = 8,
    max_rho_cap: float = np.pi / 2.0,
    growth_factor: float = 1.2,
) -> Dict[int, List[int]]:
    """把管划分成 ρ-管，并不断扩大 ρ 直到每个 bin 都足够大。

    方向分散的构型（例如一个扇形）在按角度密度估计出 ρ 时，可能每个球冠
    里只落一两根管。 本函数从通常的估计出发，放大 ρ 直到每个 bin 至少
    含有 ``min_tubes_per_bin`` 根管，或达到 ``max_rho_cap`` 为止。
    返回的划分仍保证同一 bin 内任意两根管的角间隔至多为最终的 ρ。

    Parameters
    ----------
    tubes : sequence of Tube
    rho : float or None
        初始角直径。 若为 None，则按 ``2/sqrt(n)`` 估计。
    min_tubes_per_bin : int
        每个 ρ-bin 所需的最少 δ-管数目。
    max_rho_cap : float
        允许的最大 ρ（上限为 π/2，即半球）。
    growth_factor : float
        每一步调整时 ρ 的乘性增长因子。

    Returns
    -------
    bins : dict
        ``bins[k]`` 是第 k 个 ρ-管中的 δ-管下标列表。
    """
    if not tubes:
        return {}

    tube_list = list(tubes)
    n = len(tube_list)
    if rho is None:
        rho = max(0.05, min(1.0, 2.0 / np.sqrt(max(n, 1))))
    rho = float(rho)
    if rho <= 0:
        return {}

    while rho < max_rho_cap:
        bins = partition_into_rho_tubes(tube_list, rho)
        if not bins:
            break
        if min(len(v) for v in bins.values()) >= min_tubes_per_bin:
            return bins
        next_rho = min(rho * growth_factor, max_rho_cap)
        if next_rho <= rho:
            break
        rho = next_rho

    return partition_into_rho_tubes(tube_list, rho)


def rho_tube_cover(
    tubes: Sequence[Tube],
    rho: float,
) -> Dict[int, dict]:
    """返回一个 ρ-管覆盖，连同各球冠中心与角直径。

    Returns
    -------
    cover : dict
        ``cover[k]`` 含有键 ``indices``（δ-管下标列表）、
        ``centre``（球冠中心的单位方向）与 ``diameter``
        （球冠内的最大角间隔）。
    """
    if not tubes or rho <= 0:
        return {}

    directions = np.array([t.direction.v for t in tubes], dtype=float)
    bins = partition_into_rho_tubes(tubes, rho)
    cover: Dict[int, dict] = {}
    for k, indices in bins.items():
        subset = directions[indices]
        if len(subset) <= 1:
            diameter = 0.0
        else:
            diameter = float(_angular_distance_matrix(subset).max())
        # 中心是第一根被分配的管的方向。
        centre = directions[indices[0]]
        cover[k] = {
            "indices": indices,
            "centre": centre,
            "diameter": diameter,
        }
    return cover


def _segment_intersects_box(
    a: np.ndarray,
    b: np.ndarray,
    box_lo: np.ndarray,
    box_hi: np.ndarray,
) -> bool:
    """Liang-Barsky 平板检验，不构造 Box 对象。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    t_min, t_max = 0.0, 1.0
    for i in range(3):
        d = b[i] - a[i]
        if abs(d) < 1e-15:
            if a[i] < box_lo[i] or a[i] > box_hi[i]:
                return False
        else:
            t1 = (box_lo[i] - a[i]) / d
            t2 = (box_hi[i] - a[i]) / d
            t_enter, t_exit = min(t1, t2), max(t1, t2)
            t_min = max(t_min, t_enter)
            t_max = min(t_max, t_exit)
            if t_min > t_max:
                return False
    return True


def _segment_intersects_aabb(
    ax: float, ay: float, az: float,
    bx: float, by: float, bz: float,
    x0: float, x1: float,
    y0: float, y1: float,
    z0: float, z1: float,
) -> bool:
    """标量版 Liang-Barsky 平板检验；避免数组分配。"""
    t_min, t_max = 0.0, 1.0

    dx = bx - ax
    if abs(dx) < 1e-15:
        if ax < x0 or ax > x1:
            return False
    else:
        t1 = (x0 - ax) / dx
        t2 = (x1 - ax) / dx
        te, tx = (t1, t2) if t1 < t2 else (t2, t1)
        if te > t_min:
            t_min = te
        if tx < t_max:
            t_max = tx
        if t_min > t_max:
            return False

    dy = by - ay
    if abs(dy) < 1e-15:
        if ay < y0 or ay > y1:
            return False
    else:
        t1 = (y0 - ay) / dy
        t2 = (y1 - ay) / dy
        te, tx = (t1, t2) if t1 < t2 else (t2, t1)
        if te > t_min:
            t_min = te
        if tx < t_max:
            t_max = tx
        if t_min > t_max:
            return False

    dz = bz - az
    if abs(dz) < 1e-15:
        if az < z0 or az > z1:
            return False
    else:
        t1 = (z0 - az) / dz
        t2 = (z1 - az) / dz
        te, tx = (t1, t2) if t1 < t2 else (t2, t1)
        if te > t_min:
            t_min = te
        if tx < t_max:
            t_max = tx
        if t_min > t_max:
            return False

    return True


def _build_grains_for_tube_subset(
    tubes: Sequence[Tube],
    shading: Shading | None,
    grid_n: int,
    global_indices: Sequence[int] | None = None,
    bin_index: int | None = None,
    bin_tube_count: int = 0,
    total_tube_count: int = 0,
    bin_direction: np.ndarray | None = None,
) -> GrainsDecomposition:
    """Build grains for a subset of tubes.

    The grain elongation follows the principal direction of the subset.
    ``global_indices`` maps local tube indices back to the original sequence.

    This version pre-filters cells by each tube's axis-aligned bounding box so
    that the expensive segment-box intersection test is only run on cells the
    tube can possibly hit.

    ⚠️ **``local_kt_error`` 是 proxy，不是论文的 ``C_KT-CW^loc(G)``。** 这里取
    ``n|T|/|G|``（命中该 grain 的管数 × 单管体积 ÷ grain 体积）。论文的量是
    "对**所有**凸集 ``W`` 的 Katz–Tao 凸 Wolff 误差"，而本式只用 grain 自身
    这一个凸集，且不含 ``δ`` 尺度因子——两者定义不同。

    该 proxy 被 Move #3 的后条件检查（``check_move3_condition``）与
    ``regularize_shading`` 间接依赖，故**不要把由它算出的结论当作论文的定量
    结论**。真实相交统计可用 ``error_mode="actual"`` 路径（体素重算）获得。

    **为什么写进 docstring 而不只留在行内注释**：行内注释容易被重构带走，
    而 docstring 是 `help()` 与类型提示展示的内容，且被
    ``tests/test_honest_labeling.py`` 守门（删掉即红）。
    """
    if not tubes:
        return GrainsDecomposition()

    tube_list = list(tubes)
    tube_volume = tube_list[0].volume
    if global_indices is None:
        global_indices = list(range(len(tube_list)))
    global_indices_list = list(global_indices)

    bbox = _axis_aligned_bounding_box(tubes)
    lo = bbox.bounds[:, 0]
    hi = bbox.bounds[:, 1]
    dims = hi - lo

    # 从该子集的主方向选定 grain 的拉长轴。
    principal = _principal_direction(tubes)
    long_axis = _world_axis_closest_to_direction(principal)

    grid, _ = _long_axis_grid(dims, max(1, grid_n), long_axis=long_axis)
    nx, ny, nz = int(grid[0]), int(grid[1]), int(grid[2])
    step = dims / np.array([nx, ny, nz])

    # 每个轴上的格子边界。
    x_edges = np.linspace(lo[0], hi[0], nx + 1)
    y_edges = np.linspace(lo[1], hi[1], ny + 1)
    z_edges = np.linspace(lo[2], hi[2], nz + 1)

    # 把 (ix, iy, iz) 映射到全局管下标列表。
    cell_map: Dict[Tuple[int, int, int], List[int]] = {}

    # 预计算每根管的端点与 delta，以避免属性访问的开销。
    tube_data = []
    for local_idx, t in enumerate(tube_list):
        a, b = t.endpoints
        seg_lo = np.minimum(a, b) - t.delta
        seg_hi = np.maximum(a, b) + t.delta
        tube_data.append((local_idx, a, b, t.delta, seg_lo, seg_hi))

    for local_idx, a, b, delta, seg_lo, seg_hi in tube_data:
        # 与该管 bbox 重叠的格子的保守整数范围。
        ix_min = max(0, int(np.floor((seg_lo[0] - lo[0]) / step[0])))
        ix_max = min(nx - 1, int(np.ceil((seg_hi[0] - lo[0]) / step[0])))
        iy_min = max(0, int(np.floor((seg_lo[1] - lo[1]) / step[1])))
        iy_max = min(ny - 1, int(np.ceil((seg_hi[1] - lo[1]) / step[1])))
        iz_min = max(0, int(np.floor((seg_lo[2] - lo[2]) / step[2])))
        iz_max = min(nz - 1, int(np.ceil((seg_hi[2] - lo[2]) / step[2])))

        if ix_min > ix_max or iy_min > iy_max or iz_min > iz_max:
            continue

        ax, ay, az = a[0], a[1], a[2]
        bx, by, bz = b[0], b[1], b[2]
        for ix in range(ix_min, ix_max + 1):
            cell_xlo = x_edges[ix] - delta
            cell_xhi = x_edges[ix + 1] + delta
            for iy in range(iy_min, iy_max + 1):
                cell_ylo = y_edges[iy] - delta
                cell_yhi = y_edges[iy + 1] + delta
                for iz in range(iz_min, iz_max + 1):
                    cell_zlo = z_edges[iz] - delta
                    cell_zhi = z_edges[iz + 1] + delta
                    if _segment_intersects_aabb(
                        ax, ay, az, bx, by, bz,
                        cell_xlo, cell_xhi, cell_ylo, cell_yhi, cell_zlo, cell_zhi,
                    ):
                        cell_map.setdefault((ix, iy, iz), []).append(global_indices_list[local_idx])

    grains = GrainsDecomposition()
    # 每根棱柱只预计算一次局部下标查找表。
    global_to_local = {g: i for i, g in enumerate(global_indices_list)}

    # 由被填充的格子构造棱柱。
    for (ix, iy, iz), indices in cell_map.items():
        if not indices:
            continue
        p_lo = np.array([x_edges[ix], y_edges[iy], z_edges[iz]])
        p_hi = np.array([x_edges[ix + 1], y_edges[iy + 1], z_edges[iz + 1]])
        prism_box = Box(np.column_stack([p_lo, p_hi]))

        local_indices = [global_to_local[g] for g in indices]
        density = 1.0 if shading is None else float(np.mean(shading.densities[local_indices]))
        # cell_map 里已经含有 δ-邻域命中该格子的管，
        # 故 n_hitting_grain 等于 len(indices)。 这是一个快速近似，
        # 可以接受，因为 local_kt_error 本就只是代理量。
        n_hitting_grain = len(indices)
        local_kt_error = (
            float("inf")
            if prism_box.volume <= 0
            else n_hitting_grain * tube_volume / prism_box.volume
        )
        prism = Prism(
            bounds=prism_box.bounds,
            tube_indices=tuple(indices),
            density=density,
            local_kt_error=local_kt_error,
            bin_index=bin_index,
            bin_tube_count=bin_tube_count,
            total_tube_count=total_tube_count,
            bin_direction=bin_direction,
        )
        grains.add(prism)

    return grains


def _build_oriented_grains_for_tube_subset(
    tubes: Sequence[Tube],
    shading: Shading | None,
    grid_n: int,
    global_indices: Sequence[int] | None = None,
    bin_index: int | None = None,
    bin_tube_count: int = 0,
    total_tube_count: int = 0,
    bin_direction: np.ndarray | None = None,
) -> GrainsDecomposition:
    """在**管族自己的正交坐标系**里建 grain（:class:`OrientedPrism`）。

    与 :func:`_build_grains_for_tube_subset` 的区别只在"格子建在哪个坐标系、
    体积怎么汇报"：

    - 格子沿 :func:`oriented_bounding_box` 给出的正交基切分（e_c = 长轴），
      而不是沿世界轴；
    - 每个格子是 **:class:`OrientedPrism`**，其 :attr:`~OrientedPrism.volume`
      取 **定向盒自身体积** ``a·b·c``。

    这正是先前负结果的**修正版**：负结果的病因是"格子建在定向系、却按世界
    AABB 汇报体积"（斜向格子的世界 AABB 远大于格子本身 ⇒ 总体积放大 ~30 倍）。
    把汇报口径改成定向体积后，斜向 grain 的体积与世界朝向无关，不再被放大。

    相交判定仍在**局部坐标**里做（δ-邻域在正交变换下仍是 δ-邻域），故与轴对齐
    版本同样保守：格子在局部各轴外扩 δ 后做 Liang–Barsky。
    """
    if not tubes:
        return GrainsDecomposition()

    tube_list = list(tubes)
    tube_volume = tube_list[0].volume
    if global_indices is None:
        global_indices = list(range(len(tube_list)))
    global_indices_list = list(global_indices)

    obb = oriented_bounding_box(tube_list)
    center = obb.center
    axes = obb.axes
    dims = obb.extents

    # 局部坐标系里 e_0 是主方向（最长边）——让长轴格子少切、横向多切。
    grid, _ = _long_axis_grid(dims, max(1, grid_n), long_axis=0)
    nx, ny, nz = int(grid[0]), int(grid[1]), int(grid[2])
    step = dims / np.array([nx, ny, nz])

    # 局部坐标下盒子的下界由 center 定位：世界点 → (p - center) @ axes^T。
    # OBB 关于 center 对称，故局部区间为 [-dims/2, +dims/2]。
    half = 0.5 * dims
    lo = -half
    hi = half
    x_edges = np.linspace(lo[0], hi[0], nx + 1)
    y_edges = np.linspace(lo[1], hi[1], ny + 1)
    z_edges = np.linspace(lo[2], hi[2], nz + 1)

    cell_map: Dict[Tuple[int, int, int], List[int]] = {}

    tube_data = []
    for local_idx, t in enumerate(tube_list):
        a, b = t.endpoints
        # 变换到 OBB 局部坐标（正交变换，δ-邻域保持）。
        a_l = (np.asarray(a, dtype=float) - center) @ axes.T
        b_l = (np.asarray(b, dtype=float) - center) @ axes.T
        seg_lo = np.minimum(a_l, b_l) - t.delta
        seg_hi = np.maximum(a_l, b_l) + t.delta
        tube_data.append((local_idx, a_l, b_l, t.delta, seg_lo, seg_hi))

    for local_idx, a, b, delta, seg_lo, seg_hi in tube_data:
        ix_min = max(0, int(np.floor((seg_lo[0] - lo[0]) / step[0])))
        ix_max = min(nx - 1, int(np.ceil((seg_hi[0] - lo[0]) / step[0])))
        iy_min = max(0, int(np.floor((seg_lo[1] - lo[1]) / step[1])))
        iy_max = min(ny - 1, int(np.ceil((seg_hi[1] - lo[1]) / step[1])))
        iz_min = max(0, int(np.floor((seg_lo[2] - lo[2]) / step[2])))
        iz_max = min(nz - 1, int(np.ceil((seg_hi[2] - lo[2]) / step[2])))
        if ix_min > ix_max or iy_min > iy_max or iz_min > iz_max:
            continue

        ax, ay, az = a[0], a[1], a[2]
        bx, by, bz = b[0], b[1], b[2]
        for ix in range(ix_min, ix_max + 1):
            cx_lo, cx_hi = x_edges[ix] - delta, x_edges[ix + 1] + delta
            for iy in range(iy_min, iy_max + 1):
                cy_lo, cy_hi = y_edges[iy] - delta, y_edges[iy + 1] + delta
                for iz in range(iz_min, iz_max + 1):
                    cz_lo, cz_hi = z_edges[iz] - delta, z_edges[iz + 1] + delta
                    if _segment_intersects_aabb(
                        ax, ay, az, bx, by, bz,
                        cx_lo, cx_hi, cy_lo, cy_hi, cz_lo, cz_hi,
                    ):
                        cell_map.setdefault((ix, iy, iz), []).append(global_indices_list[local_idx])

    grains = GrainsDecomposition()
    global_to_local = {g: i for i, g in enumerate(global_indices_list)}

    for (ix, iy, iz), indices in cell_map.items():
        if not indices:
            continue
        cell_lo = np.array([x_edges[ix], y_edges[iy], z_edges[iz]])
        cell_hi = np.array([x_edges[ix + 1], y_edges[iy + 1], z_edges[iz + 1]])
        cell_center_local = 0.5 * (cell_lo + cell_hi)
        cell_extents = cell_hi - cell_lo
        world_center = center + cell_center_local @ axes

        local_indices = [global_to_local[g] for g in indices]
        density = 1.0 if shading is None else float(np.mean(shading.densities[local_indices]))
        n_hitting_grain = len(indices)
        cell_volume = float(np.prod(cell_extents))
        local_kt_error = (
            float("inf")
            if cell_volume <= 0
            else n_hitting_grain * tube_volume / cell_volume
        )
        grains.add(OrientedPrism(
            center=world_center,
            axes=axes,
            extents=cell_extents,
            tube_indices=tuple(indices),
            density=density,
            local_kt_error=local_kt_error,
            bin_index=bin_index,
            bin_tube_count=bin_tube_count,
            total_tube_count=total_tube_count,
            bin_direction=bin_direction,
        ))

    return grains


def guth_grains_skeleton(
    tubes: Sequence[Tube],
    shading: Shading | None,
    grid_n: int = 4,
    rho: float | None = None,
    adaptive_rho: bool = True,
    min_tubes_per_bin: int = 8,
    frame: str = "axis",
) -> GrainsDecomposition:
    """Geometry-aware skeleton of Guth's grains decomposition.

    First partitions tubes into coarse ρ-tubes (directional clusters), then
    builds axis-aligned grains for each cluster whose elongation follows the
    cluster's principal direction.  A δ-tube is associated with a grain when its
    central segment comes within δ of the grain.  Each grain stores a proxy for
    its local Katz–Tao convex Wolff error.

    Compared with the earlier skeleton, this version aligns the grain elongation
    with the local tube direction rather than a global axis, making the grains
    much closer to the geometrically natural grains used in Wang–Zahl.

    When ``adaptive_rho`` is True (the default), the initial ρ estimate is grown
    until every ρ-bin contains at least ``min_tubes_per_bin`` tubes.  This makes
    Move #1 achievable on directionally dispersed inputs such as fans.

    ``frame`` 选择 grain 的坐标系与体积汇报口径：

    - ``"axis"``（默认）：世界轴对齐 :class:`Prism`，体积 = 世界盒体积（历史行为，
      下游 ``moves`` 可直接跑）。
    - ``"oriented"``：沿管族主方向的 :class:`OrientedPrism`，**体积 = 定向盒自身体积**
      （斜向 grain 不再因世界 AABB 而虚胖）。此时 ``moves`` 的世界轴合并路径不适用，
      调用它会显式报 ``NotImplementedYetError``（见 ``moves._require_axis_aligned``）。
    """
    if frame not in ("axis", "oriented"):
        raise InputValidationError(
            f"guth_grains_skeleton frame must be 'axis' or 'oriented', got {frame!r}."
        )
    if not tubes:
        return GrainsDecomposition()

    tube_list = list(tubes)
    n_total = len(tube_list)

    if adaptive_rho:
        bins = adaptive_partition_into_rho_tubes(
            tube_list,
            rho=rho,
            min_tubes_per_bin=min_tubes_per_bin,
        )
    else:
        if rho is None:
            rho = max(0.05, min(1.0, 2.0 / np.sqrt(max(n_total, 1))))
        bins = partition_into_rho_tubes(tube_list, rho)

    grains = GrainsDecomposition()
    for bin_idx, local_indices in bins.items():
        subset = [tube_list[i] for i in local_indices]
        subset_shading = None if shading is None else Shading(shading.densities[local_indices])
        bin_direction = _principal_direction(subset)
        builder = (
            _build_oriented_grains_for_tube_subset
            if frame == "oriented"
            else _build_grains_for_tube_subset
        )
        local_grains = builder(
            subset,
            subset_shading,
            max(1, grid_n),
            global_indices=local_indices,
            bin_index=bin_idx,
            bin_tube_count=len(local_indices),
            total_tube_count=n_total,
            bin_direction=bin_direction,
        )
        grains.prisms.extend(local_grains.prisms)

    return grains


def broadness_at_points(
    tubes: Sequence[Tube],
    points: np.ndarray,
    K: float = 10.0,
    beta: float = 1.0,
    radii: np.ndarray | None = None,
) -> dict:
    """在采样点上计算简化的 broadness 得分。

    按照 Wang–Zahl 的定义 7.6/7.7，一个方向集合 V ⊂ S² 是 (K, β)-broad 的，
    当且仅当对每个半径 r 的球冠 τ ⊂ S²，

        # {v ∈ V : v ∈ τ}  ≤  K r^β (#V)。

    本函数在每个观测到的方向上取球冠中心进行采样，并对一族给定的等比半径
    序列检验该不等式。 它返回逐点的布尔 broadness 标记以及一个全局得分。

    Parameters
    ----------
    tubes : sequence of Tube
    points : np.ndarray, shape (P, 3)
        R³ 中的采样点。
    K : float
        Broadness 常数。
    beta : float
        Broadness 指数。
    radii : np.ndarray or None
        用于检验球冠条件的半径。 若为 None，则用从 0.05 到 π/2 的默认
        等比序列。

    Returns
    -------
    result : dict
        - ``per_point_broad`` ：长度为 P 的布尔数组。
        - ``per_point_score`` ：长度为 P 的浮点数组；在所有球冠与半径上观测
          到的最大比值 ``count / (K r^β N)``（≤ 1 表示 broad）。
        - ``global_broad`` ：布尔值，当且仅当所有采样点都 broad 时为 True。
        - ``global_score`` ：所有点中的最大得分。
    """
    points = np.atleast_2d(points).astype(float)
    if radii is None:
        radii = np.geomspace(0.05, np.pi / 2.0, 10)
    else:
        radii = np.asarray(radii, dtype=float)

    tube_list = list(tubes)
    per_point_score = np.zeros(len(points), dtype=float)
    per_point_broad = np.ones(len(points), dtype=bool)

    for p_idx, pt in enumerate(points):
        # 其 δ-邻域包含该点的那些管的方向。
        dirs = []
        for t in tube_list:
            if t.contains(pt.reshape(1, -1))[0]:
                dirs.append(t.direction.v)
        if len(dirs) < 2:
            continue
        dirs = np.array(dirs, dtype=float)
        N = len(dirs)

        max_ratio = 0.0
        # 在每个观测到的方向上取球冠中心进行检验。
        cos = np.clip(dirs @ dirs.T, -1.0, 1.0)
        dist = np.arccos(cos)
        for r in radii:
            counts = np.count_nonzero(dist <= r, axis=1)
            bound = K * (r ** beta) * N
            if bound > 0:
                ratios = counts / bound
                max_ratio = max(max_ratio, float(ratios.max()))

        per_point_score[p_idx] = max_ratio
        per_point_broad[p_idx] = max_ratio <= 1.0 + 1e-12

    return {
        "per_point_broad": per_point_broad,
        "per_point_score": per_point_score,
        "global_broad": bool(np.all(per_point_broad)),
        "global_score": float(per_point_score.max()),
    }


def _directions_through_point(
    tubes: Sequence[Tube], point: np.ndarray
) -> tuple[List[int], np.ndarray]:
    """返回包含 ``point`` 的那些管的局部下标与单位方向。"""
    pt = np.atleast_2d(point).astype(float)
    indices: List[int] = []
    dirs: List[np.ndarray] = []
    for idx, t in enumerate(tubes):
        if t.contains(pt)[0]:
            indices.append(idx)
            dirs.append(t.direction.v)
    if not dirs:
        return [], np.empty((0, 3), dtype=float)
    return indices, np.asarray(dirs, dtype=float)


def _worst_cap_violation(
    dirs: np.ndarray,
    K: float,
    beta: float,
    radii: np.ndarray,
    min_count: int = 2,
) -> tuple[float, int, float, np.ndarray] | None:
    """找出违反 (K, β)-broadness 程度最大的那个球冠。

    返回 ``(ratio, centre_index, radius, member_indices)``，其中 ``ratio`` 为
    ``count / (K r^β N)``。 比值 ≤ 1 表示没有球冠违反该不等式，故该方向
    集合是 broad 的。 当没有任何一个至少含 ``min_count`` 个方向的球冠违反
    该界时，返回 ``None``。

    ``min_count`` 很重要：在被仅仅两根管穿过的采样点上，极小半径的球冠
    总会"违反"该界（``1 / (K r^2 β) > 1``），却并不指示真实的方向簇。
    要求一个最小占据数，可以让该归约聚焦于真正狭窄的簇。
    """
    n = len(dirs)
    if n < 2:
        return None
    cos = np.clip(dirs @ dirs.T, -1.0, 1.0)
    dist = np.arccos(cos)
    best: tuple[float, int, float, np.ndarray] | None = None
    for r in radii:
        counts = np.count_nonzero(dist <= r, axis=1)
        bound = K * (float(r) ** beta) * n
        if bound <= 0:
            continue
        ratios = counts / bound
        for centre_idx in np.argsort(ratios)[::-1]:
            # 用独立变量而不是复用循环名：``np.argsort`` 给出的是 numpy 整数，
            # 需要转成 Python int 才能安全用作下标（且原循环变量的语义是
            # "下标序列"，与转换后的"某个下标"不该同名）。
            centre = int(centre_idx)
            if counts[centre] < min_count:
                continue
            if ratios[centre] <= 1.0:
                break
            if best is None or ratios[centre] > best[0]:
                members = np.nonzero(dist[centre] <= r)[0]
                best = (float(ratios[centre]), centre, float(r), members)
            break
    return best


def _broadness_scale_floor(tubes: Sequence[Tube]) -> float:
    """一个方向簇有意义的最小角尺度。

    一根 δ-管在球面上张开约为 δ 量级的角，故远小于 δ 的球冠不是真正的
    方向簇。 以 ``delta`` 作为半径下限，可以避免球冠检验在任意小的半径上
    把单根管误判为违反。
    """
    deltas = [float(t.delta) for t in tubes if t.delta > 0]
    if not deltas:
        return 0.05
    return min(0.05, max(deltas))


@dataclass
class NarrowBatch:
    """一批被 broadness 前置过滤剥离的窄簇，**连同它所在的球面帽**。

    **为什么必须连 cap 一起记**（2026-09-30，L1 数学闭合）：只记"哪些管被
    剥掉了"是不完整的——被剥掉的管之所以能剥，是因为**它们的方向挤在一个
    半径 ``cap_radius`` 的小球面帽里**。而论文里这批管的处置正是
    "**在这个帽上重新走一遍同一条归纳**"（induction on a smaller cap）。

    没有 ``cap_radius`` 就无从知道"帽子多大"，也就无从复现那一步——
    这正是此前"剥了没下文"的结构性原因：数据本身不足以支撑承接。

    Attributes
    ----------
    indices
        全局管索引（原始 ``tube_list`` 中的下标）。
    cap_radius
        方向球面上该帽的半径（弧度）。``0`` 表示"未记录"——旧调用点或
        检测失败时的兼容取值，此时承接方应保守处理（见
        ``ProofOrchestrator._induct_on_narrow_batch``）。
    cap_centre
        帽中心在**该批 dirs 数组内**的下标（不是全局管索引）。
        ``-1`` 表示未记录。
    """

    indices: Tuple[int, ...]
    cap_radius: float = 0.0
    cap_centre: int = -1

    def __len__(self) -> int:
        return len(self.indices)


@dataclass
class BroadnessReduction:
    """在一族管上运行 broadness 预过滤所得的结果。

    该归约会剥离*狭窄*的方向簇——即过多管方向聚集其上的球冠——直到剩余的
    管在采样点处是 (K, β)-broad 的，或者无法再取得进展为止。 每个被剥离
    的簇记录在 ``narrow_batches`` 中，类型为 :class:`NarrowBatch`
    （**不是**裸索引元组——它连帽的几何 ``cap_radius``/``cap_centre`` 一起记；
    只要索引请用 :attr:`narrow_index_batches` 或 ``b.indices``）;
    在完整的 Wang–Zahl 实现里，这些批次正是由在一个更小球冠上做归纳来
    处理的那些。

    这是一个**计算代理**：它使用采样点与有限的半径网格，而非精确的球冠
    条件，故 ``is_broad`` 应读作"在采样点处是 broad 的"。
    """

    is_broad: bool
    K: float
    beta: float
    initial_score: float
    final_score: float
    n_tubes_in: int
    broad_indices: Tuple[int, ...] = field(default_factory=tuple)
    #: 被剥离的窄簇批次。**自 2026-09-30 起元素是 :class:`NarrowBatch`**
    #: （原来只是 ``tuple[int, ...]``）——因为承接需要知道"小帽多大"，
    #: 只记索引不足以复现论文的"小帽上的归纳"。
    #:
    #: 兼容性：``NarrowBatch`` 实现了 ``__len__``，故既有只取 ``len(b)``
    #: 的调用方不受影响；真正遍历索引的地方需改用 ``b.indices``（见
    #: ``_to_index_batches``）。
    narrow_batches: List["NarrowBatch"] = field(default_factory=list)
    stalled: bool = False

    @property
    def narrow_index_batches(self) -> List[Tuple[int, ...]]:
        """只取索引的视图（供"我只要知道剥了哪些管"的调用方）。"""
        return [b.indices for b in self.narrow_batches]


    @property
    def n_removed(self) -> int:
        """被剥离进 narrow batch 的管数。"""
        return self.n_tubes_in - len(self.broad_indices)

    @property
    def is_vacuous(self) -> bool:
        """当存活的集合太小、以致 broadness 不再有意义时为 True。"""
        return len(self.broad_indices) <= 1

    def summary(self) -> str:
        tail = ", stalled (no proper reduction exists)" if self.stalled else ""
        return (
            f"broadness pre-filter: K={self.K:g}, beta={self.beta:g}, "
            f"score {self.initial_score:.4g} -> {self.final_score:.4g}, "
            f"removed {self.n_removed}/{self.n_tubes_in} tubes in "
            f"{len(self.narrow_batches)} narrow batch(es), "
            f"is_broad={self.is_broad}{tail}"
        )


def broadness_reduction(
    tubes: Sequence[Tube],
    K: float = 10.0,
    beta: float = 1.0,
    n_points: int = 32,
    seed: int | None = 0,
    radii: np.ndarray | None = None,
    min_tubes_per_batch: int = 2,
    max_batches: int = 8,
) -> BroadnessReduction:
    """剥离狭窄的方向簇，使剩余部分成为 broad 的。

    依照 Wang–Zahl 的 broadness 归约步骤，一个未通过 (K, β)-broadness 检验
    的管集合含有一个方向聚集在一个小球冠内的子集合。 这些管被移除进一个
    *narrow batch*，然后对其剩余部分重复该过程；存活的管构成 broad 的
    子集合，可交由 Move 流水线消费。

    本函数在当前管子集的并集内采样点，并在那里运行 ``broadness_at_points``。
    在最坏的点上，它定位出违反该界程度最大的球冠，并移除穿过该点、且方向
    落在该球冠内的那些管。 如此循环，把 broadness 得分推向 1。

    Parameters
    ----------
    tubes : sequence of Tube
    K, beta : float
        Broadness 常数与指数。
    n_points : int
        每次迭代在管并集内抽取的采样点数。
    seed : int or None
        基础随机种子；第 ``i`` 次迭代用 ``seed + i``，从而各次迭代重新采样。
    radii : np.ndarray or None
        用于检验球冠条件的半径。 默认是从 ``delta``（管厚度，即几何上最小的
        有意义的角尺度）到 π/2 的等比序列。 低于管尺度的球冠被排除，因为
        否则单根管会在任意小的半径上"违反"该界。
    min_tubes_per_batch : int
        当违反的球冠将移除的管数少于该值时停止（太小、不构成有意义的簇）。
    max_batches : int
        最多剥离的 narrow batch 数目。

    Returns
    -------
    BroadnessReduction
        其中 ``broad_indices`` 给出存活的（broad 的）管下标。注意如果只有
        ≤ 1 根管存活，``is_broad`` 是空真（vacuously true），而 ``is_vacuous``
        会标记它——调用方应检查存活计数。 ``stalled`` 为 True 表示检测到
        违反、但无法归约成适当的 narrow batch（此时存活集合*并非* broad，
        调用方不应把它当作 broad）。

    Notes
    -----
    这是刻意对论文构造做的代理：它在数值上可判定、单调（在最坏球冠处移除
    之后得分不会升高），且足够廉价，可以在每次 grains 分解之前运行。
    """
    tube_list = list(tubes)
    if radii is None:
        # 一根管的角尺度约为 delta（其长宽比为 1:delta），故小于 delta 的
        # 半径不是有意义的方向簇。
        scale_floor = _broadness_scale_floor(tube_list)
        radii_arr = np.geomspace(max(scale_floor, 1e-3), np.pi / 2.0, 10)
    else:
        radii_arr = np.asarray(radii, dtype=float)
    if not tube_list:
        return BroadnessReduction(
            is_broad=True,
            K=K,
            beta=beta,
            initial_score=0.0,
            final_score=0.0,
            n_tubes_in=0,
        )

    working: set[int] = set(range(len(tube_list)))
    narrow_batches: List["NarrowBatch"] = []
    initial_score = 0.0
    current_score = 0.0
    is_broad = True
    stalled = False

    for iteration in range(max_batches):
        ordered = sorted(working)
        subset = [tube_list[i] for i in ordered]
        if len(subset) < 2:
            # 少于两根管时，采样点处的 broadness 是空真的。
            is_broad = True
            break

        points = sample_broadness_points(
            subset,
            n_points=n_points,
            seed=None if seed is None else seed + iteration,
        )
        if len(points) == 0:
            is_broad = True
            break

        result = broadness_at_points(subset, points, K=K, beta=beta, radii=radii_arr)
        current_score = float(result["global_score"])
        if iteration == 0:
            initial_score = current_score

        if result["global_broad"]:
            is_broad = True
            break

        is_broad = False
        worst = int(np.argmax(result["per_point_score"]))
        _, dirs = _directions_through_point(subset, points[worst])
        violation = _worst_cap_violation(dirs, K, beta, radii_arr)
        if violation is None:
            # 点级检验报告了违反，但没有一个携带足够方向的球冠能解释它：
            # 没有适当的对象可剥离。
            stalled = True
            break

        ratio, _centre, _radius, members = violation
        if ratio <= 1.0:
            stalled = True
            break

        batch_indices = tuple(sorted(ordered[int(j)] for j in members))
        if len(batch_indices) < min_tubes_per_batch or len(batch_indices) >= len(working):
            # 太小、无关紧要，或者移除它会把集合清空。
            stalled = True
            break

        # **把帽的几何一起记下来**（2026-09-30，L1）：光记"剥了哪些管"不足以
        # 支撑论文的"小帽上的归纳"——那一步需要知道**帽子多大**
        # （``_radius`` 是方向球面上的弧度半径）。此前只记索引，于是承接
        # 无从下手，这正是"剥了没下文"的结构性原因。
        #
        # ``_centre`` 是**该批 dirs 数组内**的下标。转成"全局管索引"以免
        # 与 ``indices`` 混淆（后者是全局的）——两者差一个 ``members`` 映射，
        # 留一个含糊的字段必然被误用。
        try:
            centre_global = int(ordered[int(members[int(_centre)])])
        except (IndexError, ValueError, TypeError):
            centre_global = -1
        narrow_batches.append(
            NarrowBatch(
                indices=batch_indices,
                cap_radius=float(_radius),
                cap_centre=centre_global,
            )
        )
        working -= set(batch_indices)
        if len(working) < 2:
            is_broad = True
            break
    else:
        # 循环在仍然失败的情况下耗尽了 max_batches：如实报告。
        if not is_broad:
            stalled = True

    final_score = current_score
    if narrow_batches:
        final_subset = [tube_list[i] for i in sorted(working)]
        if len(final_subset) >= 2:
            final_points = sample_broadness_points(
                final_subset,
                n_points=n_points,
                seed=None if seed is None else seed + 10_000,
            )
            if len(final_points) > 0:
                final_result = broadness_at_points(
                    final_subset, final_points, K=K, beta=beta, radii=radii_arr
                )
                final_score = float(final_result["global_score"])
                if not final_result["global_broad"]:
                    stalled = True
        if len(final_subset) < 2:
            final_score = 0.0
            is_broad = True

    return BroadnessReduction(
        is_broad=is_broad,
        K=K,
        beta=beta,
        initial_score=initial_score,
        final_score=final_score,
        n_tubes_in=len(tube_list),
        broad_indices=tuple(sorted(working)),
        narrow_batches=narrow_batches,
        stalled=stalled,
    )


def two_scale_grains_skeleton(
    tubes: Sequence[Tube],
    shading: Shading | None,
    rho: float,
    frame: str = "axis",
) -> Dict[int, GrainsDecomposition]:
    """Skeleton of a two-scale grains decomposition.

    Returns a mapping from coarse ρ-tube bin index to a GrainsDecomposition of
    the δ-tubes inside that bin.  Grain ``tube_indices`` are global indices
    into the original ``tubes`` sequence.

    ``frame`` 透传给 :func:`guth_grains_skeleton`；建成后 grain 类型被**原样
    保留**（用 :func:`dataclasses.replace` 只改索引与 bin 元数据），不会把
    ``OrientedPrism`` 静默降级成轴对齐 ``Prism``。
    """
    from dataclasses import replace

    bins = partition_into_rho_tubes(tubes, rho)
    result: Dict[int, GrainsDecomposition] = {}
    for bin_idx, indices in bins.items():
        bin_tubes = [tubes[i] for i in indices]
        bin_shading = None if shading is None else Shading(shading.densities[indices])
        local_grains = guth_grains_skeleton(bin_tubes, bin_shading, grid_n=2, frame=frame)
        # 把局部下标映射回全局管下标，并保留 grain 类型。
        global_grains: List[Prism | OrientedPrism] = []
        bin_direction = _principal_direction(bin_tubes)
        for p in local_grains.prisms:
            global_indices = tuple(indices[i] for i in p.tube_indices)
            global_grains.append(
                replace(
                    p,
                    tube_indices=global_indices,
                    bin_index=bin_idx,
                    bin_tube_count=len(indices),
                    total_tube_count=len(tubes),
                    bin_direction=bin_direction,
                )
            )
        result[bin_idx] = GrainsDecomposition(global_grains)
    return result


def flatten_two_scale_grains(
    two_scale: Dict[int, GrainsDecomposition]
) -> GrainsDecomposition:
    """把两尺度 grains 分解展平成一个单一的分解。

    当下游的 Move 流水线（Move #1/#2/#3）需要统一处理所有 ρ-bin，或需要
    计算全局 grain 统计量时，这很有用。
    """
    prisms: List[Prism | OrientedPrism] = []
    for grains in two_scale.values():
        prisms.extend(grains.prisms)
    return GrainsDecomposition(prisms)


def two_scale_grains_invariants(
    two_scale: Dict[int, GrainsDecomposition]
) -> dict:
    """汇总两尺度分解中所有 ρ-bin 的不变量。

    返回与 ``GrainsDecomposition.invariants()`` 相同的键，外加
    ``n_bins``（粗粒度 ρ-管的数目）。
    """
    flat = flatten_two_scale_grains(two_scale)
    inv = flat.invariants()
    inv["n_bins"] = len(two_scale)
    return inv


def grain_quality_metrics(
    grains: GrainsDecomposition,
    tubes: Sequence[Tube] | None = None,
) -> dict:
    """计算某个 grains 分解的数值质量指标。

    Parameters
    ----------
    grains : GrainsDecomposition
    tubes : sequence of Tube, optional
        提供时，用总管体积来估计体积效率。 若省略，则体积效率报告为 0。

    Returns
    -------
    metrics : dict
        - ``volume_efficiency`` ：``sum |T| / sum |G|``（衡量 grain 围绕其管
          的紧密程度的一个代理；对不相交的 grain 而言 ≤ 1 为理想）。
        - ``overlap_ratio`` ：``sum |G| / |union_bbox|``（≥ 1；越大表示
          grain 之间重叠越多）。
        - ``mean_rho`` / ``std_rho`` ：``b / c`` 的均值与标准差。
        - ``mean_c`` / ``std_c`` ：长边 ``c`` 的均值与标准差。
    """
    if grains.count == 0:
        return {
            "volume_efficiency": 0.0,
            "overlap_ratio": 0.0,
            "mean_rho": 0.0,
            "std_rho": 0.0,
            "mean_c": 0.0,
            "std_c": 0.0,
        }

    total_prism_volume = grains.total_prism_volume
    tube_volume = 0.0
    if tubes is not None:
        tube_volume = sum(t.volume for t in tubes)

    lo = np.min(np.stack([p.bounds[:, 0] for p in grains.prisms]), axis=0)
    hi = np.max(np.stack([p.bounds[:, 1] for p in grains.prisms]), axis=0)
    union_bbox_volume = float(np.prod(hi - lo))

    volume_efficiency = (
        tube_volume / total_prism_volume if total_prism_volume > 1e-24 else 0.0
    )
    overlap_ratio = (
        total_prism_volume / union_bbox_volume if union_bbox_volume > 1e-24 else 0.0
    )

    rhos = np.array([p.rho for p in grains.prisms])
    cs = np.array([p.c for p in grains.prisms])

    return {
        "volume_efficiency": float(volume_efficiency),
        "overlap_ratio": float(overlap_ratio),
        "mean_rho": float(np.mean(rhos)),
        "std_rho": float(np.std(rhos)),
        "mean_c": float(np.mean(cs)),
        "std_c": float(np.std(cs)),
    }


def sample_broadness_points(
    tubes: Sequence[Tube],
    n_points: int = 100,
    seed: int | None = None,
) -> np.ndarray:
    """在管并集内采样点，用于 broadness 检验。

    点从管的轴对齐包围盒中均匀抽取，落在每根管之外时被拒绝。 这样可以避免
    把 ``broadness_at_points`` 偏向空的区域。 若已找到足够多的点，或在有限
    次尝试之后，该函数会提前停止。

    Parameters
    ----------
    tubes : sequence of Tube
    n_points : int
        期望被接受的采样点数目。
    seed : int or None
        用于可复现的随机种子。

    Returns
    -------
    points : np.ndarray, shape (M, 3)
        被接受的采样点，其中 ``M <= n_points``。
    """
    if not tubes or n_points <= 0:
        return np.empty((0, 3))

    rng = np.random.default_rng(seed)
    bbox = _axis_aligned_bounding_box(tubes)
    lo = bbox.bounds[:, 0]
    hi = bbox.bounds[:, 1]

    accepted: List[np.ndarray] = []
    max_attempts = 10 * n_points
    attempts = 0
    while len(accepted) < n_points and attempts < max_attempts:
        batch_size = min(n_points - len(accepted), 1000)
        pts = rng.uniform(lo, hi, size=(batch_size, 3))
        inside = np.zeros(batch_size, dtype=bool)
        for t in tubes:
            inside |= t.contains(pts)
        accepted.extend(pts[inside])
        attempts += batch_size

    if not accepted:
        return np.empty((0, 3))
    points = np.array(accepted[:n_points])
    return points


def is_valid_grains_decomposition(
    grains: GrainsDecomposition,
    tubes: Sequence[Tube],
) -> bool:
    """检查某个 grains 分解的基本有效性。

    一个有效的分解必须具有正的 grain 体积，且其管下标指向真实存在的管。
    """
    if grains.count == 0:
        return True
    n = len(tubes)
    for p in grains.prisms:
        if p.volume <= 1e-24:
            return False
        if any((idx < 0) or (idx >= n) for idx in p.tube_indices):
            return False
    return True
