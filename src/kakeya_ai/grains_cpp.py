"""C++ 加速的近似 grains 分解（grains.py 的独立副本）。

本模块是 ``grains_fast.py`` 的进一步对缓存友好、精度更低的变体。 它保留
相同的几何近似（廉价的 bbox 重叠检验、对较小 ρ-bin 粗化的网格），但把最
热的循环——把管的 bbox 分配到网格格子——挪进了 ``cpp/kernel/tube_kernel.cpp``。
API 与 ``grains.py`` 保持一致，因此在对原始速度的要求高于几何精确性的
场合可以直接替换。

除非你确实想要不同的近似，否则不要在同一个计算里同时 import
``grains``、``grains_fast`` 与 ``grains_cpp``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from . import cpp_loader
from .primitives import Shading, Tube
from .wolff_axioms import Box

#: 编译扩展（``cpp/kernel/tube_kernel.cpp`` 的 pybind11 产物）。
#:
#: **2026-10-03 修正**：此前这里只有一句 ``from kakeya_ai import _tube_kernel``，
#: 失败即 ``_tk = None``，然后 :func:`_assign_tubes_to_grid_cells` **直接抛
#: ``ImportError``** —— 但**上面的注释承诺了**"fall back to the legacy ``cpp/``
#: directory layout"。⇒ **注释与实现不符，且违反本项目"任何 C++ 加载失败都静默
#: 切回纯 Python"的约定**（内部设计文档 §ABI/编译风险）。
#:
#: **2026-10-07 收敛**：三级回退的实现已抽进 :mod:`kakeya_ai.cpp_loader`——全库
#: 唯一的加载真源（此前 ``fast_kernel`` / ``nano_kernel`` / 本模块**各抄了一份**，
#: 正是局限 #22 记下的"没有单一真源"）。本模块只保留 ``_tk`` 这个状态名与
#: :func:`_load_tube_kernel` 这个入口名，语义交由 ``cpp_loader`` 保证。
_tk: Any = None


def _load_tube_kernel() -> Any:
    """加载 C++ 扩展；**三级回退，实现见 :mod:`kakeya_ai.cpp_loader`**。

    保留本函数（而非直接调用 ``cpp_loader``）是为了两个既有名字不被破坏：
    ``grains_cpp._tk``（状态）与 ``grains_cpp._load_tube_kernel``（入口）——
    :mod:`tests.geometry.test_grains_cpp_fallback` 直接用它们。语义（包内 → legacy
    ``cpp/`` → 顶层导入；全失败返回 ``None``）由 ``cpp_loader`` 单点保证。
    """
    global _tk

    _tk = cpp_loader.load_cpp_extension("_tube_kernel")
    return _tk


_load_tube_kernel()


def _assign_tubes_to_grid_cells(
    tube_bboxes: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """把管包围盒分配到网格单元；**扩展缺失时返回 None**（由调用方回退）。

    **为什么返回 ``None`` 而不是抛异常**（2026-10-03 改）：本项目的约定是
    "**任何 C++ 加载失败都静默切回纯 Python**"（内部设计文档 §ABI/编译风险，
    ``accel.py`` / ``fast_kernel`` / ``nano_kernel`` 都照此实现）。此前这里
    ``raise ImportError`` 是**唯一的例外**——它把"可选加速缺失"升级成了"功能
    不可用"，比不加速更糟。

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        ``(cell_ids, local_tube_indices)``；扩展不可用时为 ``None``。
    """
    if _tk is None:
        return None
    return _tk.assign_tubes_to_grid_cells(tube_bboxes, lo, hi, nx, ny, nz)


def _assign_tubes_to_grid_cells_py(
    tube_bboxes: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray]:
    """纯 Python / NumPy 回退：与 C++ 版**同一判据**，只是慢。

    **实现照抄 :mod:`grains_fast` 的同名热点**（那里因不做 C++ 卸载而一直走
    这条路径，故是**已被使用、已被测试的实现**，不是新写的）。三条保证：

    * **判据相同**：只做包围盒与格边界求交（本模块与 ``grains_fast`` 都刻意
      不做更细的几何测试）；
    * **单位元序相同**：逐管 + 逐格升序，与 ``grains_fast`` 逐位一致；
    * 因此它与 C++ 版应当在**同一构型上给出相同的分格结果**——这条由
      ``tests/test_grains_cpp_fallback.py`` 与 ``grains_fast`` 对拍守门。

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(cell_ids, local_tube_indices)``，语义同 C++ 版。
    """
    step = (hi - lo) / np.array([nx, ny, nz], dtype=float)
    inv_step = 1.0 / step

    cell_xlo, cell_xhi = lo[0] + np.arange(nx) * step[0], lo[0] + np.arange(1, nx + 1) * step[0]
    cell_ylo, cell_yhi = lo[1] + np.arange(ny) * step[1], lo[1] + np.arange(1, ny + 1) * step[1]
    cell_zlo, cell_zhi = lo[2] + np.arange(nz) * step[2], lo[2] + np.arange(1, nz + 1) * step[2]

    cell_ids: list[int] = []
    local_indices: list[int] = []
    for local_idx in range(tube_bboxes.shape[0]):
        tb = tube_bboxes[local_idx]
        ix_min = max(0, int(np.floor((tb[0] - lo[0]) * inv_step[0])))
        ix_max = min(nx - 1, int(np.ceil((tb[1] - lo[0]) * inv_step[0])))
        iy_min = max(0, int(np.floor((tb[2] - lo[1]) * inv_step[1])))
        iy_max = min(ny - 1, int(np.ceil((tb[3] - lo[1]) * inv_step[1])))
        iz_min = max(0, int(np.floor((tb[4] - lo[2]) * inv_step[2])))
        iz_max = min(nz - 1, int(np.ceil((tb[5] - lo[2]) * inv_step[2])))
        if ix_min > ix_max or iy_min > iy_max or iz_min > iz_max:
            continue
        for ix in range(ix_min, ix_max + 1):
            if tb[1] < cell_xlo[ix] or tb[0] > cell_xhi[ix]:
                continue
            base_x = ix * ny * nz
            for iy in range(iy_min, iy_max + 1):
                if tb[3] < cell_ylo[iy] or tb[2] > cell_yhi[iy]:
                    continue
                base_xy = base_x + iy * nz
                for iz in range(iz_min, iz_max + 1):
                    if tb[5] >= cell_zlo[iz] and tb[4] <= cell_zhi[iz]:
                        cell_ids.append(base_xy + iz)
                        local_indices.append(local_idx)

    return (
        np.asarray(cell_ids, dtype=np.int64),
        np.asarray(local_indices, dtype=np.int64),
    )


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


@dataclass
class GrainsDecomposition:
    """一组构成某个管集合之分解的棱柱（grain）。"""

    prisms: List[Prism] = field(default_factory=list)

    def add(self, prism: Prism) -> None:
        self.prisms.append(prism)

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
    """快速的近似 grain 构造。

    对缓存友好、精度更低的变体：
      * 对较小的 ρ-bin 粗化网格；
      * 用扁平的 list-of-lists 代替 dict 来做格子 -> 管的映射；
      * 用廉价的 bbox 重叠检验代替精确的线段-盒检验；
      * 把所有管的 bbox 与格子边界都预计算成数组。

    由此得到的 grain 会略大，且可能包含精确构造本会分配的那批管的一个
    超集，但当后续的 move 只需要一个粗糙骨架时，这是可以接受的。
    """
    if not tubes:
        return GrainsDecomposition()

    tube_list = list(tubes)
    n_tubes = len(tube_list)
    tube_volume = tube_list[0].volume
    if global_indices is None:
        global_indices = list(range(n_tubes))
    global_indices_list = list(global_indices)

    bbox = _axis_aligned_bounding_box(tubes)
    lo = bbox.bounds[:, 0]
    hi = bbox.bounds[:, 1]
    dims = hi - lo

    # 从该子集的主方向选定 grain 的拉长轴。
    principal = _principal_direction(tubes)
    long_axis = _world_axis_closest_to_direction(principal)

    # 对很小的 bin 粗化网格，使工作量保持 O(n_tubes) 并改善缓存利用。
    if n_tubes <= 2:
        effective_grid_n = 1
    elif n_tubes <= 8:
        effective_grid_n = 2
    else:
        effective_grid_n = max(1, grid_n)

    grid, _ = _long_axis_grid(dims, effective_grid_n, long_axis=long_axis)
    nx, ny, nz = int(grid[0]), int(grid[1]), int(grid[2])

    # 每个轴上的格子边界。
    x_edges = np.linspace(lo[0], hi[0], nx + 1)
    y_edges = np.linspace(lo[1], hi[1], ny + 1)
    z_edges = np.linspace(lo[2], hi[2], nz + 1)

    # 预计算各管的 bbox：(n_tubes, 6) = [xlo, xhi, ylo, yhi, zlo, zhi]。
    tube_bboxes = np.empty((n_tubes, 6), dtype=float)
    for i, t in enumerate(tube_list):
        a, b = t.endpoints
        ax, ay, az = a
        bx, by, bz = b
        tube_bboxes[i, 0] = min(ax, bx) - t.delta
        tube_bboxes[i, 1] = max(ax, bx) + t.delta
        tube_bboxes[i, 2] = min(ay, by) - t.delta
        tube_bboxes[i, 3] = max(ay, by) + t.delta
        tube_bboxes[i, 4] = min(az, bz) - t.delta
        tube_bboxes[i, 5] = max(az, bz) + t.delta

    # 把最热的三重循环交给 C++ 内核。 每次重叠返回一个
    # (cell_id, local_tube_index) 对；把局部下标映射回全局管下标，并用
    # NumPy 完全按格子分组，从而只遍历被填充的格子。
    #
    # **扩展缺失时回退纯 Python**（2026-10-03）：返回 ``None`` 就走下面这段
    # 逐管-逐格判定。它与 C++ 版是**同一判据**（包围盒与格边界求交），只是慢
    # —— 这正是"可选加速"应有的语义：**加速器不在，功能仍在**。
    assigned = _assign_tubes_to_grid_cells(tube_bboxes, lo, hi, nx, ny, nz)
    if assigned is None:
        cell_ids, local_tube_indices = _assign_tubes_to_grid_cells_py(
            tube_bboxes, lo, hi, nx, ny, nz
        )
    else:
        cell_ids, local_tube_indices = assigned

    grains = GrainsDecomposition()
    global_to_local = {g: i for i, g in enumerate(global_indices_list)}

    if cell_ids.size:
        global_tube_indices = np.asarray(global_indices_list, dtype=int)[local_tube_indices]
        order = np.argsort(cell_ids, kind="mergesort")
        sorted_ids = cell_ids[order]
        sorted_tubes = global_tube_indices[order]
        unique_ids, starts = np.unique(sorted_ids, return_index=True)
        segments = np.split(sorted_tubes, starts[1:])

        # 为被填充的格子预计算各轴上的分解。
        stride_yz = ny * nz
        ix_arr = unique_ids // stride_yz
        rem = unique_ids % stride_yz
        iy_arr = rem // nz
        iz_arr = rem % nz

        # 五个数组同源于 ``unique_ids``（其余四个都由它算出），长度必然相同。
        # strict 把这个隐含前提写成可见断言：若将来有人改了上游只保留部分
        # 数组，会在这里**立即报错**，而不是静默漏掉后面的格子。
        for cid, ix, iy, iz, seg in zip(
            unique_ids, ix_arr, iy_arr, iz_arr, segments, strict=True
        ):
            indices = seg.tolist()
            p_lo = np.array([x_edges[int(ix)], y_edges[int(iy)], z_edges[int(iz)]])
            p_hi = np.array([x_edges[int(ix) + 1], y_edges[int(iy) + 1], z_edges[int(iz) + 1]])
            prism_box = Box(np.column_stack([p_lo, p_hi]))

            local_indices = [global_to_local[g] for g in indices]
            density = 1.0 if shading is None else float(np.mean(shading.densities[local_indices]))
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


def guth_grains_skeleton(
    tubes: Sequence[Tube],
    shading: Shading | None,
    grid_n: int = 4,
    rho: float | None = None,
    adaptive_rho: bool = True,
    min_tubes_per_bin: int = 8,
) -> GrainsDecomposition:
    """Guth grains 分解的几何感知骨架。

    先把管划分为粗粒度的 ρ-管（方向簇），再为每个簇构造轴对齐的 grain，
    其拉长方向跟随该簇的主方向。 当一根 δ-管的中心线段与某个 grain 的
    距离在 δ 以内时，该管便与该 grain 关联。 每个 grain 存一个它的局部
    Katz–Tao 凸 Wolff 误差的代理量。

    与更早的骨架相比，本版本让 grain 的拉长方向与局部管方向对齐，而不是
    对齐全局坐标轴，使 grain 更接近 Wang–Zahl 中几何上自然的 grain。

    当 ``adaptive_rho`` 为 True（默认）时，初始的 ρ 估计会被放大，直到每个
    ρ-bin 至少含有 ``min_tubes_per_bin`` 根管。 这让 Move #1 在方向分散的
    输入（例如扇形）上也能达成。
    """
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
        local_grains = _build_grains_for_tube_subset(
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


def two_scale_grains_skeleton(
    tubes: Sequence[Tube],
    shading: Shading | None,
    rho: float,
) -> Dict[int, GrainsDecomposition]:
    """两尺度 grains 分解的骨架。

    返回一个映射：从粗粒度的 ρ-管 bin 下标到该 bin 内那些 δ-管的
    GrainsDecomposition。 棱柱的 ``tube_indices`` 是指向原始 ``tubes``
    序列的全局下标。
    """
    bins = partition_into_rho_tubes(tubes, rho)
    result: Dict[int, GrainsDecomposition] = {}
    for bin_idx, indices in bins.items():
        bin_tubes = [tubes[i] for i in indices]
        bin_shading = None if shading is None else Shading(shading.densities[indices])
        local_grains = guth_grains_skeleton(bin_tubes, bin_shading, grid_n=2)
        # 把局部下标映射回全局的管下标。
        global_prisms = []
        bin_direction = _principal_direction(bin_tubes)
        for p in local_grains.prisms:
            global_indices = tuple(indices[i] for i in p.tube_indices)
            global_prisms.append(
                Prism(
                    bounds=p.bounds,
                    tube_indices=global_indices,
                    density=p.density,
                    local_kt_error=p.local_kt_error,
                    bin_index=bin_idx,
                    bin_tube_count=len(indices),
                    total_tube_count=len(tubes),
                    bin_direction=bin_direction,
                )
            )
        result[bin_idx] = GrainsDecomposition(global_prisms)
    return result
