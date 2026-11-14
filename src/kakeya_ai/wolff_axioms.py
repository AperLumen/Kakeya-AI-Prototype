"""δ-管排布的 Wolff 公理检查。

本模块实现了 Wang–Zahl (2025) 定义 1.3 中的 Katz–Tao 凸 Wolff 公理
与 Frostman 板 Wolff 公理。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Protocol, Sequence, TypeVar

import numpy as np
from numpy.typing import ArrayLike

from .primitives import Tube
from .errors import InputValidationError


class ConvexSet(Protocol):
    """Wolff 公理检验所用的凸集协议。"""

    def contains_segment(self, a: np.ndarray, b: np.ndarray, delta: float) -> bool:
        ...

    @property
    def volume(self) -> float:
        ...


class SlabLike(Protocol):
    """Frostman 板检验所用的类板集合协议。"""

    def contains_segment(self, a: np.ndarray, b: np.ndarray, delta: float) -> bool:
        ...

    @property
    def volume(self) -> float:
        ...


#: ``Box`` 的**自身类型**，供 ``from_center_size`` 声明"返回调用它的那个子类"。
#:
#: **为什么需要它**（2026-10-01）：``from_center_size`` 此前写死 ``-> "Box"``。
#: 子类 ``Prism``（``grains.py``）**继承**这个方法，于是类型检查器把
#: ``Prism.from_center_size(...)`` 的返回值推断成 ``Box``——而 ``a``/``b``/``c``/
#: ``rho`` 这些属性只定义在 ``Prism`` 上，故 Pylance 在调用点报"属性未知"
#: （运行时正常）。用 ``TypeVar`` 绑定而非 ``typing.Self``：本包
#: ``requires-python >= 3.10``，而 ``Self`` 是 3.11+ 才有。
_BoxT = TypeVar("_BoxT", bound="Box")


@dataclass(frozen=True)
class Box:
    """轴对齐的长方体区域 [xmin,xmax] × [ymin,ymax] × [zmin,zmax]，即每条轴上的下界与上界。"""

    bounds: np.ndarray  # 形状 (3, 2)
    def __post_init__(self):
        b = np.asarray(self.bounds, dtype=float).reshape(3, 2)
        object.__setattr__(self, "bounds", b)
        if np.any(b[:, 0] >= b[:, 1]):
            raise InputValidationError("Box bounds must satisfy min < max.")

    @property
    def dimensions(self) -> np.ndarray:
        return self.bounds[:, 1] - self.bounds[:, 0]

    @property
    def volume(self) -> float:
        return float(np.prod(self.dimensions))

    def contains_segment(self, a: np.ndarray, b: np.ndarray, delta: float) -> bool:
        """若线段 [a,b] 的（闭）δ-邻域落在盒内则为 True。"""
        pts = np.vstack([np.asarray(a), np.asarray(b)])
        lo = pts.min(axis=0) - delta
        hi = pts.max(axis=0) + delta
        return bool(
            np.all(lo >= self.bounds[:, 0]) and np.all(hi <= self.bounds[:, 1])
        )

    def intersects_segment(self, a: np.ndarray, b: np.ndarray) -> bool:
        """若闭线段 [a,b] 与闭合盒相交则为 True。"""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        # 采用 Liang-Barsky 线段裁剪算法，也即逐轴的平板法。
        t_min, t_max = 0.0, 1.0
        for i in range(3):
            d = b[i] - a[i]
            if abs(d) < 1e-15:
                if a[i] < self.bounds[i, 0] or a[i] > self.bounds[i, 1]:
                    return False
            else:
                t1 = (self.bounds[i, 0] - a[i]) / d
                t2 = (self.bounds[i, 1] - a[i]) / d
                t_enter, t_exit = min(t1, t2), max(t1, t2)
                t_min = max(t_min, t_enter)
                t_max = min(t_max, t_exit)
                if t_min > t_max:
                    return False
        return True

    @classmethod
    def from_center_size(
        cls: "type[_BoxT]", center: "ArrayLike", sizes: "ArrayLike"
    ) -> "_BoxT":
        """由中心与尺寸构造 Box。

        参数标注为 ``ArrayLike`` 而不是 ``Sequence[float]``：实现第一件事就是
        ``np.asarray(..., dtype=float)``，本就接受 array-like。写成
        ``Sequence[float]`` 会把最主要的用法（直接传 ``np.ndarray``）判成类型
        错误——标注与实现意图不符时，该改的是标注，而不是把调用方降级成 list。

        返回类型用 ``_BoxT``（而不是写死 ``Box``）：子类 ``Prism`` 继承本方法，
        写死 ``Box`` 会让 ``Prism.from_center_size(...)`` 被推断成 ``Box``，
        于是 ``Prism`` 独有的属性（``a``/``b``/``c``/``rho``）在类型检查器里
        "不存在"。见 ``_BoxT`` 的定义说明。
        """
        c = np.asarray(center, dtype=float)
        s = np.asarray(sizes, dtype=float)
        return cls(np.column_stack([c - s / 2, c + s / 2]))

    def __repr__(self) -> str:
        return f"Box(bounds={self.bounds.round(4).tolist()})"


def _box_slab_intersection_volume(
    bounds: np.ndarray,
    normal: np.ndarray,
    offset: float,
    half_thickness: float,
) -> float:
    """轴对齐盒与无限板 ``{|n·x - offset| ≤ half_thickness}`` 的**精确**交集体积。

    闭式推导。设盒为 ``∏[cᵢ - Lᵢ/2, cᵢ + Lᵢ/2]``，令 ``y = x - c``，则
    ``n·x - offset = n·y + m``，``m = n·c - offset``。盒对 ``yᵢ → -yᵢ`` 对称，
    故可把法向各分量取绝对值：``n·y = Σ|nᵢ|·tᵢ``（``tᵢ`` 仍均匀在 ``[-Lᵢ/2, Lᵢ/2]``）。
    再平移 ``zᵢ = tᵢ + Lᵢ/2 ∈ [0, Lᵢ]``，得 ``n·y = u' - μ``，其中
    ``u' = Σ|nᵢ|·zᵢ``、``μ = ½Σ|nᵢ|Lᵢ``。于是交集体积分 = 盒体积 ×
    ``P[u' ∈ (μ-m-h, μ-m+h)]``。

    ``u'`` 是若干独立均匀变量的线性组合，其分布正是**广义 Irwin–Hall
    （箱样条）**，CDF 有闭式：设 ``k`` 为 ``aᵢ := |nᵢ|Lᵢ > 0`` 的维数，则

    ``F(t) = (1 / (k!·∏aᵢ)) · Σ_{v∈{0,1}^k} (-1)^{|v|} · (t - v·a)_+^k``。

    该式对 ``k≤3`` 只有至多 8 项，精确到浮点。**这就是此前"属独立课题"的那一步。**
    """
    lo, hi = np.asarray(bounds, dtype=float)[:, 0], np.asarray(bounds, dtype=float)[:, 1]
    sides = hi - lo
    if np.any(sides <= 0) or half_thickness <= 0:
        return 0.0
    box_volume = float(np.prod(sides))
    c = 0.5 * (lo + hi)
    a = np.abs(np.asarray(normal, dtype=float)) * sides
    m = float(np.dot(normal, c) - offset)
    mu = 0.5 * float(np.sum(a))

    idx = [i for i in range(a.size) if a[i] > 1e-15]
    k = len(idx)
    if k == 0:
        # 法向退化（与所有盒边正交）：板要么整盒包含，要么完全不交。
        frac = 1.0 if abs(m) <= half_thickness else 0.0
        return box_volume * frac

    ak = a[idx]
    denom = float(np.prod(ak)) * float(factorial(k))

    def cdf(t: float) -> float:
        s = 0.0
        for v in range(1 << k):
            shift = 0.0
            bits = 0
            for i in range(k):
                if v >> i & 1:
                    shift += ak[i]
                    bits += 1
            x = t - shift
            if x > 0.0:
                s += (-1.0) ** bits * x ** k
        return s / denom

    lo_t = mu - m - half_thickness
    hi_t = mu - m + half_thickness
    frac = cdf(hi_t) - cdf(lo_t)
    return box_volume * min(max(frac, 0.0), 1.0)


@dataclass(frozen=True)
class Slab:
    """板（Slab）：单位球与一张加厚平面的交集。

    为便于计算检验，我们把它近似为区域
    { |n · x - d| <= half_thickness } 再裁剪到包围盒。
    """

    normal: np.ndarray
    offset: float
    half_thickness: float
    bounding_box: Box

    def __post_init__(self):
        n = np.asarray(self.normal, dtype=float)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            raise InputValidationError("Slab normal must be non-zero.")
        object.__setattr__(self, "normal", n / norm)

    @property
    def volume(self) -> float:
        """板 ∩ 包围盒的**精确**体积（广义 Irwin–Hall 闭式）。

        **历史**：2026-09-28 记录了一条缺陷——旧实现取"主导轴盒面面积 × 厚度"，
        于是体积**完全不随法向变化**（实测 ``d=(2,2,2)``、``h=0.1`` 时法向
        ``(0,0,1)``/``(0.577,0.577,0.577)``/``(0.9,0.3,0.316)``/``(0.1,0.1,0.99)``/
        ``(0.8,0.6,0)`` 全返回 ``0.80000``，而真值在 ``0.80 ~ 1.04``），且**系统性
        偏小**（对角法向偏低约 23%）。因 ``frostman_slab_error = n/(|W|·#T)``，
        分母偏小 ⇒ 误差偏大 ⇒ ``is_sticky`` 偏 non-sticky——一条**假阴性**。

        **2026-10-09 修复**：改为调用 :func:`_box_slab_intersection_volume` 的精确
        闭式（板 ∩ 盒，广义 Irwin–Hall CDF，至多 8 项）。对 ``d=(2,2,2)``：
        轴对齐法向仍为 ``0.8000``，对角法向修正为 ``1.0358``（蒙特卡洛 ``1.0307``），
        与真值相对误差 ≲ 1%。方向污染与系统性偏小随之消除，``frostman_slab_error``
        不再因体积低估而被放大。

        Returns
        -------
        float
            板与包围盒交集的体积（``half_thickness <= 0`` 时为 0）。
        """
        return _box_slab_intersection_volume(
            self.bounding_box.bounds,
            self.normal,
            self.offset,
            self.half_thickness,
        )

    def contains_segment(self, a: np.ndarray, b: np.ndarray, delta: float) -> bool:
        """若线段 [a,b] 的 δ-邻域落在板内则为 True。"""
        a, b = np.asarray(a), np.asarray(b)
        n = self.normal
        vals = np.array([np.dot(n, a), np.dot(n, b)])
        lo, hi = vals.min() - delta, vals.max() + delta
        if hi > self.offset + self.half_thickness or lo < self.offset - self.half_thickness:
            return False
        # 还要求线段落在包围盒内（作为单位球的代理）。
        pts = np.vstack([a, b])
        box_lo = pts.min(axis=0) - delta
        box_hi = pts.max(axis=0) + delta
        return bool(
            np.all(box_lo >= self.bounding_box.bounds[:, 0])
            and np.all(box_hi <= self.bounding_box.bounds[:, 1])
        )

    @classmethod
    def from_thickness(
        cls,
        normal: "ArrayLike",
        offset: float,
        thickness: float,
        bounding_box: Box,
    ) -> "Slab":
        # 同 ``Box.from_center_size``：实现直接 ``np.asarray``，故接受 array-like。
        # 标注成 ``Sequence[float]`` 会把传 ndarray 的主要用法判成错误。
        return cls(np.asarray(normal), offset, thickness / 2, bounding_box)

    def __repr__(self) -> str:
        return (
            f"Slab(normal={self.normal.round(4).tolist()}, "
            f"offset={self.offset}, half_thickness={self.half_thickness})"
        )


@dataclass(frozen=True)
class Parallelotope:
    """旋转盒子（平行体）：``{ c + Σ_i t_i h_i e_i : |t_i| ≤ 1 }``。

    **为什么需要它**（局限 #5 / 路线图"扩大 Wolff 测试集"）：``Box`` 只覆盖
    **轴对齐**凸集，而 Katz–Tao 公理要求对**所有凸集** $W$ 成立。一族沿对角线
    的管，其"最紧凸包"是斜的——轴对齐盒只能靠**放很大**才装得下它们，于是
    ``n·|tube|/|W|`` 被系统性压小，公理看起来比实际更好满足。平行体把测试集
    从"轴对齐"扩到"任意朝向的盒子"（仍是凸集，且 ``contains``/``volume`` 精确、
    开销与 Box 同阶），是对该局限最直接的一档补强。

    局部坐标由正交归一的 ``axes``（行=e₁,e₂,e₃）给出：``x_local = axes @ (x - c)``。
    旋转保距，故 δ-邻域在局部坐标里仍是 δ-邻域。
    """

    center: np.ndarray      # (3,)
    axes: np.ndarray        # (3, 3)，行=正交归一基
    half_extents: np.ndarray  # (3,)，> 0

    def __post_init__(self):
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float).reshape(3))
        ax = np.asarray(self.axes, dtype=float).reshape(3, 3)
        object.__setattr__(self, "axes", ax)
        he = np.asarray(self.half_extents, dtype=float).reshape(3)
        if np.any(he <= 0):
            raise InputValidationError("Parallelotope half_extents must be > 0.")
        object.__setattr__(self, "half_extents", he)

    @property
    def volume(self) -> float:
        return float(np.prod(2.0 * self.half_extents))

    def to_local(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        return (p - self.center) @ self.axes.T

    def contains_segment(self, a: np.ndarray, b: np.ndarray, delta: float) -> bool:
        """为真当且仅当 segment [a,b] 的 δ-邻域完全落在平行体内。"""
        pts = self.to_local(np.vstack([np.asarray(a), np.asarray(b)]))
        lo = pts.min(axis=0) - delta
        hi = pts.max(axis=0) + delta
        # 边界情形（管端恰好贴在盒面 + δ）会出现 1 ULP 级误差，留极小容差。
        eps = 1e-12
        return bool(
            np.all(lo >= -self.half_extents - eps)
            and np.all(hi <= self.half_extents + eps)
        )

    @property
    def world_aabb_half(self) -> np.ndarray:
        """平行体在世界轴下的 AABB 半长：``(|axes|ᵀ @ half_extents)``。"""
        return np.abs(self.axes.T) @ self.half_extents

    @classmethod
    def from_frame(
        cls, center: ArrayLike, axes: ArrayLike, half_extents: ArrayLike
    ) -> "Parallelotope":
        return cls(np.asarray(center), np.asarray(axes), np.asarray(half_extents))

    def __repr__(self) -> str:
        return (
            f"Parallelotope(center={self.center.round(4).tolist()}, "
            f"half_extents={self.half_extents.round(4).tolist()})"
        )


def count_tubes_in_set(tubes: Sequence[Tube], region) -> int:
    """统计中心线段被包含在 region 内的管 T 的数量。"""
    count = 0
    for t in tubes:
        a, b = t.endpoints
        if region.contains_segment(a, b, t.delta):
            count += 1
    return count


def katz_tao_convex_error(
    tubes: Sequence[Tube],
    convex_sets: Sequence[ConvexSet],
) -> float:
    """Compute C_KT-CW(T) for a given family of convex test sets.

    Definition 1.3(A):  #{T ⊂ W} ≤ C |W| / |T|  for all convex W.

    **注意（2026-09-15）**：返回值为 0.0 有两种可能——(a) 每个测试集都被
    检验且误差都是 0，(b) **没有任何测试集包含任何管**（一个都没测到）。
    两者对调用方含义完全不同，但这里无法区分，因为要区分必须知道覆盖情况。
    需要区分时用 ``katz_tao_convex_error_checked()``（它额外返回是否有覆盖）。
    """
    err, _n_included = _katz_tao_convex_error_impl(tubes, convex_sets)
    return err


def _katz_tao_convex_error_impl(
    tubes: Sequence[Tube],
    convex_sets: Sequence[ConvexSet],
) -> tuple[float, int]:
    """返回 (误差, 被至少一个测试集包含的管数)。

    第二个返回值就是"到底测到了几根管"——为 0 时误差那是虚假的 0.0，
    不能当作"满足公理"。这个拆分是为了让调用方（如 ``build_scale_chain``）
    能如实报告"未测"而不是"通过"。
    """
    if not tubes:
        return 0.0, 0
    tube_vol = tubes[0].volume
    errors = []
    max_included = 0
    for w in convex_sets:
        n = count_tubes_in_set(tubes, w)
        max_included = max(max_included, n)
        if n == 0 or w.volume <= 0:
            continue
        errors.append(n * tube_vol / w.volume)
    return (float(max(errors)) if errors else 0.0), max_included


def katz_tao_convex_error_checked(
    tubes: Sequence[Tube],
    convex_sets: Sequence[ConvexSet],
) -> tuple[float, int]:
    """与 ``katz_tao_convex_error`` 相同，但额外返回被包含的管数。

    Returns
    -------
    (error, n_included) : tuple[float, int]
        ``n_included == 0`` 表示所有测试集都没装下任何管，此时 ``error`` 是
        虚假的 0.0，**不得当作"满足公理"**。
    """
    return _katz_tao_convex_error_impl(tubes, convex_sets)


def frostman_slab_error(
    tubes: Sequence[Tube],
    slabs: Sequence[SlabLike],
) -> float:
    """对给定的一族板检验集计算 C_F-SW(T)。

    Definition 1.3(B):  #{T ⊂ W} ≤ C |W| (#T)  for all slabs W.

    **注意（2026-09-15）**：与 ``katz_tao_convex_error`` 同理，返回 0.0 可能
    是"都测了且都是 0"，也可能是"一个都没测到"。
    """
    err, _n_included = _frostman_slab_error_impl(tubes, slabs)
    return err


def _frostman_slab_error_impl(
    tubes: Sequence[Tube],
    slabs: Sequence[SlabLike],
) -> tuple[float, int]:
    """返回 (误差, 被至少一个测试集包含的管数)。"""
    if not tubes:
        return 0.0, 0
    n_tubes = len(tubes)
    errors = []
    max_included = 0
    for w in slabs:
        n = count_tubes_in_set(tubes, w)
        max_included = max(max_included, n)
        if n == 0 or w.volume <= 0:
            continue
        errors.append(n / (w.volume * n_tubes))
    return (float(max(errors)) if errors else 0.0), max_included


def frostman_slab_error_checked(
    tubes: Sequence[Tube],
    slabs: Sequence[SlabLike],
) -> tuple[float, int]:
    """与 ``frostman_slab_error`` 相同，但额外返回被包含的管数。

    ``n_included == 0`` 时 ``error`` 是虚假的 0.0，不得当作"满足公理"。
    """
    return _frostman_slab_error_impl(tubes, slabs)


def check_katz_tao_axiom(
    tubes: Sequence[Tube],
    convex_sets: Sequence[ConvexSet],
    max_error: float,
) -> bool:
    """Return True if C_KT-CW(T) ≤ max_error over the supplied test sets.

    **警告（2026-09-15）**：当没有任何测试集包含管时，误差为 0.0，本函数会
    返回 True——那是"没测到"而非"满足"。要区分两者请用
    ``check_katz_tao_axiom_checked()``，或自行用
    ``katz_tao_convex_error_checked()`` 拿到的 ``n_included`` 判定。

    保留这个宽松语义是为了向后兼容（既有调用方与测试依赖它）；但**新代码
    应当用 checked 版本**，避免把"未测"当成"通过"。
    """
    return katz_tao_convex_error(tubes, convex_sets) <= max_error


def check_katz_tao_axiom_checked(
    tubes: Sequence[Tube],
    convex_sets: Sequence[ConvexSet],
    max_error: float,
) -> tuple[bool, int]:
    """与 ``check_katz_tao_axiom`` 相同，但额外返回被包含的管数。

    ``n_included == 0`` 时返回值不是可信的"满足"（什么都没测到）。
    """
    err, n_included = katz_tao_convex_error_checked(tubes, convex_sets)
    return err <= max_error, n_included


def check_frostman_slab_axiom(
    tubes: Sequence[Tube],
    slabs: Sequence[SlabLike],
    max_error: float,
) -> bool:
    """若在所给检验集上 C_F-SW(T) ≤ max_error 则返回 True。

    **警告（2026-09-15）**：与 ``check_katz_tao_axiom`` 同理，零覆盖时返回
    True 是"没测到"。需要区分时用 ``check_frostman_slab_axiom_checked()``。
    """
    return frostman_slab_error(tubes, slabs) <= max_error


def check_frostman_slab_axiom_checked(
    tubes: Sequence[Tube],
    slabs: Sequence[SlabLike],
    max_error: float,
) -> tuple[bool, int]:
    """与 ``check_frostman_slab_axiom`` 相同，但额外返回被包含的管数。"""
    err, n_included = frostman_slab_error_checked(tubes, slabs)
    return err <= max_error, n_included


def make_test_boxes(
    bounding_box: Box,
    n_boxes: int = 16,
    seed: int | None = None,
) -> list[Box]:
    """在 ``bounding_box`` 内生成随机的轴对齐盒。

    这些盒可用作检验凸集，用来估计 Katz–Tao 凸 Wolff 常数。 各边尺寸在
    对数空间上均匀采样，从而小盒与大盒都会被覆盖到。
    """
    if n_boxes <= 0:
        return []

    rng = np.random.default_rng(seed)
    lo = bounding_box.bounds[:, 0]
    hi = bounding_box.bounds[:, 1]
    dims = hi - lo

    boxes: list[Box] = []
    for _ in range(n_boxes):
        # 每条边占包围盒的比例。
        frac = rng.uniform(0.05, 1.0, size=3)
        size = dims * frac
        centre = rng.uniform(lo + size / 2, hi - size / 2)
        boxes.append(Box.from_center_size(centre, size))
    return boxes


def make_test_slabs(
    bounding_box: Box,
    n_slabs: int = 16,
    seed: int | None = None,
) -> list[Slab]:
    """在 ``bounding_box`` 内生成随机的类板集合。

    这些板可用于估计 Frostman 板 Wolff 常数。 法向在 S² 上均匀分布，
    厚度在对数空间上采样。
    """
    if n_slabs <= 0:
        return []

    rng = np.random.default_rng(seed)
    dims = bounding_box.bounds[:, 1] - bounding_box.bounds[:, 0]

    slabs: list[Slab] = []
    for _ in range(n_slabs):
        # 在 S² 上均匀随机的方向。
        normal = rng.normal(size=3)
        normal = normal / np.linalg.norm(normal)

        # 厚度在最小包围盒边长的 1% 到 100% 之间。
        thickness = rng.uniform(0.01, 1.0) * float(np.min(dims))

        # 选取偏移量，使该板穿过包围盒。
        centre = rng.uniform(
            bounding_box.bounds[:, 0], bounding_box.bounds[:, 1]
        )
        offset = float(np.dot(normal, centre))

        slabs.append(
            Slab.from_thickness(
                normal=normal,
                offset=offset,
                thickness=thickness,
                bounding_box=bounding_box,
            )
        )
    return slabs


def _orthonormal_frame(e0: np.ndarray) -> np.ndarray:
    """由主方向 e0 造正交归一基（第一行 = 单位 e0）。"""
    e0 = np.asarray(e0, dtype=float)
    n = float(np.linalg.norm(e0))
    e0 = e0 / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
    ref = np.eye(3)[int(np.argmin(np.abs(e0)))]
    e1 = np.cross(e0, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(e0, e1)
    return np.vstack([e0, e1, e2])


def tight_parallelotope(tubes: Sequence[Tube]) -> Parallelotope:
    """包住 ``tubes`` 的**最紧平行体**（沿其公共主方向的定向包围盒）。

    这是"内容最少"的凸集之一，因此对 Katz–Tao 比值 ``n·|tube|/|W|`` 是**最不利**
    的测试体——正是"对所有凸集取上确界"最该试的一类。要求 ``tubes`` 方向大致
    一致（同一族平行管）；方向发散时返回的盒子会很松（但仍包含全部管，语义正确）。
    """
    if not tubes:
        raise InputValidationError("tight_parallelotope requires at least one tube.")
    dirs = np.array([t.direction.v for t in tubes], dtype=float)
    mean = dirs.mean(axis=0)
    axes = _orthonormal_frame(mean)
    pts, margin = [], 0.0
    for t in tubes:
        a, b = t.endpoints
        pts.append(a)
        pts.append(b)
        margin = max(margin, float(t.delta))
    pts = np.asarray(pts, dtype=float)
    base = pts.mean(axis=0)
    local = (pts - base) @ axes.T
    lo, hi = local.min(axis=0), local.max(axis=0)
    half = 0.5 * (hi - lo) + margin
    center = base + (0.5 * (lo + hi)) @ axes
    return Parallelotope.from_frame(center, axes, np.maximum(half, 1e-9))


def make_tube_aligned_parallelotopes(
    tubes: Sequence[Tube], direction_tol: float = 0.15
) -> list[Parallelotope]:
    """按方向聚类，给出包住每一簇管的**最紧平行体**。

    ``direction_tol`` 是方向余弦阈值：两管方向内积 ``≥ 1 - direction_tol`` 视为
    同簇。返回的每个平行体都是能装下该簇的紧凸集，因而对公理检验比随机盒子更狠。
    """
    if not tubes:
        return []
    groups: list[list[Tube]] = []
    reps: list[np.ndarray] = []
    for t in tubes:
        v = np.asarray(t.direction.v, dtype=float)
        for i, rep in enumerate(reps):
            if float(np.dot(v, rep)) >= 1.0 - direction_tol:
                groups[i].append(t)
                break
        else:
            reps.append(v)
            groups.append([t])
    return [tight_parallelotope(g) for g in groups]


def make_test_parallelotopes(
    bounding_box: Box,
    n_parallelotopes: int = 16,
    seed: int | None = None,
) -> list[Parallelotope]:
    """在 ``bounding_box`` 内生成随机的**旋转过的**盒。

    每个体都得到一个均匀随机的正交标架（在 SO(3) 上 Haar 随机，
    由高斯矩阵做 QR 分解得到）以及对数均匀的半轴长，随后被收缩，
    使其世界 AABB 严格落在 ``bounding_box`` 内。与轴对齐的
    :func:`make_test_boxes` 不同，这些凸集能更紧地贴合对角管族——
    这正是扩充检验集的意义所在。
    """
    if n_parallelotopes <= 0:
        return []

    rng = np.random.default_rng(seed)
    lo = bounding_box.bounds[:, 0]
    hi = bounding_box.bounds[:, 1]
    dims = hi - lo

    bodies: list[Parallelotope] = []
    safety = 0
    while len(bodies) < n_parallelotopes and safety < 100 * n_parallelotopes:
        safety += 1
        # 在 SO(3) 上的 Haar 随机标架：做 QR 分解并修正符号。
        m = rng.normal(size=(3, 3))
        q, r = np.linalg.qr(m)
        q = q * np.sign(np.diag(r))
        # 半轴长取自对数空间（盒边长的 5%..50%）。
        frac = rng.uniform(0.05, 0.5, size=3)
        half = 0.5 * dims * frac
        # 收缩到旋转体的世界 AABB 严格落在内部为止。
        aabb_half = np.abs(q.T) @ half
        limit = 0.5 * dims * 0.98
        scale = float(np.min(limit / np.maximum(aabb_half, 1e-12)))
        if scale < 1.0:
            half = half * scale
            aabb_half = np.abs(q.T) @ half
        center = rng.uniform(lo + aabb_half, hi - aabb_half)
        bodies.append(Parallelotope.from_frame(center, q, half))
    return bodies


def make_wolff_test_sets(
    bounding_box: Box,
    n_boxes: int = 16,
    n_parallelotopes: int = 16,
    seed: int | None = None,
) -> list[ConvexSet]:
    """扩充的凸检验族：轴对齐的盒**加上**旋转过的盒。

    这就是局限 #5 所说的"更大、更一般的凸检验集"：在这族集合上通过公理
    检验，比只对 :func:`make_test_boxes` 检验是**严格更强**的陈述，因为
    它包含了那些能紧致地容纳整族对角管的朝向。
    """
    return [
        *make_test_boxes(bounding_box, n_boxes, seed=seed),
        *make_test_parallelotopes(bounding_box, n_parallelotopes, seed=seed),
    ]


def estimate_katz_tao_constant(
    tubes: Sequence[Tube],
    bounding_box: Box,
    n_test_boxes: int = 64,
    seed: int | None = None,
) -> float:
    """在一族随机检验盒上估计 C_KT-CW(T)。"""
    return katz_tao_convex_error(
        tubes, make_test_boxes(bounding_box, n_test_boxes, seed=seed)
    )


def estimate_katz_tao_constant_general(
    tubes: Sequence[Tube],
    bounding_box: Box,
    n_test_boxes: int = 32,
    n_test_parallelotopes: int = 32,
    seed: int | None = None,
) -> float:
    """Estimate C_KT-CW(T) over boxes **and** rotated boxes (局限 #5 补强）。

    取两个家族的**最大**误差——比 :func:`estimate_katz_tao_constant` 更紧的下界
    估计，因为旋转盒子能贴合轴对齐盒装不下的斜向管族。
    """
    return katz_tao_convex_error(
        tubes,
        make_wolff_test_sets(
            bounding_box,
            n_boxes=n_test_boxes,
            n_parallelotopes=n_test_parallelotopes,
            seed=seed,
        ),
    )


def adversarial_convex_error(
    tubes: Sequence[Tube],
    convex_sets: Sequence[ConvexSet] | None = None,
) -> tuple[float, int]:
    """对"最不利凸集"取上确界的误差估计，并报告覆盖情况。

    家族由三部分组成：

    1. 外部传入的 ``convex_sets``（可选）；
    2. 每根管的**自身**最紧平行体（单管盒）；
    3. 按方向聚类后的**簇**最紧平行体（含多管的紧盒）。

    第 3 类是最狠的：它把"能装下一整簇平行管的最小凸集"直接拿来做分母，正对应
    Katz–Tao 公理里"对所有凸集 $W$ 取上确界"中真正可能取到最大值的那类 $W$。

    Returns
    -------
    (error, n_included) : tuple[float, int]
        ``n_included`` 为被至少一个测试体包含的管数；为 0 表示"什么都没测到"，
        此时 ``error == 0.0`` 不可当作"满足公理"。
    """
    family: list[ConvexSet] = list(convex_sets or [])
    family.extend(tight_parallelotope([t]) for t in tubes)
    family.extend(make_tube_aligned_parallelotopes(tubes))
    return katz_tao_convex_error_checked(tubes, family)


def estimate_frostman_constant(
    tubes: Sequence[Tube],
    bounding_box: Box,
    n_test_slabs: int = 64,
    seed: int | None = None,
) -> float:
    """在一族随机检验板上估计 C_F-SW(T)。"""
    return frostman_slab_error(
        tubes, make_test_slabs(bounding_box, n_test_slabs, seed=seed)
    )
