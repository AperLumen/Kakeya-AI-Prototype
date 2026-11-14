"""Sticky Kakeya 理论的算法化实现。

本模块把 Wang–Zahl (2025) 论文中 Sticky Kakeya 定理与
Nikishin–Stein–Pisier 因子化部分转化为可执行算法，包括：

1. **尺度链构造** (Scale Chain Construction)
   给定 δ-管族 T，构造嵌套尺度链
       δ = ρ_N < ρ_{N-1} < … < ρ_0 = 1
   并在每个尺度 ρ_i 验证 Katz–Tao 凸 Wolff 公理。

2. **Sticky 判定** (Sticky Classification)
   判定管族是否满足 Sticky Kakeya 定理的输入条件：
   存在尺度链使得每一层 T_{ρ_i} 都满足 Wolff 公理 (误差 ≲ 1)。

3. **Nikishin–Stein–Pisier 因子化**
   当 #T ≪ δ^{-2} 时，通过随机平移/旋转复制把管族放大到
   #T̂ ≈ δ^{-2}，并验证并集体积在刚体变换下不变。

4. **Sticky Kakeya 体积验证**
   对 sticky 管族验证体积下界 |∪T| ≥ (#T)|T|。

5. **多尺度归纳** (Multi-scale Induction)
   Proposition 1.7 的自提升循环：E(σ,ω) ⇒ D(σ, ω−g(σ,ω))，
   通过尺度链逐步压缩 ω。

数学背景
--------
Sticky Kakeya 定理 (Katz–Tao, Hickman–Rogers–Zahl)：
若 δ-管族 T 满足 #T ≈ δ^{-2} 且对每个尺度 ρ ∈ [δ, 1]
存在 coarse-graining T_ρ 满足 Wolff 公理，则

    |∪_{T∈𝕋} T|  ≥  C · (#T) · |T|  =  C · (#T) · πδ² · length.

NSP 因子化：
若 #T ≪ δ^{-2}，取 N_copy ≈ δ^{-2} / #T 份随机刚体变换
(R_i, t_i) 的副本 T̂ = ∪_i (R_i T + t_i)，则 #T̂ ≈ δ^{-2}
且 |∪T̂| ≈ N_copy · |∪T|（体积在刚体变换下不变）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Sequence, Tuple

import numpy as np

from .grains import partition_into_rho_tubes
from .errors import InputValidationError
from .primitives import Direction, Tube
from .volume import estimate_union_volume_voxel
from .wolff_axioms import (
    Box,
    ConvexSet,
    frostman_slab_error_checked,
    katz_tao_convex_error_checked,
    make_tube_aligned_parallelotopes,
)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class ScaleLayer:
    """尺度链中的一层：尺度 ρ_i 处的管族粗粒化状态。

    Attributes
    ----------
    rho : float
        该层的尺度（管半径参数）。
    tube_count : int
        该层粗粒化后的管数 #T_ρ。
    kt_cw_error : float
        Katz–Tao 凸 Wolff 公理误差 C_KT-CW(T_ρ)。
    frostman_error : float
        Frostman slab 公理误差 C_F-SW(T_ρ)。
    wolff_ok : bool
        两种 Wolff 公理是否均满足 (误差 ≤ max_error)。
    tested : bool
        该层是否**真的被检验过**。当所有测试集都不包含任何管时（粗管半径
        过大、装不进测试集），误差函数会返回 0.0——那看起来像"满足公理"，
        实际是"什么都没测"。这种情况下 ``tested=False``，且 ``wolff_ok``
        不代表公理成立（2026-09-15 修复：此前 ρ≥0.63 的层全部假通过）。
    n_included : int
        被至少一个测试集包含的管数（=0 时 ``tested=False``）。
    """

    rho: float
    tube_count: int
    kt_cw_error: float
    frostman_error: float
    wolff_ok: bool
    tested: bool = True
    n_included: int = -1

    def __repr__(self) -> str:
        if not self.tested:
            status = "?未测"
        else:
            # 不用 ✓/✗ 等符号：终端输出与实验报告都会引用本 repr，
            # 而技术文档里出现 emoji/符号不专业（2026-09-27 定调）。
            status = "满足" if self.wolff_ok else "不满足"
        return (
            f"ScaleLayer(ρ={self.rho:.4f}, #T={self.tube_count}, "
            f"KT-CW={self.kt_cw_error:.2f}, F-SW={self.frostman_error:.4f} {status})"
        )


@dataclass
class ScaleChain:
    """完整的嵌套尺度链 δ = ρ_N < … < ρ_0 = 1。

    Attributes
    ----------
    layers : list[ScaleLayer]
        从粗 (ρ_0=1) 到细 (ρ_N=δ) 的各层状态。
    delta : float
        最细尺度 δ。
    is_sticky : bool
        是否所有层均满足 Wolff 公理。
    """

    layers: List[ScaleLayer] = field(default_factory=list)
    delta: float = 1e-2
    is_sticky: bool = False

    @property
    def n_scales(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        tag = "Sticky" if self.is_sticky else "Non-sticky"
        lines = [f"ScaleChain({tag}, δ={self.delta:.4f}, {self.n_scales} layers):"]
        for i, layer in enumerate(self.layers):
            lines.append(f"  [{i}] {layer}")
        return "\n".join(lines)


@dataclass
class NSPFactorizationResult:
    """Nikishin–Stein–Pisier 因子化的输出。

    Attributes
    ----------
    original_count : int
        原始管数 #T。
    n_copies : int
        随机刚体变换副本数。
    boosted_count : int
        放大后的管数 #T̂ ≈ n_copies × original_count。
    target_count : float
        目标管数 δ^{-2}。
    original_volume : float
        原始管族并集体积 |∪T|。
    boosted_volume : float
        放大管族并集体积 |∪T̂|。
    volume_ratio : float
        boosted_volume / (n_copies × original_volume)，应 ≈ 1。
    meets_target : bool
        boosted_count 是否接近 δ^{-2}。
    """

    original_count: int
    n_copies: int
    boosted_count: int
    target_count: float
    original_volume: float
    boosted_volume: float
    volume_ratio: float
    meets_target: bool
    #: 因子化模式：``"random"``（随机刚体副本，历史行为）或 ``"lattice"``
    #: （**确定性平移格**——把放大看成在格点上取值的**离散测度**，见
    #: :func:`nsp_factorization` 的 ``mode`` 说明）。
    mode: str = "random"
    #: ``mode="lattice"`` 时的格距（沿各轴的平移步长）；其它模式为 0。
    lattice_spacing: float = 0.0

    def __repr__(self) -> str:
        return (
            f"NSP(#T={self.original_count} → #T̂={self.boosted_count}, "
            f"target δ⁻²={self.target_count:.0f}, "
            f"copies={self.n_copies}, "
            f"vol_ratio={self.volume_ratio:.3f}, "
            f"{'达标' if self.meets_target else '未达标'})"
        )


@dataclass
class StickyKakeyaResult:
    """Sticky Kakeya 体积验证的输出。

    Attributes
    ----------
    is_sticky : bool
        管族是否满足 sticky 条件。
    volume : float
        实测并集体积 |∪T|。
    sticky_lower_bound : float
        Sticky 下界 (#T)·|T|。
    ratio : float
        volume / sticky_lower_bound，应 ≥ 1。
    scale_chain : ScaleChain
        尺度链详情。
    nsp : NSPFactorizationResult | None
        若执行了 NSP 因子化，其结果。
    """

    is_sticky: bool
    volume: float
    sticky_lower_bound: float
    ratio: float
    scale_chain: ScaleChain
    nsp: NSPFactorizationResult | None = None

    def __repr__(self) -> str:
        parts = [
            f"StickyKakeya({'Sticky' if self.is_sticky else 'Non-sticky'})",
            f"|∪T|={self.volume:.6f}",
            f"LB=(#T)|T|={self.sticky_lower_bound:.6f}",
            f"ratio={self.ratio:.3f}",
        ]
        if self.nsp is not None:
            parts.append(f"NSP={self.nsp}")
        return " | ".join(parts)


@dataclass
class InductionStep:
    """Proposition 1.7 多尺度归纳的一步。

    Attributes
    ----------
    sigma : float
        当前 σ 参数。
    omega_before : float
        压缩前的 ω。
    omega_after : float
        压缩后的 ω' = ω − g(σ, ω)。
    omega_reduction : float
        g(σ, ω) = ω − ω'。
    delta : float
        当前尺度。
    is_sticky : bool
        该步是否达到 sticky 情形。
    scale_chain : ScaleChain
        该步使用的尺度链。
    """

    sigma: float
    omega_before: float
    omega_after: float
    omega_reduction: float
    delta: float
    is_sticky: bool
    scale_chain: ScaleChain

    def __repr__(self) -> str:
        return (
            f"InductionStep(σ={self.sigma:.3f}, "
            f"ω: {self.omega_before:.4f} → {self.omega_after:.4f}, "
            f"Δω={self.omega_reduction:.4f}, "
            f"δ={self.delta:.4f}, "
            f"{'sticky' if self.is_sticky else 'non-sticky'})"
        )


# ---------------------------------------------------------------------------
# 核心算法
# ---------------------------------------------------------------------------

def random_rigid_transform(
    center: np.ndarray | None = None,
    radius: float = 2.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成一个随机刚体变换 (R, t)。

    Parameters
    ----------
    center : (3,) array or None
        平移的中心偏移。``None`` 等价于原点 ``[0,0,0]``。
    radius : float
        平移范围 [-radius, radius]。
    rng : Generator, optional
        随机数生成器。若 None 且 seed 也为 None，使用默认 rng。
    seed : int, optional
        随机种子。若 rng 未提供，用此构造 Generator。

    Returns
    -------
    R : (3, 3) ndarray
        随机旋转矩阵。
    t : (3,) ndarray
        随机平移向量。
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # ``center`` 用 None 哨兵而不是 ``np.zeros(3)`` 默认值：后者在**函数定义时**
    # 求值一次，所有调用共享同一个数组对象。ndarray 可变，后来者若写
    # ``center += offset`` 或 ``center[0] = ...``，就会静默污染后续所有调用
    # （本函数当前只读，但这份安全性不该依赖"没人会改"）。
    if center is None:
        center = np.zeros(3)

    # 随机旋转：用 QR 分解随机矩阵
    A = rng.standard_normal((3, 3))
    Q, R_upper = np.linalg.qr(A)
    # 确保行列式为 +1（不是反射）
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    R = Q

    # 随机平移
    t = rng.uniform(-radius, radius, size=3) + center
    return R, t


def apply_rigid_transform(
    tubes: Sequence[Tube],
    R: np.ndarray,
    t: np.ndarray,
) -> List[Tube]:
    """对管族施加刚体变换 (R, t)。

    对每根管的中心点和方向都做 R·x + t 变换。
    """
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float)
    transformed = []
    for tube in tubes:
        new_center = R @ tube.center + t
        new_dir_vec = R @ tube.direction.v
        new_dir = Direction(new_dir_vec)
        transformed.append(
            Tube(
                center=new_center,
                direction=new_dir,
                length=tube.length,
                delta=tube.delta,
            )
        )
    return transformed


def _lattice_translations(n: int, spacing: float) -> List[np.ndarray]:
    """返回 ``n`` 个**确定性**的格点平移向量（放大 NSP 的离散测度支撑集）。

    以立方格 ``{(i,j,k)·spacing}`` 铺开（边长为 ``ceil(n^(1/3))``），再按
    ``(‖·‖, i, j, k)`` **确定性排序**取前 ``n`` 个——离原点近的优先，使副本尽量
    集中在域中心。**无随机数**，故同输入必得同输出（``mode="lattice"`` 逐位可复现）。
    """
    if n <= 0:
        return []
    m = max(1, int(np.ceil(n ** (1.0 / 3.0))))
    r = (m - 1) // 2
    offsets = []
    for i in range(-r, m - r):
        for j in range(-r, m - r):
            for k in range(-r, m - r):
                v = np.array([i, j, k], dtype=float) * spacing
                offsets.append((float(np.linalg.norm(v)), i, j, k, v))
    offsets.sort(key=lambda o: (round(o[0], 12), o[1], o[2], o[3]))
    return [o[4] for o in offsets[:n]]


def nsp_factorization(
    tubes: Sequence[Tube],
    delta: float | None = None,
    target_factor: float = 1.0,
    grid_n: int = 64,
    bounds: np.ndarray | None = None,
    seed: int | None = None,
    mode: str = "random",
) -> Tuple[List[Tube], NSPFactorizationResult]:
    """执行 Nikishin–Stein–Pisier 因子化。

    若 #T ≪ δ^{-2}，把管族放大到 #T̂ ≈ δ^{-2}。两种构造：

    * ``mode="random"``（默认，历史行为）——N_copy ≈ δ^{-2}/#T 份**随机刚体
      变换**副本；
    * ``mode="lattice"``——**确定性平移格**：副本由一组格点平移给出，等价于把
      放大看成在格点上取值的**离散计数测度**（更接近 NSP 的测度论形式，且**去
      除随机性**、逐位可复现，不再依赖 ``seed``）。

    **为什么加 lattice**（局限 #3）：此前只有"随机刚体副本"，被如实记为"不是
    Stein–Pisier 的测度论构造"。平移格把构造从"随机"变为"结构化测度"。**仍未
    闭合的部分**：未证明格距下重叠重数的一致上界（论文的 NSP 因子化给出该界；
    本实现只做到"确定性 + 体积不变 + 计数达标"）。

    Parameters
    ----------
    tubes : sequence of Tube
        原始 δ-管族。
    delta : float, optional
        管半径 δ；None 则从 tubes[0].delta 获取。
    target_factor : float
        目标 #T̂ / δ^{-2} 的比例（默认 1.0）。
    grid_n : int
        体素网格分辨率（用于体积估计）。
    bounds : (3, 2) array, optional
        体积估计的包围盒。``None`` 时自动取**覆盖两侧管族的 2 的幂半径**
        （副本会被平移到 ±radius，固定单位盒会把它们漏掉）。显式给出时尊重。
    seed : int, optional
        随机种子。

    Returns
    -------
    boosted_tubes : list[Tube]
        放大后的管族 T̂。
    result : NSPFactorizationResult
        因子化详情。
    """
    if not tubes:
        return [], NSPFactorizationResult(
            original_count=0, n_copies=0, boosted_count=0,
            target_count=0, original_volume=0, boosted_volume=0,
            volume_ratio=0, meets_target=False,
        )

    if delta is None:
        delta = tubes[0].delta

    n_tubes = len(tubes)
    target_count = target_factor / (delta ** 2)

    # 计算需要的副本数
    if n_tubes >= target_count * 0.5:
        # 已经接近 δ^{-2}，不需要因子化
        n_copies = 1
    else:
        n_copies = max(1, int(np.round(target_count / n_tubes)))

    rng = np.random.default_rng(seed)

    # 先把副本生成出来，再统一确定域：副本会被平移到 ±radius，
    # 原来的容量必须在**同一个盒**里算，否则盒外的管被 `contains` 忽略，
    # vol_ratio 会假性偏低（实测：固定 [-1,1]³ 时报告 0.183，覆盖域下为 1.012）。
    if n_copies <= 1:
        boosted_tubes = list(tubes)
    elif mode == "lattice":
        boosted_tubes = list(tubes)  # 第一份是原始
        for t in _lattice_translations(n_copies - 1, delta):
            boosted_tubes.extend(apply_rigid_transform(tubes, np.eye(3), t))
    elif mode == "random":
        boosted_tubes = list(tubes)  # 第一份是原始
        for _ in range(n_copies - 1):
            R, t = random_rigid_transform(rng=rng)
            boosted_tubes.extend(apply_rigid_transform(tubes, R, t))
    else:
        from .errors import InputValidationError
        raise InputValidationError(f"未知 NSP mode={mode!r}（应为 'random' 或 'lattice'）")

    # 域：调用方显式给出则尊重；否则自动取覆盖两侧管族的 2 的幂半径。
    from .domain import DEFAULT_RADIUS, bounds_for_radius, radius_covering
    if bounds is None:
        bounds = bounds_for_radius(
            radius_covering(list(tubes) + list(boosted_tubes), floor=DEFAULT_RADIUS)
        )

    # 原始管族体积（在同一个域上）
    original_vol = estimate_union_volume_voxel(tubes, grid_n=grid_n, bounds=bounds)

    # 放大管族的体积（可能因为重叠而小于 n_copies × original_vol）
    boosted_vol = (
        original_vol if n_copies <= 1
        else estimate_union_volume_voxel(boosted_tubes, grid_n=grid_n, bounds=bounds)
    )

    # 体积比：理想情况下 ≈ 1（无重叠时 = n_copies × original_vol）
    expected_vol = n_copies * original_vol
    volume_ratio = boosted_vol / expected_vol if expected_vol > 0 else 0.0

    meets_target = len(boosted_tubes) >= target_count * 0.5

    result = NSPFactorizationResult(
        original_count=n_tubes,
        n_copies=n_copies,
        boosted_count=len(boosted_tubes),
        target_count=target_count,
        original_volume=original_vol,
        boosted_volume=boosted_vol,
        volume_ratio=volume_ratio,
        meets_target=meets_target,
        mode=mode,
        lattice_spacing=(2.0 * delta if mode == "lattice" and n_copies > 1 else 0.0),
    )

    return boosted_tubes, result


def coarsen_to_rho_tubes(
    tubes: Sequence[Tube],
    rho: float,
) -> List[Tube]:
    """把 δ-管族粗粒化为**真实的 ρ-管族**。

    每个 ρ-bin（由 ``partition_into_rho_tubes`` 按方向球面距离分组）拟合成
    一根有实体的 ρ-管：半径取 ρ、方向取该 bin 的主方向、中心与长度由 bin 内
    管的端点在该方向上的投影确定。

    **为什么需要它**：``build_scale_chain`` 原先只在原始 δ-管上换了个阈值算
    Wolff 误差，粗粒化的结果（bin 划分）仅被用来数个数——于是各层误差完全相同
    （实测 fan 构型 8 层都是 KT-CW=0.1257），"尺度链"名存实亡。这里提供真正
    的粗粒化，让每一层是在**不同尺度的管族**上做检验。

    Parameters
    ----------
    tubes : sequence of Tube
        δ-管族。
    rho : float
        ρ-管的半径（也就是该层的尺度）。

    Returns
    -------
    coarse : list[Tube]
        ρ-管族；``coarse[k]`` 对应第 k 个 bin。方向相近的 δ-管被合并，
        因此 ``len(coarse) <= len(tubes)``。
    """
    from .grains import _principal_direction

    if not tubes or rho <= 0:
        return []

    bins = partition_into_rho_tubes(tubes, rho)
    coarse: List[Tube] = []

    for _bin_idx, indices in bins.items():
        bin_tubes = [tubes[i] for i in indices]
        if not bin_tubes:
            continue

        direction = Direction(_principal_direction(bin_tubes))
        d = direction.v

        # 沿主方向的投影区间 → ρ-管的中心与长度
        projs: List[float] = []
        for t in bin_tubes:
            a, b = t.endpoints
            projs.append(float(np.dot(d, a)))
            projs.append(float(np.dot(d, b)))
        lo, hi = min(projs), max(projs)
        # 长度至少为 2ρ，避免单点退化；覆盖 bin 内所有端点
        length = max(hi - lo, 2.0 * rho)
        along = 0.5 * (lo + hi) * d

        # 垂直方向取 bin 内管中心在垂直于 d 的平面上的平均
        perp = np.zeros(3)
        for t in bin_tubes:
            perp = perp + (t.center - float(np.dot(d, t.center)) * d)
        perp = perp / len(bin_tubes)

        coarse.append(
            Tube(
                center=perp + along,
                direction=direction,
                length=length,
                delta=rho,
            )
        )

    return coarse


def build_scale_chain(
    tubes: Sequence[Tube],
    delta: float | None = None,
    n_scales: int = 8,
    max_wolff_error: float = 1.0,
    convex_test_sets: Sequence[ConvexSet] | None = None,
    convex_test_family: str = "boxes",
    verbose: bool = False,
) -> ScaleChain:
    """构造嵌套尺度链并逐层验证 Wolff 公理。

    从最粗尺度 ρ_0 = 1 逐步细化到 ρ_N = δ，在每层做粗粒化
    (把方向相近的管合并为 ρ-管) 并检查 Katz–Tao 凸 Wolff 公理。

    Parameters
    ----------
    tubes : sequence of Tube
        δ-管族。
    delta : float, optional
        最细尺度。None → tubes[0].delta。
    n_scales : int
        尺度链层数（含 ρ_0 和 ρ_N）。
    max_wolff_error : float
        Wolff 公理误差上界阈值。
    convex_test_sets : sequence of ConvexSet, optional
        用于 KT-CW 检验的凸集列表。若 None，自动构造默认 Box 集。
    convex_test_family : {"boxes", "expanded"}
        ``"boxes"``（默认）：只用轴对齐盒（历史行为，误差估计偏松）。
        ``"expanded"``：另加每层各簇管的**最紧平行体**（旋转盒），把检验集从
        "轴对齐"扩到"任意朝向的凸集"——这是局限 #5 的补强档，得到的误差是更紧
        的下界估计（对平行管族可从 ~0.02 提到 ~π/4）。显式传入
        ``convex_test_sets`` 时本参数被忽略。
    verbose : bool
        是否打印各层状态。

    Returns
    -------
    chain : ScaleChain
    """
    if not tubes:
        return ScaleChain(layers=[], delta=delta or 1e-2, is_sticky=False)

    if delta is None:
        delta = tubes[0].delta

    # 对数均匀分布的尺度链：δ = ρ_N < ... < ρ_0 = 1
    log_scales = np.linspace(np.log(delta), 0.0, n_scales)
    rhos = np.exp(log_scales)
    rhos[0] = delta  # 确保最细尺度精确
    rhos[-1] = 1.0   # 确保最粗尺度精确

    # 测试集：**不在这里默认生成**。
    #
    # 修复记录（2026-09-15）：旧代码在此处 `if convex_test_sets is None:`
    # 就把默认盒子（基于最细层管族）赋给了同一个变量，于是循环里那句
    # "若为 None 则按本层管族重建"永远不成立——各层仍用细尺度盒子，
    # 半径大的粗管一个都装不下，误差返回 0.0 看起来像"通过"。
    # 现在用哨兵区分"调用方显式传入"与"需要按层生成"。
    explicit_test_sets = convex_test_sets is not None

    layers: List[ScaleLayer] = []
    all_ok = True

    for rho in rhos:
        # 粗粒化：把方向相近的 δ-管合并为 ρ-管，并**在粗管上**做检验。
        #
        # 修复记录（2026-09-15）：旧实现只把 bin 划分用来数个数，误差仍在原始
        # δ-管上计算（注释里写的"用 ρ-粗粒化后的管集做检验"与代码不符），
        # 导致所有层的误差数值完全相同——尺度链把同一份数据验了 8 遍。
        # 现在各层在**真实的不同尺度管族**上检验。
        #
        # 最细层 ρ=δ 是恒等：论文的 T_δ 就是输入本身，不该合并。
        if rho > delta * (1.0 + 1e-12):
            coarse_tubes = coarsen_to_rho_tubes(tubes, rho)
            if not coarse_tubes:      # 粗粒化退化时退回原始管族（并如实反映）
                coarse_tubes = list(tubes)
        else:
            coarse_tubes = list(tubes)

        n_coarse = len(coarse_tubes)

        # 测试集**按该层管族**构造：不能沿用最细层的盒子——粗管的半径与跨度
        # 都更大，用细尺度盒子会出现"一个盒子都装不下"的情形，误差函数于是
        # 返回 0.0，看起来像"公理满足"（2026-09-15 修复）。
        if explicit_test_sets:
            # 调用方显式给了测试集：尊重它，但下面照实检测覆盖情况。
            layer_test_sets = list(convex_test_sets)
        else:
            layer_test_sets = _default_convex_test_sets(coarse_tubes)
            if convex_test_family == "expanded":
                # 局限 #5 补强：加入能紧贴斜向管族的旋转盒（最紧平行体）。
                # 轴对齐盒装不下斜向管族时误差被系统性压小，这里把它补回来。
                layer_test_sets = [
                    *layer_test_sets,
                    *make_tube_aligned_parallelotopes(coarse_tubes),
                ]
            elif convex_test_family != "boxes":
                raise InputValidationError(
                    "convex_test_family must be 'boxes' or 'expanded', "
                    f"got {convex_test_family!r}."
                )

        # 该层是否真的有管被测试集包含？计数为 0 时必须报"未测"，不能报"通过"。
        #
        # 注意：这里**不用** checked 版本的返回值去判定，而是同时看两种
        # 公理的覆盖数——两条公理用的测试集类型不同（Box vs Slab），
        # 只要有一边没覆盖，该层的结论就不可靠。
        kt_err, kt_included = katz_tao_convex_error_checked(
            coarse_tubes, layer_test_sets
        )
        layer_slabs = _default_slab_test_sets(coarse_tubes)
        fs_err, fs_included = frostman_slab_error_checked(coarse_tubes, layer_slabs)

        n_included = min(kt_included, fs_included)
        tested = n_included > 0

        # 未测到的层**不宣称满足公理**：wolff_ok=False 会保守地把整条链判为
        # 非 sticky（宁可判非 sticky，也不要基于"没测"去断言 sticky）。
        wolff_ok = tested and (kt_err <= max_wolff_error) and (fs_err <= max_wolff_error)

        if not wolff_ok:
            all_ok = False

        layer = ScaleLayer(
            rho=float(rho),
            tube_count=n_coarse,
            kt_cw_error=kt_err,
            frostman_error=fs_err,
            wolff_ok=wolff_ok,
            tested=tested,
            n_included=n_included,
        )
        layers.append(layer)

        if verbose:
            print(f"  ρ={rho:.4f}: #T_ρ={n_coarse}, "
                  f"KT-CW={kt_err:.2f}, F-SW={fs_err:.4f}, "
                  f"{'满足' if wolff_ok else '不满足'}")

    return ScaleChain(layers=layers, delta=float(delta), is_sticky=all_ok)


def classify_sticky(
    tubes: Sequence[Tube],
    delta: float | None = None,
    n_scales: int = 8,
    max_wolff_error: float = 1.0,
    convex_test_sets: Sequence[ConvexSet] | None = None,
    convex_test_family: str = "boxes",
) -> Tuple[bool, ScaleChain]:
    """判定管族是否满足 Sticky Kakeya 条件。

    Sticky 条件：存在尺度链 δ = ρ_N < … < ρ_0 = 1，
    使得每层 T_{ρ_i} 都满足 Wolff 公理 (误差 ≤ max_wolff_error)。

    Returns
    -------
    is_sticky : bool
    chain : ScaleChain
    """
    chain = build_scale_chain(
        tubes, delta=delta, n_scales=n_scales,
        max_wolff_error=max_wolff_error,
        convex_test_sets=convex_test_sets,
        convex_test_family=convex_test_family,
    )
    return chain.is_sticky, chain


def verify_sticky_kakeya(
    tubes: Sequence[Tube],
    delta: float | None = None,
    n_scales: int = 8,
    max_wolff_error: float = 1.0,
    grid_n: int = 64,
    bounds: np.ndarray | None = None,
    use_nsp: bool = True,
    seed: int | None = None,
    verbose: bool = False,
) -> StickyKakeyaResult:
    """完整的 Sticky Kakeya 体积验证流程。

    流程：
    1. 检查 #T 是否接近 δ^{-2}；若否，执行 NSP 因子化。
    2. 构造尺度链并逐层验证 Wolff 公理。
    3. 若 sticky，验证 |∪T| ≥ (#T)|T|。

    Parameters
    ----------
    tubes : sequence of Tube
    delta : float, optional
    n_scales : int
    max_wolff_error : float
    grid_n : int
        体素网格分辨率。
    bounds : (3, 2) array, optional
    use_nsp : bool
        是否在 #T ≪ δ^{-2} 时执行 NSP 因子化。
    seed : int, optional
    verbose : bool

    Returns
    -------
    result : StickyKakeyaResult
    """
    if not tubes:
        return StickyKakeyaResult(
            is_sticky=False, volume=0, sticky_lower_bound=0,
            ratio=0, scale_chain=ScaleChain(), nsp=None,
        )

    if delta is None:
        delta = tubes[0].delta

    n_tubes = len(tubes)
    target = 1.0 / (delta ** 2)

    # Step 1: NSP 因子化（如果需要）
    nsp_result = None
    working_tubes = list(tubes)

    if use_nsp and n_tubes < target * 0.5:
        if verbose:
            print(f"NSP: #T={n_tubes} ≪ δ⁻²={target:.0f}, 执行因子化...")
        working_tubes, nsp_result = nsp_factorization(
            tubes, delta=delta, grid_n=grid_n, bounds=bounds, seed=seed,
        )
        if verbose:
            print(f"  → #T̂={len(working_tubes)}")
    elif verbose:
        print(f"#T={n_tubes} ≈ δ⁻²={target:.0f}, 无需因子化")

    # Step 2: 构造尺度链
    chain = build_scale_chain(
        working_tubes, delta=delta, n_scales=n_scales,
        max_wolff_error=max_wolff_error, verbose=verbose,
    )

    # Step 3: 体积验证
    volume = estimate_union_volume_voxel(
        working_tubes, grid_n=grid_n, bounds=bounds,
    )
    tube_vol = tubes[0].volume
    effective_count = len(working_tubes)
    sticky_lb = effective_count * tube_vol

    ratio = volume / sticky_lb if sticky_lb > 0 else 0.0

    return StickyKakeyaResult(
        is_sticky=chain.is_sticky,
        volume=volume,
        sticky_lower_bound=sticky_lb,
        ratio=ratio,
        scale_chain=chain,
        nsp=nsp_result,
    )


# ---------------------------------------------------------------------------
# Proposition 1.7 多尺度归纳
# ---------------------------------------------------------------------------

def induction_step_g(
    sigma: float,
    omega: float,
    delta: float,
    zeta: float = 0.01,
) -> float:
    """**启发式**的 ω 压缩量（**不是**论文的 g(σ, ω)，勿当作定理引用）。

    ⚠️ **命名与命题号的说明。** 函数名沿用了论文的符号 ``g(σ, ω)``，且它确实
    服务于 Prop 1.7 的归纳环节，但**本函数给出的数值没有论文依据**：

    - 论文 Prop 1.7 是**存在性**结论——只保证"ω 可以压到 ω−g"，**不给闭式
      公式**。凡"把 ω 压缩多少"的定量陈述，在论文里都不是由某个公式算出的。
    - 下面的 ``g ≈ c_0 · ω · (1 − σ) · |log δ|^{-1}`` 是**量级正确的模型**：
      它复现了"δ 越小压缩越多"、"σ→1 时压缩→0"、"压缩不超过 ω"这三条定性
      性质（见 ``tests/test_sticky_kakeya.py::TestInductionStepG``），但比例
      常数 ``c_0`` 与上限 ``ω/2`` 都是**自定**，不是论文或任何文献的取值。

    内部设计文档第 II 篇已如实登记这条限定（"sticky 判定的自提升公式
    是近似……是量级正确的模型，不是论文的精确估计"）；此处 docstring 与之
    对齐，避免只看代码的人误以为这是论文公式。

    **据此可作与不可作的结论**：

    - 可作："在多尺度归纳下 ω 会收缩，且收缩速度随 δ 减小而加快"（定性）。
    - **不可作**："经过 N 步后 ω 降到某具体值"、"该循环 N 步内收敛"（定量）——
      这些完全由自定常数决定。因此 ``multiscale_induction`` 的返回序列应视为
      **启发式收敛曲线**，不得写进 ``.kpt`` 的定量结论。

    Parameters
    ----------
    sigma : float
        当前 σ 参数（σ → 0 时压缩最大）。
    omega : float
        当前 ω 参数。
    delta : float
        当前尺度。
    zeta : float
        grains 分解增长指数。

    Returns
    -------
    g : float
        ω 的压缩量，``0 <= g <= ω/2``。
    """
    if omega <= 0 or delta >= 1:
        return 0.0

    log_delta = abs(np.log(delta))
    if log_delta < 1e-10:
        return 0.0

    # 压缩量：与 ω 成正比，与 (1-σ) 成正比
    # δ^{-ζ} 增长越大 → 压缩越快
    #
    # ``c_0`` 是**自定**的比例常数（非论文、非文献取值）。它只影响压缩的
    # 绝对速度，不影响上面列出的三条定性性质——所以选它不需要依据，但也
    # 因此不能据此常数报任何定量结论。
    c_0 = 0.1  # 自定，无来源；见 docstring 的"可作与不可作的结论"
    growth = zeta * log_delta
    g = c_0 * omega * (1.0 - sigma) * growth / (1.0 + growth)

    # 上限 ω/2 同样是自定的保守取值（保证压缩量不超过剩余量的一半）。
    g = min(g, omega * 0.5)
    return float(g)


def multiscale_induction(
    tubes: Sequence[Tube],
    sigma_0: float = 0.5,
    omega_0: float = 1.0,
    delta: float | None = None,
    n_scales: int = 8,
    max_wolff_error: float = 1.0,
    max_iterations: int = 20,
    omega_threshold: float = 1e-4,
    zeta: float = 0.01,
    verbose: bool = False,
) -> List[InductionStep]:
    """执行 Proposition 1.7 的多尺度归纳循环。

    迭代地：
    1. 在当前 (σ, ω) 下构造尺度链。
    2. 若 sticky → ω 可以压缩为 ω' = ω − g(σ, ω)。
    3. 若非 sticky → 先用 grains + Moves 化简到 sticky 情形。
    4. 重复直到 ω → 0 或达到迭代上限。

    Parameters
    ----------
    tubes : sequence of Tube
    sigma_0 : float
        初始 σ。
    omega_0 : float
        初始 ω。
    delta : float, optional
    n_scales : int
    max_wolff_error : float
    max_iterations : int
    omega_threshold : float
        ω 的收敛阈值。
    zeta : float
        Grains 增长指数。
    verbose : bool

    Returns
    -------
    steps : list[InductionStep]
        **启发式的**收敛曲线——**不是**对论文 Prop 1.7 的数值复现。

    Notes
    -----
    ⚠️ **返回值的定性/定量边界。** 循环里每一步的 ω 压缩量取自
    ``induction_step_g``，而那个公式的自定常数（``c_0=0.1``、上限 ``ω/2``）
    决定了整条曲线的**下降速度**。因此：

    - 可作："ω 随迭代单调下降"这一**定性**结论（有测试守着）。
    - **不可作**："N 步后 ω ≈ 某值"、"循环在 N 步内收敛"这类**定量**结论。
      论文 Prop 1.7 只保证存在压缩，不给速度。

    **L2 环节可验证（2026-10-09，A4）**：本函数只返回曲线，**终止形态与 L2
    判据由纯函数接出**，二者都不改本函数签名：

    - :func:`classify_induction_termination` —— 判定"为什么停下"
      （``CONVERGED`` / ``STALLED`` / ``DESCENDING``）；**这是唯一的终止判定
      来源**，编排器 :meth:`ProofOrchestrator._recursion_plan` 直接复用它，
      不再自己写一份尾部启发式；
    - :func:`verify_induction_l2` —— 逐步核验 g 的单调性、下界与极限存在性
      （§7.0.2 的 L2 判据），返回 :class:`InductionL2Report`。

    **可复现**：整个循环**不含随机数**（无 ``default_rng``/``random`` 调用；
    尺度链与粗粒化都是确定性几何变换）⇒ 同一输入两次调用产生**逐位相同**的
    ``InductionStep`` 序列（有测试钉住）。这一点与 NSP 因子化不同——后者依赖
    随机刚体变换，故需显式 seed。

    **已接入 ``ProofOrchestrator``**（2026-09-30）：编排器的
    ``_recursion_plan()`` 用本函数返回曲线的**时间结构**（"还在压"还是
    "压不动了"）决定是否继续细化尺度，接入点见
    内部设计文档《工程交付基线》的"多尺度归纳接入"。

    ⚠️ **接入时的边界仍然成立**：判定结论**只进 notes（自由文本）**，
    **不写进 ``ProofMetrics`` 或 ``.kpt`` 的定量字段**——下面第 Notes 条的
    限制不因"接入"而改变。守门里有反向断言钉住这一点
    （``tests/test_multiscale_induction_wiring.py``）。
    """
    if delta is None:
        delta = tubes[0].delta if tubes else 1e-2

    if not tubes:
        return []

    sigma = sigma_0
    omega = omega_0
    steps: List[InductionStep] = []

    for iteration in range(max_iterations):
        if omega <= omega_threshold:
            if verbose:
                print(f"  Iteration {iteration}: ω={omega:.6f} ≤ {omega_threshold}, 收敛")
            break

        # 构造尺度链
        is_sticky, chain = classify_sticky(
            tubes, delta=delta, n_scales=n_scales,
            max_wolff_error=max_wolff_error,
        )

        # **只有 sticky 才压缩 ω**（2026-09-30 修正）。
        #
        # 此前是"无论 sticky 与否都压缩"，且非 sticky 时把 σ 推向 1.0
        # （注释自称"简化处理"）。那与 docstring 第 3 步（"非 sticky →
        # 先用 grains + Moves 化简到 sticky 情形"）**毫无关系**，而且
        # **方向恰好相反**：``g ∝ (1 − σ)``，σ→1 就是在关掉压缩。
        # 实测后果：30 步里 **20 步 Δω≡0**，ω 只降 0.012（1.2%），
        # 永远到不了阈值（见 §7.0.2 的 L2 诊断表）。
        #
        # 论文 Prop 1.7 的自提升**前提是 sticky**；本实现**未实现**
        # "把非 sticky 化简为 sticky"那一步（那需要 grains + Moves 的
        # 完整联动，属 L3 范围，已明确不做）。故这里**如实**：
        # 非 sticky ⇒ 本步压缩量为 0，ω 不动。循环会因此**跑满上限**，
        # 那是**正确**的表现——它说明"这条构型还没进入可压缩的前提"。
        g = induction_step_g(sigma, omega, delta, zeta=zeta) if is_sticky else 0.0
        omega_new = max(0.0, omega - g)

        step = InductionStep(
            sigma=sigma,
            omega_before=omega,
            omega_after=omega_new,
            omega_reduction=g,
            delta=delta,
            is_sticky=is_sticky,
            scale_chain=chain,
        )
        steps.append(step)

        if verbose:
            print(f"  Iteration {iteration}: σ={sigma:.3f}, "
                  f"ω: {omega:.6f} → {omega_new:.6f} (Δω={g:.6f}), "
                  f"{'sticky' if is_sticky else 'non-sticky（无压缩）'}")

        omega = omega_new

        # **非 sticky 时不再推进 σ**（2026-09-30 修正）。
        #
        # 旧代码 ``sigma = min(1.0, sigma + 0.05)`` 把 σ 推向 1，而
        # ``g ∝ (1-σ)`` —— 等于"因为没法压缩，所以把未来压缩的能力也掐掉"。
        # 现在不动机器状态：σ 保持，等待（未来实现的）"化简到 sticky"那一步。
        #
        # **不做的事**：不臆造一个"非 sticky 时如何推进"的规则——论文
        # 没给，本项目也未实现该步骤。空转是**如实**的，不是 bug。

    return steps


# ---------------------------------------------------------------------------
# A4（L2）：尺度链闭环的可判定终止与"环节可验证"
# ---------------------------------------------------------------------------
#
# **背景（§7.0.2 的 L2 定义）**："Prop 1.7 每一步满足 g 的单调性与极限
# （**不要求等于论文常数**）"。缺的从来不是代码量，而是**判据**：
# ``multiscale_induction`` 返回的是一条曲线，但"这条曲线算不算跑完了、
# 停下来的原因是什么、每一步是否满足 L2"此前**只能靠读代码判断**。
#
# 本节把这三件事变成**纯函数判定**，从而使 L2 可机器核验：
#   1. ``classify_induction_termination`` —— 终止**可判定**（收敛 / 停滞 /
#      仍在压），且是**唯一**的判定来源（编排器不再自己写一份尾部启发式，
#      避免两处漂移）；
#   2. ``verify_induction_l2`` —— 逐步检查 g 的单调性、下界与极限存在性；
#   3. 可复现性由"整个循环**无随机数**"保证（见 ``multiscale_induction``
#      docstring）；同输入 → 逐位相同的 ``InductionStep`` 序列。
#
# **不做的事**：不构造"非 sticky ⇒ sticky"的化简（那需论文完整构造 ⇒ L3）。
# L2 只要求**已有环节**可验证，不要求把它变成论文的数值复现。


class InductionTermination(Enum):
    """``multiscale_induction`` 循环的终止形态（结构性判定，非定量结论）。

    - ``CONVERGED``：ω 已降到阈值以下——**归纳真的跑完了**（闭环）。
    - ``STALLED``：末尾压缩量 ≈ 0——ω 压不动了。**在未实现"非 sticky →
      sticky 化简"（L3）的前提下，这是正确的表现**：它说明构型还没进入可压缩
      的前提，而不是"循环有 bug"。调用方据此**不该**再细化尺度。
    - ``DESCENDING``：ω 仍在单调下降，但触到了 ``max_iterations`` 预算——类似
      ``BUDGET_EXHAUSTED``，**加迭代可能有效**。与 ``STALLED`` 语义相反，二者
      不可混同（这正是旧代码把两种终止混成"跑满上限"的同一个病）。
    """

    CONVERGED = auto()
    STALLED = auto()
    DESCENDING = auto()


#: "末尾若干步"的窗口长度——判 STALLED 时看曲线尾部多少步。
_INDUCTION_TAIL: int = 5
#: 压缩量"≈ 0"的相对阈值（相对初始 ω）。绝对尺度无关，故用相对量。
_INDUCTION_STALL_RTOL: float = 1e-6


def classify_induction_termination(
    steps: Sequence[InductionStep],
    *,
    omega_threshold: float = 1e-4,
) -> Tuple[InductionTermination, str]:
    """判定归纳循环"为什么停下"（**唯一**的终止判定来源）。

    **判据（只看结构，不看 ω 的具体数值）**——顺序即优先级：

    1. **空序列** ⇒ ``CONVERGED``：一步都没跑（ω 已在阈值下）＝已收敛。
    2. **末步 ``omega_after <= omega_threshold``** ⇒ ``CONVERGED``。
    3. **末尾 ``_INDUCTION_TAIL`` 步压缩量都 ≈ 0** ⇒ ``STALLED``。
    4. **ω 全程不下降** ⇒ ``STALLED``（已到不动点）。
    5. 否则 ⇒ ``DESCENDING``（还在压，但预算用尽）。

    返回 ``(termination, reason)``；``reason`` 供人读（进 notes）。

    **为什么把"收敛"与"停滞"分开**：二者都让 ``should_recurse=False``，但
    含义相反——``CONVERGED`` 是"达到了目的"，``STALLED`` 是"前提不成立、
    压不动"。把它们混成一个"跑满上限"会让读结果的人以为"再跑跑就好了"。
    """
    if not steps:
        return InductionTermination.CONVERGED, "归纳已收敛（0 步：ω 已在阈值下）"

    if steps[-1].omega_after <= omega_threshold:
        return (
            InductionTermination.CONVERGED,
            f"ω 已降到阈值 {omega_threshold:g} 以下（{len(steps)} 步收敛）",
        )

    scale = max(abs(steps[0].omega_before), 1e-30)
    tail = steps[-min(_INDUCTION_TAIL, len(steps)):]
    if all(abs(s.omega_reduction) <= _INDUCTION_STALL_RTOL * scale for s in tail):
        return (
            InductionTermination.STALLED,
            f"归纳停滞（末尾 {len(tail)} 步压缩量 ≈ 0），再细化尺度无益",
        )

    if not any(s.omega_after < s.omega_before for s in steps):
        return InductionTermination.STALLED, "ω 不再下降（已到不动点）"

    return (
        InductionTermination.DESCENDING,
        f"归纳仍在压缩（{len(steps)} 步内 ω 单调下降），继续细化尺度",
    )


@dataclass
class InductionL2Report:
    """``multiscale_induction`` 输出的 **L2 环节可验证性**报告。

    对应 §7.0.2 的 L2 判据："每一步满足 g 的单调性与极限（不要求等于论文
    常数）"。五项逐一给出，``ok`` 是它们的合取。

    Attributes
    ----------
    g_nonnegative : bool
        每一步压缩量 ``omega_reduction >= 0``（⇒ ω 不增）。
    omega_bounded_below : bool
        每一步 ``omega_after >= 0``（ω 有下界）。
    step_arithmetic_consistent : bool
        每步满足更新式 ``omega_after == max(0, omega_before - g)``。
    monotone : bool
        ω 序列单调不增（即未出现"压缩后又反弹"）。
    termination_consistent : bool
        终止判定与曲线自洽：``CONVERGED`` ⇒ 末步 ω ≤ 阈值；``STALLED`` ⇒
        末尾压缩量 ≈ 0。二者都不成立时（如 ``DESCENDING``）恒为真。
    termination : InductionTermination
        终止形态。
    limit : float
        ω 序列的**极限**（单调不增 + 有下界 ⇒ 极限存在，取末值即可）。
    n_steps : int
        步数。
    """

    g_nonnegative: bool
    omega_bounded_below: bool
    step_arithmetic_consistent: bool
    monotone: bool
    termination_consistent: bool
    termination: InductionTermination
    limit: float
    n_steps: int

    @property
    def ok(self) -> bool:
        """L2 判据是否全部满足。"""
        return (
            self.g_nonnegative
            and self.omega_bounded_below
            and self.step_arithmetic_consistent
            and self.monotone
            and self.termination_consistent
        )

    def summary(self) -> str:
        bad = [
            name for name, v in (
                ("g≥0", self.g_nonnegative),
                ("ω≥0", self.omega_bounded_below),
                ("更新式自洽", self.step_arithmetic_consistent),
                ("单调不增", self.monotone),
                ("终止自洽", self.termination_consistent),
            ) if not v
        ]
        verdict = "通过" if self.ok else f"不通过（{', '.join(bad)}）"
        return (
            f"L2[{verdict}] {self.termination.name}, "
            f"{self.n_steps} 步, 极限 ω≈{self.limit:.6g}"
        )


def verify_induction_l2(
    steps: Sequence[InductionStep],
    *,
    omega_threshold: float = 1e-4,
) -> InductionL2Report:
    """核验 ``multiscale_induction`` 输出满足 **L2**（§7.0.2）。

    **L2 的准确定义**（不夸大）：只检查**已有环节**是否满足 g 的单调性与
    极限——**不**要求等于论文常数（那需要论文的闭式 g，而 Prop 1.7 是存在性
    结论、不给公式 ⇒ 属 L3）。

    因此本函数**只**核验结构性质（不比较 ω 的绝对值）：
    ``g >= 0``、``ω >= 0``、更新式自洽、单调不增、终止自洽。单调不增且有下界
    即保证**极限存在**（单调收敛定理），其值取末值。
    """
    n = len(steps)
    if n == 0:
        termination, _ = classify_induction_termination(steps, omega_threshold=omega_threshold)
        return InductionL2Report(
            g_nonnegative=True,
            omega_bounded_below=True,
            step_arithmetic_consistent=True,
            monotone=True,
            termination_consistent=True,
            termination=termination,
            limit=0.0,
            n_steps=0,
        )

    g_nonneg = all(s.omega_reduction >= 0 for s in steps)
    bounded = all(s.omega_after >= 0 for s in steps)
    arithmetic = all(
        abs(s.omega_after - max(0.0, s.omega_before - s.omega_reduction)) < 1e-12
        for s in steps
    )
    monotone = all(
        s.omega_after <= s.omega_before + 1e-12 for s in steps
    )

    termination, _ = classify_induction_termination(steps, omega_threshold=omega_threshold)
    if termination is InductionTermination.CONVERGED:
        term_ok = steps[-1].omega_after <= omega_threshold
    elif termination is InductionTermination.STALLED:
        scale = max(abs(steps[0].omega_before), 1e-30)
        tail = steps[-min(_INDUCTION_TAIL, n):]
        term_ok = all(abs(s.omega_reduction) <= _INDUCTION_STALL_RTOL * scale for s in tail)
    else:  # DESCENDING：无额外自洽要求
        term_ok = True

    return InductionL2Report(
        g_nonnegative=g_nonneg,
        omega_bounded_below=bounded,
        step_arithmetic_consistent=arithmetic,
        monotone=monotone,
        termination_consistent=term_ok,
        termination=termination,
        limit=float(steps[-1].omega_after),
        n_steps=n,
    )


# ---------------------------------------------------------------------------
# 辅助函数：默认凸集测试集
# ---------------------------------------------------------------------------

def _default_convex_test_sets(
    tubes: Sequence[Tube],
) -> List[Box]:
    """为给定的管族构造默认的凸集测试集。

    在管族的包围盒内构造若干不同大小的 Box。

    **边距必须 ≥ 管半径**：``Box.contains_segment(a, b, delta)`` 要求盒
    包含段的 δ-邻域，即盒子要在端点外缘再让出 δ。旧实现把边距硬编码为
    0.5，当该层管族是半径 ≥ 0.5 的粗管时，连最大的盒子都装不下任何一根管
    ——计数为 0、误差为空、返回 0.0，看着像"满足公理"（2026-09-15 修复）。
    """
    if not tubes:
        return [Box.from_center_size([0, 0, 0], [2, 2, 2])]

    # 计算包围盒与最大管半径
    all_pts = []
    max_delta = 0.0
    for t in tubes:
        a, b = t.endpoints
        all_pts.extend([a, b])
        max_delta = max(max_delta, float(t.delta))
    all_pts = np.array(all_pts)

    margin = max(0.5, 1.05 * max_delta)
    lo = all_pts.min(axis=0) - margin
    hi = all_pts.max(axis=0) + margin
    center = (lo + hi) / 2
    full_size = hi - lo

    test_sets: List[Box] = []
    # 全包围盒
    test_sets.append(Box.from_center_size(center, full_size))
    # 半尺寸盒（3个方向各切半）
    for axis in range(3):
        size = full_size.copy()
        size[axis] *= 0.5
        test_sets.append(Box.from_center_size(center, size))
    # 1/4 尺寸盒
    test_sets.append(Box.from_center_size(center, full_size * 0.25))

    return test_sets


def _default_slab_test_sets(
    tubes: Sequence[Tube],
) -> List:
    """构造默认的 Frostman slab 测试集。

    **厚度必须 ≥ 管直径**（即半厚 ≥ 管半径）：``Slab.contains_segment`` 要求
    ``|n·x| ± δ ≤ half_thickness``，所以半厚小于管半径时一根管都装不下，
    计数为 0、误差为空、返回 0.0——看着像"满足公理"。

    **offset 必须扫描**（2026-09-15 补修）：旧实现把 6 个 slab 全部固定
    ``offset=0``（过原点、法向对齐坐标轴）。当管的中心散布在原点附近之外
    （如随机构型的 ±0.3）时，要求"整根管都落在过原点的薄板内"几乎不可能，
    于是全部 slab 零覆盖——这会造成**假否定**（把可测的构型判成"未测"）。
    现在对每个法向取一组 offset，覆盖管族在该方向上的投影范围。

    Returns
    -------
    slabs : list[Slab]
        每个坐标轴方向若干个 slab，offset 覆盖管族在该方向的跨度。
    """
    from .wolff_axioms import Slab

    if not tubes:
        bb = Box.from_center_size([0, 0, 0], [2, 2, 2])
        max_delta = 0.1
        # **必须初始化 ``all_pts``**：下方 ``if len(all_pts)`` 无条件引用它。
        # 此前它只在 ``else`` 分支里赋值，空输入时到那一行就抛
        # ``UnboundLocalError``（实测确认）。空管族本就没有点，用空数组如实
        # 表达——``len(all_pts) == 0`` 会正确走到 ``offsets = [0.0]``。
        all_pts = np.empty((0, 3))
    else:
        all_pts = []
        max_delta = 0.0
        for t in tubes:
            a, b = t.endpoints
            all_pts.extend([a, b])
            max_delta = max(max_delta, float(t.delta))
        all_pts = np.array(all_pts)
        margin = max(0.5, 1.05 * max_delta)
        lo = all_pts.min(axis=0) - margin
        hi = all_pts.max(axis=0) + margin
        bb = Box(np.column_stack([lo, hi]))

    # 半厚至少取 1.05×管半径（刚好留出 δ-邻域余量）；再用 2× 的一档。
    base_half = 1.05 * max(max_delta, 1e-6)

    slabs = []
    for axis in range(3):
        normal = np.zeros(3)
        normal[axis] = 1.0
        # offset 扫描：覆盖管族在该轴上的投影区间（取中位数与若干分位）
        if len(all_pts):
            projs = all_pts[:, axis]
            offsets = [
                float(np.median(projs)),
                float(np.percentile(projs, 25)),
                float(np.percentile(projs, 75)),
                0.0,
            ]
        else:
            offsets = [0.0]
        for off in offsets:
            for half in (base_half, 2.0 * base_half):
                slabs.append(Slab(normal, off, float(half), bb))

    return slabs
