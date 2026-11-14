"""两尺度 grains 分解所用的 Move #1、#2、#3 骨架。

本模块实现了 Wang–Zahl 第 7.4 节 / 8.1–8.3 节中三个 Move 的**结构**。
完整算法需要 broadness、正则遮蔽、平衡划分覆盖，以及细致的 L^2 论证。
这里我们只在 `GrainsDecomposition` 对象上提供一些看似合理的几何变换，
以便把迭代循环搭起来。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, cast

import numpy as np

from .errors import (
    ContractViolationError,
    InputValidationError,
    NotImplementedYetError,
)
from .grains import GrainsDecomposition, OrientedPrism, Prism
from .primitives import Tube


def _require_axis_aligned(
    grains: GrainsDecomposition, where: str
) -> List[Prism]:
    """三个 Move 的世界轴几何要求 grain 是轴对齐 :class:`Prism`。

    :class:`OrientedPrism` 的 ``bounds`` 只是**保守外扩**的世界 AABB，沿世界轴做
    合并/切分（``_merge_along_long_axis`` 等）会改掉定向几何——那既不是论文的
    Move，也不是可解释的近似，只会**静默给出错误的 grain**。这里改为显式报
    ``NotImplementedYetError``：定向 grain 的 Move 需要一套在**局部正交基**里
    的合并/切分（下一步），在此之前调用方应明确知道该路径不可用。

    **返回收窄后的 ``list[Prism]``**：既是运行时的形状保证（全部是 ``Prism``），
    也把这个保证**交给类型检查器**——调用方拿到的就是 ``list[Prism]``，而不是
    ``list[Prism | OrientedPrism]``，故后续所有只接受 ``Prism`` 的合并/度量函数
    都不再报"联合类型不可分配"。
    """
    if any(isinstance(p, OrientedPrism) for p in grains.prisms):
        raise NotImplementedYetError(
            f"{where} 不支持定向 grain（OrientedPrism）：Move 的世界轴合并/切分路径"
            "会经世界 AABB 改变定向几何。定向 grain 的 Move 需在局部正交基里实现"
            "（尚未接入）。请用 frame='axis' 构造 grains，或先把定向 grain 显式转回"
            "轴对齐（会丢失定向性）。"
        )
    return [p for p in grains.prisms if isinstance(p, Prism)]


@dataclass
class MoveTrace:
    """单次 Move 应用的记录。"""

    move_name: str
    input_count: int
    output_count: int
    input_max_c: float
    output_max_c: float
    input_max_rho: float
    output_max_rho: float
    input_max_local_kt_error: float
    output_max_local_kt_error: float
    iterations: int
    satisfied: bool
    notes: list[str] = field(default_factory=list)
    #: ``satisfied`` 是**对着哪个量**判的：
    #:
    #: - ``"proxy"``：用快速代理量判的。Move #3 的后条件是论文 Lemma 8.3 的
    #:   ``C_KT-CW^loc(G) <= δ^{-ζ}``，而代理是 ``n|T|/|G|``——**两个不同的
    #:   量**。proxy 满足 ≠ 论文后条件成立。
    #: - ``"actual"``：用 ``error_mode="actual"`` 的体素重算真实相交统计判的，
    #:   与论文定义的接近程度显著更高。
    #:
    #: **为什么必须记进 trace**：``satisfied=True`` 一旦写进 ``.kpt`` 就是一条
    #: 可复核的定量事实，而它的含义随调用路径而变（``apply_moves`` 过去不透传
    #: ``error_mode``，于是那条路径**永远**是 proxy）。不记口径的话，读 trace
    #: 的人无法区分"满足了论文后条件"与"满足了代理量"。
    condition_kind: str = "proxy"
    #: 本次 Move **实际执行**了多少轮合并（不是预算上限）。
    #: 用来区分"跑满了 `max_iterations`"与"一轮就没得合并"。
    #:
    #: **为什么必须与预算分开**（缺口二）：`iterations == max_iterations`
    #: 说明"还想合并但被预算截断"（加预算可能有效），
    #: `iterations < max_iterations` 说明"提前停了"（加预算没用）。
    #: 此前这里传的是 `max_iterations`（预算），把两种情况混成一个数。
    #:
    #: **精确值，不是推的**（2026-10-09）：由 `_moveN_impl` 在合并循环里**直接
    #: 计数**得到（每成功缩小一次 grain 数记一轮）。此前是从 before/after 的
    #: grain 数变化推的**下界估计**（0 或 1），区分得开"完全没动"与"动过"，
    #: 但区分不了"动了 1 轮"与"动了 3 轮"——现在可以。
    iters_used: int = 0
    #: 本次应用**是否达成了可观测的定量增益**（缺口二的可判定判据）。
    #:
    #: 判据是**各 Move 自己的目标量**严格变化：
    #: Move #1 看 ``max_c``（应增）、Move #2 看 ``max_rho``（应减）、
    #: Move #3 看 ``max_local_kt_error``（应减）。
    #:
    #: **为什么需要它**：论文要求"单步至少放大 δ^{-ζ}"，而实现是"迭代合并
    #: 直到条件满足"，**无单步放大保证**。实测（内部分批测量脚本，
    #: 16 构型）Move #2 **100% 是空操作**（16/16 逐位不变，因为其前置条件
    #: `ρ > δ^{ω/100}` 几乎从不成立）、Move #1 空操作 13/16。没有这个字段时，
    #: "应用了 Move 但什么都没变"与"应用了 Move 并推进了"在结果里**长得一样**，
    #: 只能靠事后比对指标间接推断（且三个 Move 混在一个聚合指纹里）。
    #:
    #: ``False`` = 本次应用**没有**产生可观测增益，**不等于**"Move 无用"
    #: ——前置条件本就满足时（无需修复）它正确地什么都不做。
    achieved_gain: bool = False
    #: 目标量的**相对增益** ``(after - before) / |before|``（正=朝目标方向）。
    #: 供人读的连续量；``achieved_gain`` 是它的布尔化。``before == 0`` 时为
    #: ``None``（无法定义相对量，不编造）。
    relative_gain: Optional[float] = None

    def summary(self) -> str:
        gain = "n/a" if self.relative_gain is None else f"{self.relative_gain:+.3%}"
        return (
            f"{self.move_name}: count {self.input_count} -> {self.output_count}, "
            f"max_c {self.input_max_c:.4g} -> {self.output_max_c:.4g}, "
            f"max_rho {self.input_max_rho:.4g} -> {self.output_max_rho:.4g}, "
            f"max_local_kt_error {self.input_max_local_kt_error:.4g} -> "
            f"{self.output_max_local_kt_error:.4g}, "
            f"iters={self.iters_used}, satisfied={self.satisfied} "
            f"gain={self.achieved_gain}({gain}) "
            f"[{self.condition_kind}]"
        )


def _grain_target_c(
    p: Prism,
    rho: float,
    delta: float,
    zeta: float,
    fallback_ratio: float | None,
) -> float:
    """Move #1 对单个 prism 的目标 ``c``。

    当 ``#T_ρ/#T`` 取不到（元数据缺失且无全局 fallback）时**抛错**，而不是
    挑一个数蒙混过去。

    理由：该因子直接乘进目标值。随便取一个小值会让目标近乎 0 ⇒ Move #1
    认为"已经满足"⇒ 停止修复（旧实现返回 ``1e-12`` 正是如此）。取一个大值
    则会让 Move 无谓地过度合并。两种"编造"都会让**结果看起来正常**，所以
    这里选择暴露边界：调用方要么补上 ``tube_ratio``、要么补齐 prism 的
    ``bin_tube_count``/``total_tube_count``。

    ``check_move1_condition`` 与之不同——它在缺元数据时返回 ``False``
    （"不得宣称满足"），因为判定函数必须给出一个能用的布尔值；而修复函数
    必须知道目标是多少才能动手，所以只能报错。

    Raises
    ------
    ContractViolationError
        ``#T_ρ/#T`` 无法确定（无 fallback 且 prism 缺 bin 元数据）——
        这是**必需元数据未随数据带来**（前提不满足），不是数值不可达。
        它仍是一个 ``ValueError`` 子类（历史兼容）。
    """
    ratio = _effective_tube_ratio(p, fallback_ratio)
    if ratio is None:
        raise ContractViolationError(
            "无法为 Move #1 计算目标 c：该 prism 取不到 #T_ρ/#T"
            f"（bin_tube_count={p.bin_tube_count}, "
            f"total_tube_count={p.total_tube_count}, fallback={fallback_ratio}）。"
            "请显式传入 tube_ratio=，或让 grains 构造补齐 bin 元数据。"
            "**不要**为此改回哨兵默认值——那会让 Move #1 误判'已满足'。",
            missing=("bin_tube_count", "total_tube_count"),
        )
    return (delta ** zeta) * (rho / delta) * ratio


def _merged_bin_metadata(chunk: Sequence[Prism]) -> dict:
    """合并一组 prism 时，把 ρ-bin 元数据**继承**下来。

    **这是一个真实的既有缺陷的修复**：``_merge_along_long_axis`` 等合并函数
    过去只拷贝几何量（bounds/density/local_kt_error），把
    ``bin_index`` / ``bin_tube_count`` / ``total_tube_count`` 丢了。后果是
    合并后的 grain 取不到 ``#T_ρ/#T``——而 Move #1 的目标 ``c`` 正比于它。
    旧代码于是给哨兵 ``1e-12``，让目标近乎 0 ⇒ "已满足"⇒ 停止修复（假阳性）。

    合并后的语义：

    - ``bin_index``：同一 bin 内合并才忠实；跨 bin 合并（
      ``_merge_short_grains_cross_bin``）时置 ``None``，如实表示"不再属于
      单一 bin"。
    - ``bin_tube_count``：该 chunk 内**同一 bin** 的管数之和。跨 bin 时取
      各 bin 计数之和会让比值失真，所以跨 bin 时同样置 0（配合
      ``bin_index=None``，调用方会走 fallback 或报"取不到"）。
    - ``total_tube_count``：全局量，直接继承。

    管族方向 ``bin_direction`` 亦按同一口径处理。
    """
    bins = {p.bin_index for p in chunk}
    total = max((p.total_tube_count for p in chunk), default=0)

    if len(bins) == 1 and None not in bins:
        bin_index = next(iter(bins))
        bin_tube_count = sum(p.bin_tube_count for p in chunk)
        # 方向取 chunk 中第一个有方向的（同一 bin 内方向一致）。
        direction = next((p.bin_direction for p in chunk
                          if p.bin_direction is not None), None)
        return {
            "bin_index": bin_index,
            "bin_tube_count": bin_tube_count,
            "total_tube_count": total,
            "bin_direction": direction,
        }

    # 跨 bin 或原本就没有 bin 信息：如实置空，**不要**编造一个比值。
    return {
        "bin_index": None,
        "bin_tube_count": 0,
        "total_tube_count": total,
    }


# ---------------------------------------------------------------------------
# 定向 grain（OrientedPrism）的局部正交基合并（Q4 #12，2026-11-13）
#
# 设计：定向 grain 的合并只在**同一个方向簇**内进行（长轴近似的 grain 才有
# 共同的局部标架；跨方向合并在物理上无意义）；簇内以首个 grain 的标架为公共
# 参考系，把每个 grain 转成局部盒（center 投影 ± extents/2），按目标轴排序
# 分块取局部包围盒——合并结果是**保守覆盖**（簇内标架近似带来的轻微外扩），
# 不改变定向性。标架差异超过容差（≈5.7°）的 grain 拒绝合入，保持原样。
# ---------------------------------------------------------------------------

_FRAME_COS_TOL = 0.995  # 长轴与横向轴都需满足（≈5.7°），否则不合并


def _frames_compatible(a: np.ndarray, b: np.ndarray, tol: float = _FRAME_COS_TOL) -> bool:
    """两个正交标架是否近似重合（逐轴 |cos| ≥ tol，符号无关）。"""
    return all(abs(float(np.dot(a[i], b[i]))) >= tol for i in range(3))


def _split_by_orientation(
    grains: GrainsDecomposition, where: str
) -> tuple[list[Prism], list[OrientedPrism]]:
    """把分解按 轴对齐/定向 分家；混合分解显式报错（不静默丢掉一类）。

    返回 ``(axis_prisms, oriented_prisms)``，至多一个非空。
    """
    axis_prisms = [p for p in grains.prisms if isinstance(p, Prism)]
    oriented = [p for p in grains.prisms if isinstance(p, OrientedPrism)]
    if axis_prisms and oriented:
        raise NotImplementedYetError(
            f"{where} 不支持轴对齐与定向 grain 混合的分解：两类合并路径不同"
            "（世界轴 vs 局部正交基）。请统一用一种构造（frame='axis' 或定向）。"
        )
    return axis_prisms, oriented


def _merge_oriented(
    prisms: Sequence[OrientedPrism], factor: int, transverse: bool = False
) -> list[OrientedPrism]:
    """在局部正交基里合并定向 grain。

    ``transverse=False``：沿长轴 c 分块合并（Move #1/#2 用）；
    ``transverse=True``：沿短轴 a 分块合并（Move #3 加宽用）。
    合并盒取局部包围盒（保守覆盖），bin 元数据经 ``_merged_bin_metadata``
    继承（同 bin 合并忠实，跨 bin 如实置空）。
    """
    if len(prisms) <= 1:
        return list(prisms)

    # 方向聚类：只有标架近似重合的 grain 才能进同一合并组。
    clusters: list[list[OrientedPrism]] = []
    for prism in prisms:
        for cluster in clusters:
            if _frames_compatible(cluster[0].axes, prism.axes):
                cluster.append(prism)
                break
        else:
            clusters.append([prism])

    merged: list[OrientedPrism] = []
    axis = 0 if transverse else 2
    for cluster in clusters:
        axes = cluster[0].axes      # 簇内公共参考标架
        ref = cluster[0].center
        boxes = []
        for prism in cluster:
            u = axes @ (prism.center - ref)
            boxes.append((prism, u - prism.extents / 2, u + prism.extents / 2))
        # 先按非目标轴的局部坐标分组，再在组内按目标轴分块——与轴对齐版同构。
        other_axes = [i for i in range(3) if i != axis]
        boxes.sort(key=lambda b: (round(float(b[1][other_axes[0]]), 12),
                                  round(float(b[1][other_axes[1]]), 12),
                                  float(b[1][axis])))
        for i in range(0, len(boxes), factor):
            chunk = boxes[i : i + factor]
            lo = np.min([c[1] for c in chunk], axis=0)
            hi = np.max([c[2] for c in chunk], axis=0)
            ps = [c[0] for c in chunk]
            indices = tuple(sorted({idx for q in ps for idx in q.tube_indices}))
            merged.append(
                OrientedPrism(
                    center=ref + axes.T @ ((lo + hi) / 2),
                    axes=axes,
                    extents=np.maximum(hi - lo, 1e-12),
                    tube_indices=indices,
                    density=float(np.mean([q.density for q in ps])),
                    local_kt_error=max(q.local_kt_error for q in ps),
                    **_merged_bin_metadata(ps),
                )
            )
    return merged


def _merge_along_long_axis_per_bin(prisms: Sequence[Prism], factor: int) -> list[Prism]:
    """沿各棱柱的长轴合并，但绝不跨越 ρ-bin 边界。

    若棱柱没有 bin 元数据，则退化为一次全局合并。
    """
    if len(prisms) <= 1:
        return list(prisms)

    by_bin: dict[int | None, list[Prism]] = {}
    for p in prisms:
        by_bin.setdefault(p.bin_index, []).append(p)

    merged: list[Prism] = []
    for group in by_bin.values():
        merged.extend(_merge_along_long_axis(group, factor))
    return merged


def _merge_short_grains_cross_bin(
    prisms: Sequence[Prism],
    rho: float,
    delta: float,
    zeta: float,
    fallback_ratio: float | None,
    factor: int,
) -> list[Prism]:
    """兜底合并：把方向上相邻的失败 bin 合并起来。

    仍不满足 Move #1 的 grain 按 ρ-bin 分组；中心方向相互夹角在 ``rho``
    以内的 bin 会被合并成 super-bin，然后沿长轴重新合并。
    """
    if len(prisms) <= 1:
        return list(prisms)

    target_base = (delta ** zeta) * (rho / delta)
    failing_bins: set[int] = set()
    bin_directions: dict[int, np.ndarray] = {}
    bin_sizes: dict[int, int] = {}

    for p in prisms:
        if p.bin_index is None:
            return list(prisms)
        target = target_base * _effective_tube_ratio(p, fallback_ratio)
        if p.c < target:
            failing_bins.add(p.bin_index)
        if p.bin_direction is not None:
            bin_directions[p.bin_index] = p.bin_direction
        bin_sizes[p.bin_index] = max(bin_sizes.get(p.bin_index, 0), p.bin_tube_count)

    if not failing_bins:
        return list(prisms)

    # 按各 bin 中心方向的角间隔，为失败的 bin 建立邻接图。
    failing_list = sorted(failing_bins)
    parent = {b: b for b in failing_list}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i, b1 in enumerate(failing_list):
        d1 = bin_directions.get(b1)
        if d1 is None:
            continue
        for b2 in failing_list[i + 1 :]:
            d2 = bin_directions.get(b2)
            if d2 is None:
                continue
            cos = float(np.clip(np.dot(d1, d2), -1.0, 1.0))
            if np.arccos(cos) <= rho:
                union(b1, b2)

    components: dict[int, list[Prism]] = {}
    for p in prisms:
        if p.bin_index in failing_bins:
            components.setdefault(find(p.bin_index), []).append(p)
        else:
            # 未失败的 bin 保持不变。
            components.setdefault(-(p.bin_index or 0) - 1, []).append(p)

    merged: list[Prism] = []
    for root, group in components.items():
        if root < 0:
            merged.extend(group)
            continue
        # 沿长轴重新合并这个 super-bin。
        # 更新元数据以反映合并后的 super-bin。
        super_count = sum(bin_sizes.get(b, 0) for b in failing_list if find(b) == root)
        super_direction = np.mean(
            [bin_directions[b] for b in failing_list if find(b) == root and b in bin_directions],
            axis=0,
        )
        norm = float(np.linalg.norm(super_direction))
        if norm > 1e-12:
            super_direction = super_direction / norm
        else:
            super_direction = np.array([0.0, 0.0, 1.0])

        for p in _merge_along_long_axis(group, factor):
            merged.append(
                Prism(
                    bounds=p.bounds,
                    tube_indices=p.tube_indices,
                    density=p.density,
                    local_kt_error=p.local_kt_error,
                    bin_index=root,
                    bin_tube_count=super_count,
                    total_tube_count=p.total_tube_count,
                    bin_direction=super_direction,
                )
            )
    return merged


def _move1_impl(
    grains: GrainsDecomposition,
    rho: float,
    delta: float,
    zeta: float,
    tube_ratio: float | None = None,
    max_iterations: int = 10,
) -> tuple[GrainsDecomposition, int]:
    """``move1_longer_grains`` 的实现，**额外返回实际执行的合并轮数**。

    轮数 = 循环里**成功缩小 grain 数**的次数（每轮要么按 ρ-bin 内合并、要么
    跨相邻 ρ-bin 合并，必使 ``count`` 严格下降；合不动就 ``break``）。因此
    ``iters > 0`` 与 ``count`` 下降**互为充要**，且 ``iters`` 可达多轮——这正是
    旧实现（从 count 变化推、至多 1）区分不了的。"单步是否推进"现在可精确
    判定（见 ``MoveTrace.iters_used``）。

    公共函数 ``move1_longer_grains`` 是本函数的薄封装（只取第一个返回值），
    **签名与行为逐位不变**——把轮数单独接出来是为了给 :func:`apply_moves` 与
    编排器用，而不破坏既有直接调用者。
    """
    axis_prisms, oriented = _split_by_orientation(grains, "Move #1")
    # 两类 grain 的**类型**互不继承（``Prism`` 是世界轴对齐的，``OrientedPrism``
    # 带局部标架），但本循环对两者完全同构：唯一差别是"调哪个合并函数"。故用一个
    # 联合类型的局部变量承载，并在分支处 ``cast`` 收窄——``_split_by_orientation``
    # 保证两者至多一类非空，实际类型与 ``is_oriented`` 始终一致，只是这条推理
    # 超出类型检查器能自行串起的范围。``cast`` 只存在于类型层面，运行时不转换。
    prisms: list[Prism] | list[OrientedPrism] = axis_prisms or oriented
    is_oriented = bool(oriented)
    if not prisms:
        return grains, 0

    current = GrainsDecomposition(prisms)
    iters = 0
    for _ in range(max_iterations):
        if check_move1_condition(current, rho, delta, zeta, tube_ratio):
            break
        current_max_c = current.max_c
        if current_max_c <= 0:
            break

        # 目标值取决于各 bin 内比例最大的那个 grain。
        target_c = max(
            _grain_target_c(cast("Prism", p), rho, delta, zeta, tube_ratio)
            for p in prisms
        )
        ratio = target_c / current_max_c if current_max_c > 0 else 1.0
        factor = max(2, min(16, int(np.ceil(ratio))))

        # 先在每个 ρ-bin 内部合并。
        merged = (
            _merge_oriented(cast("Sequence[OrientedPrism]", prisms), factor=factor)
            if is_oriented
            else _merge_along_long_axis_per_bin(
                cast("Sequence[Prism]", prisms), factor=factor
            )
        )
        if len(merged) < len(prisms):
            prisms = merged
            current = GrainsDecomposition(prisms)
            iters += 1
            continue

        # 若 bin 内合并停滞，则允许跨相邻 ρ-bin 合并。
        # 定向路径（v1）没有跨 bin 兜底：_merge_short_grains_cross_bin 走
        # 世界 AABB（会改变定向几何），簇聚类已隐含 bin 语义，停滞即收工。
        if is_oriented:
            break
        merged = _merge_short_grains_cross_bin(
            cast("Sequence[Prism]", prisms), rho, delta, zeta, tube_ratio, factor
        )
        if len(merged) < len(prisms):
            prisms = merged
            current = GrainsDecomposition(prisms)
            iters += 1
            continue

        break
    return current, iters


def move1_longer_grains(
    grains: GrainsDecomposition,
    rho: float,
    delta: float,
    zeta: float,
    tube_ratio: float | None = None,
    max_iterations: int = 10,
) -> GrainsDecomposition:
    """Move #1: merge grains along their long axis until
    c >= δ^ζ (ρ/δ) (#T_ρ / #T).

    The target ratio #T_ρ/#T is evaluated per grain using the grain's ρ-bin
    metadata when available.  Iteratively merges neighbouring grains along each
    grain's longest dimension.  The merge factor is adapted based on the gap
    between the current max_c and the target.  The post-condition is checked by
    ``check_move1_condition``.

    实际执行的合并轮数由 :func:`_move1_impl` 一并算出；本函数只返回 grains
    （向后兼容的直接调用者）。需要轮数（如写 ``MoveTrace.iters_used``）请走
    ``_move1_impl`` 或 :func:`apply_moves`。
    """
    return _move1_impl(grains, rho, delta, zeta, tube_ratio, max_iterations)[0]


def _move2_impl(
    grains: GrainsDecomposition,
    omega: float,
    delta: float,
    max_iterations: int = 10,
) -> tuple[GrainsDecomposition, int]:
    """``move2_replace_square_grains`` 的实现，额外返回合并轮数（同 #1 口径）。"""
    axis_prisms2, oriented2 = _split_by_orientation(grains, "Move #2")
    # 联合类型 + 分支收窄的理由同 ``_move1_impl``。
    prisms: list[Prism] | list[OrientedPrism] = axis_prisms2 or oriented2
    is_oriented2 = bool(oriented2)
    if not prisms:
        return grains, 0

    threshold = move2_rho_threshold(omega, delta)
    current = GrainsDecomposition(prisms)
    iters = 0
    for _ in range(max_iterations):
        if check_move2_condition(current, omega, delta):
            break
        current_max_rho = current.max_rho
        if current_max_rho <= 0:
            break
        # 依据 rho 差距调整合并因子。
        ratio = current_max_rho / threshold if threshold > 0 else 1.0
        factor = max(2, min(16, int(np.ceil(ratio))))
        merged = (
            _merge_oriented(cast("Sequence[OrientedPrism]", prisms), factor=factor)
            if is_oriented2
            else _merge_along_long_axis(cast("Sequence[Prism]", prisms), factor=factor)
        )
        if len(merged) == len(prisms):
            break
        prisms = merged
        current = GrainsDecomposition(prisms)
        iters += 1
    return current, iters


def move2_replace_square_grains(
    grains: GrainsDecomposition,
    omega: float,
    delta: float,
    max_iterations: int = 10,
) -> GrainsDecomposition:
    """Move #2: if ρ > δ^{ω/100}, merge grains along long axis until
    ρ <= δ^{ω/100}.

    This makes grains longer (larger c) and therefore reduces ρ = b/c.

    实测 Move #2 **100% 是空操作**（16/16 构型逐位不变，因为前置条件
    ``ρ > δ^{ω/100}`` 几乎从不成立）——那时 ``iters_used == 0``。轮数由
    :func:`_move2_impl` 算出；本函数只返回 grains（向后兼容）。
    """
    return _move2_impl(grains, omega, delta, max_iterations)[0]


def move3_target_rho(rho: float, delta: float, zeta: float) -> float:
    """Move #3 之后 ``rho = b/c`` 的目标下界。

    在 Wang–Zahl 引理 8.3 中，加宽步骤把横向宽度 ``b`` 增大到足以保证
    ``tilde{rho} >= delta^{-zeta/1000} rho``。
    """
    if delta <= 0:
        return rho
    return float(rho * (delta ** (-zeta / 1000.0)))


def tube_segment_in_box(tube: Tube, box_bounds: np.ndarray) -> Optional[Tube]:
    """把一条无限长的 δ-管裁剪到轴对齐的盒上。

    该管被建模为过 ``tube.center``、沿 ``tube.direction`` 的直线的 δ-邻域；
    本函数把这条直线与盒作相交（对六个边界面使用平板法），再进一步与管自身
    的线段参数范围 ``[-length/2, length/2]`` 求交。

    Parameters
    ----------
    tube : Tube
    box_bounds : np.ndarray, shape (3, 2)
        轴对齐的边界 ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``。

    Returns
    -------
    Tube or None
        一根新管，方向与 delta 不变，中心落在裁剪后的线段上、长度取裁剪后
        的长度。 当直线与盒不相交、交集为空，或裁剪后长度小于
        ``tube.delta`` 时，返回 None。
    """
    bounds = np.asarray(box_bounds, dtype=float).reshape(3, 2)
    lo, hi = bounds[:, 0], bounds[:, 1]
    center = np.asarray(tube.center, dtype=float)
    d = np.asarray(tube.direction.v, dtype=float)

    t_lo, t_hi = -0.5 * tube.length, 0.5 * tube.length
    for i in range(3):
        if abs(d[i]) < 1e-15:
            # 直线与该平板平行：若整体落在盒外侧则不相交。
            if center[i] < lo[i] or center[i] > hi[i]:
                return None
            continue
        t1 = (lo[i] - center[i]) / d[i]
        t2 = (hi[i] - center[i]) / d[i]
        enter, exit_ = (t1, t2) if t1 < t2 else (t2, t1)
        t_lo = max(t_lo, enter)
        t_hi = min(t_hi, exit_)
        if t_lo > t_hi:
            return None

    seg_length = t_hi - t_lo
    if seg_length < tube.delta:
        return None
    new_center = center + 0.5 * (t_lo + t_hi) * d
    return Tube(
        center=new_center,
        direction=tube.direction,
        length=seg_length,
        delta=tube.delta,
    )


def actual_local_kt_errors(
    tubes: Sequence[Tube],
    prisms: Sequence[Prism],
    grid_n: int = 32,
    backend: str = "numpy",
) -> List[float]:
    """由真实管-相交统计得到的逐 grain 局部 KT 误差。

    对每颗棱柱，把它各自的后备管与 grain 盒作裁剪（用
    ``tube_segment_in_box``）；未命中盒的管被丢弃。 误差为
    ``n_kept * tube_volume / union_volume``，其中 ``union_volume`` 是裁剪后
    各管之并集的体素估计体积，``tube_volume`` 是裁剪后管的平均体积
    （故 ``n_kept * tube_volume`` 为保留管的总体积）。

    Parameters
    ----------
    tubes : sequence of Tube
        完整的管集合；棱柱用下标索引到其中。
    prisms : sequence of Prism
    grid_n : int
        用于并集体积估计的体素网格分辨率。
    backend : str
        ``accel.volume_voxel`` 所用的后端（默认 ``"numpy"``）。

    Returns
    -------
    errors : list[float]
        每颗棱柱一个误差；当没有管在裁剪后存活，或估计的并集体积约为 0
        时为 0.0。
    """
    from . import accel

    tube_list = list(tubes)
    errors: List[float] = []
    for prism in prisms:
        kept = [
            clipped
            for idx in prism.tube_indices
            if 0 <= idx < len(tube_list)
            for clipped in [tube_segment_in_box(tube_list[idx], prism.bounds)]
            if clipped is not None
        ]
        n_kept = len(kept)
        if n_kept == 0:
            errors.append(0.0)
            continue
        union_volume = accel.volume_voxel(kept, grid_n=grid_n, backend=backend)
        if union_volume <= 1e-24:
            errors.append(0.0)
            continue
        tube_volume = float(np.mean([t.volume for t in kept]))
        errors.append(float(n_kept * tube_volume / union_volume))
    return errors


def regularize_shading(
    grains: GrainsDecomposition, min_density: float = 0.5
) -> GrainsDecomposition:
    """确保每颗 grain 都带有至少 ``min_density`` 的遮蔽密度。

    这是 Wang–Zahl 第 7.4 节中正则遮蔽构造的一个计算代理：加宽一颗 grain
    之后，每根管落在 grain 内的那部分子集仍应占据该管的一个均匀比例。
    抬高密度下限可以防止退化的 grain 带有零有效遮蔽。
    """
    if grains.count == 0:
        return grains

    new_prisms: list[Prism] = []
    for p in grains.prisms:
        new_density = float(np.clip(max(min_density, p.density), 0.0, 1.0))
        new_prisms.append(
            Prism(
                bounds=p.bounds,
                tube_indices=p.tube_indices,
                density=new_density,
                local_kt_error=p.local_kt_error,
                bin_index=p.bin_index,
                bin_tube_count=p.bin_tube_count,
                total_tube_count=p.total_tube_count,
                bin_direction=p.bin_direction,
            )
        )
    return GrainsDecomposition(new_prisms)


def _move3_impl(
    grains: GrainsDecomposition,
    zeta: float,
    delta: float | None = None,
    max_local_error: float | None = None,
    max_iterations: int = 10,
    regularize: bool = True,
    tubes: Optional[Sequence[Tube]] = None,
    error_mode: str = "proxy",
    grid_n: int = 32,
    backend: str = "numpy",
) -> tuple[GrainsDecomposition, int]:
    """``move3_wider_grains`` 的实现，额外返回加宽轮数（同 #1 口径）。

    轮数 = 循环里**成功缩小 grain 数**的横向合并次数（``_merge_transverse``
    使 ``count`` 严格下降才算一轮；合不动就 ``break``）。
    """
    axis_prisms3, oriented3 = _split_by_orientation(grains, "Move #3")
    # 联合类型 + 分支收窄的理由同 ``_move1_impl``。
    prisms: list[Prism] | list[OrientedPrism] = axis_prisms3 or oriented3
    is_oriented3 = bool(oriented3)
    if is_oriented3 and error_mode == "actual":
        # 定向路径 v1：actual 误差重建依赖 Prism(bounds=...) 轴对齐构造。
        raise NotImplementedYetError(
            "Move #3 的 error_mode='actual' 尚不支持定向 grain（重建依赖轴对齐"
            " Prism 构造）；请用默认 proxy 模式。"
        )
    if error_mode not in ("proxy", "actual"):
        raise InputValidationError(
            f"error_mode must be 'proxy' or 'actual', got {error_mode!r}."
        )
    if error_mode == "actual" and tubes is None:
        raise ContractViolationError(
            "error_mode='actual' 需要被检验的管族 `tubes`，未传入。",
            missing=("tubes",),
        )
    if grains.count == 0:
        return grains, 0

    if max_local_error is None:
        if delta is None:
            raise ContractViolationError(
                "必须给出 `delta` 或 `max_local_error` 之一才能确定目标误差。",
                missing=("delta", "max_local_error"),
            )
        max_local_error = delta ** (-zeta)

    # 把守卫后的取值收窄到**非 Optional** 的局部变量，供循环内使用。
    #
    # 必要的理由：类型检查器无法把上面的赋值当作有效收窄（``delta`` 本身是
    # Optional，``delta ** (-zeta)`` 的推断结果仍是 ``float | None``）。
    # 于是循环里的比较与除法全部报"None 不支持运算符"——而更值得警惕的是，
    # 这对读者也是个信号：那几行的安全性依赖于几十行之前的两个守卫。
    # 显式断言把这份依赖变成眼前可见的事实。
    assert max_local_error is not None  # 上面的分支已保证
    error_ceiling: float = float(max_local_error)
    require_tubes = error_mode == "actual"
    assert not require_tubes or tubes is not None  # 上面的守卫已保证

    current = GrainsDecomposition(prisms)
    iters = 0
    for _ in range(max_iterations):
        if check_move3_condition(current, error_ceiling):
            break
        current_error = current.max_local_kt_error
        if current_error <= 0 or error_ceiling <= 0:
            break
        # 依据误差差距调整合并因子。
        ratio = current_error / error_ceiling
        factor = max(2, min(16, int(np.ceil(np.log1p(ratio)))))
        merged = (
            _merge_oriented(
                cast("Sequence[OrientedPrism]", prisms), factor=factor, transverse=True
            )
            if is_oriented3
            else _merge_transverse(cast("Sequence[Prism]", prisms), factor=factor)
        )
        if len(merged) == len(prisms):
            break
        prisms = merged
        current = GrainsDecomposition(prisms)
        iters += 1
        if require_tubes:
            # ``require_tubes`` 蕴含 ``tubes is not None``（见上面的 assert）。
            # mypy/Pylance 无法自己串起这条推理，故在此显式取用。
            assert tubes is not None
            errors = actual_local_kt_errors(
                tubes, cast("Sequence[Prism]", prisms), grid_n=grid_n, backend=backend
            )
            prisms = [
                Prism(
                    bounds=p.bounds,
                    tube_indices=p.tube_indices,
                    density=p.density,
                    local_kt_error=err,
                    bin_index=p.bin_index,
                    bin_tube_count=p.bin_tube_count,
                    total_tube_count=p.total_tube_count,
                    bin_direction=p.bin_direction,
                )
                for p, err in zip(prisms, errors, strict=True)
            ]
            current = GrainsDecomposition(prisms)
        if regularize:
            current = regularize_shading(current)
    return current, iters


def move3_wider_grains(
    grains: GrainsDecomposition,
    zeta: float,
    delta: float | None = None,
    max_local_error: float | None = None,
    max_iterations: int = 10,
    regularize: bool = True,
    tubes: Optional[Sequence[Tube]] = None,
    error_mode: str = "proxy",
    grid_n: int = 32,
    backend: str = "numpy",
) -> GrainsDecomposition:
    """Move #3: merge grains transversely to reduce local convex Wolff error.

    By widening a grain in the transverse directions (increasing ``a`` and
    ``b``), the grain volume grows faster than the number of tubes it contains,
    so the proxy local Katz–Tao error ``n|T|/|G|`` tends to decrease.  Merging
    stops once ``max_local_error`` is satisfied or no improvement is possible.

    Parameters
    ----------
    grains : GrainsDecomposition
        Input grains, typically after Moves #1 and #2.
    zeta : float
        Precision parameter used to set the default ``max_local_error``.
    delta : float or None
        Scale parameter.  If ``max_local_error`` is not given, it is set to
        ``delta**(-zeta)``.
    max_local_error : float or None
        Explicit error ceiling.  Overrides the ``delta``-based default.
    max_iterations : int
        Maximum merge iterations.
    regularize : bool
        If True (default), apply ``regularize_shading`` after each widening
        iteration so that no grain collapses to zero effective shading.
    tubes : sequence of Tube or None
        The full tube collection backing the grains.  Required when
        ``error_mode="actual"`` (ignored otherwise).
    error_mode : str
        ``"proxy"`` (default) keeps the volume-weighted proxy error and is the
        exact legacy behaviour.  ``"actual"`` recomputes each grain's local KT
        error from the actual intersection statistics of its tubes inside the
        grain box (see ``actual_local_kt_errors``) after every merge.  Any
        other value raises ``ValueError``.
    grid_n, backend
        Passed to ``actual_local_kt_errors`` when ``error_mode="actual"``.

    Returns
    -------
    GrainsDecomposition
        Widened grains with preserved ρ-bin metadata and recomputed local KT
        error.

    实际执行的加宽轮数由 :func:`_move3_impl` 一并算出；本函数只返回 grains
    （向后兼容的直接调用者）。
    """
    return _move3_impl(
        grains, zeta, delta=delta, max_local_error=max_local_error,
        max_iterations=max_iterations, regularize=regularize,
        tubes=tubes, error_mode=error_mode, grid_n=grid_n, backend=backend,
    )[0]


def _long_axis(p: Prism) -> int:
    """返回棱柱最长维度的下标。"""
    return int(np.argmax(p.dimensions))


def _transverse_axes(p: Prism) -> tuple[int, int]:
    """返回棱柱两个较短维度的下标。

    Move #3 会沿这两个横向方向加宽 grain。 返回的顺序是确定的：较短维度
    在前，较长维度在后。
    """
    axes = [0, 1, 2]
    axes.remove(_long_axis(p))
    if p.dimensions[axes[1]] < p.dimensions[axes[0]]:
        axes[0], axes[1] = axes[1], axes[0]
    # 显式返回两元素元组而不是 ``tuple(axes)``：后者类型是
    # ``tuple[int, ...]``，与声明的 ``tuple[int, int]`` 不兼容（长度无法被
    # 类型系统证明）。这里直接写出来，既满足类型，也更清楚地表达"恰有两个"。
    return (axes[0], axes[1])


def _recompute_local_kt_error(chunk: Sequence[Prism]) -> float:
    """合并 ``chunk`` 之后按体积加权的局部 KT 误差。

    对不相交的管集合，它等于 ``n |T| / |merged_grain|``；对有重叠的集合，
    它是一个保守的代理量，当合并后体积比管的并集增长得更快时，它通常随之
    下降。
    """
    if not chunk:
        return 0.0
    total_volume = sum(p.volume for p in chunk)
    if total_volume <= 1e-24:
        return max(p.local_kt_error for p in chunk)
    weighted = sum(p.local_kt_error * p.volume for p in chunk) / total_volume
    return float(weighted)


def _merge_metadata(chunk: Sequence[Prism]) -> dict:
    """从一段棱柱中提取一致的 ρ-bin 元数据。

    Move #3 在固定的 ρ-bin 内执行，故元数据应当一致。若不一致（防御性
    处理），则把 ``bin_index`` 置为 None，并对方向取平均。
    """
    bin_indices = {p.bin_index for p in chunk if p.bin_index is not None}
    bin_directions = [p.bin_direction for p in chunk if p.bin_direction is not None]
    total_counts = {p.total_tube_count for p in chunk if p.total_tube_count > 0}

    mean_dir: np.ndarray | None = None
    if bin_directions:
        mean_dir = np.mean(bin_directions, axis=0)
        norm = float(np.linalg.norm(mean_dir))
        if norm > 1e-12:
            mean_dir = mean_dir / norm
        else:
            mean_dir = None

    return {
        "bin_index": next(iter(bin_indices)) if len(bin_indices) == 1 else None,
        "bin_tube_count": max((p.bin_tube_count for p in chunk), default=0),
        "total_tube_count": next(iter(total_counts)) if len(total_counts) == 1 else 0,
        "bin_direction": mean_dir,
    }


def _merge_along_long_axis(prisms: Sequence[Prism], factor: int) -> list[Prism]:
    """沿每个棱柱的最长维度合并相邻的棱柱。

    棱柱先按其长轴下标以及两个横向轴的坐标分组。在每个分组内，沿长轴方向
    相邻的棱柱按 ``factor`` 大小分块合并。
    """
    if len(prisms) <= 1:
        return list(prisms)

    merged: list[Prism] = []

    # 按长轴下标分组。
    by_axis: dict[int, list[Prism]] = {}
    for p in prisms:
        by_axis.setdefault(_long_axis(p), []).append(p)

    for axis, group in by_axis.items():
        # 先按横向坐标排序，再按长轴坐标排序。
        trans = [i for i in range(3) if i != axis]
        group.sort(key=lambda p: (round(p.bounds[trans[0], 0], 12),
                                   round(p.bounds[trans[1], 0], 12),
                                   p.bounds[axis, 0]))

        sub_group: list[Prism] = []
        current_key = None

        # ``axis`` / ``trans`` 通过**默认参数**绑定当前迭代值。
        #
        # 直接闭包会捕获变量本身（延迟绑定）：本函数恰好在循环内被立即调用，
        # 所以当下行为是对的；但一旦有人把 ``flush`` 收集起来稍后调用（很自然
        # 的重构），就会用到**最后一次迭代**的取值，产生静默错误结果。
        # 默认参数在定义时求值，天然固化这一步的取值，不依赖调用时机。
        #
        # 只绑定本函数**真正引用**的变量：``axis``。``trans`` 只在下面的分组
        # 键里用（循环体直接引用，不经闭包），不必也不应列入参数表。
        def flush(
            sub_group: list[Prism],
            axis: int = axis,
        ) -> None:
            if not sub_group:
                return
            sub_group.sort(key=lambda p: p.bounds[axis, 0])
            for i in range(0, len(sub_group), factor):
                chunk = sub_group[i : i + factor]
                if not chunk:
                    continue
                lo = np.min([p.bounds[:, 0] for p in chunk], axis=0)
                hi = np.max([p.bounds[:, 1] for p in chunk], axis=0)
                indices = tuple(sorted({idx for p in chunk for idx in p.tube_indices}))
                density = float(np.mean([p.density for p in chunk])) if chunk else 1.0
                local_kt_error = max(p.local_kt_error for p in chunk)
                merged.append(
                    Prism(
                        bounds=np.column_stack([lo, hi]),
                        tube_indices=indices,
                        density=density,
                        local_kt_error=local_kt_error,
                        **_merged_bin_metadata(chunk),
                    )
                )

        for p in group:
            key = (round(p.bounds[trans[0], 0], 12), round(p.bounds[trans[1], 0], 12))
            if key != current_key:
                flush(sub_group)
                sub_group = []
                current_key = key
            sub_group.append(p)
        flush(sub_group)

    return merged


def _merge_transverse(prisms: Sequence[Prism], factor: int) -> list[Prism]:
    """在垂直于每个棱柱长轴的平面内合并相邻的棱柱。

    棱柱先按其长轴下标与长轴坐标分组，然后在两个横向方向上合并。
    """
    if len(prisms) <= 1:
        return list(prisms)

    merged: list[Prism] = []

    # 按长轴下标分组。
    by_axis: dict[int, list[Prism]] = {}
    for p in prisms:
        by_axis.setdefault(_long_axis(p), []).append(p)

    for axis, group in by_axis.items():
        trans = [i for i in range(3) if i != axis]
        # 先按长轴坐标排序，再按横向坐标排序。
        group.sort(key=lambda p: (round(p.bounds[axis, 0], 12),
                                   p.bounds[trans[0], 0],
                                   p.bounds[trans[1], 0]))

        sub_group: list[Prism] = []
        current_key = None

        # 同 ``_merge_along_long_axis``：用默认参数固化 ``trans``（本函数引用了
        # 它），避免闭包延迟绑定。这里**不需要** ``axis``——它只在上面排序时
        # 用到，循环体直接引用，不经闭包。
        def flush(
            sub_group: list[Prism],
            trans: list[int] = trans,
        ) -> None:
            if not sub_group:
                return
            # 按横向坐标排序，并按 factor x factor 的分块合并。
            sub_group.sort(key=lambda p: (p.bounds[trans[0], 0], p.bounds[trans[1], 0]))
            for i in range(0, len(sub_group), factor * factor):
                chunk = sub_group[i : i + factor * factor]
                if not chunk:
                    continue
                lo = np.min([p.bounds[:, 0] for p in chunk], axis=0)
                hi = np.max([p.bounds[:, 1] for p in chunk], axis=0)
                indices = tuple(sorted({idx for p in chunk for idx in p.tube_indices}))
                density = float(np.mean([p.density for p in chunk])) if chunk else 1.0
                local_kt_error = _recompute_local_kt_error(chunk)
                meta = _merge_metadata(chunk)
                merged.append(
                    Prism(
                        bounds=np.column_stack([lo, hi]),
                        tube_indices=indices,
                        density=density,
                        local_kt_error=local_kt_error,
                        bin_index=meta["bin_index"],
                        bin_tube_count=meta["bin_tube_count"],
                        total_tube_count=meta["total_tube_count"],
                        bin_direction=meta["bin_direction"],
                    )
                )

        for p in group:
            key = round(p.bounds[axis, 0], 12)
            if key != current_key:
                flush(sub_group)
                sub_group = []
                current_key = key
            sub_group.append(p)
        flush(sub_group)

    return merged


def _effective_tube_ratio(
    p: Prism | OrientedPrism, fallback: float | None
) -> float | None:
    """返回该 grain 的 ``#T_ρ/#T``；**无法确定时返回 ``None``**。

    ``None`` 的语义是"这个量取不到"，而**不是**"取一个便于通过检查的值"。
    之前这里返回 ``1e-12``（注释写 "use a tiny conservative value so the
    condition is easy"），后果是把 Move #1 的目标压到近乎 0——于是 ``c`` 小到
    荒唐（实测 ``c=1e-9``）的 grain 也会被判"满足 Item (iv)"。**那是假阳性**，
    而且方向上最危险：它让"条件已满足"这种**可写进 trace 的定量结论**凭空
    成立。

    真正保守的取向是"不确定就说不知道"：调用方据此报"无法判定"，而不是把
    未知当作通过（也不当作失败——那是另一种编造）。

    参数接受 ``Prism | OrientedPrism``：本函数只读 ``bin_tube_count`` /
    ``total_tube_count`` 两项元数据，**两种 grain 都有**，与几何朝向无关；故它
    也能在未收窄的 grain 集合上工作（``_check_move1_condition_impl`` 正是如此）。
    """
    if fallback is not None:
        return fallback
    if p.total_tube_count > 0 and p.bin_tube_count > 0:
        return p.bin_tube_count / p.total_tube_count
    # 元数据缺失（例如经过不含 bin_index 的合并）。**不给哨兵值。**
    return None


def check_move1_condition(
    grains: GrainsDecomposition,
    rho: float,
    delta: float,
    zeta: float,
    tube_ratio: float | None = None,
) -> bool:
    """Check whether Item (iv) of Proposition 7.5 holds: c >= δ^ζ (ρ/δ)(#Tρ/#T).

    When ``tube_ratio`` is None, the ratio #T_ρ/#T is read from each prism's
    bin metadata (set by ``guth_grains_skeleton``).  This is the correct
    behaviour for directionally dispersed configurations where a single global
    ratio would be misleading.

    **元数据缺失时判为不满足（``False``），并记录原因。** 这不是"保守地选一边"
    的问题：``#T_ρ/#T`` 是论文 Item (iv) 里的**因子**，拿不到它就算不出目标，
    于是"是否满足"**没有定义**。返回 ``False`` 的含义是"不能宣称满足"——
    宁可让 Move #1 多跑一轮（代价是算力），也不要凭空产生一条"已满足"的
    定量事实（代价是结论不可信）。取不到原因可用
    ``check_move1_condition_detailed()`` 查看。
    """
    if grains.count == 0:
        return True
    ok, _missing = _check_move1_condition_impl(grains, rho, delta, zeta, tube_ratio)
    return ok


def check_move1_condition_detailed(
    grains: GrainsDecomposition,
    rho: float,
    delta: float,
    zeta: float,
    tube_ratio: float | None = None,
) -> tuple[bool, int]:
    """与 ``check_move1_condition`` 相同，但额外返回"缺元数据的 grain 数"。

    调用方靠这个计数区分两种 ``False``：

    - ``missing == 0``：真的算过，结论是"不满足"——这是**判定**。
    - ``missing > 0``：有 grain 取不到 ``#T_ρ/#T``，结论是"**无法判定**"。

    二者处置不同（前者该走 Move 修复，后者该去补元数据），所以必须能分开。
    """
    if grains.count == 0:
        return True, 0
    return _check_move1_condition_impl(grains, rho, delta, zeta, tube_ratio)


def _check_move1_condition_impl(
    grains: GrainsDecomposition,
    rho: float,
    delta: float,
    zeta: float,
    tube_ratio: float | None,
) -> tuple[bool, int]:
    """``check_move1_condition`` 的实现：返回 (是否可宣称满足, 缺元数据 grain 数)。"""
    target_base = (delta ** zeta) * (rho / delta)
    missing = 0
    for p in grains.prisms:
        ratio = _effective_tube_ratio(p, tube_ratio)
        if ratio is None:
            # 取不到因子 ⇒ 无法判定 ⇒ 不得宣称满足。
            missing += 1
            continue
        if p.c < target_base * ratio:
            return False, missing
    if missing:
        return False, missing
    return True, 0


def move2_rho_threshold(omega: float, delta: float) -> float:
    """Move #2 的上限阈值 ``δ^{ω/100}``（ρ 应不超过它）。

    **单一实现**（此前 ``move2_replace_square_grains`` 与 ``check_move2_condition``
    各写一份 ``delta ** (omega / 100)``）——两处重复的公式一旦分别被改，
    "判据"与"修复"就会用**不同**的阈值（本项目已多次踩"两处各写一份"）。

    ``delta <= 0`` 时返回 ``1.0``（阈值取"任何 ρ≤1 都满足"，即不触发修复）。

    **默认参数的后果（局限 #14，2026-10-01 定位）**：``omega=0.1, delta=0.05``
    ⇒ 阈值 ``≈0.997``，而 ``ρ = b/c ≤ 1`` 几乎总是小于它 ⇒ 条件**结构性地
    近乎恒满足**，Move #2 因而**几乎不做事**。这不是 bug（论文条件 #2 的
    上界在 ρ≤1 下就平凡成立），但要意识到"在这些构型上 Move #2 给不出样本"。
    """
    return delta ** (omega / 100) if delta > 0 else 1.0


def check_move2_condition(grains: GrainsDecomposition, omega: float, delta: float) -> bool:
    """检查是否满足 ρ <= δ^{ω/100}。"""
    if grains.count == 0:
        return True
    threshold = move2_rho_threshold(omega, delta)
    return all(p.rho <= threshold for p in grains.prisms)


def check_move3_condition(grains: GrainsDecomposition, max_local_error: float) -> bool:
    """检查每颗 grain 上的代理局部 Katz–Tao 凸 Wolff 误差是否不超过
    ``max_local_error``。
    """
    if grains.count == 0:
        return True
    return grains.max_local_kt_error <= max_local_error


def apply_moves(
    grains: GrainsDecomposition,
    rho: float,
    delta: float,
    zeta: float,
    omega: float,
    tube_ratio: float | None = None,
    max_iterations: int = 10,
    tubes: Sequence[Tube] | None = None,
    error_mode: str = "proxy",
) -> tuple[GrainsDecomposition, list[MoveTrace]]:
    """Apply Moves #1, #2, #3 in order and return the result plus traces.

    The loop follows the Wang–Zahl Proposition 7.5 / Lemma 8.1–8.3 pattern:
    each Move either fixes a quantitative failure or leaves the decomposition
    unchanged.  The returned ``MoveTrace`` objects record before/after values
    so the iteration can be numerically verified.

    Parameters
    ----------
    grains : GrainsDecomposition
        Initial grains (typically the output of ``guth_grains_skeleton``).
    rho, delta, zeta, omega, tube_ratio : float
        Parameters from Proposition 7.5.
    max_iterations : int
        Maximum iterations for each individual Move.
    tubes : sequence of Tube, optional
        **Move #3 用 ``error_mode="actual"`` 时的必填项**：真实相交统计要在
        原始管族上算。过去本函数不接收 ``tubes``，于是**无论调用方怎么想，
        Move #3 都只能走 proxy**——后条件检查的是一个与论文定义不同的量。
    error_mode : str
        ``"proxy"``（``n|T|/|G|``）或 ``"actual"``（体素重算真实相交）。
        **口径会写进 ``MoveTrace.condition_kind``**，使 trace 自证
        ``satisfied`` 是对着哪个量判的。

        **本函数（纯函数）的默认仍是 ``"proxy"``**，为向后兼容既有的直接
        调用者。但**编排器 ``ProofOrchestrator`` 的默认已切到 ``"actual"``**
        （2026-10-01）：实测 proxy **不是保守上界，而是系统性偏悲观**——8/8
        构型下 proxy 报"不满足"而 actual 报"满足"（见设计记录的"缺口一"与内部分批测量）。要拿到与论文后条件可比的结论请显式传 ``"actual"``
        （并给 ``tubes``）。

    Raises
    ------
    NotImplementedYetError
        输入含定向 grain（``OrientedPrism``）——三个 Move 的合并/切分走在世界轴
        上，不适用于定向几何（见 :func:`_require_axis_aligned`）。

    Returns
    -------
    grains : GrainsDecomposition
        Grains after all three Moves.
    traces : list[MoveTrace]
        One trace per Move, recording quantitative changes.

    Raises
    ------
    ValueError
        ``error_mode="actual"`` 但未提供 ``tubes``——那会静默退回 proxy，
        等于口径说谎。
    """
    if error_mode == "actual" and not tubes:
        raise ContractViolationError(
            "error_mode='actual' 需要 tubes（真实相交统计要在原始管族上算）；"
            "未提供会静默退回 proxy，使 MoveTrace.condition_kind 说谎。",
            missing=("tubes",),
        )

    traces: list[MoveTrace] = []
    current = grains
    # 2026-11-13（Q4 #12）：定向支持已进三个 Move 的局部正交基路径，
    # 总闸撤掉——分派在各 Move 内部（_split_by_orientation）。混合分解
    # 与 error_mode='actual'+定向 仍显式报错（见各 Move 实现）。

    # Move #1：不断加长 grain，直到条目 (iv) 成立。
    m1, iters1 = _move1_impl(current, rho, delta, zeta, tube_ratio, max_iterations)
    traces.append(
        _make_trace("Move #1 (longer)", current, m1, max_iterations,
                    check_move1_condition(m1, rho, delta, zeta, tube_ratio),
                    condition_kind="exact", target="max_c", iters_used=iters1)
    )
    current = m1

    # Move #2：若 grain 太方，则减小 rho = b/c。
    m2, iters2 = _move2_impl(current, omega, delta, max_iterations)
    traces.append(
        _make_trace("Move #2 (lengthen)", current, m2, max_iterations,
                    check_move2_condition(m2, omega, delta),
                    condition_kind="exact", target="max_rho", iters_used=iters2)
    )
    current = m2

    # Move #3：横向加宽 grain 以压低局部 KT 误差。
    #
    # 口径必须**透传**：过去这里既不收 tubes 也不收 error_mode，于是这条路径
    # 永远走 proxy。而 Move #3 的后条件（Lemma 8.3 的 C_KT-CW^loc(G) <= δ^{-ζ}）
    # 与 proxy（n|T|/|G|）是不同的量，satisfied 的语义因此随调用路径漂移。
    m3, iters3 = _move3_impl(
        current, zeta, delta=delta, max_iterations=max_iterations,
        tubes=tubes, error_mode=error_mode,
    )
    traces.append(
        _make_trace("Move #3 (widen)", current, m3, max_iterations,
                    check_move3_condition(m3, delta ** (-zeta)),
                    condition_kind=error_mode, target="max_local_kt_error",
                    iters_used=iters3)
    )
    current = m3

    return current, traces


def _make_trace(
    name: str,
    before: GrainsDecomposition,
    after: GrainsDecomposition,
    iterations: int,
    satisfied: bool,
    condition_kind: str = "proxy",
    target: str = "max_local_kt_error",
    iters_used: int = 0,
) -> MoveTrace:
    """Helper to build a MoveTrace from before/after grain decompositions.

    ``condition_kind`` 记录 ``satisfied`` 是对着哪个量判的（``"exact"`` /
    ``"proxy"`` / ``"actual"``），使写进 trace 的 ``satisfied=True`` 能自证
    口径——否则"满足代理量"与"满足论文后条件"在文件里长得一模一样。

    ``target`` 是本次 Move 的**目标量**（见 :func:`_move_target_value`）：
    ``"max_c"``（Move #1，应增）/ ``"max_rho"``（Move #2，应减）/
    ``"max_local_kt_error"``（Move #3，应减）。据此算出 ``achieved_gain`` 与
    ``relative_gain``（缺口二的可判定判据）。

    ``iterations`` 参数**是调用方的预算**（`max_iterations`）——本函数**不**用它
    填 ``iters_used``（那会把"预算"当成"实际用量"）。

    ``iters_used`` 由调用方**显式传入**（来自 ``_moveN_impl`` 的真实轮数）：
    缺口二要求它是**精确值**而非从 grain 数推的 ``0/1`` 下界——调用方既然已经
    从 ``_impl`` 拿到了轮数，就不该在这里再"猜"一次。默认 0 仅为兼容直接
    构造 ``MoveTrace`` 的旧调用点。
    """
    gain_abs = _move_target_delta(target, before, after)
    rel = None if abs(gain_abs.before) < 1e-300 else gain_abs.relative
    return MoveTrace(
        move_name=name,
        input_count=before.count,
        output_count=after.count,
        input_max_c=before.max_c,
        output_max_c=after.max_c,
        input_max_rho=before.max_rho,
        output_max_rho=after.max_rho,
        input_max_local_kt_error=before.max_local_kt_error,
        output_max_local_kt_error=after.max_local_kt_error,
        iterations=iterations,
        satisfied=satisfied,
        condition_kind=condition_kind,
        iters_used=iters_used,
        achieved_gain=gain_abs.improved,
        relative_gain=rel,
    )


@dataclass(frozen=True)
class _TargetDelta:
    """目标量的 before/after 与"是否朝目标方向改进"。"""

    before: float
    after: float
    improved: bool

    @property
    def relative(self) -> float:
        return (self.after - self.before) / abs(self.before)


def _move_target_value(kind: str, g: GrainsDecomposition) -> float:
    """取某 Move 的**目标量**当前值。"""
    if kind == "max_c":
        return float(g.max_c)
    if kind == "max_rho":
        return float(g.max_rho)
    if kind == "max_local_kt_error":
        return float(g.max_local_kt_error)
    raise InputValidationError(f"未知的 Move 目标量 {kind!r}")


def _move_target_delta(kind: str, before: GrainsDecomposition,
                       after: GrainsDecomposition) -> _TargetDelta:
    """算目标量的 before/after 与"是否朝目标方向改进"。

    **方向因 Move 而异**：Move #1 要把 ``max_c`` **推大**；Move #2/#3 要把
    ``max_rho`` / ``max_local_kt_error`` **压小**。用错方向会让
    "改进"判反——故这里显式列出，不靠符号约定。
    """
    b = _move_target_value(kind, before)
    a = _move_target_value(kind, after)
    if kind == "max_c":
        improved = a > b
    else:  # max_rho / max_local_kt_error：越小越好
        improved = a < b
    return _TargetDelta(before=b, after=a, improved=improved)

