"""芯片预算模型：把几何内核抽象成"算力 / 带宽 / 并行"的可计算瓶颈（2026-10-07）。

**这个模块为什么存在**
--------------------

项目未来要**基于这套几何逻辑做芯片（加速器）研发**。做芯片不能凭形容词
（"算得很快"），必须先回答可判定的问题：

* 这个 kernel 到底是**算力受限**还是**带宽受限**？（决定该堆 ALU 还是堆 SRAM）
* 给定一个芯片的 FLOPS / 带宽 / PE 数量，**理论上限**是多少？
* 实测（Apple M4 上 ~1 ns/tube-point pair）离理论上限**还差多少**？差在哪？

本模块把这些变成**可运行的函数**：输入芯片参数，输出瓶颈、上限、以及相对
实测的差距。它是后续"算子抽象 / 映射到 PE 阵列 / 探索设计空间"的**判据基础**。

**模型来自哪里（不凭空捏造）**
----------------------------

算子计数（:data:`FLOPS_PER_PAIR` / :data:`BYTES_PER_PAIR`）**逐行数自**
``cpp/kernel/nano_kernel.cpp`` 的 :func:`point_segment_dist2_f32` 与内层循环，注释里
写明每项怎么来的。本文件顶部**不做任何未经来源的假设**——凡有假设都标
``假设``并在 :meth:`ChipSpec.explain` 里可审计。

**Roofline 判据**
----------------

算术强度 :math:`I = \\text{FLOP} / \\text{Byte}`。

* :math:`I > I^*`（机器平衡点 = 峰值算力 / 峰值带宽）⇒ **算力受限**；
* 否则 **带宽受限**。

本 kernel 的 :math:`I` 是**可算的**（见 :func:`arithmetic_intensity`），
于是"该不该为它做专用芯片"这个问题的第一半有了答案。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "ChipSpec",
    "KernelSpec",
    "TUBE_GEOMETRY_KERNEL",
    "estimate",
    "arithmetic_intensity",
    "roofline_ceiling_ns_per_pair",
    "measured_vs_model",
    "check_model_self_consistency",
    "chipspec_declared_peak_gflops",
    "implied_rate_gflops",
    "required_cores_for_rate",
    "diagnose_superpeak",
    "predict_schedule",
    "fit_e_core_efficiency",
    "fit_p_core_all_ratio",
    "fit_schedule_params",
    "SchedulePrediction",
    "ImpossibleMeasurementError",
    "M4_REFERENCE",
    "M4_P_CORE_GFLOPS",
    "M4_E_CORE_GFLOPS",
    "Estimate",
]


class ImpossibleMeasurementError(ValueError):
    """实测结果**快于理论上限**——物理上不可能，必然是模型或口径有问题。

    **为什么单列一个异常类型**：这不是普通的输入校验失败（那种是"你传错了值"），
    而是"**两个各自看起来合理的量，合起来推出一个不可能的世界**"。它值得一个
    专门的名字，因为处置方式不同——不是叫用户改输入，而是**去查那两种根因**。

    继承 ``ValueError`` 是为了对调用方友好（沿用本项目"非法输入当场拒绝"的
    惯例，见 `resource_budget.Budget.validate`）。
    """


def implied_rate_gflops(
    ns_per_pair: float, kernel: Optional[KernelSpec] = None
) -> float:
    """由实测 ns/pair **反推**出至少需要的算力（GFLOP/s）。

    这是一个**函数**、不是一个说法：实测若真在 ``ns_per_pair`` 内算完
    ``flops_per_pair`` 次运算，那它**至少**消耗了这么多算力。任何被用来做
    上限的峰值若低于它，那个上限就**必然算错**。

    单位：``FLOP / ns = GFLOP/s``。
    """
    if ns_per_pair <= 0:
        raise ValueError(f"ns_per_pair 必须为正：{ns_per_pair}")
    k = kernel if kernel is not None else TUBE_GEOMETRY_KERNEL
    return k.flops_per_pair / ns_per_pair


def chipspec_declared_peak_gflops(chip: ChipSpec) -> float:
    """由 ``ChipSpec`` 自己的字段**独立复算**峰值算力（GFLOP/s）。

    这是"模型自洽性"检查的右侧。**不读 ``peak_flops_gflops``**——那正是被
    检查的对象。复算规则：

    * **同构**（未给 ``p_cores``/``e_cores``）：``clock × simd × 2(FMA) × 总核数``；
    * **异构**（给了 P/E 核数与比例）：``P核数 × 单核峰值 + E核数 × 单核峰值 × 比例``。

    若两者对不上，说明 `ChipSpec.peak_flops_gflops` 是手填的、与其它字段不自洽
    ——这正是本函数存在的意义（**门控依赖函数、不是依赖人的说法**）。
    """
    per_core = chip.clock_ghz * chip.simd_width * 2.0  # 单核峰值（GFLOP/s）
    if chip.p_cores > 0 and chip.e_cores > 0:
        ratio = chip.e_core_rate_ratio if chip.e_core_rate_ratio > 0 else 1.0
        return per_core * (chip.p_cores + chip.e_cores * ratio)
    return per_core * chip.n_processing_elements


def check_model_self_consistency(chip: ChipSpec, *, rel_tol: float = 0.15) -> tuple[bool, str]:
    """检查 ``ChipSpec`` 自洽：``peak_flops_gflops`` 是否与其它字段算得出的一致。

    **这是"门控依赖函数"的第一半**：不再说"可能模型参数错"，而是**去算**——
    用 ``clock × simd × 2 × cores`` 复算一遍，和声明的 ``peak_flops_gflops``
    比。不一致就**指着具体字段**说哪里错了。

    返回 ``(是否自洽, 说明)``。容差 ``rel_tol`` 容忍"P/E 核不同频"这类实现细节
    （异构芯片上声明值可能高于"全 P 核同频"的复算值）。
    """
    declared = chip.peak_flops_gflops
    recomputed = chipspec_declared_peak_gflops(chip)
    if recomputed <= 0:
        return False, f"复算峰值为 {recomputed}（时钟/SIMD 字段可疑）"
    rel = abs(declared - recomputed) / recomputed
    if rel <= rel_tol:
        return True, (
            f"自洽：声明 {declared:.1f} ≈ 复算 {recomputed:.1f} GFLOP/s"
            f"（偏差 {rel*100:.0f}% ≤ {rel_tol*100:.0f}%）"
        )
    return False, (
        f"**不自洽**：声明 peak_flops_gflops={declared:.1f}，而由"
        f" clock({chip.clock_ghz}G) × simd({chip.simd_width}) × 2"
        + (
            f" × [P{chip.p_cores} + E{chip.e_cores}×{chip.e_core_rate_ratio}]"
            if chip.p_cores > 0 and chip.e_cores > 0
            else f" × cores({chip.n_processing_elements})"
        )
        + f" 复算得 {recomputed:.1f} GFLOP/s（偏差 {rel*100:.0f}%）"
    )


#: 单个 `(tube, point)` 对在**一个** SIMD 通道上一周期能完成几次该算子的运算。
#: 用于把"实测反推的算力"换算成"需要多少个核"——这是判断"是不是把多线程
#: 当单线程报"的**函数化**判据。
def required_cores_for_rate(
    implied_gflops: float, single_core_flops_gflops: float
) -> float:
    """把"实测反推的算力"换算成**至少**需要几个（单核峰值为基准的）核。

    若实测声称是单线程、但反推算力是单核峰值的 N 倍，那这个"单线程"实测
    **至少**用了 N 个核的算力 ⇒ 不是单线程（或峰值参数错）。
    """
    if single_core_flops_gflops <= 0:
        raise ValueError("single_core_flops_gflops 必须为正")
    return implied_gflops / single_core_flops_gflops


def diagnose_superpeak(
    *,
    measured_ns_per_pair: float,
    ceiling_ns_per_pair: float,
    headroom_x: float,
    chip: Optional[ChipSpec] = None,
    single_core_flops_gflops: Optional[float] = None,
) -> str:
    """当"实测快于理论上限"时，**运行检查函数**并**报告计算得到的结论**。

    **与旧版的区别（2026-10-07 用户定调）**：旧版是"把预写好的两种解释套上数字"
    ——那是文字，不是门控。本版**真的去调用函数**：

    * :func:`implied_rate_gflops` 反推实测**至少**需要多少算力；
    * :func:`chipspec_declared_peak_gflops` + :func:`check_model_self_consistency`
      **独立复算** ``ChipSpec`` 声明的峰值是否自洽；
    * :func:`required_cores_for_rate` 把反推算力换算成"至少需要几个核"，据此
      **判定**"是不是把多线程当单线程报"。

    然后按**计算出的证据**给出结论（哪一侧矛盾），而不是并列两种猜想。
    """
    implied = implied_rate_gflops(measured_ns_per_pair)
    lines = [
        "物理合理性门控：实测快于理论上限（用超过峰值的速度算东西）——不可能。",
        f"  实测         : {measured_ns_per_pair:.4f} ns/pair",
        f"  模型上限     : {ceiling_ns_per_pair:.4f} ns/pair  (headroom {headroom_x:.2f}x)",
        f"  实测反推算力 : {implied:.1f} GFLOP/s（= {TUBE_GEOMETRY_KERNEL.flops_per_pair} FLOP/pair ÷ 实测 ns）",
    ]

    findings: list[str] = []

    # 检查①（函数）：ChipSpec 自洽吗？
    if chip is not None:
        ok, why = check_model_self_consistency(chip)
        if not ok:
            findings.append(f"① 模型字段不自洽 → {why}")
        else:
            findings.append(f"① 模型字段自洽（{why}）⇒ 单看参数本身没错")

    # 检查②（函数）：实测反推的算力需要几个核？
    if single_core_flops_gflops is not None:
        cores = required_cores_for_rate(implied, single_core_flops_gflops)
        lines.append(
            f"  反推需要核数 : {cores:.2f} 个（单核 {single_core_flops_gflops:.1f} GFLOP/s）"
        )
        if cores > 1.05:
            findings.append(
                f"② 实测反推需要 ≥{cores:.1f} 个核的算力 ⇒ 这不是单线程实测"
                "（或单核峰值参数错）"
            )
        else:
            findings.append(
                "② 反推核数 ≈1 ⇒ 若声称单线程，则**单核峰值参数偏低**"
                "（实测不可能超过一个核的峰值）"
            )

    lines.append("")
    lines.append("按计算得到的结论：")
    lines += [f"  {f}" for f in findings] if findings else ["  （未提供 chip / 单核峰值，无法复算）"]
    lines.append("")
    lines.append("确知在做'对比保守下界'时可显式传 allow_superpeak=True 放行。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 算子计数：**逐行数自 cpp/kernel/nano_kernel.cpp::point_segment_dist2_f32**
# ---------------------------------------------------------------------------
#
# 每处理一个 (tube, point) 对，内层做以下浮点运算（含 FMA 记 1 次乘加，
# 但这里按"保守的独立乘+加"计数，即 FMA 不额外折扣——见下方说明）：
#
#   apx/apy/apz = p - a            → 3 SUB
#   dot = apx*abx+apy*aby+apz*abz  → 3 MUL + 2 ADD
#   t   = dot * inv_ab_len2        → 1 MUL
#   t   = clamp(t, 0, 1)           → 2 (fmin+fmax)
#   c   = a + t*ab (x,y,z)         → 3 MUL + 3 ADD
#   d   = p - c    (x,y,z)         → 3 SUB
#   dist2 = dx*dx+dy*dy+dz*dz      → 3 MUL + 2 ADD
#                                    ─────────────
#                                    合计 25 次浮点运算
#
# **为什么 25 而不是更小**：真实编译器会把若干乘加融合成 FMA（M4 上 8 条
# 命中约 4~5 条 FMA），故 25 是**上界口径**（"没吃到 FMA 折扣"）。用它做
# 上限估计是**保守**的——得到的"理论上限"偏低，不会让人高估专用芯片收益。
# 这一步的取舍如实写在 `Estimate.assumptions` 里。
FLOPS_PER_PAIR: int = 25

#: 每对一个 (tube, point) 需要读的**内存字节**。
#:
#: tube 数据（一条 7 个 float32 的预计算结构 + 点 3 个 float32）在**理想**
#: 数据复用下，每个点读一次点坐标（12 B）、每根管每个点复用其 7 个 float
#: （28 B，但**实际会被缓存**）。这里的口径是**"理论最小"**——假设管数据
#: 全在片上、点流式读入：per pair 摊销 = 点数组 12 B / 每点扫过的管数。
#: 这是**下界**（实际带宽需求 ≥ 它），故算出的带宽上限是**乐观**的。
#: 用它审视"专用芯片能把带宽压到多低"是恰当口径。
BYTES_PER_POINT: int = 12          # 点坐标 (x,y,z)，均为 float32
BYTES_PER_TUBE_STRUCT: int = 28    # 管结构字段 ax/ay/az/abx/aby/abz/inv_ab_len2（均 float32）


@dataclass(frozen=True)
class ChipSpec:
    """一块芯片（现有或假想）的关键参数。

    单位一律显式（名字里带），避免 GB/GiB、GFLOP/GFLOPs 混用。
    """

    name: str
    #: 峰值浮点算力（float32），GFLOP/s。
    peak_flops_gflops: float
    #: 峰值内存带宽，GB/s。
    peak_bandwidth_gbytes: float
    #: 可并行执行单元（PE / 核心）数。
    n_processing_elements: int
    #: 每 PE 的私有片上缓存（KB）。0 表示不建模（视为足够）。
    sram_kb_per_pe: float = 0.0
    #: 每 PE 每周期可发射的 float32 SIMD 通道数（如 NEON=4）。
    simd_width: int = 4
    #: 时钟频率（GHz）。**单核 boost** 口径（见 :data:`p_core_all_ratio`）。
    clock_ghz: float = 1.0
    #: 性能核（P-core）数。异构芯片用它做**独立复算**（见
    #: :func:`chipspec_declared_peak_gflops`）；同构芯片留 0（视为全部同频）。
    p_cores: int = 0
    #: 能效核（E-core）数。
    e_cores: int = 0
    #: E 核相对 P 核的算力比（精度/频率/宽度综合）。异构芯片用它复算。
    e_core_rate_ratio: float = 0.0
    #: **P 核全核持续时钟 / 单核 boost 时钟**（0 = 不建模降频）。真实 CPU 在
    #: 单核 boost 时能冲到 ``clock_ghz``，但多核满载时因功耗/热包络**各核降频**
    #: ——这是"单核近线性、多核变缓"在**核数 ≤ P 核数**段的主因（见
    #: :func:`predict_schedule` 的降频项）。同构芯片/不建模留 0。
    p_core_all_ratio: float = 0.0
    #: 制造工艺（制程），如 ``"TSMC N3E (3nm)"``——**文档性字段**：它解释
    #: 为何能塞进 4P+6E、E 核又有 ~0.3 的算力比，但不直接进 roofline 数学。
    process_node: str = ""
    #: 是否支持 Armv9.2 的 **SME**（Scalable Matrix Extension，矩阵扩展）。
    #: **文档性字段**：本项目的 tube 几何算子是**逐元素**（非矩阵乘），故 SME
    #: 不改变 ``flops_per_pair`` 的口径；但它标记了"算子融合成矩阵形式"的潜力
    #: 方向（见 docs/PROJECT.md 第 III 篇）。
    has_sme: bool = False
    #: 备注（来源、口径、实测依据）。
    note: str = ""

    def explain(self) -> str:
        """把参数与"由这些参数推出什么"打印成人可审阅的文本（可审计）。"""
        lines = [
            f"芯片: {self.name}",
            f"  峰值算力     : {self.peak_flops_gflops:.1f} GFLOP/s (float32)",
            f"  峰值带宽     : {self.peak_bandwidth_gbytes:.1f} GB/s",
            f"  PE 数        : {self.n_processing_elements}",
            f"  每 PE SRAM   : {self.sram_kb_per_pe:.1f} KB",
            f"  每 PE SIMD   : {self.simd_width} (float32 通道)",
            f"  时钟         : {self.clock_ghz:.2f} GHz（单核 boost）",
        ]
        if self.p_core_all_ratio > 0:
            lines.append(f"  P 核全核/单核 : {self.p_core_all_ratio:.2f}（多核满载降频）")
        if self.process_node:
            lines.append(f"  工艺         : {self.process_node}")
        if self.has_sme:
            lines.append("  SME          : 支持（Armv9.2 矩阵扩展）")
        if self.note:
            lines.append(f"  备注         : {self.note}")
        return "\n".join(lines)


@dataclass(frozen=True)
class KernelSpec:
    """一个 kernel 的算子计数（来自实测源码，不是估的）。"""

    name: str
    #: 每个 (tube, point) 对的浮点运算数。
    flops_per_pair: int
    #: 每个点读入的字节（数据流下界）。
    bytes_per_point: int
    #: 每根管的预计算结构字节（可常驻片上）。
    bytes_per_tube_struct: int
    #: 计数来源说明（哪个文件、哪一段）。
    source: str = ""

    def explain(self) -> str:
        return (
            f"kernel: {self.name}\n"
            f"  FLOP / pair      : {self.flops_per_pair}\n"
            f"  bytes / point    : {self.bytes_per_point}\n"
            f"  bytes / tube     : {self.bytes_per_tube_struct}\n"
            f"  计数来源         : {self.source}"
        )


#: 项目里实际用的 tube 几何包含测kernel（计数来自 cpp/kernel/nano_kernel.cpp）。
TUBE_GEOMETRY_KERNEL = KernelSpec(
    name="tube_point_dist2_contains",
    flops_per_pair=FLOPS_PER_PAIR,
    bytes_per_point=BYTES_PER_POINT,
    bytes_per_tube_struct=BYTES_PER_TUBE_STRUCT,
    source=(
        "cpp/kernel/nano_kernel.cpp::point_segment_dist2_f32 + "
        "voxel_union_volume_f32 内层循环（逐行计数，见本文件顶部注释）"
    ),
)


@dataclass(frozen=True)
class Estimate:
    """给定芯片 + 构型下的一次分析结果。"""

    chip: ChipSpec
    kernel: KernelSpec
    n_tubes: int
    n_points: int
    #: 算术强度（FLOP/Byte），pair 级。
    intensity: float
    #: 机器平衡点 I* = 峰值算力 / 峰值带宽（FLOP/Byte）。
    machine_balance: float
    #: 瓶颈：'compute' 或 'memory'。
    bottleneck: str
    #: 算力受限下的理论上限（ns/pair）。
    compute_ceiling_ns_per_pair: float
    #: 带宽受限下的理论上限（ns/pair）。
    memory_ceiling_ns_per_pair: float
    #: 实际 roofline 上限 = max(两者)（更慢的那个说了算）。
    roofline_ns_per_pair: float
    #: 本次构型的总工作量与总理论上限时间（毫秒）。
    total_pairs: int
    total_ceiling_ms: float
    #: 分析所用的假设（可审计）。
    assumptions: list[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = [
            self.chip.explain(),
            "",
            self.kernel.explain(),
            "",
            f"构型: {self.n_tubes} 管 × {self.n_points} 点",
            f"  总 pair 数        : {self.total_pairs:,}",
            f"  算术强度 I        : {self.intensity:.3f} FLOP/Byte",
            f"  机器平衡点 I*     : {self.machine_balance:.3f} FLOP/Byte",
            f"  ⇒ 瓶颈            : {self.bottleneck}",
            f"  算力上限          : {self.compute_ceiling_ns_per_pair:.4f} ns/pair",
            f"  带宽上限          : {self.memory_ceiling_ns_per_pair:.4f} ns/pair",
            f"  ⇒ roofline 上限   : {self.roofline_ns_per_pair:.4f} ns/pair",
            f"  ⇒ 总上限          : {self.total_ceiling_ms:.4f} ms",
        ]
        if self.assumptions:
            lines.append("  假设:")
            lines.extend(f"    - {a}" for a in self.assumptions)
        return "\n".join(lines)


def arithmetic_intensity(
    kernel: KernelSpec = TUBE_GEOMETRY_KERNEL, n_tubes: int = 64
) -> float:
    """pair 级算术强度（FLOP/Byte）。

    **分母依赖构型**：一个点被 ``n_tubes`` 根管复用 ⇒ 每个 pair 摊销
    ``bytes_per_point / n_tubes`` 字节。故强度随**管数增加而升高**（复用越多）。
    漏掉 ``n_tubes`` 会把强度算成与构型无关的低估——那是错的。

    **口径**：分母是"数据流下界"（管结构假定常驻片上），故这是**理想复用**
    下的强度、是**上界**；实际强度 ≤ 它（缓存没这么完美）。用它判断"是否
    很可能带宽受限"是恰当的第一刀。
    """
    if n_tubes <= 0:
        raise ValueError(f"n_tubes 必须为正：{n_tubes}")
    bytes_per_pair = kernel.bytes_per_point / n_tubes
    return kernel.flops_per_pair / bytes_per_pair


def roofline_ceiling_ns_per_pair(
    chip: ChipSpec, n_tubes: int
) -> tuple[float, float]:
    """返回 (算力上限, 带宽上限) 的 ns/pair。

    算力上限：每 pair 要 FLOPS_PER_PAIR 次运算，芯片每秒能算
    ``peak_flops_gflops × 1e9`` 次 ⇒ 每 pair 秒数 × 1e9 = ns。

    带宽上限：一个点被 ``n_tubes`` 根管复用 ⇒ **每个 pair** 摊销
    ``BYTES_PER_POINT / n_tubes`` 字节；芯片每秒能搬
    ``peak_bandwidth_gbytes × 1e9`` 字节。

    ``n_tubes`` 是必需参数——带宽摊销**依赖构型**（管越多、点复用越多、
    每 pair 带宽成本越低），漏掉它会把带宽上限算成与构型无关，那是错的。
    """
    k = TUBE_GEOMETRY_KERNEL
    if n_tubes <= 0:
        raise ValueError(f"n_tubes 必须为正：{n_tubes}")
    flops_per_s = chip.peak_flops_gflops * 1e9
    bytes_per_s = chip.peak_bandwidth_gbytes * 1e9

    compute_ns = k.flops_per_pair / flops_per_s * 1e9
    bytes_per_pair = k.bytes_per_point / n_tubes
    memory_ns = bytes_per_pair / bytes_per_s * 1e9
    return compute_ns, memory_ns


def measured_vs_model(
    chip: ChipSpec,
    measured_ns_per_pair: float,
    n_tubes: int,
    *,
    measured_is_single_thread: bool = True,
    single_core_flops_gflops: Optional[float] = None,
    allow_superpeak: bool = False,
) -> dict:
    """把**实测** ns/pair 与模型上限对比，**并显式处理线程口径**。

    这是把模型用于"该不该做专用芯片"决策时**最容易出错的一步**：实测通常
    来自**单线程**（本项目 ``nano/isolated/bench_micro`` 就是单线程），而
    ``chip`` 参数常写的是**全芯片**峰值。直接相除会得到虚高的"余量"
    （本机实测单线程 1.0 ns/pair vs 全芯片上限 0.098 ns/pair ⇒ 10×，
    看似芯片能再快 10 倍，其实那是"单核 vs 全芯片"的口径错配）。

    **异构芯片的正确口径**：单线程实测跑在**一个 P 核**上，故单核上限应由
    **P 核峰值**算出、而**不是**"全芯片峰值 / 核数"——后者会把 E 核拉低平均
    值，得到偏低的每核峰值（本机踩过：用平均得 1.23 ns/pair，用 P 核得
    0.71 ns/pair，差 1.7×）。故：

    * 若传 ``single_core_flops_gflops``（如 M4 的 P 核 35.2），单核上限由它算；
    * 否则从 ``chip.peak_flops_gflops / n_pe`` 退化估计（**仅适用于同构芯片**）；
    * ``measured_is_single_thread=False`` 时用全芯片上限，不做折算。

    **物理合理性门控（``allow_superpeak``）**：若算出的 ``headroom_x < 1.0``
    （实测**快于**理论上限），那意味着"用超过峰值算力的速度算东西"——**物理上
    不可能**。这只有两种解释（见 :func:`check_physical_plausibility`）：

    1. **模型参数错**（本机真实踩过：``M4_REFERENCE`` 漏算 E 核，导致
       "单核已用峰值 179%"）；
    2. **"实测"不是真的单线程实测**（如被编译器逆向优化掉、或计数口径与
       声称的不符）。

    两者都必须**当场报错**、而不是静默返回一个漂亮但假的数字。确知自己拿的是
    "保守下界"做对比时可显式传 ``allow_superpeak=True`` 放行。
    """
    compute_ns, memory_ns = roofline_ceiling_ns_per_pair(chip, n_tubes)
    ceiling = max(compute_ns, memory_ns)
    note = "ceiling 为全芯片上限"

    if measured_is_single_thread:
        if single_core_flops_gflops is not None:
            # 单核上限 = FLOPS_PER_PAIR / (P 核 GFLOP/s)，与构型无关。
            ceiling = TUBE_GEOMETRY_KERNEL.flops_per_pair / (
                single_core_flops_gflops * 1e9
            ) * 1e9
            note = f"ceiling 由单核峰值 {single_core_flops_gflops} GFLOP/s 算出"
        elif chip.n_processing_elements > 1:
            # 退化估计（同构芯片才准确）；异构芯片请显式传 single_core_flops_gflops。
            ceiling = ceiling * chip.n_processing_elements
            note = (
                "ceiling 按'全芯片/核数'退化折算（**同构假设**；异构芯片请传 "
                "single_core_flops_gflops 以用 P 核峰值）"
            )

    headroom = measured_ns_per_pair / ceiling

    # 物理合理性门控：实测快于理论上限 ⇒ 不可能。**运行检查函数**判定哪侧矛盾。
    if headroom < 1.0 and not allow_superpeak:
        raise ImpossibleMeasurementError(
            diagnose_superpeak(
                measured_ns_per_pair=measured_ns_per_pair,
                ceiling_ns_per_pair=ceiling,
                headroom_x=headroom,
                chip=chip,
                single_core_flops_gflops=single_core_flops_gflops,
            )
        )

    return {
        "measured_ns_per_pair": measured_ns_per_pair,
        "ceiling_ns_per_pair": ceiling,
        "headroom_x": headroom,
        "bottleneck": "compute" if compute_ns >= memory_ns else "memory",
        "note": note,
    }



def estimate(
    chip: ChipSpec,
    n_tubes: int,
    n_points: int,
    kernel: KernelSpec = TUBE_GEOMETRY_KERNEL,
) -> Estimate:
    """分析一次构型在该芯片上的瓶颈与理论上限。"""
    if n_tubes <= 0 or n_points <= 0:
        raise ValueError(f"n_tubes 与 n_points 必须为正：{n_tubes=} {n_points=}")

    total_pairs = n_tubes * n_points

    # --- 算力上限 ---------------------------------------------------------
    # 总 FLOP = pair 数 × 每 pair FLOP；除以芯片每秒 FLOP ⇒ 秒 ⇒ ns/pair。
    flops_per_s = chip.peak_flops_gflops * 1e9
    compute_ns = kernel.flops_per_pair / flops_per_s * 1e9

    # --- 带宽上限 ---------------------------------------------------------
    # 每个点读 12 B；一个点被 n_tubes 根管复用 ⇒ **每个 pair** 摊销
    # bytes_per_point / n_tubes。故带宽口径下每 pair 秒数 = 该摊销 / 带宽。
    bytes_per_s = chip.peak_bandwidth_gbytes * 1e9
    bytes_per_pair = kernel.bytes_per_point / n_tubes
    memory_ns = bytes_per_pair / bytes_per_s * 1e9

    # --- roofline：慢的那个说了算 ----------------------------------------
    roofline_ns = max(compute_ns, memory_ns)
    bottleneck = "compute" if compute_ns >= memory_ns else "memory"

    intensity = kernel.flops_per_pair / bytes_per_pair
    machine_balance = flops_per_s / bytes_per_s

    total_ceiling_ms = roofline_ns * total_pairs / 1e6

    return Estimate(
        chip=chip,
        kernel=kernel,
        n_tubes=n_tubes,
        n_points=n_points,
        intensity=intensity,
        machine_balance=machine_balance,
        bottleneck=bottleneck,
        compute_ceiling_ns_per_pair=compute_ns,
        memory_ceiling_ns_per_pair=memory_ns,
        roofline_ns_per_pair=roofline_ns,
        total_pairs=total_pairs,
        total_ceiling_ms=total_ceiling_ms,
        assumptions=[
            f"{kernel.flops_per_pair} FLOP/pair 是**上界口径**（未折扣 FMA；"
            "真实编译器会融合若干乘加，故实际上限更高、roofline 更乐观）",
            "带宽按数据流**下界**计（管结构假定常驻片上、点流式读入）",
            "只算 float32 运算；控制流/索引/边界检查不计",
            "未计 Python/pybind11 与调用开销（纯 kernel 上限）",
        ],
    )


#: M4 单核（P 核）峰值算力（GFLOP/s）——与 ``bench_micro.cpp`` **同源**。
#:
#: 4.4 GHz × 4 fp32 SIMD 通道 × 2（FMA）= 35.2 GFLOP/s。用于"单线程实测"
#: 与模型对比时的**单核上限**（异构芯片不能用"全芯片/核数"，那会把 E 核
#: 拉低平均值）。改这里时必须同步 ``nano/isolated/bench_micro.cpp`` 的
#: ``p_core_gflops``——两处漂移会被 `test_microbench_integration.py` 抓到。
M4_P_CORE_GFLOPS: float = 35.2

#: M4 单核（E 核）峰值算力（GFLOP/s）——保守取 P 核约 30%。
M4_E_CORE_GFLOPS: float = 10.6


# ---------------------------------------------------------------------------
# 参考芯片：Apple M4（用来**校准模型**——见 tests/test_chip_model.py）
# ---------------------------------------------------------------------------
#
# **这些数字的口径必须写清**，否则模型无意义。
#
# **2026-10-07 修正**：初版取 140 GFLOP/s（= 4 P 核 × 35.2），但那样**漏算了
# 6 个 E 核**，且让"单核峰值"变成 14 GFLOP/s——比实测还低（实测单核 0.9986
# ns/pair × 25 FLOP = **25 GFLOP/s**，反推出单核峰值 ≥ 25）。**这个矛盾是
# `nano/isolated/bench_micro` 与模型对齐时暴露的**——正是"两处口径交叉核对"
# 的价值。改法：
#   * P 核：4.4 GHz × 4 fp32 SIMD 通道 × 2（FMA）= **35.2 GFLOP/s/核**；
#   * E 核：频率与宽度都低，保守取 P 核的 ~30% ⇒ ~10.6 GFLOP/s/核；
#   * 全芯片 = 4 × 35.2 + 6 × 10.6 ≈ **204 GFLOP/s**（理论峰值，非可持续值）。
M4_REFERENCE = ChipSpec(
    name="Apple M4 (reference)",
    peak_flops_gflops=4 * M4_P_CORE_GFLOPS + 6 * M4_E_CORE_GFLOPS,
    peak_bandwidth_gbytes=120.0,
    n_processing_elements=10,
    sram_kb_per_pe=128.0,   # 保守：L1D 约 128 KB/核（Apple 未公开，取量级）
    simd_width=4,
    clock_ghz=4.4,
    p_cores=4,
    e_cores=6,
    e_core_rate_ratio=M4_E_CORE_GFLOPS / M4_P_CORE_GFLOPS,  # ~0.30
    # 全核/单核 P 时钟比：由本机实测曲线 (n,T(n)) **联合拟合**得出（见
    # ``fit_schedule_params`` 与 test_schedule_prediction.py）——单核 boost
    # 4.4 GHz，4 个 P 核满载时约 81%（≈3.6 GHz），典型的功耗/热包络降频。
    # 0 表示不建模（旧口径）。
    p_core_all_ratio=0.81,
    process_node="TSMC N3E (3nm)",
    has_sme=True,
    note=(
        "M4 参数为**理论峰值量级估计**（Apple 未公开 GFLOPS）：P 核 35.2、"
        "E 核 ~10.6（P 的 30%），全芯片 ~204 GFLOP/s；带宽 120 GB/s 为官方值。"
        "工艺 TSMC N3E、支持 SME；P 核全核降频至单核约 81%（实测拟合）。"
        "用途是校准模型、而非宣称精确。单核峰值见 M4_P_CORE_GFLOPS。"
    ),
)


def estimate_single_core_ms(
    n_tubes: int,
    n_points: int,
    chip: Optional[ChipSpec] = None,
    *,
    measured_ns_per_pair: Optional[float] = None,
    efficiency: float = 1.0,
) -> float:
    """由**问题规模**推算**单核墙钟**（毫秒）——把调度计划的单核输入从
    "调用方拍脑袋给的 ms"变成"模型从规模推出"。

    这是 ``nano.scheduling.scheduler.plan_query`` 规模感知路径（``sizes=...``）的模型侧：
    pair 级几何算子（体素 / 蒙特卡洛 / 点包含）的单核墙钟 =
    ``n_tubes × n_points`` × 单核 ns/pair。单核 ns/pair 的口径**必须分清**
    （沿用 :func:`measured_vs_model` 的异构教训：单核上限由 **P 核峰值**算，
    不是"全芯片 / 核数"——后者会把 E 核拉进平均，得到偏低的每核峰值）：

    * ``measured_ns_per_pair``（**推荐**）：``nano/isolated/bench_micro`` 的
      单线程实测值（本机 ≈1.0 ns/pair）。最诚实——已含编译器融合、流水线、
      访存延迟等一切效应；传了它，``efficiency`` 不参与（实测本身已含折扣）。
    * 否则用 **P 核 roofline 上限** × ``efficiency`` 折扣。上限是**乐观口径**
      （FLOP 上界计数、未折扣 FMA，见 :func:`estimate` 的假设），故
      ``efficiency=1`` 给出的是"理论最好"；做诚实的计划应传实测值，或按
      实测/上限之比（本机 ≈1.4）给 ``efficiency<1``。

    P 核峰值从芯片参数**推导**（异构：``peak / (p + e·e_core_rate_ratio)``；
    同构退化：``peak / n_pe``），换芯片无需另传单核峰值。带宽口径与
    :func:`estimate` 一致（每 pair 摊销 ``bytes_per_point / n_tubes``，管结构
    假定常驻片上），取 max(算力, 带宽)——慢的那个说了算。

    非 pair 级模块（grains / moves / proof）没有 pair 计数模型，**不在本函数
    口径内**——调度器对它们直接跳过规模推算（如实登记，不硬套公式）。
    """
    if n_tubes < 1 or n_points < 1:
        raise ValueError(
            f"n_tubes 与 n_points 必须为正：{n_tubes=} {n_points=}"
        )
    if measured_ns_per_pair is not None and measured_ns_per_pair <= 0:
        raise ValueError(f"measured_ns_per_pair 须为正：{measured_ns_per_pair}")
    if not (0.0 < efficiency <= 1.0):
        raise ValueError(f"efficiency 须在 (0,1]：{efficiency}")

    c = chip if chip is not None else M4_REFERENCE

    if measured_ns_per_pair is not None:
        ns_per_pair = measured_ns_per_pair
    else:
        # 单核（P 核）峰值算力：异构芯片从拓扑与 E 核算力比反推，同构退化。
        if c.p_cores > 0 and c.e_core_rate_ratio > 0:
            p_gflops = c.peak_flops_gflops / (
                c.p_cores + c.e_cores * c.e_core_rate_ratio
            )
        elif c.n_processing_elements > 1:
            p_gflops = c.peak_flops_gflops / c.n_processing_elements
        else:
            p_gflops = c.peak_flops_gflops
        compute_ns = TUBE_GEOMETRY_KERNEL.flops_per_pair / (p_gflops * 1e9) * 1e9
        bytes_per_pair = TUBE_GEOMETRY_KERNEL.bytes_per_point / n_tubes
        memory_ns = bytes_per_pair / (c.peak_bandwidth_gbytes * 1e9) * 1e9
        # efficiency 是"达到上限的几分之几"：墙钟 = 上限 / efficiency。
        ns_per_pair = max(compute_ns, memory_ns) / efficiency

    return n_tubes * n_points * ns_per_pair / 1e6


# ---------------------------------------------------------------------------
# 能耗模型（2026-10-30）——"降热"从比喻变成可计算的判据
# ---------------------------------------------------------------------------
#
# 动机（用户 2026-10-30 定调）：优化芯片调度的同时降低发热，切入点在
# **信息回传**——搬运数据的能耗远高于计算（"让进内存的信息是有用的，
# 筛选留给片上"）。时间 roofline（上节）回答"多快"，本节回答"多烫"：
# 功率 ≈ 能耗/时间，能耗 = 计算能耗 + 数据搬运能耗。
#
# **口径警示**：常数取 45nm 文献值（Horowitz, ISSCC 2014「Computing's
# Energy Problem」：fp32 乘 3.7 pJ/FLOP、DRAM 读 650 pJ/32-bit 字
# ≈ 162.5 pJ/字节）。M4 是 N3E 工艺，**绝对值会显著更低**——但
# "搬运 ≫ 计算"的量级比（几十~上百倍）跨工艺保持，这才是本模型要
# 传达的设计判据，不是功耗预测。换口径只改这里两个常量。

#: fp32 运算能耗（pJ/FLOP，45nm 乘法口径，见上节警示）。
ENERGY_PJ_PER_FLOP_F32: float = 3.7

#: DRAM 搬运能耗（pJ/字节，45nm 650 pJ/32-bit 字 ÷ 4，见上节警示）。
ENERGY_PJ_PER_DRAM_BYTE: float = 162.5

#: 融合 MC **片内采样开销**（splitmix64×3 + 线性映射，ns/样本）——
#: **实测标定 = 0.0**（2026-10-30，bench_micro 配对交替协议三连跑：
#: −0.9~−0.04 ns/样本，在 ±1 ns 噪声内）。采样指令与管扫描独立、被
#: 乱序执行隐藏 ⇒ "算力换带宽"的算力侧代价实测 ≈ 0，融合净收益就是
#: 内存往返的全额消除（搬运占比 66% → 0）。复验：
#: ``KAKEYA_BENCH_FUSED_MC=1 ./bench_micro``。
#: ⚠️ 与 ``nano/isolated/bench_micro.cpp`` 的同名 constexpr **两处同源**——
#: ``tests/test_microbench_integration.py`` 守这个同步。
FUSED_MC_SAMPLING_OVERHEAD_NS: float = 0.0


def estimate_energy_pj(
    n_tubes: int,
    n_points: int,
    kernel: KernelSpec = TUBE_GEOMETRY_KERNEL,
    *,
    bytes_moved: Optional[float] = None,
) -> dict:
    """估算一次构型的能耗（pJ）与"搬运/计算"占比——发热优化的判据。

    参数
    ----
    n_tubes, n_points : 构型（管数 × 点数/pair 数）。
    kernel : 算子计数（默认 tube 几何内核）。
    bytes_moved : **显式数据搬运量**（字节）。这是本函数的核心价值：
        对比"融合前/后"两条数据流时传入各自的实测/实算字节数（如 MC
        旧路径 numpy 生成 f64 + 转 f32 + pybind 传参 ≈ 每样本 72 B；
        融合版点数组不进内存 = 0），能耗差即"降热"量。
        缺省 = 理想数据流（点流式读一次 + 管结构读一次）：
        ``n_points × bytes_per_point + n_tubes × bytes_per_tube_struct``。

    返回 dict（键固定，供测试与调用方断言）：
    ``compute_pj`` / ``memory_pj`` / ``total_pj`` / ``memory_share``
    （搬运占比 0~1）/ ``ratio_memory_over_compute`` / ``note``。

    **不算芯片参数**：能耗常数是工艺无关的量级占位（见上节警示），
    与 :func:`estimate`（时间 roofline）互补而非复用其芯片参数。
    """
    if n_tubes < 1 or n_points < 1:
        raise ValueError(
            f"n_tubes 与 n_points 必须为正：{n_tubes=} {n_points=}"
        )
    if bytes_moved is not None and bytes_moved < 0:
        raise ValueError(f"bytes_moved 必须 ≥0：{bytes_moved}")

    total_pairs = n_tubes * n_points
    compute_pj = total_pairs * kernel.flops_per_pair * ENERGY_PJ_PER_FLOP_F32
    if bytes_moved is None:
        bytes_moved = (
            n_points * kernel.bytes_per_point
            + n_tubes * kernel.bytes_per_tube_struct
        )
    memory_pj = bytes_moved * ENERGY_PJ_PER_DRAM_BYTE

    total = compute_pj + memory_pj
    return {
        "compute_pj": compute_pj,
        "memory_pj": memory_pj,
        "total_pj": total,
        "memory_share": memory_pj / total if total > 0 else 0.0,
        "ratio_memory_over_compute": (
            memory_pj / compute_pj if compute_pj > 0 else float("inf")
        ),
        "note": (
            "能耗常数为 45nm 文献口径（Horowitz ISSCC 2014），量级比"
            "（搬运 ≫ 计算）跨工艺成立，绝对值非 N3E 实测"
        ),
    }


# ---------------------------------------------------------------------------
# 调度预测模型（2026-10-07）——从**单核实测**预测**多核调度**的结果
# ---------------------------------------------------------------------------
#
# **要回答的问题**（用户 2026-10-07 定调）："在微基准的单核延迟和吞吐量上，
# 如何把整个算法在芯片的调度层面上进行优化？"——即：**给定单核延迟 T1，用 n
# 个（异构）核跑，墙钟是多少？n 该取几？**
#
# **为什么需要预测、而不是每次实测**：本机约定"一次最多两个进程"（见
# CONTRIBUTING.md 的资源约束），逐个线程数实测既慢又占资源。若有一个**从单核数据出发**的模型能预测
# 各 n 的表现，就能在**不跑**的情况下选调度策略——预测准不准由测试用**已有的
# 实测曲线**回验（见 tests/test_schedule_prediction.py）。
#
# **模型形式**（异构核效率可加）：
#   设最快核效率 = 1，E 核效率 = `r`（0<r≤1，可从事后数据**拟合**）。
#   n 个线程按"先用 P 核、再用 E 核"分配（调度器通常如此）：
#     speedup(n) = Σ_{i<n} eff(core_i)
#   墙钟  T(n) = T1 / speedup(n) + per_thread_overhead_ms·(n-1)。
#   其中 per_thread_overhead_ms 是**每多一个线程的固定开销**（fork-join 同步
#   等），默认 0；>0 时曲线可呈 U 形（见 :func:`predict_schedule`）。
# 理想（同构、开销=0）时 eff≡1 ⇒ T(n)=T1/n（Amdahl 无串行部分的极限）。
#
# **实测校准点（M4，64 管，grid=96，fast_kernel）**：
#   n:     1     2     3     4     5     6     7     8     9    10
#   ms: 51.4  27.9  19.5  16.5  14.0  12.0  11.3   9.9   9.7  10.3
#   读法：1→5 近线性；5→9 变缓（E 核弱）；**10 反而略慢于 9**（超订/争用）。
# ⇒ 模型必须能解释"第 10 个线程是负收益"，否则选不出最优 n。


@dataclass(frozen=True)
class SchedulePrediction:
    """一次调度预测的结果（可审计）。"""

    single_core_ms: float
    p_cores: int
    e_cores: int
    e_core_efficiency: float
    #: n -> 预测墙钟（毫秒）。
    predicted_ms: dict
    #: 预测最优线程数与其墙钟。
    best_n: int
    best_ms: float
    #: 预测依据（可审计）。
    assumptions: list

    def explain(self) -> str:
        lines = [
            f"单核墙钟            : {self.single_core_ms:.3f} ms",
            f"核拓扑              : {self.p_cores}P + {self.e_cores}E",
            f"E 核效率（相对 P）  : {self.e_core_efficiency:.3f}",
            "",
            "  n    预测 ms    实测加速比(相对单核)",
        ]
        for n in sorted(self.predicted_ms):
            sp = self.single_core_ms / self.predicted_ms[n]
            lines.append(f"  {n:>2}   {self.predicted_ms[n]:8.3f}   {sp:6.2f}×")
        lines.append(f"\n⇒ 预测最优 n = {self.best_n}（{self.best_ms:.3f} ms）")
        if self.assumptions:
            lines.append("假设:")
            lines.extend(f"  - {a}" for a in self.assumptions)
        return "\n".join(lines)


def predict_schedule(
    single_core_ms: float,
    *,
    p_cores: int,
    e_cores: int,
    e_core_efficiency: float,
    max_threads: int = 0,
    serial_fraction: float = 0.0,
    per_thread_overhead_ms: float = 0.0,
    p_core_all_ratio: float = 0.0,
) -> SchedulePrediction:
    """由**单核墙钟**预测各线程数下的墙钟，并给出最优 n。

    参数
    ----
    single_core_ms : 单线程（跑在一个 P 核上）的墙钟，毫秒。
    p_cores, e_cores : 拓扑。n 个线程时"先用 P 核、再用 E 核"。
    e_core_efficiency : E 核相对 P 核的效率（0<r≤1）。**这是调度效率**（不是
        ``ChipSpec.e_core_rate_ratio`` 的理论算力比）——它含内存延迟/争用等
        效应，故可与算力比不同。可由一组实测 (n, T(n)) 拟合（见
        :func:`fit_e_core_efficiency`）。
    max_threads : 预测到几线程为止（0 = p_cores+e_cores）。
    serial_fraction : **不可并行**部分占比（Amdahl，0≤s<1）。默认 0。
    per_thread_overhead_ms : 每多一个线程的**固定开销**（毫秒），以 α·(n-1)
        形式加到墙钟（n=1 是定义点，不加）。默认 0（不开启）——它针对的是
        n 超过核数后的超订回摆（如 n=9→10 的回升），与下文的 P 核降频**是两
        回事**：降频解释 n≤P 核段的变缓，α 解释 n>核数段可能的回升。
    p_core_all_ratio : **P 核全核持续时钟 / 单核 boost 时钟**（0<d≤1，0=不建模
        降频）。真实 CPU 单核 boost 能冲到 ``clock_ghz``，但 n 个 P 核满载时
        各核降频——这是"单核近线性、多核变缓"在 **n≤P 核数**段的主因（Amdahl
        串行项解释不了它：本机实测加 s>0 反而更差）。M4 实测拟合 **d≈0.81**
        （4 P 核满载约 81% 单核频率）。默认 0 = 旧口径（无降频）。

    模型
    ----
    ``core_speed(n) = p_used·p_ratio(p_used) + e_used·r``，其中
    ``p_ratio(k) = 1 - (1-d)·(k-1)/(p_cores-1)``（线性插值：1 核=1、全 P 核=d），
    ``p_used=min(n,p_cores)``、``e_used=max(0,n-p_cores)``；
    ``speedup(n) = 1 / (s + (1-s)/core_speed(n))``；
    ``T(n) = T1 / speedup(n) + per_thread_overhead_ms·(n-1)``。
    d=0、s=0、α=0 时退化为 ``T(n) = T1 / core_speed(n)``（旧口径）。
    """
    if single_core_ms <= 0:
        raise ValueError(f"single_core_ms 必须为正：{single_core_ms}")
    if p_cores < 0 or e_cores < 0 or (p_cores + e_cores) <= 0:
        raise ValueError(f"核数非法：p={p_cores} e={e_cores}")
    if not (0.0 < e_core_efficiency <= 1.0):
        raise ValueError(f"E 核效率须在 (0,1]：{e_core_efficiency}")
    if not (0.0 <= serial_fraction < 1.0):
        raise ValueError(f"serial_fraction 须在 [0,1)：{serial_fraction}")
    if per_thread_overhead_ms < 0:
        raise ValueError(f"per_thread_overhead_ms 须 ≥0：{per_thread_overhead_ms}")
    if p_core_all_ratio != 0.0 and not (0.0 < p_core_all_ratio <= 1.0):
        raise ValueError(f"p_core_all_ratio 须为 0（不建模）或 (0,1]：{p_core_all_ratio}")

    total = p_cores + e_cores
    n_max = max_threads if max_threads > 0 else total

    predicted: dict = {}
    for n in range(1, n_max + 1):
        p_used = min(n, p_cores)
        e_used = max(0, n - p_cores)
        # P 核降频：p_ratio(k) 从 1（单核 boost）线性降到 d（全 P 核）。
        if p_core_all_ratio > 0.0 and p_cores > 1 and p_used > 1:
            p_ratio = 1.0 - (1.0 - p_core_all_ratio) * (p_used - 1) / (p_cores - 1)
        else:
            p_ratio = 1.0
        core_speed = p_used * p_ratio + e_used * e_core_efficiency
        if core_speed <= 0:
            core_speed = 1e-9
        # Amdahl：串行的 s 部分不随核增加而变快。
        speedup = 1.0 / (serial_fraction + (1.0 - serial_fraction) / core_speed)
        # 每线程固定开销 α·(n-1)：n=1 是定义点不加，α>0 可让曲线 U 形。
        predicted[n] = single_core_ms / speedup + per_thread_overhead_ms * (n - 1)

    best_n = min(predicted, key=lambda k: predicted[k])
    return SchedulePrediction(
        single_core_ms=single_core_ms,
        p_cores=p_cores,
        e_cores=e_cores,
        e_core_efficiency=e_core_efficiency,
        predicted_ms=predicted,
        best_n=best_n,
        best_ms=predicted[best_n],
        assumptions=[
            "n 个线程按'先用 P 核、再用 E 核'分配（调度器通常如此）",
            f"serial_fraction={serial_fraction:.3f}（0 = 无线程/串行开销）",
            "核心'speedup'是理想化的；s>0 把不可并行部分纳入",
            f"per_thread_overhead_ms={per_thread_overhead_ms:.3f}"
            "（0 = 无每线程开销；>0 时墙钟可随 n 回升）",
            f"p_core_all_ratio={p_core_all_ratio:.2f}"
            "（0 = 不建模降频；>0 时 P 核多核满载降频）",
            "不建模超订（n>核数）的回退",
        ],
    )


def fit_schedule_params(
    single_core_ms: float,
    measured_ms: dict,
    *,
    p_cores: int,
    e_cores: int,
) -> tuple[float, float]:
    """**联合**拟合调度模型的两个校准参数，返回 ``(e_core_efficiency, p_core_all_ratio)``。

    **为什么必须联合、而不是分开**：E 核效率 r 与 P 核降频比 d **耦合**——若不
    建模降频（d 恒 1），"P 核多核变缓"会被错误地记到 r 头上，导致 r 被低估
    （本机旧口径 r≈0.19）；分离降频后 r 升至 ≈0.44。故要用**粗→细网格**在
    (r, d) 平面上搜**最小化整条曲线的平均相对误差**，返回最优对。

    **参数**：与 :func:`fit_e_core_efficiency` / :func:`fit_p_core_all_ratio`
    同口径——``measured_ms`` 是 ``{n: 墙钟 ms}``（n 从 1 起）。

    **与两个单参数拟合器的关系**：本函数是**权威校准**；那两个是**快速顺序估计**
    （先 median 估 d、再 median 估 r），快但不如联合拟合准（顺序估计各自对离群
    点敏感）。要精确校准用本函数。
    """
    if single_core_ms <= 0:
        raise ValueError(f"single_core_ms 必须为正：{single_core_ms}")
    if not measured_ms:
        raise ValueError("measured_ms 不能为空")

    def mean_err(r: float, d: float) -> float:
        pred = predict_schedule(
            single_core_ms, p_cores=p_cores, e_cores=e_cores,
            e_core_efficiency=r, p_core_all_ratio=d,
        )
        errs = [
            abs(pred.predicted_ms[n] - t) / t
            for n, t in measured_ms.items() if t > 0 and n in pred.predicted_ms
        ]
        return sum(errs) / len(errs) if errs else float("inf")

    # 粗网格 → 细网格（各 21×21，共 ~900 次预测，纯本地、代价可忽略）。
    best = (float("inf"), 0.3, 0.8)
    for r_lo, r_hi, d_lo, d_hi, step in (
        (0.1, 0.9, 0.5, 1.0, 0.04),
        (0.3, 0.6, 0.7, 0.95, 0.005),
    ):
        r = r_lo
        while r <= r_hi + 1e-12:
            d = d_lo
            while d <= d_hi + 1e-12:
                e = mean_err(r, d)
                if e < best[0]:
                    best = (e, r, d)
                d += step
            r += step

    # 细网格最优附近的再精确到 3 位（一维局部搜索，避免网格精度不足）。
    _, r0, d0 = best
    for dr in (0.001, -0.001):
        for dd in (0.001, -0.001):
            e = mean_err(r0 + dr, d0 + dd)
            if e < best[0]:
                best = (e, r0 + dr, d0 + dd)
    return (round(best[1], 4), round(best[2], 4))


def fit_p_core_all_ratio(
    single_core_ms: float,
    measured_ms: dict,
    *,
    p_cores: int,
) -> float:
    """从实测 (n, T(n)) **拟合** P 核全核/单核降频比 d（区间 (0,1]）。
    **只用 P 核段**（``2≤n≤p_cores``）的点：那段不含 E 核效率 r，只有降频 d
    未知。对每个这样的 n，由 ``speedup(n) = n·(1-(1-d)(n-1)/(p_cores-1))``
    反解 d，再取中位数（抗离群）。``p_cores≤1`` 时无降频概念（单核谈不上
    全核），返回 1.0。
    """
    if p_cores <= 1:
        return 1.0
    ds = []
    for n, t in measured_ms.items():
        if n < 2 or n > p_cores:
            continue  # 只用 P 核段
        if t <= 0 or single_core_ms <= 0:
            continue
        # speedup = n * (1 - (1-d)(n-1)/(p-1))  ⇒  1-d = (1 - speedup/n)·(p-1)/(n-1)
        speedup = single_core_ms / t
        one_minus_d = (1.0 - speedup / n) * (p_cores - 1) / (n - 1)
        ds.append(max(0.0, min(1.0, 1.0 - one_minus_d)))
    if not ds:
        raise ValueError(
            "没有可用于拟合 P 核降频比的实测点（需要 2≤n≤p_cores 的条目）"
            f"：p_cores={p_cores}, 实测 n={sorted(measured_ms)}"
        )
    ds.sort()
    return ds[len(ds) // 2]  # 中位数，抗离群


def fit_e_core_efficiency(
    single_core_ms: float,
    measured_ms: dict,
    *,
    p_cores: int,
    e_cores: int,
    p_core_all_ratio: float = 0.0,
) -> float:
    """从一组实测 ``{n: ms}`` **拟合** E 核调度效率 r（区间 [0.05,1]）。

    **为什么单参数可拟合**：模型 ``T(n) = T1 / (p_used·p_ratio + e_used·r)``
    在 ``n>p_cores`` 段，``p_used=p_cores`` 是常数、``p_ratio`` 也固定为降频比
    ``d``——只有 r 未知，是 **T(n) 的线性化**：对每个 n 解出 r、取中位数
    （抗离群）。只用 P 核的 n（``n≤p_cores``）**不含 r**，故被跳过。

    ``p_core_all_ratio``：P 核降频比 d（默认 0=旧口径、不建模降频）。若 >0，
    则 P 核段的贡献按 ``p_cores·d`` 计（而非 ``p_cores·1``）——降频与 E 核效率
    是**两个独立的效应**，混在一起会把"P 核多核降频"错算成"E 核很弱"
    （旧口径 r≈0.19 正因此低估了 E 核：分离降频后 r≈0.44）。

    **这是"预测"的关键一步**：单核延迟给了 T1，一组便宜的实测给了 r（与 d），
    此后就能**预测任意 n**（包括没测过的），而不必逐个实测——正合本机"少跑进程"
    的约定。
    """
    rs = []
    for n, t in measured_ms.items():
        if n <= p_cores:
            continue  # 不含 r
        if t <= 0 or single_core_ms <= 0:
            continue
        e_used = n - p_cores
        if e_used <= 0:
            continue
        # T(n) = T1 / (p_cores·d + e_used·r)  ⇒  p·d + e·r = T1/T(n)
        #   ⇒  r = (T1/T(n) - p·d)/e
        speedup = single_core_ms / t
        p_contrib = p_cores * (p_core_all_ratio if p_core_all_ratio > 0 else 1.0)
        r = (speedup - p_contrib) / e_used
        rs.append(max(0.05, min(1.0, r)))
    if not rs:
        raise ValueError(
            "没有可用于拟合的实测点（需要 n > p_cores 的条目）"
            f"：p_cores={p_cores}, 实测 n={sorted(measured_ms)}"
        )
    rs.sort()
    return rs[len(rs) // 2]  # 中位数，抗离群


def main() -> int:
    """命令行入口。用法::

        python -c "from nano.model.chip_model import main; main()"

    注意：``python -m nano.model.chip_model`` 的可用性取决于 ``nano/`` 是否在 ``sys.path``
    （它不在 ``src/`` 内）。
    """
    est = estimate(M4_REFERENCE, n_tubes=64, n_points=200_000)
    print(est.explain())
    print()
    print("实测（M4, nano_kernel 单线程）≈ 1.0 ns/pair（见 docs/PROJECT.md §2.1）")
    print("**注意口径**：M4_REFERENCE 是**全芯片**算力，1.0 ns/pair 是**单线程**。")
    print("两者直接相除没有意义——按口径折算后的对比见下：")
    single = measured_vs_model(
        M4_REFERENCE, 1.0, 64,
        measured_is_single_thread=True,
        single_core_flops_gflops=M4_P_CORE_GFLOPS,
    )
    print(f"  单 P 核（{M4_P_CORE_GFLOPS} GFLOP/s，与实测同口径） "
          f"上限 {single['ceiling_ns_per_pair']:.4f} ns/pair "
          f"⇒ 实测/上限 = {single['headroom_x']:.2f}×")
    full = measured_vs_model(M4_REFERENCE, 1.0, 64, measured_is_single_thread=False)
    print(f"  全芯片（{M4_REFERENCE.peak_flops_gflops:.0f} GFLOP/s）        "
          f"上限 {full['ceiling_ns_per_pair']:.4f} ns/pair "
          f"⇒ 实测/上限 = {full['headroom_x']:.2f}×（口径错配，勿直接引用）")

    print()
    print("=== 调度预测：由单核墙钟预测多线程墙钟（示例）===")
    # 用本机实测联合拟合的 E 核调度效率与 P 核降频比演示：给定单核 51.4ms。
    r = 0.44
    d = M4_REFERENCE.p_core_all_ratio
    pred = predict_schedule(
        51.37, p_cores=M4_REFERENCE.p_cores, e_cores=M4_REFERENCE.e_cores,
        e_core_efficiency=r, p_core_all_ratio=d,
    )
    print(pred.explain())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
