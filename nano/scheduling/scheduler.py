"""动态调度器：把**一个提问**拆成**子问题**，逐个子问题决定"调度哪些模块"。

**用户的意图**（2026-10-07 定调）："这是一个**动态**的调度——要判断每一个提问、
或者说每一个提问里面的**子问题**，它需要**调度哪一些模块**。"

**与既有模块的分工**（避免重复）：

* ``src/kakeya_ai/ai/expert_router.py`` —— 把提问路由到**LLM 专家**（prompt 侧）。
  那是"用哪个模型回答"，**不是**本模块。
* 本模块（``nano/scheduling/scheduler.py``）—— 把提问拆成子问题，为每个子问题决定
  **调哪些"计算模块"**（几何内核 / 加速后端）以及**用多少资源**（后端、线程数）。
  它**消费**前两步的成果：``chip_model`` 判瓶颈、``predict_schedule`` 选线程数。

**为什么放 ``nano/``**：它属于"芯片/调度"这条线（与 chip_model 同源），不是
几何/AI 运行时组件。

**核心结构**：

1. **子问题分解**（:func:`decompose_query`）：按**启发式关键词**把提问映射到若干
   :class:`SubQuestion`——每个带一个 :class:`ModuleKind`（它需要哪类计算）。
   与 expert_router 的"关键词路由"同一思路（透明、可测、无 embedding 依赖）。
   子问题按触发词在提问里的**位置**排序（保留"先…再…"的先后），位置记入
   ``SubQuestion.pos`` 供审计；紧邻否定标记（"不要/别用/排除/除了"）会**排除**
   对应模块（"不要蒙特卡洛"不再调度 MC）。
2. **模块选择**（:meth:`ModuleKind` 的语义）：如 "volume"→体素/蒙特卡洛体积、
   "contains"→批量点包含、"grains"→管分解、"proof"→Move/编排。
3. **资源分配**（:func:`plan_query`）：对每个子问题，用 ``chip_model`` 判它**算力
   还是带宽受限**、用 ``predict_schedule`` 算出**建议线程数**——于是"调度"不只是
   "选模块"，还包括"这个模块该用几个核跑"。单核墙钟有三个来源，**优先级**：
   每模块显式（``single_core_ms_by_module``）> 全局标量（``single_core_ms``）
   > **规模模型推算**（``sizes=...`` + ``chip_model.estimate_single_core_ms``，
   由"几管 × 几点"推出，2026-10-30 补——不再依赖调用方拍脑袋给 ms）。支持**留核**
   （``reserve_cores``，给 Ollama 等留核）与 **E 核调度效率参数化**
   （``e_core_efficiency``，默认本机实测 0.44）。
4. **执行衔接**（:data:`MODULE_REGISTRY` / :func:`resolve_module`）：计划里的每个
   :class:`ModuleKind` 都带一个**真实模块与入口**（如 ``volume_voxel`` →
   ``kakeya_ai.volume.estimate_union_volume_voxel``）——静态映射、**不 import**
   主包，避免 nano（研发侧）反向依赖运行时。需要"能否 import"时用
   :func:`verify_registry`（``find_spec``，不执行模块）。
5. **执行顺序**（:func:`compute_execution_order` / :attr:`SchedulePlan.execution_order`）：
   把扁平的 ``tasks`` 排成**拓扑层**（:class:`ExecutionOrder` 的 ``waves``），
   并按**声明式依赖**（:data:`MODULE_DEPENDENCIES`，目前为空）与**核预算**
   判定每层**并行还是串行**——这是"跨子问题并发/依赖调度"的落点：两个纯计算
   子问题各要满核时会被判为**串行**（避免超订），而 host/IO 子问题可自由重叠。

**纪律**：本模块**不做**计算本身（计算在 ``accel`` / 各 kernel）；它只产出
**计划**（:class:`SchedulePlan`）——"哪些模块、按什么顺序、用什么资源"。这样
调度逻辑可单测、可审计，而执行仍走既有路径。执行衔接的静态映射
（:data:`MODULE_REGISTRY`）让计划**可自证"模块真实存在"**，但**不 import**
运行时代码。
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..model.chip_model import (
    ChipSpec,
    estimate_single_core_ms,
    predict_schedule,
)
from ..runtime.resource_arbiter import ResourceGrant

__all__ = [
    "ModuleKind",
    "ModuleTarget",
    "ProblemSize",
    "SubQuestion",
    "TaskPlan",
    "SchedulePlan",
    "ExecutionWave",
    "ExecutionOrder",
    "decompose_query",
    "plan_query",
    "compute_execution_order",
    "resolve_module",
    "verify_registry",
    "DEFAULT_E_CORE_EFFICIENCY",
    "DEFAULT_P_CORE_ALL_RATIO",
    "MODULE_REGISTRY",
    "MODULE_DEPENDENCIES",
]


class ModuleKind(str, Enum):
    """一个子问题需要的**计算模块类别**。

    取值为字符串枚举，便于打印与序列化（与 ``resource_budget`` 的可审计口径一致）。
    """

    VOLUME_VOXEL = "volume_voxel"      # 体素网格并集体积（确定性、可逐位核对）
    VOLUME_MC = "volume_mc"            # 蒙特卡洛并集体积（带噪、可给 CI）
    CONTAINS = "contains"             # 批量"点是否在管族内"
    GRAINS = "grains"                  # 管族分解（grains）
    MOVES = "moves"                    # 圣言 Move（合并/压缩）
    PROOF = "proof"                    # 端到端证明编排（无/有 LLM）


@dataclass(frozen=True)
class ModuleTarget:
    """一个 :class:`ModuleKind` 对应的**真实模块与入口**（执行衔接的静态映射）。

    **为什么放这里、且是静态数据**：本模块只产出计划、不执行计算；但计划里写
    的 ``ModuleKind`` 若没有对应的真实代码，读者无法自证"调度的模块真的存在"。
    这个映射把 ``ModuleKind → 模块名 + 入口函数名`` 钉成**可审计的静态事实**
    （数据、非 import）——nano（研发侧）**不 import** 主包运行时，避免反向依赖；
    需要"是否可 import"时用 :func:`verify_registry`（``importlib.util.find_spec``，
    不执行模块，按需调用）。
    """

    #: 模块名（如 ``kakeya_ai.volume``）。**不是** import——只是名字。
    module: str
    #: 入口函数名（如 ``estimate_union_volume_voxel``）。
    entry: str
    #: 人类可读说明（它做什么）。
    note: str = ""


#: ``ModuleKind → 真实模块与入口`` 的静态映射。**只记录名字、不 import**——
#: 计划据此自证"调度的模块真实存在、入口叫什么"，执行仍走既有路径（accel/kernel）。
#: 口径来源：主包 ``src/kakeya_ai/`` 的实际模块（2026-10-07 核对）。
MODULE_REGISTRY: dict[ModuleKind, ModuleTarget] = {
    ModuleKind.VOLUME_VOXEL: ModuleTarget(
        "kakeya_ai.volume", "estimate_union_volume_voxel",
        "体素网格并集体积（确定性）"),
    ModuleKind.VOLUME_MC: ModuleTarget(
        "kakeya_ai.volume", "estimate_union_volume_monte_carlo_with_ci",
        "蒙特卡洛并集体积（带置信区间）"),
    ModuleKind.CONTAINS: ModuleTarget(
        "kakeya_ai.accel", "contains_points",
        "批量点包含（nano→fast→numpy 统一后端）"),
    ModuleKind.GRAINS: ModuleTarget(
        "kakeya_ai.grains", "partition_into_rho_tubes",
        "管族 ρ 分解（grains）"),
    ModuleKind.MOVES: ModuleTarget(
        "kakeya_ai.moves", "apply_moves",
        "圣言 Move 合并/压缩"),
    ModuleKind.PROOF: ModuleTarget(
        "kakeya_ai.proof_orchestrator", "run_proof_skeleton",
        "端到端证明编排骨架"),
}


#: 子问题间的**声明式依赖**（``ModuleKind → 必须先完成的 ModuleKind 集合``）。
#: 目前为空：每个子问题都是对"已就绪的管族"的**原子操作**，彼此独立（无跨模块
#: 流水线）。机制已备——将来引入跨模块流水线（如 grains→moves→volume）时在此
#: 声明，:func:`compute_execution_order` 会自动据此做拓扑分层。被依赖的模块
#: **不在**计划里时视为"已就绪/由外部提供"，不会报错。
MODULE_DEPENDENCIES: dict[ModuleKind, frozenset[ModuleKind]] = {}


def resolve_module(kind: ModuleKind) -> ModuleTarget:
    """返回某 :class:`ModuleKind` 对应的真实模块与入口（静态映射，无 import）。

    未登记的类型抛 ``KeyError``——计划里出现"没登记的计算模块"应立即暴露，
    而不是静默给个空目标。
    """
    return MODULE_REGISTRY[kind]


def verify_registry(
    kinds: Optional[list[ModuleKind]] = None,
) -> dict[ModuleKind, bool]:
    """核对每个（或指定）``ModuleKind`` 的真实模块**是否可被 ``find_spec`` 解析**。

    **只 ``find_spec``、不 import 模块本身**（不执行运行时代码）——所以是"文件
    存在且包可解析"的轻量核对，而非"能跑"。**口径**：需要主包 ``kakeya_ai``
    在本进程的 ``sys.path`` 上（pytest 通过 ``pyproject`` 的 ``pythonpath=src``
    提供；裸 ``python`` 需自行加 ``src/``）。解析不到 → 该类型记 ``False``
    （诚实报告"本进程看不到"，而非假装存在）。
    """
    targets = kinds if kinds is not None else list(MODULE_REGISTRY)
    result: dict[ModuleKind, bool] = {}
    for k in targets:
        t = MODULE_REGISTRY[k]
        result[k] = importlib.util.find_spec(t.module) is not None
    return result


#: 每个模块的"计算特征"：算术强度大致量级 + 是否可确定性并行。
#: 用于**选后端与线程数**（不是精确性能模型——那是 chip_model 的职责）。
@dataclass(frozen=True)
class ModuleProfile:
    kind: ModuleKind
    #: 该模块是否**算力受限**（否则视为带宽/IO 受限）。取自 `nano/model/chip_model.py`
    #: 的 roofline 结论：tube 几何算子 I≈133 ≫ I*≈1.17 ⇒ 算力受限。
    compute_bound: bool
    #: 是否可安全并行（逐位一致）。都走"整型计数归约" ⇒ 可。
    parallel_safe: bool
    #: 人类可读说明。
    note: str = ""


#: 模块画像。**口径来源**：`chip_model` 判定 tube 几何算子算力受限（见其 §3），
#: 且 `cpp/kernel/parallel.hpp` 已证并行归约逐位一致 ⇒ 这几类都 parallel_safe。
_MODULE_PROFILES = {
    ModuleKind.VOLUME_VOXEL: ModuleProfile(ModuleKind.VOLUME_VOXEL, True, True,
                                          "体素计数（确定性、可逐位核对）"),
    ModuleKind.VOLUME_MC: ModuleProfile(ModuleKind.VOLUME_MC, True, True,
                                        "蒙特卡洛计数（带噪，可给 CI）"),
    ModuleKind.CONTAINS: ModuleProfile(ModuleKind.CONTAINS, True, True,
                                       "批量点包含（计数归约）"),
    ModuleKind.GRAINS: ModuleProfile(ModuleKind.GRAINS, True, False,
                                     "管分解（目前未并行化）"),
    ModuleKind.MOVES: ModuleProfile(ModuleKind.MOVES, True, False,
                                    "Move 合并（依赖几何条件，非纯计数）"),
    ModuleKind.PROOF: ModuleProfile(ModuleKind.PROOF, True, False,
                                    "端到端编排（含 IO/LLM，非纯计算）"),
}


#: 默认 E 核**调度效率**（相对 P 核=1）。**口径**：由本机实测 (n,T(n)) 曲线
#: **分离 P 核降频后**联合拟合得出（见 ``chip_model.fit_schedule_params``），
#: 是"调度效率"——与 ``ChipSpec.e_core_rate_ratio``（理论算力比 ~0.30）**不是
#: 同一个量**，勿混用：前者回答"加 E 核调度能快多少"（含内存/争用效应），
#: 后者回答"E 核算力多强"。旧口径（不建模降频）是 0.19，它把"P 核多核降频"
#: 错算成"E 核很弱"；分离降频后为 0.44。
DEFAULT_E_CORE_EFFICIENCY = 0.44

#: 默认 **P 核全核/单核时钟比**（降频比）。**口径**：由本机实测 (n,T(n)) 曲线
#: 联合拟合（见 ``chip_model.fit_schedule_params``）——M4 单核 boost 4.4 GHz，
#: 4 个 P 核满载时约 81%（≈3.6 GHz），典型的功耗/热包络降频。0 = 不建模降频。
DEFAULT_P_CORE_ALL_RATIO = 0.81


@dataclass(frozen=True)
class SubQuestion:
    """一个子问题：它是什么 + 它需要哪类计算。"""

    text: str
    module: ModuleKind
    #: 该子问题在原文里的触发词（可审计"为什么选了它"）。
    trigger: str = ""
    #: 触发词在提问（小写后）里的位置——审计"它在原文哪一处"，并据此保持
    #: 提问中"先…再…"的先后顺序。
    pos: int = -1


#: 关键词 → 模块。**启发式、透明、可测**（与 expert_router 同一风格）；
#: 刻意不引 embedding（本项目已验证当前 embedding 路线不可行，见局限 #8）。
#: **触发词取该模块所有关键词在提问里的最早出现**（便于否定判断与按位置排序）；
#: 子问题输出顺序按触发位置排（见 :func:`decompose_query`）——保留"先…再…"。
_MODULE_KEYWORDS: list[tuple[tuple[str, ...], ModuleKind]] = [
    (("体积", "volume", "|∪", "并集", "measure"), ModuleKind.VOLUME_VOXEL),
    (("蒙特卡洛", "monte", "mc", "采样", "sample"), ModuleKind.VOLUME_MC),
    (("包含", "contains", "是否在", "命中", "mask"), ModuleKind.CONTAINS),
    (("grains", "颗粒", "分解", "partition", "rho"), ModuleKind.GRAINS),
    (("move", "压缩", "合并", "sticky", "扩大"), ModuleKind.MOVES),
    (("证明", "prove", "proof", "编排", "orchestrat", "下界"), ModuleKind.PROOF),
]


#: 否定标记：出现在某模块**最早触发词紧邻前方**时，该模块被排除
#: （"不要蒙特卡洛""别用 mc""排除采样""除了体积"都不再调度对应模块）。
#: 只做**紧邻**否定（窗口 ≤2 字符），不引 embedding、不做句法分析——与
#: "关键词启发式"同一口径；更远的否定（"其实我并不想算体积"）不在本层处理。
_NEGATION_MARKERS = ("不要", "不用", "别用", "排除", "除了")


def _is_negated(q: str, pos: int) -> bool:
    """触发词在 ``pos`` 处是否被紧邻的否定标记修饰。

    只看 ``q[pos-2:pos]`` 的 2 字符窗口（覆盖"不要X/不用X/别用X/排除X/除了X"），
    足够覆盖常见显式否定，且不误伤单独的"不"字（"不"本身不构成排除）。
    """
    if pos < 1:
        return False
    prefix = q[max(0, pos - 2):pos]
    return any(m in prefix for m in _NEGATION_MARKERS)


def decompose_query(query: str) -> list[SubQuestion]:
    """把提问按关键词拆成子问题（每个带一个模块）。

    **动态**的一面：拆分结果**取决于提问内容**——问"算体积并验证包含"会得到
    ``[VOLUME_VOXEL, CONTAINS]`` 两个子问题；问"跑证明"得到 ``[PROOF]``。

    **顺序**：子问题按触发词在提问里的**位置**排序（不是关键词表顺序）——
    "先检查包含、再算体积"得到 ``[CONTAINS, VOLUME_VOXEL]``，保留提问表述的
    先后（调用方据此知道先做哪个）。

    **重叠关键词的取舍**：如"蒙特卡洛并集体积"同时命中 VOLUME_MC 与
    VOLUME_VOXEL，**两者都保留**（各记一次）——这是刻意的：体素口径确定、
    可逐位核对，蒙特卡洛口径带噪、可给 CI，两个口径并存正是本项目
    "确定 + 统计"双轨验证的做法；是否都要算由调用方决定。

    **显式否定**：紧邻否定标记（"不要/不用/别用/排除/除了"）会**排除**被修饰
    的模块——"不要蒙特卡洛"不调度 MC、"别用 mc"不调度 MC、"排除采样"不调度 MC。

    若没有任何关键词命中（或被否定后为空），返回**空列表**——调用方据此知道
    "没有可调度的计算模块"（例如纯闲聊），而不是硬塞一个默认模块。
    """
    if not query or not query.strip():
        return []
    q = query.lower()
    subs: list[SubQuestion] = []
    for keywords, module in _MODULE_KEYWORDS:
        # 该模块所有关键词在提问里的出现位置（取最早的那个作触发词）。
        positions = [(q.find(kw), kw) for kw in keywords if q.find(kw) >= 0]
        if not positions:
            continue
        pos, kw = min(positions)
        if _is_negated(q, pos):
            continue  # "不要X/别用X/排除X/除了X" ⇒ 该模块被显式排除
        subs.append(SubQuestion(
            text=query.strip(), module=module, trigger=kw, pos=pos,
        ))
    # 按触发位置排：保留提问里"先…再…"的先后（稳定的 sort 不改变并列顺序）。
    subs.sort(key=lambda s: s.pos)
    return subs


@dataclass(frozen=True)
class ProblemSize:
    """一个子问题的**问题规模**——调度器据此从芯片模型推算单核墙钟。

    口径：``n_tubes`` = 管族管数；``n_points`` = "点"数，含义随模块而定
    （体素 = 网格采样数 ``grid_n³``、蒙特卡洛 = 采样数、包含 = 待测点数）。
    只有 **pair 级几何算子**（``VOLUME_VOXEL`` / ``VOLUME_MC`` / ``CONTAINS``）
    有 pair 计数模型、能从规模推算；``GRAINS`` / ``MOVES`` / ``PROOF`` 不是
    pair 计数口径 ⇒ 给了 ``sizes`` 也**忽略**（如实登记，不硬套公式）。
    """

    n_tubes: int
    n_points: int

    def __post_init__(self) -> None:
        if self.n_tubes < 1 or self.n_points < 1:
            raise ValueError(
                f"ProblemSize 的管数/点数必须为正："
                f"{self.n_tubes=} {self.n_points=}"
            )


@dataclass(frozen=True)
class TaskPlan:
    """单个子问题的**调度计划**：调哪个模块 + 用什么资源。"""

    sub: SubQuestion
    backend: str
    #: 建议线程数。**语义**：纯计算且给了单核墙钟 ⇒ ``predict_schedule`` 的最优 n；
    #: 纯计算但没给 ⇒ 芯片可调度核数（总核数 - 留核）；非并行模块 ⇒ 0
    #: （host 单线程，不给"看着像并行其实没并"的数字）。
    threads: int
    #: 该模块是否算力受限（决定优化方向）。
    compute_bound: bool
    #: 可审计说明。
    rationale: str
    #: 该模块对应的**真实模块与入口**（执行衔接，来自 :data:`MODULE_REGISTRY`）。
    #: 计划据此自证"调度的模块真实存在、入口叫什么"，而不是只有个枚举名。
    target: Optional[ModuleTarget] = None


@dataclass(frozen=True)
class SchedulePlan:
    """整个提问的调度计划。"""

    query: str
    tasks: list
    #: 未被任何模块匹配的提示（如空查询）。
    note: str = ""
    assumptions: list = field(default_factory=list)
    #: 可调度核数（总核数 - 留核，或——多租户下——上层发放的授权核数）。
    #: 执行顺序的核预算裁定依据；0 = 未知（手工构造计划时可能没算过）。
    available_cores: int = 0
    #: 多租户下**上层发放的资源授权**（:class:`~nano.runtime.resource_arbiter.ResourceGrant`）。
    #: ``None`` = 单租户独占模式（可调度核数由芯片拓扑减留核算出）。这是把本
    #: 调度器从"单租户独占规划器"变成"**租户内规划器**"的落点——见
    #: :func:`plan_query` 的 ``grant`` 参数。
    grant: Optional[ResourceGrant] = None

    @property
    def execution_order(self) -> "ExecutionOrder":
        """本计划的**执行顺序**：把扁平 ``tasks`` 排成拓扑层、判定并行/串行。

        惰性计算，见 :func:`compute_execution_order`——这是"跨子问题并发/依赖
        调度"的落点（此前只按提问先后排平序、不判并发）。
        """
        return compute_execution_order(self)

    def explain(self) -> str:
        if not self.tasks:
            return f"提问: {self.query!r}\n  （无可调度的计算模块）{self.note}"
        lines = [f"提问: {self.query!r}", f"  拆出 {len(self.tasks)} 个子问题："]
        for i, t in enumerate(self.tasks, 1):
            lines.append(
                f"    {i}. [{t.sub.module.value}] 触发词={t.sub.trigger!r}"
                f"@{t.sub.pos} → 后端={t.backend} 线程={t.threads} "
                f"({'算力受限' if t.compute_bound else '非算力受限'})"
            )
            if t.target is not None:
                lines.append(f"       执行: {t.target.module}.{t.target.entry}")
            lines.append(f"       理由: {t.rationale}")
        lines.append(self.execution_order.explain())
        lines.append("假设:")
        lines.extend(f"  - {a}" for a in self.assumptions)
        return "\n".join(lines)


@dataclass(frozen=True)
class ExecutionWave:
    """执行顺序里的**一步**：若干子问题及其并行/串行安排。

    语义：
    * ``parallel=True``：``task_indices`` 里的任务**可同时跑**，各自用
      ``threads_per_task`` 的线程数（核预算已按并发摊过）。
    * ``parallel=False``：``task_indices`` 里的任务**一个接一个跑**（串行），
      各自仍用 ``threads_per_task`` 的满线程数——这是"并发会超订核预算"时的
      正确取舍（宁可串行，也不在 10 核上开出 2×9=18 线程的超订计划）。
    """

    #: 本步要执行的任务下标（指向 :class:`SchedulePlan`.tasks，0 基）。
    task_indices: tuple
    #: 是否并行执行（True=同时跑；False=串行、一次一个）。
    parallel: bool
    #: 每个任务的建议线程数（与 ``task_indices`` 对齐）。
    threads_per_task: tuple
    #: 可审计说明（为什么这样安排）。
    rationale: str


@dataclass(frozen=True)
class ExecutionOrder:
    """整个计划的执行顺序：由扁平 ``tasks`` 排成的**拓扑层**（waves）。

    来源：:func:`compute_execution_order`——先按**声明式依赖**做拓扑分层，
    再按**核预算**决定每层是并行还是串行（见 :class:`ExecutionWave`）。
    """

    waves: tuple
    #: 是否存在"真正可并行"的一步（>1 个任务且 ``parallel=True``）。
    has_parallelism: bool

    def explain(self) -> str:
        if not self.waves:
            return "  （无可执行步骤）"
        lines = ["  执行顺序（拓扑层 → 并行/串行）："]
        for k, w in enumerate(self.waves, 1):
            idx = ", ".join(f"#{i + 1}" for i in w.task_indices)
            if len(w.task_indices) == 1:
                mode = "—"
            elif w.parallel:
                mode = "并行"
            else:
                mode = "串行"
            thr = ", ".join(str(t) for t in w.threads_per_task)
            lines.append(f"    第 {k} 步: 任务[{idx}] {mode} 线程=({thr}) —— {w.rationale}")
        return "\n".join(lines)


def compute_execution_order(
    plan: SchedulePlan,
    *,
    available_cores: Optional[int] = None,
    dependencies: Optional[dict] = None,
) -> ExecutionOrder:
    """把 :class:`SchedulePlan` 的扁平 ``tasks`` 排成**可执行的拓扑层**。

    两步：
    1. **拓扑分层**：按 ``dependencies``（默认 :data:`MODULE_DEPENDENCIES`）把
       有依赖的子问题分层（Kahn 算法）。目前依赖表为空 ⇒ 所有子问题同层
       （独立）；被依赖的模块不在计划里时视为"已就绪"，不产生边。
    2. **核预算裁定**：对每层，算其中各任务线程需求之和；若超过
       ``available_cores``（默认取 ``plan.available_cores``，即总核数-留核），
       则该层**串行**（``parallel=False``，每个任务用满自己的线程数），否则
       **并行**（``parallel=True``）。线程数 0 的任务（host/IO 或非并行模块）
       不占内核预算、可自由重叠。

    **这是"跨子问题并发/依赖调度"的落点**（此前只按提问先后排平序、不判
    并发）：它让计划诚实回答"这几个子问题该并行跑还是串行跑、各用几核"，
    而不是每个子问题各自拿到满核数（那会在 10 核芯片上开出 2×9=18 线程的
    超订计划）。

    **注意**：仍是**只调度、不执行**——本函数产出"顺序与并发"的**计划**，
    执行仍走 ``accel`` / 各 kernel。
    """
    tasks = plan.tasks
    n = len(tasks)
    if n == 0:
        return ExecutionOrder(waves=(), has_parallelism=False)

    deps = dependencies if dependencies is not None else MODULE_DEPENDENCIES
    avail = available_cores if available_cores is not None else plan.available_cores

    # 依赖边：模块 m 依赖的每个前置模块 → 该模块的**所有**出现（下标）。
    by_module: dict = {}
    for i, t in enumerate(tasks):
        by_module.setdefault(t.sub.module, []).append(i)
    indegree = [0] * n
    children: list = [[] for _ in range(n)]
    for i, t in enumerate(tasks):
        for dep_kind in deps.get(t.sub.module, ()):
            for j in by_module.get(dep_kind, []):
                if j != i and i not in children[j]:
                    children[j].append(i)
                    indegree[i] += 1

    # Kahn 分层：每层 = 当前入度为 0 的节点；环则取入度最小的破环（不丢任务）。
    levels: list = []
    queue = [i for i in range(n) if indegree[i] == 0]
    done = 0
    while done < n:
        if not queue:
            remain = [i for i in range(n) if indegree[i] >= 0]
            level = [min(remain, key=lambda x: indegree[x])]
        else:
            level = list(queue)
        queue = []
        for i in level:
            indegree[i] = -1
            done += 1
            for j in children[i]:
                indegree[j] -= 1
                if indegree[j] == 0:
                    queue.append(j)
        levels.append(sorted(level))

    waves: list = []
    for level in levels:
        idxs = level
        threads = tuple(tasks[i].threads for i in idxs)
        demand = sum(tasks[i].threads for i in idxs)
        if len(idxs) == 1:
            parallel = True
            why = f"单任务，用 {threads[0]} 线程"
        elif avail > 0 and demand > avail:
            parallel = False
            why = (
                f"并发线程需求 {demand} > 可调度 {avail} 核 ⇒ 串行"
                "（各自用满线程，避免超订）"
            )
        else:
            parallel = True
            why = (
                f"并发线程需求 {demand} ≤ 可调度 {avail} 核 ⇒ 可并行"
                if avail > 0
                else "核预算未知 ⇒ 假设可并行（不判超订）"
            )
        waves.append(ExecutionWave(
            task_indices=tuple(idxs),
            parallel=parallel,
            threads_per_task=threads,
            rationale=why,
        ))

    has_parallelism = any(
        w.parallel and len(w.task_indices) > 1 for w in waves
    )
    return ExecutionOrder(waves=tuple(waves), has_parallelism=has_parallelism)


def _choose_backend(
    prof: ModuleProfile, nano_available: bool
) -> str:
    """选后端。

    * **纯计算的并行算子**（VOLUME_*/CONTAINS）且 f32 内核可用 ⇒ ``nano``
      （快、≈1ns/pair）——口径：`chip_model` 判定 tube 几何算子算力受限。
    * **非 GPU 内核类**（GRAINS/MOVES/PROOF）⇒ ``"host"``：它们不是"调哪个
      几何 kernel"的问题，而是主机上的编排/图算法，不套内核后端。
    """
    if not prof.parallel_safe:
        return "host"
    if nano_available and prof.compute_bound:
        return "nano"
    return "fast" if nano_available else "numpy"


def _lookup_size(sizes: dict, module: ModuleKind) -> Optional[ProblemSize]:
    """查 ``sizes`` 里某模块的规模（键容忍 ``ModuleKind`` 或其 ``.value`` 串）。

    未登记返回 ``None``——调用方据此回退到默认行为（用满可调度核数）。
    """
    if module in sizes:
        return sizes[module]
    return sizes.get(module.value)


def plan_query(
    query: str,
    *,
    chip: Optional[ChipSpec] = None,
    single_core_ms: Optional[float] = None,
    single_core_ms_by_module: Optional[dict] = None,
    sizes: Optional[dict] = None,
    measured_ns_per_pair: Optional[float] = None,
    model_efficiency: float = 1.0,
    e_core_efficiency: float = DEFAULT_E_CORE_EFFICIENCY,
    p_core_all_ratio: Optional[float] = None,
    reserve_cores: int = 0,
    nano_available: bool = True,
    grant: Optional[ResourceGrant] = None,
) -> SchedulePlan:
    """**动态调度**：拆提问 → 逐子问题选模块与资源。

    参数
    ----
    query : 用户提问。
    chip : 芯片参数（默认 M4 参考）。用于查拓扑（P/E 核数）与降频比。
    single_core_ms : 该子问题在单核上的**实测/估计**墙钟（毫秒），**所有纯计算
        子问题共用**的默认值。若给了，就用 ``predict_schedule`` 算出**建议线程
        数**；没给则线程=芯片可调度核数（默认全核）。**这是"预测模型"接入调度
        的地方**。
    single_core_ms_by_module : **每模块**独立单核墙钟（``{ModuleKind: ms}``，
        键也可是 ``ModuleKind.value`` 的字符串）。优先于 ``single_core_ms``——
        不同模块成本不同时用它（如 volume 和 contains 的每对耗时不同）。
    sizes : **问题规模**（``{ModuleKind: ProblemSize}``，键也可是 ``.value``
        字符串）——给了它，且没给显式 ``single_core_ms*`` 时，纯计算子问题的
        单核墙钟由 ``chip_model.estimate_single_core_ms`` **从规模推算**
        （"几管 × 几点" → 单核 ms → 建议线程数），不再依赖调用方拍脑袋。
        只对 **pair 级几何算子**（volume_voxel/volume_mc/contains）生效；
        grains/moves/proof 无 pair 计数模型，给了也**忽略**（如实登记）。
    measured_ns_per_pair : ``nano/isolated/bench_micro`` 的**单线程实测**
        ns/pair（本机 ≈1.0）。传给 ``sizes`` 推算路径做校准——最诚实（已含
        编译器/访存效应）；不传则用 roofline 上限 × ``model_efficiency``。
    model_efficiency : roofline 口径的**达成率**（0,1]，默认 1 = "理论最好"
        （乐观）。上限是 FLOP 上界计数口径（见 chip_model 假设），做诚实的
        计划应按"实测/上限之比"（本机 ≈1.4 的倒数 ⇒ 0.7 左右）给 <1 的值，
        或直接传 ``measured_ns_per_pair``。只在没有实测值时参与。
    e_core_efficiency : E 核调度效率（默认本机实测 0.44，见
        ``DEFAULT_E_CORE_EFFICIENCY``）。换芯片时传入该芯片的实测值。
    p_core_all_ratio : P 核全核/单核降频比（默认**取芯片的** ``chip.p_core_all_ratio``，
        无 chip 时回退 ``DEFAULT_P_CORE_ALL_RATIO``）。0 = 不建模降频。
    reserve_cores : **留核**数（如给 Ollama 留 2 核）。可调度核数 = 总核数 -
        留核数；线程建议不会超过它。**仅在单租户（``grant is None``）时生效**。
    nano_available : f32 内核是否可用（决定能否选 ``nano`` 后端）。
    grant : **多租户模式**下由上层（集群放置 / 节点仲裁，见
        :mod:`nano.runtime.resource_arbiter`）发放的 :class:`ResourceGrant`。给了它，
        可调度核数就从"芯片拓扑减留核"**改为"授权核数"**——于是本调度器成为
        **租户内规划器**，在"我这一份"里拆子问题、定并行/串行，而不会越权去
        用别的租户的核。``None``（默认）= 单租户独占（现有行为不变）。

    返回 :class:`SchedulePlan`——**只是计划，不执行计算**。
    """
    from ..model.chip_model import M4_REFERENCE  # 延迟导入，便于测试替换 chip

    c = chip if chip is not None else M4_REFERENCE
    total = c.p_cores + c.e_cores
    if grant is not None:
        # 多租户：核预算来自上层授权（真源），与芯片拓扑/留核解耦。
        available = grant.cores
    else:
        if reserve_cores < 0 or (total > 0 and reserve_cores >= total):
            raise ValueError(
                f"reserve_cores 须在 [0, 总核数-1]：reserve={reserve_cores} total={total}"
            )
        available = total - reserve_cores if total > 0 else 0
    if not (0.0 < e_core_efficiency <= 1.0):
        raise ValueError(f"e_core_efficiency 须在 (0,1]：{e_core_efficiency}")
    if measured_ns_per_pair is not None and measured_ns_per_pair <= 0:
        raise ValueError(f"measured_ns_per_pair 须为正：{measured_ns_per_pair}")
    if not (0.0 < model_efficiency <= 1.0):
        raise ValueError(f"model_efficiency 须在 (0,1]：{model_efficiency}")
    if sizes is not None:
        valid_keys = set(ModuleKind) | {m.value for m in ModuleKind}
        for k, v in sizes.items():
            if k not in valid_keys:
                raise ValueError(
                    f"sizes 的键必须是 ModuleKind 或其 .value：{k!r}"
                )
            if not isinstance(v, ProblemSize):
                raise ValueError(
                    f"sizes[{k!r}] 必须是 ProblemSize，得到 {type(v).__name__}"
                )
    droop = (
        p_core_all_ratio if p_core_all_ratio is not None
        else (c.p_core_all_ratio if c.p_core_all_ratio > 0 else DEFAULT_P_CORE_ALL_RATIO)
    )
    if droop != 0.0 and not (0.0 < droop <= 1.0):
        raise ValueError(f"p_core_all_ratio 须为 0（不建模）或 (0,1]：{droop}")

    subs = decompose_query(query)

    tasks: list[TaskPlan] = []
    for sub in subs:
        prof = _MODULE_PROFILES[sub.module]
        backend = _choose_backend(prof, nano_available)

        # 单核墙钟三个来源，优先级：每模块显式 > 全局标量 > 规模模型推算。
        t1: Optional[float] = None
        t1_source = ""
        if single_core_ms_by_module and sub.module in single_core_ms_by_module:
            t1 = single_core_ms_by_module[sub.module]
            t1_source = "每模块显式"
        elif single_core_ms is not None:
            t1 = single_core_ms
            t1_source = "全局显式"
        elif sizes is not None and prof.parallel_safe and prof.compute_bound:
            # 只有 pair 级几何算子有规模模型；grains/moves/proof 跳过（如实登记）。
            size = _lookup_size(sizes, sub.module)
            if size is not None:
                t1 = estimate_single_core_ms(
                    size.n_tubes, size.n_points, chip=c,
                    measured_ns_per_pair=measured_ns_per_pair,
                    efficiency=model_efficiency,
                )
                t1_source = f"规模 {size.n_tubes} 管×{size.n_points} 点推算"

        threads = 0
        why = []
        if not prof.parallel_safe:
            # 该模块尚未可安全并行（如 GRAINS/MOVES/PROOF）——不建议线程数，
            # 避免"看着像并行、其实没并"的误导（host 单线程）。
            why.append("该模块未并行化（非纯计数归约）⇒ 不设线程数（host 单线程）")
        elif t1 is not None and available > 0:
            pred = predict_schedule(
                t1, p_cores=c.p_cores, e_cores=c.e_cores,
                e_core_efficiency=e_core_efficiency,
                p_core_all_ratio=droop,
                max_threads=available,
            )
            threads = pred.best_n
            why.append(
                f"由{t1_source}单核 {t1:.1f}ms 预测最优 {pred.best_n} 线程"
                f"（{pred.best_ms:.1f}ms）"
            )
        elif available > 0:
            threads = available
            why.append(f"未给单核墙钟 ⇒ 用芯片默认全核（可调度 {available} 核）")
        else:
            why.append("芯片无可用核 ⇒ 不设线程数")

        why.append(
            f"算子{'算力受限' if prof.compute_bound else '非算力受限'}"
            f" ⇒ 后端 {backend}（{prof.note}）"
        )
        tasks.append(TaskPlan(
            sub=sub, backend=backend, threads=threads,
            compute_bound=prof.compute_bound, rationale="；".join(why),
            target=MODULE_REGISTRY.get(sub.module),
        ))

    note = "" if tasks else "（提问里没有匹配到计算模块的关键词）"
    if grant is not None:
        core_line = (
            f"多租户：核预算来自授权（租户 {grant.tenant!r}，授权 {grant.cores} 核）"
        )
    elif reserve_cores:
        core_line = f"留核 {reserve_cores} 核（可调度 {available} 核）"
    else:
        core_line = "不留核（用满全芯片）"
    if sizes is not None:
        ns_pair = (
            f"实测 {measured_ns_per_pair} ns/pair"
            if measured_ns_per_pair
            else f"roofline × 达成率 {model_efficiency:.2f}"
        )
        size_line = (
            f"规模感知：{len(sizes)} 个模块的规模经 chip_model 推算单核墙钟"
            f"（{ns_pair}）"
        )
    else:
        size_line = (
            "未给问题规模（sizes）⇒ 单核墙钟只认显式 ms，否则用默认全核"
        )
    return SchedulePlan(
        query=query,
        tasks=tasks,
        note=note,
        available_cores=available,
        grant=grant,
        assumptions=[
            "子问题分解用**关键词启发式**（透明可测；不引 embedding）",
            "后端选择依据 chip_model 的'算力受限'结论（tube 几何 I≈133≫I*）",
            f"线程数依据 predict_schedule（E 核调度效率 {e_core_efficiency:.2f}，"
            f"P 核降频比 {droop:.2f}）",
            size_line,
            core_line,
            "本计划**不执行**计算——只决定'调哪些模块、用什么资源'",
        ],
    )


def main() -> int:
    """命令行：演示几个提问的调度计划。"""
    for q in (
        "算一下这个管族的并集体积，并检查这些点是否在集合内",
        "不要蒙特卡洛，用体素算体积",
        "跑一遍证明编排，看下界是否成立",
        "你好，今天天气怎么样",
    ):
        print(plan_query(q, single_core_ms=51.4, reserve_cores=2).explain())
        print()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
