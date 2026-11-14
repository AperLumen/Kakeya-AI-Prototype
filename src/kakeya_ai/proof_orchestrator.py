"""主算法骨架：三维挂谷证明搜索的编排器。

这个模块把已有的组件（Wolff 公理检验、sticky 判定、grains 分解、
Moves #1/#2/#3、体积估计）连接成一个可运行的证明路径搜索骨架。

它本身是一个**骨架实现**：
- Move 选择默认使用启发式规则，可选接入 surrogate 模型（置信度门控）；
  后续也可替换为 LLM 评分或精确数学判据；
- 递归和尺度归纳做了简化，只演示结构；
- 不追求复刻 127 页证明的每一个细节，而是让“状态 → 检查 → 选择 Move
  → 应用 → 再检查”的主循环能够跑通。

使用方式
--------
    from kakeya_ai.proof_orchestrator import ProofOrchestrator

    orch = ProofOrchestrator(tubes, epsilon=0.05, K=2.0)
    result = orch.run()
    print(result.trace_summary())
"""

from __future__ import annotations

import site
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np

from .domain import (
    DEFAULT_RADIUS,
    bounds_for_radius,
    exceeds_domain,
    radius_covering,
)
from .grains import (
    BroadnessReduction,
    GrainsDecomposition,
    NarrowBatch,
    broadness_reduction,
    guth_grains_skeleton,
)
from .moves import (
    _move_target_delta,
    _move1_impl,
    _move2_impl,
    _move3_impl,
    check_move1_condition,
    check_move2_condition,
    check_move3_condition,
)
from .primitives import Shading, Tube
from .sticky_kakeya import (
    InductionTermination,
    NSPFactorizationResult,
    build_scale_chain,
    classify_induction_termination,
    multiscale_induction,
    nsp_factorization,
)
from .volume import theorem_1_2_lower_bound
from .errors import ContractViolationError, InputValidationError
from .proof_monitor import (
    NullMonitor,
    ProgressSnapshot,
)

if TYPE_CHECKING:
    from .ai.surrogate_scorer import SurrogateScorer


class StepCallbackLike(Protocol):
    """``on_step`` 需要满足的最小接口。

    ``(snapshot: ProgressSnapshot) -> None``。用 ``Protocol`` 而非写死类型：
    调用方可以直接传函数（:class:`~kakeya_ai.proof_monitor.CallableMonitor`
    适配），也可以传 :class:`~kakeya_ai.proof_monitor.ProofMonitor` 实例，
    还可以传自研的任意对象——只要它有 ``on_step`` 方法。

    **回调抛出的异常不会中断证明搜索**（编排器捕获后记入诊断）。观测设施
    不该成为新的失败源。
    """

    def on_step(self, snapshot: ProgressSnapshot) -> None:  # pragma: no cover
        ...


#: ``on_step`` 参数接受的两类形态：
#:
#: - 带 ``on_step`` 方法的**对象**（:class:`StepCallbackLike`，如
#:   :class:`~kakeya_ai.proof_monitor.ProofMonitor`）；
#: - **裸函数** ``(snapshot) -> None``（运行时由
#:   :class:`~kakeya_ai.proof_monitor.CallableMonitor` 自动包装）。
#:
#: **为什么单列一个别名**：运行时两条路径都支持（见 ``__init__`` 的
#: ``hasattr(on_step, "on_step")`` / ``callable(on_step)`` 分支与 docstring），
#: 但此前参数只标注了 :class:`StepCallbackLike`——只认“对象”，于是所有传
#: 裸函数（lambda 或普通 ``def``）的调用点都被类型检查器误报。把运行时事实
#: 如实写进类型，而不是让调用方逐个 ``# type: ignore``。
OnStepArg = Union[StepCallbackLike, Callable[[ProgressSnapshot], None]]


class LLMScorerLike(Protocol):
    """``llm_scorer`` 需要满足的最小接口。

    形如 ``beam_search_score`` / ``cascaded_score``：
    ``(prompt, *, state=..., **kw) -> (scores, source)``，其中 ``scores`` 是
    ``{1: float, 2: float, 3: float}``。

    为什么用 ``Protocol`` 而不是写死具体函数：LLM 调用会**真的发起 HTTP
    请求**（本项目曾因此吃过测试跑 28 秒、内存压到 0.89 GB 的亏）。Protocol
    让调用方/测试能注入桩实现，把"接线是否正确"与"LLM 是否可达"彻底解耦。
    """

    def __call__(
        self,
        prompt: str,
        *,
        state: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[Dict[int, float], str]:
        ...


#: 支持的 LLM 打分策略 → 默认实现（惰性导入的模块路径 + 函数名）。
#:
#: - ``beam``：``beam_search_score`` —— 快模型生成若干候选推理方向，慢模型
#:   对 top-k 逐一评分。适合"候选空间大、需要比较"的场景。
#: - ``cascaded``：``cascaded_score`` —— 快模型起头给一段推理，慢模型接着
#:   往下推。适合"单条线索需要深化"的场景。
#:
#: 两者签名完全兼容（同吃 ``(prompt, state=...)``、同返回
#: ``(scores, source)``），故主循环用同一段代码调它们，只靠 ``llm_strategy``
#: 区分记录。（各家策略共享同一段接线逻辑，避免两套实现漂移。）
_LLM_STRATEGIES = frozenset({"beam", "cascaded"})


def make_llm_scorer(strategy: str = "beam", **kwargs) -> "LLMScorerLike":
    """按策略名构造 LLM 打分器（注入 ``llm_scorer=`` 用）。

    Parameters
    ----------
    strategy : str
        ``"beam"`` → ``beam_search_score``；``"cascaded"`` → ``cascaded_score``。
    **kwargs
        原样转发给打分器（如 ``beam_width``、``fast_model``、``deep_model``）。

    Returns
    -------
    scorer : callable
        可直接传给 ``ProofOrchestrator(llm_scorer=...)``。

    Examples
    --------
    >>> from kakeya_ai.proof_orchestrator import make_llm_scorer
    >>> scorer = make_llm_scorer("cascaded", deep_model="qwen3.5:4b")
    >>> orch = ProofOrchestrator(tubes, llm_scorer=scorer, llm_strategy="cascaded")

    Notes
    -----
    **惰性导入**：``beam_search`` / ``cascaded_reasoning`` 会拉起
    ``ollama_client``，而后者在导入期不做事——但仍放在函数内导入，避免
    "只想用启发式"的调用方被牵连导入一堆 AI 依赖。
    """
    if strategy not in _LLM_STRATEGIES:
        raise InputValidationError(
            f"未知的 LLM 策略 {strategy!r}；可选 {sorted(_LLM_STRATEGIES)}"
        )
    if strategy == "beam":
        from .ai.beam_search import beam_search_score as _impl
    else:
        from .ai.cascaded_reasoning import cascaded_score as _impl

    def _scorer(prompt: str, *, state: Optional[dict] = None, **call_kwargs):
        return _impl(prompt, state=state, **{**kwargs, **call_kwargs})

    return _scorer


# 把 surrogate 的 move 编号（1/2/3）映射到编排器所执行的动作。
_SURROGATE_MOVE_ACTIONS = {
    1: "MOVE1_LONGER_GRAINS",
    2: "MOVE2_REPLACE_SQUARE_GRAINS",
    3: "MOVE3_WIDER_GRAINS",
}


class ProofAction(Enum):
    """证明路径上的一步动作类型。"""

    STICKY_CHECK = auto()
    BROADNESS_REDUCTION = auto()
    NSP_FACTORIZATION = auto()
    GRAINS_DECOMPOSITION = auto()
    MOVE1_LONGER_GRAINS = auto()
    MOVE2_REPLACE_SQUARE_GRAINS = auto()
    MOVE3_WIDER_GRAINS = auto()
    LOWER_BOUND_VERIFY = auto()
    RECURSION = auto()
    TERMINAL_SUCCESS = auto()
    TERMINAL_FAILURE = auto()


class FailureReason(Enum):
    """搜索**为什么没成功**——失败的可归因分类。

    为什么需要它：此前 ``success=False`` 只有一个含义"没成功"，调用方分不清
    是"预算用尽"、"走进死路"、还是"输入本身就退化了"。三者的后续处置完全不同
    （加预算 vs 换策略 vs 修输入），却都长成同一个布尔值。分类后：

    - ``NONE``：成功，或尚未终止（``exhausted`` 且当前下界成立）。
    - ``UNKNOWN``：失败但**没有归因信息**（手工构造的 ``ProofResult``、
      或未来新增的终止路径忘了写归因）。**必须与"预算用尽"分开**：
      未归因不等于预算用尽，硬填一个具体类别就是编造——本项目的基本原则是
      "宁可说不知道，也不要猜一个像样的答案"。
    - ``BUDGET_EXHAUSTED``：迭代次数用尽，搜索**没跑完**。不是判定失败，
      只是没跑到底——加 ``max_iterations`` 可能就出结果。对应
      ``ProofResult.exhausted=True``。
    - ``STICKY_LOWER_BOUND_UNMET``：判定为黏性构型，直接验下界没过。
      这是**判定性**失败：sticky 情形下没有别的 Move 可走。
    - ``RECURSION_EXHAUSTED``：需要递归细化但已到最小尺度（δ 不能再小），
      且下界仍未满足。
    - ``EMPTY_INPUT``：输入管族为空，没有可搜索的对象。
    - ``NO_PROGRESS``：三个 Move 都试过、且都**观测不到任何改进**（指标逐位
      不变）——继续迭代只会空转。**必须与 ``BUDGET_EXHAUSTED`` 分开**：
      前者加 ``max_iterations`` 只会多空转几次，后者加预算可能就出结果。
      实测背景：12 根随机构型下 `MOVE2` 连续重试 10 次、指标（rho /
      local_error / n_grains）逐位不变，而归因是 ``BUDGET_EXHAUSTED``
      ——误导人去加预算。
    """

    NONE = auto()
    UNKNOWN = auto()
    BUDGET_EXHAUSTED = auto()
    STICKY_LOWER_BOUND_UNMET = auto()
    RECURSION_EXHAUSTED = auto()
    EMPTY_INPUT = auto()
    NO_PROGRESS = auto()


class ProofScope(Enum):
    """``success`` 的**作用域**——这个布尔到底在断言哪个集合上的下界。

    **为什么必须有这个枚举**（2026-09-30 加，L1 数学闭合）：此前
    ``success=True`` 混合了两种完全不同的情形：

    - (a) 下界对**原输入管族**成立；← 数学上有意义
    - (b) 下界只对 broadness 剥离**之后**的子集成立。← 现在也会返回 ``True``

    ``uncovered_narrow_tubes`` 已经把情形 (b) 的事实**记下来**了，但没参与
    判定——读结果的人只要不看那个字段，就会把 (b) 当成 (a)。这不是"不够
    详细"，而是**同一个布尔承载了两种含义**，下游无法据它做判断。

    故拆成三值，且**互相不可混淆**：

    - ``FULL``：下界覆盖原输入**全部**管族。**这是唯一可以读作"证明成功"的取值。**
    - ``BROAD_ONLY``：下界只覆盖 broad 子集；被剥离的窄簇**未被任何步骤承接**。
      此时 ``success`` **强制为 False**（构造时校验）——因为对原输入而言
      什么都没证明。
    - ``NOT_APPLICABLE``：空输入，无下界可言。

    **与论文的差距（如实标注）**：Wang–Zahl 里被剥离的窄簇由"**小帽上的归纳**"
    单独承接，从而仍能得到原输入的下界。**本实现未实现该承接**（`grains.py`
    的 ``working -= set(batch)`` 只做了移除）。所以 ``BROAD_ONLY`` 是
    "本实现的能力边界"，不是"论文的结论"。
    """

    FULL = auto()
    BROAD_ONLY = auto()
    NOT_APPLICABLE = auto()


#: "小帽上的归纳"的最大递归深度（2026-09-30 加，防无界递归）。
#:
#: 取值理由：顶层遇到剥离时最多再下探 2 层——实测一个构型通常剥出 2–4 批，
#: 两层足够覆盖"批里还有批"的常见情形；更深说明构型本身高度退化，
#: 那时**如实报未承接**比继续下探更有价值（继续下探只会烧时间）。
#: 本项目已有"递归跑飞"的实际教训（见 ``_induction_depth`` 的注释）。
MAX_INDUCTION_DEPTH = 2


#: ``ProofScope`` → 面向人的一行说明。
PROOF_SCOPE_TEXT = {    ProofScope.FULL: "lower bound covers the full input tube family",
    ProofScope.BROAD_ONLY: (
        "lower bound holds ONLY on the broad subset; stripped narrow clusters "
        "are not covered by any step (paper: 'induction on a smaller cap', "
        "not implemented here) — this is NOT a bound on the original input"
    ),
    ProofScope.NOT_APPLICABLE: "no lower bound (empty input)",
}


#: ``FailureReason`` → 面向人的一行说明。集中在这里，避免各处拼字符串走样。
FAILURE_REASON_TEXT = {    FailureReason.NONE: "no failure (success or not yet terminated)",
    FailureReason.UNKNOWN: "failed without attribution (unknown cause)",
    FailureReason.BUDGET_EXHAUSTED: "iterations exhausted; search did not finish",
    FailureReason.STICKY_LOWER_BOUND_UNMET: "sticky configuration; lower bound not reached",
    FailureReason.RECURSION_EXHAUSTED: "cannot recurse to a finer scale; lower bound not reached",
    FailureReason.EMPTY_INPUT: "empty tube collection",
    FailureReason.NO_PROGRESS: (
        "no move made measurable progress; further iterations would spin"
    ),
}

#: ``ProofAction`` → 该 Move 的**目标量键**（缺口二：判定"单次执行是否推进"用）。
#: 方向由 :func:`~kakeya_ai.moves._move_target_delta` 决定（Move #1 推大 ``max_c``，
#: Move #2/#3 压小 ``max_rho``/``max_local_kt_error``），**不在这里再写一份**。
_MOVE_TARGET_KEY = {
    ProofAction.MOVE1_LONGER_GRAINS: "max_c",
    ProofAction.MOVE2_REPLACE_SQUARE_GRAINS: "max_rho",
    ProofAction.MOVE3_WIDER_GRAINS: "max_local_kt_error",
}


def _target_improved(kind: str, before, after) -> bool:
    """应用 Move 前后，目标量是否**朝目标方向**改进。

    薄封装，直接委托给 ``moves._move_target_delta``——判据只在那一处定义，
    避免编排器与 ``MoveTrace`` 两处各写一份而漂移（本项目已多次踩此坑）。
    ``before``/``after`` 为 ``None``（grains 尚未构建）时按"未改进"处理。
    """
    if before is None or after is None:
        return False
    return bool(_move_target_delta(kind, before, after).improved)


class SelectionSource(Enum):
    """这一步的 Move **是谁选的**——决策路径的归因。

    与 ``FailureReason`` 对称：那个回答"为什么没成功"，这个回答"谁做的决定"。
    trace 的 ``notes`` 里其实已经写了来源，但那是**自由文本**——没法统计
    "这一轮搜索里启发式兜底占多少、LLM 命中多少"。用枚举则可以直接计数。

    为什么值得记：文档把"执行层逐步覆盖方向层"当作训练闭环的目标
    （``surrogate_coverage`` 逐步扩大 → LLM 只需处理它真正擅长的部分）。
    要判断这个目标有没有在推进，前提是**能数出各路径的占比**。

    - ``LLM``：注入的 LLM 打分器命中（beam 或 cascaded）。
    - ``SURROGATE``：本地 surrogate 模型命中（置信度过阈值）。
    - ``HEURISTIC``：以上都没命中，走 Move 条件检查——**这是必须能数出来的**
      一档：它占比高说明打分器还没覆盖住实际状态分布。
    - ``NOT_APPLICABLE``：这一步根本不涉及 Move 选择（broadness 过滤、
      grains 分解、NSP 因子化、sticky 判定、递归等）。与"启发式兜底"必须
      分开——否则会把"没有选择"算成"选择失败"。
    """

    LLM = auto()
    SURROGATE = auto()
    HEURISTIC = auto()
    NOT_APPLICABLE = auto()


#: ``SelectionSource`` → 面向人的一行说明。
SELECTION_SOURCE_TEXT = {
    SelectionSource.LLM: "Move chosen by the injected LLM scorer",
    SelectionSource.SURROGATE: "Move chosen by the local surrogate model",
    SelectionSource.HEURISTIC: "Move chosen by the heuristic (no scorer hit)",
    SelectionSource.NOT_APPLICABLE: "this step does not select a Move",
}


@dataclass
class ProofMetrics:
    """某一时刻可观测的证明状态指标。"""

    delta: float
    rho: float
    a: float
    b: float
    c: float
    local_error: float
    volume_ratio: float
    union_volume: float
    lower_bound: float
    n_tubes: int
    n_grains: int
    is_sticky: bool = False
    #: 算这个指标时用的域半径（体积估计的盒 = ``[-r, r]³``）。**逐步骤记录**：
    #: 域会在 NSP 抬升管族时变大，抬升前后的步骤各自的域不同。不记的话，
    #: 复核方会在错误的盒上重算早期步骤（早期值是在小盒上算的）。
    domain_radius: float = 1.0
    #: 算这个指标时用的网格数。与域半径同理：域抬升时 grid_n 也会被抬到
    #: 2 的幂（保持分辨率），逐步骤记录才能逐步骤重算。
    grid_n: int = 0

    def __repr__(self) -> str:
        return (
            f"Metrics(δ={self.delta:.4g}, ρ={self.rho:.4g}, "
            f"err={self.local_error:.4g}, vol_ratio={self.volume_ratio:.4g}, "
            f"sticky={self.is_sticky})"
        )


class _StateChangeRequired(TypedDict):
    """必带键：只有 ``op``。"""

    op: str                    # 该字段的动作类型，取值为 keep / subset / rescale / replace 中的一种


class StateChange(_StateChangeRequired, total=False):
    """管族从前一步到本步的变换描述（``ProofStep.state_change`` 的结构）。

    用 ``TypedDict`` 而不是 ``dict``：消费端要按下标取 ``op`` / ``indices``
    等字段，裸 ``dict`` 在类型检查器里会让每次下标访问退化成 ``object``，
    刷出成片与真实缺陷混杂的假报错（本项目已有前例）。

    ``op`` 是**唯一必带键**（``_StateChangeRequired``），其余可选：四种
    ``op`` 各自只带自己那几个键——``keep`` 不带任何附加键，``subset`` 只有
    ``indices``，``rescale`` 有 ``delta`` / ``delta_hex`` / ``factor``，
    ``replace`` 有 ``tubes``。若把其余也声明成必带，会逼调用点写无意义的
    占位键；但把 ``op`` 也变成可选，读端每次 ``["op"]`` 都会报"可能不存在"。
    """

    indices: List[int]         # op == "subset"
    delta: float               # op == "rescale"
    delta_hex: str             # op == "rescale"
    factor: float              # op == "rescale"
    tubes: List[dict]          # op == "replace"


@dataclass
class ProofStep:
    """证明路径中的一步记录。"""

    iteration: int
    action: ProofAction
    before: ProofMetrics
    after: ProofMetrics
    notes: List[str] = field(default_factory=list)
    #: 该步的**管族如何从前一步得到**——复核方靠它重建中间状态。
    #: 形如 ``{"op": "keep"}`` / ``{"op": "subset", "indices": [...]}`` /
    #: ``{"op": "rescale", "factor": 0.5}`` / ``{"op": "replace", "tubes": [...]}``。
    #: 没有它就只能验哈希链、无法重算体积（曾因此在多步 trace 上只能报
    #: UNVERIFIED）。旧调用点默认 None——**只有**这些调用点会是 None，
    #: ``run()`` 产出的每一步都非 None（见 ``TestExportContract``）。
    state_change: Optional[StateChange] = None
    #: 本步的 Move 是谁选的（见 ``SelectionSource``）。不涉及 Move 选择的
    #: 步骤（broadness/grains/NSP/sticky/递归）为 ``NOT_APPLICABLE``。
    #:
    #: 与 ``notes`` 里的自由文本**并存**而不是取代它：notes 给人看细节
    #: （如 ``gap=1.2``），这个字段给机器数占比。两者缺一：只有文本则没法
    #: 统计，只有枚举则丢失"为什么"。
    #:
    #: 类型是 ``Optional[...]`` 而不是裸枚举：构造时**允许传 None**，由
    #: ``__post_init__`` 规范成 ``NOT_APPLICABLE``。此前标注成非可选、
    #: 靠 ``# type: ignore`` 压住报错——那是在类型系统里说了句假话，代价是
    #: 每个"传 None"的调用点（含测试）都要自己再 ignore 一次。
    selection_source: Optional["SelectionSource"] = None
    #: 本步（若是 Move 步）**是否达成可观测的定量增益**（缺口二）。
    #:
    #: - ``True``：Move 的目标量朝目标方向变了（Move #1 看 ``max_c`` 应增，
    #:   Move #2/#3 看量应减）。
    #: - ``False``：应用了 Move 但目标量**逐位不变**（前置条件已满足 / 无法
    #:   再合并）——即 ``NO_PROGRESS`` 的微观来源。
    #: - ``None``：非 Move 步（broadness/grains/NSP/sticky/递归），无此语义。
    #:
    #: **为什么必须是一等字段**：``training_data.proof_steps_to_samples`` 会把
    #: "实际执行过的 Move"当作**正样本**。而实测（2026-10-01）一条 trace 里
    #: 3 个 Move 样本有 **2 个是 ``MOVE2`` 的 no-op**（``achieved_gain=False``）
    #: ——拿它训练等于**教模型偏好一个什么都不做的 Move**。有了本字段，消费端
    #: 才能把 no-op 决策排除在正样本之外。``False`` 与 ``None`` **语义相反**
    #: （"明确没进展" vs "不适用"），不可混同。
    achieved_gain: Optional[bool] = None
    #: 本步（若是 Move 步）**实际执行的合并轮数**（精确值，不是预算）。
    #:
    #: - 正整数 ``n``：Move 循环成功缩小 grain 数 ``n`` 轮；
    #: - ``0``：应用了 Move 但一轮都没动（前置条件已满足 / 无法再合并）；
    #: - ``None``：非 Move 步（broadness/grains/NSP/sticky/递归），无此语义。
    #:
    #: **与 ``achieved_gain`` 互补**：后者是"动没动"的布尔，本字段是"动了几轮"。
    #: 二者同源于一次 Move 应用（``_apply_move`` 经 ``_moveN_impl`` 取得），
    #: 使 §7.0.2 缺口二里"区分得开完全没动与动过，但区分不了动 1 轮与 3 轮"
    #: 这一条**闭合**。参与哈希链（见 ``trace_format._step_payload``）。
    iters_used: Optional[int] = None

    def __post_init__(self) -> None:
        # 未显式给出时取 NOT_APPLICABLE——**不能**默认成 HEURISTIC：那会把
        # "这一步不涉及选择"混进"启发式兜底"，使统计虚高。
        if self.selection_source is None:
            self.selection_source = SelectionSource.NOT_APPLICABLE

    @property
    def source(self) -> "SelectionSource":
        """本步的来源，**保证非 None**（归一化已在 ``__post_init__`` 完成）。

        读点用它而不是直接读 ``selection_source``：字段类型是 ``Optional``
        （因为构造时接受 ``None``），但过了 ``__post_init__`` 之后就不可能
        为 ``None``。用一个带返回标注的属性把这层不变量表达出来，读点就
        不必写"永不触发"的判空分支——否则每处都要 `assert ... is not None`。
        """
        source = self.selection_source
        if source is None:            # pragma: no cover - 仅当绕过 __post_init__
            raise RuntimeError(
                "selection_source 未归一化：ProofStep 被绕过 __post_init__ 构造"
            )
        return source

    def summary(self) -> str:
        return (
            f"Step {self.iteration}: {self.action.name} "
            f"[{self.source.name}] | "
            f"before={self.before} | after={self.after}"
        )


@dataclass
class ProofState:
    """当前证明状态。"""

    tubes: List[Tube]
    shading: Optional[Shading]
    grains: Optional[GrainsDecomposition]
    delta: float
    rho: float
    iteration: int = 0
    # broadness 前置过滤剥离出的窄簇（每项是一批原始 tube 索引）。
    #: 被 broadness 前置过滤剥离的窄簇批次。**自 2026-09-30（L1）起元素是
    #: :class:`~kakeya_ai.grains.NarrowBatch`**（原来仅索引元组）——因为
    #: 承接它们需要知道"小帽多大"，只记索引不足以复现论文的"小帽上的归纳"。
    narrow_batches: List[NarrowBatch] = field(default_factory=list)
    # 当前 tubes 是否已通过 broadness 检查（避免重复采样）。
    broadness_checked: bool = False


@dataclass
@dataclass(frozen=True)
class _NarrowBatchOutcome:
    """单批窄簇的承接结果（内部类型，不导出）。"""

    index: int
    n_tubes: int
    covered: bool
    reason: str = ""


@dataclass(frozen=True)
class _NarrowCoverage:
    """全部被剥离窄簇的**承接报告**（内部类型，供 scope 判定用）。

    ``all_covered`` 为 True 时，才允许把「下界只覆盖 broad 子集」升级为
    「覆盖原输入」——即 scope 从 ``BROAD_ONLY`` 变为 ``FULL``。
    """

    covered: int
    total: int
    per_batch: List[_NarrowBatchOutcome]
    notes: List[str]

    @property
    def all_covered(self) -> bool:
        return self.total > 0 and self.covered == self.total

    def summary(self) -> str:
        if self.total == 0:
            return "无被剥离窄簇"
        if self.all_covered:
            return f"{self.covered}/{self.total} 批窄簇均已被承接"
        failed = [b for b in self.per_batch if not b.covered]
        detail = "；".join(f"#{b.index}({b.n_tubes} 根): {b.reason}" for b in failed[:3])
        return (
            f"⚠️ {self.covered}/{self.total} 批窄簇被承接，"
            f"{len(failed)} 批未承接 —— {detail}"
        )


@dataclass
class ProofResult:
    """算法骨架运行结果。"""

    success: bool
    steps: List[ProofStep]
    final_state: ProofState
    final_metrics: ProofMetrics
    message: str
    #: 主循环因达到 ``max_iterations`` 而结束（而非某个 terminal 动作）。
    #: 注意它与 ``success`` 是两回事：``success`` 只由下界计算决定，
    #: 而 ``exhausted=True`` 表示搜索本身没跑完，即使下界碰巧成立，
    #: 也不代表找到了终止路径。
    exhausted: bool = False
    #: 本次运行的**域半径**（体积估计的盒 = ``[-r, r]³``）。基准是 1.0；
    #: 管族被 NSP 副本带出基准域时抬升为 2 的幂。必须写进 trace：
    #: 复核方要在同一个域上重算，数值才可比（见 ``domain.py`` 的三条不变量）。
    domain_radius: float = 1.0
    #: 初始管族。final_state.tubes 可能已被 broadness 过滤
    #: 改小，无法反推输入；导出可复核的 trace（见 ``trace_format``）必须要
    #: 有它——否则复核方拿不到几何起点。旧调用点默认 None。
    initial_tubes: Optional[List[Tube]] = None
    #: 本次运行使用的随机种子。
    #:
    #: - 整数：NSP 因子化的随机刚体变换完全由它决定，**同一输入重跑逐位一致**。
    #: - ``None``：探索模式（每次运行的副本不同）。此时这份结果**不可复现**，
    #:   写进 trace 时不会记 ``seed``，复核方无法重放 NSP 步骤。
    #: - 是否由输入几何自动派生，见 ``seed_is_derived``。
    seed: Optional[int] = None
    #: ``seed`` 是否为"未显式指定、从输入几何派生"而来。派生值同样确定性，
    #: 标注出来是为了让复核方知道"这个 seed 不是人为挑的，而是几何的函数"。
    seed_is_derived: bool = False
    #: 失败归因。成功时为 ``FailureReason.NONE``。**与 ``success`` 独立**：
    #: ``success`` 说"下界是否成立"，``failure_reason`` 说"为什么没成功"。
    #: 两者可能出现 ``success=True`` 配 ``BUDGET_EXHAUSTED`` 的组合（预算用尽
    #: 但当前下界碰巧成立）——那仍然是"没找到终止路径"，见 ``exhausted``。
    failure_reason: "FailureReason" = None  # type: ignore[assignment]
    #: **``success`` 的作用域限制**：被 broadness 前置过滤剥离、且**未被任何
    #: 后续步骤覆盖**的管数。
    #:
    #: 为什么必须记这个数：broadness reduction 在 Wang–Zahl 里是"把窄簇转化为
    #: **小帽上的归纳**"，即被剥离的部分会由另一条归纳链单独覆盖。本实现只做了
    #: 前半步——把窄簇从集合里移除（``working -= set(batch)``），**没有**对它们
    #: 启动任何递归或归纳。于是 ``TERMINAL_SUCCESS`` 判定用的 ``volume_ratio``
    #: 是**剥离后**子集的，它**不构成对原输入管族的体积下界**。
    #:
    #: 因此：``success=True`` 且本字段 ``> 0`` 时，正确读法是
    #: **"下界在 broad 子集上成立"**，而不是"原输入成立"。此前这个限制只存在
    #: 于 ``narrow_batches``（一个只被记录、从未被消费的字段），读结果的人无从
    #: 得知；现已上浮为结果的一等字段。
    #:
    #: 2026-09-29 之前的运行结果没有这个字段，读回来是 0——那表示"生成时无此
    #: 记录"，**不等于**当时没有剥离（旧结果需查 ``steps`` 里的
    #: ``BROADNESS_REDUCTION`` 步骤自行判断）。
    uncovered_narrow_tubes: int = 0

    #: **``success`` 的作用域**（见 :class:`ProofScope`）。2026-09-30 新增。
    #:
    #: 这是 L1 数学闭合的核心：让 ``success`` 的含义**不再是二义的**。
    #: 三条不变量由 :meth:`__post_init__` 强制（不满足即抛错，不静默容忍）：
    #:
    #: 1. ``success=True`` ⟹ ``scope is ProofScope.FULL``
    #:    （只有覆盖全部输入才配称"成功"）；
    #: 2. ``scope is ProofScope.BROAD_ONLY`` ⟹ ``success=False``
    #:    （对被剥离部分什么都没证明）；
    #: 3. ``uncovered_narrow_tubes > 0`` ⟹ ``scope is BROAD_ONLY``
    #:    （两处记录同一事实，不许互相矛盾）。
    #:
    #: 默认取 ``NOT_APPLICABLE`` 而不是 ``FULL``：**默认值不该是"最乐观"的那个**。
    #: 手工构造 ``ProofResult``（测试、旧代码）忘了填时，会因不变量 1 报错，
    #: 从而**被迫显式声明**——这是有意的摩擦。
    #:
    #: 旧数据兼容：2026-09-30 之前的 ``.kpt`` / 序列化结果没有此字段，
    #: 读回来（若构造时未指定）是 ``NOT_APPLICABLE``，表示"生成时无此记录"，
    #: **不等于**当时覆盖了全部输入。判据同 ``uncovered_narrow_tubes``。
    scope: "ProofScope" = ProofScope.NOT_APPLICABLE

    #: 被剥离窄簇的**承接报告**（人读的一行摘要）。2026-09-30 新增。
    #:
    #: ``BROAD_ONLY`` 时这里说明"几批被承接、几批没接住"——它是 scope 的
    #: **证据**：光说"只在 broad 子集上成立"不够，读者需要知道承接尝试过
    #: 没有、结果如何。空串表示"无窄簇"或旧结果（生成时无此字段）。
    narrow_coverage: str = ""

    def coverage_note(self) -> str:
        """返回一句**可直接放进报告**的作用域说明。

        设计意图：让"下界只管 broad 子集"这件事在下游（报告、trace 摘要、
        人读的日志）有一个统一的、不会被遗忘的措辞——而不是每处调用方各自
        手写一段（必然漂移或漏写）。
        """
        if self.uncovered_narrow_tubes <= 0:
            if self.scope is ProofScope.FULL:
                return "下界覆盖全部输入管族（无被剥离且未覆盖的窄簇）"
            # scope 非 FULL 且无窄簇：只可能是 NOT_APPLICABLE（空输入）。
            return "无下界可言（空输入）"
        return (
            f"⚠️ 下界仅在 broad 子集上成立：有 {self.uncovered_narrow_tubes} "
            "根窄簇管被前置过滤剥离，且承接未成功"
            "（承接走『小帽上的归纳』的同尺度保守特例，见 "
            "``_induct_on_narrow_batches``；失败详情见 narrow_coverage）"
            "——因此这**不是**对原输入的下界，`success` 已强制为 False"
        )

    def __post_init__(self) -> None:
        # 未显式归因时的默认值：成功 → NONE；失败 → **UNKNOWN**。
        #
        # 这里**不能**默认成某个具体失败类别（曾错填 BUDGET_EXHAUSTED）。
        # `success=False` 只说明"没成功"，真实原因可能是黏性、递归到底、
        # 输入退化，或未来新增的终止路径；把它硬归成"预算用尽"是编造归因，
        # 会让调用方做出错误处置（去加 max_iterations，而问题根本不在预算）。
        # 走 run() 的正常路径一定会在终止分支显式写入归因，所以 UNKNOWN
        # 只出现在手工构造或新路径漏写归因的情形——那正是应该报"未知"的时候。
        if self.failure_reason is None:
            self.failure_reason = (
                FailureReason.NONE if self.success else FailureReason.UNKNOWN
            )
        self._enforce_scope_invariants()

    def _enforce_scope_invariants(self) -> None:
        """强制 ``success`` / ``scope`` / ``uncovered_narrow_tubes`` 三者自洽。

        **为什么在构造时抛错而不是"容忍并归一化"**：归一化（比如自动把
        ``success=True`` + ``BROAD_ONLY`` 改成 ``success=False``）会让
        **调用方写错却不自知**——它仍然以为自己在设 ``success``。抛错则把
        "你想断言什么"这个问题逼回给调用方。这与本项目对
        ``error_mode='actual'`` 漏传 ``tubes`` 的处置一致（外层显式抛错）。

        三条不变量见 :attr:`scope` 的文档。
        """
        if self.success and self.scope is not ProofScope.FULL:
            raise ContractViolationError(
                "success=True 但 scope 不是 FULL："
                f"scope={self.scope.name}, "
                f"uncovered_narrow_tubes={self.uncovered_narrow_tubes}。"
                "『成功』只能用于断言**原输入全部管族**的下界；"
                "若下界只覆盖 broad 子集，必须 success=False 且 "
                "scope=BROAD_ONLY（论文中该缺口由『小帽上的归纳』承接，"
                "本实现未实现）。"
            )
        if self.scope is ProofScope.BROAD_ONLY and self.success:
            # 上一条已覆盖，保留此分支只为让错误信息更直白（防御未来重排）。
            raise ContractViolationError(
                "scope=BROAD_ONLY 不允许 success=True"
            )
        if self.uncovered_narrow_tubes > 0 and self.scope is not ProofScope.BROAD_ONLY:
            raise ContractViolationError(
                f"uncovered_narrow_tubes={self.uncovered_narrow_tubes} > 0 "
                f"但 scope={self.scope.name}：两处记录同一事实，不许矛盾。"
                "有未覆盖的窄簇时，scope 必须是 BROAD_ONLY。"
            )

    def selection_counts(self) -> Dict[str, int]:
        """本结果里各**决策来源**出现的步数（含 ``NOT_APPLICABLE``）。

        从 ``steps`` **派生**而不是另存一个字段：两份真相必然漂移（比如
        手工构造的 ``ProofResult`` 忘了填计数，就会出现"计数说 LLM 选了 3 步、
        实际一步都没有"）。派生则永远自洽。

        用途：文档把"执行层逐步覆盖方向层"（``surrogate_coverage`` 扩大、
        LLM 只处理它真正擅长的部分）当作训练闭环的目标，而判断这个目标
        有没有推进的前提，就是能数出各路径的占比：``HEURISTIC`` 占比高说明
        打分器还没覆盖住实际状态分布。
        """
        counts = {source.name: 0 for source in SelectionSource}
        for step in self.steps:
            counts[step.source.name] += 1
        return counts

    def trace_summary(self) -> str:
        lines = [
            f"Proof trace: success={self.success}, {len(self.steps)} steps"
            + (", exhausted (max_iterations)" if self.exhausted else ""),
            f"Failure reason: {self.failure_reason.name} "
            f"({FAILURE_REASON_TEXT[self.failure_reason]})",
            # scope 与 success **紧邻**输出：读者看到 "success=True" 的同一屏
            # 里就该知道它管到哪。若只把 scope 放在文件末尾，等于没声明。
            f"Scope: {self.scope.name} ({PROOF_SCOPE_TEXT[self.scope]})",
        ]
        # 只列**出现过的**来源：全零的项没有信息量（常见于不含 Move 选择的
        # 短 trace），列出来反而掩盖重点。
        counts = {k: v for k, v in self.selection_counts().items() if v}
        if counts:
            shown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            lines.append(f"Move selection: {shown}")
        lines.append(f"Final: {self.final_metrics}")
        # 作用域说明**只在有剥离时**追加：没有剥离时那句话是噪音
        # （scope 行已在开头给出，这里给的是人读的细节）。
        if self.uncovered_narrow_tubes > 0:
            lines.append(f"Scope detail: {self.coverage_note()}")
        lines.append("")
        for step in self.steps:
            lines.append(f"  {step.iteration:2d}: {step.action.name:30s} "
                         f"[{step.source.name:14s}] "
                         f"vol_ratio {step.before.volume_ratio:.4g} -> {step.after.volume_ratio:.4g}")
        return "\n".join(lines)


class ProofOrchestrator:
    """编排主算法骨架：sticky 检查 → grains 分解 → Move 选择 → 应用 → 验证。"""

    def __init__(
        self,
        tubes: Sequence[Tube],
        epsilon: float = 0.05,
        K: float = 2.0,
        lambda_min: float = 0.85,
        max_iterations: int = 10,
        grid_n: int = 64,
        use_nano: bool = True,
        move_params: Optional[dict] = None,
        surrogate: Optional["SurrogateScorer"] = None,
        surrogate_threshold: float = 2.0,
        surrogate_weights_path: Optional[str] = None,
        llm_scorer: Optional["LLMScorerLike"] = None,
        llm_strategy: str = "beam",
        llm_threshold: float = 2.0,
        broadness_filter: bool = True,
        broadness_K: float = 10.0,
        broadness_beta: float = 1.0,
        broadness_points: int = 32,
        seed: Optional[int] = 0,
        on_step: Optional["OnStepArg"] = None,
        _induction_depth: int = 0,
    ):
        """
        Parameters
        ----------
        tubes : sequence of Tube
            输入 δ-管族，假定满足 Wolff 公理。
        epsilon : float
            Theorem 1.2 中的 ε 参数。
        K : float
            Theorem 1.2 中的 K 参数。
        lambda_min : float
            染色下界 λ 参数。
        max_iterations : int
            最大 Move 迭代次数。
        grid_n : int
            体素网格数（用于体积估计）。
        use_nano : bool
            是否使用 ``nano_kernel`` 加速体积估计。
        move_params : dict or None
            传给 Move 函数的可选参数，如 ``zeta``、``omega``、``error_mode``。
            ``error_mode`` 的默认值见下方 ``__init__`` 末尾的说明（**已从
            ``proxy`` 改为 ``actual``**，2026-10-01）。
        surrogate : SurrogateScorer or None
            可选的 surrogate 评分模型。提供时，置信度足够高
            （top-two gap >= ``surrogate_threshold``）的预测会直接决定
            Move 选择；否则退回启发式。
        surrogate_threshold : float
            surrogate 预测的最小置信度门控（默认 2.0）。
        surrogate_weights_path : str or None
            surrogate 权重文件路径。仅当 ``surrogate`` 为 None 时生效；
            未显式给出时探测项目根 ``data/surrogate_scorer_weights.json``。
            文件缺失或损坏时静默退回启发式。
        broadness_filter : bool
            是否在 grains 分解前做 broadness 前置过滤（默认 True）。
            过滤把方向过分集中的窄簇剥离为 narrow batch，只对剩余
            （broad）子集继续做 grains/Move；剥离批次记入状态供后续
            归纳使用。设为 False 时行为与旧版完全一致。
        broadness_K : float
            broadness 常数 K。
        broadness_beta : float
            broadness 指数 β。
        broadness_points : int
            每次 broadness 采样点数。
        seed : int or None
            **随机种子，决定主循环是否可复现**（默认 0 = 确定）。

            主循环里唯一的随机来源是 NSP 因子化：它在管数不足 δ⁻² 时用
            随机刚体变换生成副本。此前该调用**没有传递种子**，于是
            ``np.random.default_rng(None)`` 每次取新熵——**同一输入重跑会得到
            不同的证明路径**（实测某一构型 12 次运行产生 4 种动作序列，58%
            的概率第一步就终止）。这与项目"同一输入 → 可独立复核的 trace"
            这条前提直接冲突：复核方即使拿到完全相同的参数，也复现不出同一份
            trace。故默认值取 0（确定）。

            - ``0``（默认）或任意整数：**确定性**。第 k 次 NSP 调用使用
              ``seed + k``，因此同一次运行内的多次因子化不会重复同一组变换，
              而整条路径完全由输入与 seed 决定，逐位可重放。
            - ``None``：**非确定性探索模式**（显式选择）。每次运行的副本都不同，
              适合"想看看不同随机副本会走出什么路径"。此时结果**不可复现**，
              导出 trace 时不会写入 ``seed``，复核方也无法重放 NSP 步骤——
              这是刻意的：不要在文件里记一个并不生效的种子。
        on_step : callable or object, optional
            **每步之后的进度回调**（长跑可观测性，2026-09-30 新增）。

            此前的 ``run()`` 是黑盒：一次 4 轮迭代实测约 81.8 s，期间调用方
            对"跑到第几步、每步多久、是否卡住"一无所知，失败时只看到"没出
            结果"。传一个 :class:`~kakeya_ai.proof_monitor.ProofMonitor` 即可
            收集进度并在 :meth:`~kakeya_ai.proof_monitor.ProofMonitor.verdict`
            里拿到三态健康判定（``progressing`` / ``stalled`` / ``unknown``）。

            - ``None``（默认）：不观测（内部用空对象，零开销）。
            - 任意带 ``on_step(snapshot)`` 方法的对象，或裸函数
              （自动用 :class:`~kakeya_ai.proof_monitor.CallableMonitor` 包装）。

            **回调抛出的异常被捕获并记入 :attr:`diagnostics`，不中断搜索**。
        """
        self.tubes = list(tubes)
        self.epsilon = epsilon
        self.K = K
        self.lambda_min = lambda_min
        self.max_iterations = max_iterations
        self.grid_n = grid_n
        self.use_nano = use_nano
        self.move_params = move_params or {}

        # surrogate：显式传入优先，否则尝试 lazy 加载权重文件。
        self.surrogate_threshold = float(surrogate_threshold)
        self.surrogate = surrogate if surrogate is not None else self._load_surrogate(
            surrogate_weights_path
        )
        # LLM 打分器：**默认不启用**。每次调用都会真的发 HTTP 请求，因此
        # 必须由调用方显式注入——不做"自动探测 Ollama 是否可达"那类隐式启用
        # （那会让一次普通的 run() 悄悄依赖外部服务，也让测试变慢、变脆）。
        #
        # ``llm_strategy`` 只用于**记录与诊断**：实际调用哪个函数由注入的
        # 可调用对象决定（避免"传错策略却不生效"这类静默错配）。合法值见
        # ``_LLM_STRATEGIES``。
        if llm_strategy not in _LLM_STRATEGIES:
            raise InputValidationError(
                f"llm_strategy 必须是 {sorted(_LLM_STRATEGIES)} 之一，"
                f"实际为 {llm_strategy!r}"
            )
        self.llm_scorer = llm_scorer
        self.llm_strategy = llm_strategy
        self.llm_threshold = float(llm_threshold)
        #: 最近一次喂给 LLM 打分器的 prompt（供备注与调试）。
        self._last_llm_prompt: Optional[str] = None
        # _select_move 写入、step() 读取的 surrogate 备注。
        self._selection_notes: List[str] = []
        # _select_move 写入、step() 读取的决策来源（枚举，供统计占比）。
        self._last_selection_source: SelectionSource = SelectionSource.NOT_APPLICABLE
        #: 打分器"想选但被守卫拦下"的计数，键为 ``"LLM"`` / ``"SURROGATE"``。
        #:
        #: **为什么要有它**：``SelectionSource`` 只记"最终谁做的决定"，所以
        #: surrogate 选了 10 次、其中 9 次被拦下时，占比统计会显示 surrogate
        #: 贡献了 1 次——看不到那 9 次"给了答案但没用"。而这个量正是训练
        #: 最该盯的指标（详见 ``surrogate_coverage_with_gate``）。文本形式的
        #: ``_selection_notes`` 里虽有记录，但文本不适合聚合，故单列计数。
        self._scorer_gate_blocks: Dict[str, int] = {"LLM": 0, "SURROGATE": 0}
        #: 打分器"给出答案且被采纳"的计数，与上面同构，用于算有效率。
        self._scorer_gate_accepts: Dict[str, int] = {"LLM": 0, "SURROGATE": 0}
        # step() 写入、run() 读取的失败归因。在 __init__ 里初始化（而不是
        # 在 run() 里），因为 step() 可以被直接调用——那时 run() 从未跑过。
        self._last_failure_reason: Optional[FailureReason] = None
        #: step() 写入、run() 读取的"本步 Move 是否达成定量增益"（缺口二）。
        #: ``None`` = 本步不是 Move（无此语义）；``True``/``False`` 见
        #: ``ProofStep.achieved_gain`` 的说明。同样在 __init__ 初始化，因为
        #: ``step()`` 可被直接调用。
        self._last_move_achieved_gain: Optional[bool] = None
        #: step() 写入、run() 读取的"本步 Move **实际执行了多少轮合并**"（缺口二）。
        #: ``None`` = 本步不是 Move（无此语义）。由 ``_apply_move`` 经 ``_moveN_impl``
        #: 拿到**精确轮数**（不是从 grain 数推的 0/1 下界），使"动了 1 轮"与
        #: "动了 3 轮"可区分。同样在 __init__ 初始化，因为 ``step()`` 可被直接调用。
        self._last_move_iters_used: Optional[int] = None
        # 校验/重算失败的诊断信息。吞异常时必须往这里写一行，再由 step()
        # 转写进 trace："判定失败"与"判定为否"必须在记录里可区分。
        self._diagnostics: List[str] = []

        # 进度回调（长跑可观测性）。默认空对象——主循环里就不必判 None。
        # 详见 __init__ 的 docstring 与 proof_monitor 模块。
        if on_step is None:
            self._monitor: object = NullMonitor()
        elif hasattr(on_step, "on_step"):
            # 已符合协议（如 ProofMonitor 实例）。
            self._monitor = on_step
        elif callable(on_step):
            # 裸函数：包成带 on_step 方法的适配器。
            from .proof_monitor import CallableMonitor
            self._monitor = CallableMonitor(on_step)
        else:
            raise InputValidationError(
                "on_step 必须是 None、可调用对象、或带 on_step 方法的对象",
                field="on_step", value=type(on_step).__name__,
            )
        #: 单步快照累计（供调用方在 run() 后查看；回调异常不会写这里）。
        self._snapshots: List[ProgressSnapshot] = []
        #: 累计耗时（秒），用于快照的 total_elapsed。
        self._elapsed_total: float = 0.0

        # Move 尝试历史（**调度状态机的记忆**）。
        #
        # 此前 `_select_move` 是**无状态**的：每轮重新算 `m1_ok/m2_ok/m3_ok`，
        # 不记得"刚才试过哪个、效果如何"。实测后果（12 根随机构型）：
        # `MOVE2` 连续重试 **10 次**，而 rho / local_error / n_grains
        # **逐位不变**——纯空转，却把预算耗尽，归因还是误导性的
        # `BUDGET_EXHAUSTED`（加预算只会多空转）。
        #
        # 记录"某 Move 试过之后是否让指标发生变化"，即可识别这种情况：
        # 一个 Move 若在**相同指标状态**下试过且毫无改进，就不该再试。
        #: ``{move_name: (上次尝试前的指标指纹, 该次尝试是否带来变化)}``
        #:
        #: **生命周期是「一次 run()」**：:meth:`run` 开头会清空它。不这样的话，
        #: 同一个实例的第二次 run() 会「记得」上次的结论，与新建实例结果不同
        #: ——那会让「重复跑同一构型做消融」静默失真（实测证据见
        #: ``tests/test_orchestrator_run_idempotent.py``）。
        self._move_attempts: Dict[str, Tuple[tuple, bool]] = {}

        # 体积估计函数
        self._volume_fn = self._choose_volume_fn()

        # broadness 前置过滤配置。
        self.broadness_filter = broadness_filter
        self.broadness_K = float(broadness_K)
        self.broadness_beta = float(broadness_beta)
        self.broadness_points = int(broadness_points)
        # 最近一次 broadness 过滤结果（供 step 备注与外部检查使用）。
        self.last_broadness: Optional[BroadnessReduction] = None
        # 最近一次 NSP 因子化结果（供 step 备注与外部检查使用）。
        self.last_nsp: Optional[NSPFactorizationResult] = None

        #: "小帽上的归纳"的递归深度（0 = 顶层运行）。        #:
        #: **必须显式记录并设上限**（2026-09-30）：承接窄簇是对 ``run()`` 的
        #: 递归调用，没有深度上限就成了无界递归。本项目已因"递归跑飞"付出过
        #: 代价（试过一版 δ 缩放导致单次运行数分钟不返回，只能 kill）。
        #: 达到 ``MAX_INDUCTION_DEPTH`` 时**如实报未承接**，不静默放宽。
        self._induction_depth = int(_induction_depth)

        #: **判据的参照系**：`volume_ratio` 的分母取这族管的下界。
        #:
        #: 默认为**本次运行的输入**（`self.tubes`）。**不随状态演化而变**——
        #: 这是 2026-09-30 第 5 步诊断后修的：此前分母取"当前管集"，
        #: 而 broadness 剥离会把它从 9 根减到 2 根，使下界降 4.5 倍，
        #: 于是 `volume_ratio` 从 0.343 虚高到 1.545、一步就越过成功线——
        #: **拿被裁过的管子集去比它自己的下界**，这不是证明。
        #:
        #: 论文 Theorem 1.2 断言的是**原输入** $\mathbb{T}$ 的体积下界，
        #: 故参照系必须是原输入。递归承接时传该批窄簇（那**是**它的"原输入"）。
        self._reference_tubes: Sequence[Tube] = list(tubes)

        #: 参照系的 `volume_ratio`，**只算一次**（`None` = 尚未计算）。
        #:
        #: 缓存的原因见 `_compute_metrics` 里的长注释：NSP 会把
        #: `domain_radius` / `grid_n` 抬升，若每次都用当前分辨率去量**固定的**
        #: 参照系，会比出不同的值（实测 1.5992 vs 0.9995，而 1.0 是成功线
        #: ⇒ 分辨率能翻转结论）。缓存后整个过程用**同一把尺子**。
        self._reference_ratio_cache: Optional[float] = None

        # 随机种子。见 __init__ 的 docstring：主循环里唯一的随机来源是 NSP
        # 因子化，不固定它则整条证明路径不可复现。
        self.seed: Optional[int] = None if seed is None else int(seed)
        #: 派生 seed 用的"已发生多少次 NSP 调用"计数。每次调用递增，
        #: 使同一次运行内的多次因子化用**不同**的变换（否则每次都在同一组
        #: 副本上打转），同时整条路径仍完全由 base seed 决定。
        self._nsp_calls = 0

        # 体积估计的「域」。基准是历史的 [-1,1]³；管族不越界时**永远等于基准**，
        # 所以历史数值逐位不变。越过基准时才抬升为 2 的幂（见 domain.py 的三条不变量）。
        self.domain_radius: float = radius_covering(self.tubes, floor=DEFAULT_RADIUS)
        self.domain_radius_explicit = False

        # 默认 Move 参数
        self.zeta = self.move_params.get("zeta", 0.01)
        self.omega = self.move_params.get("omega", 0.1)
        self.tube_ratio = self.move_params.get("tube_ratio", None)
        # ``error_mode`` 默认 **"actual"**（2026-10-01 从 "proxy" 改）。
        #
        # 为什么改默认：Move #3 的后条件是论文 Lemma 8.3 的
        # ``C_KT-CW^loc(G) <= δ^{-ζ}``；"proxy" 用的是 ``n|T|/|G|``——
        # **两个不同的量**，proxy 满足 ≠ 论文后条件成立。实测（
        # 内部分批测量脚本，8 构型）两者**每次都给出
        # 相反结论**：proxy 报"不满足"（max_err 2.9~7.8），actual 报"满足"
        # （max_err 1.0~1.25）⇒ **proxy 不是保守上界，而是系统性偏悲观**。
        # 编排器路径**始终有 tubes**（``_apply_move`` 传 ``state.tubes``），
        # 故 actual 无额外前提；端到端实测代价仅 ~2%
        # （内部分批测量脚本：45.1s → 46.0s），
        # 且两条路径的**可观测结论完全相同**（同 6 步、同 success/scope）。
        #
        # 要恢复旧行为的部署可显式传 ``move_params={"error_mode": "proxy"}``。
        self.error_mode = self.move_params.get("error_mode", "actual")

    @staticmethod
    def _default_weights_candidates() -> List[Path]:
        """surrogate 权重的默认探测路径（按优先级）。

        两种部署形态，路径不同，都要能工作：

        1. **仓库内运行**（开发者、CI）：权重在 ``<repo>/data/``。
           ``parents[2]`` 从 ``<repo>/src/kakeya_ai/proof_orchestrator.py``
           上溯到 ``<repo>``。
        2. **已安装**（``pip install``）：``src/`` 布局消失，权重被
           ``[tool.setuptools.data-files]`` 装到 ``sys.prefix/share/kakeya_ai/``。

        此前只探测第 1 种，于是装成包后**静默退回启发式**——功能看着正常、
        实际少了一层，这种失败最难发现。两处都探，缺了才算真缺失。
        """
        here = Path(__file__).resolve()
        candidates = [
            # 1) 仓库布局：<repo>/src/kakeya_ai/proof_orchestrator.py → <repo>/data
            here.parents[2] / "data" / "surrogate_scorer_weights.json",
            # 2) 已安装布局：<prefix>/share/kakeya_ai/
            Path(sys.prefix) / "share" / "kakeya_ai" / "surrogate_scorer_weights.json",
        ]
        # 3) 用户级安装（pip install --user / pipx）可能落在 userbase。
        #
        # 用 getattr 探测而不是直接调：``site.getuserbase`` 在极精简的
        # Python 构建（如某些嵌入式分发）里可能不存在。这里给探测结果**显式
        # 标注类型**，否则 getattr(..., None) 会让它退化成 ``object``，
        # 后续 ``Path(object)`` 直接过不了类型检查。
        user_base = getattr(site, "getuserbase", None)
        if callable(user_base):
            base = str(user_base())
            candidates.append(
                Path(base) / "share" / "kakeya_ai" / "surrogate_scorer_weights.json"
            )
        return candidates

    @classmethod
    def _load_surrogate(cls, weights_path: Optional[str]) -> Optional["SurrogateScorer"]:
        """Lazy 加载 surrogate 权重；任何失败都静默退回 None（启发式）。"""
        candidates: List[Path] = []
        if weights_path is not None:
            candidates.append(Path(weights_path))
        else:
            candidates.extend(cls._default_weights_candidates())
        try:
            from .ai.surrogate_scorer import SurrogateScorer

            for path in candidates:
                if path.is_file():
                    try:
                        return SurrogateScorer.load(str(path))
                    except Exception:
                        continue
        except Exception:
            pass
        return None

    def _choose_volume_fn(self):
        """根据 use_nano 选择体积估计函数。

        **显式传入域**：体积必须在同一个域上算，否则不同轮的数值不可比。
        本方法内不读 ``self``，只依赖参数——便于外部测试核验。
        """
        from . import accel
        if self.use_nano:
            return lambda tubes, grid_n, bounds=None: accel.volume_voxel(
                tubes, grid_n=grid_n, backend="nano", bounds=bounds
            )
        return lambda tubes, grid_n, bounds=None: accel.volume_voxel(
            tubes, grid_n=grid_n, backend="numpy", bounds=bounds
        )

    def _assert_domain_covers(self, tubes: Sequence[Tube], *, where: str) -> None:
        """域哨兵：若管族跑出当前域，**扩域并留痕**。

        体积估计在给定盒上采样；盒装不下的管会被 ``Tube.contains`` 当盒外点
        忽略——体积偏小而不报错。所以每次管族变化后都该问一句。

        这里只扩不缩（见 ``domain.py`` 不变量 2）：中段回缩会让前后段数值不可比。
        """
        need = exceeds_domain(tubes, self.domain_radius)
        if need is None:
            return
        before = self.domain_radius
        self._configure_domain(tubes)
        self._diagnostics.append(
            f"domain expanded at {where}: radius {before:g} → {self.domain_radius:g} "
            f"(tubes need up to {need:.6g}); grid_n={self.grid_n}"
        )

    def _configure_domain(self, tubes: Sequence[Tube]) -> None:
        """根据当前管族调整域——只**扩不缩**，且只在越过基准时抬升。

        为什么要有这一步：NSP 因子化会把副本撒到 ``±radius``（默认 radius=2.0），
        管族于是跑出 ``[-1,1]³``。这时若不扩域，盒外的管会被 ``Tube.contains``
        当作盒外点忽略——**体积静默偏小**（实测 630 根里 509 根在盒外，
        ``volume_ratio`` 从真实的 1.404 假性跌到 0.254）。

        为什么不按管族紧贴地取盒（我早期试过的做法）：紧致盒坐标**不是** f32
        精确值，nano(f32) 与 numpy(f64) 的体素网格会错位，破坏“写出去能复核”
        （实测差异 ~1e-8）。抬升到 2 的幂则两种精度都精确，网格逐位对齐。

        越界时同时把 ``grid_n`` 抬到不小于 64 的 2 的幂：域变大后若网格数不变，
        空间分辨率会下降（体积估计变粗）。抬升后“单位体积上的采样密度”不降。
        """
        need = radius_covering(tubes, floor=self.domain_radius)
        if need > self.domain_radius:
            self.domain_radius = need
            self.grid_n = max(self.grid_n, 64)
            # grid_n 也取 2 的幂：与 2 的幂半径配合可保跨后端网格逐位相同。
            self.grid_n = int(2 ** int(np.ceil(np.log2(self.grid_n))))

    def initialize(self) -> ProofState:
        """构造初始证明状态。"""
        if not self.tubes:
            return ProofState([], None, None, self.epsilon, self.epsilon)
        delta = self.tubes[0].delta
        return ProofState(
            tubes=self.tubes,
            shading=Shading(np.ones(len(self.tubes), dtype=float)),
            grains=None,
            delta=delta,
            rho=delta,
        )

    def _compute_metrics(self, state: ProofState) -> ProofMetrics:
        """从当前状态计算可观测指标。"""
        delta = state.delta
        n_tubes = len(state.tubes)
        domain_radius = float(self.domain_radius)
        grid_n = int(self.grid_n)

        # `union_volume` / `lower_bound` = **当前状态**的量（**可复核**：
        # `verify_trace` 按 `state_change` 重建中间管族后重算，必须逐位一致）。
        union_vol = self._volume_fn(
            state.tubes,
            grid_n=grid_n,
            bounds=bounds_for_radius(domain_radius),
        )
        lower_bound = theorem_1_2_lower_bound(
            state.tubes,
            epsilon=self.epsilon,
            K=self.K,
            lambda_min=self.lambda_min,
        )
        # 判据的**参照系**：原输入（递归承接时是该批窄簇）。
        #
        # **分子分母都必须取参照系**（2026-09-30 二次修正）。第一版只改了
        # 分母，于是 NSP 一扩管（16 → 400 根）分子就暴增，`vol_ratio` 从
        # 1.60 变成 **32.7** —— 仍然是"换了判据"，只是换在分子上。
        # 实测（`mild-spread(16)`）：修分母后仍被 NSP 吹到 32.7，
        # 随后两个 Move **没改任何指标**、却报 `TERMINAL_SUCCESS`。
        #
        # 注意与 `union_volume` **字面值**的区别：那个字段仍是当前状态的
        # （复核需要），这里是**判据用的**参照量的体积。
        ref = list(self._reference_tubes) if self._reference_tubes else list(state.tubes)
        # **参照系的量按参照系的域/分辨率算，且只算一次并缓存**。
        #
        # 为什么不能沿用当前的 `domain_radius` / `grid_n`：NSP 扩管时会把域
        # 从 1.0 抬到 4.0、grid 从 8 抬到 64（见 `_try_nsp_factorization`），
        # 而**参照系管族没变**——用一个变了的分辨率去量同一个固定的对象，
        # 得到的是**测量差异**，不是数学变化。实测：同一参照系的比值在 NSP
        # 前后从 **1.5992 变成 0.9995**，而 1.0 恰好是成功线 ⇒ **分辨率能
        # 翻转结论**。这显然不能留。
        #
        # 缓存到实例上（参照系在 `__init__` 里设定后不再变），于是
        # `volume_ratio` 在整个搜索过程中是**同一把尺子**量出来的。
        if self._reference_ratio_cache is None:
            ref_union = float(self._volume_fn(
                ref, grid_n=grid_n, bounds=bounds_for_radius(domain_radius),
            ))
            ref_lb = theorem_1_2_lower_bound(
                ref, epsilon=self.epsilon, K=self.K, lambda_min=self.lambda_min,
            )
            self._reference_ratio_cache = (
                ref_union / ref_lb if ref_lb > 0 else 0.0
            )
        volume_ratio = self._reference_ratio_cache

        grains = state.grains
        if grains and grains.prisms:
            a = float(np.mean([p.a for p in grains.prisms]))
            b = float(np.mean([p.b for p in grains.prisms]))
            c = float(np.mean([p.c for p in grains.prisms]))
            rho = b / c if c > 0 else delta
            tube_vol = state.tubes[0].volume if state.tubes else 1.0
            local_errors = [
                len(p.tube_indices) * tube_vol / p.volume
                for p in grains.prisms
                if p.volume > 0
            ]
            local_error = float(max(local_errors)) if local_errors else 0.0
            n_grains = grains.count
        else:
            a = b = c = delta
            rho = delta
            local_error = 0.0
            n_grains = 0

        is_sticky = False
        if n_tubes > 0:
            try:
                scale_chain = build_scale_chain(state.tubes, delta=delta)
                is_sticky = scale_chain.is_sticky
            except Exception as exc:
                # 判定失败 ≠ 判定为「否」：记入诊断供 step() 转写进 trace。
                self._diagnostics.append(
                    f"sticky check failed ({type(exc).__name__}: {exc}); "
                    "treated as not sticky"
                )
                is_sticky = False

        return ProofMetrics(
            delta=delta,
            rho=rho,
            a=a,
            b=b,
            c=c,
            local_error=local_error,
            volume_ratio=volume_ratio,
            union_volume=union_vol,
            lower_bound=lower_bound,
            n_tubes=n_tubes,
            n_grains=n_grains,
            is_sticky=is_sticky,
            domain_radius=domain_radius,
            grid_n=grid_n,
        )

    def _build_grains(self, state: ProofState) -> GrainsDecomposition:
        """构建 grains 分解。"""
        shading = state.shading
        if shading is None and state.tubes:
            shading = Shading(np.ones(len(state.tubes), dtype=float))
        return guth_grains_skeleton(
            state.tubes,
            shading,
            grid_n=2,
            adaptive_rho=True,
            min_tubes_per_bin=4,
        )

    def _describe_state_change(
        self, before: "ProofState", after: "ProofState"
    ) -> StateChange:
        """描述“管族如何从 before 得到 after”——复核方靠它重建中间状态。

        用**对象标识**而不是几何比较：同一根管在状态间传递时不重新构造，
        所以 ``id`` 比对既快又准（几何比较会因末位浮点差异误判）。四种情形：

        - ``keep``：管族与尺度都没变（grains/Move 只改分解，不改管族）。
        - ``subset``：只是取了一部分管（broadness 剥离），记下索引。
        - ``rescale``：管族逐根对应、但尺度变了（递归 δ/2）。
        - ``replace``：其余情况（如 NSP 因子化）——**直接带上整份新管族**，
          宁可文件大一点，也不要让复核方猜。
        """
        from .trace_format import _tube_payload   # 惰性导入：避免模块级循环依赖

        if after.delta != before.delta:
            # 已先比长度再逐根比 identity，故 strict 恒成立；写上它使
            # “这里不可能长度不等”成为代码里可见的断言，而非读者的心算。
            if (len(after.tubes) == len(before.tubes)
                    and all(a is b for a, b in zip(after.tubes, before.tubes,
                                                   strict=True))):
                # 记**目标 delta 本身**而不是倍率：倍率相乘可能有舍入分歧，
                # 直接用目标值才能逐位重建。hex 供精确解析，十进制供人读。
                return {
                    "op": "rescale",
                    "delta": float(repr(float(after.delta))),
                    "delta_hex": float(after.delta).hex(),
                    "factor": after.delta / before.delta,
                }
            return {"op": "replace", "tubes": [_tube_payload(t) for t in after.tubes]}

        if len(after.tubes) == len(before.tubes) and all(
            a is b for a, b in zip(after.tubes, before.tubes, strict=True)
        ):
            return {"op": "keep"}

        # 是不是子集？（broadness 剥离）
        index_of = {id(t): i for i, t in enumerate(before.tubes)}
        indices = [index_of.get(id(t), -1) for t in after.tubes]
        if indices and all(i >= 0 for i in indices):
            return {"op": "subset", "indices": indices}

        return {"op": "replace", "tubes": [_tube_payload(t) for t in after.tubes]}

    def _apply_broadness_filter(self, state: ProofState) -> Optional[ProofState]:
        """对当前管族做 broadness 前置过滤，只在需要时返回新状态。

        返回 None 表示无需过滤（未启用、已过滤过、或本来就 broad）。
        返回新状态时，``tubes`` 只保留 broad 子集，narrow 簇以原始索引
        记录在 ``narrow_batches`` 中，grains 被重建。
        """
        if not self.broadness_filter or state.broadness_checked:
            return None
        if len(state.tubes) < 2:
            return None

        try:
            reduction = broadness_reduction(
                state.tubes,
                K=self.broadness_K,
                beta=self.broadness_beta,
                n_points=self.broadness_points,
            )
        except Exception as exc:
            # 过滤失败不应阻塞主循环；按"未过滤"处理，但必须留痕。
            self._diagnostics.append(
                f"broadness filter failed ({type(exc).__name__}: {exc}); "
                "input kept unfiltered"
            )
            return None

        self.last_broadness = reduction
        if not reduction.narrow_batches:
            # 本来就 broad（或无法进一步剥离）：标记已检查，避免重复采样。
            return ProofState(
                tubes=state.tubes,
                shading=state.shading,
                grains=state.grains,
                delta=state.delta,
                rho=state.rho,
                iteration=state.iteration,
                narrow_batches=state.narrow_batches,
                broadness_checked=True,
            )

        kept = list(reduction.broad_indices)
        if len(kept) < 2:
            # 剥离后不足两根管，broadness 检查失去意义：保持原样。
            return ProofState(
                tubes=state.tubes,
                shading=state.shading,
                grains=state.grains,
                delta=state.delta,
                rho=state.rho,
                iteration=state.iteration,
                narrow_batches=state.narrow_batches,
                broadness_checked=True,
            )

        new_tubes = [state.tubes[i] for i in kept]
        new_shading = (
            None
            if state.shading is None
            else Shading(state.shading.densities[kept])
        )
        return ProofState(
            tubes=new_tubes,
            shading=new_shading,
            grains=None,  # 管族变化后必须重建 grains
            delta=state.delta,
            rho=state.rho,
            iteration=state.iteration,
            narrow_batches=state.narrow_batches + list(reduction.narrow_batches),
            broadness_checked=True,
        )

    def _scores_to_action(
        self, scores: Dict[int, float]
    ) -> Optional[ProofAction]:
        """把 Move 分数转成动作；分数里没有可用的 Move 时返回 None。

        抽成一处是因为 surrogate 与 beam scorer **共用同一张
        ``_SURROGATE_MOVE_ACTIONS`` 映射**——两份实现会在加新 Move 时漂移。
        """
        best_move = max(scores, key=lambda m: scores.get(m, 0.0))
        action_name = _SURROGATE_MOVE_ACTIONS.get(best_move)
        return ProofAction[action_name] if action_name is not None else None

    def _try_llm_scorer(
        self, state: ProofState, metrics: ProofMetrics
    ) -> Optional[Tuple[ProofAction, float, str]]:
        """用注入的 LLM 打分器打分；置信度够高才返回 (动作, gap, source)。

        **为什么与 surrogate 并列**而不是串在它后面：两者是解决同一问题的
        两条独立路径（本地小模型 vs LLM），谁更可信取决于部署环境与状态
        分布。让调用方选一条，而不是硬编码优先级——否则"接了 LLM"这件事
        会静默改变已有部署的决策路径。

        **为什么 LLM 策略是一个参数而不是两个实现**：``beam_search_score``
        与 ``cascaded_score`` 的签名完全兼容（同吃 ``(prompt, state=...)``、
        同返回 ``(scores, source)``），差别只在内部策略——前者是"快模型出
        候选、慢模型评最优"，后者是"快模型起头、慢模型续推理"。两套几乎
        相同的接线代码必然漂移，故用 ``llm_strategy`` 选一个、共用这段逻辑。

        任何失败（LLM 不可达、返回格式不符、调用超时）都记入 ``_diagnostics``
        并返回 None，**不抛出**：主循环必须能继续走启发式，而不是因为外部
        服务抖动就整个 run 失败。
        """
        if self.llm_scorer is None:
            return None

        from .ai.beam_search import _top_two_gap
        from .ai.prompts import format_branch_scoring
        from .ai.training_data import state_from_metrics

        state_dict = state_from_metrics(metrics)
        prompt = format_branch_scoring(
            assertion="Theorem 1.2",
            delta=metrics.delta,
            rho=metrics.rho,
            a=metrics.a,
            b=metrics.b,
            c=metrics.c,
            local_error=metrics.local_error,
            volume_ratio=metrics.volume_ratio,
        )
        self._last_llm_prompt = prompt
        try:
            result = self.llm_scorer(prompt, state=state_dict)
        except Exception as exc:
            self._diagnostics.append(
                f"LLM scorer ({self.llm_strategy}) failed "
                f"({type(exc).__name__}: {exc}); falling back to heuristic"
            )
            return None

        # 两个打分器都返回 ``(scores, source)``；容忍只返回 scores 的桩实现
        # （测试常这样写），但**不猜** source——没有就如实标 "unknown"。
        if isinstance(result, tuple) and len(result) == 2:
            scores, source = result
        else:
            scores, source = result, "unknown"

        if not isinstance(scores, dict) or not scores:
            self._diagnostics.append(
                f"LLM scorer ({self.llm_strategy}) returned unusable scores "
                f"({type(scores).__name__}); falling back to heuristic"
            )
            return None

        # ``_top_two_gap`` 返回 ``(gap, best_move, second_move)``。
        top_gap, _, _ = _top_two_gap(scores)
        if top_gap < self.llm_threshold:
            return None
        action = self._scores_to_action(scores)
        if action is None:
            return None
        return action, top_gap, str(source)

    def _select_move(self, state: ProofState, metrics: ProofMetrics) -> ProofAction:
        """选择下一个 Move：默认启发式，可选 LLM / surrogate 打分器门控。

        打分器优先级：**LLM 打分器优先于 surrogate**（显式注入的 LLM 比本地
        小模型更值得信任，且它是调用方主动配置的），其次 surrogate；
        两者都不可用或置信度不足时走启发式。

        命中时备注写入 ``self._selection_notes``，使 trace 里能看出"这个 Move
        是打分器选的还是启发式选的"——否则事后无法判断决策来源。

        来源同时记入 ``self._last_selection_source``（枚举，供 ``step()``
        写进 ``ProofStep.selection_source``）：备注是给人看的细节，枚举是给
        机器数的占比。
        """
        self._selection_notes = []
        self._last_selection_source = SelectionSource.NOT_APPLICABLE
        grains = state.grains
        if grains is None or not grains.prisms:
            return ProofAction.GRAINS_DECOMPOSITION

        # 先算三个 Move 的条件是否满足 + 状态指纹。
        #
        # **必须在打分器之前算**：打分器（LLM / surrogate）此前**完全绕过**
        # 这些条件——实测后果是 surrogate 选出的 Move 有效率只有 **12%**
        # （36 次空转 / 5 次有效），而启发式是 33%。根因是它为"条件已满足"
        # 的状态选了对应的 Move（如 `m2_ok=True` 却选 MOVE2 → 100% 空转）。
        # 详见下方 `_surrogate_action_allowed`。
        m1_ok = check_move1_condition(grains, metrics.rho, metrics.delta, self.zeta, self.tube_ratio)
        m2_ok = check_move2_condition(grains, self.omega, metrics.delta)
        m3_ok = check_move3_condition(grains, 1.0 / (metrics.delta ** self.zeta))
        fingerprint = self._state_fingerprint(metrics)
        conditions = {
            ProofAction.MOVE1_LONGER_GRAINS: m1_ok,
            ProofAction.MOVE2_REPLACE_SQUARE_GRAINS: m2_ok,
            ProofAction.MOVE3_WIDER_GRAINS: m3_ok,
        }

        # 1. LLM 打分器（需调用方显式注入）。
        llm = self._try_llm_scorer(state, metrics)
        if llm is not None:
            action, gap, source = llm
            if self._scored_action_allowed(action, conditions, fingerprint):
                self._last_selection_source = SelectionSource.LLM
                self._scorer_gate_accepts["LLM"] += 1
                self._selection_notes.append(
                    f"move selected by LLM scorer "
                    f"(strategy={self.llm_strategy}, source={source}, gap={gap:.3g})"
                )
                return action
            self._scorer_gate_blocks["LLM"] += 1
            self._selection_notes.append(
                f"LLM 选的 {action.name} 被守卫拦下"
                f"（{'条件已满足' if conditions.get(action) else '同状态下试过且无效'}）"
                f"，退回启发式"
            )

        # 2. surrogate 置信度门控的 Move 选择。
        if self.surrogate is not None:
            # 状态 dict 由 ``training_data.state_from_metrics`` 统一构造——特征的
            # 取法只有一份。此前这里手写 4 个字段，与训练端各写一份，扩维时
            # 必然漂移（训练用 A 组、推理用 B 组，模型静默给出无意义分数）。
            from .ai.training_data import state_from_metrics

            state_dict = state_from_metrics(metrics)
            try:
                scores = self.surrogate.predict_scores(state_dict)
                gap = self.surrogate.confidence(scores)
            except Exception as exc:
                scores, gap = None, 0.0
                self._diagnostics.append(
                    f"surrogate prediction failed ({type(exc).__name__}: {exc}); "
                    "falling back to heuristic"
                )
            if scores is not None and gap >= self.surrogate_threshold:
                # 用显式取值而不是 ``key=scores.get``：后者的签名含
                # ``-> float | None`` 这一重载，类型检查会因"None 不可比较"
                # 而报错。这里的 scores 保证非空（predict_scores 始终给出
                # 三个 Move），但仍显式给默认值——不依赖"键一定在"这个隐含前提。
                action = self._scores_to_action(scores)
                if action is not None and self._scored_action_allowed(
                    action, conditions, fingerprint
                ):
                    self._last_selection_source = SelectionSource.SURROGATE
                    self._scorer_gate_accepts["SURROGATE"] += 1
                    self._selection_notes.append(
                        f"move selected by surrogate (gap={gap:.3g})"
                    )
                    return action
                if action is not None:
                    # 打分器给了答案但被守卫拦下 → 显式记下，否则"surrogate
                    # 覆盖率"的统计会把这次算成"生效"（实际没生效）。
                    self._scorer_gate_blocks["SURROGATE"] += 1
                    self._selection_notes.append(
                        f"surrogate 选的 {action.name} 被守卫拦下"
                        f"（{'条件已满足' if conditions.get(action) else '同状态下试过且无效'}）"
                        f"，退回启发式"
                    )

        # 走到这里就是**启发式兜底**——必须显式标出来。它的占比正是
        # “打分器还没覆盖住实际状态分布”的度量（文档把降低这个占比当作
        # 训练闭环的目标），所以绝不能漏记或误记为 NOT_APPLICABLE。
        self._last_selection_source = SelectionSource.HEURISTIC

        # 简单启发式：哪个条件差就优先修哪个
        #
        # **但要先排除"在相同状态下试过且无效"的 Move**（调度状态机的记忆）。
        # 没有这一步时，实测会出现 `MOVE2` 连续重试 10 次而指标逐位不变
        # 的纯空转（12 根随机构型）。
        candidates = []
        if not m1_ok:
            candidates.append(ProofAction.MOVE1_LONGER_GRAINS)
        if not m3_ok and metrics.local_error > 1.0:
            candidates.append(ProofAction.MOVE3_WIDER_GRAINS)
        if not m2_ok:
            candidates.append(ProofAction.MOVE2_REPLACE_SQUARE_GRAINS)

        for action in candidates:
            if self._is_known_futile(action, fingerprint):
                self._selection_notes.append(
                    f"skipped {action.name}: 已在相同状态试过且指标无变化"
                )
                continue
            return action

        if candidates:
            # 所有候选都试过且无效 → 不再返回它们（否则就是空转）。
            self._selection_notes.append(
                f"all candidate moves exhausted (tried & futile): "
                f"{[a.name for a in candidates]}"
            )
            self._last_failure_reason = FailureReason.NO_PROGRESS
            return ProofAction.LOWER_BOUND_VERIFY

        # 都满足则尝试递归细化
        return ProofAction.RECURSION

    @staticmethod
    def _state_fingerprint(metrics: "ProofMetrics") -> tuple:
        """把指标压成可比较的"状态指纹"（用于判断"同状态是否试过"）。

        只取**对 Move 有效性有判别力**的量：rho / local_error / n_grains /
        volume_ratio。用精确值而非容差——本项目对同一输入要求逐位可复现，
        容差会引入"看起来一样其实不同"的判断。
        """
        return (
            float(metrics.rho),
            float(metrics.local_error),
            float(metrics.volume_ratio),
            int(metrics.n_grains),
            float(metrics.delta),
        )

    def _scored_action_allowed(
        self,
        action: ProofAction,
        conditions: Dict[ProofAction, bool],
        fingerprint: tuple,
    ) -> bool:
        """打分器（LLM / surrogate）选的 Move 是否该被采纳。

        **为什么需要这道守卫**：打分器此前**完全绕过** Move 条件检查与失效
        记忆，直接返回。实测后果（12 个随机构型 × 3 个规模）：

        ==========  ========  ======  ========
        选择来源     空转      有效     有效率
        ==========  ========  ======  ========
        SURROGATE   36        5       **12%**
        HEURISTIC   2         1       33%
        ==========  ========  ======  ========

        surrogate 的 Move 选择**比启发式还差**——根因是它为"条件已满足"的
        状态选了对应 Move（如 ``m2_ok=True`` 却选 MOVE2），而那必然是空转
        （Move 的作用正是让条件从不满足变为满足）。

        两条判据（与启发式路径**完全一致**，这样三条路径才可比）：

        1. **条件未满足才值得做**：``conditions[action]`` 为 True 时该 Move
           已经不需要了——选它只会空转；
        2. **同状态下试过且无效则不重试**（复用 :meth:`_is_known_futile`）。

        被拦下**不算错误**：调用方退回启发式，且 :attr:`_selection_notes`
        会显式记下"被守卫拦下"——否则 surrogate 覆盖率统计会把这次算成"生效"
        （实际没生效，是"给了答案但没用"）。
        """
        if action not in conditions:
            # 不是三个 Move 之一（如直接给了 RECURSION）——打分器不负责这类
            # 决策，交给启发式。如实不放行，但不记为"条件已满足"。
            return False
        if conditions[action]:
            return False
        if self._is_known_futile(action, fingerprint):
            return False
        return True

    def scorer_gate_stats(self) -> Dict[str, Dict[str, float]]:
        """打分器给答案的"有效率"，按来源（LLM / SURROGATE）分列。

        这是 :func:`~kakeya_ai.ai.surrogate_coverage_with_gate` 的**在线对应物**：
        那个函数在离线状态集上算，本方法在一次真实运行后算——**且两者口径
        一致**（都问"给了答案且答案可用"的比例，而不是"敢不敢给答案"）。

        Returns
        -------
        dict
            ``{"SURROGATE": {"accepts": int, "blocks": int, "offered": int,
            "effectiveness": float}, "LLM": {...}}``。
            ``offered = accepts + blocks``，是打分器**真的给出候选**的次数；
            ``effectiveness = accepts / offered``，即"给了答案且能用"的占比。
            ``offered == 0`` 时 ``effectiveness`` 为 ``0.0``（**不返回 nan**：
            调用方通常要拿它做比较或累加，nan 会静默污染下游统计）。

        Notes
        -----
        ``offered`` 与 ``SelectionSource`` 的占比**不是一回事**：后者只统计
        "最终谁做的决定"，所以被拦下的那些会算到 HEURISTIC 头上——从占比看
        不出"打分器试过但没用"。本方法补上这一维。
        """
        out: Dict[str, Dict[str, float]] = {}
        for src, accepts in self._scorer_gate_accepts.items():
            blocks = self._scorer_gate_blocks.get(src, 0)
            offered = accepts + blocks
            out[src] = {
                "accepts": float(accepts),
                "blocks": float(blocks),
                "offered": float(offered),
                "effectiveness": (accepts / offered) if offered else 0.0,
            }
        return out

    def move_is_admissible(
        self, action: int, grains, metrics: "ProofMetrics"
    ) -> bool:
        """离线度量用的"该 Move 会不会被采纳"判据。

        **存在的理由**：:func:`~kakeya_ai.ai.surrogate_coverage` 度量的是
        "surrogate 敢不敢给答案"（top-two gap ≥ 阈值），**不含**这个 Move
        是否可用。两者实测不等（修复前 88% 的自信用答案是空转，会被守卫
        拦下），所以度量"实际覆盖"需要一个与推理**同源**的判据。

        **本方法与 :meth:`_scored_action_allowed` 共用同一实现**——它就是把
        :meth:`_select_move` 内部那套"算 conditions 再判守卫"原样搬出来，
        而不是另写一份（本项目已多次因"两处各写一份"导致口径漂移）。
        改条件或改守卫时**两处必须同时改**，守门测试会盯住这一点。

        签名与 :func:`~kakeya_ai.ai.accelerated_scorer.surrogate_coverage_with_gate`
        的 ``is_admissible(action, state)`` 不同：那个收 ``(int, dict)``，本
        方法收 ``(int, grains, metrics)``。传参时用 ``lambda a, s: ...`` 适配：
        状态字典里没有 grains 对象，条件算不出来——**所以度量必须在有 grains
        的调用点做**（如 :meth:`step` 内），不能事后从字典重建。

        Parameters
        ----------
        action : int
            1 / 2 / 3，对应三个 Move。
        grains : Grains
            真实 grains（Move 条件依赖它，无法从 state 字典重建）。
        metrics : ProofMetrics
            当前轮的指标。

        Returns
        -------
        bool
            该 Move 是否可用（条件未满足 + 同状态下未试过无效）。
        """
        reverse = {
            1: ProofAction.MOVE1_LONGER_GRAINS,
            2: ProofAction.MOVE2_REPLACE_SQUARE_GRAINS,
            3: ProofAction.MOVE3_WIDER_GRAINS,
        }
        action_enum = reverse.get(int(action))
        if action_enum is None:
            return False
        conditions = {
            ProofAction.MOVE1_LONGER_GRAINS: check_move1_condition(
                grains, metrics.rho, metrics.delta, self.zeta, self.tube_ratio
            ),
            ProofAction.MOVE2_REPLACE_SQUARE_GRAINS: check_move2_condition(
                grains, self.omega, metrics.delta
            ),
            ProofAction.MOVE3_WIDER_GRAINS: check_move3_condition(
                grains, 1.0 / (metrics.delta ** self.zeta)
            ),
        }
        return self._scored_action_allowed(
            action_enum, conditions, self._state_fingerprint(metrics)
        )

    def _is_known_futile(self, action: ProofAction,
                         fingerprint: tuple) -> bool:
        """该 Move 是否"在**这个**状态下试过且没带来任何变化"。

        条件同时成立才算无效：
        1. 曾在**相同指纹**的状态下尝试过；
        2. 那次尝试后指标**逐位不变**（``changed is False``）。

        若状态变了（指纹不同），之前"无效"的结论不适用——Move 可能在新
        状态下有效。这正是"记忆"而非"永久拉黑"的区别。
        """
        rec = self._move_attempts.get(action.name)
        return rec is not None and rec[0] == fingerprint and rec[1] is False

    def _record_attempt(self, action: ProofAction, before_fingerprint: tuple,
                        after: "ProofMetrics") -> None:
        """记录一次 Move 尝试的结果（供 :meth:`_is_known_futile` 使用）。"""
        changed = self._state_fingerprint(after) != before_fingerprint
        self._move_attempts[action.name] = (before_fingerprint, changed)

    def _apply_move(self, state: ProofState, action: ProofAction) -> ProofState:
        """应用选中的 Move，返回新状态。

        三个 Move 都经 ``_moveN_impl``（**返回真实合并轮数**），把轮数暂存到
        ``self._last_move_iters_used``（缺口二：使"动了 1 轮"与"动了 3 轮"可
        区分，而非从 grain 数推的 0/1 下界）。公共的 ``moveN_*`` 只是薄封装，
        这里不用它——否则拿不到轮数。
        """
        grains = state.grains
        if grains is None:
            grains = self._build_grains(state)

        if action == ProofAction.MOVE1_LONGER_GRAINS:
            new_grains, iters = _move1_impl(
                grains, rho=state.rho, delta=state.delta,
                zeta=self.zeta, tube_ratio=self.tube_ratio
            )
        elif action == ProofAction.MOVE2_REPLACE_SQUARE_GRAINS:
            new_grains, iters = _move2_impl(
                grains, omega=self.omega, delta=state.delta
            )
        elif action == ProofAction.MOVE3_WIDER_GRAINS:
            # ``grains.tube_indices`` 索引的是**当前状态**的管族（经 broadness
            # 过滤后可能与 ``self.tubes`` 长度不同），因此 actual 模式的误差
            # 统计必须用 ``state.tubes``；传错管族会静默算错误差。
            #
            # ``grid_n`` **必须透传** ``self.grid_n``：actual 模式的体素分辨率
            # 应与运行时其余体积估计一致。不传会静默用函数默认值 32，而当
            # ``self.grid_n`` 被域抬升改到 64 时两者不符（分辨率差异会改变
            # actual 误差的数值）。
            new_grains, iters = _move3_impl(
                grains, zeta=self.zeta, delta=state.delta,
                tubes=state.tubes, error_mode=self.error_mode,
                grid_n=self.grid_n,
            )
        else:
            new_grains, iters = grains, 0
        self._last_move_iters_used = iters

        return ProofState(
            tubes=state.tubes,
            shading=state.shading,
            grains=new_grains,
            delta=state.delta,
            rho=state.rho,
            iteration=state.iteration + 1,
            narrow_batches=list(state.narrow_batches),
            broadness_checked=state.broadness_checked,
        )

    def _recursion_plan(self, state: ProofState) -> dict:
        """用多尺度归纳判断"该不该递归、递归到多细"。

        **这是 :func:`~kakeya_ai.sticky_kakeya.multiscale_induction` 的接入点**
        ——此前编排器 import 了它却从不调用，``RECURSION`` 分支是**无脑把 δ 减半**
        （``_try_recurse`` 的旧实现），没有任何"是否需要更细尺度"的判断。后果：
        递归会一直走到 ``delta <= 1e-4`` 的硬下界才停，即使 ω 早就压不动了。

        **严格遵守 ``multiscale_induction`` 的定量边界**（其 docstring 明确警告
        "不可作 'N 步后 ω ≈ 某值' 这类定量结论"）：本函数**只取曲线的时间结构**
        ——"还在压"还是"压不动了"——用于**调度决策**，并把理由写进 notes（自由
        文本）。**绝不把 ω 的数值写进 `.kpt` 的定量字段**。

        **判定来源唯一（2026-10-09，A4）**：终止形态由
        :func:`~kakeya_ai.sticky_kakeya.classify_induction_termination` 判定——
        此前这里自己写了一份"尾部压缩量 ≈ 0 / 是否还在下降"的启发式，
        与 L2 核验（``verify_induction_l2``）各写一份，会漂移。现在只调用那
        一处，本函数负责把它的三态映射到调度语义。

        Returns
        -------
        dict，键：

        - ``should_recurse``：是否建议继续细化；
        - ``reason``：判定依据（人类可读，进 notes）；
        - ``n_steps``：归纳循环实际跑了几步（仅用于诊断）；
        - ``stalled``：是否判定为停滞；
        - ``termination``：终止形态名（``CONVERGED`` / ``STALLED`` /
          ``DESCENDING``，仅用于诊断；自由文本，不进定量字段）。
        """
        if not state.tubes:
            return {"should_recurse": False, "reason": "空管族",
                    "n_steps": 0, "stalled": True,
                    "termination": InductionTermination.STALLED.name}

        omega_threshold = float(self.move_params.get("induction_omega_threshold", 1e-4))
        try:
            steps = multiscale_induction(
                state.tubes,
                sigma_0=self.move_params.get("sigma_0", 0.5),
                omega_0=float(self.omega),
                delta=state.delta,
                max_iterations=int(self.move_params.get("induction_max_iterations", 20)),
                omega_threshold=omega_threshold,
                zeta=self.zeta,
            )
        except Exception as exc:  # 归纳失败不该中断证明搜索
            self._diagnostics.append(
                f"multiscale_induction 失败（{type(exc).__name__}: {exc}）；"
                "递归决策退回默认（允许递归）"
            )
            return {"should_recurse": True,
                    "reason": f"归纳不可用（{type(exc).__name__}），按默认允许递归",
                    "n_steps": 0, "stalled": False,
                    "termination": "UNAVAILABLE"}

        # 唯一判定来源：只按**结构**分类（收敛 / 停滞 / 仍在压）。
        termination, reason = classify_induction_termination(
            steps, omega_threshold=omega_threshold
        )

        # 映射到调度语义：
        # - CONVERGED（ω 到阈值）与 STALLED（压不动）都不该再细化尺度；
        # - DESCENDING（还在压）才继续递归。
        # ``stalled`` 只对 STALLED 为真——CONVERGED 是"达到目的"而非"卡住"，
        # 混成一个布尔会让读结果的人以为"再跑跑就好了"。
        if termination is InductionTermination.DESCENDING:
            should_recurse, stalled = True, False
        else:
            should_recurse, stalled = False, termination is InductionTermination.STALLED

        return {"should_recurse": should_recurse,
                "reason": reason,
                "n_steps": len(steps), "stalled": stalled,
                "termination": termination.name}

    def _try_recurse(self, state: ProofState) -> tuple[Optional[ProofState], dict]:
        """尝试尺度递归：先问多尺度归纳"该不该递归"，再决定细化。

        **与旧实现的区别**：旧版是"无脑 δ/2，只要 ``delta > 1e-4`` 就递归"
        ——没有"是否需要更细尺度"的判断，递归会一直撞到硬下界才停。现在先取
        :meth:`_recursion_plan` 的结论：归纳判定停滞（ω 压不动了）时**不再递归**，
        交由调用方去验下界。

        这是 ``multiscale_induction`` 从"被 import 但从不调用"变为**真正参与
        调度**的落点（此前登记为 内部设计文档已知局限 #2）。

        Returns
        -------
        (new_state or None, plan)
            ``new_state`` 为 None 表示不递归；``plan`` 是判定依据（供 notes）。
        """
        plan = self._recursion_plan(state)

        if not plan["should_recurse"]:
            return None, plan

        if state.delta <= 1e-4:
            # 硬下界：归纳说还能压，但尺度已到最小可表示粒度。
            plan = dict(plan, reason=plan["reason"] + "；但 δ 已到最小尺度下界")
            return None, plan

        new_delta = state.delta / 2.0
        new_tubes = [
            Tube(
                center=t.center,
                direction=t.direction,
                length=t.length,
                delta=new_delta,
            )
            for t in state.tubes
        ]
        # 管族变了就校验域：递归本身不移动管心，但中心本来就可能已在边界外
        # （如 NSP 后），不检查就会在下一轮悄悄漏算盒外那段。
        self._assert_domain_covers(new_tubes, where="recursion")
        new_state = ProofState(
            tubes=new_tubes,
            shading=state.shading,
            grains=None,
            delta=new_delta,
            rho=state.rho,
            iteration=state.iteration + 1,
            narrow_batches=list(state.narrow_batches),
            broadness_checked=False,
        )
        return new_state, plan

    def _next_nsp_seed(self) -> Optional[int]:
        """给本次 NSP 调用分配种子；``self.seed is None`` 时返回 ``None``。

        同一次运行内的第 k 次调用用 ``seed + k``：**既保证整条路径由 base seed
        唯一决定**（可复现），又让多次因子化不会反复使用同一组随机变换
        （否则"重新因子化"只是把同一批副本再撒一遍，策略上毫无变化）。

        ⚠️ **``_nsp_calls`` 必须在 :meth:`run` 开头归零**：k 是「**本次运行内**
        的调用序号」。不归零的话，同一实例的第二次 run() 第一处 NSP 会用
        ``seed + 1``，**生成另一组管族**——实测同一实例连跑三次是
        4 步 / 4 步 / 2 步，而三个独立实例都是 4 步。
        """
        if self.seed is None:
            return None
        value = self.seed + self._nsp_calls
        self._nsp_calls += 1
        return value

    def _try_nsp_factorization(self, state: ProofState) -> Optional[ProofState]:
        """如果 #T << δ^{-2}，尝试 NSP 因子化放大管族。

        ``nsp_factorization`` 返回 ``(boosted_tubes, result)`` **二元组**；
        早期实现误当对象访问 ``result.tubes``，异常被 ``except`` 吞掉，导致
        本分支从未真正执行过（每次只留下一行诊断）。修复后同时把实测数值
        写进诊断，使 trace 里能看出因子化到底做没做、放大多少。

        **种子必须显式传递**（2026-09-17 修复）：此前的调用没传 ``seed``，
        函数内部 ``np.random.default_rng(None)`` 每次取新熵，于是同一输入重跑
        会得到不同的副本、不同的体积比、乃至不同的动作序列。这使 trace
        的"可独立复核"落空。种子由 ``_next_nsp_seed()`` 按调用序号派生，
        并写进结果与 trace 的 ``params``。
        """
        n_tubes = len(state.tubes)
        delta = state.delta
        target_count = int(round(delta ** (-2)))
        if n_tubes >= 0.5 * target_count:
            return None
        try:
            nsp_seed = self._next_nsp_seed()
            boosted_tubes, nsp_result = nsp_factorization(
                state.tubes, delta=delta, seed=nsp_seed
            )
            self.last_nsp = nsp_result
            if boosted_tubes and nsp_result.meets_target:
                # 副本会跑到域外：先扩域（+grid_n）再落成新状态。不扩就往下走的话，
                # 下一轮 ``_compute_metrics`` 会静默漏掉盒外管。
                self._assert_domain_covers(boosted_tubes, where="nsp")
                self._diagnostics.append(
                    f"NSP factorized #T={nsp_result.original_count} → "
                    f"#T̂={nsp_result.boosted_count} "
                    f"(target δ⁻²={nsp_result.target_count:.0f}, "
                    f"copies={nsp_result.n_copies}, "
                    f"vol_ratio={nsp_result.volume_ratio:.3f}, "
                    f"domain_radius={self.domain_radius:g}, "
                    f"grid_n={self.grid_n}, seed={nsp_seed})"
                )
                return ProofState(
                    tubes=list(boosted_tubes),
                    shading=None,
                    grains=None,
                    delta=delta,
                    rho=state.rho,
                    iteration=state.iteration + 1,
                    narrow_batches=list(state.narrow_batches),
                    broadness_checked=False,
                )
            # 跑通但没达到目标：也要留痕，否则与“根本没跑”无法区分。
            self._diagnostics.append(
                f"NSP factorization ran but missed target: "
                f"#T̂={nsp_result.boosted_count} < "
                f"{0.5 * nsp_result.target_count:.0f} "
                f"(vol_ratio={nsp_result.volume_ratio:.3f}); step skipped"
            )
        except Exception as exc:
            self._diagnostics.append(
                f"NSP factorization failed ({type(exc).__name__}: {exc}); "
                "step skipped"
            )
        return None

    def step(
        self,
        state: ProofState,
        before: Optional[ProofMetrics] = None,
    ) -> tuple[ProofState, ProofAction, List[str]]:
        """执行单步并返回 (新状态, 动作, 备注)。

        ``notes`` 除了动作说明，还会带上本步累积的诊断（校验/重算失败等原因），
        保证 trace 里能区分"判定失败"与"判定为否"。

        Parameters
        ----------
        state : ProofState
            当前状态。
        before : ProofMetrics, optional
            已经算好的 ``state`` 指标；给出时不再重复计算（体积估计是主循环
            里最贵的一步，``run()`` 会预先算好传进来）。
        """
        self._diagnostics = []
        # 每步重置：只有真正的 Move 步会把它设成 True/False，其余步骤留 None。
        self._last_move_achieved_gain = None
        self._last_move_iters_used = None
        if before is None:
            before = self._compute_metrics(state)
        new_state, action, notes = self._step_impl(state, before)
        # 无论走哪条返回路径，都把本步诊断附在 notes 末尾。
        notes = list(notes) + list(self._diagnostics)
        return new_state, action, notes

    def _step_impl(
        self,
        state: ProofState,
        before: ProofMetrics,
    ) -> tuple[ProofState, ProofAction, List[str]]:
        notes: List[str] = []
        action: ProofAction

        # 每步开头清空归因槽位：否则上一步写下的原因会残留，让"本步终止"
        # 被误归因成上一步的类型。
        self._last_failure_reason = None
        # 同理清空决策来源：本步若是 broadness/grains/NSP/sticky 等不选 Move
        # 的动作，来源必须如实为 NOT_APPLICABLE，而不是继承上一步的
        # "LLM 选的"。默认值就是 NOT_APPLICABLE，直接重置即可。
        self._last_selection_source = SelectionSource.NOT_APPLICABLE

        # 1. broadness 前置过滤：把方向集中的窄簇剥离出去。
        #    Wang–Zahl 中 broadness reduction 是最外层的结构化简——窄构型
        #    直接转化为"小帽上的归纳"，因此必须先于 sticky 判定执行；
        #    否则宽松的 sticky 阈值会把所有构型提前短路，该步骤永不触发。
        broad_state = self._apply_broadness_filter(state)
        if broad_state is not None:
            if len(broad_state.tubes) != len(state.tubes):
                reduction = self.last_broadness
                notes.append(
                    f"broadness filter stripped {len(state.tubes) - len(broad_state.tubes)} "
                    f"narrow tube(s) in {0 if reduction is None else len(reduction.narrow_batches)} "
                    f"batch(es); {len(broad_state.tubes)} broad tube(s) remain"
                )
                if reduction is not None and reduction.stalled:
                    notes.append("broadness filter stalled: remaining set is not broad")
                return broad_state, ProofAction.BROADNESS_REDUCTION, notes
            # 未发生剥离：仅标记已检查。管族与 grains 未变，传入的指标仍然有效。
            state = broad_state

        # 2. sticky 检查
        if before.is_sticky:
            action = ProofAction.STICKY_CHECK
            notes.append("sticky configuration detected")
            # sticky 情形直接验证下界即可
            if before.volume_ratio >= 1.0:
                action = ProofAction.TERMINAL_SUCCESS
            else:
                action = ProofAction.TERMINAL_FAILURE
                self._last_failure_reason = FailureReason.STICKY_LOWER_BOUND_UNMET
            return state, action, notes

        # 3. NSP 因子化：如果管数不够 δ^{-2}
        nsp_state = self._try_nsp_factorization(state)
        if nsp_state is not None:
            action = ProofAction.NSP_FACTORIZATION
            notes.append("NSP factorization applied to reach target tube count")
            return nsp_state, action, notes

        # 3. grains 分解
        if state.grains is None:
            grains = self._build_grains(state)
            state = ProofState(
                tubes=state.tubes,
                shading=state.shading,
                grains=grains,
                delta=state.delta,
                rho=state.rho,
                iteration=state.iteration,
                narrow_batches=list(state.narrow_batches),
                broadness_checked=state.broadness_checked,
            )
            action = ProofAction.GRAINS_DECOMPOSITION
            notes.append(f"built grains: count={grains.count}")
            return state, action, notes

        # 4. 选择并应用 Move
        action = self._select_move(state, before)
        notes.extend(self._selection_notes)
        if action in (
            ProofAction.MOVE1_LONGER_GRAINS,
            ProofAction.MOVE2_REPLACE_SQUARE_GRAINS,
            ProofAction.MOVE3_WIDER_GRAINS,
        ):
            # 记录"这个 Move 在**当前指标状态**下是否带来变化"——供下一次
            # `_select_move` 跳过"同状态下试过且无效"的 Move。这是调度从
            # "无状态启发式"变为"有记忆的状态机"的落点。
            before_fp = self._state_fingerprint(before)
            new_state = self._apply_move(state, action)
            after_metrics = self._compute_metrics(new_state)
            self._record_attempt(action, before_fp, after_metrics)
            if self._state_fingerprint(after_metrics) == before_fp:
                notes.append(
                    f"{action.name} 未改变指标（rho/local_error/n_grains/"
                    "volume_ratio 逐位不变）——该 Move 在此状态下无效"
                )
            notes.append(f"applied {action.name}")
            # 缺口二：Move **单次执行**是否达成定量增益（对着它自己的目标量）。
            # 必须记录，否则"应用了但没推进"（= `NO_PROGRESS` 的微观来源）与
            # "应用了并推进了"在 trace 里分不开。实测 Move #2 空操作率 100%、
            # Move #1 空操作率 13/16（内部分批测量脚本）。
            gain_kind = _MOVE_TARGET_KEY.get(action)
            if gain_kind is not None:
                improved = _target_improved(gain_kind, state.grains, new_state.grains)
                # 同时落到**结构化字段**（供 `run()` 写进 ProofStep / trace），
                # 而不只是 notes 里的自由文本——消费端要能**机器判定** no-op。
                self._last_move_achieved_gain = bool(improved)
                notes.append(
                    f"{action.name} target={gain_kind} "
                    f"achieved_gain={improved}"
                    + ("" if improved else "（本次未推进：前置条件已满足或无法再合并）")
                )
            # 缺口二（补强）：Move 本步**实际执行的合并轮数**（精确值）。
            # `achieved_gain` 只说"动没动"，轮数说"动了几轮"——把噪声/空操作
            # 与真正的多轮推进分开（``iters_used`` 由 ``_apply_move`` 经
            # ``_moveN_impl`` 取得，非从 grain 数推的 0/1 下界）。
            if self._last_move_iters_used is not None:
                notes.append(f"{action.name} iters_used={self._last_move_iters_used}")
            # Move #3 的后条件口径（``proxy`` / ``actual``）**必须随步记录**：
            # 它决定 ``satisfied`` 是对着哪个量判的（缺口一）。只有 Move #3 有
            # 两种口径，其余 Move 的后条件无此二义。
            if action == ProofAction.MOVE3_WIDER_GRAINS:
                notes.append(f"move3 error_mode={self.error_mode}")
            return new_state, action, notes

        # 5. 递归细化
        if action == ProofAction.RECURSION:
            recurse_state, plan = self._try_recurse(state)
            # 把归纳的判定依据写进 notes（自由文本，**不写进 .kpt 的定量字段**
            # ——multiscale_induction 的 docstring 明确警告其输出不可作定量结论）。
            notes.append(f"recursion plan: {plan['reason']}")
            if recurse_state is not None:
                notes.append(f"recursed to finer scale δ={recurse_state.delta:.4g}")
                return recurse_state, ProofAction.RECURSION, notes
            # 上面已 return，故这里不必再套 ``else``（RET505）——
            # 少一层嵌套，两条路径的先后关系反而更直白。
            notes.append("recursion not possible; verifying lower bound")
            action = ProofAction.LOWER_BOUND_VERIFY
            if before.volume_ratio >= 1.0:
                action = ProofAction.TERMINAL_SUCCESS
            else:
                action = ProofAction.TERMINAL_FAILURE
                self._last_failure_reason = FailureReason.RECURSION_EXHAUSTED
            return state, action, notes

        # 6. 直接验下界
        #
        # `_select_move` 在"三个 Move 都在同状态下试过且无效"时会走这条路
        # ——此时继续迭代只会空转（实测：`MOVE2` 连续 10 次而指标逐位不变）。
        # 归因用 `NO_PROGRESS`，**不是** `BUDGET_EXHAUSTED`：后者会误导人
        # 去加 `max_iterations`，而加预算只会多空转几次。
        if action == ProofAction.LOWER_BOUND_VERIFY:
            if before.volume_ratio >= 1.0:
                action = ProofAction.TERMINAL_SUCCESS
            else:
                action = ProofAction.TERMINAL_FAILURE
                if self._last_failure_reason is None:
                    # `_select_move` 未归因（如递归路径未设）时，如实标 UNKNOWN
                    # ——不猜。猜成 BUDGET_EXHAUSTED 会误导处置方向。
                    self._last_failure_reason = FailureReason.UNKNOWN
            return state, action, notes

        # 兜底
        return state, action, notes

    def _seed_fields(self) -> dict:
        """构造要写进 ``ProofResult`` 的种子字段（三处返回点共用）。

        单处实现，避免三处各写一遍而在后续改动中走样。
        """
        return {"seed": self.seed, "seed_is_derived": False}

    def _induct_on_narrow_batches(self, state: ProofState) -> "_NarrowCoverage":
        """对被剥离的窄簇做**"小帽上的归纳"**，返回覆盖报告。

        **这一步是 L1 的另一半**（2026-09-30）。此前 ``BROAD_ONLY`` 是**终态**
        ——窄簇被剥掉后无人承接，于是对原输入什么都没证明，只能如实报"不成功"。
        论文的做法是：既然这批管的方向挤在一个半径 ``cap_radius`` 的小帽里，
        就**在这个帽上重走一遍同一条论证**（induction on a smaller cap）。

        本实现的做法（**如实说明是代理，不是论文的完整构造**）：
        对每批窄簇，用**同一套配置**递归跑一次 ``run()``。递归的自然终止
        条件是管的**尺度**——传入的 δ 取该批自身的特征尺度（``cap_radius``
        作为方向张角的代理），使每层递归处理的是"更小帽上的同一问题"。

        **为什么用"递归跑一遍"而不是"直接算体积下界"**：被剥离的窄簇是
        一个**完整的子问题**（它自己也可能需要 grains/Move/sticky 检查），
        只算它的并集体积会漏掉"这批管本身是否满足论证前提"。递归保证了
        与主流程**同一条判据**（判据只有一处实现，见 §7.0.2 的"判据必须
        与推理同源"）。

        Returns
        -------
        _NarrowCoverage
            每批的承接结果 + 汇总。``all_covered`` 为 True 时才允许把 scope
            提升为 ``FULL``。
        """
        batches = list(state.narrow_batches)
        if not batches:
            return _NarrowCoverage(covered=0, total=0, per_batch=[], notes=[])

        # **深度上限**：到了就如实报未承接，不再下探。
        # 不静默放宽——"没承接"与"承接到底仍失败"是两件事，必须可区分。
        if self._induction_depth >= MAX_INDUCTION_DEPTH:
            return _NarrowCoverage(
                covered=0,
                total=len(batches),
                per_batch=[
                    _NarrowBatchOutcome(
                        index=i, n_tubes=len(b.indices), covered=False,
                        reason=f"已达最大归纳深度 {MAX_INDUCTION_DEPTH}",
                    )
                    for i, b in enumerate(batches)
                ],
                notes=[
                    f"达到 MAX_INDUCTION_DEPTH={MAX_INDUCTION_DEPTH}，"
                    "不再下探（避免无界递归）"
                ],
            )

        covered = 0
        per_batch: List["_NarrowBatchOutcome"] = []
        notes: List[str] = []

        for i, batch in enumerate(batches):
            tubes = [state.tubes[j] for j in batch.indices if 0 <= j < len(state.tubes)]
            if len(tubes) < 2:
                # 单根管谈不上并集体积（退化），如实记为未覆盖——**不跳过**：
                # 跳过会让"未覆盖"凭空消失，正是此前那个缺口的形态。
                per_batch.append(
                    _NarrowBatchOutcome(
                        index=i, n_tubes=len(tubes), covered=False,
                        reason="退化批次（少于 2 根管，无并集体积可言）",
                    )
                )
                continue

            # **递归的尺度：保持与原 δ 相同（不缩放）**。
            #
            # 试过一版 ``δ' = δ · cap_radius``，**放弃了**——原因是实测把
            # δ 从 0.1 压到 0.005 后，``grid_n=8`` 的体素网格完全分辨不出
            # 这么细的管，运行**卡死**（δ=0.1 时 0.02 s 完成，δ=0.005 时
            # 数分钟不返回）。
            #
            # 更根本的问题是**量纲**：``cap_radius`` 是**方向球面上的弧度**
            # （方向聚拢程度），``δ`` 是**空间粗细**，两者不是同一个量，
            # 直接把弧度乘到 δ 上没有依据。论文里"更小帽 ⇒ 更细尺度"的
            # 定量关系需要显式推导（形如 ``δ' ~ δ·r^β``），**本实现未做该
            # 推导**，故不臆造一个乘子。
            #
            # 保持原 δ 在数学上对应"**在同一尺度上、对方向子集重走论证**"
            # ——这是"小帽上的归纳"的**保守特例**（不比论文强，但不引入
            # 未经验证的尺度变换）。如实标注在 §7.0.2。

            # **`seed` 处置放在 `try` 之外**（2026-09-30 补）。
            #
            # 原因：下面的 `except` 是"**承接失败不该让主运行崩掉**"的兜底，
            # 它兜的是**数学上的失败**（递归跑完但没承接住）。而 `sub_seed`
            # 是**纯算术**——它若出错（如对 `None` 做加法），那是**编程错误**，
            # 不是"这批没承接住"。混在同一个 `except` 里会让编程错误被静默
            # 记成"承接失败"，**故障看起来像结论**。
            #
            # 这个坑实测踩过：`seed=None` 时 `self.seed + 1000 + i` 抛
            # `TypeError`，被 `except` 吞掉后只写进 `_diagnostics`，
            # 报告成"第 i 批承接时异常：TypeError"——**测试全绿**，
            # 而真正的 bug（`None` 不能做加法）从未暴露。
            #
            # 语义：`self.seed is None`（非确定探索模式）时传 `None`，
            # 让递归层继承同一语义（不是"退回某个默认种子"）。否则错开
            # 各批次的种子，避免不同批在相同管族上取到同一组随机变换。
            sub_seed = None if self.seed is None else self.seed + 1000 + i

            try:
                sub = ProofOrchestrator(
                    tubes,
                    max_iterations=self.max_iterations,
                    grid_n=self.grid_n,
                    epsilon=self.epsilon,
                    seed=sub_seed,
                    # **不传 surrogate**：递归里再走一遍 AI 打分会让成本爆炸，
                    # 且这一层的目的是"承接证明"而非"练打分器"。
                    surrogate_weights_path="/nonexistent/weights.json",
                    # 承接层**不再做 broadness 剥离**：否则会无限递归
                    # （每层都剥出新的窄簇）。论文里递归靠"帽更小"终止，
                    # 这里靠"不再剥离 + 深度上限"终止。
                    broadness_filter=False,
                    # 深度 +1：承接层自己再遇到剥离时，由上限拦下。
                    _induction_depth=self._induction_depth + 1,
                ).run()
            except (
                # **编程错误必须直接冒泡，不能被当成"这批没承接住"**
                # （2026-09-30 补）。这三类在承接流程里只会来自代码缺陷：
                #   * TypeError      —— 类型用错（如对 None 做算术）
                #   * AttributeError —— 属性/方法不存在（改名后漏改调用点）
                #   * NameError      —— 名字不存在（重构残留）
                # 把它们吞掉会让**故障看起来像结论**：报告说"承接未成功"，
                # 实际是代码坏了。实测 `seed=None` 的 TypeError 就这样潜伏过。
                TypeError,
                AttributeError,
                NameError,
            ):
                raise
            except Exception as exc:  # 承接失败不该让主运行崩掉
                # 走到这里的才是**真正的数学/数值失败**（如奇异矩阵、
                # 不收敛），如实记为"这批未承接"，不放大也不掩盖。
                self._diagnostics.append(
                    f"窄簇承接第 {i} 批失败（{type(exc).__name__}: {exc}）"
                )
                per_batch.append(
                    _NarrowBatchOutcome(
                        index=i, n_tubes=len(tubes), covered=False,
                        reason=f"承接时异常：{type(exc).__name__}",
                    )
                )
                continue

            ok = bool(sub.success) and sub.scope is ProofScope.FULL
            if ok:
                covered += 1
            else:
                notes.append(
                    f"第 {i} 批（{len(tubes)} 根）承接未成功："
                    f"scope={sub.scope.name}"
                )
            per_batch.append(
                _NarrowBatchOutcome(
                    index=i, n_tubes=len(tubes), covered=ok,
                    reason="" if ok else FAILURE_REASON_TEXT.get(
                        sub.failure_reason, sub.message
                    ),
                )
            )

        return _NarrowCoverage(
            covered=covered,
            total=len(batches),
            per_batch=per_batch,
            notes=notes,
        )

    def _scope_fields(
        self, state: ProofState, success: bool, coverage: "_NarrowCoverage",
    ) -> dict:
        """构造"``success`` 作用域"字段（各处返回点共用）。

        **2026-09-30 起收 ``coverage``**（L1 的另一半）：若有被剥离的窄簇，
        但**它们全部承接成功**，则下界确实覆盖了原输入 ⇒ scope 提升为
        ``FULL``。未全部承接才落到 ``BROAD_ONLY``。

        这是本函数与早先版本的**唯一行为差别**：早先只要 ``uncovered > 0``
        就一律 ``BROAD_ONLY``（因为那时根本没有承接机制，剥了就是没了）。

        - 无窄簇 ⇒ ``FULL``（下界确实覆盖全部输入）；
        - 有窄簇且**全部承接** ⇒ ``FULL``（论文的"小帽上的归纳"已走通）；
        - 有窄簇且**部分未承接** ⇒ ``BROAD_ONLY``（强制 ``success=False``）。

        注意：``BROAD_ONLY`` 时**不是**"什么都没做"——broad 子集上的下界
        依然成立，只是作用域比读者以为的窄。
        """
        if coverage.total == 0:
            # 无剥离 ⇒ 下界的**作用域**就是全部输入。
            #
            # **注意这里不再看 `success`**（2026-09-30 修正）：scope 描述的是
            # "结论**管到哪**"，success 描述"结论**成不成立**"——两者独立。
            # 此前把它们绑在一起（`FULL if success else NOT_APPLICABLE`），
            # 于是在"承接成功但 broad 子集未达标"时给出 `NOT_APPLICABLE`
            # ——那读起来像"没有下界"，而实际上是"**有**下界，只是没到线"。
            return {"uncovered_narrow_tubes": 0, "scope": ProofScope.FULL}
        if coverage.all_covered:
            # 窄簇全部承接 ⇒ 作用域回到**全部输入**（论文的"小帽上的归纳"
            # 已走通）。同样与 success 无关。
            return {"uncovered_narrow_tubes": 0, "scope": ProofScope.FULL}
        # **未全部承接**：仍然只对 broad 子集成立。
        # ``uncovered`` 只数**未承接成功的**那些批次——承接成功的不再是"未覆盖"。
        return {
            "uncovered_narrow_tubes": sum(
                b.n_tubes for b in coverage.per_batch if not b.covered
            ),
            "scope": ProofScope.BROAD_ONLY,
        }

    @staticmethod
    def _effective_success(scope: "ProofScope", wanted: bool) -> bool:
        """按 scope **裁决** ``success``（单一入口，避免各处各判）。

        规则只有一条：**只有 ``FULL`` 才允许 ``True``**。
        ``BROAD_ONLY``（下界不管被剥离部分）与 ``NOT_APPLICABLE``（空输入）
        一律 ``False``——对被剥离部分什么都没证明。

        **为什么不把 ``success`` 塞进 ``_scope_fields`` 的返回值**：那样会与
        调用点显式传的 ``success=`` 撞键（实测踩到 ``got multiple values for
        keyword argument 'success'``）。拆成两个纯函数后，职责清楚：
        ``_scope_fields`` 回答"作用域是什么"，本函数回答"因此 success 是什么"。
        """
        return bool(wanted) and scope is ProofScope.FULL

    def _emit_step(self, snapshot: ProgressSnapshot) -> None:
        """把快照交给回调，并把异常降级为诊断（**不中断搜索**）。

        为什么吞异常：观测设施（日志、UI 上报、监控打点）不该让一次跑了
        一分多钟的证明搜索失败。但也不能静默——异常内容写进
        :attr:`diagnostics`，事后能查到"回调坏了"。
        """
        self._snapshots.append(snapshot)
        try:
            self._monitor.on_step(snapshot)  # type: ignore[attr-defined]
        except Exception as exc:  # 刻意吞掉任何回调异常，不让观测代码拖垮搜索
            self._diagnostics.append(
                f"on_step 回调在第 {snapshot.iteration} 步抛出 "
                f"{type(exc).__name__}: {exc}（已忽略，搜索继续）"
            )

    @property
    def snapshots(self) -> List[ProgressSnapshot]:
        """``run()`` 期间收集到的单步快照（只读用途）。

        无论 ``on_step`` 传的是什么（甚至传 ``None``），这里都会累积——
        它记录的是**主循环真的发生了什么**，与回调是否配置无关。
        """
        return list(self._snapshots)

    @property
    def diagnostics(self) -> List[str]:
        """被降级处理的问题记录（只读用途）。

        含两类：① ``on_step`` 回调抛出的异常（已忽略、搜索继续）；
        ② 校验/重算失败但被吞掉的诊断。**"判定失败"与"判定为否"靠这里区分**。
        """
        return list(self._diagnostics)

    def monitor(self) -> object:
        """返回 ``on_step`` 背后的监视器对象（默认 :class:`NullMonitor`）。

        ``run()`` 结束后可用它取判定：若传入的是
        :class:`~kakeya_ai.proof_monitor.ProofMonitor`，可调
        ``orch.monitor().verdict()`` 拿三态健康结论。
        """
        return self._monitor

    def run(self) -> ProofResult:
        """运行主算法骨架直到终止或达到最大迭代次数。

        **幂等约定（2026-09-30 修）**：``run()`` 每次都是"新的一次运行"，
        与"新建一个同参数实例再跑"结果**必须相同**。因此开头会把所有
        "按次运行"的状态归零：

        - ``_scorer_gate_blocks/_accepts``：打分器门控统计（否则两次混在一起）；
        - ``_nsp_calls``：NSP 种子派生计数器（``seed + k``，k 是**本次运行内**
          的第 k 次调用）。此前不归零，于是同一个实例第二次 ``run()`` 的第一处
          NSP 用 ``seed + 1``——**与新建实例的结果不同**（实测 4 步 vs 4 步但
          管族不同，累积到第三次变成 2 步）。这会让"重复跑同一构型做消融"
          静默失真。
        - ``_move_attempts``：Move 失效记忆。同理——上一条证明它会让第二次
          运行"记得"上次的结论。**它是"本次运行内"的记忆，不是跨运行的缓存**。
        - ``_snapshots`` / ``_diagnostics``：观测记录，同理。

        保留跨运行的东西只有 ``_elapsed_total``（累计耗时，本身就是累计语义）
        与不可变的配置。
        """
        # —— 按次运行的状态归零（见 docstring 的幂等约定）——
        self._scorer_gate_blocks = {"LLM": 0, "SURROGATE": 0}
        self._scorer_gate_accepts = {"LLM": 0, "SURROGATE": 0}
        self._nsp_calls = 0
        self._move_attempts = {}
        self._snapshots = []
        self._diagnostics = []
        if not self.tubes:
            return ProofResult(
                success=False,
                steps=[],
                final_state=ProofState([], None, None, self.epsilon, self.epsilon),
                final_metrics=ProofMetrics(
                    delta=0.0, rho=0.0, a=0.0, b=0.0, c=0.0,
                    local_error=0.0, volume_ratio=0.0, union_volume=0.0,
                    lower_bound=0.0, n_tubes=0, n_grains=0,
                ),
                message="empty tube collection",
                initial_tubes=[],
                domain_radius=self.domain_radius,
                failure_reason=FailureReason.EMPTY_INPUT,
                **self._seed_fields(),
                # 空输入：无下界可言，故 NOT_APPLICABLE。
                uncovered_narrow_tubes=0,
                scope=ProofScope.NOT_APPLICABLE,
            )

        state = self.initialize()

        steps: List[ProofStep] = []
        for it in range(self.max_iterations):
            _t0 = time.perf_counter()
            before = self._compute_metrics(state)
            prev_state = state
            state, action, notes = self.step(state, before=before)
            after = self._compute_metrics(state)

            step = ProofStep(
                iteration=it,
                action=action,
                before=before,
                after=after,
                notes=notes,
                state_change=self._describe_state_change(prev_state, state),
                selection_source=self._last_selection_source,
                achieved_gain=self._last_move_achieved_gain,
                iters_used=self._last_move_iters_used,
            )
            steps.append(step)

            # 进度快照 + 回调（长跑可观测性）。**回调异常不中断搜索**——
            # 观测设施不该成为新的失败源；异常记进诊断，供事后排查。
            _elapsed = time.perf_counter() - _t0
            self._elapsed_total += _elapsed
            self._emit_step(
                ProgressSnapshot(
                    iteration=it,
                    action=action.name,
                    elapsed=_elapsed,
                    total_elapsed=self._elapsed_total,
                    volume_ratio=float(after.volume_ratio),
                    n_grains=int(after.n_grains),
                    n_tubes=int(after.n_tubes),
                    terminal=action in (
                        ProofAction.TERMINAL_SUCCESS,
                        ProofAction.TERMINAL_FAILURE,
                    ),
                    notes=tuple(notes),
                )
            )

            if action in (ProofAction.TERMINAL_SUCCESS, ProofAction.TERMINAL_FAILURE):
                success = action == ProofAction.TERMINAL_SUCCESS
                if success:
                    reason = FailureReason.NONE
                else:
                    # ``step()`` 在终止分支里写入归因。若没写（新增终止路径
                    # 忘了补归因），如实标 UNKNOWN —— 绝不能猜一个具体类别：
                    # 猜错会让调用方去加预算，而真正的问题可能是别处。
                    reason = (
                        self._last_failure_reason
                        if self._last_failure_reason is not None
                        else FailureReason.UNKNOWN
                    )
                # **先承接窄簇，再定 scope**（2026-09-30，L1 的另一半）。
                # 此前 ``BROAD_ONLY`` 是终态——剥了没人管。现在对每批窄簇
                # 在它自己的小帽上递归跑一次；**全部承接成功**时才能把
                # scope 提升为 FULL（即真正对原输入成立）。
                coverage = self._induct_on_narrow_batches(state)
                _scope = self._scope_fields(state, success, coverage)
                return ProofResult(
                    steps=steps,
                    final_state=state,
                    final_metrics=after,
                    message=FAILURE_REASON_TEXT[reason],
                    initial_tubes=list(self.tubes),
                    domain_radius=self.domain_radius,
                    failure_reason=reason,
                    **self._seed_fields(),
                    success=self._effective_success(_scope["scope"], success),
                    narrow_coverage=coverage.summary(),
                    **_scope,
                )

        # 达到最大迭代次数：搜索未跑完（exhausted）。
        # ``success`` 仍然只由下界计算决定（volume_ratio 口径不变），但
        # ``exhausted=True`` 明确告知调用方这是"预算用尽"而非"判定终止"：
        # 即使本次下界成立，也不等于找到了终止路径。
        #
        # ``steps[-1].after`` 而不是循环外的 ``after``：后者在
        # ``max_iterations == 0``（循环零次）时**未绑定**——虽然下面的条件
        # 表达式会因 ``steps`` 为空而短路，**运行时安全**，但静态分析无法
        # 证明这一点（会报 possibly-unbound）。改成从 ``steps`` 里取，
        # 类型与语义都直白：有步就用最后一步的结果，没有就现算一次。
        final_metrics = steps[-1].after if steps else self._compute_metrics(state)
        # 同上：先承接窄簇，再定 scope。
        want_success = final_metrics.volume_ratio >= 1.0
        coverage = self._induct_on_narrow_batches(state)
        _scope = self._scope_fields(state, want_success, coverage)
        return ProofResult(
            steps=steps,
            final_state=state,
            final_metrics=final_metrics,
            message=f"reached max_iterations={self.max_iterations}",
            exhausted=True,
            initial_tubes=list(self.tubes),
            domain_radius=self.domain_radius,
            failure_reason=FailureReason.BUDGET_EXHAUSTED,
            **self._seed_fields(),
            success=self._effective_success(_scope["scope"], want_success),
            narrow_coverage=coverage.summary(),
            **_scope,
        )


def run_proof_skeleton(
    tubes: Sequence[Tube],
    epsilon: float = 0.05,
    K: float = 2.0,
    lambda_min: float = 0.85,
    max_iterations: int = 10,
    grid_n: int = 64,
    use_nano: bool = False,
    move_params: Optional[dict] = None,
    surrogate: Optional["SurrogateScorer"] = None,
    surrogate_threshold: float = 2.0,
    surrogate_weights_path: Optional[str] = None,
    llm_scorer: Optional["LLMScorerLike"] = None,
    llm_strategy: str = "beam",
    llm_threshold: float = 2.0,
    broadness_filter: bool = True,
    broadness_K: float = 10.0,
    broadness_beta: float = 1.0,
    broadness_points: int = 32,
    seed: Optional[int] = 0,
) -> ProofResult:
    """便捷入口：运行主算法骨架。

    ``seed`` 说明见 ``ProofOrchestrator.__init__``：默认 ``0``（确定），
    传 ``None`` 显式选择非确定性探索模式。
    """
    orch = ProofOrchestrator(
        tubes=tubes,
        epsilon=epsilon,
        K=K,
        lambda_min=lambda_min,
        max_iterations=max_iterations,
        grid_n=grid_n,
        use_nano=use_nano,
        move_params=move_params,
        surrogate=surrogate,
        surrogate_threshold=surrogate_threshold,
        surrogate_weights_path=surrogate_weights_path,
        llm_scorer=llm_scorer,
        llm_strategy=llm_strategy,
        llm_threshold=llm_threshold,
        broadness_filter=broadness_filter,
        broadness_K=broadness_K,
        broadness_beta=broadness_beta,
        broadness_points=broadness_points,
        seed=seed,
    )
    return orch.run()
