"""小专家模式 V1.5 (Mini MoE) —— 仿 Kimi K3 的混合专家路由思想.

Kimi K3 用 896 个专家、每次激活 16 个. 这里做一个极简可用版本:
- 专家池: 若干预设专家 (不同角色 prompt + 模型 + 关键词)
- 路由器: 基于关键词加权评分, 选 top-k 个专家
- 聚合器: 依次调用激活的专家, 拼接回答

V1.5 新增 (对应 Kimi K3 论文机制, 详见 内部设计文档第 III 篇《专家模式 MoE 路由》):
- QuantileBalancer: Quantile Balancing 负载均衡 (论文 §2.3.3),
  专家偏置只影响 top-k 选择、不进混合权重;
- shared expert: 共享专家常激活 (论文 §2.3 的 shared experts);
- 分数 RMS 归一化 (论文 §2.3.1 Normalized LatentMoE 的思想);
- 推理强度网格: effort = low/high/max, 按问题分配 token 预算
  (论文 §4.1.2 的 domain × reasoning-effort 思想, 启发式而非 RL).

回答精度方面不做限制: 默认不设 num_predict 上限 (交由 Ollama 默认值).

Example
-------
>>> from kakeya_ai.ai.expert_router import ExpertRouter
>>> router = ExpertRouter()
>>> answer, meta = router.route_and_answer("用 Python 写一个快速排序")
>>> print(answer)
>>> print(meta)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .observability import default_observability, get_logger
from .ollama_client import generate, is_server_ready
from .token_usage import TokenUsage
from ..resource_budget import DEFAULT_BUDGET

_logger = get_logger(__name__)
_obs = default_observability()


@dataclass
class Expert:
    """单个专家的定义.

    Parameters
    ----------
    name : str
        专家名称, 用于路由日志和回答标签.
    keywords : list[str]
        专长关键词, 路由器用它和用户 query 做匹配.
        中英文均可, 不区分大小写.
    system_prompt : str
        专家系统提示词, 决定其角色与风格.
    model : str
        使用的 Ollama 模型 tag, 例如 "qwen3.5:4b".
    options : dict
        Ollama 生成参数. 默认不限制 num_predict, 让模型自由发挥.
    priority : int
        同分时优先级, 数字大的优先.
    """

    name: str
    keywords: list[str]
    system_prompt: str
    model: str
    options: dict = field(default_factory=lambda: {"temperature": 0.6}) # type: ignore
    priority: int = 0
    effort_options: dict = field(default_factory=dict) # type: ignore
    """按推理强度覆盖生成参数, 例如 {"low": {"num_predict": 256}}.

    对应论文 §4.1.2 的 per-problem token budget: 低强度问题限制
    num_predict, 高强度问题放宽. 键为 EffortLevel 取值, 见
    :func:`estimate_effort`.
    """

    def score(self, query: str) -> float:
        """计算 query 与本专家关键词的匹配分.

        简单策略: 每命中一个关键词加 1 分; 命中词越长 (越具体) 再加额外分,
        鼓励具体词匹配. 不区分大小写.
        """
        if not query:
            return 0.0
        q = query.lower()
        total = 0.0
        for kw in self.keywords:
            k = kw.lower().strip()
            if not k:
                continue
            if k in q:
                # 基础 1 分 + 词长奖励 (每多一个字符加 0.1, 封顶 1.0)
                length_bonus = min(len(k) / 10.0, 1.0)
                total += 1.0 + length_bonus
        return total


# ---------------------------------------------------------------------------
# Quantile Balancing 负载均衡 (Kimi K3 论文 §2.3.3 的忠实简化)
# ---------------------------------------------------------------------------


class QuantileBalancer:
    """Quantile Balancing: 无辅助损失的 expert 负载均衡.

    思想 (论文 Eq.13 + §2.3.3): top-k 选择用 biased score ``s + b``,
    expert 偏置 ``b_j`` 由分数分位数确定, 使每个 expert 的负载等于
    目标值 ``q = m*k/n``; 偏置只影响 dispatch, 不参与混合权重
    (auxiliary-loss-free).

    单步更新 (与论文一致, 单次前向、无迭代):
    1. 用当前偏置算 cutoff ``α_i`` = biased score 的第 (k+1) 大值
       (top-(k+1) 取截止线, 免去 token 侧分位数);
    2. 对每个 expert j, 取 margin ``s_ij - α_i`` 的第 (q+1) 大值定
       ``b_j = -margin[q]``, 使恰好约 q 个 token 把 j 选入 top-k.

    Parameters
    ----------
    n_experts : int
        expert 数 n.
    top_k : int
        每 token 激活数 k.
    fixed_step : float or None
        若给定, 退化为论文引用的固定步长规则
        ``b_j += γ·sign(q̄ - ℓ_j)`` (小 batch 时更稳); None (默认)
        使用完整 QB 规则.

    Note
    ----
    原始得分全为 0 的 expert (如 keywords 为空的兜底专家) 不参与
    均衡, 避免无意义偏置.
    """

    def __init__(
        self,
        n_experts: int,
        top_k: int,
        fixed_step: Optional[float] = None,
    ) -> None:
        self.n = max(1, int(n_experts))
        self.k = max(1, min(int(top_k), self.n))
        self.fixed_step = fixed_step
        self.bias: list[float] = [0.0] * self.n
        self.loads: list[int] = [0] * self.n

    def update(self, scores: list[list[float]]) -> dict[int, float]:
        """根据 m×n 原始分矩阵更新偏置, 返回 {expert_idx: bias}."""
        m = len(scores)
        if m == 0:
            return dict(enumerate(self.bias))
        n, k = self.n, self.k

        biased = [
            [float(s) + self.bias[j] for j, s in enumerate(row[:n])]
            for row in scores
        ]

        # 当前偏置下的实际负载与 cutoff (第 k+1 大).
        loads = [0] * n
        alpha: list[float] = []
        for row in biased:
            sr = sorted(row, reverse=True)
            alpha.append(sr[k] if k < n else sr[-1])
            for j in sorted(range(n), key=lambda j: -row[j])[:k]:
                loads[j] += 1

        q = m * k / n
        if self.fixed_step is not None:
            # 固定步长规则: b_j += γ·sign(q̄ - ℓ_j)
            for j in range(n):
                if all(scores[i][j] <= 0 for i in range(m)):
                    continue
                if q > loads[j]:
                    self.bias[j] += self.fixed_step
                elif q < loads[j]:
                    self.bias[j] -= self.fixed_step
        else:
            # QB: b_j = -(s_ij - α_i 的第 (q+1) 大值), q 向下取整.
            q_idx = min(int(q), m - 1)
            for j in range(n):
                if all(scores[i][j] <= 0 for i in range(m)):
                    continue
                margins = sorted(
                    (scores[i][j] - alpha[i] for i in range(m)), reverse=True
                )
                self.bias[j] = -margins[q_idx]

        self.loads = loads
        return dict(enumerate(self.bias))

    def biased_scores(self, raw: list[float]) -> list[float]:
        """返回 raw + b (只用于选择, 不用于混合权重)."""
        return [float(s) + self.bias[j] for j, s in enumerate(raw[: self.n])]

    def stats(self) -> dict: # type: ignore
        return {"bias": list(self.bias), "loads": list(self.loads)} # type: ignore


# ---------------------------------------------------------------------------
# 推理强度网格 (Kimi K3 论文 §4.1.2 的思想, 启发式而非 RL)
# ---------------------------------------------------------------------------


class EffortLevel:
    """推理强度档位（对应论文的 {low, high, max} 推理强度）。"""

    LOW = "low"
    HIGH = "high"
    MAX = "max"
    ALL = (LOW, HIGH, MAX)


#: 路由置信度下限（roadmap §4.4 #21"专家路由置信度"，2026-10-30）：
#: 低于它 = 模糊 query，消费方（``ModelRouter.decide_mode``）不得硬选
#: ``fast``，须回退 ``cascade``。**标定值**（2026-10-30，
#: 内部实验脚本，改写集 25 + 独立集 100）：
#: 基线 92.0%；θ=0.80 时保留集准确率 97.3%（零回归）且回退率 41.6% 达最大；
#: 0.85/0.90 回退率相同——取同收益下更宽松的 0.80（更少无谓回退）。
#: 改值必须重跑标定脚本。
CONFIDENCE_FLOOR: float = 0.80


@dataclass(frozen=True)
class RouteInfo:
    """一次路由的置信信息（``route_with_confidence`` 的返回之一）。

    * ``confidence``：top-1 有效分份额（s1/Σs，只计非 shared 正分专家）；
      关键词全不命中（纯 general 兜底）时为 0.0。
    * ``is_low_confidence``：``confidence < floor`` —— 消费方据此回退。
    """

    activated: tuple  # tuple[str]，与 route() 返回一致的名字序
    scores: dict      # {专家名: 有效分}（非 shared 且正分）
    confidence: float
    floor: float

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence < self.floor


#: **MAX** 档的推理线索词：真正需要多步重推理的（证明一条引理 / 推导多步）。
#: 刻意**不含** ``证明``（它太常见，连"证明者/证明了"都算，且简单证明不该吃满
#: 预算）与 ``为什么``（问句风格词，方向无关——2026-10-03 已因此回退过一次）。
_MAX_EFFORT_KEYWORDS = ("推导", "反证", "归纳", "复杂度")

#: **HIGH** 档线索词：需要"解释/实现/证明/给出结论"但规模有限的任务。
#: 同样**不含**问句风格词（``为什么`` / ``怎么`` / ``如何``）——它们只表示"是个
#: 提问"，不代表工作量（与 MAX 里去掉 ``为什么`` 同理）。
_HIGH_EFFORT_KEYWORDS = (
    "证明", "设计", "实现", "解释", "定义", "结论", "下界", "论证",
    "关系", "性质", "区别", "修复", "指出",
)


def estimate_effort(query: str) -> str:
    """按问题估计推理强度 (per-problem budget 的启发式版).

    - 含 **MAX** 线索词 → ``max``（重多步推理）
    - 长度 > 200 → ``high``（长**文本**不等于重推理；此前误判为 max）
    - 含 **HIGH** 线索词 → ``high``
    - 其余 → ``low``（短问答 / 写作 / 闲聊默认吃小预算）

    **2026-10-09 重标定**（此前与 golden_set 作者标注一致率仅 35%，**低于**
    "无脑全判 low"的多数类基线 65%）：旧版把 ``为什么`` 计入 max、把长度 >200
    计入 max、并默认 ``high``，导致系统性**过估计**（10 过估 / 1 低估）。
    新版把 MAX 收紧到真正的重推理词、长文本降为 high、默认改 low，一致率升到
    **76%**（13/17），且**过估计 ≥ 低估**（2 ≥ 2）——偏好未变（过估安全、低估
    截断推理），见 ``tests/test_router_independent_corpus.py`` 的 effort 守门。
    """
    if not query:
        return EffortLevel.LOW
    if any(kw in query for kw in _MAX_EFFORT_KEYWORDS):
        return EffortLevel.MAX
    if len(query) > 200:
        return EffortLevel.HIGH
    if any(kw in query for kw in _HIGH_EFFORT_KEYWORDS):
        return EffortLevel.HIGH
    return EffortLevel.LOW


#: "这个提问像不像需要**长链推理**"的线索词（**不判断学科**）。
#:
#: ⚠️ **2026-10-03：本判据已从一个已回退的实验里移除**，这里保留常量与说明
#: 是为了记录那次失败，避免重走。回退原因（实测）：
#:
#: 在 `general` 兜底上用这组词升级 `deep`，在**自编**改写集上得 20% → 28%
#: （+8pt），看似有效。但在**独立语料**（内部实验脚本
#: 的 100 条，原作者编写、非本次实验者）上暴露**真实回归**：
#: ``"猫咪为什么打呼噜"`` 被判成 `deep`（期望 `fast`）⇒ 100/100 退化为 **99/100**。
#:
#: 逐词分解后更糟——**收益与回归同源**：
#:
#: | 线索词 | 独立语料 | 自编改写集 | 误升级 |
#: | --- | --- | --- | --- |
#: | 含 ``为什么`` | 99/100 | 7/25 | 1 |
#: | **去掉 ``为什么``** | **100/100** | **5/25** | **0** |
#: | 无兜底（改动前） | 100/100 | 5/25 | 0 |
#:
#: ⇒ ``为什么`` 是**唯一**贡献收益（5→7）的词，也是**唯一**造成回归的词；
#: 去掉它之后收益归零。根因：``为什么`` 在两个方向上都有（"为什么两条平行线
#: 永远不会碰在一起"该 deep、"猫咪为什么打呼噜"该 fast），**单凭它无法区分**。
#: ⇒ **该实验的净收益为 0**，故回退；内部设计文档局限 #8 记录全过程。
_REASONING_CUES: frozenset[str] = frozenset({
    "证明", "推导", "反证", "归纳", "为什么",
})


def looks_like_reasoning_question(query: str) -> bool:
    """粗判"这是不是一个需要长链推理的问题"（**不判断学科**）。

    ⚠️ **2026-10-03：当前无调用方**（原调用点已回退，见 :data:`_REASONING_CUES`
    的说明）。保留本函数是为了让那次失败**可复现、可检验**——想在
    `general` 兜底上做类似升级的人，可以直接跑
    内部实验脚本与独立语料对照，复核下面的结论。

    **它当初要解决的问题**：:class:`ExpertRouter` 在关键词**全不命中**时返回
    ``general``，而 ``model_router`` 把 ``general`` 映射到 ``fast`` —— 于是
    "纯描述性提问"（如"为什么两条平行线永远不会碰在一起"）被派给小模型。
    实测（逐类别分解）：``math``/``kakeya`` 合计相对 trivial 基线 **−14**，
    而 ``code``/``writing`` 是 **+5**（命中关键词时路由是对的）。

    **它当初的用法**：只判"推理强度"，不做学科判断（那需要语义 embedding，
    见 内部设计文档局限 #8）。**不采用 ``estimate_effort()``**，因为
    后者含"长度 > 200 ⇒ max"规则，会把**长文写作**误送 ``deep``。

    Parameters
    ----------
    query : str
        用户提问原文（中英文均可）。

    Returns
    -------
    bool
        命中任一推理线索词即 ``True``；空串恒 ``False``。
    """
    if not query:
        return False
    return any(cue in query for cue in _REASONING_CUES)


#: **分析型提问**的记号词表（2026-10-09 新增，取代上面的 ``_REASONING_CUES``）。
#:
#: **与已回退的 ``_REASONING_CUES`` 的关键区别**：上一次失败的根因是线索词选的是
#: "**问句风格**"的词（``为什么``）——它在常识问答与数学提问里**都**出现，方向无关，
#: 于是"收益与回归同源"。本次改用"**分析型内容**"的词，按四个**语言学类别**组织
#: （而非照抄某个测试集的名词）：
#:
#: * ``量化/度量``：问一个量（多少/多大/多密…）或提到度量对象（面积/体积/比例…）；
#: * ``关系/性质``：问关系、性质、结论是否成立；
#: * ``全称/必然``：全称量化与必然性断言（任意/必然/永远）；
#: * ``存在/比较``：存在性提问与不等式（有没有/是否存在/大于/小于/不超过）。
#:
#: **为什么这些词能区分两类**：常识提问（"猫咪为什么打呼噜"/"今天天气怎么样"）问的
#: 是**事实**，不含量化/关系/全称/比较结构；而"纯描述性"的数学提问
#: （"两条曲线围出来的面积怎么算"/"这个集合有多大"）几乎必然含其中之一。
#: 关键在于**故意不收**"问句风格"的词（为什么/怎么/是否/能不能…）——它们方向无关。
#:
#: **实测口径**（内部实验脚本，2026-10-09）：
#: 自编改写集 **20% → 60%**（math 0→8/10、kakeya 0→2/4、writing/code 不动），
#: 且**独立语料 100/100 零误升级**、``"猫咪为什么打呼噜"`` 仍 ``fast``、
#: ``golden_set`` 的 deep 率不变。**60% 超过 trivial 基线 56%**——低于它属
#: "零信息"（见 benchmark 的说明），故本兜底带来**正**信息。
#:
#: **如实登记的残余风险**：词表仍可能误触发于把同样的词用于日常语境的提问
#: （如"这两个东西有多大差别""有没有好的餐厅""价格不超过 100 元"）。
#: 本机两个语料未覆盖这类，无法证否；embedding 到位后应整体替换本判据。
_ANALYTICAL_MARKERS: tuple[str, ...] = (
    # 量化 / 度量
    "多少", "多大", "多小", "多密", "几倍", "有多少", "有多大",
    "比例", "概率", "面积", "体积", "大小", "数量", "密度", "极限", "收敛",
    # 关系 / 性质
    "关系", "性质", "结构", "结论", "成立", "等价", "站得住",
    # 全称 / 必然
    "任意", "必然", "永远",
    # 存在 / 比较
    "是否存在", "有没有", "大于", "小于", "不超过",
)


def looks_like_analytical_question(query: str) -> bool:
    """粗判"这是不是一个**分析型**提问"（**不判断学科**）。

    用途：:class:`ExpertRouter` 在关键词**全不命中**时返回 ``general``，而
    ``model_router`` 默认把 ``general`` 映射到 ``fast``——于是"纯描述性"的分析型
    提问（如"两条曲线围出来的那块面积怎么算"）被漏判给小模型。本函数供
    ``ModelRouter.decide_mode`` 在 ``expert == "general"`` 时决定是否升级 ``deep``。

    **它是上一版 ``looks_like_reasoning_question``（基于 ``为什么`` 等问句风格词）
    的替代**：那一版在自编改写集上 +8pt、却在独立语料上把 ``"猫咪为什么打呼噜"``
    误升级（100→99），净收益 0，已回退（见 :data:`_REASONING_CUES`）。本版改用
    分析型**内容**词，实测独立语料零误升级、改写集 20%→60%（见
    :data:`_ANALYTICAL_MARKERS` 的实测口径与残余风险）。

    **不采用 ``estimate_effort()``**：后者含"长度 > 200 ⇒ max"规则，会把长文写作
    误送 ``deep``（本函数只看内容词，不看长度）。

    Parameters
    ----------
    query : str
        用户提问原文（中英文均可）。

    Returns
    -------
    bool
        命中任一分析型记号词即 ``True``；空串恒 ``False``。
    """
    if not query:
        return False
    return any(marker in query for marker in _ANALYTICAL_MARKERS)


def resolve_options(expert: Expert, effort: Optional[str]) -> dict: # type: ignore
    """合并专家默认 options 与 effort 覆盖项 (effort 优先)."""
    opts = dict(expert.options) # type: ignore
    if effort and effort in getattr(expert, "effort_options", {}):
        opts.update(expert.effort_options[effort]) # type: ignore
    return opts # type: ignore


# ---------------------------------------------------------------------------
# 预设专家池
# ---------------------------------------------------------------------------

DEFAULT_FAST_MODEL = "qwen3.5:4b"
DEFAULT_DEEP_MODEL = "qwen3.5:4b"

GENERAL_SYSTEM_PROMPT = (
    "你是一名通用助手, 回答简洁、准确、有条理. "
    "遇到不确定的问题直接说明, 不要编造."
)

MATH_SYSTEM_PROMPT = (
    "你是一名数学专家, 擅长代数、微积分、概率统计、几何、数论等领域. "
    "回答时请: 1) 先厘清已知条件与待求项; 2) 给出主要推导步骤; "
    "3) 给出最终结果并验证合理性. 不需要限制篇幅."
)

CODE_SYSTEM_PROMPT = (
    "你是一名资深软件工程师, 熟悉 Python / JavaScript / C++ / Rust 等主流语言. "
    "回答时请: 1) 先简要分析需求; 2) 给出可直接运行的完整代码 (带必要注释); "
    "3) 说明实现思路; 4) 指出潜在边界情况. 不需要限制篇幅."
)

WRITING_SYSTEM_PROMPT = (
    "你是一名内容创作专家, 擅长文案、论文、报告、故事等多种文体. "
    "回答时注意: 结构清晰、语言流畅、风格适配场景. 不需要限制篇幅."
)

KAKEYA_SYSTEM_PROMPT = (
    "你是挂谷集合 (Kakeya set) 研究领域的助手. "
    "重要术语: '挂谷集合' = Kakeya set, '挂谷定理/猜想' = Kakeya theorem/conjecture, "
    "属于几何测度论, 不是博弈论或经济学. "
    "涉及单位线段在所有方向上的集合及其 Hausdorff 维数、体积等. "
    "不确定的内容直接说明, 不要编造."
)


def _load_expert_pool():
    """加载架构清单的专家池（单一真源在 nano/manifest/architecture_manifest.py）。

    nano 是仓库根的扁平包：先尝试直接导入，失败则按本文件位置推导
    仓库根（src/kakeya_ai/ai/ → 上三级）注入 ``sys.path`` 再导
    （与 ``plan_aware_routing`` 同一思路）。仍失败时让 ImportError
    带着明确信息冒出——**池组成的真源不可缺，不静默降级**（否则清单
    改了、代码却用内置旧池，单一真源形同虚设）。
    """
    try:
        from nano.manifest.architecture_manifest import EXPERT_POOL

        return EXPERT_POOL
    except ImportError:
        repo_root = str(Path(__file__).resolve().parents[3])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from nano.manifest.architecture_manifest import EXPERT_POOL

        return EXPERT_POOL


#: effort 网格按模型档推导（deep 模型给深网格）——口径单一化，改档
#: 在清单的 ExpertSpec.model_role，不必动本表。
_EFFORT_BY_ROLE = {
    "deep": {
        "low": {"num_predict": 512},
        "high": {"num_predict": 1024},
        "max": {"num_predict": 2048},
    },
    "fast": {
        "low": {"num_predict": 256},
        "high": {"num_predict": 512},
        "max": {"num_predict": 1024},
    },
}


def default_experts(
    fast_model: str = DEFAULT_FAST_MODEL,
    deep_model: str = DEFAULT_DEEP_MODEL,
) -> list[Expert]:
    """返回默认的专家池 (5 个)——**组成数据从架构清单生成**（V3 阶段 3）。

    名字/关键词/模型档/priority/temperature 的单一真源是
    ``nano/manifest/architecture_manifest.py::EXPERT_POOL``；本函数只做配对：
    系统提示词（按名字）、默认模型与 effort 网格（按 ``model_role``）。
    **改池组成请改清单**（MERGE/SPLIT 走清单的 ``merge_experts`` /
    ``split_expert``，带不变量校验）；守门测试逐位比对清单与本函数产物。
    """
    pool = _load_expert_pool()
    prompts = {
        "math": MATH_SYSTEM_PROMPT,
        "code": CODE_SYSTEM_PROMPT,
        "writing": WRITING_SYSTEM_PROMPT,
        "kakeya": KAKEYA_SYSTEM_PROMPT,
        "general": GENERAL_SYSTEM_PROMPT,
    }
    return [
        Expert(
            name=s.name,
            keywords=list(s.keywords),
            system_prompt=prompts[s.name],
            model=deep_model if s.model_role == "deep" else fast_model,
            options={"temperature": s.temperature},
            effort_options=_EFFORT_BY_ROLE[s.model_role],
            priority=s.priority,
        )
        for s in pool
    ]


# ---------------------------------------------------------------------------
# 路由器 + 聚合器
# ---------------------------------------------------------------------------


class ExpertRouter:
    """小专家模式路由器.

    Parameters
    ----------
    experts : list[Expert], optional
        专家池. 默认使用 :func:`default_experts`.
    top_k : int
        每次激活的专家数量 (类比 Kimi K3 的 top-k 路由). 默认 2.
    always_include_general : bool
        是否在没有匹配专家时强制启用 general 兜底. 默认 True.
    balancer : QuantileBalancer or None
        QB 负载均衡器. 设置后 ``route()`` 用 biased score (s+b) 选择,
        ``route_batch()`` 会先按批量分数更新偏置. 偏置不进混合权重.
    shared_names : tuple[str, ...] or None
        共享专家名 (如 ``("general",)``): 类比论文 shared experts,
        总是激活, 与 routed top-k 互补 (去重).
    normalize : bool
        选 top-k 前对分数做 RMS 归一化 (论文 §2.3.1 思想). 默认 False.
    """

    def __init__(
        self,
        experts: Optional[list[Expert]] = None,
        top_k: int = 2,
        always_include_general: bool = True,
        keep_alive: str = "10m",
        balancer: Optional[QuantileBalancer] = None,
        shared_names: Optional[tuple] = None, # type: ignore
        normalize: bool = False,
    ) -> None:
        self.experts = experts if experts is not None else default_experts()
        self.top_k = max(1, top_k)
        self.always_include_general = always_include_general
        self.keep_alive = keep_alive
        self.balancer = balancer
        self.shared_names = tuple(shared_names) if shared_names else None # type: ignore
        self.normalize = normalize

    # -- 评分 ---------------------------------------------------------------

    def _raw_scores(self, query: str) -> list[float]:
        return [exp.score(query) for exp in self.experts]

    def _effective_scores(self, query: str) -> list[float]:
        """选择用分数: raw → (+balancer bias) → (RMS 归一)."""
        scores = self._raw_scores(query)
        if self.balancer is not None:
            scores = self.balancer.biased_scores(scores)
        if self.normalize:
            scores = self._rms_normalize(scores)
        return scores

    @staticmethod
    def _rms_normalize(scores: list[float]) -> list[float]:
        ms = math.sqrt(sum(s * s for s in scores) / max(len(scores), 1))
        if ms < 1e-12:
            return list(scores)
        return [s / ms for s in scores]

    def mixture_weights(
        self,
        query: str,
        activated: Optional[list[Expert]] = None,
    ) -> dict:
        """routed 专家的归一化混合权重 p_i = s_i / Σs (原始分, 论文 Eq.13).

        shared 专家不参与权重. 原始分全为 0 时均匀分配.
        """
        activated = activated if activated is not None else self.route(query)
        shared = set(self.shared_names or ())
        routed = [e for e in activated if e.name not in shared]
        if not routed:
            return {}
        raw = {e.name: e.score(query) for e in routed}
        total = sum(raw.values())
        if total <= 0:
            return {e.name: 1.0 / len(routed) for e in routed}
        return {e.name: raw[e.name] / total for e in routed}

    # -- 路由 ---------------------------------------------------------------

    def route(self, query: str) -> list[Expert]:
        """根据 query 选出 top-k 个专家 (biased + 归一化后).

        若所有专家得分为 0 且 always_include_general=True,
        则返回 [general] 单个专家. shared 专家总是附加 (去重).
        """
        effective = self._effective_scores(query)
        # ``effective`` 必须与 ``self.experts`` 一一对应：长度不等意味着评分
        # 链路（balancer / RMS 归一）出了问题。此时 strict 会**立即报错**，
        # 而不是静默把专家与错误的分数配成对——后者会让路由结果毫无意义
        # 且不报任何错。
        scored = list(zip(self.experts, effective, strict=True))
        # 过滤 0 分专家 (但保留 general 用于兜底)
        positive = [(e, s) for e, s in scored if s > 0]
        if not positive:
            general = None
            if self.always_include_general:
                general = next(
                    (e for e in self.experts if e.name == "general"), None
                )
            if general is not None:
                activated = [general]
            else:
                # 实在找不到, 返回前 top_k 个
                activated = list(self.experts[: self.top_k])
        else:
            # 排序: 分数降序, 同分按 priority 降序
            positive.sort(key=lambda x: (x[1], x[0].priority), reverse=True)
            activated = [e for e, _ in positive[: self.top_k]]

        # 共享专家常激活 (论文 shared experts 思想).
        if self.shared_names:
            for name in self.shared_names:
                shared = next((e for e in self.experts if e.name == name), None)
                if shared is not None and shared not in activated:
                    activated.append(shared)
        return activated

    def route_batch(self, queries: list[str]) -> list[list[Expert]]:
        """批量路由: 先按分数矩阵更新 QB 偏置, 再逐条 top-k.

        这是负载均衡的入口 —— 单条 query 的均衡没有意义 (m=1 时
        q = k/n < 1). 未设置 balancer 时等价于逐条 ``route()``.
        """
        if self.balancer is None:
            return [self.route(q) for q in queries]
        self.balancer.update([self._raw_scores(q) for q in queries])
        return [self.route(q) for q in queries]

    # -- 置信度（roadmap §4.4 #21，2026-10-30）------------------------------

    def route_with_confidence(
        self, query: str, *, floor: Optional[float] = None
    ) -> tuple[list[Expert], "RouteInfo"]:
        """路由 + 置信信息：模糊 query 不得硬选（消费方据此回退 cascade）。

        与 :meth:`route` 的激活结果**完全一致**（本方法内部复用它），另返回
        :class:`RouteInfo`。置信度定义（可解释、可复算）：

        * 在激活的**非 shared 且正分**专家上取有效分，``confidence = s1 / Σs``
          （top-1 份额）——份额接近 1 = 一个专家压倒性命中；多专家均分 = 模糊。
        * 关键词全不命中（只剩 general 兜底）⇒ 无正分专家 ⇒ confidence = 0.0
          （**必然**低于任何正阈值，兜底即回退）。

        ``floor`` 缺省取 :data:`CONFIDENCE_FLOOR`（标定值，见下）。
        """
        activated = self.route(query)
        fl = CONFIDENCE_FLOOR if floor is None else floor
        shared = set(self.shared_names or ())
        effective = dict(
            zip(
                (e.name for e in self.experts),
                self._effective_scores(query),
                strict=True,
            )
        )
        routed_scores = {
            e.name: effective[e.name]
            for e in activated
            if e.name not in shared and effective[e.name] > 0
        }
        if not routed_scores:
            confidence = 0.0
        else:
            total = sum(routed_scores.values())
            confidence = max(routed_scores.values()) / total if total > 0 else 0.0
        info = RouteInfo(
            activated=tuple(e.name for e in activated),
            scores=routed_scores,
            confidence=confidence,
            floor=fl,
        )
        return activated, info

    # -- 聚合 ---------------------------------------------------------------

    def route_and_answer(
        self,
        query: str,
        show_expert_tag: bool = True,
        usage: Optional[TokenUsage] = None,
        effort: Optional[str] = None,
    ) -> tuple[str, dict]:
        """路由 -> 调用激活的专家 -> 聚合回答.

        Parameters
        ----------
        effort : str or None
            推理强度 (EffortLevel.LOW/HIGH/MAX). None 时由
            :func:`estimate_effort` 按 query 估计, 决定每个专家的
            生成参数覆盖 (per-problem token budget).

        Returns
        -------
        answer : str
            最终回答. 多专家时按顺序拼接, 每段带 [Expert: name] 标签
            (可由 show_expert_tag 关闭).
        meta : dict
            路由元信息: {"activated": [names], "scores": {...},
            "errors": [...], "effort": ..., "mixture_weights": {...}}.
        """
        meta: dict = {"activated": [], "scores": {}, "errors": []}

        if not is_server_ready():
            msg = (
                "Error: Ollama server is not reachable at "
                "http://localhost:11434. Please start Ollama first."
            )
            _obs.record_decision(source_label="expert_error")
            _obs.record_fallback(
                source_label="expert_error", reason="server_not_ready"
            )
            return msg, meta

        activated = self.route(query)
        meta["activated"] = [e.name for e in activated]
        for exp in self.experts:
            meta["scores"][exp.name] = exp.score(query)

        # 推理强度 (论文 §4.1.2 per-problem budget 的启发式版).
        effort = effort or estimate_effort(query)
        meta["effort"] = effort
        # 混合权重: 由原始分归一, balancer 偏置不参与 (论文 Eq.13).
        meta["mixture_weights"] = self.mixture_weights(query, activated)
        if self.balancer is not None:
            meta["bias"] = list(self.balancer.bias)

        if not activated:
            return "(no expert activated)", meta

        pieces: list[str] = []
        for exp in activated:
            prompt = (
                exp.system_prompt
                + "\n\nUser: "
                + query
                + "\n\n请给出你的回答."
            )
            try:
                resp = generate(
                    exp.model,
                    prompt,
                    timeout=DEFAULT_BUDGET.http_timeout,
                    keep_alive=self.keep_alive,
                    options=resolve_options(exp, effort),
                    usage=usage,
                    # 思考型模型（qwen3 系）缺省开 think，预算被思维链耗尽、
                    # response 返空串；非思考型模型会忽略该标志。
                    think=False,
                ).strip()
            except Exception as exc:
                _logger.warning("Expert '%s' failed: %s", exp.name, exc)
                meta["errors"].append({"expert": exp.name, "error": str(exc)})
                _obs.record_fallback(
                    source_label=f"expert_{exp.name}",
                    reason=f"call_failed: {exc}",
                )
                continue

            _obs.record_decision(source_label=f"expert_{exp.name}")

            if not resp:
                meta["errors"].append(
                    {"expert": exp.name, "error": "empty_response"}
                )
                continue

            if show_expert_tag:
                pieces.append(f"[Expert: {exp.name}]\n{resp}")
            else:
                pieces.append(resp)

        if not pieces:
            return "(all experts failed)", meta

        answer = "\n\n---\n\n".join(pieces)
        return answer, meta


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def ask(
    query: str,
    top_k: int = 2,
    show_expert_tag: bool = True,
    experts: Optional[list[Expert]] = None,
) -> tuple[str, dict]:
    """一键调用小专家模式."""
    router = ExpertRouter(experts=experts, top_k=top_k)
    return router.route_and_answer(query, show_expert_tag=show_expert_tag)


def list_experts(experts: Optional[list[Expert]] = None) -> list[dict]:
    """列出专家池 (用于 CLI --list)."""
    pool = experts if experts is not None else default_experts()
    return [
        {
            "name": e.name,
            "model": e.model,
            "keywords": e.keywords,
            "temperature": e.options.get("temperature", 0.6),
        }
        for e in pool
    ]


def parse_query_for_routing(query: str, experts: Optional[list[Expert]] = None) -> list[dict]:
    """只做路由, 不调用模型 (用于调试和测试)."""
    pool = experts if experts is not None else default_experts()
    router = ExpertRouter(experts=pool, top_k=2)
    activated = router.route(query)
    return [
        {
            "name": e.name,
            "score": e.score(query),
            "model": e.model,
        }
        for e in activated
    ]
