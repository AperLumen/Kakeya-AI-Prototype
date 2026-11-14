"""架构清单（architecture manifest）——芯片/调度/内核/专家池的**单一真源**。

**用户意图**（2026-10-30 定调）：\"整个优化基于模型——模型的信息分发调度要
通过这个架构去修改；架构尽可能做成插件形式（如 skill），让模型实现
平替架构的更迭。\"

本清单就是那份**机器可读的架构**，三层消费：

1. **模型（agent）**：改 cpp/ / nano/ / 调度相关代码前必读本清单；
   改架构 = **先改本清单**（原则/常量/DAG/专家池），
   再按各条目的 ``enforcement`` 落实（改代码 / 重跑标定 / 重渲染），最后
   跑守门。专家池更迭走 :func:`merge_experts` / :func:`split_expert`
   （带不变量校验），不手改池数据。
2. **人**：同一份清单是审查模型改动对不对的判据。
3. **守门测试**（``tests/tooling/test_architecture_manifest.py``）：保证
   \"清单 ↔ C++ ↔ Python\" 三方不漂移——这正是本项目反复踩过的
   \"同一物理量两处各记一个数\"的系统性解法。

**未知问题（如实登记，不假装已知）**：清单/SKILL 注入后，模型的真实
决策是否**系统性偏向**架构主张、以及能否\"平替架构更迭\"——这是经验
问题，本文件不断言。测量实验已登记为可执行数据
（:data:`SKILL_INFLUENCE_EXPERIMENT` + :data:`EXPERIMENT_PROTOCOL`），
当前**无结论**；在那之前，本清单的价值 = 单一真源 + 防漂移，独立于
"模型会不会照它想"这个问题成立。

**渲染产物的真源**：``docs/diagrams/scheduler_flow.mmd`` 是**生成物**
（本文件的 ``SCHEDULER_FLOW_*`` 数据经 :func:`render_scheduler_flow_mmd`
渲染）。改图结构 = 改这里的数据 → ``.venv/bin/python -m nano.manifest.architecture_manifest``
→ 需要位图时再跑 ``bash scripts/docs_build/render_diagrams.sh scheduler_flow``
（PNG 是可选产物、不入库）。手改 mmd 会被守门判红。英文孪生 ``scheduler_flow_en.mmd`` 是人工维护的翻译，不在本清单
真源范围内（如实登记）。

**生成产物的真源（专家池）**：``src/kakeya_ai/ai/expert_router.py::
default_experts()`` 从本清单的 :data:`EXPERT_POOL` **生成**（系统提示词
留在代码，按名字配对）——改池组成 = 改这里 + 跑池守门。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

__all__ = [
    "Principle",
    "SharedQuantity",
    "DagNode",
    "DagEdge",
    "ExpertSpec",
    "SkillInfluenceTask",
    "PRINCIPLES",
    "SHARED_QUANTITIES",
    "SCHEDULER_FLOW_STATEMENTS",
    "SCHEDULER_FLOW_NODES",
    "SCHEDULER_FLOW_EDGES",
    "SCHEDULER_FLOW_CLASSDEFS",
    "EXPERT_POOL",
    "merge_experts",
    "split_expert",
    "SKILL_INFLUENCE_EXPERIMENT",
    "EXPERIMENT_PROTOCOL",
    "render_scheduler_flow_mmd",
    "main",
]

#: 仓库根（本文件在 nano/manifest/ 下，上溯两级）。
_REPO_ROOT = Path(__file__).resolve().parents[2]
#: 生成物路径（DAG 渲染落点）。
SCHEDULER_FLOW_MMD = _REPO_ROOT / "docs" / "diagrams" / "scheduler_flow.mmd"


# ---------------------------------------------------------------------------
# 架构原则（模型的"决策宪法"；每条带 enforcement = 落实与守门机制）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Principle:
    """一条架构原则：陈述 + 落实机制（改代码时对照执行）。"""

    id: str
    title: str
    statement: str
    #: 落实/守门机制：谁保证这条原则不被破坏。
    enforcement: str


PRINCIPLES: tuple[Principle, ...] = (
    Principle(
        "plan-only",
        "只产出计划、不执行计算",
        "调度器（nano/scheduling/scheduler.py）只产出 SchedulePlan——\"调哪些模块、用"
        "什么资源、按什么顺序\"；计算永远走 accel / 各 kernel 的既有路径。"
        "MODULE_REGISTRY 是静态名字映射、不 import 主包（nano 研发侧不反向"
        "依赖运行时）。",
        "tests/test_scheduler.py 全组；test_every_module_kind_has_a_target"
        " / test_registry_verifies_resolvable",
    ),
    Principle(
        "bit-identical",
        "跨后端 / 跨切法逐位一致",
        "整型计数归约按确定顺序求和，故并行（均分/加权/动态领活）、不同"
        "后端（nano/fast/numpy）对同一输入必须逐位相同；浮点累加不走并行"
        "归约。",
        "tests/test_kernel_parallel.py、tests/test_accel_dispatcher.py、"
        "cpp/abi/test_abi.c（纯 C 层）；A/B 模式每组都验 count_match",
    ),
    Principle(
        "dataflow-fusion",
        "数据流纪律：采样⊕判定融合",
        "一次性消费的采样点（MC）由片内 splitmix64 现算——点数组不进内存"
        "（旧路径每样本 ~72 B 往返、信息零复用）。体素内核同原则：坐标由"
        "索引现算。\"让进内存的信息是有用的，筛选留给片上。\"",
        "tests/test_nano_kernel.py::test_fused_deterministic_same_seed；"
        "能耗量化 nano/model/chip_model.py::estimate_energy_pj（搬运占比 66%→0）",
    ),
    Principle(
        "criterion-first",
        "先判据后动手",
        "任何调度/架构优化须先有可判定的收益判据再实现；负结果如实登记，"
        "不加复杂度。判据工具三个 opt-in：KAKEYA_BENCH_PARTITION（异构"
        "分派）、KAKEYA_PARALLEL_STATS（失衡比）、KAKEYA_BENCH_FUSED_MC"
        "（采样开销）。",
        "架构清单的已做/未做登记口径；bench_micro.cpp"
        " 各段注释含标定数据与测量学坑（提出循环假快 145×、背靠背计时"
        "频率倾斜 ±2 ns）",
    ),
    Principle(
        "single-source",
        "同一物理量唯一真源 + 同源守门",
        "跨层共享的物理量（FLOP/pair、P 核峰值、分派权重、采样开销…）"
        "登记在本清单的 SHARED_QUANTITIES；C++/Python 两侧只准引用、"
        "不准另写数值；守门测试解析两侧源码与本清单比对。",
        "tests/tooling/test_architecture_manifest.py（三方）；"
        "tests/test_microbench_integration.py（两处同源的既有守门）",
    ),
)


# ---------------------------------------------------------------------------
# 共享物理量（唯一真源登记；守门测试做 清单 ↔ C++ ↔ Python 三方比对）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SharedQuantity:
    """一个跨层共享的物理量：值 + 各层锚点 + 口径注释。"""

    name: str
    value: float
    unit: str
    #: Python 侧锚点（文件 :: 符号；无 Python 常量时写明原因）。
    layer_python: str
    #: C++ 侧锚点（文件 :: 常量/函数），可多处。
    layer_cpp: tuple[tuple[str, str], ...]
    note: str


SHARED_QUANTITIES: tuple[SharedQuantity, ...] = (
    SharedQuantity(
        "e_core_partition_weight",
        0.60,
        "相对 P 核 = 1.0",
        # ⚠️ 无 Python 常量：scheduler.DEFAULT_E_CORE_EFFICIENCY（0.44）是
        # **另一个量**（predict_schedule 速度模型参数，由均分曲线拟合）——
        # 两者口径不同、不相等是对的，守门测试专门守这个区别。
        "（无 Python 常量；与 DEFAULT_E_CORE_EFFICIENCY 是两个量）",
        (
            ("cpp/kernel/parallel.hpp", "e_core_partition_weight"),
            ("nano/isolated/bench_micro.cpp", "E_CORE_WEIGHT"),
        ),
        "A/B 实测标定（KAKEYA_BENCH_PARTITION=1，多轮一致 w=0.60 ⇒ "
        "+4.4~4.8%）；见 parallel.hpp 的标定实验注释",
    ),
    SharedQuantity(
        "fused_mc_sampling_overhead_ns",
        0.0,
        "ns/样本",
        "nano/model/chip_model.py :: FUSED_MC_SAMPLING_OVERHEAD_NS",
        (("nano/isolated/bench_micro.cpp", "FUSED_MC_SAMPLING_OVERHEAD_NS"),),
        "配对交替协议标定（KAKEYA_BENCH_FUSED_MC=1，三连跑 −0.9~−0.04 "
        "ns，±1 ns 噪声内）⇒ \"算力换带宽\"的算力侧代价 ≈ 0",
    ),
)


# ---------------------------------------------------------------------------
# 专家池（小专家 V3 阶段 3，2026-10-30）——池组成的**单一真源**
# ---------------------------------------------------------------------------
#
# `src/kakeya_ai/ai/expert_router.py::default_experts()` 从本池**生成**
# 专家对象（系统提示词留在代码，按名字配对；effort 网格按 model_role
# 推导：deep={512,1024,2048}、fast={256,512,1024}）。模型（agent）要改
# 池组成 = 改这里 + 重跑守门（tests/tooling/test_architecture_manifest.py 的
# 池一致性用例）——这是"模型平替架构更迭"的第一个真实落点。
#
# ⚠️ 依赖方向：本文件**不 import** 主包（nano 保持无 src 依赖）；
# 是 expert_router 惰性导入本清单（与 plan_aware_routing 同模式）。

@dataclass(frozen=True)
class ExpertSpec:
    """一个专家的**组成数据**（系统提示词等表现层留在代码）。"""

    name: str
    keywords: tuple[str, ...]
    #: "fast" | "deep"——决定默认模型与 effort 网格。
    model_role: str
    priority: int
    temperature: float
    note: str = ""


#: 默认专家池（5 个）。general 是兜底：无关键词、恒 0 分、必被激活，
#: **不可合并/拆分**（``merge_experts``/``split_expert`` 强制此不变量）。
EXPERT_POOL: tuple[ExpertSpec, ...] = (
    ExpertSpec(
        "math", (
            "math", "数学", "证明", "定理", "推导", "方程", "积分",
            "微分", "概率", "统计", "矩阵", "线性代数", "几何", "数论",
            "求导", "极限", "分布", "方差", "期望", "贝叶斯",
        ),
        "deep", 2, 0.3, "数学推理（深模型）",
    ),
    ExpertSpec(
        "code", (
            "code", "代码", "编程", "python", "javascript", "js", "ts",
            "java", "c++", "rust", "go", "函数", "bug", "调试", "算法",
            "排序", "实现", "脚本", "程序", "api", "sql", "html", "css",
            "react", "vue", "实现一个", "docker", "kubernetes", "前端",
            "git", "性能优化", "rate limiter", "设计一个", "正则",
            "二分查找",
        ),
        "fast", 2, 0.4, "代码与工程",
    ),
    ExpertSpec(
        "writing", (
            "写", "文案", "文章", "论文", "报告", "总结", "润色", "改写",
            "故事", "小说", "演讲", "邮件", "草稿", "翻译", "摘要", "大纲",
            "开头", "结尾", "视频脚本", "短视频",
        ),
        "fast", 1, 0.7, "写作与润色",
    ),
    ExpertSpec(
        "kakeya", (
            "kakeya", "挂谷", "挂谷集合", "挂谷定理", "挂谷猜想",
            "hausdorff", "维数", "测度", "besicovitch", "方向", "tube",
            "δ-tube", "几何测度",
        ),
        "deep", 3, 0.3, "挂谷猜想领域（研究词汇；计算词汇由计划先验补）",
    ),
    ExpertSpec(
        "general", (),
        "fast", 0, 0.6, "兜底：无关键词、恒 0 分、总会被激活",
    ),
)


# ---------------------------------------------------------------------------
# 池的架构操作（MERGE / SPLIT）——带不变量校验，模型更迭池时**走这里**
# ---------------------------------------------------------------------------

def _spec(pool: tuple[ExpertSpec, ...], name: str) -> ExpertSpec:
    for s in pool:
        if s.name == name:
            return s
    raise ValueError(f"池中不存在专家 {name!r}")


def merge_experts(
    pool: tuple[ExpertSpec, ...],
    a: str,
    b: str,
    into: str,
    *,
    note: str = "",
) -> tuple[ExpertSpec, ...]:
    """合并两个专家为一个新专家（架构更迭操作，返回**新**池）。

    不变量（违反即 ValueError，不静默产出坏池）：
    * ``a``/``b`` 都在池中、互不相同、且**都不是 general**（兜底不可合并）；
    * ``into`` 不在池中（新名字，避免"合并"掩盖成"改名"）；
    * 关键词取并集去重（排序保持稳定，便于守门逐位比对）；
    * ``model_role`` 取较强档（``deep`` 优先）——合并后的专家承接较强
      任务，避免把深模型任务静默降级给快模型。
    """
    sa, sb = _spec(pool, a), _spec(pool, b)
    if a == b:
        raise ValueError("不能把一个专家与它自己合并")
    if "general" in (a, b):
        raise ValueError("general 是兜底专家，不可合并")
    if any(s.name == into for s in pool):
        raise ValueError(f"{into!r} 已在池中——合并必须用新名字（改名请另做）")
    merged = ExpertSpec(
        name=into,
        keywords=tuple(sorted(set(sa.keywords) | set(sb.keywords))),
        model_role="deep" if "deep" in (sa.model_role, sb.model_role) else "fast",
        priority=max(sa.priority, sb.priority),
        temperature=sa.temperature,
        note=note or f"由 {a} + {b} 合并",
    )
    out = [s for s in pool if s.name not in (a, b)]
    out.append(merged)
    return tuple(out)


def split_expert(
    pool: tuple[ExpertSpec, ...],
    name: str,
    into: tuple[str, str],
    *,
    keywords: tuple[tuple[str, ...], tuple[str, ...]],
    notes: tuple[str, str] = ("", ""),
) -> tuple[ExpertSpec, ...]:
    """拆分一个专家为两个新专家（架构更迭操作，返回**新**池）。

    不变量：
    * ``name`` 在池中且**不是 general**（兜底不可拆分）；
    * 两个新名字都不在池中、互不相同；
    * ``keywords`` 必须是原关键词集的**划分**（并集 = 原集、不相交）——
      防"拆分"时悄悄丢词或造词；
    * 两子专家的 model_role 必须**不弱于**原专家（拆分不得降级）。
    """
    src = _spec(pool, name)
    if name == "general":
        raise ValueError("general 是兜底专家，不可拆分")
    n1, n2 = into
    if n1 == n2:
        raise ValueError("两个新名字必须不同")
    if any(s.name in into for s in pool):
        raise ValueError("新名字与池中既有专家冲突")
    k1, k2 = keywords
    if set(k1) | set(k2) != set(src.keywords) or set(k1) & set(k2):
        raise ValueError("keywords 必须是原关键词集的划分（并集=原集、不相交）")
    out = [s for s in pool if s.name != name]
    for n, kw, note in ((n1, k1, notes[0]), (n2, k2, notes[1])):
        out.append(ExpertSpec(
            name=n, keywords=tuple(kw), model_role=src.model_role,
            priority=src.priority, temperature=src.temperature,
            note=note or f"由 {name} 拆分",
        ))
    return tuple(out)


# ---------------------------------------------------------------------------
# 未知问题的测量实验（SKILL 注入是否改变模型决策）——登记为可执行数据
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SkillInfluenceTask:
    """一个跨层决策任务：SKILL 影响实验的待执行任务。"""

    id: str
    prompt: str
    layers: tuple[str, ...]
    rubric: tuple[str, ...]


#: 任务集（3 个跨层任务）。每个任务在**有无 SKILL.md 注入**下各执行一遍
#: （不同会话，同模型），按 rubric 人工或半自动评分。
SKILL_INFLUENCE_EXPERIMENT: tuple[SkillInfluenceTask, ...] = (
    SkillInfluenceTask(
        "add-backend",
        "给体积估计新增一个 'percentile 报告' 后端字段（从 ninja kernel "
        "到 accel 调度到 with_ci API 的纵切改动）。",
        ("cpp/", "src/kakeya_ai/", "nano/manifest/architecture_manifest.py", "tests/"),
        ("是否先改/查架构清单再动代码", "守门一次通过率",
         "改动落点引用清单条目的比例"),
    ),
    SkillInfluenceTask(
        "retune-partition-weight",
        "把 E 核分派权重从 0.60 调到 0.70（跨 parallel.hpp / bench_micro / "
        "architecture_manifest 的同源改动）。",
        ("cpp/", "nano/", "tests/"),
        ("是否重跑 KAKEYA_BENCH_PARTITION=1 标定", "三处同源是否同步改",
         "是否跑 tests/tooling/test_architecture_manifest.py"),
    ),
    SkillInfluenceTask(
        "merge-experts",
        "把 writing 与 code 合并为一个 'tech_write' 专家（池组成更迭）。",
        ("nano/manifest/architecture_manifest.py", "src/kakeya_ai/ai/", "tests/"),
        ("是否走 merge_experts 而非手改", "池守门（清单↔default_experts）"
         "是否仍绿", "是否更新了受影响的守门测试"),
    ),
)

#: 执行口径与**第一轮结果**（2026-10-31 执行，如实登记：n 小、存在污染，
#: 结论为弱证据）。
EXPERIMENT_PROTOCOL: str = (
    "同一模型、同一仓库状态；任务集各任务执行两遍——A 组会话注入 SKILL.md"
    "（消费协议生效）、B 组不注入；收集两次的 diff 与守门输出，按各任务 "
    "rubric 评分。指标汇总：清单引用率、守门一次通过率、人工审查发现的"
    "架构违背数。"
    "\n\n"
    "**第一轮结果**（3 任务 × 2 臂，5 个有效会话）："
    "① add-backend——A 臂守门一次通过、B 臂计数守门迭代一轮：唯一显示"
    "两臂差异的任务（SKILL 对'先查守门涉及面'的过程纪律有弱正效应）；"
    "② retune-partition-weight——两臂完全同构，B 臂自发读仓库根 SKILL.md "
    "并遵循其流程：**自选择污染**，注入与不注入在该任务上不可分；"
    "③ merge-experts——两臂同构，且都做成'声明产物 + 手写等价物'而非"
    "调用 merge_experts API：SKILL 协议亦未促成 API 调用。"
    "**效度威胁**：共享工作树 = SKILL.md 与清单对两臂都可见，B 臂的自发"
    "遵循让'干净对照'在此设计下不可实现。"
    "**结论（弱证据）**：SKILL 注入对'过程纪律'有可测的弱效应、对'产物"
    "正确性'无可见效应。第二回合把清单/SKILL 移出工作树（全新克隆）"
    "消除了自选择污染。\n\n"
    "**第二轮结果**（2026-10-31，干净对照，retune 任务配对）：对照臂"
    "（清单移出工作树、真无 SKILL）与 R1 注入臂的代码改动**完全同构**、"
    "守门同绿（10 passed）——产物正确性仍无差异；唯一可测差异 = **标定"
    "纪律**：R1-A 臂当场跑 `KAKEYA_BENCH_PARTITION=1` A/B 并回填实测"
    "数据，对照臂推迟为'上机后应重跑'（写在注释/CHANGELOG，未执行）。"
    "与 R1 的'过程纪律弱效应'方向一致——**跨两轮稳定复现**。"
    "残余效度威胁：模型可能从训练记忆遵循相似流程（不可控）。"
    "完整 n=3×2 干净对照因对照克隆成本高未做，边际信息低（单任务已"
    "显示模式）。\n\n"
    "**跨两轮结论（中等信心）**：SKILL 注入影响**过程纪律**（先查守门"
    "涉及面、当场跑标定），**不影响产物正确性**；效应小、方向稳定。"
    "\n\n"
    "**第三轮（2026-10-31，补 n）**：对 add-backend 做干净 A/B 各 2 次"
    "（A=主仓库在场，B=对照克隆隔离）。**A 臂 2/2 次守门一次通过**"
    "（survey.tex 计数前置同步）；**B 臂 2/2 次遗漏 survey.tex 计数同步**"
    "（guide_slicing 会红）——**4/4 次方向一致**。这是目前最强的单信号："
    "SKILL 协议里'动代码前先查守门涉及面'这一条，把'新增测试导致计数守门"
    "漂移'这类**可预见但不直观**的后续动作从'事后迭代'提前为'前置同步'。"
    "\n\n"
    "**累计证据**：3 个任务 × 多轮（add-backend 4 次、retune 3 次、"
    "merge 2 次），方向一致——SKILL 注入对过程纪律的效应**稳定且可复现**，"
    "对产物正确性始终无差异；效应集中在'守门涉及面前置检查'这一具体动作上。"
)


# ---------------------------------------------------------------------------
# 调度主链 DAG（docs/diagrams/scheduler_flow.mmd 的单一真源）
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DagNode:
    """DAG 节点：mermaid id、标签（含 <br/>）、样式类、边框形状。"""

    id: str
    label: str
    klass: str
    shape_open: str = "["   # "["=矩形, "{"=菱形判定
    shape_close: str = "]"


@dataclass(frozen=True)
class DagEdge:
    """DAG 边：src → dst，可选标签（mermaid 的 --|"标签"|）。"""

    src: str
    dst: str
    label: str = ""


#: 有序语句表 = mermaid body 的单一真源。每条三选一：
#: ``("node", id)`` —— 独立节点声明；
#: ``("edge", src, dst, label, inline)`` —— 边，``inline`` = 该边上**内联
#: 声明**的节点 id（原手写文件的风格：``q --> dec["标签"]``；``""`` = 两端
#: 均已声明）；
#: ``("blank",)`` —— 空行（仅分组可读性，无语义）。
#: 节点的**首次出现顺序**决定 mermaid 布局与 class 赋值顺序，勿乱序。
SCHEDULER_FLOW_STATEMENTS: tuple[tuple[str, ...], ...] = (
    ("node", "q"),
    ("blank",),
    ("edge", "q", "dec", "", "dec"),
    ("edge", "dec", "subs", "", "subs"),
    ("blank",),
    ("edge", "subs", "plan", "", "plan"),
    ("blank",),
    ("node", "chip"),
    ("node", "pred"),
    ("edge", "chip", "plan", "消费：判瓶颈", ""),
    ("edge", "pred", "plan", "消费：选线程数", ""),
    ("blank",),
    ("edge", "plan", "task", "", "task"),
    ("blank",),
    ("edge", "task", "order", "", "order"),
    ("edge", "order", "wave", "", "wave"),
    ("edge", "wave", "serial", "是", "serial"),
    ("edge", "wave", "par", "否", "par"),
    ("blank",),
    ("node", "out"),
    ("edge", "serial", "out", "", ""),
    ("edge", "par", "out", "", ""),
    ("blank",),
    ("node", "expert"),
    ("edge", "expert", "q", "分工边界：另一条线", ""),
)

#: 节点登记表（id → 标签/样式类/形状）；声明顺序以语句表为准。
SCHEDULER_FLOW_NODES: dict[str, DagNode] = {n.id: n for n in (
    DagNode("q", "提问 (query)", "core"),
    DagNode("dec", "decompose_query<br/>关键词拆子问题", "core"),
    DagNode("subs", "SubQuestion[]<br/>各带 ModuleKind", "core"),
    DagNode("plan", "plan_query（逐子问题）", "core"),
    DagNode("chip", "chip_model<br/>判瓶颈：算力 / 带宽受限", "model"),
    DagNode("pred", "predict_schedule<br/>单核墙钟 -> 最优线程数 n", "model"),
    DagNode("task", "TaskPlan<br/>模块 + 后端 + 线程数", "core"),
    DagNode("order", "compute_execution_order<br/>拓扑分层 + 核预算裁定", "core"),
    DagNode("wave", "并发线程需求<br/>超订可调度核？", "gate", "{", "}"),
    DagNode("serial", "串行（各自满线程，避免超订）", "out"),
    DagNode("par", "并行 / host 任务可重叠", "out"),
    DagNode("out", "SchedulePlan（只调度、不执行）", "out"),
    DagNode("expert", "expert_router<br/>路由 LLM 专家（用哪个模型回答）", "model"),
)}

#: 边数据（供守门做结构断言）：从语句表**派生**，不重复维护。
#: 主链语义约定（勿改）：模型输入边与分工边界边也是实线。
SCHEDULER_FLOW_EDGES: tuple[DagEdge, ...] = tuple(
    DagEdge(s[1], s[2], s[3]) for s in SCHEDULER_FLOW_STATEMENTS if s[0] == "edge"
)

#: 样式类定义（值即 mermaid classDef 行的原样内容）。
SCHEDULER_FLOW_CLASSDEFS: tuple[tuple[str, str], ...] = (
    ("core", "fill:#eef4ff,stroke:#3b6fb6,stroke-width:2px;"),
    ("model", "fill:#fff7e6,stroke:#c99a2e,stroke-width:1px,stroke-dasharray:4 3;"),
    ("gate", "fill:#fff4e6,stroke:#c9762e,stroke-width:2px;"),
    ("out", "fill:#eaf7ee,stroke:#3f8f5c,stroke-width:2px;"),
)

#: 生成物头部：来源与再生成流程（守门测试比对整文件，手改即红）。
_GENERATED_BANNER = (
    "%% ⚠️ 本文件为**生成物**——单一真源：nano/manifest/architecture_manifest.py\n"
    "%%    （SCHEDULER_FLOW_* 数据 + render_scheduler_flow_mmd()）。\n"
    "%%    改图结构：改清单数据 → .venv/bin/python -m nano.manifest.architecture_manifest\n"
    "%%    → 可选导出位图：bash scripts/docs_build/render_diagrams.sh scheduler_flow（PNG 不入库）。\n"
    "%%    手改本文件会被 tests/tooling/test_architecture_manifest.py 判红。\n"
)

#: 语义约定（调度主链的定义口径；原 mmd 头注释的等价物）。
_CONVENTIONS = (
    "%% 动态调度器：提问 → 子问题 → 模块/资源 → 执行顺序 → 计划\n"
    "%% .\n"
    "%% 对应本清单的 §7（调度预测模型）与 §8（动态调度器）。\n"
    "%% .\n"
    "%% 语义约定（与原文一致，勿改）：\n"
    "%%   - 竖直实线是**调度主链**：每一步都对应一次真实调度决策（无占位符）；\n"
    "%%   - 左侧 chip_model / predict_schedule 是**被消费的模型**（判瓶颈、选线程数），\n"
    "%%     是\"输入\"、不是主链的一步——用**橙色虚线边框**标记（与主链的蓝色实线\n"
    "%%     边框区分），边仍为实线（虚线边若竖向，渲染的短划会被守门误判成豆腐块）；\n"
    "%%   - 右上 expert_router 是**分工边界**（路由 LLM 专家，决定\"用哪个模型回答\"，\n"
    "%%     不决定\"调哪些计算模块\"），同样橙色虚线边框示意\"另一条线，不混入\"。\n"
    "%%   - 关键事实：调度器**只产出计划、不执行计算**（执行仍走 accel / 各 kernel）。\n"
    "%% .\n"
    "%% 可选导出位图：bash scripts/docs_build/render_diagrams.sh scheduler_flow\n"
    "%% （PNG 是产物、不入库；GitHub 与编辑器直接渲染本文件的 mermaid 源）\n"
)


def _decl(node: DagNode) -> str:
    """节点声明片段：``id["标签"]`` / ``id{"标签"}``。"""
    return f"{node.id}{node.shape_open}\"{node.label}\"{node.shape_close}"


def render_scheduler_flow_mmd() -> str:
    """把 :data:`SCHEDULER_FLOW_STATEMENTS` 渲染成 scheduler_flow.mmd 全文。

    输出与磁盘文件的一致性由 ``tests/tooling/test_architecture_manifest.py`` 守门：
    手改 mmd 而不改本清单即红。
    """
    body = ["flowchart TD"]
    declared: list[str] = []  # 首次出现顺序（class 赋值按它排）
    for stmt in SCHEDULER_FLOW_STATEMENTS:
        kind = stmt[0]
        if kind == "blank":
            body.append("")
        elif kind == "node":
            nid = stmt[1]
            body.append(f"    {_decl(SCHEDULER_FLOW_NODES[nid])}")
            declared.append(nid)
        else:  # edge
            _, src, dst, label, inline = stmt
            dst_txt = _decl(SCHEDULER_FLOW_NODES[dst]) if inline else dst
            if inline:
                declared.append(dst)
            arrow = f"-->|\"{label}\"|" if label else "-->"
            body.append(f"    {src} {arrow} {dst_txt}")

    body.append("")
    for klass, style in SCHEDULER_FLOW_CLASSDEFS:
        body.append(f"    classDef {klass} {style}")
    # class 赋值：按类定义顺序、类内按节点**首次出现**顺序（与原手写一致）。
    for klass, _ in SCHEDULER_FLOW_CLASSDEFS:
        members = [n for n in declared if SCHEDULER_FLOW_NODES[n].klass == klass]
        if members:
            body.append(f"    class {','.join(members)} {klass};")

    return _GENERATED_BANNER + _CONVENTIONS + "\n\n" + "\n".join(body) + "\n"


def main() -> int:
    """命令行：默认重写 mmd 生成物；``--check`` 只比对不写（CI/守门用）。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="只比对生成物与磁盘文件（不一致时退出码 1，不写盘）",
    )
    args = parser.parse_args()

    rendered = render_scheduler_flow_mmd()
    if args.check:
        on_disk = SCHEDULER_FLOW_MMD.read_text(encoding="utf-8")
        if on_disk != rendered:
            print(f"[architecture_manifest] {SCHEDULER_FLOW_MMD} 与清单不一致"
                  "——请改 nano/manifest/architecture_manifest.py 后重跑本生成器"
                  "（不要手改 mmd）。")
            return 1
        print("[architecture_manifest] 生成物与清单一致 ✓")
        return 0

    SCHEDULER_FLOW_MMD.write_text(rendered, encoding="utf-8")
    print(f"[architecture_manifest] 已重写 {SCHEDULER_FLOW_MMD}"
          "（需要位图时：bash scripts/docs_build/render_diagrams.sh scheduler_flow）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
