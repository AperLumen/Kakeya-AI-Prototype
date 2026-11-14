"""驻留准入控制：任何东西进内存之前，先过三道初筛。

设计原则（用户定调，2026-11-04）：**运行内存里存的东西一定要有用**。
以 Ollama 模型驻留为首个应用场景（gemma4:12b 常驻 ~8 GB 的教训），但
机制本身是通用的——任何"占大块内存的驻留对象"都可走同一套筛子。

三道初筛，按"最便宜先查"的顺序执行，任一不过即拒（fail-fast）：

1. **需求筛**（:func:`demand_gate`）：请求必须带真实用途与预估消耗，
   拒绝"先加载备用""以防万一"这类推测性驻留——它们正是内存里
   "存了但没在用"的东西的主要来源。
2. **资源筛**（:func:`resource_gate`）：装得下且留守卫线。用
   :func:`kakeya_ai.resource.probe.memory_snapshot` 的全字段（不止
   available 标量）：扣除驻留对象体积后仍须 ≥ 2 GB 守卫线，且
   内存压力（pressure）低于阈值。守卫线默认复用
   :func:`kakeya_ai.resource.allocator.reserve_bytes`。
3. **效用筛**（:func:`utility_gate`）：这个对象对该任务是最低充分选择。
   能力过剩（杀鸡用牛刀）和能力不足（打蚊子用大炮）都拒——
   内存要留给"恰好够用"的东西。

全部筛子是纯函数（快照与请求作为入参），可注入、可单测；
编排入口 :func:`admit` 返回 :class:`AdmissionVerdict`，拒绝时给出
``gate``（哪道筛子）与 ``suggestion``（怎么改：换更小的模型、
驱逐低效驻留、或干脆不做这件事）。

**场景清单与开关决策（Q4 #16，2026-11-13 评估定案）**：

1. **交互式 CLI**（``scripts/interfaces/ollama_chat.py``，人面前、单模型
   加载）：demand/resource **hard gate**（拒绝并给建议）＋ utility
   **warn-only**（实测最低充分选择可继续）——现状即定案。
2. **库路径 / 级联内部**（``cascaded_reasoning`` / ``two_stage`` /
   ``beam`` 等程序化调用）：**不做 admission hard gate**——2026-11-06
   教训：库路径 hard gate 误杀级联（程序化调用的"需求"无法被初筛正确
   估计）。该路径只有 ``memory_guard`` 的观测警告（warn-only），保持。
3. **长任务**（编排器 / 训练 / 批量评测）：hard gate 若启用只能在
   **启动前一次性**做——运行中被拒的代价是整个 episode 作废；运行中
   只观测。现状：这些路径不经过 ``admit()``，符合定案；将来若加启动前
   检查，必须一次性而非逐调用。
4. **CI / 测试**：无 Ollama，admission 不适用（skip），保持。
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from . import allocator, probe

__all__ = [
    "AdmissionVerdict",
    "LoadRequest",
    "ModelProfile",
    "MODEL_REGISTRY",
    "admit",
    "demand_gate",
    "resource_gate",
    "utility_gate",
]

# 守卫线：复用 allocator 的默认预留（2 GB），保持全库一个口径。
RESERVE_BYTES = allocator.DEFAULT_RESERVE_BYTES
# 内存压力阈值：pressure = 1 - available/total，超过即视为吃紧。
PRESSURE_LIMIT = 0.85

# 用途 → 需求强度（0~1）。粗粒度三档即可支撑"最低充分"判定：
# 日常问答/轻推理 0.4，写作/总结 0.6，数学推导/前沿概念 0.9。
PURPOSE_DEMAND: Dict[str, float] = {
    "chat": 0.4,
    "write": 0.6,
    "math": 0.9,
}


@dataclass(frozen=True)
class ModelProfile:
    """驻留对象的画像：体积与能力分（0~1，按用途维度）。"""

    name: str
    footprint_bytes: int
    capability: float  # 综合能力分；细粒度维度将来需要再拆

    @property
    def footprint_gb(self) -> float:
        return self.footprint_bytes / 2**30


@dataclass(frozen=True)
class LoadRequest:
    """一次驻留请求：要装什么、为谁装、预计用多少。"""

    model: ModelProfile
    purpose: str  # 键于 PURPOSE_DEMAND；未登记按最高需求 0.9 处理（偏保守）
    estimated_tokens: int = 1024

    @property
    def demand(self) -> float:
        return PURPOSE_DEMAND.get(self.purpose, 0.9)


@dataclass(frozen=True)
class AdmissionVerdict:
    """准入裁定：过/不过、卡在哪道筛、建议怎么办。"""

    allowed: bool
    gate: str  # 卡在哪道筛子：需求 / 资源 / 效用；全过记 "ok"
    reason: str
    suggestion: str = ""


# 本机已登记模型（实测常驻统一内存的体积，2026-11-03/04）。
# 新模型进本机后应在此登记，否则效用筛按"未知=最贵"处理。
MODEL_REGISTRY: Dict[str, ModelProfile] = {
    "qwen3.5:4b": ModelProfile("qwen3.5:4b", 4 * 2**30, capability=0.6),
    "gemma4:12b": ModelProfile("gemma4:12b", 8 * 2**30, capability=0.9),
}


def demand_gate(request: LoadRequest) -> Optional[str]:
    """需求筛：拒绝推测性驻留。返回拒绝原因（None 表示通过）。

    判据只有一条——请求必须说清"为谁、干什么"。说不清的（空用途、
    零消耗预估）一律视为"以防万一"式加载，这正是要拦的东西。
    """
    if not request.purpose:
        return "用途为空：推测性驻留（'先加载备用'）不占内存"
    if request.estimated_tokens <= 0:
        return "预估消耗为 0：无事可做的驻留请求"
    return None


def resource_gate(
    request: LoadRequest,
    memory: "probe.MemorySnapshot",
    reserve_bytes: int = RESERVE_BYTES,
    pressure_limit: float = PRESSURE_LIMIT,
) -> Optional[str]:
    """资源筛：装得下且留守卫线、压力不超限。返回拒绝原因（None 通过）。"""
    remaining = memory.available - request.model.footprint_bytes
    if remaining < reserve_bytes:
        return (
            f"装不下：可用 {memory.available / 2**30:.2f}G - 驻留 "
            f"{request.model.footprint_gb:.1f}G = {remaining / 2**30:.2f}G "
            f"< 守卫线 {reserve_bytes / 2**30:.1f}G"
        )
    if memory.pressure > pressure_limit:
        return f"内存压力过高：pressure={memory.pressure:.2f} > {pressure_limit:.2f}"
    return None


def utility_gate(
    request: LoadRequest,
    registry: Dict[str, ModelProfile] = MODEL_REGISTRY,
) -> Optional[str]:
    """效用筛：必须是"最低充分"选择。返回拒绝原因（None 通过）。

    不足（capability < demand）拒；过剩——存在更小且仍满足需求的
    已登记模型——也拒，建议换那个更小的。
    """
    demand = request.demand
    if request.model.capability < demand:
        return (
            f"能力不足：{request.model.name} capability={request.model.capability:.2f} "
            f"< 需求 {demand:.2f}"
        )
    smaller_sufficient = [
        p
        for p in registry.values()
        if p.footprint_bytes < request.model.footprint_bytes
        and p.capability >= demand
    ]
    if smaller_sufficient:
        best = min(smaller_sufficient, key=lambda p: p.footprint_bytes)
        return (
            f"能力过剩：{best.name}（{best.footprint_gb:.0f}G）已满足需求 "
            f"{demand:.2f}，不必驻留 {request.model.name}"
        )
    return None


def _suggest_eviction(
    request: LoadRequest,
    residents: Dict[str, "ModelProfile"],
    memory: "probe.MemorySnapshot",
) -> str:
    """资源筛被拒时的建议：驱逐效用最低的驻留者（若驱逐后能装下）。"""
    if not residents:
        return "无驻留者可驱逐；等 keep_alive 到期卸载或释放其他内存后再试"
    # 请求的目标模型不计入驱逐候选（驻留表里同名的就是自己）。
    others = [p for n, p in residents.items() if n != request.model.name]
    if not others:
        return "无其他驻留者可驱逐；等 keep_alive 到期卸载或释放其他内存后再试"
    victims = sorted(others, key=lambda p: p.capability)
    victim = victims[0]
    freed = memory.available + victim.footprint_bytes - request.model.footprint_bytes
    if freed >= RESERVE_BYTES:
        return (
            f"可先驱逐效用最低的 {victim.name}（capability={victim.capability:.2f}），"
            f"驱逐后余量 {freed / 2**30:.1f}G ≥ 守卫线"
        )
    return f"驱逐 {victim.name} 后仍不够（余量 {freed / 2**30:.1f}G），需释放更多"


def admit(
    request: LoadRequest,
    residents: Optional[Dict[str, ModelProfile]] = None,
    memory: Optional["probe.MemorySnapshot"] = None,
) -> AdmissionVerdict:
    """三道初筛的编排入口。任一不过即拒，全过才放行。"""
    fail = demand_gate(request)
    if fail:
        return AdmissionVerdict(False, "demand", fail)

    snap = memory if memory is not None else probe.memory_snapshot()
    fail = resource_gate(request, snap)
    if fail:
        suggestion = _suggest_eviction(request, residents or {}, snap)
        return AdmissionVerdict(False, "resource", fail, suggestion)

    fail = utility_gate(request)
    if fail:
        return AdmissionVerdict(
            False, "utility", fail, "换建议的更小模型，或确认任务确实需要它"
        )
    return AdmissionVerdict(True, "ok", f"{request.model.name} 可驻留")
