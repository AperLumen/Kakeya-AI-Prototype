"""多租户资源仲裁：把"这张芯片有多少核"变成"**这个租户**被授予多少核"。

**为什么需要它**（2026-10-08）：``nano/scheduling/scheduler.py`` 是**单租户独占**的规划器——
它直接从芯片拓扑算"可调度核数"，隐含"整块芯片归我一个租户"。在云服务厂商 /
多租户共用一台机器时，N 个租户各自的 :func:`~nano.scheduling.scheduler.plan_query` 都独立
宣布"我要 8 核"，它们**互不知情**，于是在 10 核节点上开出 N×8 线程——这正是
:func:`~nano.scheduling.scheduler.compute_execution_order` 已修复的"**单个计划内**超订"的
**跨租户放大版**。单个计划内的裁定救不了跨计划，必须有一层**节点级账本**。

**分层定位**（本模块 = L1，scheduler = L2）::

    L0  集群/放置（选节点、发放 grant）        —— 不在本仓库范围
    L1  节点本地仲裁（ResourceGrant/NodeLedger）—— **本模块**
    L2  租户内规划（nano/scheduling/scheduler.py）        —— 消费 grant 拆子问题
    L3  执行 + 强制（accel / cgroups）         —— OS 真正限核

**本模块只回答"给多少"（授信）**，不回答"怎么用"——后者是 scheduler 的事；
它**不 import** scheduler（避免环），只被 scheduler 依赖。

**OS 强制的边界（如实登记）**：``NodeLedger`` 是**语义层**的记账——它算出
"这个租户最多该用几核"，但**不真正限制** CPU。真正的强制在 Linux 上用
``cgroups``（``cpu.max`` / ``memory.max``）；本机是 macOS，只能语义模拟。
故本模块的价值是"给调度器一个**可注入的核预算**"（把"芯片固有核数"换成
"上层发放的授权核数"），让整套调度**从单租户独占变成可插多租户**——这是那场
架构跃迁里**最小、且不破坏现有行为**的一步。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "ResourceGrant",
    "AdmissionDecision",
    "QueueTicket",
    "NodeLedger",
    "POLICY_EQUAL_SHARE",
    "POLICY_FIFO",
    "PRIORITY_INTERACTIVE",
    "PRIORITY_BATCH",
]

#: 准入策略：**等分**（默认）。同时活跃的租户越多，每个租户上限越低
#: （``(总核数 - 保留) // 活跃租户数``）。抗"一个大租户饿死别人"。
POLICY_EQUAL_SHARE = "equal_share"
#: 准入策略：**先到先得**。能吃多少给多少，直到空闲耗尽。吞吐优先、公平靠后。
POLICY_FIFO = "fifo"

#: 优先级类：**交互式**（默认，数值最小 = 最先服务）。对应"有人在等结果"的
#: 请求（如一次 API 调用）。队列按 ``(priority, seq)`` 升序出队。
PRIORITY_INTERACTIVE = 0
#: 优先级类：**批处理**。可等，让位给交互式。
PRIORITY_BATCH = 1


@dataclass(frozen=True)
class ResourceGrant:
    """上层（集群放置 / 节点仲裁）发放给**某个租户**的一份资源授权。

    这是把 ``nano/scheduling/scheduler.py`` 从"单租户独占规划器"改成"**租户内规划器**"
    的关键：``plan_query(grant=...)`` 之后，可调度核数不再是"芯片有多少核"，
    而是"**这个租户被授予多少核**"。

    单位一律显式：``cores`` 是逻辑核数，``memory_bytes`` 是字节，
    ``gpu_units`` 是 GPU 配额单位（0 = 不用 GPU；本项目 GPU 只服务 LLM）。
    """

    #: 租户标识（名字/ID）。同名字重复 admit 视为**同一个租户**（幂等）。
    tenant: str
    #: 授予的**可调度核数**（≥1）。这是 scheduler 的核预算真源。
    cores: int
    #: 授予的**内存配额**（字节）。0 = 不限制（本项目当前口径）。
    memory_bytes: int = 0
    #: 授予的 **GPU 配额**（单位由上层定义）。0 = 不用 GPU。
    gpu_units: int = 0
    #: 可审计说明（这份授权从哪来、为什么是这个数）。
    note: str = ""

    def __post_init__(self) -> None:
        if not self.tenant:
            raise ValueError("ResourceGrant.tenant 不能为空")
        if self.cores < 1:
            raise ValueError(f"cores 必须 ≥1：{self.cores}")
        if self.memory_bytes < 0:
            raise ValueError(f"memory_bytes 必须 ≥0：{self.memory_bytes}")
        if self.gpu_units < 0:
            raise ValueError(f"gpu_units 必须 ≥0：{self.gpu_units}")

    def explain(self) -> str:
        parts = [f"租户 {self.tenant!r}：核 {self.cores}"]
        if self.memory_bytes:
            parts.append(f"内存 {self.memory_bytes / 1024 ** 3:.2f} GB")
        if self.gpu_units:
            parts.append(f"GPU {self.gpu_units}")
        if self.note:
            parts.append(self.note)
        return "；".join(parts)


@dataclass(frozen=True)
class QueueTicket:
    """一份**排队等待**的资源请求（准入队列的一张票）。

    :meth:`NodeLedger.submit` 在当下给不出资源时，返回一张票而不是硬拒——
    释放资源后 :meth:`NodeLedger.drain` 按 ``(priority, seq)`` 升序服务它们。
    这是"**不抢占**也要让等待者最终拿到资源"的落点：不做抢占（不中断在跑的
    租户），但**排队 + 释放后重平衡**，避免后到者被永久饿死。

    ``seq`` 是同优先级内的**先到先服务**序号（由 :class:`NodeLedger` 单调分配）。
    """

    tenant: str
    cores: int
    memory_bytes: int = 0
    gpu_units: int = 0
    #: 优先级类（数值越小越先服务，见 :data:`PRIORITY_INTERACTIVE`）。
    priority: int = PRIORITY_INTERACTIVE
    #: 同优先级内的到达序号（FIFO 的 tiebreak）。
    seq: int = 0


@dataclass(frozen=True)
class AdmissionDecision:
    """一次准入判定的结果（**永远返回**，而不是抛异常——准入失败是常规结果）。

    三态（由 :attr:`status` / :attr:`granted` / :attr:`queued` 区分）：

    * ``granted=True``：:attr:`grant` 是发放的 :class:`ResourceGrant`；
    * ``queued=True``：:attr:`ticket` 是一张排队票（:meth:`NodeLedger.submit`
      在资源不足时入队）；
    * 其余（``granted=False`` 且无票）：**拒绝**，:attr:`reason` 说明为什么
      （空闲不足 / 参数非法）。

    调用方据 ``status`` 决定"跑 / 等 / 拒"。
    """

    granted: bool
    grant: Optional[ResourceGrant]
    reason: str
    #: 排队票（仅在 :meth:`NodeLedger.submit` 入队时非空）。
    ticket: Optional[QueueTicket] = None

    @property
    def queued(self) -> bool:
        """是否处于"已入队等待"状态（未授予但有票）。"""
        return self.ticket is not None and not self.granted

    @property
    def status(self) -> str:
        """三态字符串：``"granted"`` / ``"queued"`` / ``"rejected"``。"""
        if self.granted:
            return "granted"
        if self.queued:
            return "queued"
        return "rejected"


@dataclass
class NodeLedger:
    """**节点级**资源账本：单机多租户的"唯一真源"。

    它记账三件事——容量（``total_*``）、预留给系统/LLM 的部分（``reserved_*``）、
    已发放给各租户的授权（``_allocations``）——并据此做**准入**。

    **为什么是"账本"而不是"锁"**：本模块只回答"该给多少"，不强制。真正的
    限制由 L3（cgroups / 线程池尺寸）施加；账本保证**决策一致**（不会账面上
    超发）。这样 L2（scheduler）拿到的永远是"合法预算"。

    **线程安全**：当前实现**不加锁**——假定调用方在单线程仲裁路径上使用
    （如一个调度循环）。多线程并发准入需外部加锁；这点如实登记，不假装安全。
    """

    #: 节点总逻辑核数（如 M4 = 4P + 6E = 10）。
    total_cores: int
    #: 节点总内存（字节）。0 = 不记账内存。
    total_memory_bytes: int = 0
    #: 节点总 GPU 配额单位。0 = 不用 GPU。
    total_gpu_units: int = 0
    #: 预留给系统 / LLM（Ollama）等的核数（不计入可分配）。
    reserved_cores: int = 0
    #: 预留内存（字节）。
    reserved_memory_bytes: int = 0
    #: 预留 GPU 配额。
    reserved_gpu_units: int = 0
    #: 默认准入策略（见 :data:`POLICY_EQUAL_SHARE` / :data:`POLICY_FIFO`）。
    policy: str = POLICY_EQUAL_SHARE
    #: 租户 → 已发放授权。内部记账，勿直接改。
    _allocations: dict = field(default_factory=dict, repr=False)
    #: 准入队列（等待中的 :class:`QueueTicket`）。内部记账，勿直接改。
    _queue: list = field(default_factory=list, repr=False)
    #: 单调到达序号（队列 FIFO 的 tiebreak）。内部记账，勿直接改。
    _seq: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.total_cores < 1:
            raise ValueError(f"total_cores 必须 ≥1：{self.total_cores}")
        if self.total_memory_bytes < 0 or self.total_gpu_units < 0:
            raise ValueError("total_memory_bytes / total_gpu_units 必须 ≥0")
        if not (0 <= self.reserved_cores < self.total_cores):
            raise ValueError(
                f"reserved_cores 须在 [0, total_cores)："
                f"reserved={self.reserved_cores} total={self.total_cores}"
            )
        if self.policy not in (POLICY_EQUAL_SHARE, POLICY_FIFO):
            raise ValueError(f"未知准入策略：{self.policy!r}")

    # ------------------------------------------------------------------
    # 只读视图
    # ------------------------------------------------------------------

    @property
    def allocatable_cores(self) -> int:
        """可分配核数（总量 - 保留）——即"全给一个租户"时的上限。"""
        return self.total_cores - self.reserved_cores

    @property
    def allocated_cores(self) -> int:
        """已发放给所有租户的核数之和。"""
        return sum(g.cores for g in self._allocations.values())

    @property
    def free_cores(self) -> int:
        """尚未发放的核数（可分配 - 已发放）。恒 ≥0。"""
        return self.allocatable_cores - self.allocated_cores

    def grant_for(self, tenant: str) -> Optional[ResourceGrant]:
        """查某租户当前的授权（未授予返回 ``None``）。"""
        return self._allocations.get(tenant)

    def active_tenants(self) -> list:
        """当前持有授权的租户名（排序，便于审计与测试）。"""
        return sorted(self._allocations)

    # ------------------------------------------------------------------
    # 准入 / 释放
    # ------------------------------------------------------------------

    def admit(
        self,
        tenant: str,
        *,
        cores: int,
        memory_bytes: int = 0,
        gpu_units: int = 0,
        policy: Optional[str] = None,
    ) -> AdmissionDecision:
        """为 ``tenant`` 准入并发放一份 :class:`ResourceGrant`。

        **幂等**：该租户已有授权时，直接返回既有授权（不重复记账）——避免
        "同一租户重复 admit 导致账面翻倍"这类静默漂移。

        **公平**（``equal_share``，默认）：授予上限 =
        ``max(1, (总核数 - 保留) // 当前活跃租户数(含本次))``——同时活跃的租户
        越多，每人上限越低。``fifo`` 则能吃多少给多少，直到空闲耗尽。

        **⚠️ 在线贪心 + 不抢占的固有边界（如实登记）**：``admit`` 只在**准入那一刻**
        按当时的活跃数设上限，**不做事后回收**。若租户 A 先到、独占 8 核，租户 B
        后到时会因空闲为 0 而被拒——即使按等分 B 本应得 4。这是 ``admit`` 的
        **瞬时判定**语义（给不了就拒）。

        **缓解**：面向调度循环的调用方应改用 :meth:`submit`——它给不了就**入队**
        （返回票），待 A :meth:`release` 后 :meth:`drain` 按 ``(优先级, 先到)``
        兑现，从而**避免后到者被永久饿死**。仍**不做抢占**（不中断在跑的租户，
        ``ResourceGrant`` 一旦发放即不可变）；真正的强隔离靠 L0（放置层）或
        L3（cgroup 硬限）。别把 ``admit`` 当成强隔离。

        **内存/GPU**：按"空闲量"上限截断；为 0 表示不限制/不使用。

        返回 :class:`AdmissionDecision`——**失败也是正常返回**（``granted=False``），
        调用方据此排队或拒绝，不必 ``try/except``。
        """
        pol = policy or self.policy

        # 幂等：已持有 ⇒ 直接返回既有授权。
        existing = self._allocations.get(tenant)
        if existing is not None:
            return AdmissionDecision(True, existing, "该租户已持有授权（幂等返回）")

        if cores < 1:
            return AdmissionDecision(False, None, f"请求核数非法（须 ≥1）：{cores}")

        # 公平上限：等分时按"含本次的活跃租户数"分摊。
        if pol == POLICY_EQUAL_SHARE:
            n = len(self._allocations) + 1
            fair_cores = max(1, self.allocatable_cores // n)
        else:
            fair_cores = self.allocatable_cores

        granted_cores = min(cores, self.free_cores, fair_cores)
        if granted_cores < 1:
            return AdmissionDecision(
                False, None,
                f"空闲核不足（空闲 {self.free_cores}，公平上限 {fair_cores}）= 拒绝",
            )

        free_mem = max(0, self.total_memory_bytes - self.reserved_memory_bytes
                       - sum(g.memory_bytes for g in self._allocations.values()))
        free_gpu = max(0, self.total_gpu_units - self.reserved_gpu_units
                       - sum(g.gpu_units for g in self._allocations.values()))
        granted_mem = min(memory_bytes, free_mem) if self.total_memory_bytes else memory_bytes
        granted_gpu = min(gpu_units, free_gpu) if self.total_gpu_units else gpu_units

        note = f"准入策略 {pol}（公平上限 {fair_cores}，空闲 {self.free_cores}）"
        grant = ResourceGrant(
            tenant=tenant, cores=granted_cores,
            memory_bytes=granted_mem, gpu_units=granted_gpu, note=note,
        )
        self._allocations[tenant] = grant
        return AdmissionDecision(True, grant, note)

    def release(self, tenant: str) -> Optional[ResourceGrant]:
        """释放某租户的授权，归还其核/内存/GPU 到空闲池。

        返回被释放的授权（未持有时返回 ``None``）——便于"释放给我看"的审计。

        **不自动排空队列**：调用方（调度循环）应在 ``release`` 后显式调
        :meth:`drain`，让排队者获得资源。拆成两步是为了可测、可审计，且让
        "释放"与"重平衡"各自的时机显式可见。
        """
        return self._allocations.pop(tenant, None)

    # ------------------------------------------------------------------
    # 准入队列（P1：不抢占，但排队 + 释放后重平衡）
    # ------------------------------------------------------------------

    def submit(
        self,
        tenant: str,
        *,
        cores: int,
        memory_bytes: int = 0,
        gpu_units: int = 0,
        priority: int = PRIORITY_INTERACTIVE,
        policy: Optional[str] = None,
    ) -> AdmissionDecision:
        """请求资源：**能立即给就发授权，给不了就入队**（不硬拒）。

        与 :meth:`admit` 的分工：``admit`` 是"现在能不能给"的**瞬时判定**
        （给不了就拒）；``submit`` 是面向**调度循环**的请求入口——给不了就
        返回一张 :class:`QueueTicket`，等 :meth:`release` + :meth:`drain`
        再兑现。**不抢占**（不动正在跑的租户），但通过排队避免后到者被饿死。

        **幂等**：该租户已持有授权 ⇒ 返回既有授权；已在队列中 ⇒ 返回既有票。
        """
        d = self.admit(
            tenant, cores=cores, memory_bytes=memory_bytes,
            gpu_units=gpu_units, policy=policy,
        )
        if d.granted:
            return d
        existing = self._queued_ticket(tenant)
        if existing is not None:
            return AdmissionDecision(
                False, None, "该租户已在队列中（幂等返回）", ticket=existing,
            )
        ticket = QueueTicket(
            tenant=tenant, cores=cores, memory_bytes=memory_bytes,
            gpu_units=gpu_units, priority=priority, seq=self._seq,
        )
        self._seq += 1
        self._queue.append(ticket)
        return AdmissionDecision(False, None, f"入队等待（{d.reason}）", ticket=ticket)

    def drain(self) -> list:
        """按 ``(priority, seq)`` 升序，把队列里**能装下**的请求准入。

        对每张票重试 :meth:`admit`；成功者发放授权并从队列移除，失败者留队。
        **不中断正在跑的租户**（无抢占）——只是把刚释放/本就空闲的容量分配给
        等待者。返回本次**新准入**的 :class:`AdmissionDecision` 列表。

        典型用法（调度循环）：``ledger.release(tenant); ledger.drain()``。
        """
        admitted: list = []
        remaining: list = []
        for ticket in sorted(self._queue, key=lambda t: (t.priority, t.seq)):
            d = self.admit(
                ticket.tenant, cores=ticket.cores,
                memory_bytes=ticket.memory_bytes, gpu_units=ticket.gpu_units,
            )
            if d.granted:
                admitted.append(d)
            else:
                remaining.append(ticket)
        self._queue = remaining
        return admitted

    def pending(self) -> list:
        """当前排队中的票（按服务顺序 ``(priority, seq)`` 排序，可审计）。"""
        return sorted(self._queue, key=lambda t: (t.priority, t.seq))

    @property
    def queued_count(self) -> int:
        """队列长度。"""
        return len(self._queue)

    def _queued_ticket(self, tenant: str) -> Optional[QueueTicket]:
        for t in self._queue:
            if t.tenant == tenant:
                return t
        return None

    def explain(self) -> str:
        """人可审阅的账本快照（容量 / 保留 / 已发放 / 空闲 / 各租户）。"""
        lines = [
            f"节点账本：总核 {self.total_cores}，保留 {self.reserved_cores}，"
            f"可分配 {self.allocatable_cores}",
            f"  已发放 {self.allocated_cores}，空闲 {self.free_cores}，"
            f"策略 {self.policy}",
        ]
        if self.total_memory_bytes:
            lines.append(f"  内存：总 {self.total_memory_bytes / 1024 ** 3:.2f} GB，"
                         f"保留 {self.reserved_memory_bytes / 1024 ** 3:.2f} GB")
        if self.total_gpu_units:
            lines.append(f"  GPU：总 {self.total_gpu_units}，保留 {self.reserved_gpu_units}")
        if not self._allocations:
            lines.append("  （无活跃租户）")
        for tenant in sorted(self._allocations):
            lines.append(f"  - {self._allocations[tenant].explain()}")
        if self._queue:
            lines.append(f"  队列：{len(self._queue)} 个等待 "
                         f"（按优先级/先到服务）")
            for t in self.pending():
                lines.append(f"  - 排队 {t.tenant!r}：要 {t.cores} 核"
                             f"（优先级 {t.priority}，seq {t.seq}）")
        return "\n".join(lines)
