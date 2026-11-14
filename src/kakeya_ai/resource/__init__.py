"""资源调度子包：内存、CPU、GPU 配额的统一申请与释放。

当前主要实现内存配额（``MemoryQuota`` / ``MemoryAllocator``），因为项目最紧迫
的约束是 16 GB 统一内存下 GPU（LLM）与几何计算的争用。未来可沿同一接口扩展
CPU 核、HTTP 并发槽等资源的调度。

设计要点
--------

1. **显式配额**：任务调用 ``MemoryAllocator.request`` 申请本次需要的内存，
   用完后必须 ``release``。守卫从"该不该放行"升级为"有没有足够配额可分配"。
2. **向后兼容**：``kakeya_ai.memory_guard`` 的公开 API 全部保留并继续导出，
   内部转发到本包实现。外部代码与测试不必立即迁移。
3. **与 ``memory_guard`` 同语义**：默认仍是"持续不足才拒绝"，避免页回收造成
   的瞬时低谷误拦任务。strict 模式保留给"马上要分配大缓冲"的场景。

使用示例
--------

::

    from kakeya_ai.resource import allocator

    with allocator.acquire(required=500_000_000) as quota:
        # 已确认留出系统保留 2 GB + 本次 500 MB 后仍有余量。
        result = run_mlx_kernel(...)
    # 退出 with 自动释放配额
"""

from __future__ import annotations

from .admission import (
    MODEL_REGISTRY,
    AdmissionVerdict,
    LoadRequest,
    ModelProfile,
    admit,
    demand_gate,
    resource_gate,
    utility_gate,
)
from .allocator import (
    DEFAULT_RESERVE_BYTES,
    InsufficientMemoryError,
    MemoryAllocator,
    MemoryQuota,
    RESERVE_ENV,
    acquire,
    allocator,
    available_memory,
    check_gpu_headroom,
    describe,
    has_gpu_headroom,
    is_sustained_low,
    recent_trend,
    release_all,
    reserve_bytes,
    reset_cache,
    warn_if_low_memory,
)
from .probe import (
    SUSTAINED_INTERVAL,
    SUSTAINED_SAMPLES,
    VERDICT_TTL,
    VM_STAT_TIMEOUT,
    CpuSnapshot,
    MemorySnapshot,
    SystemSnapshot,
    cpu_snapshot,
    memory_snapshot,
    snapshot,
)

__all__ = [
    "DEFAULT_RESERVE_BYTES",
    "InsufficientMemoryError",
    "MODEL_REGISTRY",
    "MemoryAllocator",
    "MemoryQuota",
    "AdmissionVerdict",
    "LoadRequest",
    "ModelProfile",
    "RESERVE_ENV",
    "SUSTAINED_INTERVAL",
    "SUSTAINED_SAMPLES",
    "VERDICT_TTL",
    "VM_STAT_TIMEOUT",
    "acquire",
    "admit",
    "allocator",
    "available_memory",
    "check_gpu_headroom",
    "cpu_snapshot",
    "demand_gate",
    "describe",
    "has_gpu_headroom",
    "is_sustained_low",
    "memory_snapshot",
    "MemorySnapshot",
    "CpuSnapshot",
    "SystemSnapshot",
    "recent_trend",
    "release_all",
    "reserve_bytes",
    "reset_cache",
    "resource_gate",
    "snapshot",
    "utility_gate",
    "warn_if_low_memory",
]
