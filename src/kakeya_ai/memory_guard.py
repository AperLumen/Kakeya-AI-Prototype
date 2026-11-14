"""内存守卫：在使用 GPU 前确保留够系统余量。

.. deprecated::
   本模块已拆分为 ``kakeya_ai.resource`` 子包。这里保留旧 API 以兼容既有代码；
   新代码请使用 ``kakeya_ai.resource`` 的配额 API（``MemoryAllocator`` /
   ``MemoryQuota`` / ``acquire``）。

原 docstring 的完整内容已迁移到 ``kakeya_ai.resource.allocator`` 与
``kakeya_ai.resource.probe``。
"""

from __future__ import annotations

from .resource import (
    DEFAULT_RESERVE_BYTES,
    InsufficientMemoryError,
    RESERVE_ENV,
    SUSTAINED_INTERVAL,
    SUSTAINED_SAMPLES,
    VERDICT_TTL,
    VM_STAT_TIMEOUT,
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

__all__ = [
    "DEFAULT_RESERVE_BYTES",
    "InsufficientMemoryError",
    "RESERVE_ENV",
    "SUSTAINED_INTERVAL",
    "SUSTAINED_SAMPLES",
    "VERDICT_TTL",
    "VM_STAT_TIMEOUT",
    "available_memory",
    "check_gpu_headroom",
    "describe",
    "has_gpu_headroom",
    "is_sustained_low",
    "recent_trend",
    "release_all",
    "reserve_bytes",
    "reset_cache",
    "warn_if_low_memory",
]
