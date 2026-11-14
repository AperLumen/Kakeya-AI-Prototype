"""内存配额分配器：在 GPU/LLM 任务前确保系统余量，并跟踪已分配配额。

设计要点
--------

1. **全局单例**：``allocator`` 是进程内唯一的 ``MemoryAllocator`` 实例。所有
   内存配额向它申请，避免多个模块各自估算、重复扣除。
2. **配额只跟踪，不真正 malloc**：本模块不做系统级内存锁定（那需要 root 和
   平台特定 API），而是维护一个"已承诺给任务"的字典。调用方拿到 ``MemoryQuota``
   后仍按正常方式分配数组；分配器保证在承诺总量+系统保留之外还有余量。
3. **可重入**：同一任务可多次申请额外配额；释放时按 quota id 归还，不会误放
   别人的配额。
4. **strict 模式保留**：给"马上要按 ``required`` 大小分配实打实的缓冲"的场景用，
   此时只做瞬时判定。

与 ``memory_guard`` 的关系
--------------------------

``memory_guard`` 的公开 API 现在由本模块提供：

- ``available_memory``、``is_sustained_low``、``recent_trend``、``reset_cache``
  转发到 ``resource.probe``。
- ``check_gpu_headroom`` / ``has_gpu_headroom`` 在请求配额为 0 时等价于原语义。
- ``InsufficientMemoryError`` 保留同一异常类型，位于 ``resource.allocator``，
  但 ``memory_guard`` 仍导出同名异常以兼容既有 import。
"""

from __future__ import annotations

import threading
import warnings
from contextlib import contextmanager
from typing import Dict, Generator, Optional

from ..config import as_bytes
from ..errors import BackendUnavailableError
from . import probe

#: 默认为系统保留的空闲内存（字节）。2 GB。
DEFAULT_RESERVE_BYTES = 2 * 1024**3

#: 允许用环境变量覆盖（测试与特殊部署用）。
RESERVE_ENV = "KAKEYA_GPU_MEMORY_RESERVE"

#: 模块级常量便捷导出（兼容 ``memory_guard`` 旧 API）。
SUSTAINED_SAMPLES = probe.SUSTAINED_SAMPLES
SUSTAINED_INTERVAL = probe.SUSTAINED_INTERVAL
VERDICT_TTL = probe.VERDICT_TTL
VM_STAT_TIMEOUT = probe.VM_STAT_TIMEOUT


class InsufficientMemoryError(BackendUnavailableError):
    """空闲内存持续不足以满足预留与本次配额要求。"""

    def __init__(
        self,
        available: int,
        required: int,
        reserve: int,
        allocated: int = 0,
        samples: int = 0,
        window: float = 0.0,
    ) -> None:
        self.available = available
        self.required = required
        self.reserve = reserve
        self.allocated = allocated
        self.samples = samples
        self.window = window
        detail = (f"（连续 {samples} 次探测、约 {window:.2f} s 内均如此）"
                  if samples > 1 else "")
        super().__init__(
            f"空闲内存持续不足{detail}：可用 {available / 1024**3:.2f} GB，"
            f"已分配 {allocated / 1024**3:.2f} GB，"
            f"本次需要 {required / 1024**3:.2f} GB，"
            f"且需为系统保留 {reserve / 1024**3:.2f} GB。",
            backend="system-memory",
        )


class MemoryQuota:
    """一次内存配额的句柄。

    用户通常不直接构造，而是通过 ``MemoryAllocator.acquire`` 获得。
    作为上下文管理器使用时，退出即释放。
    """

    def __init__(self, quota_id: int, allocated: int) -> None:
        self.quota_id = quota_id
        self.allocated = allocated
        self._released = False

    def __enter__(self) -> "MemoryQuota":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        """归还本次配额。幂等：重复调用无副作用。"""
        if not self._released:
            allocator.release(self.quota_id)
            self._released = True

    def __repr__(self) -> str:
        return f"<MemoryQuota id={self.quota_id} allocated={self.allocated}>"


class MemoryAllocator:
    """进程内内存配额分配器。

    使用全局单例 ``allocator``；不要自己构造第二个实例，否则已分配配额会
    重复计算。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._id_counter = 0
        self._quotas: Dict[int, int] = {}

    @property
    def total_allocated(self) -> int:
        """当前已承诺给任务的内存总量（字节）。"""
        with self._lock:
            return sum(self._quotas.values())

    def reserve_bytes(self) -> int:
        """为系统保留的内存下限（字节）。"""
        value = as_bytes(RESERVE_ENV, None)
        return DEFAULT_RESERVE_BYTES if value is None else value

    def request(
        self,
        required: int = 0,
        reserve: Optional[int] = None,
        *,
        strict: bool = False,
    ) -> MemoryQuota:
        """申请 ``required`` 字节内存配额。

        成功返回 ``MemoryQuota``，失败抛出 ``InsufficientMemoryError``。
        调用方应在任务结束后调用 ``quota.release()`` 或配合 ``with`` 使用。
        """
        res = self.reserve_bytes() if reserve is None else reserve
        need = required + res + self.total_allocated

        if strict:
            avail = probe.available_memory()
            if avail <= 0:
                return self._allocate(required)
            if avail < need:
                raise InsufficientMemoryError(
                    avail, required, res, self.total_allocated
                )
            return self._allocate(required)

        sustained, taken = probe.is_sustained_low(need)
        if sustained:
            raise InsufficientMemoryError(
                probe.available_memory(), required, res, self.total_allocated,
                samples=taken, window=taken * probe.SUSTAINED_INTERVAL,
            )
        return self._allocate(required)

    def _allocate(self, required: int) -> MemoryQuota:
        with self._lock:
            self._id_counter += 1
            qid = self._id_counter
            self._quotas[qid] = required
        return MemoryQuota(qid, required)

    def release(self, quota_id: int) -> int:
        """归还指定 id 的配额，返回实际归还字节数。"""
        with self._lock:
            return self._quotas.pop(quota_id, 0)

    def release_all(self) -> int:
        """归还全部配额（主要用于测试与进程退出清理），返回总字节数。"""
        with self._lock:
            total = sum(self._quotas.values())
            self._quotas.clear()
            return total

    @contextmanager
    def acquire(
        self,
        required: int = 0,
        reserve: Optional[int] = None,
        *,
        strict: bool = False,
    ) -> Generator[MemoryQuota, None, None]:
        """``request`` 的上下文管理器版本，退出时自动释放。"""
        quota = self.request(required, reserve, strict=strict)
        try:
            yield quota
        finally:
            quota.release()


#: 进程内全局内存分配器。
allocator = MemoryAllocator()


#: 模块级 ``acquire`` 便捷函数，等价于 ``allocator.acquire``。
acquire = allocator.acquire


# ---------------------------------------------------------------------------
# 兼容 ``memory_guard`` 旧 API 的顶层函数
# ---------------------------------------------------------------------------

def reserve_bytes() -> int:
    """为系统保留的内存下限（字节）。"""
    return allocator.reserve_bytes()


def check_gpu_headroom(
    required: int = 0,
    reserve: Optional[int] = None,
    *,
    strict: bool = False,
) -> None:
    """确认留出 ``reserve`` 后仍能满足 ``required``；不足则抛异常。"""
    allocator.request(required, reserve, strict=strict).release()


def has_gpu_headroom(
    required: int = 0,
    reserve: Optional[int] = None,
    *,
    strict: bool = False,
) -> bool:
    """``check_gpu_headroom`` 的布尔版本。"""
    try:
        check_gpu_headroom(required, reserve, strict=strict)
        return True
    except InsufficientMemoryError:
        return False


#: 模块级 ``available_memory`` 便捷引用（兼容 ``memory_guard`` 旧 API）。
#: 为了**让测试 monkeypatch ``memory_guard.available_memory`` 能同时影响本模块**，
#: 这里不包成 wrapper，而是直接引用 ``probe.available_memory``；
#: ``memory_guard`` 也引用同一函数对象，因此 patch 一处即可全局生效。
available_memory = probe.available_memory

#: 模块级 ``is_sustained_low`` 便捷引用（兼容 ``memory_guard`` 旧 API）。
#: 与 ``available_memory`` 同理，直接引用 ``probe`` 中的函数对象，
#: 使测试 monkeypatch ``memory_guard.is_sustained_low`` 时全局生效。
is_sustained_low = probe.is_sustained_low

#: 模块级 ``recent_trend`` 便捷引用（兼容 ``memory_guard`` 旧 API）。
recent_trend = probe.recent_trend

#: 模块级 ``reset_cache`` 便捷引用（兼容 ``memory_guard`` 旧 API）。
reset_cache = probe.reset_cache


def describe() -> str:
    """一行摘要，便于日志与 benchmark 输出。"""
    avail = probe.available_memory()
    res = reserve_bytes()
    allocated = allocator.total_allocated
    trend = probe.recent_trend()
    suffix = ""
    if trend:
        suffix = f" / 近 {len(trend)} 次采样最低 {min(trend) / 1024**3:.2f} GB"
    return (
        f"可用 {avail / 1024**3:.2f} GB / 保留 {res / 1024**3:.2f} GB "
        f"/ 已分配 {allocated / 1024**3:.2f} GB "
        f"/ 余量 {max(0, avail - res - allocated) / 1024**3:.2f} GB{suffix}"
    )


def warn_if_low_memory(caller: str = "kakeya_ai") -> None:
    """内存持续低于预留线时发出警告（``ollama_client`` 调用前用）。"""
    try:
        sustained, samples = probe.is_sustained_low(reserve_bytes())
        if sustained:
            warnings.warn(
                f"{caller}: 空闲内存持续低于 2 GB 预留线"
                f"（连续 {samples} 次探测；{describe()}），"
                f"Ollama 加载模型可能触发 swap。",
                stacklevel=3,
            )
    except Exception:  # pragma: no cover
        pass


def release_all() -> int:
    """释放全局分配器中的所有配额（主要用于测试）。"""
    return allocator.release_all()
