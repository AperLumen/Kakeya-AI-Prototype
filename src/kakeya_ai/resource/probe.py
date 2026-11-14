"""内存与系统状态探测：按平台读数，供守卫判定与调度观测。

本模块只做"读数"，不做判定。所有平台相关代码集中在这里，便于测试隔离。

两类读数
--------

* **标量读数**（守卫热路径）：:func:`available_memory` / :func:`is_sustained_low`
  —— 只取"可用内存"一个数，口径最小、调用最频。
* **全字段快照**（观测/调度路径）：:func:`memory_snapshot` /
  :func:`cpu_snapshot` / :func:`snapshot` —— 一次返回内存各分页类别、swap、
  CPU 拓扑、负载均值与每核利用率。代价高于标量读数（macOS 侧每核利用率需
  两次 ``host_processor_info`` 采样差分，默认间隔 0.05 s），**不要**把它放进
  逐对计算的内层循环；调度、诊断、benchmark 归因用。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

#: 判定"持续低内存"所需的最少连续采样次数。
SUSTAINED_SAMPLES = 4

#: 连续采样之间的间隔（秒）。4 × 0.25 s ≈ 0.75 s 的确认窗口——
#: 足够滤掉页回收造成的瞬时低谷，又不至于让调用方干等。
SUSTAINED_INTERVAL = 0.25

#: 已确认的判定结果缓存时长（秒）。内存压力不会在毫秒级消失，
#: 缓存可避免每次调用都付一遍确认窗口的开销（级联调用下很关键）。
VERDICT_TTL = 5.0

#: ``vm_stat`` 子进程的探活超时（秒）。
#:
#: **不是资源预算项**：``vm_stat`` 是本地命令，正常在毫秒级返回；这个值只用于
#: 防止命令挂死。预算管的是"任务能花多久"，不是"本地命令多快返回"——
#: 与 ``ollama_client.PROBE_TIMEOUT`` 同一性质（探活用固定小值）。
VM_STAT_TIMEOUT = 5.0

#: (时间戳, 可用字节) 采样历史，供诊断与 ``recent_trend()`` 使用。
_history: list[tuple[float, int]] = []

#: (计算时刻, 判定时的阈值, 是否持续低于阈值) 的判定缓存。
#: 必须连阈值一起缓存——否则先按 2 GB 判定"通过"后，10 GB 的请求会错误复用它。
_verdict: Optional[tuple[float, int, bool]] = None


def _run(cmd: list, timeout: float) -> Optional[str]:
    """统一子进程执行：成功返回 stdout 文本；任何失败返回 None。

    探测铁律：宁可读不到，不许抛。所有外部命令（``vm_stat`` / ``sysctl``）
    都经这里，调用方只看返回值是否为 None。
    """
    try:
        return subprocess.run(
            list(cmd), capture_output=True, text=True, check=True, timeout=timeout,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _sysctl_int(name: str) -> Optional[int]:
    """读一个整数 sysctl；失败返回 None。"""
    out = _run(["sysctl", "-n", name], VM_STAT_TIMEOUT)
    if out is None:
        return None
    try:
        return int(out.strip())
    except ValueError:
        return None


def _page_size() -> int:
    """系统页大小（字节）；读不到退化为 4096。"""
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return 4096


def _loadavg() -> Tuple[float, float, float]:
    """负载均值 (1/5/15 分钟)；取不到时三元组全 0。"""
    try:
        a, b, c = os.getloadavg()
        return (float(a), float(b), float(c))
    except (OSError, AttributeError):
        return (0.0, 0.0, 0.0)


def _parse_meminfo(text: str) -> Dict[str, int]:
    """解析 ``/proc/meminfo``：返回 ``{字段名: 字节数}``。纯函数，便于测试。

    形如 ``MemAvailable:   12345678 kB`` —— ``kB`` 单位换算成字节，无单位行
    原样取数；解析不了的行直接跳过。
    """
    fields: Dict[str, int] = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if not sep:
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        fields[key.strip()] = value * 1024 if len(parts) > 1 and parts[1] == "kB" else value
    return fields


def _parse_vm_stat(text: str) -> Tuple[int, Dict[str, int]]:
    """解析 ``vm_stat`` 输出：返回 ``(page_size, {页类名: 页数})``。纯函数。

    行形如 ``Pages free:   12345.`` —— 数字带千分位逗号、行尾有句点；
    页面大小单独出现在 ``page size of 16384`` 行。只保留纯数字部分，
    避免逗号与句点污染。
    """
    page_size = 4096
    stats: Dict[str, int] = {}
    for line in text.splitlines():
        if "page size of" in line:
            try:
                page_size = int(line.split("page size of")[1].split()[0])
            except (IndexError, ValueError):
                pass
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        digits = "".join(ch for ch in val if ch.isdigit())
        if digits:
            # 部分 macOS 版本给 compressor 行加引号（"Pages occupied by
            # compressor":），归一化掉，保证跨版本键名一致。
            stats[key.strip().strip('"')] = int(digits)
    return page_size, stats


def _parse_swapusage(text: str) -> Tuple[int, int]:
    """解析 ``sysctl vm.swapusage``：返回 ``(total, used)``（字节）。纯函数。

    行形如 ``total = 6144.00M  used = 4502.50M  free = 1641.50M  (encrypted)``。
    """
    factors = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
    tokens = text.replace("=", " ").split()

    def _field(name: str) -> int:
        for i, tok in enumerate(tokens):
            if tok == name and i + 1 < len(tokens):
                raw = tokens[i + 1]
                suffix = raw[-1].upper()
                if suffix in factors:
                    return int(float(raw[:-1]) * factors[suffix])
                try:
                    return int(float(raw))
                except ValueError:
                    return 0
        return 0

    return _field("total"), _field("used")


def _parse_proc_stat(text: str) -> List[Tuple[int, int]]:
    """解析 ``/proc/stat`` 每核行：返回 ``[(busy, total)]`` tick 对。纯函数。

    形如 ``cpu3  1024 0 300 9000 22 0 12 0 0 0``；busy = 除 idle/iowait 外
    全部字段之和；total = 全部字段之和。
    """
    pairs: List[Tuple[int, int]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[0].startswith("cpu") or not parts[0][3:].isdigit():
            continue
        try:
            ticks = [int(x) for x in parts[1:]]
        except ValueError:
            continue
        idle = ticks[3] + (ticks[4] if len(ticks) > 4 else 0)
        pairs.append((sum(ticks) - idle, sum(ticks)))
    return pairs


def _host_processor_ticks() -> Optional[List[Tuple[int, int]]]:
    """macOS：``host_processor_info`` 读每核 ``(busy, total)`` tick；失败返回 None。

    返回结构与 :func:`_parse_proc_stat` 对齐，供差分统一处理。Mach 分配的缓冲
    用 ``vm_deallocate`` 归还，避免泄漏。仅 Darwin 可用。
    """
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        u32 = ctypes.c_uint
        lib.mach_host_self.restype = u32
        lib.mach_task_self.restype = u32
        lib.host_processor_info.restype = ctypes.c_int
        lib.host_processor_info.argtypes = [
            u32,
            u32,
            ctypes.POINTER(u32),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
            ctypes.POINTER(u32),
        ]
        lib.vm_deallocate.restype = ctypes.c_int
        lib.vm_deallocate.argtypes = [u32, ctypes.c_void_p, ctypes.c_size_t]

        count = u32(0)
        info = ctypes.POINTER(ctypes.c_int)()
        info_count = u32(0)
        # flavor 2 = PROCESSOR_LOAD_INFO；每核 4 个 int：user/system/idle/nice
        kr = lib.host_processor_info(
            lib.mach_host_self(),
            2,
            ctypes.byref(count),
            ctypes.byref(info),
            ctypes.byref(info_count),
        )
        if kr != 0 or count.value == 0 or not info:
            return None
        pairs: List[Tuple[int, int]] = []
        for i in range(count.value):
            user, system, idle, nice = (int(info[i * 4 + j]) for j in range(4))
            total = user + system + idle + nice
            pairs.append((total - idle, total))
        lib.vm_deallocate(
            lib.mach_task_self(),
            ctypes.cast(info, ctypes.c_void_p),
            ctypes.c_size_t(info_count.value * ctypes.sizeof(ctypes.c_int)),
        )
        return pairs
    except (OSError, AttributeError, ValueError):
        return None


def _per_core_utilization(
    before: List[Tuple[int, int]], after: List[Tuple[int, int]]
) -> List[float]:
    """两组 tick 采样差分得出每核利用率（0..1）。

    tick 倒退或总 tick 不增时该核记 0；两次采样核数不一致（间隔内热插拔）
    时按较短者截断 —— 宁可缺失，不编造。
    """
    utils: List[float] = []
    for (b_busy, b_total), (a_busy, a_total) in zip(before, after, strict=False):
        d_total = a_total - b_total
        d_busy = a_busy - b_busy
        if d_total <= 0 or d_busy < 0:
            utils.append(0.0)
        else:
            utils.append(min(1.0, d_busy / d_total))
    return utils


@dataclass(frozen=True)
class MemorySnapshot:
    """内存全字段快照（字节）。

    Linux 无 speculative / wired / compressed 的分账（``vm_stat`` 专有），
    macOS 无 ``cached`` —— 这些字段在对应平台如实为 0：宁可读不到，不编造。
    """

    source: str  # 读数来源：linux / darwin / fallback
    page_size: int  # 系统页大小（字节）
    total: int  # 物理内存总量
    available: int  # 可用内存（内核估计口径）
    free: int  # 完全空闲
    active: int  # 活跃页
    inactive: int  # 非活跃页（可回收）
    speculative: int  # 投机页（可回收，macOS 专有）
    wired: int  # 内核常驻（不可回收，macOS 专有）
    compressed: int  # 压缩内存占用（macOS 专有）
    swap_total: int  # swap 总量
    swap_used: int  # swap 已用
    cached: int  # page cache（Linux 专有）

    @property
    def pressure(self) -> float:
        """内存压力 0..1：``(active + wired + compressed) / total``；总量未知时为 0。"""
        if self.total <= 0:
            return 0.0
        return (self.active + self.wired + self.compressed) / self.total


@dataclass(frozen=True)
class CpuSnapshot:
    """CPU 全字段快照。

    ``performance_cores`` / ``efficiency_cores`` 为 P/E 拓扑（Linux 的
    sysctl 无此项，如实为 0）；``per_core_utilization`` 为两次采样差分的
    每核利用率（读不到 tick 时为空元组）；``loadavg`` 为 1/5/15 分钟负载均值。
    """

    cores: int  # 逻辑核总数
    performance_cores: int = 0  # 性能核数（P 核）
    efficiency_cores: int = 0  # 能效核数（E 核）
    per_core_utilization: Tuple[float, ...] = ()
    loadavg: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def schedulable_cores(self, max_util: float = 1.0, reserve: int = 0) -> int:
        """可调度核数（至少 1）。

        ``max_util``：利用率不超过该值的核才算空闲可调度；默认 1.0 即只看
        拓扑，不看负载。``reserve``：额外保底余量（给系统/前台留核）。
        读不到每核利用率时退化为 ``max(1, cores - reserve)``。
        """
        budget = max(1, self.cores - max(0, reserve))
        if not self.per_core_utilization or max_util >= 1.0:
            return budget
        idle = sum(1 for u in self.per_core_utilization if u <= max_util)
        return max(1, min(budget, idle))


@dataclass(frozen=True)
class SystemSnapshot:
    """系统全字段快照：内存 + CPU 两件合装。"""

    memory: MemorySnapshot
    cpu: CpuSnapshot


def _memory_snapshot_linux() -> MemorySnapshot:
    """Linux：``/proc/meminfo`` 组装全字段快照。

    swap 已用 = SwapTotal - SwapFree；cached 取 ``Cached + SReclaimable``
    （可回收 page cache 的完整口径）；``vm_stat`` 专有的分页类别如实为 0。
    """
    try:
        with open("/proc/meminfo", "r", encoding="ascii", errors="replace") as fh:
            fields = _parse_meminfo(fh.read())
    except OSError:
        fields = {}
    swap_total = fields.get("SwapTotal", 0)
    return MemorySnapshot(
        source="linux",
        page_size=_page_size(),
        total=fields.get("MemTotal", 0),
        available=fields.get("MemAvailable", 0),
        free=fields.get("MemFree", 0),
        active=fields.get("Active", 0),
        inactive=fields.get("Inactive", 0),
        speculative=0,
        wired=0,
        compressed=0,
        swap_total=swap_total,
        swap_used=swap_total - fields.get("SwapFree", 0),
        cached=fields.get("Cached", 0) + fields.get("SReclaimable", 0),
    )


def _memory_snapshot_macos() -> MemorySnapshot:
    """macOS：``vm_stat`` + ``sysctl vm.swapusage`` 组装全字段快照。"""
    out = _run(["vm_stat"], VM_STAT_TIMEOUT)
    page_size, stats = (4096, {}) if out is None else _parse_vm_stat(out)
    swap_out = _run(["sysctl", "vm.swapusage"], VM_STAT_TIMEOUT)
    swap_total, swap_used = (0, 0) if swap_out is None else _parse_swapusage(swap_out)

    def _pages(name: str) -> int:
        return stats.get(name, 0) * page_size

    return MemorySnapshot(
        source="darwin",
        page_size=page_size,
        total=_sysctl_int("hw.memsize") or 0,
        available=_pages("Pages free")
        + _pages("Pages inactive")
        + _pages("Pages speculative"),
        free=_pages("Pages free"),
        active=_pages("Pages active"),
        inactive=_pages("Pages inactive"),
        speculative=_pages("Pages speculative"),
        wired=_pages("Pages wired down"),
        compressed=_pages("Pages occupied by compressor"),
        swap_total=swap_total,
        swap_used=swap_used,
        cached=0,
    )


def _memory_snapshot_fallback() -> MemorySnapshot:
    """未知平台：总量/可用走 sysconf，其余字段如实为 0。"""
    try:
        total = int(os.sysconf("SC_PHYS_PAGES")) * _page_size()
    except (ValueError, OSError, AttributeError):
        total = 0
    return MemorySnapshot(
        source="fallback",
        page_size=_page_size(),
        total=total,
        available=_fallback_available(),
        free=0,
        active=0,
        inactive=0,
        speculative=0,
        wired=0,
        compressed=0,
        swap_total=0,
        swap_used=0,
        cached=0,
    )


def memory_snapshot() -> MemorySnapshot:
    """内存全字段快照。读不到的字段如实为 0；可用读不到时退化到 sysconf。"""
    if sys.platform.startswith("linux"):
        snap = _memory_snapshot_linux()
    elif sys.platform == "darwin":
        snap = _memory_snapshot_macos()
    else:
        snap = _memory_snapshot_fallback()
    if snap.available <= 0:
        fallback = _fallback_available()
        if fallback > 0:
            snap = replace(snap, available=fallback)
    return snap


def _read_proc_stat_pairs() -> Optional[List[Tuple[int, int]]]:
    """读 ``/proc/stat`` 每核 ``(busy, total)`` tick；失败返回 None。"""
    try:
        with open("/proc/stat", "r", encoding="ascii", errors="replace") as fh:
            return _parse_proc_stat(fh.read())
    except OSError:
        return None


def _cpu_snapshot_linux(sample_interval: float) -> CpuSnapshot:
    """Linux：``/proc/stat`` 两次采样差分；核数取采样到的 ``cpuN`` 行数。"""
    before = _read_proc_stat_pairs()
    pairs: Tuple[float, ...] = ()
    cores = len(before) if before else 0
    if before:
        time.sleep(max(0.0, sample_interval))
        after = _read_proc_stat_pairs()
        if after is not None:
            pairs = tuple(_per_core_utilization(before, after))
    if cores <= 0:
        try:
            cores = int(os.sysconf("SC_NPROCESSORS_ONLN"))
        except (ValueError, OSError, AttributeError):
            cores = 0
    return CpuSnapshot(
        cores=cores,
        performance_cores=0,
        efficiency_cores=0,
        per_core_utilization=pairs,
        loadavg=_loadavg(),
    )


def _cpu_topology_macos() -> Tuple[int, int, int]:
    """macOS P/E 拓扑：``(性能核数, 能效核数, 总核数)``；读不到记 0。"""
    perf = _sysctl_int("hw.perflevel0.logicalcpu")
    eff = _sysctl_int("hw.perflevel1.logicalcpu")
    total = _sysctl_int("hw.ncpu")
    return (perf or 0, eff or 0, total or 0)


def _cpu_snapshot_macos(sample_interval: float) -> CpuSnapshot:
    """macOS：sysctl 拓扑 + ``host_processor_info`` 两次采样差分。"""
    perf, eff, total = _cpu_topology_macos()
    before = _host_processor_ticks()
    pairs: Tuple[float, ...] = ()
    if before is not None:
        time.sleep(max(0.0, sample_interval))
        after = _host_processor_ticks()
        if after is not None:
            pairs = tuple(_per_core_utilization(before, after))
    return CpuSnapshot(
        cores=total or len(before or []),
        performance_cores=perf,
        efficiency_cores=eff,
        per_core_utilization=pairs,
        loadavg=_loadavg(),
    )


def _cpu_snapshot_generic() -> CpuSnapshot:
    """未知平台：只有核数与负载，拓扑与利用率如实留空/0。"""
    try:
        cores = int(os.sysconf("SC_NPROCESSORS_ONLN"))
    except (ValueError, OSError, AttributeError):
        cores = 0
    return CpuSnapshot(cores=cores, loadavg=_loadavg())


def cpu_snapshot(sample_interval: float = 0.05) -> CpuSnapshot:
    """CPU 拓扑 + 负载 + 每核利用率快照。

    每核利用率需两次采样差分，间隔 ``sample_interval``（默认 0.05 s）；
    读不到 tick 时 ``per_core_utilization`` 为空元组，其余字段照常。
    """
    if sys.platform.startswith("linux"):
        return _cpu_snapshot_linux(sample_interval)
    if sys.platform == "darwin":
        return _cpu_snapshot_macos(sample_interval)
    return _cpu_snapshot_generic()


def snapshot(sample_interval: float = 0.05) -> SystemSnapshot:
    """内存 + CPU 一次性全字段快照。调度观测与 benchmark 归因的入口。"""
    return SystemSnapshot(memory=memory_snapshot(), cpu=cpu_snapshot(sample_interval))


def _available_memory_linux() -> int:
    """Linux：``MemAvailable``（字节）。读不到返回 0。

    为什么用 ``MemAvailable`` 而不是 ``MemFree``：``MemFree`` 只是"完全空闲
    的页"，**不含** page cache 与 reclaimable slab —— 而后者在压力下可被内核
    立即回收。用 ``MemFree`` 会严重低估可用内存（典型偏差数倍），导致
    守卫在内存充足的机器上误拒 GPU 任务。

    ``MemAvailable`` 是内核自 3.14 起提供、专门为此设计的估计量，
    语义与 macOS 侧的 ``free + inactive + speculative`` 对齐。
    """
    try:
        with open("/proc/meminfo", "r", encoding="ascii", errors="replace") as fh:
            fields = _parse_meminfo(fh.read())
    except OSError:
        return 0
    return fields.get("MemAvailable", 0)


def _available_memory_macos() -> int:
    """macOS：``vm_stat`` 的 free + inactive + speculative（字节）。读不到返回 0。"""
    out = _run(["vm_stat"], VM_STAT_TIMEOUT)
    if out is None:
        return 0
    page_size, stats = _parse_vm_stat(out)
    total_pages = sum(
        stats.get(k, 0)
        for k in ("Pages free", "Pages inactive", "Pages speculative")
    )
    if total_pages <= 0:
        return 0
    return total_pages * page_size


def _fallback_available() -> int:
    """无专用实现或专用实现失败时的退化估计（字节）。"""
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(size)
    except (ValueError, OSError, AttributeError):
        return 0


def _read_available_memory() -> int:
    """按平台分派读取可用内存（字节）。"""
    if sys.platform.startswith("linux"):
        value = _available_memory_linux()
        if value > 0:
            return value
        return _fallback_available()

    if sys.platform == "darwin":
        value = _available_memory_macos()
        if value > 0:
            return value
        return _fallback_available()

    return _fallback_available()


def available_memory() -> int:
    """当前可用内存（字节），**瞬时值**。"""
    return _read_available_memory()


def reset_cache() -> None:
    """清空采样历史与判定缓存。"""
    global _verdict
    _history.clear()
    _verdict = None


def recent_trend() -> list[int]:
    """返回最近的采样序列（字节）。"""
    return [v for _t, v in _history]


def is_sustained_low(
    threshold: int,
    samples: Optional[int] = None,
    interval: Optional[float] = None,
    use_cache: bool = True,
) -> tuple[bool, int]:
    """连续探测，判断可用内存是否**持续**低于 ``threshold``。

    单次低于不算数——页回收造成的瞬时低谷很常见，稍纵即逝。只有连续
    ``samples`` 次都低于阈值才判为真。探测不到内存（读数为 0）时返回
    ``(False, n)``——宁可放行，也不因探测失败误杀任务。
    """
    global _verdict
    if samples is None:
        samples = SUSTAINED_SAMPLES
    if interval is None:
        interval = SUSTAINED_INTERVAL
    now = time.monotonic()

    if use_cache and _verdict is not None:
        ts, cached_threshold, sustained = _verdict
        if now - ts < VERDICT_TTL and cached_threshold == threshold:
            return sustained, 0

    taken = 0
    below_all = True
    for i in range(max(1, samples)):
        last = available_memory()
        _history.append((now + i * interval, last))
        if len(_history) > 64:
            del _history[: len(_history) - 64]
        taken += 1
        if last <= 0:
            _verdict = (now, threshold, False)
            return False, taken
        if last >= threshold:
            below_all = False
            break
        if i < samples - 1:
            time.sleep(interval)

    _verdict = (now, threshold, below_all)
    return below_all, taken
