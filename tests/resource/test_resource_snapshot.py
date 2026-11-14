"""resource.probe 快照层（内存/CPU 全字段读数）的测试。

fixture 全部取自真实机器输出格式：``vm_stat`` / ``sysctl vm.swapusage``
来自 M4 macOS 实测，``/proc/meminfo`` / ``/proc/stat`` 取标准 Linux 格式。
解析器是纯函数，喂 fixture 即可验证，不触碰真实系统；只有冒烟用例走真机。
"""

from unittest.mock import mock_open

import pytest

from kakeya_ai.resource import probe
from kakeya_ai.resource.probe import (
    CpuSnapshot,
    MemorySnapshot,
    cpu_snapshot,
    memory_snapshot,
    snapshot,
)

#: M4 macOS 实测风格的 vm_stat 输出。刻意保留三类格式细节：
#: 千分位逗号（``72,421.``）、行尾句点、compressor 键带引号。
VM_STAT_SAMPLE = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              72,421.
Pages active:                           352,220.
Pages inactive:                         297,836.
Pages speculative:                       75,411.
Pages throttled:                              0.
Pages wired down:                       131,072.
Pages purgeable:                         12,345.
"Pages occupied by compressor":          40,000.
Pages stored in compressor:             120,000.
File-backed pages:                      200,000.
Anonymous pages:                        300,000.
Pageins:                                500,000.
Pageouts:                                   100.
Swapins:                                     50.
Swapouts:                                   200.
"""

#: ``sysctl vm.swapusage`` 实测格式（macOS 14/15）。
SWAPUSAGE_SAMPLE = (
    "total = 6144.00M  used = 4502.50M  free = 1641.50M  (encrypted)\n"
)

#: 标准 Linux /proc/meminfo 片段。
MEMINFO_SAMPLE = """\
MemTotal:       16000000 kB
MemFree:         1000000 kB
MemAvailable:    8000000 kB
Active:          4000000 kB
Inactive:        3000000 kB
Cached:          2000000 kB
SwapTotal:       4000000 kB
SwapFree:        3000000 kB
SReclaimable:     500000 kB
HugePages_Total:       0
"""

#: 标准 Linux /proc/stat 片段：首行是全部核汇总（必须被忽略）。
PROC_STAT_SAMPLE = """\
cpu  1000 0 300 7000 50 0 25 0 0 0
cpu0  500 0 150 3500 25 0 12 0 0 0
cpu1  500 0 150 3500 25 0 13 0 0 0
intr 123456
"""


def _macos_probe(monkeypatch):
    """把 _run / _sysctl_int 接到 macOS fixture 上，返回 memory_snapshot() 结果。"""
    def fake_run(cmd, timeout):
        if cmd[:1] == ["vm_stat"]:
            return VM_STAT_SAMPLE
        if cmd[:2] == ["sysctl", "vm.swapusage"]:
            return SWAPUSAGE_SAMPLE
        return None

    monkeypatch.setattr(probe, "_run", fake_run)
    monkeypatch.setattr(
        probe,
        "_sysctl_int",
        lambda name: 16 * 1024**3 if name == "hw.memsize" else None,
    )
    return memory_snapshot()


class TestParseVmStat:
    def test_real_sample(self):
        page_size, stats = probe._parse_vm_stat(VM_STAT_SAMPLE)
        assert page_size == 16384
        assert stats["Pages free"] == 72421
        assert stats["Pages active"] == 352220
        # 引号键必须被归一化，否则跨 macOS 版本读不到压缩机页。
        assert stats["Pages occupied by compressor"] == 40000
        assert stats["Pages wired down"] == 131072

    def test_default_page_size_when_missing(self):
        page_size, stats = probe._parse_vm_stat("Pages free: 100.\n")
        assert page_size == 4096
        assert stats == {"Pages free": 100}

    def test_ignores_lines_without_digits(self):
        _, stats = probe._parse_vm_stat("Mach Virtual Memory Statistics:\n")
        assert stats == {}


class TestParseSwapusage:
    def test_megabyte_units(self):
        total, used = probe._parse_swapusage(SWAPUSAGE_SAMPLE)
        assert total == 6144 * 1024**2
        assert used == int(4502.50 * 1024**2)

    def test_plain_bytes_without_unit(self):
        total, used = probe._parse_swapusage("total = 1073741824 used = 0\n")
        assert total == 1073741824
        assert used == 0

    def test_missing_fields_are_zero(self):
        assert probe._parse_swapusage("total = 1024.00M\n") == (1024 * 1024**2, 0)
        assert probe._parse_swapusage("") == (0, 0)


class TestParseMeminfo:
    def test_kilobyte_conversion(self):
        fields = probe._parse_meminfo(MEMINFO_SAMPLE)
        assert fields["MemAvailable"] == 8000000 * 1024
        assert fields["Cached"] == 2000000 * 1024
        # 无 kB 单位的行原样取数（HugePages_Total 是个数，不是字节）。
        assert fields["HugePages_Total"] == 0

    def test_garbage_lines_skipped(self):
        fields = probe._parse_meminfo("MemFree: abc kB\n:::broken\nMemFree: 1 kB\n")
        assert fields["MemFree"] == 1024


class TestParseProcStat:
    def test_aggregate_line_excluded(self):
        pairs = probe._parse_proc_stat(PROC_STAT_SAMPLE)
        assert len(pairs) == 2
        # core0：busy = total(4187) - idle(3500) - iowait(25) = 662。
        assert pairs[0] == (662, 4187)

    def test_empty_on_no_cpu_lines(self):
        assert probe._parse_proc_stat("intr 1\nctxt 2\n") == []


class TestPerCoreUtilization:
    def test_half_busy_core(self):
        # before: busy 100/total 200；after: busy 200/total 400 → Δbusy 100 / Δtotal 200。
        utils = probe._per_core_utilization([(100, 200)], [(200, 400)])
        assert utils == [0.5]

    def test_tick_regression_marks_zero(self):
        # 计数器归零/热插拔导致 tick 倒退：该核记 0，不编造。
        utils = probe._per_core_utilization([(500, 1000)], [(10, 600)])
        assert utils == [0.0]

    def test_length_mismatch_truncates(self):
        utils = probe._per_core_utilization([(1, 2), (1, 2)], [(2, 4)])
        assert len(utils) == 1

    def test_idle_core_is_zero(self):
        utils = probe._per_core_utilization([(100, 200)], [(100, 300)])
        assert utils == [0.0]


class TestSchedulableCores:
    def test_no_utilization_falls_back_to_topology(self):
        snap = CpuSnapshot(cores=10, per_core_utilization=())
        assert snap.schedulable_cores() == 10
        assert snap.schedulable_cores(reserve=2) == 8

    def test_max_util_filters_busy_cores(self):
        snap = CpuSnapshot(cores=4, per_core_utilization=(0.1, 0.9, 0.2, 1.0))
        assert snap.schedulable_cores(max_util=0.5) == 2

    def test_result_never_below_one(self):
        snap = CpuSnapshot(cores=4, per_core_utilization=(1.0, 1.0, 1.0, 1.0))
        assert snap.schedulable_cores(max_util=0.5) == 1
        assert CpuSnapshot(cores=0).schedulable_cores() == 1

    def test_reserve_capped_by_budget(self):
        snap = CpuSnapshot(cores=4, per_core_utilization=(0.0, 0.0, 0.0, 0.0))
        assert snap.schedulable_cores(max_util=0.5, reserve=10) == 1


class TestMemorySnapshot:
    def test_pressure_math(self):
        snap = MemorySnapshot(
            source="test", page_size=4096, total=16 * 1024**3,
            available=0, free=0, active=4 * 1024**3, inactive=0,
            speculative=0, wired=2 * 1024**3, compressed=2 * 1024**3,
            swap_total=0, swap_used=0, cached=0,
        )
        assert snap.pressure == 0.5

    def test_pressure_zero_when_total_unknown(self):
        snap = MemorySnapshot(
            source="test", page_size=0, total=0, available=0, free=0,
            active=1, inactive=0, speculative=0, wired=0, compressed=0,
            swap_total=0, swap_used=0, cached=0,
        )
        assert snap.pressure == 0.0


class TestMacosSnapshotWiring:
    def test_fields_from_fixture(self, monkeypatch):
        snap = _macos_probe(monkeypatch)
        page = 16384
        assert snap.source == "darwin"
        assert snap.page_size == page
        assert snap.total == 16 * 1024**3
        assert snap.free == 72421 * page
        assert snap.active == 352220 * page
        assert snap.inactive == 297836 * page
        assert snap.speculative == 75411 * page
        assert snap.wired == 131072 * page
        assert snap.compressed == 40000 * page
        assert snap.cached == 0  # macOS 无此项，如实为 0
        assert snap.swap_total == 6144 * 1024**2
        assert snap.swap_used == int(4502.50 * 1024**2)
        # 可用口径与标量读数一致：free + inactive + speculative。
        assert snap.available == (72421 + 297836 + 75411) * page

    def test_scalar_matches_snapshot_semantics(self, monkeypatch):
        _macos_probe(monkeypatch)
        assert probe._available_memory_macos() == memory_snapshot().available

    def test_degenerate_inputs_yield_honest_zeros(self, monkeypatch):
        monkeypatch.setattr(probe, "_run", lambda cmd, timeout: None)
        monkeypatch.setattr(probe, "_sysctl_int", lambda name: None)
        snap = memory_snapshot()
        assert snap.source == "darwin"
        assert snap.total == 0 and snap.swap_total == 0 and snap.compressed == 0
        # vm_stat 读不到时退化到 sysconf 口径，可用内存不应是负数/异常值。
        assert snap.available >= 0


class TestLinuxSnapshotWiring:
    def test_memory_fields(self, monkeypatch):
        monkeypatch.setattr("builtins.open", mock_open(read_data=MEMINFO_SAMPLE))
        monkeypatch.setattr(probe, "_page_size", lambda: 4096)
        snap = probe._memory_snapshot_linux()
        assert snap.source == "linux"
        assert snap.total == 16000000 * 1024
        assert snap.available == 8000000 * 1024
        assert snap.cached == (2000000 + 500000) * 1024
        assert snap.swap_used == (4000000 - 3000000) * 1024
        # vm_stat 专有的分页类别在 Linux 如实为 0。
        assert snap.wired == 0 and snap.compressed == 0 and snap.speculative == 0

    def test_cpu_diff(self, monkeypatch):
        before = [(100, 200), (150, 300)]
        after = [(150, 400), (150, 450)]
        # 两次采样各返回一份：side_effect 迭代器按调用顺序给出。
        samples = iter([before, after])
        monkeypatch.setattr(probe, "_read_proc_stat_pairs", lambda: next(samples))
        snap = probe._cpu_snapshot_linux(sample_interval=0.0)
        assert snap.cores == 2
        assert snap.per_core_utilization == (0.25, 0.0)
        assert len(snap.loadavg) == 3

    def test_dispatch_to_linux_builder(self, monkeypatch):
        monkeypatch.setattr(probe.sys, "platform", "linux")
        monkeypatch.setattr("builtins.open", mock_open(read_data=MEMINFO_SAMPLE))
        monkeypatch.setattr(probe, "_page_size", lambda: 4096)
        snap = memory_snapshot()
        assert snap.source == "linux"
        assert snap.available == 8000000 * 1024

    def test_meminfo_unreadable_falls_back(self, monkeypatch):
        # meminfo 打不开时字段全 0，可用内存退化到 sysconf 口径而非抛异常。
        def broken_open(*args, **kwargs):
            raise OSError("no /proc")

        monkeypatch.setattr(probe.sys, "platform", "linux")
        monkeypatch.setattr("builtins.open", broken_open)
        snap = memory_snapshot()
        assert snap.source == "linux"
        assert snap.total == 0
        assert snap.available >= 0


class TestSmoke:
    def test_snapshot_on_real_machine(self):
        # 真机冒烟：走真实 sysctl/系统调用，只验证形状与一致性，不验证具体数值。
        sys_snap = snapshot()
        assert isinstance(sys_snap.memory, MemorySnapshot)
        assert isinstance(sys_snap.cpu, CpuSnapshot)
        assert sys_snap.cpu.cores >= 1
        assert len(sys_snap.cpu.loadavg) == 3
        assert sys_snap.memory.available >= 0

    def test_cpu_snapshot_shape(self):
        snap = cpu_snapshot()
        assert snap.cores >= 1
        assert len(snap.per_core_utilization) in (0, snap.cores)
