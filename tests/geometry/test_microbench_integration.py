"""守门：`nano/isolated/bench_micro` 与 `chip_model` 必须**说同一套话**（2026-10-07）。

**为什么需要它**：这是"纳秒内核项目 × 芯片模型"的**合流点**。两处各自维护了
同一个物理量（每 pair 的 FLOP 数 = 25；M4 的 P 核峰值 = 35.2 GFLOP/s）。
**若两处漂移**，一个报告说"离峰值 70%"、另一个说"离峰值 40%"，做设计决策时
看哪个都会错——这正是本项目多次踩过的"同一件事两处各记一个数"。

**本文件守三件事**：

1. **共享常量一致**：C++ 微基准里的 ``FLOPS_PER_PAIR`` / ``p_core_gflops``
   必须与 ``chip_model.py`` 的对应常量**数值相同**（用正则从源码抽出比对，
   不靠运行——因为它们是编译期常量，运行也看不到）。同理，E 核**分派权重**
   在 ``bench_micro.cpp`` 与 ``cpp/kernel/parallel.hpp`` 两处必须一致（2026-10-30
   异构分派新增；它是 A/B 标定的 0.60，**不是** scheduler 的 0.44 调度效率）。
2. **微基准能编译并给出物理合理的结果**（本机有 C++ 工具链时）：串行 ns/pair
   为正、落在一个宽泛但可辩护的区间（0.2~5 ns/pair），且并行计数与串行**逐位
   相同**。
3. **模型与微基准的单核口径一致**：由模型的 ``M4_P_CORE_GFLOPS`` 推出的单核
   上限，必须与微基准**同一公式**（``FLOPS_PER_PAIR / P核峰值``）——把"两处
   用不同公式"这种隐蔽漂移挡在门外。

无 C++ 工具链或无法编译时自动 skip（不把"没工具链"误判成"代码错"）。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from nano.model.chip_model import (
    M4_P_CORE_GFLOPS,
    TUBE_GEOMETRY_KERNEL,
    roofline_ceiling_ns_per_pair,
)

REPO = Path(__file__).resolve().parents[2]
MICRO_SRC = REPO / "nano" / "isolated" / "bench_micro.cpp"
PARALLEL_SRC = REPO / "cpp" / "kernel" / "parallel.hpp"
CHIP_SRC = REPO / "nano" / "chip_model.py"


def _micro_text() -> str:
    return MICRO_SRC.read_text(encoding="utf-8")


def _extract_const(text: str, name: str) -> float:
    """从 C++ 源码抽出 ``constexpr <type> NAME = <number>;`` 的数值。"""
    m = re.search(rf"{name}\s*=\s*([0-9.]+)", text)
    assert m is not None, f"在源码里找不到常量 {name} —— 名字变了就改本测试"
    return float(m.group(1))


class TestSharedConstantsAgree:
    """两处对"同一物理量"的常量必须相同——这是交叉核对的核心。"""

    def test_flops_per_pair_matches(self):
        """微基准的 ``FLOPS_PER_PAIR`` 必须等于 chip_model 的 ``flops_per_pair``。

        两处都写 25（逐行数自 nano_kernel.cpp）。若一处改了另一处没改，
        roofline 对比就会用错分子。
        """
        micro_val = _extract_const(_micro_text(), "FLOPS_PER_PAIR")
        assert micro_val == TUBE_GEOMETRY_KERNEL.flops_per_pair, (
            f"bench_micro.cpp 的 FLOPS_PER_PAIR={micro_val} 与 chip_model 的 "
            f"{TUBE_GEOMETRY_KERNEL.flops_per_pair} 不一致 —— 两处必须同源"
        )

    def test_p_core_gflops_matches(self):
        """微基准的 ``p_core_gflops`` 必须等于 chip_model 的 ``M4_P_CORE_GFLOPS``。"""
        micro_val = _extract_const(_micro_text(), "p_core_gflops")
        assert micro_val == M4_P_CORE_GFLOPS, (
            f"bench_micro.cpp 的 p_core_gflops={micro_val} 与 chip_model 的 "
            f"M4_P_CORE_GFLOPS={M4_P_CORE_GFLOPS} 不一致 —— 两处必须同源"
        )

    def test_micro_references_chip_model(self):
        """微基准的注释里必须点明它与 chip_model 的关系（可追溯）。"""
        text = _micro_text()
        assert "chip_model" in text, (
            "bench_micro.cpp 应说明它与 nano.chip_model 的关系，"
            "否则后人不知道两处数字为何必须一致"
        )

    def test_e_core_partition_weight_matches(self):
        """E 核**分派权重**两处同源：bench_micro 的 ``E_CORE_WEIGHT`` 必须与
        ``cpp/kernel/parallel.hpp`` 的 ``e_core_partition_weight()`` 返回值相同。

        **口径注意**：它是 A/B 实测标定的**分派权重**（0.60），**不是**
        ``scheduler.DEFAULT_E_CORE_EFFICIENCY``（0.44，predict_schedule
        速度模型参数，由均分曲线拟合）——两者是不同量、不相等是对的。
        本测试只守 C++ 两处一致；并守 parallel.hpp 注释里仍声明这个区别
        （防"两个不同量又被合并成一个"的漂移）。
        """
        from nano.scheduling.scheduler import DEFAULT_E_CORE_EFFICIENCY

        micro_val = _extract_const(_micro_text(), "E_CORE_WEIGHT")
        hpp_text = PARALLEL_SRC.read_text(encoding="utf-8")
        m = re.search(
            r"e_core_partition_weight\(\).*?return\s+([0-9.]+)", hpp_text, re.S
        )
        assert m is not None, (
            "parallel.hpp 里找不到 e_core_partition_weight() 的 return 值——"
            "函数改名就一起改本测试"
        )
        hpp_val = float(m.group(1))
        assert micro_val == hpp_val, (
            f"bench_micro.cpp 的 E_CORE_WEIGHT={micro_val} 与 parallel.hpp 的 "
            f"分派权重 {hpp_val} 不一致 —— 两处必须同源"
            "（改权重必须重新跑 A/B 标定，见 KAKEYA_BENCH_PARTITION）"
        )
        assert "DEFAULT_E_CORE_EFFICIENCY" in hpp_text, (
            "parallel.hpp 的注释里应点明分派权重与 scheduler 的 "
            "DEFAULT_E_CORE_EFFICIENCY 不是一个量（口径说明被删掉了？）"
        )
        assert micro_val != DEFAULT_E_CORE_EFFICIENCY, (
            "分派权重与 predict_schedule 的 E 核调度效率变成同一个数了——"
            "两者口径不同（见 parallel.hpp 注释），出现同值即是漂移信号，"
            "请核实后改本测试"
        )

    def test_fused_mc_sampling_overhead_matches(self):
        """融合 MC 采样开销两处同源：bench_micro.cpp 的
        ``FUSED_MC_SAMPLING_OVERHEAD_NS`` 必须与 chip_model 的同名常量一致。

        **标定结论**（2026-10-30，配对交替协议）：开销在 ±1 ns 噪声内，
        按 0.0 计——采样指令被乱序执行隐藏。若哪天芯片/编译器让该值
        真的 >0，须重跑 ``KAKEYA_BENCH_FUSED_MC=1 ./bench_micro`` 标定后
        两处同改（注释里有完整实验口径）。
        """
        from nano.model.chip_model import FUSED_MC_SAMPLING_OVERHEAD_NS

        micro_val = _extract_const(_micro_text(), "FUSED_MC_SAMPLING_OVERHEAD_NS")
        assert micro_val == FUSED_MC_SAMPLING_OVERHEAD_NS, (
            f"bench_micro.cpp 的采样开销 {micro_val} 与 chip_model 的 "
            f"{FUSED_MC_SAMPLING_OVERHEAD_NS} 不一致 —— 两处必须同源"
            "（改值须用 KAKEYA_BENCH_FUSED_MC=1 重跑标定）"
        )
        hpp_text = _micro_text()
        assert "配对交替" in hpp_text, (
            "bench_micro.cpp 里应保留'配对交替'的标定协议说明——"
            "背靠背整段计时会被频率爬升倾斜（实测 ±2 ns 假差）"
        )


class TestSingleCoreCeilingFormulaMatches:
    """模型推出的单核上限，必须与微基准用**同一公式**。"""

    def test_single_core_ceiling_formula(self):
        """单核上限 = FLOP/pair / P 核峰值——两处口径一致。"""
        # 模型侧：用 P 核峰值算出的单核上限。
        compute_ns, _ = roofline_ceiling_ns_per_pair(
            _single_p_core_spec(), n_tubes=64
        )
        # 微基准侧：同一条公式（从源码注释与常量重建）。
        micro_expected = (
            _extract_const(_micro_text(), "FLOPS_PER_PAIR")
            / (_extract_const(_micro_text(), "p_core_gflops") * 1e9)
            * 1e9
        )
        assert compute_ns == pytest.approx(micro_expected), (
            f"模型单核上限 {compute_ns:.4f} 与微基准公式 {micro_expected:.4f} "
            "不一致 —— 两处对'单核上限'用了不同算法"
        )


def _single_p_core_spec():
    from dataclasses import replace

    from nano.model.chip_model import M4_REFERENCE

    return replace(
        M4_REFERENCE,
        name="1 P-core",
        peak_flops_gflops=M4_P_CORE_GFLOPS,
        n_processing_elements=1,
    )


def _build_micro(tmp_dir: Path) -> Path:
    """编译微基准；返回二进制路径。链不通则抛 RuntimeError（由测试转成 skip）。"""
    if shutil.which("clang++") is None:
        raise RuntimeError("本机没有 clang++")
    out = tmp_dir / "bench_micro"

    def _try(isysroot: str | None) -> subprocess.CompletedProcess:
        cmd = [
            "clang++", "-O3", "-std=c++17", "-pthread",
            "-ffast-math", str(MICRO_SRC), "-o", str(out),
        ]
        if isysroot:
            cmd += ["-isysroot", isysroot]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=180)

    # macOS：默认 SDK 可能比本机 ld 新而链不通（本机实测 MacOSX27.0 坏、
    # 26.5 好）。逐个 SDK 试到能链通的为止——与 cpp/build_nano.sh 同思路。
    if sys.platform == "darwin":
        sdk_root = Path("/Library/Developer/CommandLineTools/SDKs")
        sdk_dirs = sorted(sdk_root.glob("MacOSX*.sdk")) if sdk_root.is_dir() else []
        last: subprocess.CompletedProcess | None = None
        for iso in [None] + [str(p) for p in reversed(sdk_dirs)]:
            proc = _try(iso)
            if proc.returncode == 0:
                return out
            last = proc
        raise RuntimeError(
            f"所有 SDK 都无法链通微基准（环境问题，非代码错）："
            f"{(last.stderr if last else '')[-200:]}"
        )
    proc = _try(None)
    if proc.returncode != 0:
        raise RuntimeError(f"编译微基准失败（环境问题，非代码错）：{proc.stderr[-200:]}")
    return out


@pytest.fixture(scope="module")
def micro_binary(tmp_path_factory) -> Path:
    """编译一次，整模块复用。"""
    try:
        return _build_micro(tmp_path_factory.mktemp("micro"))
    except RuntimeError as exc:
        pytest.skip(str(exc))


class TestMicrobenchRuns:
    """能编译就实测一遍：结果物理合理，且并行计数逐位等于串行。"""

    def test_serial_result_is_physically_plausible(self, micro_binary):
        """串行 ns/pair 必须为正、且在可辩护区间（0.2~5 ns/pair）。

        **计量规模（2026-10-30 修）**：32 万 pair/调用 × 10 轮——早期用
        2 万 pair × 5 轮（总计 ~160 µs）太小：一次 ~100 µs 的调度抢占摊
        到均值就是 ~0.6 ns/pair，背景负载（Xprotect 扫描等）即可把读数
        顶出 5.0 上界（实测 5.09 误报）。规模放大 20× 后，同等抢占只摊
        ~0.03 ns/pair；而真回归（向量化失效等 10× 变慢）仍远在区间外。
        区间语义不变，成本 <1 s。
        """
        proc = subprocess.run(
            [str(micro_binary), "16", "20000", "10"],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, f"微基准非零退出：{proc.stderr[-300:]}"
        m = re.search(r"Serial:\s+[\d.]+ ms/call\s+\|\s+([\d.]+) ns/pair", proc.stdout)
        assert m is not None, f"输出里找不到 Serial ns/pair：\n{proc.stdout}"
        ns_per_pair = float(m.group(1))
        assert 0.2 <= ns_per_pair <= 5.0, (
            f"串行 {ns_per_pair} ns/pair 超出可辩护区间（0.2~5）—— "
            "要么机器异常、要么计数口径变了"
        )

    def test_parallel_counts_bit_identical(self, micro_binary):
        """微基准自己会断言"并行计数 == 串行计数"；退出码非零即是失败。

        这条把"纯 C++ 并行归约逐位一致"也纳入守门（此前只在 Python 层测过）。
        """
        proc = subprocess.run(
            [str(micro_binary), "16", "4000", "3"],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, (
            f"微基准报告并行计数与串行不一致（或运行失败）：\n{proc.stdout}\n{proc.stderr}"
        )
        # 输出里所有 count_match 必须是 yes。
        if "count_match" in proc.stdout:
            assert "NO" not in proc.stdout, (
                f"微基准报告了 count_match=NO：\n{proc.stdout}"
            )

    def test_reports_topology_and_roofline(self, micro_binary):
        """输出必须包含拓扑与 roofline 段（即"合流"确实发生了）。"""
        proc = subprocess.run(
            [str(micro_binary), "16", "2000", "3"],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, proc.stderr[-300:]
        out = proc.stdout
        assert "Topology:" in out, "微基准未报告拓扑"
        assert "Roofline" in out, "微基准未报告 roofline 对比"
        assert "chip_model" in out, "微基准未点明与 chip_model 的关系"

    def test_normal_run_passes_plausibility_gate(self, micro_binary):
        """正常实测（单核慢于上限）必须**退出码 0**——门控不该误伤。"""
        proc = subprocess.run(
            [str(micro_binary), "16", "4000", "5"],
            capture_output=True, text=True, timeout=120,
        )
        assert proc.returncode == 0, (
            f"正常实测被物理合理性门控误伤（退出码 {proc.returncode}）：\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
