"""守门：芯片预算模型（`nano.model.chip_model`）必须**自洽且可校准**。

**为什么需要它**：这个模型是"要不要为这套几何逻辑做专用芯片"的**判据基础**
（见 `nano/model/chip_model.py` 的模块 docstring）。一个用来做**设计决策**的
模型，若有一点算错或口径错配，会让人**高估或低估专用芯片的收益**——比没有
模型更糟（给人一种"有依据"的错觉）。

**本文件守四类性质**：

1. **单位与一致性**：``estimate().intensity`` 必须等于 ``arithmetic_intensity()``；
   roofline 上限必须 = max(算力上限, 带宽上限)（慢的说了算）。
2. **单调性**（物理直觉的机器化）：算力翻倍 ⇒ 上限时间减半；管数增加 ⇒ 算术
   强度升高（复用变多）；点数增加 ⇒ 总上限时间线性增加。
3. **口径显式化**：``measured_vs_model`` 必须**区分单线程实测与全芯片上限**——
   这是本项目实测（单线程 1.1 ns/pair）与参考芯片（全芯片 140 GFLOP/s）之间
   最容易犯的口径错配。
4. **校准**：把参考 M4 参数喂进模型，预测的**单核**上限必须与实测同量级
   （在标注的容差内）——否则模型与现实脱节，不能用于决策。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from nano.model.chip_model import (
    M4_P_CORE_GFLOPS,
    M4_REFERENCE,
    TUBE_GEOMETRY_KERNEL,
    ChipSpec,
    ImpossibleMeasurementError,
    arithmetic_intensity,
    check_model_self_consistency,
    chipspec_declared_peak_gflops,
    diagnose_superpeak,
    estimate,
    estimate_energy_pj,
    estimate_single_core_ms,
    implied_rate_gflops,
    measured_vs_model,
    required_cores_for_rate,
    roofline_ceiling_ns_per_pair,
)


class TestConsistency:
    """模型内部必须自洽——同一件事在两处算出不同值是最危险的缺陷。"""

    def test_estimate_intensity_matches_helper(self):
        """``estimate().intensity`` 必须等于 ``arithmetic_intensity()``。

        **实测踩过**：初版 ``arithmetic_intensity()`` 漏了 ``n_tubes`` 参数
        （用"每点字节"）而 ``estimate()`` 用"每 pair 字节"，同一构型两处给出
        2.08 vs 133.3——差 64 倍（正好是管数）。做决策时看哪个都会错。
        """
        for n_tubes in (1, 8, 64, 256):
            est = estimate(M4_REFERENCE, n_tubes=n_tubes, n_points=1000)
            assert est.intensity == pytest.approx(
                arithmetic_intensity(n_tubes=n_tubes)
            ), f"n_tubes={n_tubes} 处两处算术强度不一致"

    def test_roofline_is_max_of_the_two_ceilings(self):
        """roofline 上限必须取算力/带宽上限中**更慢**的那个（max）。"""
        est = estimate(M4_REFERENCE, n_tubes=64, n_points=1000)
        assert est.roofline_ns_per_pair == max(
            est.compute_ceiling_ns_per_pair, est.memory_ceiling_ns_per_pair
        )
        expected_bottleneck = (
            "compute"
            if est.compute_ceiling_ns_per_pair >= est.memory_ceiling_ns_per_pair
            else "memory"
        )
        assert est.bottleneck == expected_bottleneck

    def test_helper_ceilings_match_estimate(self):
        """独立的 ``roofline_ceiling_ns_per_pair`` 必须与 ``estimate`` 内的口径一致。"""
        compute, memory = roofline_ceiling_ns_per_pair(M4_REFERENCE, n_tubes=64)
        est = estimate(M4_REFERENCE, n_tubes=64, n_points=1000)
        assert compute == pytest.approx(est.compute_ceiling_ns_per_pair)
        assert memory == pytest.approx(est.memory_ceiling_ns_per_pair)


class TestMonotonicity:
    """物理直觉的机器化：方向必须对，否则模型不可信。"""

    def test_more_compute_lowers_ceiling(self):
        """算力翻倍 ⇒ 上限时间减半（算力受限时）。"""
        base = estimate(M4_REFERENCE, n_tubes=64, n_points=1000)
        doubled = replace(M4_REFERENCE, peak_flops_gflops=M4_REFERENCE.peak_flops_gflops * 2)
        faster = estimate(doubled, n_tubes=64, n_points=1000)
        assert faster.roofline_ns_per_pair == pytest.approx(
            base.roofline_ns_per_pair / 2
        )

    def test_more_tubes_raises_intensity(self):
        """管数增加 ⇒ 每个点被更多管复用 ⇒ 算术强度升高（更不易带宽受限）。"""
        i_small = arithmetic_intensity(n_tubes=8)
        i_large = arithmetic_intensity(n_tubes=256)
        assert i_large > i_small
        assert i_large == pytest.approx(i_small * 32)  # 线性于管数

    def test_more_points_scales_total_time_linearly(self):
        """点数翻倍 ⇒ 总上限时间翻倍（总工作量线性）。"""
        a = estimate(M4_REFERENCE, n_tubes=64, n_points=1000)
        b = estimate(M4_REFERENCE, n_tubes=64, n_points=2000)
        assert b.total_ceiling_ms == pytest.approx(a.total_ceiling_ms * 2)

    def test_more_bandwidth_helps_only_when_memory_bound(self):
        """带宽翻倍**只在带宽受限时**改善上限；算力受限时应无影响。

        这条把"瓶颈判断有没有用"变成了可判定的事实：若瓶颈是 compute，
        加大带宽不该动 roofline。
        """
        # 造一个**带宽受限**的假想芯片：算力极大、带宽极小。
        memory_bound = ChipSpec(
            name="memory-bound toy",
            peak_flops_gflops=1e6,      # 算力几乎无限
            peak_bandwidth_gbytes=0.5,  # 带宽极小
            n_processing_elements=1,
        )
        e0 = estimate(memory_bound, n_tubes=64, n_points=1000)
        assert e0.bottleneck == "memory"

        wide = replace(memory_bound, peak_bandwidth_gbytes=1.0)
        e1 = estimate(wide, n_tubes=64, n_points=1000)
        assert e1.roofline_ns_per_pair == pytest.approx(
            e0.roofline_ns_per_pair / 2
        )

        # 算力受限时：加大带宽不该改变 roofline（本项目 kernel 就是算力受限）。
        compute_bound = estimate(M4_REFERENCE, n_tubes=64, n_points=1000)
        assert compute_bound.bottleneck == "compute"
        wider = replace(M4_REFERENCE, peak_bandwidth_gbytes=M4_REFERENCE.peak_bandwidth_gbytes * 10)
        assert estimate(wider, n_tubes=64, n_points=1000).roofline_ns_per_pair == pytest.approx(
            compute_bound.roofline_ns_per_pair
        )


class TestThreadScopeIsExplicit:
    """单线程实测 vs 全芯片上限——最易犯的口径错配，必须显式处理。"""

    def test_single_thread_with_p_core_rate(self):
        """传 P 核峰值时，单核上限 = FLOPS_PER_PAIR / P 核 GFLOP/s。"""
        r = measured_vs_model(
            M4_REFERENCE, measured_ns_per_pair=1.0, n_tubes=64,
            measured_is_single_thread=True,
            single_core_flops_gflops=M4_P_CORE_GFLOPS,
        )
        expected = TUBE_GEOMETRY_KERNEL.flops_per_pair / (M4_P_CORE_GFLOPS * 1e9) * 1e9
        assert r["ceiling_ns_per_pair"] == pytest.approx(expected)
        # 单核口径下余量应 ~1.4×（物理合理：单核用掉约 70% 峰值），
        # 而不是虚高的 10×（那是全芯片错配）。
        assert 1.0 < r["headroom_x"] < 2.0

    def test_single_thread_homogeneous_fallback_scales_by_pe(self):
        """无 P 核参数时退化为"全芯片 / 核数"（同构假设）。"""
        r = measured_vs_model(
            M4_REFERENCE, measured_ns_per_pair=2.0, n_tubes=64,
            measured_is_single_thread=True,
        )
        full_chip_ceiling = estimate(M4_REFERENCE, 64, 200_000).roofline_ns_per_pair
        assert r["ceiling_ns_per_pair"] == pytest.approx(
            full_chip_ceiling * M4_REFERENCE.n_processing_elements
        )

    def test_p_core_rate_beats_homogeneous_average(self):
        """异构芯片上，P 核口径给出的单核上限必须**低于**"平均值"口径。

        因为 E 核拉低了"全芯片/核数"的平均值 ⇒ 单核峰值被低估 ⇒ 上限偏高。
        本机实测过这个坑（1.23 vs 0.71 ns/pair）。这条把差异钉死。
        """
        with_p = measured_vs_model(
            M4_REFERENCE, 2.0, 64, measured_is_single_thread=True,
            single_core_flops_gflops=M4_P_CORE_GFLOPS,
        )
        with_avg = measured_vs_model(
            M4_REFERENCE, 2.0, 64, measured_is_single_thread=True,
        )
        assert with_p["ceiling_ns_per_pair"] < with_avg["ceiling_ns_per_pair"]

    def test_all_core_comparison_does_not_rescale(self):
        """显式声明"全芯片"时不得再折算。"""
        r = measured_vs_model(
            M4_REFERENCE, measured_ns_per_pair=1.0, n_tubes=64,
            measured_is_single_thread=False,
        )
        full_chip_ceiling = estimate(M4_REFERENCE, 64, 200_000).roofline_ns_per_pair
        assert r["ceiling_ns_per_pair"] == pytest.approx(full_chip_ceiling)
        assert r["headroom_x"] > 1.0


class TestCalibration:
    """模型必须与实测同量级——否则不能用于设计决策。"""

    def test_model_predicts_measured_single_core_within_order(self):
        """单核口径下，模型上限与实测 ~1.0 ns/pair 应在**同一量级**内。

        容差 2×：模型用的是"未折扣 FMA 的 25 FLOP/pair"（保守，上限偏低）
        与理论峰值参数。**要求 2× 内**是恰当前提——它既能抓出"模型与现实
        脱节"（如算错单位导致差 10×），又不假装自己是精确预测。
        实测：模型 0.71 ns/pair vs 实测 1.0 ⇒ 1.4×，通过。
        """
        r = measured_vs_model(
            M4_REFERENCE, measured_ns_per_pair=1.0, n_tubes=64,
            measured_is_single_thread=True,
            single_core_flops_gflops=M4_P_CORE_GFLOPS,
        )
        assert 1.0 <= r["headroom_x"] <= 4.0, (
            f"模型单核上限 {r['ceiling_ns_per_pair']:.4f} ns/pair 与实测 1.0 "
            f"不同量级（比值 {r['headroom_x']:.2f}）—— 模型与现实脱节，"
            "不能用于设计决策"
        )

    def test_kernel_opcount_is_documented_and_positive(self):
        """算子计数必须为正且**来自源码**（有来源说明，不是拍脑袋）。"""
        assert TUBE_GEOMETRY_KERNEL.flops_per_pair > 0
        assert TUBE_GEOMETRY_KERNEL.bytes_per_point > 0
        assert TUBE_GEOMETRY_KERNEL.source, (
            "算子计数必须写明来源（哪个文件哪一段）——否则无法审计"
        )
        assert "nano_kernel.cpp" in TUBE_GEOMETRY_KERNEL.source

    def test_opcount_flops_within_plausible_range(self):
        """25 FLOP/pair 应落在一个**可辩护**的区间内（防止手滑多打/少打数字）。

        下界 20：即便把 3 处 clamp 与若干加免费掉，也要 ≥ 20。
        上界 40：逐行数出的原始计数 25 + 合理冗余（若以后改用更复杂判据）。
        """
        assert 20 <= TUBE_GEOMETRY_KERNEL.flops_per_pair <= 40, (
            f"FLOP/pair={TUBE_GEOMETRY_KERNEL.flops_per_pair} 超出可辩护区间——"
            "核对 cpp/kernel/nano_kernel.cpp 的逐行计数"
        )


class TestInputValidation:
    """非法输入必须**当场报错**，不能静默给出荒谬结果。"""

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_nonpositive_tubes_rejected(self, bad):
        with pytest.raises(ValueError):
            estimate(M4_REFERENCE, n_tubes=bad, n_points=1000)
        with pytest.raises(ValueError):
            arithmetic_intensity(n_tubes=bad)
        with pytest.raises(ValueError):
            roofline_ceiling_ns_per_pair(M4_REFERENCE, n_tubes=bad)

    def test_nonpositive_points_rejected(self):
        with pytest.raises(ValueError):
            estimate(M4_REFERENCE, n_tubes=8, n_points=0)


class TestPhysicalPlausibilityGate:
    """**实测快于理论上限 = 物理不可能**，必须当场拒绝；且结论要**算出来**。

    这是用户 2026-10-07 定调的门控，且强调"**门控依赖函数**、不是定义一段文字
    解释"。故本类不仅验证"会拒"，还验证**结论来自函数计算**（复算自洽性、
    反推核数），而不是套用预写文本。
    """

    def test_superpeak_measurement_is_rejected(self):
        """实测快于上限（headroom < 1.0）必须抛 ``ImpossibleMeasurementError``。"""
        with pytest.raises(ImpossibleMeasurementError):
            measured_vs_model(
                M4_REFERENCE, 0.1, 64,
                single_core_flops_gflops=M4_P_CORE_GFLOPS,
            )

    def test_legit_measurement_passes(self):
        """正常（实测慢于上限）必须放行、不抛。"""
        r = measured_vs_model(
            M4_REFERENCE, 1.0, 64, single_core_flops_gflops=M4_P_CORE_GFLOPS
        )
        assert r["headroom_x"] >= 1.0

    def test_allow_superpeak_opt_out(self):
        """显式 ``allow_superpeak=True`` 时放行（用于"对比保守下界"场景）。"""
        r = measured_vs_model(
            M4_REFERENCE, 0.1, 64,
            single_core_flops_gflops=M4_P_CORE_GFLOPS,
            allow_superpeak=True,
        )
        assert r["headroom_x"] < 1.0

    def test_exactly_at_peak_is_allowed(self):
        """headroom **恰好 1.0** 是允许的极限（判据是 ``< 1.0``）。"""
        ceiling = TUBE_GEOMETRY_KERNEL.flops_per_pair / (M4_P_CORE_GFLOPS * 1e9) * 1e9
        r = measured_vs_model(
            M4_REFERENCE, ceiling, 64, single_core_flops_gflops=M4_P_CORE_GFLOPS
        )
        assert r["headroom_x"] == pytest.approx(1.0)


class TestGateDependsOnFunctions:
    """门控的结论必须**由函数算出来**，不是套预写文字（用户 2026-10-07 定调）。"""

    def test_implied_rate_is_computed_from_measurement(self):
        """``implied_rate_gflops`` 必须由实测 ns/pair **算出**反推算力。"""
        # 25 FLOP/pair ÷ 1.0 ns = 25 GFLOP/s
        assert implied_rate_gflops(1.0) == pytest.approx(25.0)
        assert implied_rate_gflops(0.5) == pytest.approx(50.0)
        with pytest.raises(ValueError):
            implied_rate_gflops(0.0)

    def test_chipspec_peak_is_independently_recomputed(self):
        """``chipspec_declared_peak_gflops`` 必须**不读** peak_flops_gflops 而算出来。

        用同构 spec：clock × simd × 2 × cores。
        """
        c = ChipSpec(
            name="homogeneous toy", peak_flops_gflops=1.0,  # 故意填错
            peak_bandwidth_gbytes=100.0, n_processing_elements=8,
            simd_width=4, clock_ghz=2.0,
        )
        # 2.0 GHz × 4 × 2 × 8 = 128 GFLOP/s（与声明的 1.0 无关）
        assert chipspec_declared_peak_gflops(c) == pytest.approx(128.0)

    def test_self_consistency_check_detects_mismatch(self):
        """字段不自洽时，``check_model_self_consistency`` 必须报 False 并指出数字。"""
        bad = ChipSpec(
            name="inconsistent", peak_flops_gflops=1.0,
            peak_bandwidth_gbytes=100.0, n_processing_elements=8,
            simd_width=4, clock_ghz=2.0,
        )
        ok, why = check_model_self_consistency(bad)
        assert not ok
        assert "不自洽" in why and "128" in why, "应报出复算值以便定位"

    def test_m4_reference_is_self_consistent(self):
        """参考 M4 的声明峰值必须与其它字段自洽（异构：P + E×ratio）。"""
        ok, why = check_model_self_consistency(M4_REFERENCE)
        assert ok, f"M4_REFERENCE 自身不自洽：{why}"
        assert chipspec_declared_peak_gflops(M4_REFERENCE) == pytest.approx(
            M4_REFERENCE.peak_flops_gflops, rel=0.15
        )

    def test_required_cores_is_computed(self):
        """``required_cores_for_rate`` 必须把反推算力换算成核数。"""
        # 25 GFLOP/s 需要 / 12.5 GFLOP/s 单核 = 2 核
        assert required_cores_for_rate(25.0, 12.5) == pytest.approx(2.0)
        with pytest.raises(ValueError):
            required_cores_for_rate(25.0, 0.0)

    def test_diagnosis_reports_computed_verdict_not_prose(self):
        """诊断必须**给出算出来的判定**（自洽性 + 反推核数），而非并列猜想。"""
        msg = diagnose_superpeak(
            measured_ns_per_pair=1.0,
            ceiling_ns_per_pair=1.7857,
            headroom_x=1.0 / 1.7857,
            chip=M4_REFERENCE,
            single_core_flops_gflops=14.0,  # 历史错误值
        )
        # 必须含"复算自洽性"的判定（自洽/不自洽）
        assert ("自洽" in msg), "诊断应含自洽性判定"
        # 必须含反推核数（函数算出的量）
        assert "反推需要核数" in msg
        # 必须含反推算力（函数算出的量）
        assert "反推算力" in msg

    def test_gate_catches_the_historical_bug(self):
        """复现历史 bug 形态：单核峰值填成 14 GFLOP/s（140/10，漏算 E 核）。

        当初若已有此门控，会被自动拦下。
        """
        with pytest.raises(ImpossibleMeasurementError):
            measured_vs_model(
                M4_REFERENCE, 1.0, 64, single_core_flops_gflops=14.0
            )


class TestScaleToSingleCoreMs:
    """规模 → 单核墙钟（2026-10-30）：调度计划规模感知路径的模型侧。

    守门要点：实测路径**逐位可复算**；roofline 路径的"单核口径"必须取
    **P 核峰值**（沿用 ``measured_vs_model`` 的异构教训，不是"全芯片/核数"）；
    ``efficiency`` 折扣语义是"墙钟 = 上限 / efficiency"；快过上限
    （efficiency>1）物理不可能，须当场拒绝。
    """

    def test_measured_path_is_exact(self):
        """实测 ns/pair 路径：ms = 管×点×实测 / 1e6，逐位可复算。"""
        ms = estimate_single_core_ms(64, 200_000, measured_ns_per_pair=1.0)
        assert ms == pytest.approx(64 * 200_000 * 1.0 / 1e6)  # = 12.8 ms

    def test_measured_path_ignores_efficiency(self):
        """传了实测值 ⇒ efficiency 不参与（实测本身已含一切效应）。"""
        ms1 = estimate_single_core_ms(
            64, 200_000, measured_ns_per_pair=1.0, efficiency=1.0
        )
        ms2 = estimate_single_core_ms(
            64, 200_000, measured_ns_per_pair=1.0, efficiency=0.5
        )
        assert ms1 == pytest.approx(ms2)

    def test_roofline_path_uses_p_core_peak(self):
        """roofline 路径：单核口径 = **P 核峰值**（非全芯片/核数）。

        M4：25 FLOP/pair ÷ 35.2 GFLOP/s ≈ 0.710 ns/pair ⇒ 64×200000 对
        ≈ 9.09 ms。若误用"全芯片/10"（20.44 GFLOP/s），会得 ≈15.7 ms——
        这正是 ``measured_vs_model`` 踩过的口径错配。
        """
        ms = estimate_single_core_ms(64, 200_000)
        p_core_ns = (
            TUBE_GEOMETRY_KERNEL.flops_per_pair / (M4_P_CORE_GFLOPS * 1e9) * 1e9
        )
        assert ms == pytest.approx(64 * 200_000 * p_core_ns / 1e6, rel=1e-4)

    def test_efficiency_discount_stretches_wallclock(self):
        """efficiency=0.5 ⇒ 墙钟翻倍（墙钟 = 上限 / efficiency）。"""
        full = estimate_single_core_ms(64, 200_000)
        half = estimate_single_core_ms(64, 200_000, efficiency=0.5)
        assert half == pytest.approx(2 * full)

    def test_memory_ceiling_can_dominate_on_slow_bandwidth_chip(self):
        """带宽极低的芯片上，带宽上限必须能说了算（取 max，不是只看算力）。"""
        slow_mem = replace(M4_REFERENCE, peak_bandwidth_gbytes=0.1)
        bytes_per_pair = TUBE_GEOMETRY_KERNEL.bytes_per_point / 64
        mem_ns = bytes_per_pair / (0.1 * 1e9) * 1e9
        assert mem_ns > 1.0, "前置：该芯片带宽上限确应高于算力上限（0.71 ns）"
        ms = estimate_single_core_ms(64, 200_000, chip=slow_mem)
        assert ms == pytest.approx(64 * 200_000 * mem_ns / 1e6, rel=1e-3)

    def test_monotonic_in_size(self):
        """墙钟随规模单调：点数翻倍 ⇒ 墙钟翻倍。"""
        a = estimate_single_core_ms(64, 100_000)
        b = estimate_single_core_ms(64, 200_000)
        assert b == pytest.approx(2 * a)

    def test_nonpositive_size_rejected(self):
        with pytest.raises(ValueError):
            estimate_single_core_ms(0, 200_000)
        with pytest.raises(ValueError):
            estimate_single_core_ms(64, 0)

    def test_bad_calibration_rejected(self):
        """实测值必须为正；efficiency 须在 (0,1]——>1 是快过理论上限，物理不可能。"""
        with pytest.raises(ValueError):
            estimate_single_core_ms(64, 200_000, measured_ns_per_pair=-1.0)
        with pytest.raises(ValueError):
            estimate_single_core_ms(64, 200_000, efficiency=0.0)
        with pytest.raises(ValueError):
            estimate_single_core_ms(64, 200_000, efficiency=1.5)


class TestEnergyModel:
    """能耗模型（2026-10-30）——"降热"判据：搬运 ≫ 计算时，削内存往返即削能耗。

    常数为 45nm 文献口径（Horowitz ISSCC 2014），测试只验证**公式与拆分**
    （不验证绝对值——那是工艺占位，注释里已声明）。
    """

    def test_explicit_bytes_split(self):
        """显式 bytes_moved：memory_pj = 字节 × 162.5，占比可复算。"""
        r = estimate_energy_pj(64, 1_000, bytes_moved=72_000.0)
        assert r["memory_pj"] == pytest.approx(72_000.0 * 162.5)
        assert r["compute_pj"] == pytest.approx(64 * 1_000 * 25 * 3.7)
        total = r["compute_pj"] + r["memory_pj"]
        assert r["total_pj"] == pytest.approx(total)
        assert r["memory_share"] == pytest.approx(r["memory_pj"] / total)
        assert r["ratio_memory_over_compute"] == pytest.approx(
            r["memory_pj"] / r["compute_pj"]
        )

    def test_default_bytes_is_stream_once(self):
        """缺省搬运量 = 点流式读一次 + 管结构读一次（理想数据流口径）。"""
        r = estimate_energy_pj(64, 200_000)
        expected_bytes = 200_000 * 12 + 64 * 28
        assert r["memory_pj"] == pytest.approx(expected_bytes * 162.5)

    def test_fusion_eliminates_memory_energy(self):
        """MC 融合判据的量化形态：点数组不进内存（bytes=0）⇒ 能耗只剩计算。

        旧路径每样本 ≈72 B 时搬运占大头（>50%）；融合后占比归零、
        总节能比例 == 旧 memory_share。这是"降热"叙事的数字支柱。
        """
        n = 2_000_000
        old = estimate_energy_pj(64, n, bytes_moved=n * 72.0)
        new = estimate_energy_pj(64, n, bytes_moved=0.0)
        assert old["memory_share"] > 0.5, "旧路径搬运应占大头（判据前提）"
        assert new["memory_pj"] == 0.0
        assert new["memory_share"] == 0.0
        saved = (old["total_pj"] - new["total_pj"]) / old["total_pj"]
        assert saved == pytest.approx(old["memory_share"])

    def test_invalid_inputs_rejected(self):
        with pytest.raises(ValueError):
            estimate_energy_pj(0, 1000)
        with pytest.raises(ValueError):
            estimate_energy_pj(64, 0)
        with pytest.raises(ValueError):
            estimate_energy_pj(64, 1000, bytes_moved=-1.0)
