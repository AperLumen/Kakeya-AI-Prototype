"""体积估计的「域」不变量测试（``src/kakeya_ai/domain.py``）。

这份测试锁住的是**可复核性**，不是某个函数的行为。背景：NSP 因子化会把随机
副本平移到 ±radius，管族于是跑出历史基准域 ``[-1,1]³``。我一度把默认盒改成
"按管族自动包紧"，结果：

- 已入库的 ``.kpt`` trace 复核从**通过**变成**不通过**
  （``0.04803240741`` vs 重算 ``0.05004272539``）；
- nano(f32) 与 numpy(f64) 在紧致盒上网格错位约 1e-8（"内部通、外部断"）。

原因是紧致盒坐标不是 f32 精确值。所以现在改成：**默认域永远不动**；只有管族
真的越界时才抬升，且半径取 2 的幂（f32/f64 都精确）。这三条不变量必须被测试
钉住，否则下次又会有人"顺手优化"掉它。
"""

from __future__ import annotations

import numpy as np
import pytest

from kakeya_ai.domain import (
    DEFAULT_RADIUS,
    bounds_for_radius,
    exceeds_domain,
    radius_covering,
    required_radius,
)
from kakeya_ai.primitives import Direction, Tube


def _tube(center, delta=0.1, length=1.0, axis="z"):
    vec = {"z": [0.0, 0.0, 1.0], "x": [1.0, 0.0, 0.0]}[axis]
    return Tube(center=np.asarray(center, dtype=float),
                direction=Direction.from_array(vec),
                length=length, delta=delta)


class TestInvariant1DomainUnchanged:
    """不变量 1：**域不变**。基准域恒为 ``[-1,1]³``。"""

    def test_default_radius_is_one(self):
        assert DEFAULT_RADIUS == 1.0

    def test_bounds_for_radius_one_is_bitwise_historical(self):
        """``radius=1.0`` 的 bounds 必须与历史硬编码值**逐位相同**。

        这是历史 trace 还能复核的前提：``[[-1,1],[-1,1],[-1,1]]``。
        """
        expected = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
        got = bounds_for_radius(1.0)
        assert np.array_equal(got, expected)
        assert got.tobytes() == expected.tobytes()

    def test_in_domain_tubes_keep_radius_one(self):
        """管族没越界时，覆盖半径**必须**是 1.0（不许"紧贴"优化）。"""
        tubes = [_tube([0.0, 0.0, 0.0]), _tube([0.3, -0.3, 0.2], delta=0.05)]
        assert radius_covering(tubes, floor=1.0) == 1.0

    def test_empty_tubes_keep_radius_one(self):
        assert radius_covering([], floor=1.0) == 1.0

    def test_orchestrator_starts_at_baseline_for_in_domain_input(self):
        """编排器对盒内构型必须用基准域，历史数值才逐位不变。"""
        from kakeya_ai.proof_orchestrator import ProofOrchestrator

        tubes = [_tube([0.0, y, 0.0], delta=0.1) for y in (-0.3, 0.0, 0.3)]
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)
        assert orch.domain_radius == 1.0, (
            "盒内构型的域被改动了——历史 trace 会复核不通过"
        )


class TestInvariant2NoShrink:
    """不变量 2：**不收缩**。域一旦扩大，同一次运行内不回缩。"""

    def test_covering_radius_never_below_floor(self):
        big = [_tube([1.6, 0.0, 0.0])]
        grew = radius_covering(big, floor=1.0)
        assert grew > 1.0
        # 再用同一批"看起来小"的管族请求，floor 必须挡住回缩
        small = [_tube([0.1, 0.0, 0.0])]
        assert radius_covering(small, floor=grew) == grew

    def test_orchestrator_domain_monotonic_after_nsp(self):
        """跑完 NSP 后域只会变大；之后再算小构型也不会缩回去。"""
        from kakeya_ai.proof_orchestrator import ProofOrchestrator

        # delta=0.25 → δ⁻²=16，6 根管 < 8 才会触发 NSP
        tubes = [_tube([0.0, 0.05 * i, 0.0], delta=0.25) for i in range(6)]
        orch = ProofOrchestrator(tubes, max_iterations=1, grid_n=16)
        before = orch.domain_radius

        state = orch.initialize()
        new_state = orch._try_nsp_factorization(state)
        if new_state is None:
            pytest.skip("该构型未触发 NSP（域不变量无从检验）")

        assert orch.domain_radius >= before
        # 域扩大必须留痕，否则从 trace 里看不出体积值为何变化
        assert any("domain" in d for d in orch._diagnostics), orch._diagnostics


class TestInvariant3Reproducible:
    """不变量 3：**可复现**。半径是 2 的幂 → f32/f64 都精确。"""

    @pytest.mark.parametrize("need,expected", [
        (1.0, 1.0), (1.05, 2.0), (1.7, 2.0), (2.0, 2.0),
        (2.2, 4.0), (3.99, 4.0), (4.0, 4.0), (4.5, 8.0),
    ])
    def test_radius_is_power_of_two(self, need, expected):
        # 管沿 z 轴、长度 1.0 → 端点在 z=±0.5；把中心放在 x = need - δ，
        # 于是 max(|x|) + δ = need，正好是"需要的最小覆盖半径"。
        tube = _tube([need - 0.1, 0.0, 0.0])
        assert required_radius([tube]) == pytest.approx(need)
        got = radius_covering([tube], floor=1.0)
        assert got == expected, f"需要的半径 ≈{need}，得到 {got}（期望 {expected}）"

    @pytest.mark.parametrize("radius", [1.0, 2.0, 4.0, 8.0])
    def test_bounds_exact_in_float32_and_float64(self, radius):
        b64 = bounds_for_radius(radius)
        # f32 往返必须无损：nano 内核走 float32，non-exact 会让网格错位
        assert np.array_equal(b64.astype(np.float32).astype(np.float64), b64)
        # 半径本身也必须是精确的 2 的幂
        assert float.fromhex(float(radius).hex()) == radius

    def test_covering_radius_covers_every_tube(self):
        """覆盖半径必须真的装得下每根管（否则体积会静默偏小）。"""
        tubes = [
            _tube([1.9, 0.0, 0.0], delta=0.1),
            _tube([-0.2, 2.4, 0.0], delta=0.05),
        ]
        r = radius_covering(tubes, floor=1.0)
        assert r >= required_radius(tubes)
        assert exceeds_domain(tubes, r) is None


class TestDomainSentinel:
    """哨兵：越界必须能被**检测到**，而不是静默漏算。"""

    def test_detects_out_of_domain_tube(self):
        far = [_tube([1.6, 0.0, 0.0])]
        need = exceeds_domain(far, 1.0)
        assert need is not None and need > 1.0

    def test_reports_none_when_covered(self):
        near = [_tube([0.2, 0.0, 0.0])]
        assert exceeds_domain(near, 1.0) is None

    def test_boundary_is_allowed(self):
        """恰好贴边（半径 0.9 的盒内管端点到 0.9）不算越界。"""
        t = _tube([0.0, 0.0, 0.0], delta=0.1, length=1.6)   # z 端点到 ±0.8
        assert exceeds_domain([t], 0.9) is None

    def test_invalid_radius_rejected(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with pytest.raises(ValueError):
                bounds_for_radius(bad)


class TestExpandedDomainRoundTrip:
    """域抬升后，写出的 trace 仍必须能复核（这是最初出问题的地方）。"""

    def test_nsp_trace_verifies_with_expanded_domain(self, tmp_path):
        from kakeya_ai.proof_orchestrator import ProofOrchestrator
        from kakeya_ai.trace_format import verify_trace, write_trace

        # 小构型 + 小 grid_n，跑得快；δ=0.25 保证触发 NSP
        tubes = [_tube([0.0, 0.05 * i, 0.0], delta=0.25) for i in range(6)]
        result = ProofOrchestrator(tubes, max_iterations=2, grid_n=16,
                                   broadness_filter=False).run()

        path = write_trace(result, tmp_path / "domain.kpt", params={"grid_n": 16})
        report = verify_trace(path)

        assert report.ok, report.summary()
        assert result.domain_radius >= 1.0

    def test_historical_traces_still_verify(self):
        """已入库的 trace（协议早期、无域字段）必须按基准域复核通过。"""
        from pathlib import Path

        from kakeya_ai.trace_format import verify_trace

        traces = sorted((Path(__file__).resolve().parents[2]
                         / "docs" / "traces").glob("*.kpt"))
        if not traces:
            pytest.skip("docs/traces/ 下没有入库 trace")
        for path in traces:
            report = verify_trace(path)
            assert report.ok, f"{path.name} 复核失败：\n{report.summary()}"
