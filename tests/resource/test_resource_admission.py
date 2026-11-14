"""驻留准入控制（resource.admission）的单元测试。

快照全部手工构造注入，不读真机——三道筛子的判定是纯函数，
真机读数只发生在 admit() 未注入 memory 的默认路径（由专门的
冒烟用例覆盖）。守卫线 2 GB 与实测常数保持同一口径。
"""

import pytest

from kakeya_ai.resource import admission
from kakeya_ai.resource.admission import (
    MODEL_REGISTRY,
    AdmissionVerdict,
    LoadRequest,
    ModelProfile,
    admit,
    demand_gate,
    resource_gate,
    utility_gate,
)
from kakeya_ai.resource.probe import MemorySnapshot

GIB = 2**30


def _snap(avail_gib: float, total_gib: float = 16.0) -> MemorySnapshot:
    """构造指定可用量的内存快照（pressure = 1 - available/total）。"""
    avail = int(avail_gib * GIB)
    return MemorySnapshot(
        source="test",
        page_size=16384,
        total=int(total_gib * GIB),
        available=avail,
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


QWEN = MODEL_REGISTRY["qwen3.5:4b"]  # 4G, capability 0.6
GEMMA = MODEL_REGISTRY["gemma4:12b"]  # 8G, capability 0.85


class TestDemandGate:
    """第 1 道筛：拒绝推测性驻留。"""

    def test_empty_purpose_rejected(self):
        req = LoadRequest(model=QWEN, purpose="", estimated_tokens=512)
        assert demand_gate(req) is not None

    def test_zero_tokens_rejected(self):
        req = LoadRequest(model=QWEN, purpose="chat", estimated_tokens=0)
        assert demand_gate(req) is not None

    def test_normal_request_passes(self):
        req = LoadRequest(model=QWEN, purpose="chat", estimated_tokens=1024)
        assert demand_gate(req) is None

    def test_demand_lookup_unknown_purpose_is_conservative(self):
        # 未登记的用途按最高需求 0.9 处理——宁可拒错，不可放进占内存的废物。
        req = LoadRequest(model=QWEN, purpose="quantum-chemistry", estimated_tokens=1024)
        assert req.demand == 0.9


class TestResourceGate:
    """第 2 道筛：装得下且留守卫线、压力不超限。"""

    def test_fits_with_room_passes(self):
        req = LoadRequest(model=QWEN, purpose="chat")
        assert resource_gate(req, _snap(avail_gib=8.0)) is None

    def test_below_reserve_rejected(self):
        # 可用 9G - 8G 驻留 = 1G < 2G 守卫线。
        req = LoadRequest(model=GEMMA, purpose="chat")
        reason = resource_gate(req, _snap(avail_gib=9.0))
        assert reason is not None and "守卫线" in reason

    def test_high_pressure_rejected_even_if_fits(self):
        # 总 16G、可用 1G → pressure 0.94，远超阈值；也装不下。
        req = LoadRequest(model=QWEN, purpose="chat")
        reason = resource_gate(req, _snap(avail_gib=1.0))
        assert reason is not None

    def test_pressure_limit_enforced(self):
        # 可用 10G/16G → pressure 0.375，应通过。
        req = LoadRequest(model=QWEN, purpose="chat")
        assert resource_gate(req, _snap(avail_gib=10.0)) is None


class TestUtilityGate:
    """第 3 道筛：最低充分选择——不足拒，过剩也拒。"""

    def test_underqualified_rejected(self):
        req = LoadRequest(model=QWEN, purpose="math")  # demand 0.9 > 0.6
        reason = utility_gate(req)
        assert reason is not None and "能力不足" in reason

    def test_oversized_rejected_with_smaller_suggestion(self):
        # chat 需求 0.4，4G 的 qwen 已满足 → 拒 8G 的 gemma。
        req = LoadRequest(model=GEMMA, purpose="chat")
        reason = utility_gate(req)
        assert reason is not None and "能力过剩" in reason

    def test_minimal_sufficient_passes(self):
        req = LoadRequest(model=QWEN, purpose="chat")  # 4G ≥ 0.4，无更小者
        assert utility_gate(req) is None

    def test_largest_model_passes_when_demand_high(self):
        req = LoadRequest(model=GEMMA, purpose="math")  # 需求 0.9，只有它够
        assert utility_gate(req) is None


class TestAdmit:
    """编排入口：顺序执行、fail-fast、拒绝带建议。"""

    def test_demand_fail_fast_before_resource(self):
        req = LoadRequest(model=GEMMA, purpose="", estimated_tokens=1024)
        v = admit(req, memory=_snap(avail_gib=1.0))  # 资源也不足，但应先报需求筛
        assert not v.allowed and v.gate == "demand"

    def test_resource_fail_suggests_eviction(self):
        residents = {"gemma4:12b": GEMMA}
        req = LoadRequest(model=ModelProfile("big", 10 * GIB, 0.9), purpose="math")
        v = admit(req, residents=residents, memory=_snap(avail_gib=5.0))
        assert not v.allowed and v.gate == "resource"
        assert "驱逐" in v.suggestion

    def test_resource_fail_without_residents(self):
        req = LoadRequest(model=GEMMA, purpose="chat")
        v = admit(req, residents={}, memory=_snap(avail_gib=3.0))
        assert not v.allowed and v.gate == "resource"

    def test_utility_fail_has_suggestion(self):
        req = LoadRequest(model=GEMMA, purpose="chat")
        v = admit(req, memory=_snap(avail_gib=14.0))
        assert not v.allowed and v.gate == "utility" and v.suggestion

    def test_full_pass(self):
        req = LoadRequest(model=QWEN, purpose="chat")
        v = admit(req, memory=_snap(avail_gib=10.0))
        assert v.allowed and v.gate == "ok"

    def test_verdict_is_frozen(self):
        v = AdmissionVerdict(True, "ok", "x")
        with pytest.raises(AttributeError):
            v.allowed = False  # type: ignore[misc]


class TestSmoke:
    """默认路径冒烟：admit 未注入 memory 时读真机快照，不得抛。"""

    def test_admit_real_machine_no_raise(self):
        req = LoadRequest(model=QWEN, purpose="chat")
        v = admit(req)
        assert isinstance(v.allowed, bool)
