"""resource 子包（内存配额分配器）的测试。"""

import pytest

from kakeya_ai.resource import (
    DEFAULT_RESERVE_BYTES,
    InsufficientMemoryError,
    MemoryAllocator,
    MemoryQuota,
    acquire,
    allocator,
    available_memory,
    check_gpu_headroom,
    has_gpu_headroom,
    is_sustained_low,
    recent_trend,
    release_all,
    reserve_bytes,
    reset_cache,
)


class TestMemoryAllocator:
    def setup_method(self):
        release_all()
        reset_cache()

    def test_singleton_allocator(self):
        from kakeya_ai.resource.allocator import allocator as mod_allocator
        assert allocator is mod_allocator
        assert allocator.total_allocated == 0

    def test_request_and_release(self, monkeypatch):
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.available_memory", lambda: 8 * 1024**3
        )
        q = allocator.request(required=1 * 1024**3)
        assert isinstance(q, MemoryQuota)
        assert allocator.total_allocated == 1 * 1024**3
        q.release()
        assert allocator.total_allocated == 0

    def test_release_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.available_memory", lambda: 8 * 1024**3
        )
        q = allocator.request(required=1 * 1024**3)
        q.release()
        q.release()
        assert allocator.total_allocated == 0

    def test_total_allocated_accounts_multiple_quotas(self, monkeypatch):
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.available_memory", lambda: 16 * 1024**3
        )
        q1 = allocator.request(required=1 * 1024**3)
        q2 = allocator.request(required=2 * 1024**3)
        assert allocator.total_allocated == 3 * 1024**3
        q1.release()
        assert allocator.total_allocated == 2 * 1024**3
        q2.release()
        assert allocator.total_allocated == 0

    def test_acquire_context_manager(self, monkeypatch):
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.available_memory", lambda: 8 * 1024**3
        )
        with allocator.acquire(required=1 * 1024**3) as q:
            assert allocator.total_allocated == 1 * 1024**3
            assert isinstance(q, MemoryQuota)
        assert allocator.total_allocated == 0

    def test_module_level_acquire(self, monkeypatch):
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.available_memory", lambda: 8 * 1024**3
        )
        with acquire(required=1 * 1024**3) as _:
            assert allocator.total_allocated == 1 * 1024**3
        assert allocator.total_allocated == 0

    def test_strict_rejects_when_insufficient(self, monkeypatch):
        # 0 配额申请：avail=1GB，need=0 + 2GB(reserve) + 0 = 2GB > 1GB
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.available_memory", lambda: 1 * 1024**3
        )
        with pytest.raises(InsufficientMemoryError):
            allocator.request(required=0, strict=True)

    def test_lenient_passes_transient_low(self, monkeypatch):
        """瞬时低谷不应被持续判定拦下。"""
        readings = iter([1 * 1024**3, 8 * 1024**3])
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.available_memory",
            lambda: next(readings, 8 * 1024**3),
        )
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.SUSTAINED_INTERVAL", 0.0
        )
        q = allocator.request(required=0)
        q.release()

    def test_request_considers_already_allocated(self, monkeypatch):
        """已分配配额应计入本次申请的判断。"""
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.available_memory", lambda: 5 * 1024**3
        )
        q1 = allocator.request(required=2 * 1024**3)
        # 已分配 2 GB + 保留 2 GB + 本次 2 GB = 6 GB > 可用 5 GB
        with pytest.raises(InsufficientMemoryError):
            allocator.request(required=2 * 1024**3, strict=True)
        q1.release()

    def test_reserve_env_override(self, monkeypatch):
        monkeypatch.setenv("KAKEYA_GPU_MEMORY_RESERVE", str(512 * 1024**2))
        assert reserve_bytes() == 512 * 1024**2

    def test_error_includes_allocated(self, monkeypatch):
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.available_memory", lambda: 1 * 1024**3
        )
        with pytest.raises(InsufficientMemoryError) as ei:
            allocator.request(required=0, strict=True)
        assert "已分配" in str(ei.value)

    def test_error_includes_allocated_with_existing_quota(self, monkeypatch):
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.available_memory", lambda: 8 * 1024**3
        )
        q1 = allocator.request(required=3 * 1024**3)
        try:
            monkeypatch.setattr(
                "kakeya_ai.resource.probe.available_memory", lambda: 1 * 1024**3
            )
            with pytest.raises(InsufficientMemoryError) as ei:
                allocator.request(required=0, strict=True)
            assert "已分配" in str(ei.value)
            assert "3.00 GB" in str(ei.value)
        finally:
            q1.release()


class TestProbeCompatibility:
    def setup_method(self):
        reset_cache()

    def test_available_memory_is_sane(self):
        avail = available_memory()
        assert 0 < avail < 1024 * 1024**3

    def test_is_sustained_low_short_circuits_when_ample(self, monkeypatch):
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.available_memory", lambda: 8 * 1024**3
        )
        sustained, taken = is_sustained_low(
            2 * 1024**3, samples=4, interval=0.0, use_cache=False
        )
        assert sustained is False
        assert taken == 1

    def test_recent_trend_records_samples(self, monkeypatch):
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.available_memory", lambda: 5 * 1024**3
        )
        is_sustained_low(2 * 1024**3, samples=4, interval=0.0, use_cache=False)
        assert recent_trend()[-1] == 5 * 1024**3

    def test_check_gpu_headroom_and_has_gpu_headroom(self, monkeypatch):
        monkeypatch.setattr(
            "kakeya_ai.resource.probe.available_memory", lambda: 8 * 1024**3
        )
        check_gpu_headroom(0)
        assert has_gpu_headroom(0) is True
