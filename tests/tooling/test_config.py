"""``kakeya_ai.config`` 的测试。

配置层是"部署方唯一能看到的环境变量清单"，所以测试重点不在内部实现，
而在**对外承诺的那几条语义**：

- 每个已注册变量都能被读出来，且类型正确；
- 非法值一律**回退默认值而不抛异常**（服务不该因可选配置写错而起不来）；
- 各模块既有的常量与行为**逐位不变**（本次重构是纯搬运，不是改行为）。
"""

from __future__ import annotations

import logging

import pytest

from kakeya_ai import config


class TestRegistry:
    """注册表是唯一真源：名字、类型、默认值都在一处。"""

    def test_spec_is_non_empty_and_has_unique_names(self):
        names = [spec.name for spec in config.ENV_SPEC]
        assert names, "注册表不该为空"
        assert len(names) == len(set(names)), "变量名重复会让 get() 产生歧义"

    def test_every_spec_has_a_reader(self):
        """注册表里的每种 kind 都必须有对应读取器，否则 get() 会在运行时 KeyError。"""
        for spec in config.ENV_SPEC:
            assert spec.kind in config._READERS, (
                f"{spec.name} 声明了未知类型 {spec.kind!r}"
            )

    def test_env_names_matches_spec_order(self):
        assert config.env_names() == tuple(s.name for s in config.ENV_SPEC)

    def test_known_variables_are_registered(self):
        """本次重构涉及的那些变量必须都在清单里——漏一个就等于没收敛。"""
        for name in ("OLLAMA_HOSTS", "OBS_JSONL_PATH", "CHAT_DEEP_NUM_PREDICT",
                     "KAKEYA_GPU_MEMORY_RESERVE"):
            assert name in config.env_names(), f"{name} 未登记"


class TestReaders:
    """读取器的类型转换与回退语义。"""

    def test_raw_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("OBS_JSONL_PATH", raising=False)
        assert config.raw("OBS_JSONL_PATH", "fallback") == "fallback"

    def test_raw_returns_empty_string_when_set_empty(self, monkeypatch):
        """显式设成空串与"未设置"在字符串读取器里等价（既有行为）。"""
        monkeypatch.setenv("OBS_JSONL_PATH", "")
        assert config.raw("OBS_JSONL_PATH", "fallback") == ""

    def test_as_int_parses_and_falls_back(self, monkeypatch):
        monkeypatch.setenv("CHAT_DEEP_NUM_PREDICT", " 2048 ")
        assert config.as_int("CHAT_DEEP_NUM_PREDICT") == 2048

        monkeypatch.setenv("CHAT_DEEP_NUM_PREDICT", "abc")
        assert config.as_int("CHAT_DEEP_NUM_PREDICT", 99) == 99

    def test_as_int_rejects_float_notation(self, monkeypatch):
        """as_int 刻意不接受浮点写法——该是整数的场景不该被宽容。"""
        monkeypatch.setenv("CHAT_DEEP_NUM_PREDICT", "2048.0")
        assert config.as_int("CHAT_DEEP_NUM_PREDICT", 99) == 99

    def test_as_bytes_accepts_float_notation(self, monkeypatch):
        """as_bytes 保留历史宽容度（memory_guard 原实现用 int(float(...))）。"""
        monkeypatch.setenv("KAKEYA_GPU_MEMORY_RESERVE", "2e9")
        assert config.as_bytes("KAKEYA_GPU_MEMORY_RESERVE") == 2_000_000_000

        monkeypatch.setenv("KAKEYA_GPU_MEMORY_RESERVE", "2147483648")
        assert config.as_bytes("KAKEYA_GPU_MEMORY_RESERVE") == 2147483648

    def test_as_bytes_falls_back(self, monkeypatch):
        monkeypatch.setenv("KAKEYA_GPU_MEMORY_RESERVE", "not-a-number")
        assert config.as_bytes("KAKEYA_GPU_MEMORY_RESERVE", 123) == 123

    def test_as_csv_strips_and_drops_empties(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOSTS", "http://a:11434, http://b:11434,,")
        assert config.as_csv("OLLAMA_HOSTS") == ["http://a:11434", "http://b:11434"]

    def test_as_csv_all_blank_falls_back_to_default(self, monkeypatch):
        """**纯**空白视为"未设置"，回退给定默认值。"""
        monkeypatch.setenv("OLLAMA_HOSTS", "   ")
        assert config.as_csv("OLLAMA_HOSTS", ["default"]) == ["default"]

    def test_as_csv_only_separators_yields_empty_list(self, monkeypatch):
        """只有分隔符时（非空白）不触发回退，过滤后就是空列表。

        调用方需要自己处理"键存在但解析为空"这一情形——``ollama_client``
        即把空列表当作"未配置"回退到单 host（见下方模块接线测试）。
        """
        monkeypatch.setenv("OLLAMA_HOSTS", " , ")
        assert config.as_csv("OLLAMA_HOSTS", ["default"]) == []

    @pytest.mark.parametrize("value,expected", [
        ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
        ("0", False), ("false", False), ("no", False), ("off", False),
        ("anything", False),
    ])
    def test_flag_truthy_values(self, monkeypatch, value, expected):
        monkeypatch.setenv("KAKEYA_TEST_FLAG", value)
        assert config.flag("KAKEYA_TEST_FLAG") is expected

    def test_flag_unset_uses_default(self, monkeypatch):
        monkeypatch.delenv("KAKEYA_TEST_FLAG", raising=False)
        assert config.flag("KAKEYA_TEST_FLAG", default=True) is True
        assert config.flag("KAKEYA_TEST_FLAG", default=False) is False

    def test_flag_empty_uses_default(self, monkeypatch):
        monkeypatch.setenv("KAKEYA_TEST_FLAG", "  ")
        assert config.flag("KAKEYA_TEST_FLAG", default=True) is True


class TestBadValuesAreLoggedNotFatal:
    """非法值必须留下痕迹——"改了没生效"要能在日志里查到。"""

    def test_invalid_int_logs_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("CHAT_DEEP_NUM_PREDICT", "xyz")
        with caplog.at_level(logging.WARNING, logger="kakeya_ai.config"):
            config.as_int("CHAT_DEEP_NUM_PREDICT", 1)
        assert any("CHAT_DEEP_NUM_PREDICT" in r.message for r in caplog.records)

    def test_invalid_bytes_logs_warning(self, monkeypatch, caplog):
        monkeypatch.setenv("KAKEYA_GPU_MEMORY_RESERVE", "xyz")
        with caplog.at_level(logging.WARNING, logger="kakeya_ai.config"):
            config.as_bytes("KAKEYA_GPU_MEMORY_RESERVE", 1)
        assert any("KAKEYA_GPU_MEMORY_RESERVE" in r.message for r in caplog.records)


class TestGetAndSnapshot:
    def test_get_uses_spec_type_and_default(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOSTS", raising=False)
        assert config.get("OLLAMA_HOSTS") == []

    def test_get_rejects_unregistered_name(self):
        """未登记的名字必须报错，而不是静默给空值。"""
        with pytest.raises(KeyError, match="未注册"):
            config.get("SOME_RANDOM_VAR")

    def test_snapshot_covers_every_registered_var(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_HOSTS", raising=False)
        snap = config.snapshot()
        assert set(snap.as_dict()) == set(config.env_names())

    def test_snapshot_marks_overridden_keys(self, monkeypatch):
        monkeypatch.setenv("OBS_JSONL_PATH", "/tmp/obs-test")
        monkeypatch.delenv("OLLAMA_HOSTS", raising=False)
        snap = config.snapshot()
        assert "OBS_JSONL_PATH" in snap.overridden
        assert "OLLAMA_HOSTS" not in snap.overridden

    def test_describe_env_lists_all_names(self):
        text = config.describe_env()
        for name in config.env_names():
            assert name in text


class TestModuleWiringUnchanged:
    """各模块的取值行为必须与重构前逐位一致（本次是纯搬运）。"""

    def test_memory_guard_defaults_to_2gb(self, monkeypatch):
        from kakeya_ai import memory_guard as mg
        monkeypatch.delenv("KAKEYA_GPU_MEMORY_RESERVE", raising=False)
        assert mg.reserve_bytes() == mg.DEFAULT_RESERVE_BYTES == 2 * 1024**3

    def test_memory_guard_env_override(self, monkeypatch):
        from kakeya_ai import memory_guard as mg
        monkeypatch.setenv("KAKEYA_GPU_MEMORY_RESERVE", str(512 * 1024**2))
        assert mg.reserve_bytes() == 512 * 1024**2

    def test_memory_guard_bad_env_falls_back(self, monkeypatch):
        from kakeya_ai import memory_guard as mg
        monkeypatch.setenv("KAKEYA_GPU_MEMORY_RESERVE", "not-a-number")
        assert mg.reserve_bytes() == mg.DEFAULT_RESERVE_BYTES

    def test_ollama_hosts_unset_is_single_host(self, monkeypatch):
        from kakeya_ai.ai import ollama_client as oc
        monkeypatch.delenv(oc.OLLAMA_HOSTS_ENV, raising=False)
        assert oc._resolve_hosts("http://x:11434/api/generate") == [
            "http://x:11434/api/generate"
        ]

    def test_ollama_hosts_normalizes_entries(self, monkeypatch):
        from kakeya_ai.ai import ollama_client as oc
        monkeypatch.setenv(oc.OLLAMA_HOSTS_ENV, "http://a:11434, http://b:11434/")
        assert oc._resolve_hosts("http://ignored") == [
            "http://a:11434/api/generate",
            "http://b:11434/api/generate",
        ]

    def test_ollama_hosts_blank_falls_back(self, monkeypatch):
        """全空白等于未设置：回退单 host（既有行为，见 test_ollama_hosts.py）。"""
        from kakeya_ai.ai import ollama_client as oc
        monkeypatch.setenv(oc.OLLAMA_HOSTS_ENV, " , ")
        assert oc._resolve_hosts("http://only:11434/api/generate") == [
            "http://only:11434/api/generate"
        ]

    def test_obs_sink_disabled_when_unset(self, monkeypatch):
        from kakeya_ai.ai import observability
        monkeypatch.delenv(observability.OBS_JSONL_PATH_ENV, raising=False)
        assert observability._get_sink() is None
