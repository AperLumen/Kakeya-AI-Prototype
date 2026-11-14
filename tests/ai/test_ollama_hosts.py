"""ollama_client 中 OLLAMA_HOSTS 轮询故障转移的测试（P2#9）。"""

import json
import urllib.error

import pytest

import kakeya_ai.ai.ollama_client as oc
from kakeya_ai.ai.ollama_client import (
    DEFAULT_OLLAMA_URL,
    _resolve_hosts,
    generate,
    generate_stream,
    is_server_ready,
)

HOST_A = "http://ollama-a:11434"
HOST_B = "http://ollama-b:11434"
HOST_C = "http://ollama-c:11434/api/generate"


@pytest.fixture(autouse=True)
def _reset_cursor(monkeypatch):
    monkeypatch.delenv(oc.OLLAMA_HOSTS_ENV, raising=False)
    monkeypatch.setattr(oc, "_host_cursor", 0)
    yield
    oc._host_cursor = 0


class _FakeResponse:
    """http.client.HTTPResponse 的替身（非流式）。"""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeStreamResponse:
    """流式 http.client.HTTPResponse 的替身。"""

    def __init__(self, lines):
        self._lines = [line.encode("utf-8") for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(self._lines)


def _patch_urlopen(monkeypatch, handler):
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return handler(req.full_url)

    monkeypatch.setattr(oc.urllib.request, "urlopen", fake_urlopen)
    return calls


class TestResolveHosts:
    def test_unset_env_returns_default_url(self, monkeypatch):
        monkeypatch.delenv(oc.OLLAMA_HOSTS_ENV, raising=False)
        assert _resolve_hosts(DEFAULT_OLLAMA_URL) == [DEFAULT_OLLAMA_URL]

    def test_base_urls_get_generate_suffix(self, monkeypatch):
        monkeypatch.setenv(oc.OLLAMA_HOSTS_ENV, f"{HOST_A}, {HOST_B}/")
        assert _resolve_hosts(DEFAULT_OLLAMA_URL) == [
            f"{HOST_A}/api/generate",
            f"{HOST_B}/api/generate",
        ]

    def test_full_endpoints_kept(self, monkeypatch):
        monkeypatch.setenv(oc.OLLAMA_HOSTS_ENV, f"{HOST_C}, {HOST_A}")
        assert _resolve_hosts(DEFAULT_OLLAMA_URL) == [
            HOST_C,
            f"{HOST_A}/api/generate",
        ]

    def test_empty_env_falls_back_to_url(self, monkeypatch):
        monkeypatch.setenv(oc.OLLAMA_HOSTS_ENV, " , ")
        assert _resolve_hosts(DEFAULT_OLLAMA_URL) == [DEFAULT_OLLAMA_URL]


class TestGenerateFailover:
    def test_falls_over_to_next_host(self, monkeypatch):
        monkeypatch.setenv(oc.OLLAMA_HOSTS_ENV, f"{HOST_A}, {HOST_B}")

        def handler(url):
            if url.startswith(HOST_A):
                raise urllib.error.URLError(ConnectionResetError("host a down"))
            return _FakeResponse(
                {"response": "from-b", "done": True,
                 "prompt_eval_count": 11, "eval_count": 7}
            )

        calls = _patch_urlopen(monkeypatch, handler)
        text = generate("m", "p", max_retries=0, retry_backoff=0)
        assert text == "from-b"
        assert calls == [f"{HOST_A}/api/generate", f"{HOST_B}/api/generate"]
        # 游标越过 host B（索引 1），回绕到 host A。
        assert oc._host_cursor == 0

    def test_round_robin_cursor_rotates(self, monkeypatch):
        monkeypatch.setenv(oc.OLLAMA_HOSTS_ENV, f"{HOST_A}, {HOST_B}")
        calls = _patch_urlopen(
            monkeypatch,
            lambda url: _FakeResponse({"response": "ok", "done": True}),
        )
        generate("m", "p", max_retries=0, retry_backoff=0)
        assert calls[-1] == f"{HOST_A}/api/generate"
        assert oc._host_cursor == 1
        generate("m", "p", max_retries=0, retry_backoff=0)
        assert calls[-1] == f"{HOST_B}/api/generate"
        assert oc._host_cursor == 0

    def test_all_hosts_failing_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv(oc.OLLAMA_HOSTS_ENV, f"{HOST_A}, {HOST_B}")

        def handler(url):
            raise urllib.error.URLError(ConnectionResetError("down"))

        _patch_urlopen(monkeypatch, handler)
        with pytest.raises(RuntimeError, match="after 0 retries"):
            generate("m", "p", max_retries=0, retry_backoff=0)

    def test_single_host_without_env_matches_legacy_behavior(self, monkeypatch):
        monkeypatch.delenv(oc.OLLAMA_HOSTS_ENV, raising=False)
        calls = _patch_urlopen(
            monkeypatch,
            lambda url: _FakeResponse({"response": "ok", "done": True}),
        )
        text = generate("m", "p", max_retries=0, retry_backoff=0)
        assert text == "ok"
        assert calls == [DEFAULT_OLLAMA_URL]

    def test_unreachable_first_host_retries_before_failover(self, monkeypatch):
        """先做单主机重试；只有在重试耗尽后才故障转移。"""
        monkeypatch.setenv(oc.OLLAMA_HOSTS_ENV, f"{HOST_A}, {HOST_B}")
        attempts = {"a": 0}

        def handler(url):
            if url.startswith(HOST_A):
                attempts["a"] += 1
                raise urllib.error.URLError(TimeoutError("slow"))
            return _FakeResponse({"response": "ok", "done": True})

        calls = _patch_urlopen(monkeypatch, handler)
        text = generate("m", "p", max_retries=1, retry_backoff=0)
        assert text == "ok"
        assert attempts["a"] == 2  # host A 上 1 次初始 + 1 次重试
        assert calls[-1] == f"{HOST_B}/api/generate"


class TestGenerateStreamFailover:
    def test_stream_falls_over_to_next_host(self, monkeypatch):
        monkeypatch.setenv(oc.OLLAMA_HOSTS_ENV, f"{HOST_A}, {HOST_B}")

        def handler(url):
            if url.startswith(HOST_A):
                raise urllib.error.URLError(ConnectionResetError("host a down"))
            return _FakeStreamResponse(
                ['{"response": "hi", "done": false}',
                 '{"response": "!", "done": true}']
            )

        calls = _patch_urlopen(monkeypatch, handler)
        chunks = list(generate_stream("m", "p", max_retries=0, retry_backoff=0))
        assert "".join(chunks) == "hi!"
        assert calls == [f"{HOST_A}/api/generate", f"{HOST_B}/api/generate"]


class TestIsServerReadyHosts:
    def test_ready_when_any_host_responds(self, monkeypatch):
        monkeypatch.setenv(oc.OLLAMA_HOSTS_ENV, f"{HOST_A}, {HOST_B}")

        def handler(url):
            if url.startswith(HOST_A):
                raise urllib.error.URLError(ConnectionResetError("down"))
            return _FakeResponse({})

        calls = _patch_urlopen(monkeypatch, handler)
        assert is_server_ready() is True
        assert calls == [f"{HOST_A}/api/tags", f"{HOST_B}/api/tags"]

    def test_unready_when_all_hosts_fail(self, monkeypatch):
        monkeypatch.setenv(oc.OLLAMA_HOSTS_ENV, f"{HOST_A}, {HOST_B}")

        def handler(url):
            raise urllib.error.URLError(ConnectionResetError("down"))

        _patch_urlopen(monkeypatch, handler)
        assert is_server_ready() is False
