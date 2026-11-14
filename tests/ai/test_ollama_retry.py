"""Ollama 客户端与两阶段打分器中重试/超时处理的测试。"""

import json

import pytest

import kakeya_ai.ai.ollama_client as ollama_client
from kakeya_ai.ai.ollama_client import _should_retry, generate
from kakeya_ai.errors import InputValidationError
from kakeya_ai.ai.two_stage_scorer import two_stage_score


class _FakeResponse:
    """``http.client.HTTPResponse`` 的最小替身。"""

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, behavior):
    """用一个带计数的假实现替换 urlopen，其行为由 ``behavior`` 决定。"""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return behavior(calls["n"])

    monkeypatch.setattr(ollama_client.urllib.request, "urlopen", fake_urlopen)
    return calls


class TestShouldRetry:
    def test_urlerror_without_reason_is_retryable(self):
        import urllib.error

        exc = urllib.error.URLError("boom")
        # ``URLError.reason`` 在存根里是只读属性（实际可写）。本用例刻意把
        # 它置为 None 以测"无 reason 时仍可重试"——那是负向用例的固有摩擦。
        exc.reason = None  # type: ignore[misc]  # 抹掉 __init__ 设的那个字符串 reason
        assert _should_retry(exc) is True

    def test_urlerror_with_transient_reason_is_retryable(self):
        import urllib.error

        exc = urllib.error.URLError(TimeoutError("timed out"))
        assert _should_retry(exc) is True

    def test_bare_transient_exceptions_are_retryable(self):
        assert _should_retry(TimeoutError("t")) is True
        assert _should_retry(ConnectionResetError("r")) is True
        assert _should_retry(BrokenPipeError("b")) is True

    def test_other_exceptions_are_not_retryable(self):
        assert _should_retry(ValueError("nope")) is False
        assert _should_retry(RuntimeError("nope")) is False


class TestGenerateRetries:
    def test_recovers_after_transient_failures(self, monkeypatch):
        def behavior(n):
            if n <= 2:
                raise ConnectionResetError(f"reset {n}")
            return _FakeResponse({"response": "hello", "done": True})

        calls = _patch_urlopen(monkeypatch, behavior)
        text = generate("m", "p", retry_backoff=0)
        assert text == "hello"
        assert calls["n"] == 3

    def test_exhausts_retries_and_raises_runtime_error(self, monkeypatch):
        def behavior(n):
            raise TimeoutError(f"timed out {n}")

        calls = _patch_urlopen(monkeypatch, behavior)
        with pytest.raises(RuntimeError, match="after 2 retries"):
            generate("m", "p", retry_backoff=0)
        assert calls["n"] == 3  # 1 次初始尝试 + 2 次重试

    def test_non_retryable_exception_raises_immediately(self, monkeypatch):
        def behavior(n):
            raise ValueError("bad payload")

        calls = _patch_urlopen(monkeypatch, behavior)
        with pytest.raises(ValueError, match="bad payload"):
            generate("m", "p", retry_backoff=0)
        assert calls["n"] == 1


class TestNegativeMaxRetries:
    """``max_retries < 0`` 时重试循环一次都不执行——不能静默返回 None。

    回归：``_generate_with_retries`` / ``_ollama_api_request`` 的唯一返回点在
    循环体内，循环不执行时会**落空返回 None**。调用方拿 None 当 dict 用，
    报错位置离根因很远（表现为 AttributeError/TypeError，看不出是参数问题）。
    现改为显式抛出说明性错误。
    """

    def test_generate_with_retries_rejects_negative(self):
        with pytest.raises(RuntimeError, match="max_retries=-1"):
            ollama_client._generate_with_retries(
                "m", "p", "http://127.0.0.1:1/api/generate", 0.01, "5m",
                None, -1, 0.01,
            )

    def test_ollama_api_request_rejects_negative(self):
        with pytest.raises(RuntimeError, match="max_retries=-1"):
            ollama_client._ollama_api_request(
                "m", "p", "http://127.0.0.1:1/api/generate",
                timeout=0.01, max_retries=-1,
            )

    def test_never_returns_none_for_negative_retries(self):
        """两个函数的返回值都不该是 None——这是本项要防的具体后果。"""
        for call in (
            lambda: ollama_client._generate_with_retries(
                "m", "p", "http://127.0.0.1:1/api/generate", 0.01, "5m",
                None, -1, 0.01,
            ),
            lambda: ollama_client._ollama_api_request(
                "m", "p", "http://127.0.0.1:1/api/generate",
                timeout=0.01, max_retries=-1,
            ),
        ):
            try:
                result = call()
            except RuntimeError:
                continue
            raise AssertionError(f"应抛 RuntimeError，实际返回 {result!r}")


class TestTwoStageScoreRetryFallback:
    def test_fast_model_failure_falls_back_to_heuristic(self, monkeypatch):
        """当 generate 持续失败（即便有重试）时，回落。"""

        def fake_generate(*args, **kwargs):
            raise ConnectionResetError("down")

        monkeypatch.setattr(
            "kakeya_ai.ai.two_stage_scorer.generate", fake_generate
        )
        # 与 test_two_stage_score_fast_failure_fallback 同样的情形：没有明确的
        # 启发式赢家，因此会尝试调用快模型（然后失败）。
        state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
        scores, source = two_stage_score("prompt", state=state)

        assert source == "heuristic"
        assert all(0 <= v <= 10 for v in scores.values())

    def test_retries_kwargs_are_threaded_to_generate(self, monkeypatch):
        seen = {}

        def fake_generate(model, prompt, **kwargs):
            seen["max_retries"] = kwargs.get("max_retries")
            raise ConnectionResetError("down")

        monkeypatch.setattr(
            "kakeya_ai.ai.two_stage_scorer.generate", fake_generate
        )
        state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
        scores, source = two_stage_score(
            "prompt",
            state=state,
            fast_retries=5,
            deep_retries=7,
        )
        assert source == "heuristic"
        assert seen["max_retries"] == 5


class TestThinkPassthrough:
    """``think`` 参数：None 默认不写字段（行为不变）；给了才写进 payload。"""

    def _capture(self, monkeypatch):
        captured = {}

        class _Resp:
            def __init__(self):
                self._b = json.dumps({"response": "ok"}).encode()

            def read(self):
                return self._b

            def __enter__(self):
                return self

            def __exit__(self, *e):
                return False

        def fake_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _Resp()

        monkeypatch.setattr(ollama_client.urllib.request, "urlopen", fake_urlopen)
        return captured

    def test_think_none_omits_field(self, monkeypatch):
        cap = self._capture(monkeypatch)
        generate("m", "p")
        assert "think" not in cap["payload"], "默认不得引入新字段（向后兼容）"

    def test_think_false_is_sent(self, monkeypatch):
        cap = self._capture(monkeypatch)
        generate("m", "p", think=False)
        assert cap["payload"]["think"] is False

    def test_think_true_is_sent(self, monkeypatch):
        cap = self._capture(monkeypatch)
        generate("m", "p", think=True)
        assert cap["payload"]["think"] is True


class TestFormatPassthrough:
    """``format`` 参数（约束解码）：None 默认不写字段；给了就写进 payload。"""

    def _capture(self, monkeypatch):
        captured = {}

        class _Resp:
            def __init__(self):
                self._b = json.dumps({"response": "ok"}).encode()

            def read(self):
                return self._b

            def __enter__(self):
                return self

            def __exit__(self, *e):
                return False

        def fake_urlopen(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _Resp()

        monkeypatch.setattr(ollama_client.urllib.request, "urlopen", fake_urlopen)
        return captured

    def test_format_none_omits_field(self, monkeypatch):
        cap = self._capture(monkeypatch)
        generate("m", "p")
        assert "format" not in cap["payload"], "默认不得引入新字段（向后兼容）"

    def test_format_dict_is_sent_verbatim(self, monkeypatch):
        cap = self._capture(monkeypatch)
        schema = {"type": "object", "properties": {"purpose": {"enum": ["a"]}}}
        generate("m", "p", format=schema)
        assert cap["payload"]["format"] == schema

    def test_format_json_string_is_sent(self, monkeypatch):
        cap = self._capture(monkeypatch)
        generate("m", "p", format="json")
        assert cap["payload"]["format"] == "json"

    def test_gbnf_string_is_rejected_before_the_request(self, monkeypatch):
        """**回归（2026-10-10）**：本测试此前断言 GBNF 会被发出去——那是**错的**假设。

        Ollama **不收 GBNF**：实测 0.34.4 对 ``format=<GBNF>`` 回
        ``HTTP 500 invalid format: ...; expected "json" or a valid JSON Schema object``。
        更要命的是当时没人在发请求前校验，而解码层有 ``except Exception`` 兜底 ——
        于是"文法写错"表现成"模型没选"，故障完全静默（实测踩过）。

        现在应当**当场**抛错，且**一个请求都不发**。
        """
        cap = self._capture(monkeypatch)
        with pytest.raises(InputValidationError):
            generate("m", "p", format='root ::= "yes" | "no"')
        assert "payload" not in cap, "校验必须发生在发请求之前"
