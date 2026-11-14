"""AI 推理可观测层的测试。"""

import pytest

from kakeya_ai.ai.observability import (
    LLMObservability,
    observe_llm_call,
)


def test_metrics_accumulate():
    obs = LLMObservability()
    obs.record_llm_call("qwen3:4b", prompt_len=100, duration_ms=50.0, success=True, prompt_tokens=30, completion_tokens=20)
    obs.record_llm_call("qwen3:4b", prompt_len=200, duration_ms=60.0, success=True)
    obs.record_llm_call("deepseek-r1:7b", prompt_len=500, duration_ms=200.0, success=False)

    summary = obs.summary()
    assert summary["llm_calls"]["qwen3:4b"]["success"] == 2
    assert summary["llm_calls"]["qwen3:4b"]["failure"] == 0
    assert summary["llm_calls"]["deepseek-r1:7b"]["failure"] == 1

    qwen_duration = summary["llm_duration_ms"]["qwen3:4b"]
    assert qwen_duration["count"] == 2
    assert qwen_duration["total_ms"] == pytest.approx(110.0)
    assert qwen_duration["mean_ms"] == pytest.approx(55.0)
    assert qwen_duration["prompt_tokens_total"] == 30
    assert qwen_duration["completion_tokens_total"] == 20
    assert qwen_duration["prompt_len_total"] == 300


def test_parse_failure_recorded():
    obs = LLMObservability()
    long_response = "x" * 500
    obs.record_parse_failure(stage="parse_json_scores", response_snippet=long_response, reason="no_json_match")

    summary = obs.summary()
    assert summary["parse_failures"]["parse_json_scores"] == 1
    assert len(obs.recent_parse_failures) == 1
    assert len(obs.recent_parse_failures[0]["snippet"]) <= 200


def test_recent_parse_failures_bounded():
    obs = LLMObservability(max_recent_failures=3)
    for i in range(5):
        obs.record_parse_failure(stage="stage", response_snippet=str(i), reason="reason")
    assert len(obs.recent_parse_failures) == 3
    assert obs.recent_parse_failures[0]["snippet"] == "2"
    assert obs.recent_parse_failures[-1]["snippet"] == "4"


def test_llm_call_timing():
    obs = LLMObservability()
    obs.record_llm_call("model", prompt_len=10, duration_ms=10.0, success=True)
    duration = obs.summary()["llm_duration_ms"]["model"]
    assert duration["count"] == 1
    assert duration["mean_ms"] == pytest.approx(10.0)
    assert duration["std_ms"] == pytest.approx(0.0)


def test_decorator_records_success_and_failure():
    obs = LLMObservability()

    @observe_llm_call(observability=obs)
    def good_call(model: str, prompt: str) -> str:
        return "ok"

    @observe_llm_call(observability=obs)
    def bad_call(model: str, prompt: str) -> str:
        raise ValueError("boom")

    assert good_call("qwen3:4b", "hello") == "ok"
    with pytest.raises(ValueError):
        bad_call("deepseek-r1:7b", "world")

    summary = obs.summary()
    assert summary["llm_calls"]["qwen3:4b"]["success"] == 1
    assert summary["llm_calls"]["deepseek-r1:7b"]["failure"] == 1
    assert summary["llm_duration_ms"]["qwen3:4b"]["count"] == 1
    assert summary["llm_duration_ms"]["deepseek-r1:7b"]["count"] == 1


def test_decorator_records_generator():
    obs = LLMObservability()

    @observe_llm_call(observability=obs)
    def stream_call(model: str, prompt: str):
        yield "chunk1"
        yield "chunk2"

    chunks = list(stream_call("qwen3:4b", "prompt"))
    assert chunks == ["chunk1", "chunk2"]
    summary = obs.summary()
    assert summary["llm_calls"]["qwen3:4b"]["success"] == 1


def test_fallback_and_decision_recorded():
    obs = LLMObservability()
    obs.record_fallback("heuristic", reason="fast_model_failed")
    obs.record_decision("heuristic", state_key="key1")
    obs.record_decision("cache", state_key="key2")

    summary = obs.summary()
    assert summary["fallbacks"]["heuristic"] == 1
    assert summary["decisions"]["heuristic"] == 1
    assert summary["decisions"]["cache"] == 1


def test_reset():
    obs = LLMObservability()
    obs.record_llm_call("m", prompt_len=1, duration_ms=1.0, success=True)
    obs.record_parse_failure("s", "x")
    obs.reset()
    summary = obs.summary()
    assert summary["llm_calls"] == {}
    assert summary["parse_failures"] == {}
    assert summary["recent_parse_failures"] == []
