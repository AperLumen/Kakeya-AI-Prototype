"""token 用量统计的测试。"""

from kakeya_ai.ai.token_usage import TokenUsage


def test_token_usage_starts_zero():
    u = TokenUsage()
    assert u.prompt_tokens == 0
    assert u.completion_tokens == 0
    assert u.total_tokens == 0
    assert u.calls == 0


def test_token_usage_adds_from_ollama_response():
    u = TokenUsage()
    u.add_from_ollama_response({"prompt_eval_count": 100, "eval_count": 50})
    assert u.prompt_tokens == 100
    assert u.completion_tokens == 50
    assert u.total_tokens == 150
    assert u.calls == 1


def test_token_usage_accumulates_multiple_calls():
    u = TokenUsage()
    u.add_from_ollama_response({"prompt_eval_count": 10, "eval_count": 20})
    u.add_from_ollama_response({"prompt_eval_count": 30, "eval_count": 40})
    assert u.prompt_tokens == 40
    assert u.completion_tokens == 60
    assert u.total_tokens == 100
    assert u.calls == 2


def test_token_usage_summary():
    u = TokenUsage()
    u.add_from_ollama_response({"prompt_eval_count": 5, "eval_count": 10})
    summary = u.summary()
    assert summary["prompt_tokens"] == 5
    assert summary["completion_tokens"] == 10
    assert summary["total_tokens"] == 15
    assert summary["calls"] == 1
