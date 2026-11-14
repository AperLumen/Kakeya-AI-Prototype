"""两阶段快思考打分器的测试。

这些测试通过 monkeypatch `kakeya_ai.ai.ollama_client.generate` 避免真实 LLM 调用。
"""

import pytest

from kakeya_ai.ai.cache import DecisionCache
from kakeya_ai.ai.two_stage_scorer import (
    DEFAULT_DEEP_MODEL,
    DEFAULT_FAST_MODEL,
    parse_json_scores,
    two_stage_score,
)


def test_parse_json_scores_with_move_labels():
    response = 'Some reasoning. {"move_1": 3.5, "move_2": 7.0, "move_3": 4.0}'
    scores = parse_json_scores(response)
    assert scores == {1: 3.5, 2: 7.0, 3: 4.0}


def test_parse_json_scores_with_integer_keys():
    response = '{"1": 9.0, "2": 4.0, "3": 3.0}'
    scores = parse_json_scores(response)
    assert scores == {1: 9.0, 2: 4.0, 3: 3.0}


def test_parse_json_scores_missing_returns_none():
    assert parse_json_scores("No JSON here.") is None


def test_two_stage_score_heuristic_clear_winner(monkeypatch):
    """启发式有明确赢家时应当完全绕过 LLM 调用。"""
    calls = []

    def fake_generate(model, prompt, timeout, **kwargs):
        calls.append((model, prompt[:20], timeout))
        return '{"move_1": 10, "move_2": 0, "move_3": 0}'

    monkeypatch.setattr("kakeya_ai.ai.ollama_client.generate", fake_generate)

    state = {"delta": 0.01, "rho": 0.95, "local_error": 10.0, "volume_ratio": 2.0}
    scores, source = two_stage_score("prompt", state=state, ambiguity_margin=2.0)

    assert source == "heuristic"
    assert len(calls) == 0
    assert all(0 <= v <= 10 for v in scores.values())


def test_two_stage_score_fast_only(monkeypatch):
    """快模型有信心 -> 不调用深度模型。"""
    calls = []

    def fake_generate(model, prompt, timeout, **kwargs):
        calls.append(model)
        return '{"move_1": 9.0, "move_2": 4.0, "move_3": 3.0}'

    # 打补丁的是 two_stage_scorer 内部导入的那个名字。
    monkeypatch.setattr("kakeya_ai.ai.two_stage_scorer.generate", fake_generate)

    # 用一个不会产生明确启发式赢家的状态。
    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    scores, source = two_stage_score("prompt", state=state, ambiguity_margin=2.0)

    assert source == "fast"
    assert scores == {1: 9.0, 2: 4.0, 3: 3.0}
    assert calls == [DEFAULT_FAST_MODEL]


def test_two_stage_score_deep_rescore(monkeypatch):
    """快模型模棱两可 -> 深度模型对前两名 move 重新打分。"""
    calls = []

    def fake_generate(model, prompt, timeout, **kwargs):
        calls.append(model)
        # fast/deep 已统一为同一模型（qwen3.5:4b，2026-11 收敛），桩按
        # prompt 特征判别：deep 重打分调用带 "Re-score ONLY"。
        if "Re-score ONLY" in prompt:
            return '{"move_1": 8.0, "move_2": 4.0}'
        return '{"move_1": 6.0, "move_2": 5.5, "move_3": 4.0}'

    monkeypatch.setattr("kakeya_ai.ai.two_stage_scorer.generate", fake_generate)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    scores, source = two_stage_score("prompt", state=state, ambiguity_margin=2.0)

    assert source == "deep"
    assert scores[1] == 8.0
    assert scores[2] == 4.0
    assert scores[3] == 4.0
    assert calls == [DEFAULT_FAST_MODEL, DEFAULT_DEEP_MODEL]


def test_two_stage_score_cache_hit():
    """缓存的结果应被直接返回，而不调用 generate。"""
    cache = DecisionCache()
    state = {"delta": 0.01, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    cache.set(state, {"move_1": 2.0, "move_2": 7.0, "move_3": 5.0})

    scores, source = two_stage_score("prompt", cache=cache, state=state)

    assert source == "cache"
    assert scores == {1: 2.0, 2: 7.0, 3: 5.0}


def test_two_stage_score_fast_failure_fallback(monkeypatch):
    """若快模型失败且不存在启发式赢家，则回落。"""

    def fake_generate(model, prompt, timeout, **kwargs):
        raise RuntimeError("model not available")

    monkeypatch.setattr("kakeya_ai.ai.two_stage_scorer.generate", fake_generate)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    scores, source = two_stage_score("prompt", state=state)

    assert source == "heuristic"
    assert all(0 <= v <= 10 for v in scores.values())


# ---------------------------------------------------------------------------
# 失败归因必须区分"异常"与"解析失败"（2026-10-01）
#
# 与 test_cascaded_reasoning.py 同一纪律：此前 `except Exception: fast_scores = None`
# 再统一报 `reason="fast_model_or_parse_failed"`，把"网络/超时/模型没加载"与
# "返回了但解析不出 JSON"塌成一个字符串——排查时无从判断该查网络还是改 prompt。
# ---------------------------------------------------------------------------


def _capture_fallbacks(monkeypatch):
    import kakeya_ai.ai.two_stage_scorer as tss

    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        tss._obs, "record_fallback",
        lambda source_label, reason="": captured.append((source_label, reason)),
    )
    return captured


def test_fast_exception_reason_carries_exception_type(monkeypatch):
    def fake_generate(*args, **kwargs):
        raise TimeoutError("ollama busy")

    monkeypatch.setattr("kakeya_ai.ai.two_stage_scorer.generate", fake_generate)
    captured = _capture_fallbacks(monkeypatch)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    two_stage_score("prompt", state=state)

    reasons = " ".join(r for _s, r in captured)
    assert "TimeoutError" in reasons, f"归因未带异常类型：{captured}"
    assert "ollama busy" in reasons, f"归因未带异常消息：{captured}"


def test_fast_parse_failure_is_not_marked_as_exception(monkeypatch):
    def fake_generate(*args, **kwargs):
        return "no json here"  # 不抛异常，但解析不出分

    monkeypatch.setattr("kakeya_ai.ai.two_stage_scorer.generate", fake_generate)
    captured = _capture_fallbacks(monkeypatch)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    two_stage_score("prompt", state=state)

    reasons = " ".join(r for _s, r in captured)
    assert "parse_failed" in reasons, f"解析失败未被识别：{captured}"
    assert "exception" not in reasons, f"解析失败被误标成异常：{captured}"


def test_deep_parse_failure_does_not_crash(monkeypatch):
    """**deep 阶段解析失败（不抛异常）不得崩**（回归：UnboundLocalError）。

    ``generate`` 成功返回、但 ``parse_json_scores`` 解析不出（返回 ``None``）
    时**不会**进 ``except``，于是若 ``deep_failure`` 只在 ``except`` 里赋值，
    到下面拼 reason 那行就抛 ``UnboundLocalError``。这条路径此前**没有任何
    测试覆盖**（既有用例要么 deep 成功、要么 deep 抛异常），实测确认会崩。

    正确行为：回落 fast 分数，source 为 "fast"，reason 标 parse_failed。
    """
    calls = {"n": 0}

    def fake_generate(model, prompt, timeout, **kwargs):
        calls["n"] += 1
        # 第 1 次（fast）可解析；第 2 次（deep）返回不可解析内容。
        return '{"move_1": 8, "move_2": 3, "move_3": 2}' if calls["n"] == 1 else "not json"

    monkeypatch.setattr("kakeya_ai.ai.two_stage_scorer.generate", fake_generate)
    captured = _capture_fallbacks(monkeypatch)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    # ambiguity_margin 拉高 → 跳过启发式短路，强制走 fast+deep 两阶段。
    scores, source = two_stage_score(
        "prompt", state=state, ambiguity_margin=100.0,
        use_heuristic_fallback=False, fast_model="f", deep_model="d",
    )

    assert source == "fast", f"deep 解析失败应回落 fast，实际 {source}"
    assert scores == {1: 8.0, 2: 3.0, 3: 2.0}
    reasons = " ".join(r for _s, r in captured)
    assert "parse_failed" in reasons, f"deep 解析失败未被识别：{captured}"
