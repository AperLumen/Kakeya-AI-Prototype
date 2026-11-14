"""beam-search 打分的测试：快模型产出候选 + 7B 剪枝。"""

import json

import pytest

from kakeya_ai.ai.beam_search import (
    DEFAULT_DEEP_MODEL,
    DEFAULT_FAST_MODEL,
    Candidate,
    _parse_deep_evaluation,
    _parse_self_scores,
    beam_search_score,
    format_deep_beam_evaluation_prompt,
    format_fast_beam_prompt,
    format_self_scoring_prompt,
    parse_candidate_reasonings,
)
from kakeya_ai.ai.cache import DecisionCache
from kakeya_ai.ai.reasoning_trace import ReasoningTraceStore


BEAM_RESPONSE = """\
Candidate 1: Rho is close to 1 so grains are too square; reducing rho is the priority. ### move_2 ### 8
Candidate 2: Local convex Wolff error is high, so widening grains should help. ### move_3 ### 7
Candidate 3: The volume ratio is below the bound, so lengthening grains is safe. ### move_1 ### 5
Candidate 4: Uncertainty is high but move 2 still seems slightly better. ### move_2 ### 4
"""

DEEP_RESPONSE = '{"move_1": 4, "move_2": 9, "move_3": 5, "candidate_quality": 8}'
SELF_SCORES_RESPONSE = '{"scores": [8, 6, 5, 4]}'


def test_parse_candidate_reasonings_extracts_candidates():
    candidates = parse_candidate_reasonings(BEAM_RESPONSE, beam_width=4)
    assert len(candidates) == 4
    assert candidates[0].recommended_move == 2
    assert candidates[0].confidence == 8.0
    assert "square" in candidates[0].reasoning


def test_format_fast_beam_prompt_includes_candidate_format():
    prompt = format_fast_beam_prompt("BASE STATE", beam_width=3)
    assert "BASE STATE" in prompt
    assert "Generate exactly 3 distinct candidate reasoning directions" in prompt
    assert "Candidate i:" in prompt
    assert "###" in prompt


def test_format_self_scoring_prompt_lists_candidates():
    candidates = parse_candidate_reasonings(BEAM_RESPONSE, beam_width=4)
    prompt = format_self_scoring_prompt("BASE STATE", candidates)
    assert "Score each candidate" in prompt
    assert candidates[0].reasoning in prompt
    assert '"scores":' in prompt


def test_format_deep_beam_evaluation_prompt_includes_anchors():
    candidate = Candidate(reasoning="Rho is high.", recommended_move=2, confidence=8, index=1)
    anchors = [("Previous: move_2 was chosen because rho was large.", 0.8)]
    prompt = format_deep_beam_evaluation_prompt("BASE STATE", candidate, anchors)
    assert "BASE STATE" in prompt
    assert "<candidate_reasoning>" in prompt
    assert "Previous:" in prompt
    assert '"candidate_quality"' in prompt


def _make_fake_generate(beam_response, deep_response, self_response=None):
    calls = []

    def fake_generate(model, prompt, timeout, **kwargs):
        calls.append((model, prompt))
        # fast/deep 已统一为同一模型（qwen3.5:4b，2026-11 收敛），桩按
        # prompt 特征判别角色：自评分带 "Score each candidate"，候选生成带
        # "Generate exactly"，深评带 "Candidate reasoning to evaluate"。
        if "Score each candidate" in prompt:
            return self_response or '{"scores": [8, 7, 6, 5]}'
        if "Generate exactly" in prompt:
            return beam_response
        if "Candidate reasoning to evaluate" in prompt:
            return deep_response
        return ""

    return fake_generate, calls


def test_beam_search_score_uses_cache():
    cache = DecisionCache()
    state = {"delta": 0.01, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    cache.set(state, {"move_1": 2.0, "move_2": 7.0, "move_3": 5.0})

    scores, source = beam_search_score("prompt", cache=cache, state=state)
    assert source == "cache"
    assert scores == {1: 2.0, 2: 7.0, 3: 5.0}


def test_beam_search_score_heuristic_clear_winner(monkeypatch):
    calls = []

    def fake_generate(model, prompt, timeout, **kwargs):
        calls.append(model)
        return BEAM_RESPONSE

    monkeypatch.setattr("kakeya_ai.ai.beam_search.generate", fake_generate)

    # 高 rho 使 move_2 成为明确的启发式赢家。
    state = {"delta": 0.01, "rho": 0.95, "local_error": 0.0, "volume_ratio": 0.5}
    scores, source = beam_search_score("prompt", state=state, ambiguity_margin=2.0)

    assert source == "heuristic"
    assert len(calls) == 0
    assert all(0 <= v <= 10 for v in scores.values())


def test_beam_search_parallel_deep_eval_matches_serial(monkeypatch):
    """max_workers>1 时 deep 评估并发执行，结果与串行完全一致。

    钉住两点（2026-11-06 引入并发）：①确实并发（max in-flight == 3）；
    ②聚合结果与串行路径相同（保序 map + 同一 _evaluate 语义）。
    """
    import threading
    import time

    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    def fake_generate(model, prompt, timeout, **kwargs):
        nonlocal in_flight, max_in_flight
        if "Generate exactly" in prompt:
            return BEAM_RESPONSE
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.05)
        with lock:
            in_flight -= 1
        return DEEP_RESPONSE

    monkeypatch.setattr("kakeya_ai.ai.beam_search.generate", fake_generate)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    scores, source = beam_search_score(
        "prompt",
        state=state,
        beam_width=4,
        deep_eval_top=3,
        max_workers=3,
    )

    assert source == "beam"
    assert scores == {1: 4.0, 2: 9.0, 3: 5.0}
    assert max_in_flight == 3, f"并发未发生：max_in_flight={max_in_flight}"


def test_beam_search_score_deep_mode(monkeypatch):
    fake_generate, calls = _make_fake_generate(BEAM_RESPONSE, DEEP_RESPONSE)
    monkeypatch.setattr("kakeya_ai.ai.beam_search.generate", fake_generate)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    scores, source = beam_search_score(
        "prompt",
        state=state,
        beam_width=4,
        deep_eval_top=2,
        scoring_mode="deep",
    )

    assert source == "beam"
    assert scores == {1: 4.0, 2: 9.0, 3: 5.0}

    deep_calls = [p for _, p in calls if "Candidate reasoning to evaluate" in p]
    assert len(deep_calls) == 2


def test_beam_search_score_self_mode(monkeypatch):
    fake_generate, calls = _make_fake_generate(BEAM_RESPONSE, DEEP_RESPONSE, SELF_SCORES_RESPONSE)
    monkeypatch.setattr("kakeya_ai.ai.beam_search.generate", fake_generate)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    scores, source = beam_search_score(
        "prompt",
        state=state,
        beam_width=4,
        deep_eval_top=2,
        scoring_mode="self",
    )

    assert source == "beam"
    # 深度模型被 mock 成对它看到的每个候选都返回相同分数。
    assert scores == {1: 4.0, 2: 9.0, 3: 5.0}

    # fast/deep 已统一为同一模型，计数同样按 prompt 特征分角色。
    fast_calls = [p for _, p in calls if "Generate exactly" in p or "Score each candidate" in p]
    assert len(fast_calls) == 2  # 候选生成 + 自评分
    deep_calls = [p for _, p in calls if "Candidate reasoning to evaluate" in p]
    assert len(deep_calls) == 2


def test_beam_search_score_stores_trace(monkeypatch):
    fake_generate, _ = _make_fake_generate(BEAM_RESPONSE, DEEP_RESPONSE)
    monkeypatch.setattr("kakeya_ai.ai.beam_search.generate", fake_generate)

    store = ReasoningTraceStore()
    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    beam_search_score(
        "prompt",
        state=state,
        beam_width=4,
        deep_eval_top=2,
        trace_store=store,
    )

    assert store.size == 1
    anchors = store.get_anchors("rho square grain move_2", top_k=1)
    assert len(anchors) == 1


def test_beam_search_score_fast_failure_fallback(monkeypatch):
    def fake_generate(model, prompt, timeout, **kwargs):
        raise RuntimeError("fast model down")

    monkeypatch.setattr("kakeya_ai.ai.beam_search.generate", fake_generate)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    scores, source = beam_search_score("prompt", state=state)

    assert source == "heuristic"
    assert all(0 <= v <= 10 for v in scores.values())


def test_beam_search_dynamic_budget_uses_confidence(monkeypatch):
    beam = """\
Candidate 1: High confidence reasoning. ### move_2 ### 9
Candidate 2: Low confidence reasoning. ### move_3 ### 3
"""
    calls = []

    def fake_generate(model, prompt, timeout, **kwargs):
        calls.append((model, prompt, kwargs.get("options", {}).get("num_predict")))
        # fast/deep 已统一为同一模型，按 prompt 特征判别（候选生成带
        # "Generate exactly"）。
        if "Generate exactly" in prompt:
            return beam
        return DEEP_RESPONSE

    monkeypatch.setattr("kakeya_ai.ai.beam_search.generate", fake_generate)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    beam_search_score(
        "prompt",
        state=state,
        beam_width=2,
        deep_eval_top=2,
        dynamic_budget=True,
    )

    deep_predicts = [n for _, p, n in calls if "Candidate reasoning to evaluate" in p]
    assert len(deep_predicts) == 2
    # 置信度更高的候选应获得更小的 token 预算。
    assert deep_predicts[0] < deep_predicts[1]


def test_beam_search_dynamic_budget_can_be_disabled(monkeypatch):
    beam = """\
Candidate 1: High confidence reasoning. ### move_2 ### 9
Candidate 2: Low confidence reasoning. ### move_3 ### 3
"""
    calls = []

    def fake_generate(model, prompt, timeout, **kwargs):
        calls.append((model, prompt, kwargs.get("options", {}).get("num_predict")))
        # 同上：按 prompt 特征判别（候选生成带 "Generate exactly"）。
        if "Generate exactly" in prompt:
            return beam
        return DEEP_RESPONSE

    monkeypatch.setattr("kakeya_ai.ai.beam_search.generate", fake_generate)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    beam_search_score(
        "prompt",
        state=state,
        beam_width=2,
        deep_eval_top=2,
        dynamic_budget=False,
        deep_options={"num_predict": 800},
    )

    deep_predicts = [n for _, p, n in calls if "Candidate reasoning to evaluate" in p]
    assert len(deep_predicts) == 2
    assert deep_predicts[0] == deep_predicts[1] == 800


def test_parse_self_scores_falls_back_to_bracket_list():
    # JSON 格式错误，但存在一个方括号列表。
    response = "some text [8, 6, 5, 4] more text"
    scores = _parse_self_scores(response, expected=4)
    assert scores == [8.0, 6.0, 5.0, 4.0]


def test_parse_self_scores_returns_none_when_unparseable():
    response = "no scores here at all"
    assert _parse_self_scores(response, expected=4) is None


def test_parse_self_scores_logs_fallback_failure(caplog):
    # JSON 无效，且方括号列表含非数值数据。
    response = "{invalid} then [a, b]"
    with caplog.at_level("WARNING", logger="kakeya_ai.ai.beam_search"):
        assert _parse_self_scores(response, expected=4) is None
    assert any("self-scores" in rec.message.lower() for rec in caplog.records)


def test_parse_deep_evaluation_extracts_quality_and_scores():
    response = '{"move_1": 4, "move_2": 9, "move_3": 5, "candidate_quality": 8}'
    scores, quality = _parse_deep_evaluation(response)
    assert scores == {1: 4.0, 2: 9.0, 3: 5.0}
    assert quality == 8.0


def test_parse_deep_evaluation_logs_quality_failure(caplog):
    # move 分数有效，但 candidate_quality 不是数值。
    response = '{"move_1": 4, "move_2": 9, "move_3": 5, "candidate_quality": "high"}'
    with caplog.at_level("WARNING", logger="kakeya_ai.ai.beam_search"):
        scores, quality = _parse_deep_evaluation(response)
    assert scores == {1: 4.0, 2: 9.0, 3: 5.0}
    assert quality == 5.0  # 默认值
    assert any("candidate_quality" in rec.message.lower() for rec in caplog.records)


def test_beam_search_deep_unparseable_falls_back_to_fast(monkeypatch):
    fake_generate, _ = _make_fake_generate(BEAM_RESPONSE, "not valid json")
    monkeypatch.setattr("kakeya_ai.ai.beam_search.generate", fake_generate)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    scores, source = beam_search_score(
        "prompt",
        state=state,
        beam_width=4,
        deep_eval_top=2,
        scoring_mode="deep",
    )

    assert source == "fast"
    assert all(0 <= v <= 10 for v in scores.values())
