"""级联式 fast → 7B 推理的测试。

通过 monkeypatch `cascaded_reasoning` 模块内的 `generate` 函数来避免真实 LLM 调用。
"""

from kakeya_ai.ai.cache import DecisionCache
from kakeya_ai.ai.cascaded_reasoning import (
    DEFAULT_DEEP_MODEL,
    DEFAULT_FAST_MODEL,
    cascaded_score,
    format_deep_continuation_prompt,
    format_fast_thinker_prompt,
)


def test_format_fast_thinker_prompt_includes_reasoning_instruction():
    prompt = format_fast_thinker_prompt("BASE STATE")
    assert "BASE STATE" in prompt
    assert "Do not output JSON yet" in prompt


def test_format_deep_continuation_prompt_includes_fast_reasoning():
    prompt = format_deep_continuation_prompt("BASE STATE", "move 2 looks good")
    assert "BASE STATE" in prompt
    assert "move 2 looks good" in prompt
    assert "<fast_thinking>" in prompt
    assert '{"move_1": <0-10 score>' in prompt


def test_cascaded_score_heuristic_clear_winner(monkeypatch):
    calls = []

    def fake_generate(model, prompt, timeout, **kwargs):
        calls.append(model)
        return '{"move_1": 10, "move_2": 0, "move_3": 0}'

    monkeypatch.setattr("kakeya_ai.ai.cascaded_reasoning.generate", fake_generate)

    state = {"delta": 0.01, "rho": 0.95, "local_error": 0.0, "volume_ratio": 0.5}
    scores, source = cascaded_score("prompt", state=state, ambiguity_margin=2.0)

    assert source == "heuristic"
    assert len(calls) == 0
    assert all(0 <= v <= 10 for v in scores.values())


def test_cascaded_score_fast_failure_fallback(monkeypatch):
    def fake_generate(model, prompt, timeout, **kwargs):
        raise RuntimeError("fast model down")

    monkeypatch.setattr("kakeya_ai.ai.cascaded_reasoning.generate", fake_generate)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    scores, source = cascaded_score("prompt", state=state)

    assert source == "heuristic"
    assert all(0 <= v <= 10 for v in scores.values())


def test_cascaded_score_continues_reasoning(monkeypatch):
    calls = []

    def fake_generate(model, prompt, timeout, **kwargs):
        calls.append((model, "reasoning" in prompt.lower(), "json" in prompt.lower()))
        # fast/deep 已统一为同一模型（qwen3.5:4b，2026-11 收敛），桩不能
        # 再按模型名判别——改按 prompt 特征：fast 计划调用带
        # "Do not output JSON yet"，deep 接续调用内嵌 <fast_thinking>。
        if "Do not output JSON yet" in prompt:
            return "Move 2 looks most promising because rho is moderate."
        return '{"move_1": 4, "move_2": 8, "move_3": 3}'

    monkeypatch.setattr("kakeya_ai.ai.cascaded_reasoning.generate", fake_generate)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    scores, source = cascaded_score("prompt", state=state, always_use_deep=True)

    assert source == "cascaded"
    assert scores == {1: 4.0, 2: 8.0, 3: 3.0}
    assert len(calls) == 2
    assert calls[0][0] == DEFAULT_FAST_MODEL
    assert calls[1][0] == DEFAULT_DEEP_MODEL


def test_cascaded_score_cache_hit():
    cache = DecisionCache()
    state = {"delta": 0.01, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    cache.set(state, {"move_1": 2.0, "move_2": 7.0, "move_3": 5.0})

    scores, source = cascaded_score("prompt", cache=cache, state=state)

    assert source == "cache"
    assert scores == {1: 2.0, 2: 7.0, 3: 5.0}


def test_cascaded_score_deep_failure_uses_fast(monkeypatch):
    def fake_generate(model, prompt, timeout, **kwargs):
        # 同上：按 prompt 特征判别（fast 的提示词里含「先别输出 JSON」这句）。
        if "Do not output JSON yet" in prompt:
            return '{"move_1": 3, "move_2": 7, "move_3": 4}'
        raise RuntimeError("deep model down")

    monkeypatch.setattr("kakeya_ai.ai.cascaded_reasoning.generate", fake_generate)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    scores, source = cascaded_score("prompt", state=state, always_use_deep=True)

    assert source == "fast"
    assert scores == {1: 3.0, 2: 7.0, 3: 4.0}


def test_dynamic_budget_does_not_mutate_caller_options(monkeypatch):
    """动态预算必须拷贝 deep_options 再改，不得污染调用方的 dict。

    回归钉（2026-11-06 修复）：此前 `deep_options["num_predict"] = ...` 原地
    改写——同一 dict 被多次调用复用时，第一次算出的预算会成为第二次的
    base（预算漂移）。beam_search 的 candidate_deep_options 拷贝是正确范本。
    """
    def fake_generate(model, prompt, timeout, **kwargs):
        # fast 返回可解析分数（gap=0.5 < margin）→ 动态预算路径必触发。
        if "Do not output JSON yet" in prompt:
            return '{"move_1": 6.0, "move_2": 5.5, "move_3": 4.0}'
        return '{"move_1": 4, "move_2": 8, "move_3": 3}'

    monkeypatch.setattr("kakeya_ai.ai.cascaded_reasoning.generate", fake_generate)

    shared = {"num_predict": 1024, "temperature": 0.4}
    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    cascaded_score("prompt", state=state, always_use_deep=True, deep_options=shared)

    assert shared == {"num_predict": 1024, "temperature": 0.4}, (
        f"调用方的 deep_options 被改写：{shared}"
    )


# ---------------------------------------------------------------------------
# 失败归因必须能区分"异常"与"空响应/解析失败"（2026-10-01）
#
# **为什么**：此前 `except Exception: fast_response = ""` 把异常吞成空串，再统一报
# `reason="fast_model_failed"`。于是**三种处置方向完全不同**的失败——
# ① 调用抛异常（网络/超时/模型没加载）、② 模型真返回空串、③ 返回了但解析不出
# JSON——在观测里**塌成同一个字符串**，排查时无从判断该查网络还是该改 prompt。
# 这组测试钉住：归因里必须带上**异常类型**，且与"空响应"可分。
# ---------------------------------------------------------------------------


def _capture_fallbacks(monkeypatch):
    """把 `_obs.record_fallback` 的调用捕获到列表（返回该列表）。"""
    import kakeya_ai.ai.cascaded_reasoning as cr

    captured: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cr._obs, "record_fallback",
        lambda source_label, reason="": captured.append((source_label, reason)),
    )
    return captured


def test_fast_exception_reason_carries_exception_type(monkeypatch):
    """fast 调用抛异常 ⇒ 归因里必须出现异常类型名与消息。"""
    def fake_generate(model, prompt, timeout, **kwargs):
        raise TimeoutError("ollama busy")

    monkeypatch.setattr("kakeya_ai.ai.cascaded_reasoning.generate", fake_generate)
    captured = _capture_fallbacks(monkeypatch)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    cascaded_score("prompt", state=state)

    reasons = " ".join(r for _s, r in captured)
    assert "TimeoutError" in reasons, f"归因未带异常类型：{captured}"
    assert "ollama busy" in reasons, f"归因未带异常消息：{captured}"


def test_fast_empty_response_is_distinguishable_from_exception(monkeypatch):
    """模型返回空串（未抛异常）⇒ 归因必须是"空响应"，**不含**异常标记。

    这条与上一条配对：若两者归因相同，说明又塌回了一个字符串。
    """
    def fake_generate(model, prompt, timeout, **kwargs):
        return "   "  # 非空对象但 strip 后为空

    monkeypatch.setattr("kakeya_ai.ai.cascaded_reasoning.generate", fake_generate)
    captured = _capture_fallbacks(monkeypatch)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    cascaded_score("prompt", state=state)

    reasons = " ".join(r for _s, r in captured)
    assert "empty_response" in reasons, f"空响应未被识别：{captured}"
    assert "exception" not in reasons, f"空响应被误标成异常：{captured}"


def test_deep_exception_reason_carries_exception_type(monkeypatch):
    """deep 调用抛异常 ⇒ 归因里必须出现异常类型名（与 fast 对称）。"""
    def fake_generate(model, prompt, timeout, **kwargs):
        # 同上：按 prompt 特征判别（fast 的提示词里含「先别输出 JSON」这句）。
        if "Do not output JSON yet" in prompt:
            return '{"move_1": 3, "move_2": 7, "move_3": 4}'
        raise ConnectionError("deep unreachable")

    monkeypatch.setattr("kakeya_ai.ai.cascaded_reasoning.generate", fake_generate)
    captured = _capture_fallbacks(monkeypatch)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    cascaded_score("prompt", state=state, always_use_deep=True)

    reasons = " ".join(r for _s, r in captured)
    assert "ConnectionError" in reasons, f"deep 归因未带异常类型：{captured}"
    assert "deep unreachable" in reasons, f"deep 归因未带异常消息：{captured}"


def test_deep_parse_failure_reason_is_not_marked_as_exception(monkeypatch):
    """deep 返回不可解析内容（未抛异常）⇒ 归因标"解析失败"，不标异常。"""
    def fake_generate(model, prompt, timeout, **kwargs):
        # 同上：按 prompt 特征判别（fast 的提示词里含「先别输出 JSON」这句）。
        if "Do not output JSON yet" in prompt:
            return '{"move_1": 3, "move_2": 7, "move_3": 4}'
        return "I think move 2 is best."  # 无 JSON

    monkeypatch.setattr("kakeya_ai.ai.cascaded_reasoning.generate", fake_generate)
    captured = _capture_fallbacks(monkeypatch)

    state = {"delta": 0.1, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.0}
    cascaded_score("prompt", state=state, always_use_deep=True)

    reasons = " ".join(r for _s, r in captured)
    assert "parse_failed" in reasons, f"解析失败未被识别：{captured}"
    assert "exception" not in reasons, f"解析失败被误标成异常：{captured}"
