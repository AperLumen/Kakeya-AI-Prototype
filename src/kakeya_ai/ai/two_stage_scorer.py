"""两阶段快思考打分器：快速起草 + 7B 重打分。

阶段 1：一个小/快模型（默认 qwen3.5:4b）产出初始走法分数。
阶段 2：若前两名落在某个模糊度边际之内，则由更大的模型（默认 qwen3.5:4b）
只对这两个 top 候选重新打分。

这样既让常见路径保持快速，又为棘手决策保留了深层推理。
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional, Tuple

from .cache import DecisionCache
from .heuristics import heuristic_move_scores
from .observability import default_observability, get_logger
from .ollama_client import generate
from ..resource_budget import DEFAULT_BUDGET
from .token_usage import TokenUsage


_logger = get_logger(__name__)
_obs = default_observability()


DEFAULT_FAST_MODEL = "qwen3.5:4b"
DEFAULT_DEEP_MODEL = "qwen3.5:4b"

DEFAULT_FAST_RETRIES = 2
DEFAULT_DEEP_RETRIES = 2


def parse_json_scores(response: str) -> Optional[Dict[int, float]]:
    """从类 JSON 的响应中抽取 {1: score, 2: score, 3: score}。"""
    # 推理型模型可能把思考内容包在 <think>...</think> 标签里。
    # 先剥离它们，这样 JSON 抽取作用在最终答案上。
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    # 尝试在响应里找出形如 {"move_1": X, "move_2": Y, "move_3": Z} 或 {1: X, ...} 这样的 JSON 对象，以便后续解析出分数
    match = re.search(r"\{[^}]*(?:\"move_1\"|\"1\")[^}]*\}", response)
    if not match:
        _obs.record_parse_failure(
            stage="parse_json_scores",
            response_snippet=response,
            reason="no_json_match",
        )
        return None
    try:
        raw = json.loads(match.group(0))
    except Exception as exc:
        _obs.record_parse_failure(
            stage="parse_json_scores",
            response_snippet=match.group(0),
            reason=f"json_loads_error: {exc}",
        )
        return None
    result: Dict[int, float] = {}
    for k, v in raw.items():
        try:
            key = int(str(k).split("_")[-1])
            result[key] = float(v)
        except Exception as exc:
            _obs.record_parse_failure(
                stage="parse_json_scores",
                response_snippet=f"{k}={v}",
                reason=f"key_value_conversion_error: {exc}",
            )
            continue
    if not result:
        _obs.record_parse_failure(
            stage="parse_json_scores",
            response_snippet=match.group(0),
            reason="no_valid_move_scores",
        )
    return result if result else None


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """把分数裁剪到 [0, 10]，并确保所有走法下标都存在。"""
    return {i: max(0.0, min(10.0, scores.get(i, 5.0))) for i in (1, 2, 3)}


def _top_two_gap(scores: Dict[int, float]) -> Tuple[float, int, int]:
    """返回 (gap, best_move, second_best_move)。"""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    gap = ranked[0][1] - ranked[1][1]
    return gap, ranked[0][0], ranked[1][0]


def _return(
    scores: Dict[int, float],
    source: str,
    reason: str = "",
    state: Optional[dict] = None,
) -> Tuple[Dict[int, float], str]:
    """归一化分数、记录决策并返回。"""
    normalized = _normalize_scores(scores)
    state_key = ""
    if state is not None:
        state_key = DecisionCache().key(state, (1, 2, 3))
    _obs.record_decision(source_label=source, state_key=state_key)
    if reason:
        _obs.record_fallback(source_label=source, reason=reason)
    return normalized, source


def two_stage_score(
    prompt: str,
    cache: Optional[DecisionCache] = None,
    state: Optional[dict] = None,
    ambiguity_margin: float = 2.0,
    fast_model: str = DEFAULT_FAST_MODEL,
    deep_model: str = DEFAULT_DEEP_MODEL,
    fast_timeout: float = DEFAULT_BUDGET.fast_timeout,
    deep_timeout: float = DEFAULT_BUDGET.deep_timeout,
    fast_retries: int = DEFAULT_FAST_RETRIES,
    deep_retries: int = DEFAULT_DEEP_RETRIES,
    use_heuristic_fallback: bool = True,
    keep_alive: str = "10m",
    fast_options: Optional[dict] = None,
    deep_options: Optional[dict] = None,
    usage: Optional[TokenUsage] = None,
) -> Tuple[Dict[int, float], str]:
    """返回走法分数与来源标签。

    Source 标签：
      - "heuristic"：未使用 LLM
      - "fast"：只用了快速模型
      - "deep"：用了深层（7B）模型重打分
      - "cache"：缓存结果
    """
    # 1. 缓存查找。
    if cache is not None and state is not None:
        cached = cache.get(state)
        if cached is not None:
            parsed = parse_json_scores(json.dumps(cached))
            if parsed is not None:
                return _return(parsed, "cache", state=state)

    # 2. 启发式预检：若胜者明确，则跳过所有 LLM 调用。
    if use_heuristic_fallback and state is not None:
        h_scores = heuristic_move_scores(
            None,
            state.get("rho", 0.5),
            state.get("delta", 0.01),
            state.get("local_error", 1.0),
            state.get("volume_ratio", 1.0),
        )
        gap, _, _ = _top_two_gap(h_scores)
        if gap >= ambiguity_margin:
            result = _normalize_scores(h_scores)
            if cache is not None and state is not None:
                cache.set(state, {f"move_{k}": v for k, v in result.items()})
            return _return(result, "heuristic", state=state)

    # 3. 快速模型起草打分。
    if fast_options is None:
        fast_options = {"num_predict": 512, "temperature": 0.6}
    #: 归因：区分**调用抛异常**与**解析失败**（2026-10-01）。
    #:
    #: 此前 `except Exception: fast_scores = None`，再统一报
    #: `reason="fast_model_or_parse_failed"`——把"网络/超时/模型没加载"与"返回了
    #: 但解析不出 JSON"**塌成一个字符串**，排查时分不清该查网络还是改 prompt。
    #: （`parse_json_scores` 内部**自己会**记 parse failure；这里的异常只覆盖
    #: "调用本身抛了异常"那条路径，两者不重叠。）
    fast_failure: str = ""
    try:
        fast_response = generate(
            fast_model,
            prompt,
            timeout=fast_timeout,
            keep_alive=keep_alive,
            options=fast_options,
            usage=usage,
            max_retries=fast_retries,
            # 思考型 fast 模型（如 qwen3.5:4b）须关 think，否则 num_predict
            # 被思维链耗尽、返回空串；非思考型模型会忽略该标志。
            think=False,
        )
        fast_scores = parse_json_scores(fast_response)
    except Exception as exc:
        # 任何失败都走启发式兜底，但**必须保留异常类型**用于归因。
        fast_scores = None
        fast_failure = f"{type(exc).__name__}: {exc}"
    _fast_reason = (
        f"fast_model_exception: {fast_failure}" if fast_failure
        else "fast_model_parse_failed"
    )

    if fast_scores is None:
        # 快速模型失败；回落到启发式。
        if state is not None:
            result = _normalize_scores(
                heuristic_move_scores(
                    None,
                    state.get("rho", 0.5),
                    state.get("delta", 0.01),
                    state.get("local_error", 1.0),
                    state.get("volume_ratio", 1.0),
                )
            )
            if cache is not None:
                cache.set(state, {f"move_{k}": v for k, v in result.items()})
            return _return(
                result,
                "heuristic",
                reason=_fast_reason,
                state=state,
            )
        return _return(
            {1: 5.0, 2: 5.0, 3: 5.0},
            "heuristic",
            reason=f"{_fast_reason}_no_state",
            state=state,
        )

    fast_scores = _normalize_scores(fast_scores)

    # 4. 决定是否需要深层重打分。
    gap, best, second = _top_two_gap(fast_scores)
    if gap >= ambiguity_margin:
        # 快速模型有把握；无需深层模型。
        if cache is not None and state is not None:
            cache.set(state, {f"move_{k}": v for k, v in fast_scores.items()})
        return _return(fast_scores, "fast", state=state)

    # 5. 深层模型（7B）只对模糊的前两名走法重新打分。
    deep_prompt = (
        prompt
        + f"\nThe fast model scored them as: move_{best}={fast_scores[best]:.1f}, "
        + f"move_{second}={fast_scores[second]:.1f}. These two are close. "
        + "Re-score ONLY move_"
        + str(best)
        + " and move_"
        + str(second)
        + " and return JSON: {\"move_"
        + str(best)
        + "\": <int>, \"move_"
        + str(second)
        + "\": <int>}."
    )
    if deep_options is None:
        deep_options = {"num_predict": 1024, "temperature": 0.4}
    #: deep 模型的失败原因；调用成功且解析成功时留空串。
    #:
    #: **必须在 try 之前初始化**：``generate`` 成功返回、但
    #: ``parse_json_scores`` 解析不出（返回 None）时**不抛异常**，于是
    #: ``except`` 不执行、``deep_failure`` 未绑定——而下面要用它拼 reason，
    #: 会抛 ``UnboundLocalError``（实测确认）。
    #: 与 ``fast_failure: str = ""`` 同一处置（那里本来就对了，这里漏了）。
    #: 空串的语义是"没有异常"，与"异常信息为空串"不冲突（后者不可能）。
    deep_failure: str = ""
    try:
        deep_response = generate(
            deep_model,
            deep_prompt,
            timeout=deep_timeout,
            keep_alive=keep_alive,
            options=deep_options,
            usage=usage,
            max_retries=deep_retries,
            # 思考型 deep 模型（如 qwen3.5:4b）须关 think，否则 num_predict
            # 被思维链耗尽、返回空串；非思考型模型会忽略该标志。
            think=False,
        )
        deep_partial = parse_json_scores(deep_response)
    except Exception as exc:
        deep_partial = None
        deep_failure = f"{type(exc).__name__}: {exc}"

    if deep_partial is not None:
        # 把深层重打分合并进快速分数。
        final_scores = dict(fast_scores)
        for k, v in deep_partial.items():
            final_scores[k] = v
        final_scores = _normalize_scores(final_scores)
        if cache is not None and state is not None:
            cache.set(state, {f"move_{k}": v for k, v in final_scores.items()})
        return _return(final_scores, "deep", state=state)

    # 深层模型失败；使用快速分数。
    _deep_reason = (
        f"deep_model_exception: {deep_failure}" if deep_failure
        else "deep_model_parse_failed"
    )
    if cache is not None and state is not None:
        cache.set(state, {f"move_{k}": v for k, v in fast_scores.items()})
    return _return(
        fast_scores,
        "fast",
        reason=_deep_reason,
        state=state,
    )
