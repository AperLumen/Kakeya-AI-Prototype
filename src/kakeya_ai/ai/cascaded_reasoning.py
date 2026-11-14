"""级联快思考：快速模型起头做推理，7B 接着往下做。

不把小模型仅仅当作打分器，而是把它当作"快思考者"来产出初步分析。然后由更大的
模型接续该分析，输出最终的走法分数。

这样推理方向的开销很便宜（qwen3.5:4b），而重型模型只需精炼并收尾。
"""

from __future__ import annotations

import json
from typing import Optional, Tuple

from .cache import DecisionCache
from ..errors import InputValidationError
from .heuristics import heuristic_move_scores
from .observability import default_observability
from ..resource_budget import DEFAULT_BUDGET
from .ollama_client import generate
from .token_budget import compute_uncertainty_budget
from .token_usage import TokenUsage
from .two_stage_scorer import parse_json_scores


_obs = default_observability()


DEFAULT_FAST_MODEL = "qwen3.5:4b"
DEFAULT_DEEP_MODEL = "qwen3.5:4b"


def format_fast_thinker_prompt(base_state_prompt: str) -> str:
    """让快速模型只产出一小段推理轨迹。

    我们从基础 prompt 中剥离所有 few-shot 示例，以保持快速调用简短、不超出
    小模型的 token 预算。
    """
    core = base_state_prompt.split("Examples:")[0].strip()
    return (
        core
        + "\n\nFirst, in 2-3 sentences, explain which move looks most promising "
        "and why. Do not output JSON yet; only give the reasoning."
    )


def extract_anchors(
    fast_reasoning: str,
    fast_scores: dict | None,
    state: dict | None,
) -> str:
    """把 fast 的推理压成一行**注意力锚点**（Q4 #18，2026-11-13）。

    动机：raw 推理（2-3 句自然语言）喂给 deep 时，deep 要重读重理解，
    还会被 fast 的措辞带偏（锚定效应）；而结构化锚点只留**可核验的量**：
    状态特征值 + fast 的选择与分差。deep 据此聚焦验证而非复述。

    全部离线计算（不发 LLM 请求），缺什么就缺省——不编造锚点。
    """
    parts = []
    if state:
        for k in ("rho", "delta", "local_error", "volume_ratio"):
            if k in state:
                parts.append(f"{k}={float(state[k]):.4g}")
    if fast_scores:
        try:
            norm = _normalize_scores(fast_scores)
            gap, top, second = _top_two_gap(norm)
            parts.append(f"fast选Move#{top}(分差{gap:.2f})")
        except Exception:
            pass
    return "锚点: " + ", ".join(parts) if parts else ""


def format_deep_anchor_prompt(base_state_prompt: str, anchors: str) -> str:
    """锚点版的 deep 提示词：只给可核验的量，要求独立验证后出分。"""
    return (
        base_state_prompt
        + "\n\nA fast thinker extracted these anchors (verify, don't echo):\n"
        + "<anchors>\n"
        + anchors.strip()
        + "\n</anchors>\n\n"
        + "Independently verify and return exactly this JSON and nothing else:\n"
        + '{"move_1": <0-10 score>, "move_2": <0-10 score>, "move_3": <0-10 score>}'
    )


def format_deep_continuation_prompt(
    base_state_prompt: str,
    fast_reasoning: str,
) -> str:
    """让深层模型接续快思考者的推理。"""
    return (
        base_state_prompt
        + "\n\nA fast thinker started the analysis with:\n"
        + "<fast_thinking>\n"
        + fast_reasoning.strip()
        + "\n</fast_thinking>\n\n"
        + "Continue from this reasoning, refine it if needed, and return exactly "
        + "this JSON and nothing else:\n"
        + '{"move_1": <0-10 score>, "move_2": <0-10 score>, "move_3": <0-10 score>}'
    )


def _normalize_scores(scores: dict) -> dict:
    """把分数裁剪到 [0, 10]，并确保所有走法下标都存在。"""
    return {i: max(0.0, min(10.0, scores.get(i, 5.0))) for i in (1, 2, 3)}


def _top_two_gap(scores: dict) -> Tuple[float, int, int]:
    """返回 (gap, best_move, second_best_move)。"""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    gap = ranked[0][1] - ranked[1][1]
    return gap, ranked[0][0], ranked[1][0]


def _return(
    scores: dict,
    source: str,
    reason: str = "",
    state: Optional[dict] = None,
) -> Tuple[dict, str]:
    """归一化分数、记录决策并返回。"""
    normalized = _normalize_scores(scores)
    state_key = ""
    if state is not None:
        state_key = DecisionCache().key(state, (1, 2, 3))
    _obs.record_decision(source_label=source, state_key=state_key)
    if reason:
        _obs.record_fallback(source_label=source, reason=reason)
    return normalized, source


def cascaded_score(
    base_state_prompt: str,
    cache: Optional[DecisionCache] = None,
    state: Optional[dict] = None,
    ambiguity_margin: float = 2.0,
    fast_model: str = DEFAULT_FAST_MODEL,
    deep_model: str = DEFAULT_DEEP_MODEL,
    fast_timeout: float = DEFAULT_BUDGET.fast_timeout,
    deep_timeout: float = DEFAULT_BUDGET.deep_timeout,
    use_heuristic_fallback: bool = True,
    always_use_deep: bool = True,
    keep_alive: str = "10m",
    fast_options: Optional[dict] = None,
    deep_options: Optional[dict] = None,
    dynamic_budget: bool = True,
    usage: Optional[TokenUsage] = None,
    deep_context: str = "raw",
) -> Tuple[dict, str]:
    """用级联的 fast → 7B 推理返回走法分数。

    Source 标签：
      - "heuristic"：未使用 LLM
      - "fast"：只用了快速模型（少见；由 ambiguity_margin 把关）
      - "cascaded"：深层模型接续了快速推理
      - "cache"：缓存结果
    """
    # 1. 缓存查找。
    if cache is not None and state is not None:
        cached = cache.get(state)
        if cached is not None:
            parsed = parse_json_scores(json.dumps(cached))
            if parsed is not None:
                return _return(parsed, "cache", state=state)

    # 2. 对平凡决策做启发式预检。
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

    # 3. 快速模型开始推理。
    fast_prompt = format_fast_thinker_prompt(base_state_prompt)
    if fast_options is None:
        fast_options = {"num_predict": 512, "temperature": 0.6}
    #: 记录 fast 调用**为什么**没给出可用结果——用于归因（见下）。
    #:
    #: **为什么必须记异常类型**（2026-10-01）：此前 `except Exception: fast_response = ""`
    #: 把异常吞成空串，再统一报 `reason="fast_model_failed"`。于是
    #: **"调用抛异常"（网络/超时/模型没加载）** 与 **"模型真的返回空"** 与
    #: **"返回了但解析不出 JSON"** 三种**处置方向完全不同**的失败，在观测里
    #: **塌成同一个字符串**——排查时无从区分"该查网络"还是"该改 prompt"。
    #: 这与本项目记过的「归因必须能分开，否则处置方向会错」是同一族。
    fast_failure: str = ""
    try:
        fast_response = generate(
            fast_model,
            fast_prompt,
            timeout=fast_timeout,
            keep_alive=keep_alive,
            options=fast_options,
            usage=usage,
            # 思考型 fast 模型（如 qwen3.5:4b）须关 think，否则 num_predict
            # 被思维链耗尽、返回空串；非思考型模型会忽略该标志。
            think=False,
        )
    except Exception as exc:
        fast_response = ""
        fast_failure = f"{type(exc).__name__}: {exc}"

    if not fast_response.strip():
        # 快速模型失败；回落到启发式。
        #
        # 归因**区分两种情况**：`fast_failure` 非空 = 调用抛了异常（附类型）；
        # 为空 = 调用没抛但返回了空串。二者处置不同，不能塌成一个字符串。
        _fast_reason = (
            f"fast_model_exception: {fast_failure}" if fast_failure
            else "fast_model_empty_response"
        )
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

    fast_reasoning = fast_response.strip()

    # 可选：若快速推理已经隐含了明显的胜者，且允许我们跳过深层模型，则尝试
    # 从该推理中提取分数。这里刻意保守：只有在"存在明确的启发式胜者（已在上面
    # 处理）"或"调用方设置 always_use_deep=False 且快速响应含可解析分数并具有
    # 较大 gap"时，才跳过深层模型。
    fast_scores = parse_json_scores(fast_reasoning)
    if (
        not always_use_deep
        and fast_scores is not None
        and _top_two_gap(_normalize_scores(fast_scores))[0] >= ambiguity_margin
    ):
        result = _normalize_scores(fast_scores)
        if cache is not None and state is not None:
            cache.set(state, {f"move_{k}": v for k, v in result.items()})
        return _return(result, "fast", state=state)

    # 4. 深层模型接续快思考者的推理。
    #    deep_context="raw"（默认，接续原始推理）/"anchors"（Q4 #18：
    #    只给结构化锚点，deep 独立验证而非复述——实测见内部实验记录）。
    if deep_context == "anchors":
        anchors = extract_anchors(fast_reasoning, fast_scores, state)
        if anchors:
            deep_prompt = format_deep_anchor_prompt(base_state_prompt, anchors)
        else:
            deep_prompt = format_deep_continuation_prompt(
                base_state_prompt, fast_reasoning)
    elif deep_context == "raw":
        deep_prompt = format_deep_continuation_prompt(
            base_state_prompt, fast_reasoning)
    else:
        raise InputValidationError(
            f"deep_context 只能是 'raw'/'anchors'，收到 {deep_context!r}"
        )
    if deep_options is None:
        deep_options = {"num_predict": 1024, "temperature": 0.4}
    #: **拷贝再改**（2026-11-06 修）：动态预算（下条）会改写
    #: ``num_predict``——若直接改调用方传入的 dict，本次算出的预算会成为
    #: 下次调用的 base（预算漂移），并污染调用方状态。与
    #: ``beam_search`` 的 ``candidate_deep_options = dict(deep_options)``
    #: 是同款处理，两处对齐。
    deep_options = dict(deep_options)

    if dynamic_budget and fast_scores is not None:
        base_budget = int(deep_options.get("num_predict", 1024))
        gap = _top_two_gap(_normalize_scores(fast_scores))[0]
        deep_options["num_predict"] = compute_uncertainty_budget(
            gap,
            base=base_budget,
            min_tokens=max(128, base_budget // 4),
            max_tokens=max(base_budget, 2048),
        )

    #: 与 fast 同理：记**为什么** deep 没给出可用结果（见 `fast_failure` 的说明）。
    #: 注意 `parse_json_scores` 内部**自己会**记 parse failure——故这里的异常只
    #: 覆盖"调用本身抛了异常"（网络/超时/模型没加载）；那条路径的归因不重叠。
    deep_failure: str = ""
    try:
        deep_response = generate(
            deep_model,
            deep_prompt,
            timeout=deep_timeout,
            keep_alive=keep_alive,
            options=deep_options,
            usage=usage,
            # qwen3.5:4b 是思考型模型：不关闭 think 时 num_predict 预算会被
            # 思维链吃光、最终回答为空（实测 1024/2048 token 均触顶、resp=''）。
            # think=False 对 deepseek-r1 等模型无效（服务端忽略），故可无条件传。
            think=False,
        )
        deep_scores = parse_json_scores(deep_response)
    except Exception as exc:
        deep_scores = None
        deep_failure = f"{type(exc).__name__}: {exc}"

    if deep_scores is not None:
        result = _normalize_scores(deep_scores)
        if cache is not None and state is not None:
            cache.set(state, {f"move_{k}": v for k, v in result.items()})
        return _return(result, "cascaded", state=state)

    # 深层模型失败；若有快速模型分数就用它，否则用启发式。
    #
    # 归因**区分异常与解析失败**：`deep_failure` 非空 = deep 调用抛了异常（附类型）；
    # 为空 = 调用成功但解析不出 JSON（那条路径已由 `parse_json_scores` 记 parse failure）。
    _deep_reason = (
        f"deep_model_exception: {deep_failure}" if deep_failure
        else "deep_model_parse_failed"
    )
    if fast_scores is not None:
        result = _normalize_scores(fast_scores)
        if cache is not None and state is not None:
            cache.set(state, {f"move_{k}": v for k, v in result.items()})
        return _return(
            result,
            "fast",
            reason=_deep_reason,
            state=state,
        )

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
            reason=f"{_deep_reason}_no_fast_scores",
            state=state,
        )

    return _return(
        {1: 5.0, 2: 5.0, 3: 5.0},
        "heuristic",
        reason=f"{_deep_reason}_no_state",
        state=state,
    )
