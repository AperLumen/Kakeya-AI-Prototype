"""Beam-search 快思考：快速模型提出候选推理，7B 模型打分/剪枝。

支持两种打分模式：

- ``scoring_mode="deep"``（默认）：快速模型生成 ``beam_width`` 条带置信度的
  候选推理方向；7B 模型评估其中 top ``deep_eval_top`` 条候选并返回走法分数。
- ``scoring_mode="self"``：快速模型先生成候选，然后自己给它们重新打分；7B
  模型只按快速模型的自评分评估 top ``deep_eval_top`` 条候选。

历史推理轨迹可通过 ``trace_store`` 传入；最相关的历史摘要会被检索出来，作为
每次 7B 调用的注意力锚点。
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .cache import DecisionCache
from .heuristics import heuristic_move_scores
from .observability import default_observability
from ..resource_budget import DEFAULT_BUDGET
from .ollama_client import generate
from .reasoning_trace import ReasoningTraceStore
from .token_budget import compute_deep_budget
from .token_usage import TokenUsage
from .two_stage_scorer import parse_json_scores


DEFAULT_FAST_MODEL = "qwen3.5:4b"
DEFAULT_DEEP_MODEL = "qwen3.5:4b"

#: deep 评估的默认并发数：**刻意串行**。并发是调用方显式传参的选择，不是
#: 默认行为——默认并发会让"单机一次最多两个进程"的资源纪律形同虚设。
#: 其余预算默认值（超时等）统一取自 ``DEFAULT_BUDGET``，此处只单独钉住这一项。
DEFAULT_BEAM_WORKERS = 1

logger = logging.getLogger(__name__)
_obs = default_observability()


@dataclass
class Candidate:
    """快速模型给出的一条候选推理方向。"""

    reasoning: str
    recommended_move: int
    confidence: float
    index: int


def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """把分数裁剪到 [0, 10]，并确保所有走法下标都存在。"""
    return {i: max(0.0, min(10.0, scores.get(i, 5.0))) for i in (1, 2, 3)}


def _top_two_gap(scores: Dict[int, float]) -> Tuple[float, int, int]:
    """返回 (gap, best_move, second_best_move)。"""
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[0][1] - ranked[1][1], ranked[0][0], ranked[1][0]


def _state_key(state: dict) -> str:
    """与 DecisionCache 中的约定一致的稳定 key。"""
    return str(
        (
            round(state.get("delta", 0.0), 6),
            round(state.get("rho", 0.0), 3),
            round(state.get("local_error", 0.0), 3),
            round(state.get("volume_ratio", 0.0), 3),
            (1, 2, 3),
        )
    )


def format_fast_beam_prompt(base_state_prompt: str, beam_width: int) -> str:
    """向快速模型索取 ``beam_width`` 条候选推理方向。"""
    # 去掉 few-shot 示例，让快速调用保持简短。
    core = base_state_prompt.split("Examples:")[0].strip()
    return (
        core
        + f"\n\nGenerate exactly {beam_width} distinct candidate reasoning directions. "
        "Each candidate must be one concise sentence that recommends one of the moves "
        "and gives a one-sentence justification. "
        "Output each candidate on its own line in this exact format:\n"
        "Candidate i: <reasoning sentence> ### <move_1/move_2/move_3> ### <confidence 0-10>\n\n"
        "Example:\n"
        "Candidate 1: Rho is close to 1 so grains are too square; reducing rho is the priority. ### move_2 ### 8"
    )


def parse_candidate_reasonings(response: str, beam_width: int) -> List[Candidate]:
    """解析快速模型的候选行。

    期望的行格式：
        Candidate i: <reasoning> ### <move_N> ### <confidence>

    最多返回 ``beam_width`` 条候选。无效行会被跳过。
    解析前会剥离推理/思考标签（例如 <think>...</think>）。
    """
    # 若存在 DeepSeek-R1 风格的思考块则移除。
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    candidates: List[Candidate] = []
    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line.lower().startswith("candidate"):
            _obs.record_parse_failure(
                stage="parse_candidate_reasonings",
                response_snippet=line,
                reason="line_not_starting_with_candidate",
            )
            continue
        parts = [p.strip() for p in line.split("###")]
        if len(parts) != 3:
            _obs.record_parse_failure(
                stage="parse_candidate_reasonings",
                response_snippet=line,
                reason=f"expected_3_parts_got_{len(parts)}",
            )
            continue
        try:
            reasoning_part = parts[0].split(":", 1)[1].strip()
            move_match = re.search(r"(\d+)", parts[1])
            if not move_match:
                _obs.record_parse_failure(
                    stage="parse_candidate_reasonings",
                    response_snippet=line,
                    reason="no_move_digit_in_part",
                )
                continue
            recommended_move = int(move_match.group(1))
            confidence = float(parts[2])
        except Exception as exc:
            _obs.record_parse_failure(
                stage="parse_candidate_reasonings",
                response_snippet=line,
                reason=f"parse_exception: {exc}",
            )
            continue
        candidates.append(
            Candidate(
                reasoning=reasoning_part,
                recommended_move=recommended_move,
                confidence=confidence,
                index=len(candidates) + 1,
            )
        )
        if len(candidates) >= beam_width:
            break
    if not candidates:
        _obs.record_parse_failure(
            stage="parse_candidate_reasonings",
            response_snippet=response,
            reason="no_candidates_parsed",
        )
    return candidates


def format_self_scoring_prompt(base_state_prompt: str, candidates: List[Candidate]) -> str:
    """让快速模型对自己的候选推理重新打分。"""
    lines = "\n".join(
        f"{i}. {c.reasoning} (recommended {c.recommended_move}, confidence {c.confidence})"
        for i, c in enumerate(candidates, 1)
    )
    return (
        base_state_prompt
        + "\n\nThe fast model generated these candidate reasonings:\n"
        + lines
        + "\n\nScore each candidate 0-10 based on how likely its reasoning leads to the best move. "
        "Return exactly this JSON and nothing else:\n"
        '{"scores": [<score_1>, <score_2>, ...]}'
    )


def _parse_self_scores(response: str, expected: int) -> Optional[List[float]]:
    """解析自评分的 JSON 列表；解析失败则返回 None。"""
    try:
        data = json.loads(response)
        scores = data.get("scores") if isinstance(data, dict) else data
        if isinstance(scores, list) and len(scores) >= expected:
            return [float(s) for s in scores[:expected]]
    except Exception as exc:
        logger.warning("Failed to parse self-scores JSON: %s", exc)
        _obs.record_parse_failure(
            stage="parse_self_scores",
            response_snippet=response,
            reason=f"json_loads_error: {exc}",
        )

    # 回落方案：抓取第一个方括号包裹的数字列表。
    match = re.search(r"\[\s*([\d\s.,]+)\s*\]", response)
    if match:
        try:
            nums = [float(x) for x in match.group(1).split(",") if x.strip()]
            if len(nums) >= expected:
                return nums[:expected]
        except Exception as exc:
            logger.warning("Failed to parse self-scores fallback list: %s", exc)
            _obs.record_parse_failure(
                stage="parse_self_scores",
                response_snippet=match.group(0),
                reason=f"fallback_list_error: {exc}",
            )
            return None
    _obs.record_parse_failure(
        stage="parse_self_scores",
        response_snippet=response,
        reason="no_valid_score_list",
    )
    return None


def format_deep_beam_evaluation_prompt(
    base_state_prompt: str,
    candidate: Candidate,
    anchors: List[Tuple[str, float]],
) -> str:
    """让深层模型在可选锚点下评估一条候选。"""
    anchor_block = ""
    if anchors:
        anchor_block = (
            "\n\nRelevant historical reasoning anchors (use them only as context):\n"
            + "\n".join(f"- {summary}" for summary, _ in anchors)
        )
    return (
        base_state_prompt
        + anchor_block
        + "\n\nCandidate reasoning to evaluate:\n<candidate_reasoning>\n"
        + candidate.reasoning
        + "\n</candidate_reasoning>\n\n"
        "Assess the candidate reasoning above. Then score each move 0-10 as the final decision "
        "influenced by this reasoning. Also give an overall candidate_quality 0-10. "
        "Return exactly this JSON and nothing else:\n"
        '{"move_1": <0-10>, "move_2": <0-10>, "move_3": <0-10>, "candidate_quality": <0-10>}'
    )


def _parse_deep_evaluation(response: str) -> Tuple[Optional[Dict[int, float]], float]:
    """从深层评估响应中解析出走法分数与 candidate_quality。"""
    # parse_json_scores 已剥离 <think> 块；对 quality 也做同样处理。
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    scores = parse_json_scores(cleaned)
    quality: float = 5.0
    quality_parsed = False
    try:
        match = re.search(r"\{[^}]*\}", cleaned)
        if match:
            data = json.loads(match.group(0))
            quality = float(data.get("candidate_quality", quality))
            quality_parsed = True
    except Exception as exc:
        logger.warning("Failed to parse candidate_quality: %s", exc)
        _obs.record_parse_failure(
            stage="parse_deep_evaluation",
            response_snippet=cleaned,
            reason=f"candidate_quality_error: {exc}",
        )
    if scores is None and not quality_parsed:
        _obs.record_parse_failure(
            stage="parse_deep_evaluation",
            response_snippet=cleaned,
            reason="no_scores_or_quality",
        )
    return scores, quality


def _aggregate_evaluated(
    evaluated: List[Tuple[Candidate, Dict[int, float], float]]
) -> Dict[int, float]:
    """以 candidate_quality 为权重对走法分数做加权平均。"""
    if not evaluated:
        return {1: 5.0, 2: 5.0, 3: 5.0}

    total_weight = 0.0
    aggregated = {1: 0.0, 2: 0.0, 3: 0.0}
    for _candidate, scores, quality in evaluated:
        weight = max(0.0, quality)
        if weight <= 0:
            continue
        for move in (1, 2, 3):
            aggregated[move] += weight * scores.get(move, 5.0)
        total_weight += weight

    if total_weight <= 0:
        return {1: 5.0, 2: 5.0, 3: 5.0}
    return _normalize_scores({k: v / total_weight for k, v in aggregated.items()})


def _fallback_heuristic(state: dict) -> Dict[int, float]:
    return _normalize_scores(
        heuristic_move_scores(
            None,
            state.get("rho", 0.5),
            state.get("delta", 0.01),
            state.get("local_error", 1.0),
            state.get("volume_ratio", 1.0),
        )
    )


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


def beam_search_score(
    base_state_prompt: str,
    cache: Optional[DecisionCache] = None,
    state: Optional[dict] = None,
    beam_width: int = 4,
    deep_eval_top: int = 3,
    ambiguity_margin: float = 2.0,
    scoring_mode: str = "deep",
    fast_model: str = DEFAULT_FAST_MODEL,
    deep_model: str = DEFAULT_DEEP_MODEL,
    fast_timeout: float = DEFAULT_BUDGET.fast_timeout,
    deep_timeout: float = DEFAULT_BUDGET.deep_timeout,
    use_heuristic_fallback: bool = True,
    keep_alive: str = "10m",
    fast_options: Optional[dict] = None,
    deep_options: Optional[dict] = None,
    trace_store: Optional[ReasoningTraceStore] = None,
    n_anchors: int = 2,
    dynamic_budget: bool = True,
    max_workers: int = DEFAULT_BEAM_WORKERS,
    usage: Optional[TokenUsage] = None,
) -> Tuple[Dict[int, float], str]:
    """用对快速模型推理的 beam search 返回走法分数。

    Source 标签：
      - "heuristic"：未使用 LLM
      - "fast"：只有快速模型产出了可用输出
      - "beam"：深层模型评估了候选推理
      - "cache"：缓存结果

    ``max_workers > 1`` 时 deep 评估并发执行（模式同 ``PathExecutor``；
    服务端 ``OLLAMA_NUM_PARALLEL > 1`` 才有实效），默认 1 保持串行。
    """
    if deep_eval_top > beam_width:
        deep_eval_top = beam_width

    # 1. 缓存查找。
    if cache is not None and state is not None:
        cached = cache.get(state)
        if cached is not None:
            parsed = parse_json_scores(json.dumps(cached))
            if parsed is not None:
                return _return(parsed, "cache", state=state)

    # 2. 对平凡决策做启发式预检。
    if use_heuristic_fallback and state is not None:
        h_scores = _fallback_heuristic(state)
        gap, _, _ = _top_two_gap(h_scores)
        if gap >= ambiguity_margin:
            result = _normalize_scores(h_scores)
            if cache is not None and state is not None:
                cache.set(state, {f"move_{k}": v for k, v in result.items()})
            return _return(result, "heuristic", state=state)

    # 3. 快速模型生成候选推理。
    # 推理型模型（DeepSeek-R1）会为 <think> 块消耗 token，所以我们给快速模型
    # 比普通打分器更大的预算。
    if fast_options is None:
        fast_options = {"num_predict": 1024, "temperature": 0.7}
    fast_prompt = format_fast_beam_prompt(base_state_prompt, beam_width)
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
        logger.warning("Fast model candidate generation failed: %s", exc)
        fast_response = ""

    candidates = parse_candidate_reasonings(fast_response, beam_width)

    if not candidates:
        # 快速模型失败，或没有产出可解析的候选。
        if state is not None:
            result = _fallback_heuristic(state)
            if cache is not None:
                cache.set(state, {f"move_{k}": v for k, v in result.items()})
            return _return(
                result,
                "heuristic",
                reason="no_parseable_candidates",
                state=state,
            )
        return _return(
            {1: 5.0, 2: 5.0, 3: 5.0},
            "heuristic",
            reason="no_parseable_candidates_no_state",
            state=state,
        )

    # 4. 选出深层模型将要评估的 top 候选。
    if scoring_mode == "self":
        self_prompt = format_self_scoring_prompt(base_state_prompt, candidates)
        try:
            self_response = generate(
                fast_model,
                self_prompt,
                timeout=fast_timeout,
                keep_alive=keep_alive,
                options=fast_options,
                usage=usage,
                # 思考型 fast 模型（如 qwen3.5:4b）须关 think，否则 num_predict
                # 被思维链耗尽、返回空串；非思考型模型会忽略该标志。
                think=False,
            )
            self_scores = _parse_self_scores(self_response, len(candidates))
        except Exception as exc:
            logger.warning("Fast model self-scoring failed: %s", exc)
            self_scores = None

        if self_scores is not None:
            # ``_parse_self_scores`` 保证返回长度恰为 ``expected``（用
            # ``[:expected]`` 截断，且仅在 ``>= expected`` 时返回），故 strict
            # 恒成立——写上它使这条契约成为代码里可见的依赖。
            for candidate, score in zip(candidates, self_scores, strict=True):
                candidate.confidence = score
        # 若自评分失败，则回落到原始置信度分数。
        candidates.sort(key=lambda c: c.confidence, reverse=True)
    else:
        # 默认 deep 模式：信任快速模型自身的置信度。
        candidates.sort(key=lambda c: c.confidence, reverse=True)

    selected = candidates[:deep_eval_top]

    # 5. 深层模型在注意力锚点下逐条评估选中的候选。
    if deep_options is None:
        deep_options = {"num_predict": 1024, "temperature": 0.4}

    state_key = _state_key(state) if state is not None else ""

    # 预取（全部在主线程串行）：prompt 构造、动态预算、anchors 读取。
    # 这样 trace_store 的访问无需考虑并发，只有"真正的 HTTP 生成调用"
    # 可能进线程池。
    plans: List[Tuple[Candidate, str, dict]] = []
    for candidate in selected:
        anchors: List[Tuple[str, float]] = []
        if trace_store is not None:
            anchors = trace_store.get_anchors(
                candidate.reasoning,
                exclude_state_key=state_key,
                top_k=n_anchors,
            )
        deep_prompt = format_deep_beam_evaluation_prompt(
            base_state_prompt, candidate, anchors
        )
        candidate_deep_options = dict(deep_options)
        if dynamic_budget:
            base_budget = int(candidate_deep_options.get("num_predict", 1024))
            candidate_deep_options["num_predict"] = compute_deep_budget(
                candidate.confidence,
                base=base_budget,
                min_tokens=max(128, base_budget // 4),
                max_tokens=max(base_budget, 2048),
            )
        plans.append((candidate, deep_prompt, candidate_deep_options))

    def _evaluate(plan: Tuple[Candidate, str, dict]):
        """执行一次 deep 评估；失败返回 None（保持原串行的回落语义）。"""
        candidate, deep_prompt, opts = plan
        try:
            deep_response = generate(
                deep_model,
                deep_prompt,
                timeout=deep_timeout,
                keep_alive=keep_alive,
                options=opts,
                usage=usage,
                # 思考型 deep 模型（如 qwen3.5:4b）须关 think，否则 num_predict
                # 被思维链耗尽、返回空串；非思考型模型会忽略该标志。
                think=False,
            )
            deep_scores, quality = _parse_deep_evaluation(deep_response)
        except Exception as exc:
            logger.warning("Deep model evaluation failed: %s", exc)
            deep_scores, quality = None, 0.0
        if deep_scores is None:
            return None
        return (candidate, _normalize_scores(deep_scores), quality)

    if max_workers > 1 and len(plans) > 1:
        # 并发评估（2026-11-06）：deep_eval_top 次深评原本严格串行，每次是
        # 一次完整 HTTP 往返（本机 12B 满额 ~60s）。并发可摊薄排队与
        # prefill。**注意**：服务端 OLLAMA_NUM_PARALLEL=1（默认）时请求
        # 仍会排队，实效须配合服务端开启；本机内存受限，故默认保持 1
        # （串行，行为与此前一致）。模式对齐 PathExecutor._execute_parallel。
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_evaluate, plans))  # map 保序
    else:
        results = [_evaluate(plan) for plan in plans]
    evaluated = [r for r in results if r is not None]

    if evaluated:
        result = _aggregate_evaluated(evaluated)
        source = "beam"
        # 把得分最高候选的推理记录为该状态的轨迹。
        best_candidate = max(evaluated, key=lambda x: x[2])[0]
        trace_reasoning = best_candidate.reasoning
    else:
        # 深层模型对所有候选都失败了：回落到仅用快速模型的聚合。
        fast_only = {1: 0.0, 2: 0.0, 3: 0.0}
        weight_sum = 0.0
        for candidate in selected or candidates:
            weight = max(0.0, candidate.confidence)
            if weight <= 0:
                continue
            fast_only[candidate.recommended_move] += weight * 10.0
            weight_sum += weight
        if weight_sum > 0:
            for move in fast_only:
                fast_only[move] /= weight_sum
            result = _normalize_scores(fast_only)
        else:
            result = {1: 5.0, 2: 5.0, 3: 5.0}
        source = "fast"
        trace_reasoning = "; ".join(c.reasoning for c in (selected or candidates))

    result = _normalize_scores(result)

    # 6. 更新缓存与轨迹存储。
    if cache is not None and state is not None:
        cache.set(state, {f"move_{k}": v for k, v in result.items()})

    if trace_store is not None and state is not None:
        trace_store.add(state_key, trace_reasoning, result)

    fallback_reason = ""
    if source == "fast":
        fallback_reason = "deep_evaluation_failed_for_all_candidates"
    return _return(result, source, reason=fallback_reason, state=state)
