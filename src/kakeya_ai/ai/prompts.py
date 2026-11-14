"""AI 快思考推理层的 prompt 模板。

这些 prompt 设计为填入具体的证明状态数据后，发送给快思考模型
（DeepSeek-R1-Distill、Kimi、Qwen3 等）。

为了让 Ollama 的前缀感知 KV-cache 复用尽可能有效，分支打分 prompt 被拆成
静态前缀（角色、走法、示例、输出格式）与动态状态块。静态前缀放在最前，
这样重复请求就能共享尽量长的 token 前缀。
"""

from __future__ import annotations

from typing import Optional

from .prompt_cache import PromptCache


# 分支打分的静态前缀：所有不随状态变化的内容。
BRANCH_SCORING_PREFIX = """You are pruning branches in a 3D Kakeya proof search.

Moves:
{move_descriptions}

Examples:
State: δ=0.01, ρ=0.99, local_error=8.0, volume_ratio=0.9 → Output: {{"move_1": 4, "move_2": 7, "move_3": 9}}
State: δ=0.01, ρ=0.1, local_error=0.3, volume_ratio=1.5 → Output: {{"move_1": 5, "move_2": 3, "move_3": 4}}

Score each move 0–10. Return exactly this JSON and nothing else:
{{"move_1": <int>, "move_2": <int>, "move_3": <int>}}
"""

# 为兼容仍导入原始模板的代码而保留。
BRANCH_SCORING_PROMPT = BRANCH_SCORING_PREFIX + """
State: assertion={assertion}, δ={delta}, ρ={rho}, grain={a}×{b}×{c}, local_error={local_error}, volume_ratio={volume_ratio}.
"""

MOVE_DESCRIPTION_TEMPLATE = "{i}. {description}"

MOVE_DESCRIPTIONS = {
    1: "Lengthen grains (increase c).",
    2: "Lengthen square grains (reduce ρ).",
    3: "Widen grains (reduce local convex Wolff error).",
}

LEMMACANDIDATE_PROMPT = """We are proving {assertion} for δ-tubes in R^3.
Obstacles: {obstacles}
Suggest one concise lemma (one sentence) and one sentence justification."""


COUNTEREXAMPLE_PROMPT = """Tube arrangement: n={n_tubes}, δ={delta}, C_KT-CW={kt_error}, C_F-SW={fs_error}, |∪T|={union_volume}, lower_bound={lower_bound}.
Propose one small perturbation that could make the volume estimate tight. One paragraph."""


def _move_descriptions(move_indices: list[int]) -> str:
    return "\n".join(
        MOVE_DESCRIPTION_TEMPLATE.format(
            i=i, description=MOVE_DESCRIPTIONS.get(i, "Unknown move")
        )
        for i in move_indices
    )


def format_branch_scoring_prefix(move_indices: list[int] | None = None) -> str:
    """返回分支打分 prompt 的静态前缀。"""
    if move_indices is None:
        move_indices = [1, 2, 3]
    return BRANCH_SCORING_PREFIX.format(
        move_descriptions=_move_descriptions(move_indices)
    )


def format_branch_scoring_dynamic(
    assertion: str,
    delta: float,
    rho: float,
    a: float,
    b: float,
    c: float,
    local_error: float,
    volume_ratio: float,
) -> str:
    """返回分支打分 prompt 的动态状态块。"""
    return (
        f"State: assertion={assertion}, δ={delta}, ρ={rho}, "
        f"grain={a}×{b}×{c}, local_error={local_error}, volume_ratio={volume_ratio}."
    )


def format_branch_scoring(
    assertion: str,
    delta: float,
    rho: float,
    a: float,
    b: float,
    c: float,
    local_error: float,
    volume_ratio: float,
    move_indices: list[int] | None = None,
    prompt_cache: Optional[PromptCache] = None,
) -> str:
    """填入分支打分 prompt。

    Parameters
    ----------
    prompt_cache : PromptCache, optional
        若提供，则静态 prompt 前缀会被缓存并在多次调用间复用，从而为
        Ollama KV-cache 复用最大化前缀共享。
    """
    if move_indices is None:
        move_indices = [1, 2, 3]

    if prompt_cache is not None:
        cache_key = f"branch_scoring:{','.join(str(i) for i in move_indices)}"
        prefix = prompt_cache.get_or_build(
            cache_key, lambda: format_branch_scoring_prefix(move_indices)
        )
    else:
        prefix = format_branch_scoring_prefix(move_indices)

    dynamic_block = format_branch_scoring_dynamic(
        assertion=assertion,
        delta=delta,
        rho=rho,
        a=a,
        b=b,
        c=c,
        local_error=local_error,
        volume_ratio=volume_ratio,
    )

    return PromptCache.build_full_prompt(prefix, dynamic_block)
