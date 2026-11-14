"""AI 快思考推理层。"""

from .cache import DecisionCache
from .heuristics import heuristic_move_scores, should_call_llm
from .nesterov_optimizer import NesterovOptimizer
from .observability import (
    LLMObservability,
    configure_logging,
    default_observability,
    get_logger,
    observe_llm_call,
)
from .ollama_client import generate, is_server_ready
from .prompts import (
    BRANCH_SCORING_PROMPT,
    COUNTEREXAMPLE_PROMPT,
    LEMMACANDIDATE_PROMPT,
    format_branch_scoring,
    # 下面两个名字已在 ``__all__`` 里声明，但此前**没有导入**——于是
    # ``from kakeya_ai.ai import *`` 会直接抛 AttributeError 崩掉
    # （星号导入要求 __all__ 里每个名字都是模块属性）。补齐导入即可。
    format_branch_scoring_dynamic,
    format_branch_scoring_prefix,
)
from .beam_search import beam_search_score
from .cascaded_reasoning import cascaded_score
from .expert_router import (
    Expert,
    ExpertRouter,
    default_experts,
    list_experts,
    parse_query_for_routing,
)
from .prompt_cache import PromptCache
from .reasoning_trace import ReasoningTrace, ReasoningTraceStore, TraceVectorizer
from .surrogate_scorer import SurrogateScorer
from .token_budget import compute_deep_budget, compute_uncertainty_budget
from .token_usage import TokenUsage
from .training_data import FEATURE_NAMES, build_training_data, state_from_metrics
from .two_stage_scorer import two_stage_score

__all__ = [
    "generate",
    "is_server_ready",
    "DecisionCache",
    "heuristic_move_scores",
    "should_call_llm",
    "two_stage_score",
    "cascaded_score",
    "beam_search_score",
    "ReasoningTrace",
    "ReasoningTraceStore",
    "TraceVectorizer",
    "PromptCache",
    "compute_deep_budget",
    "compute_uncertainty_budget",
    "TokenUsage",
    "LLMObservability",
    "default_observability",
    "get_logger",
    "observe_llm_call",
    "configure_logging",
    "SurrogateScorer",
    "NesterovOptimizer",
    "build_training_data",
    # 特征清单是训练端与推理端的**唯一真源**（维数与顺序都按它），
    # ``state_from_metrics`` 是「ProofMetrics → 特征 dict」的唯一入口，
    # 两者都是调用方要用的公共接口，故对外导出。
    "FEATURE_NAMES",
    "state_from_metrics",
    "BRANCH_SCORING_PROMPT",
    "LEMMACANDIDATE_PROMPT",
    "COUNTEREXAMPLE_PROMPT",
    "format_branch_scoring",
    "format_branch_scoring_prefix",
    "format_branch_scoring_dynamic",
    # 小专家模式 V1 (Mini MoE)
    "Expert",
    "ExpertRouter",
    "default_experts",
    "list_experts",
    "parse_query_for_routing",
]
