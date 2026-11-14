"""小专家模式 (Mini MoE) 的最小测试.

只测试路由逻辑 (不调用 Ollama), 保证:
- 专家池构建正确
- 关键词评分正确
- top-k 路由选对专家
- 兜底逻辑生效
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import pytest

from kakeya_ai.ai.expert_router import (
    CONFIDENCE_FLOOR,
    EffortLevel,
    Expert,
    ExpertRouter,
    QuantileBalancer,
    default_experts,
    estimate_effort,
    list_experts,
    parse_query_for_routing,
    resolve_options,
)


# ---------------------------------------------------------------------------
# 专家池
# ---------------------------------------------------------------------------


def test_default_experts_has_five():
    pool = default_experts()
    assert len(pool) == 5
    names = {e.name for e in pool}
    assert names == {"math", "code", "writing", "kakeya", "general"}


def test_general_expert_has_no_keywords():
    pool = default_experts()
    general = next(e for e in pool if e.name == "general")
    assert general.keywords == []
    assert general.score("anything") == 0.0


def test_list_experts_shape():
    infos = list_experts()
    assert len(infos) == 5
    for info in infos:
        assert {"name", "model", "keywords", "temperature"} <= set(info.keys())


# ---------------------------------------------------------------------------
# 评分
# ---------------------------------------------------------------------------


def test_score_case_insensitive():
    pool = default_experts()
    code = next(e for e in pool if e.name == "code")
    assert code.score("用 PYTHON 写代码") > 0
    assert code.score("Python") > 0
    assert code.score("PYTHON") > 0


def test_score_increases_with_more_keywords_hit():
    pool = default_experts()
    code = next(e for e in pool if e.name == "code")
    s1 = code.score("python")
    s2 = code.score("python javascript")
    s3 = code.score("python javascript rust")
    assert s1 < s2 < s3


def test_score_zero_for_unrelated_query():
    pool = default_experts()
    code = next(e for e in pool if e.name == "code")
    assert code.score("今天天气怎么样") == 0.0


def test_score_empty_query():
    pool = default_experts()
    for e in pool:
        assert e.score("") == 0.0


def test_longer_keyword_gets_bonus():
    exp = Expert(
        name="test",
        keywords=["a", "abcdefghij"],
        system_prompt="x",
        model="m",
    )
    # 长度 1: 1 + min(1/10, 1.0) = 1.1
    assert exp.score("a") == pytest.approx(1.1)
    assert exp.score("abcdefghij") > exp.score("a")


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


def test_route_code_query_activates_code():
    router = ExpertRouter(top_k=1)
    activated = router.route("用 Python 写一个快速排序算法")
    names = [e.name for e in activated]
    assert "code" in names


def test_route_math_query_activates_math():
    router = ExpertRouter(top_k=1)
    activated = router.route("证明根号2是无理数, 给出推导步骤")
    names = [e.name for e in activated]
    assert "math" in names


def test_route_kakeya_query_activates_kakeya():
    router = ExpertRouter(top_k=1)
    activated = router.route("什么是 Kakeya 集合的 Hausdorff 维数?")
    names = [e.name for e in activated]
    assert "kakeya" in names


def test_route_writing_query_activates_writing():
    router = ExpertRouter(top_k=1)
    activated = router.route("帮我写一篇关于秋天的文章")
    names = [e.name for e in activated]
    assert "writing" in names


def test_route_unmatched_falls_back_to_general():
    router = ExpertRouter(top_k=2)
    activated = router.route("今天天气怎么样")
    names = [e.name for e in activated]
    assert "general" in names


def test_route_top_k_limit():
    router = ExpertRouter(top_k=1)
    # 同时命中 code 和 math, 但 top_k=1 只激活最高分
    activated = router.route("用 python 实现一个数学算法")
    assert len(activated) == 1


def test_route_top_k_two():
    router = ExpertRouter(top_k=2)
    activated = router.route("用 python 实现一个数学算法")
    assert len(activated) <= 2
    names = [e.name for e in activated]
    # code 和 math 都应被命中 (具体哪个进 top-2 取决于分数)
    assert "code" in names or "math" in names


def test_route_no_general_when_experts_match():
    router = ExpertRouter(top_k=2, always_include_general=True)
    activated = router.route("python 代码")
    names = [e.name for e in activated]
    # 有匹配时不应包含 general
    assert "general" not in names


def test_route_kakeya_priority_beats_others():
    # kakeya priority=3 最高, 当与 math/code 同分时应优先
    router = ExpertRouter(top_k=1)
    # "几何" 同时命中 math 和 kakeya
    activated = router.route("kakeya 几何")
    assert activated[0].name == "kakeya"


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


def test_parse_query_for_routing_returns_dicts():
    result = parse_query_for_routing("python 代码")
    assert isinstance(result, list)
    assert len(result) >= 1
    for item in result:
        assert {"name", "score", "model"} <= set(item.keys())


def test_parse_query_for_routing_general_fallback():
    result = parse_query_for_routing("今天天气怎么样")
    assert len(result) == 1
    assert result[0]["name"] == "general"


# ---------------------------------------------------------------------------
# 自定义专家池
# ---------------------------------------------------------------------------


def test_custom_expert_pool():
    custom = [
        Expert(
            name="custom",
            keywords=["foo", "bar"],
            system_prompt="custom expert",
            model="qwen3:4b",
        ),
    ]
    router = ExpertRouter(experts=custom, top_k=1)
    activated = router.route("foo bar baz")
    assert len(activated) == 1
    assert activated[0].name == "custom"


def test_custom_pool_no_general_no_match_returns_first():
    custom = [
        Expert(name="a", keywords=["x"], system_prompt="", model="m"),
        Expert(name="b", keywords=["y"], system_prompt="", model="m"),
    ]
    router = ExpertRouter(experts=custom, top_k=1, always_include_general=False)
    activated = router.route("zzz")  # 不命中任何关键词
    assert len(activated) == 1
    # 没匹配且不允许兜底时, 返回池里前 top_k 个
    assert activated[0].name == "a"


# ---------------------------------------------------------------------------
# 分位数均衡 Quantile Balancing（Kimi K3 论文 §2.3.3）
# ---------------------------------------------------------------------------


# 8 tokens x 4 experts: expert 0 每行都最高 (过热), 其余分数互有高低.
_QB_SCORES = [
    [10.0, 1.0, 2.0, 3.0],
    [10.0, 3.0, 1.0, 2.0],
    [10.0, 2.0, 3.0, 1.0],
    [10.0, 1.5, 2.5, 3.5],
    [10.0, 3.5, 1.5, 2.5],
    [10.0, 2.5, 3.5, 1.5],
    [10.0, 1.0, 3.0, 2.0],
    [10.0, 2.0, 1.0, 3.0],
]


def _topk_loads(bal: QuantileBalancer, scores, k: int = 1) -> list[int]:
    """用 biased score 做 top-k 选择, 统计每个 expert 的负载."""
    loads = [0] * bal.n
    for row in scores:
        biased = bal.biased_scores(row)
        for j in sorted(range(bal.n), key=lambda j: -biased[j])[:k]:
            loads[j] += 1
    return loads


def test_quantile_balancer_reduces_peak_load():
    bal = QuantileBalancer(n_experts=4, top_k=1)
    bal.update(_QB_SCORES)
    loads = _topk_loads(bal, _QB_SCORES)
    assert sum(loads) == 8
    # 无偏置时 expert 0 独占 (8,0,0,0); QB 后峰值负载应显著下降.
    assert max(loads) <= 4
    assert max(loads) < 8


def test_quantile_balancer_fixed_step_fallback():
    # 固定步长规则 (论文引用的 b += γ·sign(q̄ - ℓ_j)): 多轮更新后过热下降.
    bal = QuantileBalancer(n_experts=4, top_k=1, fixed_step=2.0)
    for _ in range(20):
        bal.update(_QB_SCORES)
    loads = _topk_loads(bal, _QB_SCORES)
    assert sum(loads) == 8
    assert max(loads) < 8


def test_quantile_balancer_skips_zero_score_experts():
    scores = [[5.0, 0.0], [4.0, 0.0]]
    bal = QuantileBalancer(n_experts=2, top_k=1)
    bal.update(scores)
    # 全 0 列 (兜底专家) 不参与均衡.
    assert bal.bias[1] == 0.0
    assert bal.bias[0] != 0.0


def test_route_batch_balances_expert_loads():
    # 20 条 query 都命中 math 且 math 分数最高; 不开均衡时 math 在
    # top-1 下全被激活. 每个 query 还带一个次要关键词, 使其余专家
    # 也有正分 (QB 只能在有正分的专家间重新分配负载).
    topics = ["python", "文章", "kakeya"]
    queries = [f"数学 证明 方程 求解 {i} {topics[i % 3]}" for i in range(20)]

    plain = ExpertRouter(top_k=1)
    math_plain = sum(
        1 for a in plain.route_batch(queries) if "math" in [e.name for e in a]
    )

    balanced = ExpertRouter(
        top_k=1, balancer=QuantileBalancer(n_experts=5, top_k=1)
    )
    batches = balanced.route_batch(queries)
    math_balanced = sum(1 for a in batches if "math" in [e.name for e in a])

    assert math_plain == 20
    assert math_balanced < math_plain
    assert math_balanced >= 1  # 未被完全压死


def test_mixture_weights_use_raw_scores_not_bias():
    pool = default_experts()
    router = ExpertRouter(
        experts=pool, top_k=2, balancer=QuantileBalancer(n_experts=5, top_k=2)
    )
    # ``balancer`` 声明是 ``Optional[QuantileBalancer]``（构造参数默认 None，
    # 但上面显式传了）。断言让"确实带了 balancer"在代码里可见，
    # 否则下一行的属性赋值会报 "None 没有 bias"。
    assert router.balancer is not None
    router.balancer.bias = [-3.0, 1.0, 0.0, 0.0, 0.0]
    query = "用 python 证明 数学 定理"
    activated = router.route(query)
    weights = router.mixture_weights(query, activated)
    total = sum(e.score(query) for e in activated)
    for e in activated:
        # 偏置只影响选择, 混合权重 p_i = s_i / Σs 用原始分 (论文 Eq.13).
        assert weights[e.name] == pytest.approx(e.score(query) / total)


# ---------------------------------------------------------------------------
# 共享专家 (Kimi K3 论文 §2.3 shared experts)
# ---------------------------------------------------------------------------


def test_shared_expert_always_active():
    router = ExpertRouter(top_k=1, shared_names=("general",))
    activated = router.route("用 Python 写代码")
    assert "general" in [e.name for e in activated]
    assert "code" in [e.name for e in activated]
    # 无匹配 query 时 shared 专家即兜底.
    activated2 = router.route("今天天气怎么样")
    assert "general" in [e.name for e in activated2]


def test_shared_expert_deduplicated():
    router = ExpertRouter(top_k=2, shared_names=("general",))
    activated = router.route("zzz 无匹配")
    names = [e.name for e in activated]
    assert names.count("general") == 1


# ---------------------------------------------------------------------------
# 推理强度网格 (Kimi K3 论文 §4.1.2 的思想, 启发式)
# ---------------------------------------------------------------------------


def test_estimate_effort():
    """``estimate_effort`` 的分档（2026-10-09 重标定后的语义）。

    * **low**：短问答 / 写作 / 闲聊（默认档）；
    * **high**：需解释/实现/给出结论但规模有限，或**长文本**（长度 ≠ 重推理）；
    * **max**：真正需要多步重推理的线索词（推导/反证/归纳/复杂度）。
    """
    # 短问答 → low
    assert estimate_effort("什么是挂谷猜想") == EffortLevel.LOW
    # 写作/摘要请求：不含重推理词、长度 <200 → low（旧版误判 high）
    mid = "帮我总结这段话的主要观点，要求简洁准确，不要遗漏关键信息，控制在三百字以内"
    assert 30 <= len(mid) <= 200
    assert estimate_effort(mid) == EffortLevel.LOW
    # 多步推理线索 → max
    assert estimate_effort("请详细证明这个定理并给出完整推导") == EffortLevel.MAX
    assert estimate_effort("用数学归纳法证明") == EffortLevel.MAX
    # **长文本不等于重推理**：长度 >200 只到 high（旧版误判 max）
    assert estimate_effort("证明" + "x" * 250) == EffortLevel.HIGH
    # 实现/解释类（有限规模的推理任务）→ high
    assert estimate_effort("实现一个 LRU 缓存") == EffortLevel.HIGH


def test_resolve_options_merges_effort_override():
    exp = Expert(
        name="t",
        keywords=["x"],
        system_prompt="",
        model="m",
        options={"temperature": 0.5},
        effort_options={"low": {"num_predict": 128, "temperature": 0.2}},
    )
    assert resolve_options(exp, "low") == {"temperature": 0.2, "num_predict": 128}
    assert resolve_options(exp, "high") == {"temperature": 0.5}
    assert resolve_options(exp, None) == {"temperature": 0.5}


def test_default_experts_define_effort_options():
    """默认专家必须定义 ``effort_options``——否则 ``effort`` 是空转（#3）。

    此前 ``route_and_answer(effort=...)`` 走 ``resolve_options``，但**没有专家
    定义 ``effort_options``** ⇒ effort 算了却不影响生成参数（dead code）。
    """
    for e in default_experts():
        assert e.effort_options, f"{e.name} 缺 effort_options"
        for lvl in ("low", "high", "max"):
            assert lvl in e.effort_options, f"{e.name} 缺 {lvl}"
            assert "num_predict" in e.effort_options[lvl]
        # low < max（低强度少给 token）
        assert e.effort_options["low"]["num_predict"] < e.effort_options["max"]["num_predict"]


def test_resolve_options_applies_effort_override():
    """``resolve_options`` 应把 effort 的 num_predict 覆盖进专家 options。"""
    exp = default_experts()[0]
    low = resolve_options(exp, "low")
    hi = resolve_options(exp, "max")
    assert low["num_predict"] < hi["num_predict"]
    # effort=None 时不含 num_predict 覆盖（保持专家默认）
    assert "num_predict" not in resolve_options(exp, None)


# ---------------------------------------------------------------------------
# 路由置信度（roadmap §4.4 #21，2026-10-30）
# ---------------------------------------------------------------------------


def _two_expert_router() -> ExpertRouter:
    """构造确定性双专家路由器：各一个关键词，互不重叠。"""
    return ExpertRouter(
        experts=[
            Expert(name="alpha", keywords=["甲虫"], system_prompt="s", model="m"),
            Expert(name="beta", keywords=["贝塔"], system_prompt="s", model="m"),
        ],
        always_include_general=False,
    )


def test_confidence_clear_single_hit():
    """单专家压倒性命中 ⇒ 置信度 1.0，不触发低置信。"""
    router = _two_expert_router()
    _, info = router.route_with_confidence("甲虫有几条腿")
    assert info.confidence == 1.0
    assert not info.is_low_confidence
    assert info.floor == CONFIDENCE_FLOOR


def test_confidence_zero_hit_is_pure_fallback():
    """关键词全不命中 ⇒ 无正分专家 ⇒ 置信度 0.0，必触发低置信。"""
    router = _two_expert_router()
    _, info = router.route_with_confidence("今天天气怎么样")
    assert info.confidence == 0.0
    assert info.is_low_confidence
    assert info.scores == {}, "兜底时 scores 应为空（general 不计入正分）"


def test_confidence_tie_is_ambiguous():
    """两专家均分（同分各命中一次）⇒ top-1 份额 0.5 < 阈值 ⇒ 模糊。"""
    router = _two_expert_router()
    _, info = router.route_with_confidence("甲虫和贝塔有什么关系")
    assert info.confidence == 0.5
    assert info.is_low_confidence


def test_confidence_matches_route_activation():
    """route_with_confidence 的激活结果必须与 route() 完全一致。"""
    router = _two_expert_router()
    for q in ("甲虫", "贝塔", "都不是"):
        activated, info = router.route_with_confidence(q)
        assert [e.name for e in activated] == list(info.activated)
        assert activated == router.route(q)


def test_confidence_floor_override():
    """floor 参数可覆盖标定阈值（测试/实验用）。"""
    router = _two_expert_router()
    _, info = router.route_with_confidence("甲虫", floor=1.01)
    assert info.is_low_confidence, "floor > 1 时任何有限置信度都应判低"
