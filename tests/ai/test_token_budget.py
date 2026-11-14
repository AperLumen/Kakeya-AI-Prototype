"""动态 token 预算辅助函数的测试。"""

from kakeya_ai.ai.token_budget import compute_deep_budget, compute_uncertainty_budget


def test_deep_budget_rewards_high_confidence():
    low = compute_deep_budget(confidence=1.0, base=1024)
    high = compute_deep_budget(confidence=9.0, base=1024)
    assert low > high


def test_deep_budget_respects_bounds():
    assert compute_deep_budget(confidence=0.0, min_tokens=256, max_tokens=2048) >= 256
    assert compute_deep_budget(confidence=10.0, min_tokens=256, max_tokens=2048) >= 256
    assert compute_deep_budget(confidence=0.0, min_tokens=256, max_tokens=2048) <= 2048


def test_uncertainty_budget_rewards_large_gap():
    uncertain = compute_uncertainty_budget(top_two_gap=0.5, base=1024)
    certain = compute_uncertainty_budget(top_two_gap=8.0, base=1024)
    assert uncertain > certain


def test_uncertainty_budget_within_bounds():
    budget = compute_uncertainty_budget(top_two_gap=100.0, base=1024, min_tokens=256, max_tokens=2048)
    assert 256 <= budget <= 2048
