"""AI 快思考优化的测试。"""

from kakeya_ai.ai.cache import DecisionCache
from kakeya_ai.ai.heuristics import heuristic_move_scores, should_call_llm


def test_heuristic_move_scores():
    scores = heuristic_move_scores(None, rho=0.9, delta=0.01, local_error=8.0, volume_ratio=0.9)
    assert 0 <= scores[1] <= 10
    assert 0 <= scores[2] <= 10
    assert 0 <= scores[3] <= 10
    # 高 rho 与高局部误差应偏向 move 2 与 move 3。
    assert scores[2] > scores[1]
    assert scores[3] > scores[1]


def test_should_call_llm_when_close():
    scores = {1: 6.0, 2: 5.5, 3: 4.0}
    assert should_call_llm(scores, margin=2.0) is True


def test_should_not_call_llm_when_clear():
    scores = {1: 9.0, 2: 4.0, 3: 3.0}
    assert should_call_llm(scores, margin=2.0) is False


def test_decision_cache():
    cache = DecisionCache()
    state = {"delta": 0.01, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.2}
    result = {"move_1": 3, "move_2": 5, "move_3": 7}
    cache.set(state, result)
    assert cache.get(state) == result
    assert cache.size == 1
