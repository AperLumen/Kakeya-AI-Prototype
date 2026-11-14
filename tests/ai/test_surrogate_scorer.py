"""代理打分器与 Nesterov 优化器的测试。"""

import numpy as np
import pytest

from kakeya_ai.ai import (
    NesterovOptimizer,
    SurrogateScorer,
)
from kakeya_ai.ai.training_data import (
    FEATURE_NAMES,
    build_training_data,
    heuristic_labels,
    sample_states,
)


#: 特征维数的唯一真源：测试不硬编码 4，否则扩维后测试仍“通过”但没覆盖新路径。
N_FEATURES = len(FEATURE_NAMES)


#: 一组包含全部 ``FEATURE_NAMES`` 键的中性 state（旧测试只给 4 个键，
#: 缺失的靠默认值补齐也能跑，但显式给全更能反映真实调用）。
def _neutral_state():
    return {
        "rho": 0.5,
        "delta": 0.01,
        "local_error": 1.0,
        "volume_ratio": 1.0,
        "log_n_tubes": 0.0,
        "n_grains_norm": 0.0,
        "sticky": 0.0,
        "aspect": 1.0,
    }


def test_surrogate_forward_shape():
    model = SurrogateScorer(input_dim=N_FEATURES, hidden_dims=[8], output_dim=3, seed=0)
    X = np.random.randn(10, N_FEATURES)
    out = model.forward(X)
    assert out.shape == (10, 3)


def test_surrogate_predict_scores_normalized():
    model = SurrogateScorer(input_dim=N_FEATURES, hidden_dims=[8], output_dim=3, seed=0)
    state = _neutral_state()
    scores = model.predict_scores(state)
    assert set(scores.keys()) == {1, 2, 3}
    for v in scores.values():
        assert 0.0 <= v <= 10.0


def test_surrogate_confidence():
    model = SurrogateScorer(input_dim=N_FEATURES, hidden_dims=[8], output_dim=3, seed=0)
    state = _neutral_state()
    scores = model.predict_scores(state)
    gap = model.confidence(scores)
    assert gap >= 0.0


def test_nesterov_faster_than_gd_on_quadratic():
    """在简单二次函数上，Nesterov 应比朴素 GD 用更少的步数收敛。"""

    def loss_and_grad(w):
        target = np.array([3.0, -2.0])
        diff = w - target
        loss = 0.5 * np.sum(diff**2)
        grad = diff
        return loss, grad

    def steps_to_threshold(mu, threshold=1e-4, max_steps=200):
        w = np.array([0.0, 0.0])
        opt = NesterovOptimizer(lr=0.05, mu=mu)
        for step in range(max_steps):
            lookahead = w + opt.mu * opt._velocity.get(0, np.zeros_like(w))
            _, grad = loss_and_grad(lookahead)
            params = {0: w.copy()}
            opt.step(params, {0: grad})
            w = params[0]
            loss, _ = loss_and_grad(w)
            if loss < threshold:
                return step
        return max_steps

    gd_steps = steps_to_threshold(mu=0.0)
    nesterov_steps = steps_to_threshold(mu=0.9)
    # Nesterov 应当以严格更少的步数达到阈值。
    assert nesterov_steps < gd_steps


def test_surrogate_training_reduces_loss():
    X, Y, _ = build_training_data(n_states=300, seed=42, use_llm_if_available=False)
    model = SurrogateScorer(input_dim=X.shape[1], hidden_dims=[8], output_dim=3, seed=42)
    optimizer = NesterovOptimizer(lr=1e-3, mu=0.9, max_grad_norm=1.0)
    history = model.fit(
        X,
        Y,
        optimizer=optimizer,
        epochs=500,
        val_split=0.1,
        verbose=False,
        seed=42,
    )
    assert history["train_loss"][-1] < history["train_loss"][0]
    # 验证损失不应爆炸。
    assert np.isfinite(history["val_loss"][-1])


def test_surrogate_save_and_load(tmp_path):
    model = SurrogateScorer(input_dim=N_FEATURES, hidden_dims=[8], output_dim=3, seed=0)
    state = _neutral_state()
    before = model.predict_scores(state)

    path = tmp_path / "weights.json"
    model.save(str(path))
    loaded = SurrogateScorer.load(str(path))
    after = loaded.predict_scores(state)

    assert loaded.input_dim == model.input_dim
    assert loaded.hidden_dims == model.hidden_dims
    assert loaded.output_dim == model.output_dim
    for k in before:
        assert before[k] == pytest.approx(after[k], abs=1e-6)


def test_heuristic_labels_shape_and_noise():
    states = sample_states(10, seed=0)
    Y = heuristic_labels(states, noise=0.0, seed=0)
    assert Y.shape == (10, 3)
    assert np.all((Y >= 0) & (Y <= 10))


# ---------------------------------------------------------------------------
# 训练端与推理端的特征顺序必须一致（扩维后最大的风险点）
# ---------------------------------------------------------------------------
#
# 此前训练端（build_training_data 里手写 4 个字段）与推理端
# （SurrogateScorer._state_to_vector 里再手写一遍）各写一份特征列表。两份
# 一旦顺序/成员漂移，模型会拿着"第 3 列是 local_error"的权重去读
# "第 3 列是 volume_ratio"的输入——**不报错，只是分数变得无意义**。
# 现在两处都走 FEATURE_NAMES，这组测试把它钉住。


def test_training_columns_follow_feature_names():
    """X 的列顺序必须与 ``FEATURE_NAMES`` 一一对应（用可辨识的取值验证）。"""
    from kakeya_ai.ai.training_data import state_to_features

    # 构造一个每个特征取值互不相同的 state，逐一核对列位置。
    probe = {
        "rho": 0.11,
        "delta": 0.22,
        "local_error": 0.33,
        "volume_ratio": 0.44,
        "log_n_tubes": 0.55,
        "n_grains_norm": 0.66,
        "sticky": 0.77,
        "aspect": 0.88,
    }
    cols = state_to_features(probe)
    assert cols == [probe[name] for name in FEATURE_NAMES]


def test_inference_vector_order_matches_training():
    """推理端 ``_state_to_vector`` 的取值顺序必须与训练端一致。

    做法：给模型装上"恒等"统计量（mean=0, std=1）使标准化成为恒等映射，
    再喂一个逐特征递增的 state；输出向量的分量顺序应等于 ``FEATURE_NAMES``
    的顺序。若两处顺序不一致，这个断言会失败。
    """
    probe_values = [0.1 * (i + 1) for i in range(N_FEATURES)]
    probe = dict(zip(FEATURE_NAMES, probe_values, strict=True))

    model = SurrogateScorer(
        input_dim=N_FEATURES, hidden_dims=[8], output_dim=3, seed=0,
        feature_mean=np.zeros(N_FEATURES),
        feature_std=np.ones(N_FEATURES),
    )
    vec = model._state_to_vector(probe)
    np.testing.assert_allclose(vec, probe_values)


def test_state_from_metrics_fills_all_features():
    """``state_from_metrics`` 必须产出每个 ``FEATURE_NAMES`` 键。"""
    from kakeya_ai.ai.training_data import state_from_metrics

    class _M:
        rho = 0.4
        delta = 0.02
        local_error = 1.7
        volume_ratio = 1.1
        n_tubes = 25
        n_grains = 5
        is_sticky = True
        a = 0.01
        c = 0.16

    state = state_from_metrics(_M())
    assert set(state.keys()) == set(FEATURE_NAMES)
    assert state["log_n_tubes"] == pytest.approx(np.log1p(25))
    assert state["n_grains_norm"] == pytest.approx(5 / 25)
    assert state["sticky"] == 1.0
    assert state["aspect"] == pytest.approx(16.0)


def test_feature_names_prefix_is_historical_four():
    """前 4 个特征名与顺序**不可变**——旧权重文件按它解释。

    这条测试是"向后兼容"的守门：谁把 ``local_error`` 挪到第 2 位，
    旧权重就会被按错误的列解读。
    """
    assert FEATURE_NAMES[:4] == ("rho", "delta", "local_error", "volume_ratio")
