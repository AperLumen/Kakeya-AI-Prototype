"""volume.py 的测试。"""

import numpy as np

from kakeya_ai.primitives import Direction, Shading, Tube, TubeCollection
from kakeya_ai.volume import (
    assertion_d_lower_bound,
    estimate_shaded_union_volume_voxel,
    estimate_union_volume_monte_carlo,
    estimate_union_volume_monte_carlo_with_ci,
    estimate_union_volume_voxel,
    estimate_volume_vs_theorem_1_2,
    monte_carlo_confidence_interval,
    theorem_1_2_lower_bound,
)


def test_disjoint_tubes_volume():
    """互不相交的平行管：并集体积 ≈ 各体积之和。"""
    tubes = []
    for y in [-0.5, 0.0, 0.5]:
        tubes.append(
            Tube(
                center=np.array([0.0, y, 0.0]),
                direction=Direction.from_array([0.0, 0.0, 1.0]),
                length=0.5,
                delta=0.05,
            )
        )
    vol = estimate_union_volume_voxel(tubes, grid_n=256)
    total = sum(t.volume for t in tubes)
    # 允许 ~15% 的离散化误差。
    assert vol > 0.8 * total


def test_shaded_union_smaller_than_full():
    """λ=0.5 的遮蔽并集不应大于完整并集。"""
    tubes = []
    for y in [-0.5, 0.0, 0.5]:
        tubes.append(
            Tube(
                center=np.array([0.0, y, 0.0]),
                direction=Direction.from_array([0.0, 0.0, 1.0]),
                length=0.5,
                delta=0.05,
            )
        )
    full = estimate_union_volume_voxel(tubes, grid_n=256)
    shaded = estimate_shaded_union_volume_voxel(
        tubes, Shading(np.array([0.5, 0.5, 0.5])), grid_n=256
    )
    assert 0 < shaded <= full


def test_theorem_1_2_lower_bound_positive():
    tubes = [Tube(center=np.zeros(3), direction=Direction.from_array([0.0, 0.0, 1.0]), length=1.0, delta=0.1)]
    bound = theorem_1_2_lower_bound(tubes, epsilon=0.01, K=2.0, lambda_min=0.5)
    assert bound > 0


def test_assertion_d_lower_bound():
    tubes = [Tube(center=np.zeros(3), direction=Direction.from_array([0.0, 0.0, 1.0]), length=1.0, delta=0.1)]
    bound = assertion_d_lower_bound(tubes, sigma=0.5, omega=0.0, epsilon=0.01, kappa=1.0)
    assert bound > 0


def test_monte_carlo_reasonable():
    tubes = []
    for y in [-0.5, 0.0, 0.5]:
        tubes.append(
            Tube(
                center=np.array([0.0, y, 0.0]),
                direction=Direction.from_array([0.0, 0.0, 1.0]),
                length=0.5,
                delta=0.05,
            )
        )
    vol_mc = estimate_union_volume_monte_carlo(tubes, n_samples=100_000, seed=42)
    total = sum(t.volume for t in tubes)
    assert 0.7 * total < vol_mc < 1.3 * total



def test_monte_carlo_confidence_interval_bounds():
    """置信区间必须包含估计值，且落在 [0, V] 之内。"""
    n_inside, n_samples, box_volume = 100, 1000, 8.0
    estimate, lower, upper = monte_carlo_confidence_interval(
        n_inside, n_samples, box_volume, confidence=0.95
    )
    assert 0.0 <= lower <= estimate <= upper <= box_volume


def test_monte_carlo_with_ci_matches_point_estimate():
    """`estimate_union_volume_monte_carlo_with_ci` 的估计值与朴素 MC 估计器一致。"""
    tubes = []
    for y in [-0.5, 0.0, 0.5]:
        tubes.append(
            Tube(
                center=np.array([0.0, y, 0.0]),
                direction=Direction.from_array([0.0, 0.0, 1.0]),
                length=0.5,
                delta=0.05,
            )
        )
    point = estimate_union_volume_monte_carlo(tubes, n_samples=100_000, seed=42)
    with_ci = estimate_union_volume_monte_carlo_with_ci(
        tubes, n_samples=100_000, seed=42
    )
    assert np.isclose(point, with_ci["estimate"], rtol=1e-12)
    assert 0.0 <= with_ci["relative_error"] < 1.0


def test_monte_carlo_with_ci_empty_tubes():
    result = estimate_union_volume_monte_carlo_with_ci([], n_samples=1000)
    assert result["estimate"] == 0.0
    assert result["n_samples"] == 0


def test_monte_carlo_with_ci_nano_backend():
    """``backend="nano"`` 走融合采样：同 seed 逐位确定，统计上贴 numpy 参考。

    C++ 扩展不可用时 skip（环境能力，不是代码错）。
    """
    from kakeya_ai.nano_kernel import is_cpp_available

    if not is_cpp_available():
        import pytest

        pytest.skip("C++ 扩展不可用")

    tubes = []
    for y in [-0.5, 0.0, 0.5]:
        tubes.append(
            Tube(
                center=np.array([0.0, y, 0.0]),
                direction=Direction.from_array([0.0, 0.0, 1.0]),
                length=0.5,
                delta=0.05,
            )
        )
    a = estimate_union_volume_monte_carlo_with_ci(
        tubes, n_samples=500_000, seed=42, backend="nano"
    )
    b = estimate_union_volume_monte_carlo_with_ci(
        tubes, n_samples=500_000, seed=42, backend="nano"
    )
    assert a["n_inside"] == b["n_inside"], "同 seed 计数必须逐位相同"
    assert a["estimate"] == b["estimate"]

    ref = estimate_union_volume_monte_carlo(
        tubes, n_samples=500_000, seed=42
    )
    assert abs(a["estimate"] - ref) < 0.005, (
        f"融合采样 {a['estimate']} 与 numpy 参考 {ref} 超统计容差"
    )
    assert 0.0 <= a["relative_error"] < 1.0


def test_monte_carlo_with_ci_nano_requires_seed():
    """``backend="nano"`` 缺 seed 必须显式报错（不静默给 numpy 结果）。"""
    import pytest

    from kakeya_ai.errors import InputValidationError

    tubes = [
        Tube(
            center=np.zeros(3),
            direction=Direction.from_array([0.0, 0.0, 1.0]),
            length=0.5,
            delta=0.05,
        )
    ]
    with pytest.raises(InputValidationError):
        estimate_union_volume_monte_carlo_with_ci(
            tubes, n_samples=1000, backend="nano"
        )


def test_estimate_volume_vs_theorem_1_2_shape():
    tubes = [
        Tube(
            center=np.zeros(3),
            direction=Direction.from_array([0.0, 0.0, 1.0]),
            length=1.0,
            delta=0.1,
        )
    ]
    comparison = estimate_volume_vs_theorem_1_2(
        tubes, epsilon=0.01, K=2.0, lambda_min=0.5, n_samples=20_000, seed=42
    )
    assert "volume_estimate" in comparison
    assert "theorem_1_2_lower_bound" in comparison
    assert isinstance(comparison["satisfies"], bool)
