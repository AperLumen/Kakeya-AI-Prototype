"""primitives.py 的测试。"""

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Shading, Tube, TubeCollection


def test_direction_normalization():
    d = Direction.from_array([2.0, 0.0, 0.0])
    assert np.allclose(d.v, [1.0, 0.0, 0.0])


def test_direction_angle():
    d1 = Direction.from_array([1.0, 0.0, 0.0])
    d2 = Direction.from_array([0.0, 1.0, 0.0])
    assert np.isclose(d1.angle_to(d2), np.pi / 2)


def test_tube_volume():
    t = Tube(
        center=np.array([0.0, 0.0, 0.0]),
        direction=Direction.from_array([0.0, 0.0, 1.0]),
        length=1.0,
        delta=0.1,
    )
    assert np.isclose(t.volume, np.pi * 0.1 ** 2 * 1.0)


def test_tube_contains():
    t = Tube(
        center=np.array([0.0, 0.0, 0.0]),
        direction=Direction.from_array([0.0, 0.0, 1.0]),
        length=1.0,
        delta=0.1,
    )
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.6], [0.2, 0.0, 0.0]])
    mask = t.contains(pts)
    assert mask[0]  # 中心
    assert mask[1]  # 沿轴向
    assert not mask[2]  # 半径之外


def test_shading_density_validation():
    with pytest.raises(ValueError):
        Shading(np.array([1.1, 0.5]))


def test_tube_collection_add():
    tc = TubeCollection()
    t = Tube(
        center=np.array([0.0, 0.0, 0.0]),
        direction=Direction.from_array([0.0, 0.0, 1.0]),
        length=1.0,
        delta=0.1,
    )
    tc.add(t, density=0.8)
    assert tc.count == 1
    assert tc.shading is not None
    assert np.isclose(tc.shading.densities[0], 0.8)
