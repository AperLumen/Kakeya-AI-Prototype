"""grains.py 的测试。"""

import numpy as np
import pytest

from kakeya_ai.errors import InputValidationError
from kakeya_ai.grains import (
    GrainsDecomposition,
    _segment_intersects_aabb,
    OrientedBoundingBox,
    Prism,
    broadness_at_points,
    flatten_two_scale_grains,
    grain_quality_metrics,
    guth_grains_skeleton,
    is_valid_grains_decomposition,
    oriented_bounding_box,
    partition_into_rho_tubes,
    rho_tube_cover,
    sample_broadness_points,
    two_scale_grains_invariants,
    two_scale_grains_skeleton,
)
from kakeya_ai.primitives import Direction, Tube


def make_fan(n: int = 12):
    tubes = []
    for theta in np.linspace(0.1, np.pi - 0.1, n):
        for phi in np.linspace(0, 2 * np.pi, n, endpoint=False):
            tubes.append(
                Tube(
                    center=np.zeros(3),
                    direction=Direction.from_spherical(theta, phi),
                    length=1.0,
                    delta=0.05,
                )
            )
    return tubes


def test_prism_rho():
    p = Prism.from_center_size([0, 0, 0], [0.1, 0.2, 1.0])
    # `rho` 是 `Prism` **自己的** property（= b/c）。此前写成
    # `p.bounding_box.rho`，而 `Prism`（及其父类 `Box`）**根本没有
    # `bounding_box` 属性**——该用例运行时 `AttributeError` 恒失败，
    # 一直没被发现（该文件长期没被跑过）。
    assert np.isclose(p.rho, 0.2)
    assert p.a <= p.b <= p.c


def test_partition_into_rho_tubes():
    tubes = make_fan(n=8)
    bins = partition_into_rho_tubes(tubes, rho=0.2)
    assert len(bins) > 0
    total = sum(len(v) for v in bins.values())
    assert total == len(tubes)  # 每根管都被恰好分配一次


def test_partition_into_rho_tubes_diameter_bound():
    tubes = make_fan(n=8)
    rho = 0.2
    cover = rho_tube_cover(tubes, rho)
    for info in cover.values():
        assert info["diameter"] <= rho + 1e-12
        # 检查中心方向是单位长度。
        assert np.isclose(np.linalg.norm(info["centre"]), 1.0)


def test_rho_tube_cover_is_complete():
    tubes = make_fan(n=6)
    cover = rho_tube_cover(tubes, rho=0.3)
    assigned = sorted(idx for info in cover.values() for idx in info["indices"])
    assert assigned == list(range(len(tubes)))


def test_guth_grains_skeleton():
    tubes = make_fan(n=6)
    grains = guth_grains_skeleton(tubes, None, grid_n=2)
    assert grains.count > 0
    assert all(isinstance(p, Prism) for p in grains.prisms)
    # 棱柱体积总和应不小于并集体积，且不超过 bbox 体积。
    assert 0 < grains.total_prism_volume <= 8.0


def test_two_scale_grains_skeleton():
    tubes = make_fan(n=6)
    result = two_scale_grains_skeleton(tubes, None, rho=0.2)
    assert len(result) > 0
    for grains in result.values():
        assert isinstance(grains, GrainsDecomposition)
        assert grains.count >= 0


def test_two_scale_grains_invariants():
    tubes = make_fan(n=6)
    result = two_scale_grains_skeleton(tubes, None, rho=0.3)
    for grains in result.values():
        inv = grains.invariants()
        assert inv["count"] >= 0
        assert inv["total_volume"] > 0
        assert 0 <= inv["min_rho"] <= inv["max_rho"] <= 1.0
        assert 0 < inv["min_c"] <= inv["max_c"]
        assert inv["max_local_kt_error"] >= 0
        assert inv["avg_tubes_per_grain"] >= 0
        assert inv["total_tubes_accounted"] >= 0
        assert 0.0 <= inv["avg_density"] <= 1.0


def test_two_scale_grains_respect_rho_diameter():
    """每个 ρ-bin 只应包含方向落在 ρ 之内的管。"""
    tubes = make_fan(n=8)
    rho = 0.25
    result = two_scale_grains_skeleton(tubes, None, rho=rho)
    from kakeya_ai.grains import partition_into_rho_tubes
    bins = partition_into_rho_tubes(tubes, rho)
    for bin_idx, grains in result.items():
        indices = set(grains.prisms[0].tube_indices) if grains.prisms else set()
        for p in grains.prisms:
            indices |= set(p.tube_indices)
        # bin 里的每个管索引都必须属于预期的 ρ-bin。
        assert indices.issubset(set(bins[bin_idx]))


def test_broadness_fan_is_broad():
    tubes = make_fan(n=8)
    points = np.random.default_rng(42).uniform(-0.3, 0.3, size=(20, 3))
    result = broadness_at_points(tubes, points, K=20.0, beta=1.0)
    # 铺开的扇形在宽松常数下应当算宽泛。
    assert result["global_score"] <= 1.0 + 1e-6


def test_broadness_parallel_is_not_broad():
    tubes = [
        Tube(
            center=np.array([0.0, y, 0.0]),
            direction=Direction.from_array([0.0, 0.0, 1.0]),
            length=1.0,
            delta=0.25,
        )
        for y in np.linspace(-0.4, 0.4, 8)
    ]
    points = np.array([[0.0, 0.0, 0.0]])
    result = broadness_at_points(tubes, points, K=1.0, beta=1.0)
    # 平行管的所有方向相同；多根管穿过同一点时，小半径处的帽会违反
    # 严格的 K=1 宽泛性界。
    assert not result["global_broad"]


def test_flatten_two_scale_grains():
    tubes = make_fan(n=6)
    two_scale = two_scale_grains_skeleton(tubes, None, rho=0.3)
    flat = flatten_two_scale_grains(two_scale)
    expected_count = sum(g.count for g in two_scale.values())
    assert flat.count == expected_count
    # 所有原始棱柱都应被保留。
    assert flat.total_tubes_accounted == sum(
        g.total_tubes_accounted for g in two_scale.values()
    )


def test_two_scale_grains_invariants_consistent_with_flattened():
    """展平后的两尺度 grains 与分组版本invariants 必须一致。

    回归（2026-09-17 由 ruff F811 发现）：本函数原名 ``test_two_scale_grains_invariants``，
    与上方第 86 行同名，导致**它从未被执行**——pytest 只收集到同名中的后定义者，
    前者被静默覆盖。改名后两个用例都会跑；这也是把规则集锁住 F811 的直接动机。
    """
    tubes = make_fan(n=6)
    two_scale = two_scale_grains_skeleton(tubes, None, rho=0.3)
    inv = two_scale_grains_invariants(two_scale)
    flat = flatten_two_scale_grains(two_scale)
    flat_inv = flat.invariants()
    assert inv["count"] == flat_inv["count"]
    assert inv["total_volume"] == flat_inv["total_volume"]
    assert inv["n_bins"] == len(two_scale)
    assert inv["max_local_kt_error"] == flat_inv["max_local_kt_error"]


def test_grain_quality_metrics_basic():
    tubes = make_fan(n=6)
    grains = guth_grains_skeleton(tubes, None, grid_n=2)
    metrics = grain_quality_metrics(grains, tubes)
    assert 0.0 <= metrics["volume_efficiency"] <= 1.0 + 1e-6
    assert metrics["overlap_ratio"] >= 1.0 - 1e-6
    assert metrics["mean_rho"] >= 0.0
    assert metrics["std_rho"] >= 0.0
    assert metrics["mean_c"] > 0.0
    assert metrics["std_c"] >= 0.0


def test_grain_quality_metrics_empty_grains():
    empty = GrainsDecomposition()
    metrics = grain_quality_metrics(empty)
    assert metrics["volume_efficiency"] == 0.0
    assert metrics["overlap_ratio"] == 0.0


def test_sample_broadness_points():
    tubes = make_fan(n=6)
    points = sample_broadness_points(tubes, n_points=30, seed=42)
    assert points.shape[1] == 3
    assert 0 < len(points) <= 30
    # 每个采样点都必须落在至少一根管内。
    for pt in points:
        assert any(t.contains(pt.reshape(1, -1))[0] for t in tubes)


def test_sample_broadness_points_empty():
    points = sample_broadness_points([], n_points=30)
    assert points.shape == (0, 3)


def test_is_valid_grains_decomposition():
    tubes = make_fan(n=6)
    grains = guth_grains_skeleton(tubes, None, grid_n=2)
    assert is_valid_grains_decomposition(grains, tubes)


def test_is_valid_grains_decomposition_empty():
    assert is_valid_grains_decomposition(GrainsDecomposition(), [])


class TestOrientedBoundingBox:
    """定向包围盒原语（局限 #4 的"旋转包围盒"部分）。"""

    def test_contains_all_tube_neighbourhoods(self):
        """OBB 必须包含每根管的 δ-邻域（端点 ±δ 在内）。"""
        d = Direction(np.array([1.0, 1.0, 1.0]))
        tubes = [
            Tube(center=np.array([0.05 * i, 0.05 * i, 0.05 * i]), direction=d, delta=0.02)
            for i in range(4)
        ]
        obb = oriented_bounding_box(tubes)
        for t in tubes:
            assert obb.contains(np.asarray(t.endpoints), margin=t.delta)

    def test_smaller_than_axis_aligned_for_diagonal(self):
        """对斜向单管，OBB 体积应显著小于轴对齐盒。"""
        from kakeya_ai.grains import _axis_aligned_bounding_box

        d = Direction(np.array([1.0, 1.0, 1.0]))
        tube = Tube(center=np.zeros(3), direction=d, delta=0.02)
        obb = oriented_bounding_box([tube])
        aabb = _axis_aligned_bounding_box([tube])
        assert obb.volume < aabb.volume
        assert aabb.volume / obb.volume > 3.0

    def test_principal_direction_matches_tube(self):
        d = Direction(np.array([0.0, 0.0, 1.0]))
        tube = Tube(center=np.zeros(3), direction=d, delta=0.03)
        obb = oriented_bounding_box([tube])
        assert np.allclose(np.abs(obb.principal_direction), [0, 0, 1])

    def test_axes_orthonormal(self):
        d = Direction(np.array([1.0, 2.0, 3.0]))
        obb = oriented_bounding_box([Tube(center=np.zeros(3), direction=d, delta=0.01)])
        assert np.allclose(obb.axes @ obb.axes.T, np.eye(3), atol=1e-9)

    def test_empty_returns_unit_box(self):
        obb = oriented_bounding_box([])
        assert isinstance(obb, OrientedBoundingBox)
        assert obb.volume > 0

    def test_to_local_world_roundtrip(self):
        d = Direction(np.array([1.0, 1.0, 2.0]))
        tube = Tube(center=np.array([0.1, -0.2, 0.3]), direction=d, delta=0.02)
        obb = oriented_bounding_box([tube])
        pts = np.array([[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]])
        assert np.allclose(obb.to_world(obb.to_local(pts)), pts)


class TestOrientedGrainsWiring:
    """局限 #4：定向 grain（``OrientedPrism``）接入 grains 构造路径。

    先前负结果：把格子建在定向系、按世界 AABB 汇报体积 ⇒ 总体积放大 ~30×。
    本轮把汇报口径改成**定向盒自身体积**并接入 ``guth_grains_skeleton``；
    本组钉住"体积不再虚胖、且轴对齐管族下与轴对齐路径逐位一致"。
    """

    def _diagonal_family(self, n=8, delta=0.05):
        d = Direction(np.array([1.0, 1.0, 1.0]))
        return [
            Tube(center=np.array([0.02 * i, 0.01 * i, 0.0]), direction=d,
                 length=1.0, delta=delta)
            for i in range(n)
        ]

    def test_oriented_prism_reports_own_volume_not_world_aabb(self):
        """核心口径：volume = 定向盒体积，而非世界 AABB 体积。"""
        from kakeya_ai.grains import OrientedPrism

        g = OrientedPrism(
            center=np.zeros(3),
            axes=np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) / np.sqrt(2),
            extents=np.array([0.1, 0.2, 1.0]),
        )
        assert g.volume == pytest.approx(0.1 * 0.2 * 1.0)
        # 世界 AABB 体积必须明显更大（斜盒），且 bounds 只是保守外扩。
        b = g.bounds
        aabb_vol = float(np.prod(b[:, 1] - b[:, 0]))
        assert aabb_vol > g.volume
        # a ≤ b ≤ c 的排序不变量。
        assert g.a <= g.b <= g.c

    def test_oriented_frame_matches_axis_frame_for_axis_aligned_tubes(self):
        """管族本身就轴对齐时，两条路径必须给出**逐位一致**的结果。"""
        d = Direction(np.array([0.0, 0.0, 1.0]))
        tubes = [
            Tube(center=np.array([0.05 * i, 0.03 * i, 0.0]), direction=d,
                 length=1.0, delta=0.05)
            for i in range(6)
        ]
        ax = guth_grains_skeleton(tubes, None, grid_n=4, frame="axis")
        ori = guth_grains_skeleton(tubes, None, grid_n=4, frame="oriented")
        assert ori.count == ax.count
        assert ori.total_prism_volume == pytest.approx(ax.total_prism_volume, rel=1e-9)
        assert ori.max_local_kt_error == pytest.approx(ax.max_local_kt_error, rel=1e-9)

    def test_oriented_frame_tightens_volume_for_diagonal_tubes(self):
        """斜向管族：定向 grain 总体积必须**小于**轴对齐（修复虚胖），
        且 KT proxy 因此**更大**（轴对齐盒把斜向 grain 包得太松）。"""
        tubes = self._diagonal_family()
        ax = guth_grains_skeleton(tubes, None, grid_n=4, frame="axis")
        ori = guth_grains_skeleton(tubes, None, grid_n=4, frame="oriented")
        assert ori.count > 0 and ax.count > 0
        # 关键：**没有**放大（负结果的病征）。
        assert ori.total_prism_volume < ax.total_prism_volume
        assert ori.max_local_kt_error > ax.max_local_kt_error

    def test_oriented_grains_are_actually_oriented_and_contain_their_tubes(self):
        """建出的确实是 ``OrientedPrism``，且每根登记在册的管**确实与该 grain 相交**
        （grain 是网格 cell，管横跨多个 cell——与轴对齐路径同语义）。"""
        from kakeya_ai.grains import OrientedPrism

        tubes = self._diagonal_family()
        ori = guth_grains_skeleton(tubes, None, grid_n=4, frame="oriented")
        assert ori.prisms
        assert all(isinstance(p, OrientedPrism) for p in ori.prisms)
        assert ori.is_axis_aligned is False
        for g in ori.prisms:
            # `prisms` 的元素类型是 `Prism | OrientedPrism`（同一个容器两种 grain，
            # 见 grains.py）；上面那句 `all(isinstance(...))` **不会**把类型收窄传给
            # 循环变量，故这里逐项再断言一次——既收窄类型，也把假设写成检查。
            assert isinstance(g, OrientedPrism)
            assert g.tube_indices, "空 grain 不应被建出"
            half = 0.5 * g.extents
            for idx in g.tube_indices:
                a, b = tubes[idx].endpoints
                d = tubes[idx].delta
                pts = g.to_local(np.vstack([a, b]))
                lo, hi = -half - d, half + d
                # 精确的线段–盒相交（与构造时同一判定），在局部坐标下。
                assert _segment_intersects_aabb(
                    pts[0, 0], pts[0, 1], pts[0, 2],
                    pts[1, 0], pts[1, 1], pts[1, 2],
                    lo[0], hi[0], lo[1], hi[1], lo[2], hi[2],
                )

    def test_axis_frame_is_default_and_rejects_bad_frame(self):
        tubes = self._diagonal_family(4)
        default = guth_grains_skeleton(tubes, None, grid_n=3)
        explicit = guth_grains_skeleton(tubes, None, grid_n=3, frame="axis")
        assert default.is_axis_aligned and explicit.is_axis_aligned
        assert default.total_prism_volume == pytest.approx(explicit.total_prism_volume)
        with pytest.raises(InputValidationError):
            guth_grains_skeleton(tubes, None, grid_n=3, frame="nonsense")

    def test_two_scale_preserves_oriented_grain_type(self):
        """两尺度路径必须**原样保留** grain 类型，不得静默降级成 Prism。"""
        from kakeya_ai.grains import OrientedPrism, two_scale_grains_skeleton

        tubes = self._diagonal_family(6)
        res = two_scale_grains_skeleton(tubes, None, rho=0.5, frame="oriented")
        prisms = [p for g in res.values() for p in g.prisms]
        assert prisms, "两尺度分解不应为空"
        assert all(isinstance(p, OrientedPrism) for p in prisms)
        # 轴对齐路径仍是 Prism。
        res_ax = two_scale_grains_skeleton(tubes, None, rho=0.5)
        assert all(
            isinstance(p, Prism) for g in res_ax.values() for p in g.prisms
        )

    def test_decomposition_accepts_plain_prism_list(self):
        """构造参数接受 ``list[Prism]``（协变 ``Iterable``）——类型收敛的回归。

        回归背景：``prisms`` 字段是 ``list[Prism | OrientedPrism]``，若构造参数也
        声明成该 ``list``，``GrainsDecomposition([<Prism>, ...])`` 会被静态检查器
        判为非法（``list`` 不变），实测在 ``moves.py`` 的 Move 循环里报了几十处。
        构造参数改为 ``Iterable[Prism | OrientedPrism]``（协变）后消除。
        """
        d = Direction(np.array([0.0, 0.0, 1.0]))
        grains = guth_grains_skeleton(
            [Tube(center=np.zeros(3), direction=d, delta=0.05)],
            None, grid_n=3, frame="axis",
        )
        plain = list(grains.prisms)          # list[Prism]
        rebuilt = GrainsDecomposition(plain)  # 必须可构造
        assert rebuilt.count == grains.count
        assert rebuilt.is_axis_aligned is True
        assert GrainsDecomposition(prisms=plain).count == grains.count
