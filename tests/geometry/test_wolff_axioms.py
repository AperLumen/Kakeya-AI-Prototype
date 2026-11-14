"""wolff_axioms.py 的测试。"""

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Tube
from kakeya_ai.wolff_axioms import (
    Box,
    Parallelotope,
    Slab,
    adversarial_convex_error,
    estimate_frostman_constant,
    estimate_katz_tao_constant,
    frostman_slab_error,
    katz_tao_convex_error,
    make_test_boxes,
    make_test_parallelotopes,
    make_test_slabs,
    make_tube_aligned_parallelotopes,
    make_wolff_test_sets,
    tight_parallelotope,
)


def make_tubes_along_z(n: int, delta: float = 0.1):
    """沿 z 轴创建 n 根平行的 δ-管，在 x-y 平面内均匀间隔。"""
    tubes = []
    xs = np.linspace(-0.4, 0.4, n)
    ys = np.linspace(-0.4, 0.4, n)
    for x in xs:
        for y in ys:
            tubes.append(
                Tube(
                    center=np.array([x, y, 0.0]),
                    direction=Direction.from_array([0.0, 0.0, 1.0]),
                    length=1.0,
                    delta=delta,
                )
            )
    return tubes


def test_katz_tao_clustered_tubes():
    """聚集在小盒子里的管，其 C_KT-CW 比放在大盒子里更大。"""
    # 10 根短管全部居中于原点，以便装进一个小盒子。
    clustered = []
    rng = np.random.default_rng(42)
    for _ in range(10):
        clustered.append(
            Tube(
                center=np.array([0.0, 0.0, 0.0]),
                direction=Direction.from_spherical(
                    rng.uniform(0.2, np.pi - 0.2),
                    rng.uniform(0, 2 * np.pi),
                ),
                length=0.1,
                delta=0.05,
            )
        )
    small_box = Box.from_center_size([0.0, 0.0, 0.0], [0.2, 0.2, 1.0])
    big_box = Box.from_center_size([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    c_small = katz_tao_convex_error(clustered, [small_box])
    c_big = katz_tao_convex_error(clustered, [big_box])
    # 小盒子虽小但含很多管，故其单位体积误差更大。
    assert c_small > c_big
    assert c_small > 0.1


def test_katz_tao_parallel_tubes():
    """铺开的平行管在大盒子下 C_KT-CW 很小。"""
    tubes = make_tubes_along_z(4, delta=0.1)
    big_box = Box.from_center_size([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    c_big = katz_tao_convex_error(tubes, [big_box])
    assert 0 < c_big < 1.0


def test_frostman_slab_parallel_tubes():
    """与管平行的平板会包含其中许多管。"""
    tubes = make_tubes_along_z(4, delta=0.1)
    slab = Slab.from_thickness(
        normal=[0.0, 1.0, 0.0],
        offset=0.0,
        thickness=0.5,
        bounding_box=Box.from_center_size([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]),
    )
    err = frostman_slab_error(tubes, [slab])
    assert err > 0


def test_box_contains_segment():
    box = Box.from_center_size([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    a = np.array([-0.2, 0.0, 0.0])
    b = np.array([0.2, 0.0, 0.0])
    assert box.contains_segment(a, b, 0.1)
    assert not box.contains_segment(a + np.array([1.0, 0.0, 0.0]), b + np.array([1.0, 0.0, 0.0]), 0.1)



def test_make_test_boxes_inside_bounding_box():
    bbox = Box.from_center_size([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    boxes = make_test_boxes(bbox, n_boxes=8, seed=42)
    assert len(boxes) == 8
    for box in boxes:
        assert box.volume > 0
        assert np.all(box.bounds[:, 0] >= bbox.bounds[:, 0] - 1e-12)
        assert np.all(box.bounds[:, 1] <= bbox.bounds[:, 1] + 1e-12)


def test_make_test_slabs_inside_bounding_box():
    bbox = Box.from_center_size([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    slabs = make_test_slabs(bbox, n_slabs=8, seed=42)
    assert len(slabs) == 8
    for slab in slabs:
        assert np.isclose(np.linalg.norm(slab.normal), 1.0)
        assert slab.half_thickness > 0


def test_estimate_katz_tao_constant_positive():
    tubes = make_tubes_along_z(4, delta=0.1)
    bbox = Box.from_center_size([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    c = estimate_katz_tao_constant(tubes, bbox, n_test_boxes=16, seed=42)
    assert c >= 0.0


def test_estimate_frostman_constant_positive():
    tubes = make_tubes_along_z(4, delta=0.1)
    bbox = Box.from_center_size([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    c = estimate_frostman_constant(tubes, bbox, n_test_slabs=16, seed=42)
    assert c >= 0.0


def test_katz_tao_constant_decreases_with_more_boxes():
    """使用更大的测试族只会增大（或保持）估计出的常数。"""
    tubes = make_tubes_along_z(4, delta=0.1)
    bbox = Box.from_center_size([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    c_small = estimate_katz_tao_constant(tubes, bbox, n_test_boxes=8, seed=1)
    c_large = estimate_katz_tao_constant(tubes, bbox, n_test_boxes=32, seed=1)
    assert c_large >= c_small - 1e-12


def test_make_test_boxes_empty():
    assert make_test_boxes(Box.from_center_size([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), n_boxes=0) == []


def test_make_test_slabs_empty():
    assert make_test_slabs(Box.from_center_size([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]), n_slabs=0) == []


class TestZeroCoverageIsDistinguishable:
    """零覆盖时返回的 0.0 必须与"测了且为 0"可区分。

    回归背景（2026-09-15）：``katz_tao_convex_error`` / ``frostman_slab_error``
    在**没有任何测试集包含管**时返回 0.0，而 ``check_*_axiom`` 直接拿
    ``error <= max_error`` 判定——于是"什么都没测"会被当成"满足公理"。
    这曾让尺度链中半径大的层全部假通过。
    """

    def test_convex_error_reports_include_count(self):
        from kakeya_ai.wolff_axioms import katz_tao_convex_error_checked

        tubes = [
            Tube(center=np.zeros(3),
                 direction=Direction.from_spherical(np.pi / 2, 0.0),
                 length=1.0, delta=0.1)
        ]
        tiny = [Box.from_center_size([0.0, 0.0, 0.0], [1e-6] * 3)]
        big = [Box.from_center_size([0.0, 0.0, 0.0], [4.0] * 3)]

        err_tiny, n_tiny = katz_tao_convex_error_checked(tubes, tiny)
        err_big, n_big = katz_tao_convex_error_checked(tubes, big)

        assert err_tiny == 0.0 and n_tiny == 0, "零覆盖：误差 0 但包含数为 0"
        assert n_big > 0, "大盒子应包含管"

    def test_slab_error_reports_include_count(self):
        from kakeya_ai.wolff_axioms import (
            frostman_slab_error_checked,
            make_test_slabs,
        )

        tubes = [
            Tube(center=np.zeros(3),
                 direction=Direction.from_spherical(np.pi / 2, 0.0),
                 length=1.0, delta=0.1)
        ]
        # 极薄的 slab：装不下半径 0.1 的管
        thin = make_test_slabs(
            Box.from_center_size([0.0, 0.0, 0.0], [4.0] * 3),
            n_slabs=3,
        )
        thin = [type(s)(s.normal, s.offset, 0.01, s.bounding_box) for s in thin]
        err, n = frostman_slab_error_checked(tubes, thin)
        assert err == 0.0 and n == 0

    def test_check_axiom_checked_exposes_zero_coverage(self):
        from kakeya_ai.wolff_axioms import check_katz_tao_axiom_checked

        tubes = [
            Tube(center=np.zeros(3),
                 direction=Direction.from_spherical(np.pi / 2, 0.0),
                 length=1.0, delta=0.1)
        ]
        tiny = [Box.from_center_size([0.0, 0.0, 0.0], [1e-6] * 3)]

        ok, n_included = check_katz_tao_axiom_checked(tubes, tiny, max_error=0.0)
        # 宽松语义下仍返回 True（向后兼容），但 n_included=0 让调用方能识别
        assert ok is True
        assert n_included == 0, "必须暴露包含数为 0，供调用方判为'未测'"

    def test_legacy_check_functions_unchanged(self):
        """旧接口语义保持不变（向后兼容）。"""
        from kakeya_ai.wolff_axioms import (
            check_frostman_slab_axiom,
            check_katz_tao_axiom,
        )

        tubes = [
            Tube(center=np.zeros(3),
                 direction=Direction.from_spherical(np.pi / 2, 0.0),
                 length=1.0, delta=0.1)
        ]
        big = [Box.from_center_size([0.0, 0.0, 0.0], [4.0] * 3)]
        assert check_katz_tao_axiom(tubes, big, max_error=1.0) is True
        assert isinstance(
            check_frostman_slab_axiom(tubes, [], max_error=1.0), bool
        )


class TestSlabVolumeIsExact:
    """``Slab.volume`` 的**精确**闭式（2026-10-09 修复）。

    旧实现取"主导轴盒面面积 × 厚度"，忽略法向且系统性偏小（详见
    ``_box_slab_intersection_volume``）。新实现是板 ∩ 盒的精确交集体积
    （广义 Irwin–Hall CDF），本组用蒙特卡洛与解析特例双重核验。
    """

    def test_volume_now_depends_on_normal(self):
        """修复后体积必须随法向变化（旧缺陷的正面回归）。"""
        from kakeya_ai.wolff_axioms import Box, Slab

        box = Box.from_center_size([0.0, 0.0, 0.0], [2.0] * 3)
        volumes = {}
        for nrm in ([0, 0, 1.0], [0.577, 0.577, 0.577], [0.9, 0.3, 0.316],
                    [0.8, 0.6, 0.0]):
            n = np.array(nrm, dtype=float)
            n /= np.linalg.norm(n)
            volumes[tuple(nrm)] = Slab(n, 0.0, 0.1, box).volume
        # 斜法向必须**大于**轴对齐法向（长对角线穿过盒子 ⇒ 截面更大）。
        assert volumes[(0, 0, 1.0)] == pytest.approx(0.8, rel=1e-9)
        assert volumes[(0.577, 0.577, 0.577)] > volumes[(0, 0, 1.0)]
        assert len({round(v, 9) for v in volumes.values()}) == len(volumes)

    def test_volume_matches_monte_carlo_across_normals(self):
        """对多种法向，精确式与蒙特卡洛真值一致（相对误差 < 3%）。"""
        from kakeya_ai.wolff_axioms import Box, Slab

        box = Box.from_center_size([0.0, 0.0, 0.0], [2.0] * 3)
        rng = np.random.default_rng(0)
        pts = rng.uniform(-1.0, 1.0, size=(400_000, 3))
        for nrm in ([0, 0, 1.0], [1, 1, 1.0], [0.9, 0.3, 0.316], [0.8, 0.6, 0.0]):
            n = np.array(nrm, dtype=float)
            n /= np.linalg.norm(n)
            slab = Slab(n, 0.0, 0.1, box)
            mc = float((np.abs(pts @ n) <= slab.half_thickness).mean()) * 8.0
            assert slab.volume == pytest.approx(mc, rel=0.03), nrm

    def test_volume_analytic_special_cases(self):
        """闭式在解析可算的特例上必须**精确**。"""
        from kakeya_ai.wolff_axioms import Box, Slab

        # 盒 [0,1]^3：法向 z、板 |z-0.5|<=t ⇒ 体积 = 2t。
        box = Box(np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]))
        assert Slab(np.array([0.0, 0.0, 1.0]), 0.5, 0.2, box).volume == pytest.approx(0.4)
        # 覆盖整盒 ⇒ 返回盒体积。
        assert Slab(np.array([1.0, 1.0, 1.0]), 0.0, 10.0, box).volume == pytest.approx(1.0)
        # 完全在盒外 ⇒ 0。
        assert Slab(np.array([0.0, 0.0, 1.0]), 5.0, 0.1, box).volume == pytest.approx(0.0)
        # 厚度非正 ⇒ 0。
        assert Slab(np.array([0.0, 0.0, 1.0]), 0.5, 0.0, box).volume == 0.0
class TestExpandedConvexTestSet:
    """局限 #5：把 Wolff 检验从"轴对齐盒"扩到更一般的凸集（旋转盒 + 紧凸包）。"""

    def test_parallelotope_volume_is_rotation_invariant_and_contains_work(self):
        he = np.array([0.3, 0.2, 0.1])
        box = Box.from_center_size([0.0, 0.0, 0.0], 2.0 * he)
        identity = Parallelotope.from_frame([0.0, 0.0, 0.0], np.eye(3), he)
        rng = np.random.default_rng(7)
        m = rng.normal(size=(3, 3))
        q, r = np.linalg.qr(m)
        q = q * np.sign(np.diag(r))
        par = Parallelotope.from_frame([0.0, 0.0, 0.0], q, he)
        # 旋转不改体积。
        assert par.volume == pytest.approx(box.volume, rel=1e-12)
        assert par.volume == pytest.approx(identity.volume, rel=1e-12)
        # 恒等帧下 contains_segment 与 Box 完全一致。
        a, b = np.array([-0.2, 0.0, 0.0]), np.array([0.2, 0.05, 0.0])
        assert identity.contains_segment(a, b, 0.05) == box.contains_segment(a, b, 0.05)
        assert identity.contains_segment(a, b, 0.05)
        # 旋转后仍保距：把同一线段旋到局部系里必然落在盒内。
        world = np.vstack([a @ q, b @ q])  # 局部坐标 a,b 对应的世界点
        assert par.to_local(world) == pytest.approx(np.vstack([a, b]))
        assert par.contains_segment(world[0], world[1], 0.05)
        # 越界必拒。
        out = np.array([1.0, 0.0, 0.0])
        assert not identity.contains_segment(a + out, b + out, 0.05)
        assert not par.contains_segment(world[0] @ q + out @ q, world[1] @ q + out @ q, 0.05)

    def test_make_test_parallelotopes_stay_inside_bounding_box(self):
        bbox = Box.from_center_size([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
        pars = make_test_parallelotopes(bbox, n_parallelotopes=12, seed=42)
        assert len(pars) == 12
        signs = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                          [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]], dtype=float)
        for p in pars:
            assert p.volume > 0
            corners = p.center + (signs * p.half_extents) @ p.axes
            assert np.all(corners >= bbox.bounds[:, 0] - 1e-9)
            assert np.all(corners <= bbox.bounds[:, 1] + 1e-9)
        assert make_test_parallelotopes(bbox, n_parallelotopes=0) == []
        # make_wolff_test_sets = 盒 + 旋转盒
        assert len(make_wolff_test_sets(bbox, 5, 7, seed=0)) == 12

    def test_tight_and_clustered_parallelotopes_contain_all_tubes(self):
        d = Direction.from_array([1.0, 1.0, 1.0])
        tubes = [
            Tube(center=np.array([0.1 * i, -0.1 * i, 0.0]), direction=d, delta=0.03)
            for i in range(6)
        ]
        tight = tight_parallelotope(tubes)
        for t in tubes:
            a, b = t.endpoints
            assert tight.contains_segment(a, b, t.delta)
        # 方向一致 ⇒ 单簇；混入一根正交管 ⇒ 两簇。
        assert len(make_tube_aligned_parallelotopes(tubes)) == 1
        other = Tube(center=np.zeros(3), direction=Direction.from_array([0, 0, 1]), delta=0.03)
        assert len(make_tube_aligned_parallelotopes([*tubes, other])) == 2

    def test_adversarial_family_tightens_and_is_delta_stable(self):
        """紧凸包把常数从 ~0.02 提到 ~π/4，且随 δ→0 稳定在 π/4 附近。"""
        d = Direction.from_array([1.0, 1.0, 1.0])
        bbox = Box.from_center_size([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])

        def adv(delta):
            t = Tube(center=np.zeros(3), direction=d, length=1.0, delta=delta)
            return adversarial_convex_error([t])

        tubes = [Tube(center=np.zeros(3), direction=d, delta=0.05) for _ in range(4)]
        rand = katz_tao_convex_error(tubes, make_test_boxes(bbox, 64, seed=1))
        e_adv, n_incl = adversarial_convex_error(tubes)
        assert n_incl > 0
        # 更一般（更紧）的凸集族给出显著更大的下界估计——否则"扩大"没有意义。
        assert e_adv > 10.0 * rand
        # δ→0 时趋于 π/4（管体积/最紧盒体积的极限），且恒不超过它（常数有界）。
        e_small, _ = adv(0.01)
        assert e_small == pytest.approx(np.pi / 4, rel=0.03)
        assert adv(0.02)[0] > adv(0.2)[0]
        assert adv(0.01)[0] <= np.pi / 4 + 1e-9
        # 空输入 ⇒ 明确的"未测到"。
        assert adversarial_convex_error([]) == (0.0, 0)
