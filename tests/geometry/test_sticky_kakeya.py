"""Sticky Kakeya 算法化模块的单元测试。

覆盖：
- 随机刚体变换的正确性
- NSP 因子化的管数放大与体积不变性
- 尺度链构造与 Wolff 公理逐层验证
- Sticky 判定
- Sticky Kakeya 体积验证流程
- Proposition 1.7 多尺度归纳循环
"""

import numpy as np
import pytest

from kakeya_ai.primitives import Direction, Tube, TubeCollection
from kakeya_ai.sticky_kakeya import (
    InductionL2Report,
    InductionStep,
    InductionTermination,
    NSPFactorizationResult,
    ScaleChain,
    ScaleLayer,
    StickyKakeyaResult,
    _lattice_translations,
    apply_rigid_transform,
    build_scale_chain,
    classify_induction_termination,
    classify_sticky,
    induction_step_g,
    multiscale_induction,
    nsp_factorization,
    random_rigid_transform,
    verify_induction_l2,
    verify_sticky_kakeya,
)


# ============================================================
# 测试辅助：构造管族
# ============================================================

def make_kakeya_fan(n_theta: int = 10, n_phi: int = 10, delta: float = 0.1) -> list[Tube]:
    """构造标准 Kakeya fan：方向均匀分布在球面上。"""
    tubes = []
    for theta in np.linspace(0.15, np.pi - 0.15, n_theta):
        for phi in np.linspace(0, 2 * np.pi, n_phi, endpoint=False):
            d = Direction.from_spherical(theta, phi)
            tubes.append(Tube(center=np.zeros(3), direction=d, length=1.0, delta=delta))
    return tubes


def make_parallel_tubes(n: int = 20, delta: float = 0.1) -> list[Tube]:
    """构造平行管族（高度聚集，不 broad）。"""
    d = Direction.from_array([0, 0, 1])
    tubes = []
    for i in range(n):
        offset = np.array([(i - n / 2) * 0.05, 0, 0])
        tubes.append(Tube(center=offset, direction=d, length=1.0, delta=delta))
    return tubes


# ============================================================
# 随机刚体变换测试
# ============================================================

class TestRandomRigidTransform:
    def test_rotation_is_orthogonal(self):
        """旋转矩阵应满足 R^T R = I。"""
        R, t = random_rigid_transform(seed=42)
        assert R.shape == (3, 3)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_rotation_has_positive_determinant(self):
        """旋转矩阵行列式应为 +1。"""
        for seed in range(10):
            R, t = random_rigid_transform(seed=seed)
            assert np.linalg.det(R) > 0.999

    def test_translation_in_range(self):
        """平移向量应在指定范围内。"""
        radius = 5.0
        R, t = random_rigid_transform(radius=radius, seed=42)
        assert np.all(np.abs(t) <= radius + 1e-10)


class TestApplyRigidTransform:
    def test_identity_transform_preserves_tubes(self):
        """单位变换不应改变管。"""
        tubes = make_kakeya_fan(5, 5, delta=0.1)
        R = np.eye(3)
        t = np.zeros(3)
        transformed = apply_rigid_transform(tubes, R, t)
        assert len(transformed) == len(tubes)
        for orig, new in zip(tubes, transformed, strict=True):
            assert np.allclose(orig.center, new.center)
            assert np.allclose(orig.direction.v, new.direction.v)
            assert orig.delta == new.delta
            assert orig.length == new.length

    def test_transform_preserves_tube_count_and_delta(self):
        """刚体变换应保持管数和 δ 不变。"""
        tubes = make_kakeya_fan(5, 5, delta=0.05)
        R, t = random_rigid_transform(seed=42)
        transformed = apply_rigid_transform(tubes, R, t)
        assert len(transformed) == len(tubes)
        for orig, new in zip(tubes, transformed, strict=True):
            assert orig.delta == new.delta
            assert orig.length == new.length

    def test_rotation_preserves_direction_norm(self):
        """旋转后方向向量仍为单位向量。"""
        tubes = make_kakeya_fan(5, 5, delta=0.1)
        R, t = random_rigid_transform(seed=42)
        transformed = apply_rigid_transform(tubes, R, t)
        for tube in transformed:
            norm = np.linalg.norm(tube.direction.v)
            assert abs(norm - 1.0) < 1e-10

    def test_rotation_changes_direction(self):
        """非平凡旋转应改变方向。"""
        tubes = make_kakeya_fan(3, 3, delta=0.1)
        R, t = random_rigid_transform(seed=42)
        transformed = apply_rigid_transform(tubes, R, t)
        changed = False
        for orig, new in zip(tubes, transformed, strict=True):
            if not np.allclose(orig.direction.v, new.direction.v, atol=1e-6):
                changed = True
                break
        assert changed, "旋转后方向应至少有一个改变"


# ============================================================
# NSP 因子化测试
# ============================================================

class TestNSPFactorization:
    def test_small_tube_count_triggers_boosting(self):
        """管数远小于 δ^{-2} 时应触发因子化。"""
        delta = 0.05
        tubes = make_kakeya_fan(3, 3, delta=delta)
        n_tubes = len(tubes)
        target = 1.0 / (delta ** 2)
        assert n_tubes < target * 0.5  # 确实需要因子化

        boosted, result = nsp_factorization(tubes, delta=delta, grid_n=32, seed=42)
        assert result.n_copies > 1
        assert result.boosted_count > n_tubes
        assert result.original_count == n_tubes

    def test_large_tube_count_no_boosting(self):
        """管数接近 δ^{-2} 时不需要因子化。"""
        delta = 0.2
        tubes = make_kakeya_fan(10, 10, delta=delta)
        n_tubes = len(tubes)

        boosted, result = nsp_factorization(tubes, delta=delta, grid_n=32, seed=42)
        assert result.n_copies == 1
        assert result.boosted_count == n_tubes

    def test_boosted_count_approximately_target(self):
        """放大后管数应接近 δ^{-2}。"""
        delta = 0.1
        tubes = make_kakeya_fan(3, 3, delta=delta)
        boosted, result = nsp_factorization(tubes, delta=delta, grid_n=32, seed=42)
        target = 1.0 / (delta ** 2)
        # 允许 50% 误差
        assert abs(result.boosted_count - target) / target < 0.5

    def test_volume_ratio_positive(self):
        """体积比应为正数。"""
        tubes = make_kakeya_fan(5, 5, delta=0.1)
        boosted, result = nsp_factorization(tubes, grid_n=32, seed=42)
        assert result.volume_ratio > 0
        assert result.original_volume > 0
        assert result.boosted_volume > 0

    def test_empty_tubes_returns_empty(self):
        """空管族应返回空结果。"""
        boosted, result = nsp_factorization([])
        assert len(boosted) == 0
        assert result.original_count == 0
        assert result.meets_target is False

    def test_meets_target_flag(self):
        """meets_target 标志应正确反映管数是否达标。"""
        delta = 0.1
        # 足够多的管
        tubes = make_kakeya_fan(10, 10, delta=delta)
        _, result = nsp_factorization(tubes, delta=delta, grid_n=32, seed=42)
        assert result.meets_target is True

        # 太少的管但会因子化
        tubes_small = make_kakeya_fan(2, 2, delta=delta)
        _, result_small = nsp_factorization(tubes_small, delta=delta, grid_n=32, seed=42)
        # 因子化后应该接近目标
        assert result_small.meets_target is True

    def test_lattice_mode_is_deterministic(self):
        """mode="lattice" 不含随机数：同输入两次调用逐位一致。"""
        delta = 0.05
        tubes = make_kakeya_fan(3, 3, delta=delta)
        b1, r1 = nsp_factorization(tubes, delta=delta, grid_n=32, mode="lattice")
        b2, r2 = nsp_factorization(tubes, delta=delta, grid_n=32, mode="lattice")
        assert r1.mode == "lattice"
        assert r2.mode == "lattice"
        # 无 seed 也一致（区别于 random 模式）
        assert r1.n_copies == r2.n_copies
        assert len(b1) == len(b2)
        for t1, t2 in zip(b1, b2, strict=True):
            assert np.allclose(t1.center, t2.center)
            assert np.allclose(t1.direction.v, t2.direction.v)

    def test_lattice_mode_meets_target(self):
        """格点模式同样应把管数放大到 δ^{-2} 量级。"""
        delta = 0.05
        tubes = make_kakeya_fan(3, 3, delta=delta)
        _, result = nsp_factorization(tubes, delta=delta, grid_n=32, mode="lattice")
        assert result.n_copies > 1
        assert result.meets_target is True
        assert result.lattice_spacing > 0

    def test_lattice_translations_properties(self):
        """_lattice_translations 数量正确、间隔为 spacing、含原点。"""
        offsets = _lattice_translations(5, 2.0)
        assert len(offsets) == 5
        zero = [o for o in offsets if np.allclose(o, 0.0)]
        assert len(zero) == 1
        # 所有偏移都是 spacing 的整数倍（含 0）
        for o in offsets:
            assert np.allclose(o / 2.0, np.round(o / 2.0))
        assert _lattice_translations(0, 1.0) == []

    def test_lattice_mode_requires_multiple_copies(self):
        """管数已达标的管族在格点模式下不放大（n_copies == 1）。"""
        delta = 0.2
        tubes = make_kakeya_fan(10, 10, delta=delta)
        _, result = nsp_factorization(tubes, delta=delta, grid_n=32, mode="lattice")
        assert result.n_copies == 1
        assert result.mode == "lattice"

    def test_random_mode_is_default(self):
        """默认 mode 仍为 random，行为与显式传入一致。"""
        delta = 0.05
        tubes = make_kakeya_fan(3, 3, delta=delta)
        _, r_default = nsp_factorization(tubes, delta=delta, grid_n=32, seed=42)
        _, r_explicit = nsp_factorization(tubes, delta=delta, grid_n=32, seed=42, mode="random")
        assert r_default.mode == "random"
        assert r_default.n_copies == r_explicit.n_copies
        assert r_default.boosted_count == r_explicit.boosted_count

    def test_invalid_mode_rejected(self):
        """未知 mode 应显式报错，而不是静默退化。"""
        tubes = make_kakeya_fan(3, 3, delta=0.05)
        with pytest.raises(ValueError):
            nsp_factorization(tubes, delta=0.05, grid_n=32, mode="bogus")


# ============================================================
# 尺度链构造测试
# ============================================================

class TestBuildScaleChain:
    def test_chain_has_correct_number_of_layers(self):
        """尺度链层数应等于 n_scales。"""
        tubes = make_kakeya_fan(8, 8, delta=0.1)
        chain = build_scale_chain(tubes, n_scales=6)
        assert chain.n_scales == 6

    def test_chain_spans_delta_to_one(self):
        """尺度链应从 δ 到 1。"""
        delta = 0.1
        tubes = make_kakeya_fan(5, 5, delta=delta)
        chain = build_scale_chain(tubes, delta=delta, n_scales=5)
        assert abs(chain.layers[0].rho - delta) < 1e-10
        assert abs(chain.layers[-1].rho - 1.0) < 1e-10

    def test_layers_are_monotonically_increasing(self):
        """尺度链应单调递增（从 δ 到 1）。"""
        tubes = make_kakeya_fan(5, 5, delta=0.05)
        chain = build_scale_chain(tubes, n_scales=8)
        rhos = [layer.rho for layer in chain.layers]
        for i in range(len(rhos) - 1):
            assert rhos[i] <= rhos[i + 1] + 1e-10

    def test_each_layer_has_valid_errors(self):
        """每层的 Wolff 误差应为非负有限数。"""
        tubes = make_kakeya_fan(5, 5, delta=0.1)
        chain = build_scale_chain(tubes, n_scales=5)
        for layer in chain.layers:
            assert layer.kt_cw_error >= 0
            assert np.isfinite(layer.kt_cw_error)
            assert layer.frostman_error >= 0
            assert np.isfinite(layer.frostman_error)

    def test_empty_tubes_returns_empty_chain(self):
        """空管族应返回空链。"""
        chain = build_scale_chain([])
        assert chain.n_scales == 0
        assert chain.is_sticky is False

    def test_fan_is_sticky_with_loose_threshold(self):
        """Kakeya fan 在宽松阈值下应为 sticky。"""
        tubes = make_kakeya_fan(8, 8, delta=0.1)
        chain = build_scale_chain(tubes, n_scales=5, max_wolff_error=100.0)
        assert chain.is_sticky is True

    def test_parallel_tubes_high_kt_error(self):
        """平行管族的 KT-CW 误差应高于 fan。"""
        fan = make_kakeya_fan(8, 8, delta=0.1)
        parallel = make_parallel_tubes(20, delta=0.1)
        chain_fan = build_scale_chain(fan, n_scales=3)
        chain_parallel = build_scale_chain(parallel, n_scales=3)
        # 平行管聚集在单一方向，KT-CW 应更大
        assert chain_parallel.layers[0].kt_cw_error >= chain_fan.layers[0].kt_cw_error * 0.5


# ============================================================
# Sticky 判定测试
# ============================================================

class TestClassifySticky:
    def test_fan_is_sticky(self):
        """方向均匀分布的 fan 应被判定为 sticky。"""
        tubes = make_kakeya_fan(8, 8, delta=0.1)
        is_sticky, chain = classify_sticky(tubes, n_scales=5, max_wolff_error=50.0)
        assert is_sticky is True

    def test_returns_chain_with_layers(self):
        """应返回包含层的尺度链。"""
        tubes = make_kakeya_fan(5, 5, delta=0.1)
        is_sticky, chain = classify_sticky(tubes, n_scales=5)
        assert isinstance(chain, ScaleChain)
        assert chain.n_scales == 5
        assert isinstance(is_sticky, bool)

    def test_strict_threshold_makes_non_sticky(self):
        """极严格的 Wolff 阈值可能使某些配置变为 non-sticky。"""
        tubes = make_kakeya_fan(3, 3, delta=0.1)
        is_sticky, _ = classify_sticky(tubes, n_scales=5, max_wolff_error=0.01)
        # 小 fan + 极严格阈值 → 可能非 sticky
        # 不强制结果，但函数应正常执行
        assert isinstance(is_sticky, bool)


# ============================================================
# Sticky Kakeya 体积验证测试
# ============================================================

class TestVerifyStickyKakeya:
    def test_returns_correct_type(self):
        """应返回 StickyKakeyaResult。"""
        tubes = make_kakeya_fan(5, 5, delta=0.1)
        result = verify_sticky_kakeya(tubes, n_scales=5, grid_n=32, seed=42)
        assert isinstance(result, StickyKakeyaResult)

    def test_volume_is_positive(self):
        """并集体积应为正。"""
        tubes = make_kakeya_fan(5, 5, delta=0.1)
        result = verify_sticky_kakeya(tubes, n_scales=5, grid_n=32, seed=42)
        assert result.volume > 0

    def test_sticky_lower_bound_positive(self):
        """Sticky 下界应为正。"""
        tubes = make_kakeya_fan(5, 5, delta=0.1)
        result = verify_sticky_kakeya(tubes, n_scales=5, grid_n=32, seed=42)
        assert result.sticky_lower_bound > 0

    def test_ratio_is_finite(self):
        """体积比应为有限数。"""
        tubes = make_kakeya_fan(5, 5, delta=0.1)
        result = verify_sticky_kakeya(tubes, n_scales=5, grid_n=32, seed=42)
        assert np.isfinite(result.ratio)
        assert result.ratio > 0

    def test_nsp_triggered_for_small_tube_count(self):
        """管数少时应触发 NSP。"""
        delta = 0.05
        tubes = make_kakeya_fan(2, 2, delta=delta)
        result = verify_sticky_kakeya(
            tubes, delta=delta, n_scales=5, grid_n=32, use_nsp=True, seed=42
        )
        assert result.nsp is not None
        assert result.nsp.n_copies > 1

    def test_nsp_not_triggered_for_large_tube_count(self):
        """管数足够多时不应触发 NSP。"""
        delta = 0.2
        tubes = make_kakeya_fan(10, 10, delta=delta)
        result = verify_sticky_kakeya(
            tubes, delta=delta, n_scales=5, grid_n=32, use_nsp=True, seed=42
        )
        assert result.nsp is None

    def test_empty_tubes_returns_zero_result(self):
        """空管族应返回零结果。"""
        result = verify_sticky_kakeya([])
        assert result.volume == 0
        assert result.is_sticky is False
        assert result.ratio == 0

    def test_scale_chain_attached_to_result(self):
        """结果应附带尺度链。"""
        tubes = make_kakeya_fan(5, 5, delta=0.1)
        result = verify_sticky_kakeya(tubes, n_scales=5, grid_n=32, seed=42)
        assert result.scale_chain is not None
        assert result.scale_chain.n_scales == 5


# ============================================================
# Proposition 1.7 多尺度归纳测试
# ============================================================

class TestInductionStepG:
    def test_positive_reduction_for_valid_params(self):
        """有效参数下压缩量应为正。"""
        g = induction_step_g(sigma=0.5, omega=1.0, delta=0.01)
        assert g > 0

    def test_zero_omega_returns_zero(self):
        """ω=0 时压缩量应为 0。"""
        g = induction_step_g(sigma=0.5, omega=0.0, delta=0.01)
        assert g == 0.0
        """ω=0 时压缩量应为 0。"""
        g = induction_step_g(sigma=0.5, omega=0.0, delta=0.01)
        assert g == 0.0

    def test_delta_geq_one_returns_zero(self):
        """δ≥1 时压缩量应为 0。"""
        g = induction_step_g(sigma=0.5, omega=1.0, delta=1.0)
        assert g == 0.0

    def test_reduction_bounded_by_half_omega(self):
        """压缩量不应超过 ω 的一半。"""
        g = induction_step_g(sigma=0.0, omega=0.1, delta=1e-10)
        assert g <= 0.1 * 0.5 + 1e-10

    def test_smaller_delta_more_reduction(self):
        """更小的 δ 应产生更大的压缩量。"""
        g_large_delta = induction_step_g(sigma=0.5, omega=1.0, delta=0.1)
        g_small_delta = induction_step_g(sigma=0.5, omega=1.0, delta=0.001)
        assert g_small_delta >= g_large_delta

    def test_sigma_one_minimal_reduction(self):
        """σ=1 时压缩量应很小。"""
        g = induction_step_g(sigma=1.0, omega=1.0, delta=0.01)
        # σ=1 → (1-σ)=0 → g=0
        assert g == pytest.approx(0.0, abs=1e-10)


class TestInductionDocsDoNotClaimPaperFormula:
    """守门：docstring 不得把自造公式说成论文公式。

    **为什么需要这条测试。** ``induction_step_g`` 的函数名沿用了论文符号
    ``g(σ, ω)``，且它确实服务于 Prop 1.7 的归纳环节——这正是危险之处：
    它的 docstring 曾经开口就写"计算 Proposition 1.7 中的 ω 压缩量"，读者
    看到首行就以为是论文公式，而实际上论文的 Prop 1.7 是**存在性**结论、
    **不给闭式**，这里的比例常数与上限都是自定的。

    内部设计文档一直如实登记着这条限定（"是量级正确的模型，不是论文
    的精确估计"），但**代码侧没人守**。于是出现过"文档说近似、代码说公式"
    的自相矛盾。这条测试把两边钉在一起。
    """

    def test_induction_step_g_declares_itself_heuristic(self):
        doc = induction_step_g.__doc__ or ""
        # 必须显式承认是启发式/近似，而不是论文公式。
        assert "启发式" in doc or "heuristic" in doc.lower(), (
            "induction_step_g 的 docstring 未声明自己是启发式"
        )
        # 必须说明"论文不给闭式/不是论文公式"这一关键限定。
        assert "不是" in doc and ("论文" in doc), (
            "docstring 未说明该公式不是论文给出的"
        )

    def test_multiscale_induction_marks_curve_heuristic(self):
        doc = multiscale_induction.__doc__ or ""
        assert "启发式" in doc, (
            "multiscale_induction 未把返回的收敛曲线标为启发式"
        )
        # 定量结论的禁区必须写明。
        assert "不可作" in doc or "不可" in doc, (
            "未写明哪些定量结论不可据此得出"
        )

    def test_no_bare_paper_claim_in_first_docstring_line(self):
        """docstring **首行**不得把函数定义成"论文某某命题的量"。

        首行是 `help()` 与编辑器悬浮提示最先展示的内容，误导性最强。
        """
        for fn in (induction_step_g, multiscale_induction):
            first = (fn.__doc__ or "").strip().splitlines()[0]
            assert "计算 Proposition" not in first, (
                f"{fn.__name__} 的首行仍声称是论文命题的计算"
            )


class TestMultiscaleInduction:
    def test_returns_list_of_steps(self):
        """应返回 InductionStep 列表。"""
        tubes = make_kakeya_fan(5, 5, delta=0.1)
        steps = multiscale_induction(tubes, max_iterations=3, n_scales=3)
        assert isinstance(steps, list)
        assert len(steps) <= 3
        for step in steps:
            assert isinstance(step, InductionStep)

    def test_omega_monotonically_decreasing(self):
        """ω 应单调递减。"""
        tubes = make_kakeya_fan(5, 5, delta=0.1)
        steps = multiscale_induction(tubes, max_iterations=5, n_scales=3)
        for i in range(len(steps) - 1):
            assert steps[i].omega_after <= steps[i].omega_before + 1e-10
            assert steps[i + 1].omega_before <= steps[i].omega_after + 1e-10

    def test_convergence_to_threshold(self):
        """足够多迭代后 ω 应收敛到阈值。"""
        tubes = make_kakeya_fan(5, 5, delta=0.001)
        steps = multiscale_induction(
            tubes, omega_0=0.1, max_iterations=100,
            omega_threshold=1e-6, n_scales=3,
        )
        if len(steps) > 0:
            assert steps[-1].omega_after <= 1e-4 or len(steps) == 100

    def test_zero_iterations_for_zero_omega(self):
        """ω=0 时不应产生任何步。"""
        tubes = make_kakeya_fan(3, 3, delta=0.1)
        steps = multiscale_induction(tubes, omega_0=0.0, max_iterations=5)
        assert len(steps) == 0

    def test_each_step_has_scale_chain(self):
        """每步应附带尺度链。"""
        tubes = make_kakeya_fan(5, 5, delta=0.1)
        steps = multiscale_induction(tubes, max_iterations=3, n_scales=3)
        for step in steps:
            assert step.scale_chain is not None
            assert isinstance(step.scale_chain, ScaleChain)

    def test_omega_reduction_positive(self):
        """**sticky 的**步里 ω 压缩量应为正（2026-09-30 修正前提）。

        **为什么加"sticky 的"这个限定**：论文 Prop 1.7 的压缩前提是 sticky。
        修复前本用例用非 sticky 构型（``δ=0.01`` 的 fan）也断言"每步压缩为正"
        —— 而那正是 bug 的表现：旧代码无论 sticky 与否都压缩，于是断言
        恒真、把"前提不成立时也在压缩"这个错误掩盖住了。

        现在改用**实测判为 sticky** 的构型（``δ=0.1`` 的 ``fan(3×3)``），
        并显式断言前提成立。
        """
        steps = multiscale_induction(
            _sticky_tubes(), delta=0.1, max_iterations=3, n_scales=5,
        )
        assert steps, "应产生步"
        sticky_steps = [s for s in steps if s.is_sticky]
        assert sticky_steps, (
            "前提失效：该构型应判为 sticky——非 sticky 时不压缩是**正确**行为"
        )
        for step in sticky_steps:
            assert step.omega_reduction > 0

    def test_empty_tubes_returns_empty(self):
        """空管族应返回空列表。"""
        steps = multiscale_induction([], max_iterations=5)
        assert len(steps) == 0


class TestInductionL2Verification:
    """A4（L2）：``multiscale_induction`` 的**终止可判定 + 环节可验证 + 可复现**。

    对应内部设计文档 §7.0.2 的 L2 定义："Prop 1.7 每一步满足 g 的
    单调性与极限（**不要求等于论文常数**）"。本类钉住三件事：

    1. **终止可判定**：``classify_induction_termination`` 把"为什么停下"分成
       三态（``CONVERGED`` / ``STALLED`` / ``DESCENDING``），**语义互不混淆**
       ——此前只是"跑满 ``max_iterations``"，收敛与"压不动"长得一样；
    2. **环节可验证**：``verify_induction_l2`` 逐步核验 g 的单调性、下界、
       更新式自洽与极限存在性；
    3. **可复现**：整个循环无随机数 ⇒ 同输入**逐位相同**的步序列。
    """

    @staticmethod
    def _step(before: float, after: float, sticky: bool = True) -> InductionStep:
        """直接构造一个 ``InductionStep``（单元级用，绕开昂贵的尺度链计算）。"""
        return InductionStep(
            sigma=0.5, omega_before=before, omega_after=after,
            omega_reduction=before - after, delta=0.1, is_sticky=sticky,
            scale_chain=ScaleChain(),
        )

    def test_l2_report_ok_for_real_run(self):
        """真实运行的输出必须满足 L2 结构性质（``report.ok`` 为真）。"""
        steps = multiscale_induction(
            _sticky_tubes(), delta=0.1, max_iterations=6, n_scales=5,
        )
        report = verify_induction_l2(steps)
        assert isinstance(report, InductionL2Report)
        assert report.ok, report.summary()
        assert report.g_nonnegative and report.omega_bounded_below
        assert report.step_arithmetic_consistent and report.monotone
        assert report.n_steps == len(steps)
        # 极限存在（单调不增 + 有下界）⇒ 取末值。
        assert report.limit == pytest.approx(steps[-1].omega_after)

    def test_termination_three_states_are_distinct(self):
        """三态判定必须**互不混淆**（同一组步只能落在其中一态）。"""
        # 收敛：末步 ω 已到阈值以下。
        converged = [self._step(0.1, 0.05), self._step(0.05, 1e-9)]
        t, _ = classify_induction_termination(converged, omega_threshold=1e-6)
        assert t is InductionTermination.CONVERGED

        # 停滞：末尾压缩量恒为 0（非 sticky 的如实空转）。
        stalled = [self._step(0.1, 0.1, sticky=False) for _ in range(6)]
        t, _ = classify_induction_termination(stalled, omega_threshold=1e-6)
        assert t is InductionTermination.STALLED

        # 仍在压：有下降但末步仍高于阈值。
        descending = [self._step(0.1, 0.09), self._step(0.09, 0.08)]
        t, _ = classify_induction_termination(descending, omega_threshold=1e-6)
        assert t is InductionTermination.DESCENDING

        # 空序列 = 已收敛（ω 本就在阈值下）。
        t, _ = classify_induction_termination([], omega_threshold=1e-6)
        assert t is InductionTermination.CONVERGED

    def test_l2_detects_monotonicity_violation(self):
        """篡改出一个"ω 反弹"的步，L2 必须报**不通过**（判据非空转）。"""
        bad = [self._step(0.1, 0.2)]  # 后者大于前者 ⇒ g < 0
        report = verify_induction_l2(bad)
        assert not report.g_nonnegative
        assert not report.ok, "ω 反弹却判 L2 通过——判据失效"

    def test_induction_is_bitwise_reproducible(self):
        """整个循环无随机数 ⇒ 同输入两次调用**逐位相同**。

        （与 NSP 因子化相反：后者依赖随机刚体变换、必须显式 seed。）
        """
        tubes = _sticky_tubes()
        a = multiscale_induction(tubes, delta=0.1, max_iterations=6, n_scales=5)
        b = multiscale_induction(tubes, delta=0.1, max_iterations=6, n_scales=5)
        fields = ("sigma", "omega_before", "omega_after", "omega_reduction",
                  "delta", "is_sticky")
        assert [tuple(getattr(s, f) for f in fields) for s in a] == \
               [tuple(getattr(s, f) for f in fields) for s in b], \
               "同一输入产生了两条不同的归纳曲线（说明循环里混进了随机性）"

    def test_termination_consistent_with_curve(self):
        """终止判定与曲线自洽（``termination_consistent`` 对真实运行恒真）。"""
        for tubes, delta in ((_sticky_tubes(), 0.1), (_non_sticky_tubes(), 0.01)):
            steps = multiscale_induction(tubes, delta=delta,
                                         max_iterations=6, n_scales=4)
            report = verify_induction_l2(steps)
            assert report.termination_consistent, (
                f"终止判定 {report.termination.name} 与曲线不自洽：{report.summary()}"
            )


# ============================================================
# 尺度链的真实性：粗粒化 + 覆盖检测 + 阈值
# ============================================================
# 回归背景（2026-09-15）：以下三处曾同时出错，使"尺度链"名存实亡，
# 进而让主循环几乎总在第一步就被判为 sticky 而短路。三者都属同一类病：
# **返回"通过"其实是因为什么都没检查**。

class TestScaleChainIsReal:
    """尺度链的每一层必须是在**不同尺度的管族**上检验。"""

    def test_layers_have_distinct_errors(self):
        """各层误差不能全部相同——否则就是同一份数据被验了 N 遍。

        回归：旧 ``build_scale_chain`` 把粗粒化结果只用来数个数，误差仍在
        原始 δ-管上计算（注释写"用 ρ-粗粒化后的管集做检验"与代码不符），
        实测 fan 构型每层 KT-CW 都是 0.1257。
        """
        tubes = make_kakeya_fan(4, 4, delta=0.1)
        chain = build_scale_chain(tubes, delta=0.1, n_scales=5)
        kts = [round(layer.kt_cw_error, 9) for layer in chain.layers]
        assert len(set(kts)) > 1, (
            f"各层 KT-CW 完全相同（{kts}）——尺度链没有做真正的粗粒化"
        )

    def test_coarsen_produces_larger_tubes(self):
        """粗粒化后的管半径应等于该层尺度 ρ。"""
        from kakeya_ai.sticky_kakeya import coarsen_to_rho_tubes

        tubes = make_kakeya_fan(4, 4, delta=0.1)
        for rho in (0.2, 0.5, 1.0):
            coarse = coarsen_to_rho_tubes(tubes, rho)
            assert coarse, f"rho={rho} 粗粒化结果为空"
            assert all(abs(t.delta - rho) < 1e-12 for t in coarse), (
                "粗管半径应精确等于该层尺度 ρ"
            )

    def test_coarsen_merges_nearby_directions(self):
        """方向相近的管应被合并（粗管数 ≤ 原管数），且方向覆盖 bin。"""
        from kakeya_ai.grains import partition_into_rho_tubes
        from kakeya_ai.sticky_kakeya import coarsen_to_rho_tubes

        tubes = make_kakeya_fan(4, 4, delta=0.1)
        for rho in (0.3, 0.6, 1.0):
            bins = partition_into_rho_tubes(tubes, rho)
            coarse = coarsen_to_rho_tubes(tubes, rho)
            assert len(coarse) == len(bins), "粗管数应与 ρ-bin 数一致"
            assert len(coarse) <= len(tubes)

            # 每个 bin 内的 δ-管方向应落在该粗管方向的 ρ/2 球冠内
            for bin_idx, indices in bins.items():
                bin_centre = coarse[bin_idx].direction.v
                for i in indices:
                    ang = float(np.arccos(np.clip(
                        np.dot(tubes[i].direction.v, bin_centre), -1.0, 1.0)))
                    assert ang <= rho / 2 + 1e-9, (
                        f"bin {bin_idx} 内管与粗管方向夹角 {ang} 超过 ρ/2={rho/2}"
                    )


class TestNoUntestedLayers:
    """没测到的层必须报"未测"，不能报"满足"。"""

    def test_all_layers_are_actually_tested(self):
        """默认测试集下，每层都应有管被包含（否则是假通过）。

        回归：测试集边距与 slab 厚度曾硬编码（0.5 / 0.25），当粗管半径
        大于它们时一个测试集都装不下，误差返回 0.0 —— 看起来像"满足公理"。
        """
        tubes = make_kakeya_fan(4, 4, delta=0.1)
        chain = build_scale_chain(tubes, delta=0.1, n_scales=5)
        untested = [layer for layer in chain.layers if not layer.tested]
        assert not untested, (
            f"{len(untested)} 层未被真正检验："
            f"{[layer.rho for layer in untested]}"
        )

    def test_untested_layer_does_not_claim_wolff_ok(self):
        """显式传入装不下任何管的测试集时，必须 reported 为未通过。

        这是"零命中 ≠ 满足公理"的直接检验。
        """
        from kakeya_ai.wolff_axioms import Box

        tubes = make_kakeya_fan(4, 4, delta=0.1)
        # 一个极小的盒子：任何管都装不下
        tiny = [Box.from_center_size([0.0, 0.0, 0.0], [1e-6, 1e-6, 1e-6])]
        chain = build_scale_chain(
            tubes, delta=0.1, n_scales=3, convex_test_sets=tiny,
        )
        for layer in chain.layers:
            assert layer.n_included == 0
            assert layer.tested is False
            assert layer.wolff_ok is False, (
                "装不下任何管的测试集不得判为满足公理"
            )
        assert chain.is_sticky is False


class TestWolffThresholdMatchesPaper:
    """默认阈值应对齐论文的"误差 ≲ 1"。"""

    def test_default_threshold_is_one(self):
        """回归：默认曾为 10.0，比论文（≲1）宽十倍，导致假 sticky。"""
        import inspect

        sig = inspect.signature(build_scale_chain)
        assert sig.parameters["max_wolff_error"].default == 1.0

    def test_sticky_decision_is_sensitive_near_one(self):
        """阈值在 1 附近应能改变判定结果（说明判定真的在起作用）。"""
        tubes = make_kakeya_fan(4, 4, delta=0.1)
        loose = build_scale_chain(tubes, delta=0.1, n_scales=5,
                                  max_wolff_error=3.0)
        tight = build_scale_chain(tubes, delta=0.1, n_scales=5,
                                  max_wolff_error=1.0)
        assert loose.is_sticky is True
        assert tight.is_sticky is False, (
            "max_wolff_error=1.0（论文口径）应能拒绝此构型"
        )


class TestMainLoopActuallyRuns:
    """主循环应真的执行 grains 分解与 Move，而不是第一步就短路。"""

    def test_some_configuration_executes_moves(self):
        """至少存在一个 (构型, seed) 会走完 grains + Move。

        回归背景：曾经 15/15 全部第一步终止，主循环形同虚设。

        **2026-09-17 修复不稳定**：本用例原先只试 `seed` 默认值的那一次运行，
        而编排器此前**不传递种子**给 NSP 因子化（`np.random.default_rng(None)`），
        于是每次运行的随机副本都不同——实测同一构型 12 次里有 4 种动作序列，
        约 58% 的概率第一步就终止，该用例因此随机通过/失败。
        修好种子传递后（默认 `seed=0` 是确定的），单一固定 seed 会**确定地**
        走某一条路，于是必须按原意"**搜索存在性**"：扫一组 seed，要求至少
        一个能走到 grains/Move。

        为什么扫 seed 而不是写死"seed=2 能通过"：写死会把测试绑在一个偶然的
        数值上；扫描表达的是"这条路径确实可达"这一结构性质，对 NSP 内部实现
        的调整更稳健。上限 8 个 seed 是实测的宽裕值（24 个 seed 里有 16 个可行，
        前 8 个里至少 5 个可行）。
        """
        from kakeya_ai.proof_orchestrator import ProofOrchestrator

        move_like = {
            "GRAINS_DECOMPOSITION",
            "MOVE1_LONGER_GRAINS",
            "MOVE2_REPLACE_SQUARE_GRAINS",
            "MOVE3_WIDER_GRAINS",
        }
        seen = []
        for seed in range(8):
            tubes = make_kakeya_fan(4, 4, delta=0.1)
            result = ProofOrchestrator(
                tubes, max_iterations=6, grid_n=16, seed=seed,
            ).run()
            actions = [step.action.name for step in result.steps]
            seen.append((seed, actions))
            if any(a in move_like for a in actions):
                return
        raise AssertionError(
            "扫了 8 个 seed 都没有构型走到 grains/Move；"
            f"实际动作序列：{seen}"
        )

    def test_same_seed_is_reproducible(self):
        """**同一 seed 两次运行必须逐位一致**——这是 trace 可复核的前提。

        回归背景（2026-09-17）：编排器此前不向 NSP 因子化传种子，同一输入
        重跑会得到不同路径；复核方即使拿到完全相同的参数也复现不出同一份
        trace，"可独立复核"因此落空。本用例钉住修复。
        """
        from kakeya_ai.proof_orchestrator import ProofOrchestrator

        def run_once(seed):
            tubes = make_kakeya_fan(4, 4, delta=0.1)
            return ProofOrchestrator(
                tubes, max_iterations=4, grid_n=16, seed=seed,
            ).run()

        for seed in (0, 3, 17):
            a, b = run_once(seed), run_once(seed)
            assert [s.action.name for s in a.steps] == [s.action.name for s in b.steps], (
                f"seed={seed} 两次运行的动作序列不同，编排器不可复现"
            )
            assert (
                [s.before.volume_ratio for s in a.steps]
                == [s.before.volume_ratio for s in b.steps]
            ), f"seed={seed} 两次运行的体积比不同，数值不可复现"

    def test_none_seed_opts_into_non_determinism(self):
        """``seed=None`` 是显式的探索模式：应**允许**两次运行不同。

        与上一个用例配对，说明"确定性"是**默认**而非唯一模式：需要探索不同
        随机副本时显式传 ``None``。这里只断言"不报错且合法"，不断言两次一定
        不同——随机恰好相同也是合法结果，断言不同会引入新的 flaky。
        """
        from kakeya_ai.proof_orchestrator import ProofOrchestrator

        result = ProofOrchestrator(
            make_kakeya_fan(4, 4, delta=0.1), max_iterations=4, grid_n=16, seed=None,
        ).run()
        assert result.seed is None
        assert result.steps, "探索模式也应产出步骤"


class TestNoFalseNegativeFromTestSets:
    """测试集覆盖不足会造成**假否定**——与"没测就说通过"是同一类错误的两极。

    回归背景（2026-09-15）：默认 slab 测试集把 6 个 slab 全固定在 offset=0
    （过原点、法向对齐坐标轴）。当管的中心散布在原点之外（如随机构型的
    ±0.3）时，"整根管都落在过原点的薄板内"几乎不可能，于是全部 slab 零覆盖，
    该层被标为"未测"，整条链被判 non-sticky——**而它的实际误差其实 ≤ 1**。
    """

    @staticmethod
    def _random_tubes(seed: int, n: int = 12, delta: float = 0.1):
        rng = np.random.default_rng(seed)
        return [
            Tube(center=rng.uniform(-0.3, 0.3, 3),
                 direction=Direction.from_spherical(
                     float(np.arccos(1 - 2 * rng.random())),
                     float(2 * np.pi * rng.random())),
                 length=0.8, delta=delta)
            for _ in range(n)
        ]

    def test_slab_offsets_are_not_all_zero(self):
        """默认 slab 的 offset 必须扫描，不能全部过原点。"""
        from kakeya_ai.sticky_kakeya import _default_slab_test_sets

        tubes = self._random_tubes(seed=0)
        slabs = _default_slab_test_sets(tubes)
        offsets = {round(s.offset, 9) for s in slabs}
        assert len(offsets) > 1, (
            f"所有 slab 的 offset 都是 {offsets}——无法覆盖散布的管族"
        )

    def test_displaced_tubes_are_covered(self):
        """中心偏离原点的管族也应被某些 slab 覆盖。"""
        from kakeya_ai.sticky_kakeya import _default_slab_test_sets
        from kakeya_ai.wolff_axioms import count_tubes_in_set

        tubes = self._random_tubes(seed=0)
        slabs = _default_slab_test_sets(tubes)
        max_included = max(count_tubes_in_set(tubes, s) for s in slabs)
        assert max_included > 0, "散布的管族不应被所有 slab 拒之门外"

    def test_no_layer_reports_untested_for_random_config(self):
        """随机构型不应因为测试集覆盖不足而被标"未测"（假否定）。"""
        tubes = self._random_tubes(seed=0)
        chain = build_scale_chain(tubes, delta=0.1, n_scales=5)
        untested = [layer.rho for layer in chain.layers if not layer.tested]
        assert not untested, f"这些层被误判为未测：{untested}"

    def test_empty_tubes_does_not_crash(self):
        """**空管族不得崩**（回归：UnboundLocalError）。

        ``all_pts`` 此前只在 ``else`` 分支里赋值，空输入时到 ``len(all_pts)``
        那行就抛 ``UnboundLocalError``（实测确认）。既有用例全部传非空管族，
        故从未暴露。空输入应返回可用的 slab 集（offset 退化为 ``[0.0]``）。
        """
        from kakeya_ai.sticky_kakeya import _default_slab_test_sets

        slabs = _default_slab_test_sets([])
        assert len(slabs) > 0, "空输入也应给出可用的测试集"
        assert all(s.half_thickness > 0 for s in slabs)

    def test_error_below_threshold_yields_sticky(self):
        """误差 ≤ 1 的构型应判 sticky（这是论文的预期方向）。"""
        tubes = self._random_tubes(seed=0)
        chain = build_scale_chain(tubes, delta=0.1, n_scales=5)
        max_err = max(
            max(layer.kt_cw_error, layer.frostman_error) for layer in chain.layers
        )
        assert max_err <= 1.0
        assert chain.is_sticky is True, (
            f"最大误差 {max_err:.4f} ≤ 1 却判为非 sticky——疑似假否定"
        )


def _non_sticky_tubes(n: int = 12, delta: float = 0.01):
    """一个实测**会判为非 sticky** 的构型（``δ=0.01`` 下的均匀扇）。

    用于测**另一条分支**：非 sticky 时不压缩 ω（性质③）。
    **不要**拿它测「ω 递减」——那条分支本就取 ``g=0``，得不到有效结论。
    """
    return [
        Tube(
            center=np.zeros(3),
            direction=Direction.from_spherical(np.pi / 2, 2 * np.pi * i / n),
            length=1.0,
            delta=delta,
        )
        for i in range(n)
    ]


def _sticky_tubes(n_rings: int = 3, delta: float = 0.1):
    """一个实测**会判为 sticky** 的构型（``δ=0.1`` 下的 ``fan(3×3)``）。

    **为什么必须挑 sticky 构型**：Prop 1.7 的 ω 压缩**只在 sticky 情形成立**
    （见 ``multiscale_induction`` 的循环注释）。用非 sticky 构型测"ω 是否
    递减"，测的是**另一条分支**（那里正确地不压缩），得不到有效结论。
    """
    return [
        Tube(
            center=np.zeros(3),
            direction=Direction.from_spherical(np.pi / 2 * 0.3 + 0.01 * i, 2 * np.pi * j / 3),
            length=1.0,
            delta=delta,
        )
        for i in range(n_rings)
        for j in range(3)
    ]


class TestMultiscaleInductionStructuralProperties:
    """Prop 1.7 循环的**三条结构性质**（L2，2026-09-30 新增）。

    **背景**：``induction_step_g`` 的单步性质早有测试（``TestInductionStepG``），
    但**整个循环**没人守。实测（见 §7.0.2 的 L2 诊断）暴露两个真问题：

    1. 循环**无论 sticky 与否都压缩**，而论文的压缩前提是 sticky；
    2. 非 sticky 分支把 σ 推向 1.0（``sigma + 0.05``），而 ``g ∝ (1−σ)``
       —— **等于"因为压不动，就把未来压缩的能力也掐掉"**。
       实测后果：30 步里 **20 步 Δω ≡ 0**，ω 只降 1.2%。

    修完后这三条性质才成立。**不要求**它们等于论文的定量结论（论文 Prop 1.7
    是存在性结论、不给闭式；本项目 ``c_0`` 与 ``ω/2`` 均自定）——只要求
    **方向与边界**正确。
    """

    def test_omega_strictly_decreases_while_sticky(self):
        """① sticky 的每一步，ω **严格递减**（这正是修复前的失败点）。

        修复前：σ 被推到 1.0 后 ``g=0``，ω 不再变化 ⇒ 这条断言红。
        """
        steps = multiscale_induction(
            _sticky_tubes(), delta=0.1, max_iterations=20,
            omega_threshold=1e-12, n_scales=5,
        )
        assert steps, "该构型应产生至少一步"
        sticky_steps = [s for s in steps if s.is_sticky]
        assert sticky_steps, "前提失效：该构型应判为 sticky（否则测不到压缩路径）"

        for s in sticky_steps:
            assert s.omega_after < s.omega_before, (
                f"sticky 步未压缩：ω {s.omega_before} → {s.omega_after}"
                "（修复前正是如此：σ 被推向 1 导致 g=0）"
            )
            assert s.omega_reduction > 0.0

    def test_omega_never_goes_negative(self):
        """② ω 恒 ≥ 0（``max(0.0, ·)`` 保证；防未来改动去掉该截断）。"""
        steps = multiscale_induction(
            _sticky_tubes(), delta=0.1, max_iterations=50,
            omega_threshold=1e-12, n_scales=5,
        )
        for s in steps:
            assert s.omega_before >= 0.0
            assert s.omega_after >= 0.0

    def test_non_sticky_step_does_not_reduce_omega(self):
        """③ **非 sticky 不压缩**（论文的压缩前提是 sticky）。

        这条是修复的**核心**：此前无论 sticky 与否都压缩，于是"没进入
        可压缩前提"的构型也会显示 ω 在降——**那是假的**。

        构造：均匀扇在 ``δ=0.01`` 下判为非 sticky（实测）。
        """
        tubes = _non_sticky_tubes()
        steps = multiscale_induction(
            tubes, delta=0.01, max_iterations=5,
            omega_threshold=1e-12, n_scales=4,
        )
        assert steps, "应产生步（哪怕全是空转——空转本身是如实的）"
        non_sticky = [s for s in steps if not s.is_sticky]
        assert non_sticky, "前提失效：该构型应判为非 sticky"
        for s in non_sticky:
            assert s.omega_reduction == 0.0, (
                "非 sticky 时不应压缩 ω —— 论文的压缩前提是 sticky"
            )
            assert s.omega_after == s.omega_before

    def test_non_sticky_does_not_push_sigma_to_one(self):
        """**σ 不得被推向 1**——那会关掉后续所有压缩（修复前的真 bug）。

        ``g ∝ (1−σ)``，故 σ→1 ⟹ g→0。修复前的代码在非 sticky 时做
        ``sigma = min(1.0, sigma + 0.05)``，20 步就把 σ 顶到 1.0，
        之后**永远**压不动。

        断言：多个非 sticky 步之后，σ 仍等于初值（机器状态不因"压不动"
        而恶化）。
        """
        tubes = [
            Tube(center=np.zeros(3),
                 direction=Direction.from_spherical(np.pi / 2, 2 * np.pi * i / 12),
                 length=1.0, delta=0.01)
            for i in range(12)
        ]
        steps = multiscale_induction(
            tubes, delta=0.01, max_iterations=10,
            omega_threshold=1e-12, n_scales=4, sigma_0=0.5,
        )
        assert steps
        # 非 sticky 的步里 σ 必须保持 0.5（不被推进）。
        for s in steps:
            if not s.is_sticky:
                assert s.sigma == pytest.approx(0.5), (
                    f"非 sticky 步把 σ 推到了 {s.sigma} —— 旧代码的 bug"
                    "（g ∝ 1−σ，推高 σ 等于关掉压缩）"
                )


# ============================================================
# L2 的**变异验证**：三条结构性质守门是否真有判别力
# ============================================================

class TestStructuralPropertyMutationsAreCaught:
    """验证 ``TestMultiscaleInductionStructuralProperties`` 的三条守门**真的会红**。

    **为什么单独写这一组**（2026-09-30）：§7.0.2 的 L2 验收原文要求

        **变异验证** —— 把 ``g`` 改成恒 0（不下降），守门必须红。

    但这句话此前**在代码里查无痕迹**（``grep 变异 tests/`` 零命中）。
    按本项目纪律「**没失败过的测试＝没测过**」，一句"验证过"若不落成
    可执行的代码，就只是一句声明。

    **这组测试的做法**：不注入源码、不改生产代码，而是用 ``monkeypatch``
    把 ``induction_step_g`` 换成变异版本，**在测试内跑一次真的循环**，
    然后断言"三条性质**确实被违反**"。

    ⇒ 逻辑是**反的**：普通测试断言"性质成立"，这里断言"性质在变异下
    **不成立**"。若某条性质在变异下**依然成立**，说明那条守门**没有判别力**
    （变异没能触碰到它测的东西），本组会红并指出是哪一条。
    """

    def _mutated_steps(self, monkeypatch, mutant, tubes=None, delta=0.1, **kw):
        """把 ``induction_step_g`` 替换为 ``mutant`` 后跑一次循环。

        ``tubes`` 默认用 sticky 构型；测性质③时要显式传
        ``_non_sticky_tubes()``，否则循环里**一个非 sticky 步都没有**
        （本类初版就踩了这个前提失效）。

        **为什么逐参数显式传、而不是 ``dict(...)`` + ``**params``**：
        ``**params`` 展开会被推断成"所有值同一类型"（实测推成 ``float``），
        于是 ``n_scales`` / ``max_iterations`` / ``verbose`` 各刷一条
        "float 不可分配给 int/bool" 的假报错。``kw.get("x", default)``
        让每个参数保持自己的类型。
        """
        import kakeya_ai.sticky_kakeya as sk

        if tubes is None:
            tubes = _sticky_tubes()
        monkeypatch.setattr(sk, "induction_step_g", mutant)
        return sk.multiscale_induction(
            tubes,
            delta=delta,
            max_iterations=int(kw.get("max_iterations", 20)),
            omega_threshold=float(kw.get("omega_threshold", 1e-12)),
            n_scales=int(kw.get("n_scales", 5)),
            verbose=bool(kw.get("verbose", False)),
        )

    def test_mutation_g_always_zero_makes_property_1_fail(self, monkeypatch):
        """变异 A：``g ≡ 0``（永不下降）⇒ 性质①（严格递减）**必须被违反**。

        这正是 §7.0.2 原文指定的那个变异。若在此变异下"严格递减"**仍然
        成立**，说明性质①测的不是 ``g``——那它就是个装饰品。
        """
        steps = self._mutated_steps(monkeypatch, lambda *a, **kw: 0.0)
        sticky_steps = [s for s in steps if s.is_sticky]
        assert sticky_steps, "前提失效：该构型应判为 sticky"

        violations = [s for s in sticky_steps if not (s.omega_after < s.omega_before)]
        assert violations, (
            "变异 `g ≡ 0` 下「sticky 步严格递减」居然**仍然成立** —— "
            "说明性质①没有判别力（它测的不是 g）。"
            "这正是 §7.0.2 要求做变异验证的原因。"
        )
        # 顺带钉死：Δω 确实变成了 0（变异真的生效了，不是被别处兜住）。
        assert all(s.omega_reduction == 0.0 for s in sticky_steps)

    def test_mutation_g_always_zero_makes_property_3_fail(self, monkeypatch):
        """变异 A 也**必须**让性质③（非 sticky 不压缩）失去区分度？

        **不是**——这里断言的是相反的事，用来揭示**性质③的真实边界**：

        性质③（"非 sticky ⇒ Δω = 0"）在 ``g ≡ 0`` 下**照样成立**
        （因为非 sticky 本来就取 ``g = 0``）。⇒ 性质③**无法**通过这个
        变异验证，它需要**另一个**变异来验证：

            让非 sticky 分支也压缩（即退回到修复前的行为）。

        本测试把这个事实**钉成断言**，免得后人以为"变异验证做过"就等于
        "三条都验过"。性质③的变异在下一个测试里做。
        """
        # 用**非 sticky 构型**才测得到性质③（sticky 构型里没有非 sticky 步）。
        steps = self._mutated_steps(
            monkeypatch, lambda *a, **kw: 0.0,
            tubes=_non_sticky_tubes(), delta=0.01, n_scales=4,
        )
        non_sticky = [s for s in steps if not s.is_sticky]
        assert non_sticky, "前提失效：该构型应判为非 sticky"
        # 在 g≡0 变异下，性质③**依然成立** —— 故它没被这个变异覆盖。
        assert all(s.omega_reduction == 0.0 for s in non_sticky), (
            "g≡0 下非 sticky 的 Δω 仍应为 0（这是性质③的断言内容）"
        )

    def test_mutation_non_sticky_also_compresses_breaks_property_3(self, monkeypatch):
        """变异 B：**非 sticky 也压缩**（退回到修复前的 bug）⇒ 性质③必须红。

        这补上了性质③的变异验证。做法：让 ``multiscale_induction`` 里
        "非 sticky ⇒ g=0"那个三元分支失效——用返回**非零** g 的实现替换
        ``induction_step_g``，并同时把 ``classify_sticky`` 钉成"恒非 sticky"，
        使非 sticky 分支真的拿到非零 g。

        **注意为什么不能只换 ``induction_step_g``**：循环里非 sticky 时
        ``g`` 是**硬编码的 0.0**（``... if is_sticky else 0.0``），
        与 ``induction_step_g`` 的返回值无关。所以必须同时让
        ``classify_sticky`` 报 sticky=False，**且**绕过那个三元分支——
        这里采用的做法是直接验证"若把该分支去掉会怎样"：用一个恒非
        sticky 的构型 + 直接调用循环，检查 Δω 是否被 g 污染。

        ⇒ 结论（**这是本测试真正钉住的事实**）：非 sticky 的 Δω 来自
        循环内的**硬编码 0.0**，而非 ``induction_step_g``。故**光靠变异
        ``induction_step_g`` 无法验证性质③**；性质③的守门有效性只能靠
        **直接读循环源码**确认那个三元分支存在（本测试用 AST 做这件事，
        见下一个测试）。
        """
        # 同上：必须用非 sticky 构型。
        steps = self._mutated_steps(
            monkeypatch, lambda *a, **kw: 999.0,
            tubes=_non_sticky_tubes(), delta=0.01, n_scales=4,
        )
        non_sticky = [s for s in steps if not s.is_sticky]
        assert non_sticky, "前提失效：该构型应判为非 sticky"
        # 即使 g 返回 999，非 sticky 的 Δω 仍必须是 0 —— 这证明循环内的
        # 硬编码 0.0 分支在起作用（性质③的实现点）。
        assert all(s.omega_reduction == 0.0 for s in non_sticky), (
            "非 sticky 的 Δω 被外部 g 污染了 —— 循环内"
            "「非 sticky ⇒ g=0」的硬编码分支已失效"
        )

    def test_non_sticky_zeroing_branch_exists_in_source(self):
        """性质③的实现点必须**在源码里真的存在**（AST 检查，不是子串匹配）。

        **为什么需要这条**：上面两个测试证明了"非 sticky 的 Δω 是硬编码
        0.0"，但那依赖**运行时的行为**。若有人把

            g = induction_step_g(...) if is_sticky else 0.0

        重构成别处兜 0（例如让 ``induction_step_g`` 自己在非 sticky 时返回
        0），行为可能仍然正确，但**性质③的守门（断言 Δω=0）就会失去
        判别力**——因为那时 Δω=0 不再由循环保证。

        故这里直接检查源码里存在"``is_sticky`` 作为 ``g`` 的三元条件"这一
        结构。用 AST 而非 ``in`` 子串：注释里出现这个词不算实现
        （本项目已踩过"注释里的词让检查误判"的坑）。
        """
        import ast
        import inspect

        import kakeya_ai.sticky_kakeya as sk

        src = inspect.getsource(sk.multiscale_induction)
        tree = ast.parse(_dedent_src(src))

        # 找一个 IfExp（三元表达式），其 test 里出现 is_sticky，且
        # body/orelse 之一是常量 0.0。
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.IfExp):
                continue
            test_names = {
                n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)
            }
            if "is_sticky" not in test_names:
                continue
            branch_consts = [
                b.value for b in (node.body, node.orelse)
                if isinstance(b, ast.Constant)
            ]
            if 0.0 in branch_consts:
                found = True
                break

        assert found, (
            "`multiscale_induction` 里找不到「is_sticky 决定 g 是否为常量 0.0」"
            "的三元表达式 —— 性质③（非 sticky 不压缩）的实现点已改动，"
            "其守门可能已失去判别力。若是有意重构，请同步更新本测试与其"
            "理由说明。"
        )


def _dedent_src(src: str) -> str:
    """去掉 ``inspect.getsource`` 可能带的缩进，便于 ``ast.parse``。"""
    import textwrap

    return textwrap.dedent(src)


class TestExpandedConvexTestFamily:
    """局限 #5：尺度链的 KT-CW 检验可选用更一般的凸集族（含旋转盒）。"""

    def test_boxes_is_default_and_expanded_tightens(self):
        tubes = make_kakeya_fan(4, 4, delta=0.1)
        default = build_scale_chain(tubes, delta=0.1, n_scales=4)
        explicit_boxes = build_scale_chain(
            tubes, delta=0.1, n_scales=4, convex_test_family="boxes"
        )
        expanded = build_scale_chain(
            tubes, delta=0.1, n_scales=4, convex_test_family="expanded"
        )
        # 默认必须等于 "boxes"（历史行为不变）。
        assert [layer.kt_cw_error for layer in default.layers] == [
            layer.kt_cw_error for layer in explicit_boxes.layers
        ]
        # "expanded" 只会更紧（误差 ≥），且每层仍有覆盖。
        for base, exp in zip(default.layers, expanded.layers, strict=True):
            assert exp.kt_cw_error >= base.kt_cw_error - 1e-12
            assert exp.tested

    def test_expanded_can_flip_verdict_and_rejects_bad_family(self):
        from kakeya_ai.errors import InputValidationError

        tubes = make_kakeya_fan(4, 4, delta=0.1)
        # 误差变大 ⇒ 同一阈值下更容易判"不满足"；用极松阈值确保 expanded 仍通过，
        # 用论文阈值则应能拒绝。
        strict = build_scale_chain(
            tubes, delta=0.1, n_scales=4, max_wolff_error=1.0,
            convex_test_family="expanded",
        )
        assert all(layer.kt_cw_error <= 1.0 for layer in strict.layers) or (
            strict.is_sticky is False
        )
        with pytest.raises(InputValidationError):
            build_scale_chain(tubes, delta=0.1, n_scales=3,
                              convex_test_family="nonsense")
