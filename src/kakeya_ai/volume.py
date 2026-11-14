"""δ-管之并集及其遮蔽的体积估计。

提供确定性的基于体素的估计器，以及一个蒙特卡洛估计器。
它们面向数值实验，并用于对定理 1.2 / 断言 D / 断言 E 的回归测试。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.special import erfcinv

from .primitives import Shading, Tube
from .errors import InputValidationError


def _make_grid(bounds: np.ndarray, n: int):
    """在 bounds 上返回 n×n×n 网格的体素中心与体素体积。"""
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    dims = hi - lo
    voxel_size = dims / n
    voxel_volume = float(np.prod(voxel_size))

    xs = np.linspace(lo[0] + voxel_size[0] / 2, hi[0] - voxel_size[0] / 2, n)
    ys = np.linspace(lo[1] + voxel_size[1] / 2, hi[1] - voxel_size[1] / 2, n)
    zs = np.linspace(lo[2] + voxel_size[2] / 2, hi[2] - voxel_size[2] / 2, n)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid, voxel_volume


def estimate_union_volume_voxel(
    tubes: Sequence[Tube],
    grid_n: int = 128,
    bounds: np.ndarray | None = None,
) -> float:
    """基于体素估计 |∪ T|。

    Parameters
    ----------
    tubes : sequence of Tube
    grid_n : int
        每个轴向上的体素数。
    bounds : (3, 2) array, optional
        包围盒；默认为单位球的外接立方体 [-1,1]^3。

    Returns
    -------
    volume : float
    """
    if not tubes:
        return 0.0
    if bounds is None:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    grid, voxel_volume = _make_grid(bounds, grid_n)

    inside = np.zeros(grid.shape[0], dtype=bool)
    for t in tubes:
        inside |= t.contains(grid)
    return float(np.count_nonzero(inside)) * voxel_volume


def estimate_shaded_union_volume_voxel(
    tubes: Sequence[Tube],
    shading: Shading,
    grid_n: int = 128,
    bounds: np.ndarray | None = None,
) -> float:
    """基于体素估计 |∪ Y(T)|。

    这里 Y(T) 被建模为 T 沿其长度方向的居中 λ 比例部分，
    其中 λ 是 T 的遮蔽密度。 这是一种简单的合成遮蔽；
    真实实验可以用任意子集来替换它。

    Parameters
    ----------
    tubes : sequence of Tube
    shading : Shading
    grid_n : int
    bounds : (3, 2) array, optional

    Returns
    -------
    volume : float
    """
    if not tubes:
        return 0.0
    if len(shading.densities) != len(tubes):
        raise InputValidationError("Shading length must match number of tubes.")
    if bounds is None:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    grid, voxel_volume = _make_grid(bounds, grid_n)

    inside = np.zeros(grid.shape[0], dtype=bool)
    # 长度已在函数开头显式校验（不等则提前 raise），故 strict 恒成立；
    # 写上它使"这里已保证等长"成为代码里可见的依赖，而不靠向上翻十行。
    for t, lam in zip(tubes, shading.densities, strict=True):
        if lam <= 0:
            continue
        pts = grid
        a, b = t.endpoints
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-24:
            t_in = np.linalg.norm(pts - a, axis=1) <= t.delta
        else:
            t_proj = np.sum((pts - a) * ab, axis=1) / ab_len2
            t_in = (t_proj >= (1 - lam) / 2) & (t_proj <= (1 + lam) / 2)
            t_in &= t.contains(pts)
        inside |= t_in
    return float(np.count_nonzero(inside)) * voxel_volume


def estimate_union_volume_monte_carlo(
    tubes: Sequence[Tube],
    n_samples: int = 200_000,
    bounds: np.ndarray | None = None,
    seed: int | None = None,
) -> float:
    """对 |∪ T| 的蒙特卡洛估计。

    Parameters
    ----------
    tubes : sequence of Tube
    n_samples : int
    bounds : (3, 2) array, optional
    seed : int, optional

    Returns
    -------
    volume : float
    """
    if not tubes:
        return 0.0
    if bounds is None:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    rng = np.random.default_rng(seed)
    pts = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, 3))
    box_volume = float(np.prod(bounds[:, 1] - bounds[:, 0]))

    inside = np.zeros(n_samples, dtype=bool)
    for t in tubes:
        inside |= t.contains(pts)
    return float(np.count_nonzero(inside)) / n_samples * box_volume


def monte_carlo_confidence_interval(
    n_inside: int,
    n_samples: int,
    box_volume: float,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """返回蒙特卡洛体积估计的 Wilson 得分置信区间。

    估计量为 ``volume = (n_inside / n_samples) * box_volume``。
    选用 Wilson 区间，是因为它在比例很小或很极端时也表现良好。

    Parameters
    ----------
    n_inside : int
        被接受的蒙特卡洛样本数。
    n_samples : int
        样本总数。
    box_volume : float
        采样盒的体积。
    confidence : float
        置信水平，落在 (0, 1) 内。

    Returns
    -------
    estimate, lower, upper : float
        体积估计与置信上下界。
    """
    if n_samples <= 0 or box_volume <= 0:
        return 0.0, 0.0, 0.0

    p_hat = float(n_inside) / float(n_samples)
    # 双侧正态分位数。
    alpha = 1.0 - confidence
    z = float(np.sqrt(2.0) * erfcinv(alpha))

    z2_n = (z * z) / n_samples
    denom = 1.0 + z2_n
    centre = (p_hat + 0.5 * z2_n) / denom
    half_width = (
        z
        * np.sqrt(
            (p_hat * (1.0 - p_hat) / n_samples) + (z * z) / (4.0 * n_samples * n_samples)
        )
        / denom
    )

    lower = max(0.0, centre - half_width) * box_volume
    upper = min(1.0, centre + half_width) * box_volume
    estimate = p_hat * box_volume
    return float(estimate), float(lower), float(upper)


def estimate_union_volume_monte_carlo_with_ci(
    tubes: Sequence[Tube],
    n_samples: int = 200_000,
    bounds: np.ndarray | None = None,
    seed: int | None = None,
    confidence: float = 0.95,
    backend: str | None = None,
) -> dict:
    """带置信区间的 |∪ T| 蒙特卡洛估计。

    Parameters
    ----------
    tubes : sequence of Tube
    n_samples : int
    bounds : (3, 2) array, optional
    seed : int, optional
    confidence : float
    backend : str, optional
        ``None``（默认）= 纯 numpy 参考路径（与
        :func:`estimate_union_volume_monte_carlo` 同随机流，同 seed 时
        两者逐位一致——测试钉住这个不变量）；``"nano"`` = 融合采样路径
        （片内 splitmix64 生成采样点，点数组不进内存——算子融合，
        能耗动机见 PROJECT.md 第 III 篇 §6；**要求显式 seed**，同 seed
        结果逐位确定、与并行调度无关）。置信区间的计算两条路径共用。

    Returns
    -------
    result : dict
        - ``estimate`` ：体积估计。
        - ``ci_lower`` / ``ci_upper`` ：置信区间的上下界。
        - ``relative_error`` ：半宽 / 估计值（估计值为 0 时为 0）。
        - ``n_inside`` ：被接受的样本数。
        - ``n_samples`` ：样本总数。
    """
    if not tubes:
        return {
            "estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "relative_error": 0.0,
            "n_inside": 0,
            "n_samples": 0,
        }
    if bounds is None:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    box_volume = float(np.prod(bounds[:, 1] - bounds[:, 0]))

    if backend == "nano":
        # 融合路径：n_inside 来自片内采样（nano 不可用/缺入口时
        # nano_mc_count_inside 会显式报错，不静默降级——调用方显式
        # 选了 nano，给它 numpy 结果会撒谎）。
        if seed is None:
            raise InputValidationError(
                "backend='nano' 需要显式 seed（融合 RNG 的确定性依赖种子；"
                "要每次随机请用默认的 numpy 路径）"
            )
        from .nano_kernel import nano_mc_count_inside

        n_inside = nano_mc_count_inside(tubes, n_samples, bounds, seed)
    else:
        rng = np.random.default_rng(seed)
        pts = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_samples, 3))
        inside = np.zeros(n_samples, dtype=bool)
        for t in tubes:
            inside |= t.contains(pts)
        n_inside = int(np.count_nonzero(inside))

    estimate, lower, upper = monte_carlo_confidence_interval(
        n_inside, n_samples, box_volume, confidence=confidence
    )
    half_width = 0.5 * (upper - lower)
    relative_error = half_width / estimate if estimate > 1e-24 else 0.0

    return {
        "estimate": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "relative_error": relative_error,
        "n_inside": n_inside,
        "n_samples": n_samples,
    }


def estimate_volume_vs_theorem_1_2(
    tubes: Sequence[Tube],
    epsilon: float,
    K: float,
    lambda_min: float,
    n_samples: int = 100_000,
    seed: int | None = None,
) -> dict:
    """把蒙特卡洛并集体积估计与定理 1.2 的下界作比较。

    返回一个 dict，含体积估计、下界，以及一个布尔标志 ``satisfies``——
    当估计值不小于下界（在蒙特卡洛置信区间意义下）时为 True。
    """
    lower_bound = theorem_1_2_lower_bound(tubes, epsilon, K, lambda_min)
    mc = estimate_union_volume_monte_carlo_with_ci(tubes, n_samples=n_samples, seed=seed)
    return {
        "volume_estimate": mc["estimate"],
        "ci_lower": mc["ci_lower"],
        "ci_upper": mc["ci_upper"],
        "theorem_1_2_lower_bound": lower_bound,
        "satisfies": mc["ci_lower"] >= lower_bound,
    }


def theorem_1_2_lower_bound(
    tubes: Sequence[Tube],
    epsilon: float,
    K: float,
    lambda_min: float,
) -> float:
    """定理 1.2 的右端项：δ^ε λ^K Σ|T|。

    这是一个**目标**下界；实际的并集体积应当与它作比较。
    """
    if not tubes:
        return 0.0
    delta = tubes[0].delta
    return (delta ** epsilon) * (lambda_min ** K) * sum(t.volume for t in tubes)


def assertion_d_lower_bound(
    tubes: Sequence[Tube],
    sigma: float,
    omega: float,
    epsilon: float,
    kappa: float,
) -> float:
    """断言 D(σ, ω) 的右端下界表达式，其中 κ 为常数、δ 为管半径：κ δ^{ω+ε} (#T)|T| ((#T)|T|^{1/2})^{-σ}。"""
    if not tubes:
        return 0.0
    n = len(tubes)
    t = tubes[0]
    return kappa * (t.delta ** (omega + epsilon)) * n * t.volume * ((n * t.volume ** 0.5) ** (-sigma))


def assertion_e_lower_bound(
    tubes: Sequence[Tube],
    sigma: float,
    omega: float,
    epsilon: float,
    kappa: float,
    m: float,
    ell: float,
) -> float:
    """断言 E(σ, ω) 的右端下界表达式，其中 κ 为常数、m 与 ℓ 为附加参数：κ δ^{ω+ε} m^{-1} (#T)|T| (m^{-3/2} ℓ (#T)|T|^{1/2})^{-σ}。"""
    if not tubes:
        return 0.0
    n = len(tubes)
    t = tubes[0]
    return (
        kappa
        * (t.delta ** (omega + epsilon))
        * (m ** -1)
        * n
        * t.volume
        * ((m ** -1.5) * ell * n * (t.volume ** 0.5)) ** (-sigma)
    )
