"""体积估计的「域」（domain）——显式、可写盘、不静默改的基准区域。

**为什么单独抽出来。** 体积估计必须在**同一个域**上做，否则数值不可比。
``[-1,1]³`` 是本项目的历史基准域，它不是一个随手写的默认值：

- 已入库的 ``.kpt`` 证明轨迹，其 ``union_volume`` / ``lower_bound`` 全是在它下面
  算出来的；改动它会让历史记录复核不通过（实测：``0.04803240741`` → 重算
  ``0.05004272539``，见内部设计文档）。
- ``±1.0`` 与 2 的幂在 float32 / float64 下都**精确可表示**，所以 nano(f32) 与
  numpy(f64) 的体素网格逐位对齐、"跨后端逐位一致"这条不变量成立。
- ``verify_trace`` 用它重算：写出去的文件与复核的重算能对上，靠的就是域一致。

**三条不变量（改本模块前必读）**

1. **域不变**：默认域永远等于历史基准 ``[-1,1]³``。只有调用方**显式**给出更大的
   域（例如 NSP 因子化把副本撒到更远处）才允许扩大。
2. **不收缩**：域一旦扩大，在同一次运行内不回缩——否则中段与前后段的数值不可比。
3. **可复现**：域半径必须写进 trace（``params.domain_radius``），复核方据此重算。

**为什么扩展时半径取 2 的幂**：``±2^k`` 在 f32/f64 下都精确，配合 2 的幂的 ``grid_n``
可让两种精度的体素网格逐位相同。此前"自动包紧"方案（按管族取紧致包围盒）失败的原因
正是紧致盒的坐标**不是** f32 精确值，导致 nano 与 numpy 的网格错位约 1e-8——
"内部通、外部断"。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .primitives import Tube
from .errors import InputValidationError

#: 历史基准域半径。**不要改**：已入库 trace 的数值全部依赖它。
DEFAULT_RADIUS = 1.0

#: 各后端共用同一套 bounds 边界表达：其形状依次为 [[xlo,xhi],[ylo,yhi],[zlo,zhi]]，
#: 即每个轴各自的区间端点。
Bounds = np.ndarray


def bounds_for_radius(radius: float) -> Bounds:
    """把域半径变成各后端接受的 ``bounds``（对称立方体）。

    ``radius=1.0`` 时返回的数组与历史硬编码值**逐位相同**（``[[-1,1],[-1,1],[-1,1]]``），
    因此默认路径上所有历史数值不变。
    """
    r = float(radius)
    if not math.isfinite(r) or r <= 0:
        raise InputValidationError(f"域半径必须为正的有限值，实际为 {radius!r}")
    return np.array([[-r, r], [-r, r], [-r, r]], dtype=float)


def _ceil_pow2(x: float) -> float:
    """不小于 ``x`` 的最小 2 的幂（用于保证 f32/f64 精确表示）。"""
    if not math.isfinite(x) or x <= 0:
        return DEFAULT_RADIUS
    e = int(math.ceil(math.log2(x)))
    r = float(2.0 ** e)
    # log2 的浮点误差可能让 2^e 略微小于 x，补一档。
    while r < x:
        r *= 2.0
    return r


def required_radius(tubes: Sequence[Tube]) -> float:
    """包住**整根管**（端点 + 半径 δ）所需的最小对称域半径。

    返回**未取整**的值；要用于建网格请用 :func:`radius_covering`。
    """
    if not tubes:
        return 0.0
    m = 0.0
    for t in tubes:
        a, b = t.endpoints
        r = float(t.delta)
        for p in (a, b):
            m = max(m, float(np.max(np.abs(p))) + r)
    return m


def radius_covering(
    tubes: Sequence[Tube],
    *,
    floor: float = DEFAULT_RADIUS,
) -> float:
    """给定管族，返回**可直接用于建网格**的域半径。

    规则（三条不变量的落地）：

    - 半径是 2 的幂，且不小于 ``floor`` → 保证"不收缩"与 f32/f64 精确；
    - ``floor`` 默认为历史基准 1.0 → 管族没越界时结果恒为 1.0，
      历史数值逐位不变；
    - 越界时向上取到 2 的幂（如需要 1.9 → 2.0，需要 2.4 → 4.0）。

    Examples
    --------
    >>> radius_covering([], floor=1.0)
    1.0
    """
    need = max(float(floor), required_radius(tubes))
    return max(DEFAULT_RADIUS, _ceil_pow2(need))


def exceeds_domain(tubes: Sequence[Tube], radius: float) -> Optional[float]:
    """管族是否越界；越界则返回所需半径，未越界返回 ``None``。

    这是"**不静默丢数据**"的哨兵：体积估计的网格若装不下某根管，
    ``Tube.contains`` 会把它当盒外点忽略——体积偏小而不报错。调用方应当在
    放进网格前先问一句。
    """
    need = required_radius(tubes)
    if need > float(radius) + 0.0:      # 严格大于才算越界，边界上允许
        return need
    return None
