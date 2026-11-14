"""Kakeya AI 项目的基础几何原语。

δ-管（δ-tube）被建模为 R^3 中单位线段的 δ-邻域。
为便于数值计算，我们保存线段的端点、方向、半径 δ 以及预先算好的体积。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
from .errors import InputValidationError


Vec3 = np.ndarray


@dataclass(frozen=True)
class Direction:
    """R^3 中的单位方向向量。"""

    v: np.ndarray

    def __post_init__(self):
        v = np.asarray(self.v, dtype=float)
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            raise InputValidationError("Direction vector must be non-zero.")
        object.__setattr__(self, "v", v / norm)

    @classmethod
    def from_spherical(cls, theta: float, phi: float) -> "Direction":
        """由球坐标角构造一个单位方向。"""
        return cls(
            np.array(
                [
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta),
                ]
            )
        )

    @classmethod
    def from_array(cls, arr: Sequence[float]) -> "Direction":
        return cls(np.asarray(arr, dtype=float))

    def angle_to(self, other: "Direction") -> float:
        """夹角间隔，以弧度计（0 到 π）。"""
        cos = float(np.clip(np.dot(self.v, other.v), -1.0, 1.0))
        return float(np.arccos(cos))

    def __repr__(self) -> str:
        return f"Direction({self.v.round(4).tolist()})"


@dataclass(frozen=True)
class Tube:
    """δ-管：单位线段的 δ-邻域。

    Attributes
    ----------
    center : np.ndarray
        中心线段的中点。
    direction : Direction
        管的单位方向。
    length : float
        中心线段的长度（默认 1.0）。
    delta : float
        横截面半径（粗细为 2δ）。
    """

    center: np.ndarray
    direction: Direction
    length: float = 1.0
    delta: float = 1e-2

    def __post_init__(self):
        c = np.asarray(self.center, dtype=float)
        object.__setattr__(self, "center", c)
        if self.length <= 0 or self.delta <= 0:
            raise InputValidationError("Tube length and delta must be positive.")

    @property
    def endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """中心线段的两个端点。"""
        half = 0.5 * self.length * self.direction.v
        return self.center - half, self.center + half

    @property
    def volume(self) -> float:
        """近似体积：π * δ^2 * length。"""
        return np.pi * self.delta * self.delta * self.length

    def contains(self, points: np.ndarray) -> np.ndarray:
        """返回落在距线段 δ 范围内的点的布尔掩码。

        Parameters
        ----------
        points : np.ndarray, shape (N, 3)

        Returns
        -------
        mask : np.ndarray, shape (N,), bool
        """
        pts = np.atleast_2d(points).astype(float)
        a, b = self.endpoints
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-24:
            return np.linalg.norm(pts - a, axis=1) <= self.delta
        t = np.clip(np.sum((pts - a) * ab, axis=1) / ab_len2, 0.0, 1.0)
        closest = a + np.outer(t, ab)
        dist = np.linalg.norm(pts - closest, axis=1)
        return dist <= self.delta

    def __repr__(self) -> str:
        return (
            f"Tube(center={self.center.round(4).tolist()}, "
            f"dir={self.direction.v.round(4).tolist()}, "
            f"length={self.length}, delta={self.delta})"
        )


@dataclass(frozen=True)
class Shading:
    """管集合的遮蔽（Shading）Y(T)。

    为简单起见，我们为每根管保存一个密度 λ ∈ [0,1]；实际的子集可在之后
    再采样。 λ = 1 表示 Y(T) = T。
    """

    densities: np.ndarray

    def __post_init__(self):
        d = np.asarray(self.densities, dtype=float)
        if np.any(d < 0) or np.any(d > 1):
            raise InputValidationError("Shading densities must lie in [0,1].")
        object.__setattr__(self, "densities", d)

    @property
    def lambda_min(self) -> float:
        return float(np.min(self.densities))

    def total_volume(self, tubes: Sequence[Tube]) -> float:
        """Σ |Y(T)| = Σ λ(T) |T|。

        ``strict=True``：``Shading`` 与 ``tubes`` 必须一一对应（调用方在
        体积估计前会显式校验长度，见 ``volume.py``）。长度不等意味着上游有
        bug，此时**显式报错**远好于静默少算几根管的份额。
        """
        return sum(
            t.volume * lam
            for t, lam in zip(tubes, self.densities, strict=True)
        )


@dataclass
class TubeCollection:
    """一组管（T）连同可选的遮蔽（Y）。"""

    tubes: List[Tube] = field(default_factory=list)
    shading: Shading | None = None

    def __post_init__(self):
        if self.shading is not None and len(self.shading.densities) != len(self.tubes):
            raise InputValidationError("Shading length must match number of tubes.")

    def add(self, tube: Tube, density: float = 1.0) -> None:
        self.tubes.append(tube)
        if self.shading is None:
            self.shading = Shading(np.array([float(density)]))
        else:
            old = self.shading.densities
            self.shading = Shading(np.append(old, [float(density)]))

    @property
    def count(self) -> int:
        return len(self.tubes)

    @property
    def total_tube_volume(self) -> float:
        return sum(t.volume for t in self.tubes)

    def __repr__(self) -> str:
        return f"TubeCollection(count={self.count}, total_volume={self.total_tube_volume:.6g})"
