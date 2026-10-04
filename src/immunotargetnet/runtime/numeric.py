"""Numeric helpers shared by the encoders and the evidence readers.

Ref: Sec. 2.2 (variance-stabilising within-cohort contrast, inverse-variance pooling).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-9


def variance_stabilising_transform(values: np.ndarray, *, offset: float = 1.0) -> np.ndarray:
    """log2(x + offset) with negatives clamped, the transform the cohort layer corrects for."""
    clamped = np.clip(values, 0.0, None)
    transformed: np.ndarray = np.log2(clamped + offset)
    return transformed


def inverse_variance_pool(effects: np.ndarray, variances: np.ndarray) -> tuple[float, float]:
    """Combine per-cohort contrasts; returns (pooled effect, pooled standard error)."""
    weights = 1.0 / np.maximum(variances, EPS)
    pooled = float(np.sum(weights * effects) / np.sum(weights))
    standard_error = float(np.sqrt(1.0 / np.sum(weights)))
    return pooled, standard_error


def moderated_contrast(group_a: np.ndarray, group_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature difference of means with a pooled-variance standard error.

    Both blocks are (samples, features); the return value is (effect, standard error),
    each of length `features`.
    """
    n_a, n_features = group_a.shape
    n_b = group_b.shape[0]
    effect = group_a.mean(axis=0) - group_b.mean(axis=0)
    var_a = group_a.var(axis=0, ddof=1) if n_a > 1 else np.zeros(n_features)
    var_b = group_b.var(axis=0, ddof=1) if n_b > 1 else np.zeros(n_features)
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1)
    standard_error = np.sqrt(pooled * (1.0 / max(n_a, 1) + 1.0 / max(n_b, 1)))
    return effect, standard_error


def zscore(values: np.ndarray, axis: int = -1) -> np.ndarray:
    centred = values - values.mean(axis=axis, keepdims=True)
    scale = np.maximum(values.std(axis=axis, keepdims=True), 1e-6)
    out: np.ndarray = centred / scale
    return out


def rank_transform(values: np.ndarray) -> np.ndarray:
    """Average-rank transform, used by the UCell-style signature score."""
    return np.argsort(np.argsort(values, kind="stable"), kind="stable").astype(np.float64)


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), EPS)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), EPS)
    out: np.ndarray = left_norm @ right_norm.T
    return out


def trapezoid_integral(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2:
        return 0.0
    widths = np.diff(xs)
    areas = widths * (ys[:-1] + ys[1:]) / 2.0
    return float(np.sum(areas))


@dataclass(frozen=True)
class ScalePair:
    """A location/scale pair fitted on a training block and reapplied elsewhere."""

    location: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, axis: int = 0) -> ScalePair:
        location = values.mean(axis=axis)
        scale = np.maximum(values.std(axis=axis), 1e-6)
        return cls(location=location, scale=scale)

    def apply(self, values: np.ndarray) -> np.ndarray:
        out: np.ndarray = (values - self.location) / self.scale
        return out

    def invert(self, values: np.ndarray) -> np.ndarray:
        out: np.ndarray = values * self.scale + self.location
        return out
