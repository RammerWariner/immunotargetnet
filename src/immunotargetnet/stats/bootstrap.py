"""Blocked bootstrap over gene blocks.

Ref: Sec. 2.7.3 (95% blocked-bootstrap intervals from 1000 resamples, with genes that
recur across cohorts never treated as independent observations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

DEFAULT_RESAMPLES: Final[int] = 1000
DEFAULT_LEVEL: Final[float] = 0.95
PARITY_LEVEL: Final[float] = 0.90


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    lower: float
    upper: float
    level: float
    resamples: int

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


def percentile_interval(
    samples: np.ndarray, *, level: float = DEFAULT_LEVEL, point: float | None = None
) -> BootstrapInterval:
    alpha = (1.0 - level) / 2.0
    lower = float(np.percentile(samples, 100.0 * alpha))
    upper = float(np.percentile(samples, 100.0 * (1.0 - alpha)))
    centre = float(np.mean(samples)) if point is None else point
    return BootstrapInterval(point=centre, lower=lower, upper=upper, level=level, resamples=int(samples.size))


def bootstrap_metric(
    per_unit: np.ndarray,
    metric: object,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260101,
    level: float = DEFAULT_LEVEL,
) -> BootstrapInterval:
    """Resample units with replacement and report the interval of the recomputed metric."""
    if not callable(metric):
        raise TypeError("metric must be callable on a resampled block")
    values = np.asarray(per_unit, dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selection = generator.integers(0, values.size, size=values.size)
        draws[index] = float(metric(values[selection]))
    return percentile_interval(draws, level=level, point=float(metric(values)))


def blocked_bootstrap_difference(
    left: np.ndarray,
    right: np.ndarray,
    metric: object,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260101,
    level: float = PARITY_LEVEL,
) -> BootstrapInterval:
    """Paired bootstrap of the difference between two methods on the same units."""
    if not callable(metric):
        raise TypeError("metric must be callable on a resampled block")
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.size != right_values.size:
        raise ValueError("paired bootstrap needs matching unit counts")
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selection = generator.integers(0, left_values.size, size=left_values.size)
        draws[index] = float(metric(left_values[selection]) - metric(right_values[selection]))
    point = float(metric(left_values) - metric(right_values))
    return percentile_interval(draws, level=level, point=point)


def gene_blocks(units: list[tuple[str, str]]) -> dict[str, np.ndarray]:
    """Group evaluation units by gene so a gene recurring across cohorts is one block."""
    blocks: dict[str, list[int]] = {}
    for index, (gene, _cohort) in enumerate(units):
        blocks.setdefault(gene, []).append(index)
    return {gene: np.asarray(rows, dtype=np.int64) for gene, rows in blocks.items()}


def block_resample(per_unit: np.ndarray, blocks: dict[str, np.ndarray], *, seed: int) -> np.ndarray:
    """Resample whole gene blocks with replacement."""
    generator = np.random.default_rng(seed)
    keys = sorted(blocks)
    chosen = generator.integers(0, len(keys), size=len(keys))
    indices = np.concatenate([blocks[keys[index]] for index in chosen])
    return np.asarray(per_unit[indices], dtype=np.float64)
