"""Two one-sided tests for the parity margin.

Ref: Sec. 2.6 (a 90% interval of the paired mean difference must sit inside the +-0.010
band; overlapping marginal intervals are explicitly not the test).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

PARITY_MARGIN: Final[float] = 0.010
PARITY_LEVEL: Final[float] = 0.90
MINIMUM_MEANINGFUL_KL_DROP: Final[float] = 0.20
MINIMUM_MEANINGFUL_FAITHFULNESS_GAIN: Final[float] = 0.25
MINIMUM_MEANINGFUL_TAU_GAIN: Final[float] = 0.08
PRIMARY_THRESHOLD: Final[float] = 0.50


@dataclass(frozen=True)
class EquivalenceVerdict:
    estimate: float
    level: float
    lower: float
    upper: float
    margin: float
    equivalent: bool
    detail: str


def paired_difference_interval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    level: float = PARITY_LEVEL,
    resamples: int = 1000,
    seed: int = 20260101,
) -> tuple[float, float, float]:
    """Percentile interval of the paired mean difference."""
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = np.array(
        [
            float(difference[generator.integers(0, difference.size, size=difference.size)].mean())
            for _ in range(resamples)
        ],
        dtype=np.float64,
    )
    alpha = (1.0 - level) / 2.0
    return (
        float(difference.mean()),
        float(np.percentile(draws, 100.0 * alpha)),
        float(np.percentile(draws, 100.0 * (1.0 - alpha))),
    )


def two_one_sided_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    margin: float = PARITY_MARGIN,
    level: float = PARITY_LEVEL,
    resamples: int = 1000,
    seed: int = 20260101,
) -> EquivalenceVerdict:
    """Parity holds only when the whole paired interval lies inside the margin."""
    estimate, lower, upper = paired_difference_interval(
        left, right, level=level, resamples=resamples, seed=seed
    )
    inside = (-margin <= lower) and (upper <= margin)
    if inside:
        detail = "paired interval lies inside the parity band"
    elif lower < -margin:
        detail = "paired interval extends below the parity band"
    else:
        detail = "paired interval extends above the parity band"
    return EquivalenceVerdict(
        estimate=estimate,
        level=level,
        lower=lower,
        upper=upper,
        margin=margin,
        equivalent=inside,
        detail=detail,
    )


def exceeds_minimum_meaningful(gain: float, *, axis: str) -> tuple[bool, float]:
    thresholds = {
        "immune_relevance_divergence": MINIMUM_MEANINGFUL_KL_DROP,
        "attribution_faithfulness": MINIMUM_MEANINGFUL_FAITHFULNESS_GAIN,
        "rank_stability": MINIMUM_MEANINGFUL_TAU_GAIN,
    }
    try:
        threshold = thresholds[axis]
    except KeyError as error:
        raise KeyError(f"no minimum meaningful difference defined for {axis}") from error
    return gain >= threshold, gain / threshold


def margin_multiple(gain: float, *, margin: float = PARITY_MARGIN) -> float:
    return abs(gain) / margin


def primary_criterion_met(value: float, *, threshold: float = PRIMARY_THRESHOLD) -> tuple[bool, float]:
    return value >= threshold, value - threshold
