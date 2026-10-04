"""Attribution faithfulness under deletion.

Ref: Sec. 2.7.2 (whether removing high-attribution evidence perturbs the score more than
removing low-attribution evidence) and Sec. 2.5 (the attribution is the simplex coordinate
the model was optimised on, not a retrospective explanation).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np

from immunotargetnet.runtime.numeric import trapezoid_integral

# Fractions of the evidence axes removed at each point of the deletion curve. With four
# layers these are one, two and three layers.
DELETION_GRID: Final[tuple[float, ...]] = (0.25, 0.5, 0.75)

ScoreFromMask = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class DeletionCurve:
    fractions: np.ndarray
    high_order_drop: np.ndarray
    low_order_drop: np.ndarray

    def area_between(self) -> float:
        difference = self.high_order_drop - self.low_order_drop
        return trapezoid_integral(self.fractions, difference)


def deletion_curve(
    attribution: np.ndarray,
    score_from_mask: ScoreFromMask,
    *,
    fractions: tuple[float, ...] = DELETION_GRID,
) -> DeletionCurve:
    """Remove the most- and least-attributed evidence axes and record the score each way.

    `score_from_mask` takes a boolean mask over the evidence axes and returns the score
    vector under that mask, so the deletion is performed through the model itself.
    """
    order = np.argsort(-np.asarray(attribution, dtype=np.float64), kind="stable")
    n_axes = order.size
    if n_axes < 2:
        raise ValueError("attribution faithfulness needs at least two evidence axes")
    high: list[float] = []
    low: list[float] = []
    accepted: list[float] = []
    for fraction in fractions:
        drop = max(1, min(n_axes - 1, round(fraction * n_axes)))
        high_mask = np.ones(n_axes, dtype=bool)
        low_mask = np.ones(n_axes, dtype=bool)
        high_mask[order[:drop]] = False
        low_mask[order[-drop:]] = False
        high.append(float(np.asarray(score_from_mask(high_mask), dtype=np.float64).mean()))
        low.append(float(np.asarray(score_from_mask(low_mask), dtype=np.float64).mean()))
        accepted.append(float(fraction))
    return DeletionCurve(
        fractions=np.asarray(accepted, dtype=np.float64),
        high_order_drop=np.asarray(high, dtype=np.float64),
        low_order_drop=np.asarray(low, dtype=np.float64),
    )


def attribution_faithfulness(attribution: np.ndarray, score_from_mask: ScoreFromMask) -> float:
    """AUCdel: the area between the high-attribution and low-attribution deletion curves."""
    return deletion_curve(attribution, score_from_mask).area_between()


def uniform_attribution(n_axes: int) -> np.ndarray:
    return np.full(n_axes, 1.0 / max(n_axes, 1), dtype=np.float64)
