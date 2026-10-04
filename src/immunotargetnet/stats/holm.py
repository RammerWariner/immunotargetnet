"""Holm step-down multiplicity control.

Ref: Sec. 2.7.3 (Holm correction within each family of comparisons).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HolmResult:
    raw: np.ndarray
    adjusted: np.ndarray
    rejected: np.ndarray
    family_size: int
    alpha: float

    def summary(self) -> dict[str, float]:
        return {
            "family_size": float(self.family_size),
            "rejected": float(self.rejected.sum()),
            "smallest_raw": float(self.raw.min()) if self.raw.size else float("nan"),
        }


def holm_step_down(p_values: np.ndarray, alpha: float = 0.05) -> HolmResult:
    """Adjusted p-values carrying the running maximum of the scaled values.

    The running maximum is what makes the adjusted values monotone in the raw order; a
    running minimum would collapse every value to the Bonferroni level.
    """
    raw = np.asarray(p_values, dtype=np.float64)
    if raw.size == 0:
        return HolmResult(raw=raw, adjusted=raw.copy(), rejected=raw.astype(bool), family_size=0, alpha=alpha)
    order = np.argsort(raw, kind="stable")
    family = raw.size
    scaled = np.empty(family, dtype=np.float64)
    running = 0.0
    for position, index in enumerate(order):
        candidate = float(raw[index]) * (family - position)
        running = max(running, candidate)
        scaled[index] = min(running, 1.0)
    rejected = scaled <= alpha
    return HolmResult(raw=raw, adjusted=scaled, rejected=rejected, family_size=family, alpha=alpha)


def holm_report(labels: list[str], p_values: np.ndarray, alpha: float = 0.05) -> list[dict[str, float]]:
    result = holm_step_down(p_values, alpha=alpha)
    return [
        {
            "index": float(index),
            "raw": float(result.raw[index]),
            "adjusted": float(result.adjusted[index]),
            "rejected": float(result.rejected[index]),
        }
        for index, _ in enumerate(labels)
    ]
