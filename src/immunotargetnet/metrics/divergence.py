"""Immune-relevance divergence.

Ref: Sec. 2.7.2 (KL divergence to the held-out programme labels, plus Jensen-Shannon as
its bounded reading) and the constraint axis of Sec. 2.6.
"""

from __future__ import annotations

import numpy as np

from immunotargetnet.runtime.numeric import EPS

NATURAL_LOG: float = 1.0


def _normalise(distribution: np.ndarray) -> np.ndarray:
    floored = np.clip(np.asarray(distribution, dtype=np.float64), EPS, None)
    return np.asarray(floored / floored.sum(axis=-1, keepdims=True), dtype=np.float64)


def kl_divergence(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """KL(predicted || target) per gene, in nats."""
    left = _normalise(predicted)
    right = _normalise(target)
    return np.asarray((left * (np.log(left) - np.log(right))).sum(axis=-1) / NATURAL_LOG)


def mean_kl_divergence(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(kl_divergence(predicted, target).mean())


def jensen_shannon_divergence(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """JS divergence in nats; bounded by ln 2 and therefore readable as a ceiling."""
    left = _normalise(predicted)
    right = _normalise(target)
    midpoint = 0.5 * (left + right)
    value = 0.5 * (
        (left * (np.log(left) - np.log(midpoint))).sum(axis=-1)
        + (right * (np.log(right) - np.log(midpoint))).sum(axis=-1)
    )
    return np.asarray(value)


def mean_jensen_shannon_divergence(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(jensen_shannon_divergence(predicted, target).mean())


def kl_drop(constrained: float, unconstrained: float) -> float:
    """Signed improvement on the constraint axis: positive means the constraint reduced it."""
    return float(unconstrained - constrained)


def js_ceiling() -> float:
    return float(np.log(2.0))
