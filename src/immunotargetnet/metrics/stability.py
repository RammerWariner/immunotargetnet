"""Rank-stability metrics.

Ref: Sec. 2.7.2 (Kendall tau under cohort resampling, pathway holdout and modality dropout,
with the fifth percentile reported beside the mean).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

FIFTH_PERCENTILE: Final[float] = 5.0


@dataclass(frozen=True)
class StabilitySummary:
    mean_tau: float
    fifth_percentile: float
    n_resamples: int
    taus: np.ndarray


def rank_positions(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def kendall_tau_b(left_scores: np.ndarray, right_scores: np.ndarray) -> float:
    """Tie-corrected Kendall tau between two rankings of the same items."""
    left = np.asarray(left_scores, dtype=np.float64)
    right = np.asarray(right_scores, dtype=np.float64)
    if left.size != right.size:
        raise ValueError("rankings must cover the same items")
    if left.size < 2:
        return 1.0
    concordant = 0.0
    discordant = 0.0
    tied_left = 0.0
    tied_right = 0.0
    for i in range(left.size - 1):
        dl = left[i + 1 :] - left[i]
        dr = right[i + 1 :] - right[i]
        sign = np.sign(dl) * np.sign(dr)
        concordant += float((sign > 0).sum())
        discordant += float((sign < 0).sum())
        tied_left += float(((dl == 0) & (dr != 0)).sum())
        tied_right += float(((dr == 0) & (dl != 0)).sum())
    denominator = np.sqrt((concordant + discordant + tied_left) * (concordant + discordant + tied_right))
    if denominator == 0.0:
        return 1.0
    return float((concordant - discordant) / denominator)


def stability_over_resamples(
    reference_scores: np.ndarray,
    resampled_scores: list[np.ndarray],
    *,
    percentile: float = FIFTH_PERCENTILE,
) -> StabilitySummary:
    taus = np.array(
        [kendall_tau_b(reference_scores, candidate) for candidate in resampled_scores],
        dtype=np.float64,
    )
    if taus.size == 0:
        return StabilitySummary(
            mean_tau=float("nan"), fifth_percentile=float("nan"), n_resamples=0, taus=taus
        )
    return StabilitySummary(
        mean_tau=float(taus.mean()),
        fifth_percentile=float(np.percentile(taus, percentile)),
        n_resamples=int(taus.size),
        taus=taus,
    )


def pearson_of_ranks(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation, used for the relevance-readiness co-variation of Sec. 3.6."""
    left_ranks = rank_positions(np.asarray(left, dtype=np.float64))
    right_ranks = rank_positions(np.asarray(right, dtype=np.float64))
    left_centred = left_ranks - left_ranks.mean()
    right_centred = right_ranks - right_ranks.mean()
    denominator = float(np.sqrt((left_centred**2).sum() * (right_centred**2).sum()))
    if denominator == 0.0:
        return 0.0
    return float((left_centred * right_centred).sum() / denominator)
