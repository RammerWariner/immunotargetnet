"""Recovery metrics.

Ref: Sec. 2.7.2 (NDCG at several cut-offs, mean reciprocal rank, top-fifty recall) and
the primary success criterion of Sec. 2.6 (NDCG@20 >= 0.50).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

PRIMARY_CUTOFF: Final[int] = 20
TOP_K_CUTOFFS: Final[tuple[int, ...]] = (10, 20, 30, 50, 100)
RECALL_CUTOFF: Final[int] = 50


@dataclass(frozen=True)
class RecoveryCurve:
    cutoffs: np.ndarray
    recovered: np.ndarray

    def at(self, cutoff: int) -> float:
        position = int(np.searchsorted(self.cutoffs, cutoff))
        position = min(position, self.recovered.size - 1)
        return float(self.recovered[position])


def ranking_order(scores: np.ndarray) -> np.ndarray:
    """Descending score order with ties broken by index, so the ranking is deterministic."""
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")


def discounted_gain(labels: np.ndarray, cutoff: int) -> float:
    """Cumulated gain with the standard log2 discount of Sec. 2.7.2."""
    selected = labels[:cutoff]
    if selected.size == 0:
        return 0.0
    positions = np.arange(1, selected.size + 1, dtype=np.float64)
    discounted = selected / np.log2(positions + 1.0)
    return float(discounted.sum())


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = ranking_order(scores)
    ranked = labels[order]
    ideal = labels[ranking_order(labels)]
    best = discounted_gain(ideal, k)
    if best <= 0.0:
        return 0.0
    return discounted_gain(ranked, k) / best


def mean_reciprocal_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    """Reciprocal rank of the first relevant item, averaged over the pool."""
    order = ranking_order(scores)
    ranked = labels[order]
    relevant = np.flatnonzero(ranked > 0)
    if relevant.size == 0:
        return 0.0
    return 1.0 / float(relevant[0] + 1)


def reciprocal_rank_per_query(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Per-gene reciprocal rank, the unit the paired tests of Sec. 2.7.3 operate on."""
    order = ranking_order(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size) + 1
    inverse = np.where(labels > 0, 1.0 / ranks, 0.0)
    return np.asarray(inverse, dtype=np.float64)


def recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int = RECALL_CUTOFF) -> float:
    order = ranking_order(scores)[:k]
    positives = float((labels > 0).sum())
    if positives == 0.0:
        return 0.0
    return float((labels[order] > 0).sum()) / positives


def cumulative_recovery(labels: np.ndarray, scores: np.ndarray, cutoffs: np.ndarray) -> RecoveryCurve:
    """Cumulative fraction of the held-out target set recovered by each cut-off."""
    order = ranking_order(scores)
    ranked = (labels[order] > 0).astype(np.float64)
    total = ranked.sum()
    cumulative = np.cumsum(ranked)
    values = np.array(
        [0.0 if total == 0 else cumulative[min(int(k), cumulative.size) - 1] / total for k in cutoffs],
        dtype=np.float64,
    )
    return RecoveryCurve(cutoffs=np.asarray(cutoffs, dtype=np.float64), recovered=values)


def primary_recovery(labels: np.ndarray, scores: np.ndarray) -> float:
    return ndcg_at_k(labels, scores, PRIMARY_CUTOFF)


def cut_off_grid() -> np.ndarray:
    return np.arange(10, 101, 10, dtype=np.float64)
