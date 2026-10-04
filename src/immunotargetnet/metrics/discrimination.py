"""Readiness metrics.

Ref: Sec. 2.7.2 (area under the precision-recall curve, area under the ROC curve and
expected calibration error against the platform tractability annotations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

ECE_BINS: Final[int] = 10


@dataclass(frozen=True)
class CalibrationCurve:
    bin_lower: np.ndarray
    mean_confidence: np.ndarray
    mean_accuracy: np.ndarray
    counts: np.ndarray

    def expected_error(self) -> float:
        weights = self.counts / max(float(self.counts.sum()), 1.0)
        gaps = np.abs(self.mean_confidence - self.mean_accuracy)
        return float((weights * gaps).sum())


def area_under_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC, tie-aware through the average rank."""
    positive = labels > 0
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    sorted_scores = scores[order]
    index = 0
    while index < sorted_scores.size:
        stop = index + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[index]:
            stop += 1
        if stop - index > 1:
            ranks[order[index:stop]] = ranks[order[index:stop]].mean()
        index = stop
    rank_sum = float(ranks[positive].sum())
    statistic = rank_sum - n_positive * (n_positive + 1) / 2.0
    return statistic / (n_positive * n_negative)


def area_under_precision_recall(labels: np.ndarray, scores: np.ndarray) -> float:
    """Average precision with ties handled as one step of the precision-recall curve.

    A run of equal scores is one operating point, so a constant predictor scores its own
    prevalence rather than one.
    """
    positive = labels > 0
    n_positive = int(positive.sum())
    if n_positive == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    ranked = positive[order]
    sorted_scores = np.asarray(scores, dtype=np.float64)[order]
    cumulative = np.cumsum(ranked).astype(np.float64)
    total = 0.0
    previous_recall = 0.0
    index = 0
    while index < ranked.size:
        stop = index + 1
        while stop < ranked.size and sorted_scores[stop] == sorted_scores[index]:
            stop += 1
        last = stop - 1
        precision = cumulative[last] / float(last + 1)
        recall = cumulative[last] / float(n_positive)
        total += precision * (recall - previous_recall)
        previous_recall = recall
        index = stop
    return float(total)


def prevalence_baseline(labels: np.ndarray) -> float:
    """AUPRC of a constant predictor equals the positive rate."""
    return float((labels > 0).mean())


def calibration_curve(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = ECE_BINS
) -> CalibrationCurve:
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1)
    confidence = np.zeros(bins, dtype=np.float64)
    accuracy = np.zeros(bins, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.float64)
    for bin_index in range(bins):
        selected = assignment == bin_index
        counts[bin_index] = float(selected.sum())
        if counts[bin_index] == 0:
            continue
        confidence[bin_index] = float(probabilities[selected].mean())
        accuracy[bin_index] = float((labels[selected] > 0).mean())
    return CalibrationCurve(
        bin_lower=edges[:-1],
        mean_confidence=confidence,
        mean_accuracy=accuracy,
        counts=counts,
    )


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = ECE_BINS) -> float:
    curve = calibration_curve(probabilities, labels, bins=bins)
    return curve.expected_error()


def tractability_filter_ranking(annotation_flag: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Rule-based tractability filter applied to a method's own ranked output.

    Ref: Table 1 footnote - the two rows carrying the dagger apply the same filter, so the
    filter is a reordering rather than a rescoring.
    """
    order = np.argsort(-scores, kind="stable")
    filtered = np.concatenate([order[annotation_flag[order]], order[~annotation_flag[order]]])
    restored = np.empty_like(filtered)
    restored[filtered] = np.arange(filtered.size)
    return np.asarray(-restored, dtype=np.float64)
