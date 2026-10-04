"""Metrics: each is checked against a reference written from the definition."""

from __future__ import annotations

import math

import numpy as np
import pytest

from immunotargetnet.metrics.attribution import DELETION_GRID, attribution_faithfulness, deletion_curve
from immunotargetnet.metrics.discrimination import (
    area_under_precision_recall,
    area_under_roc,
    calibration_curve,
    expected_calibration_error,
    prevalence_baseline,
    tractability_filter_ranking,
)
from immunotargetnet.metrics.divergence import (
    jensen_shannon_divergence,
    kl_divergence,
    mean_jensen_shannon_divergence,
    mean_kl_divergence,
)
from immunotargetnet.metrics.ranking import (
    PRIMARY_CUTOFF,
    cumulative_recovery,
    discounted_gain,
    mean_reciprocal_rank,
    ndcg_at_k,
    ranking_order,
    recall_at_k,
    reciprocal_rank_per_query,
)
from immunotargetnet.metrics.stability import kendall_tau_b, pearson_of_ranks, rank_positions


def test_ranking_order_is_descending_and_stable() -> None:
    scores = np.array([0.5, 0.9, 0.9, 0.1])
    assert ranking_order(scores).tolist() == [1, 2, 0, 3]


def test_ndcg_matches_a_direct_computation() -> None:
    labels = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    order = ranking_order(scores)
    ideal = ranking_order(labels)
    expected = discounted_gain(labels[order], 3) / discounted_gain(labels[ideal], 3)
    assert ndcg_at_k(labels, scores, 3) == pytest.approx(expected)


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert ndcg_at_k(labels, scores, 4) == pytest.approx(1.0)


def test_ndcg_is_zero_without_positives() -> None:
    assert ndcg_at_k(np.zeros(5), np.arange(5.0), 3) == 0.0


def test_mean_reciprocal_rank_uses_the_first_hit() -> None:
    labels = np.array([0.0, 1.0, 0.0, 1.0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert mean_reciprocal_rank(labels, scores) == pytest.approx(0.5)


def test_reciprocal_rank_per_query_is_sparse() -> None:
    labels = np.array([0.0, 1.0, 0.0])
    scores = np.array([3.0, 2.0, 1.0])
    values = reciprocal_rank_per_query(labels, scores)
    assert values[1] == pytest.approx(0.5)
    assert values[[0, 2]].tolist() == [0.0, 0.0]


def test_recall_at_k_counts_recovered_positives() -> None:
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert recall_at_k(labels, scores, 2) == pytest.approx(0.5)


def test_cumulative_recovery_is_monotone() -> None:
    labels = np.array([1.0, 0.0, 1.0, 1.0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    curve = cumulative_recovery(labels, scores, np.array([1.0, 2.0, 3.0, 4.0]))
    assert curve.recovered[0] <= curve.recovered[1] <= curve.recovered[2] <= curve.recovered[3]
    assert curve.recovered[-1] == pytest.approx(1.0)
    assert curve.at(2) == pytest.approx(curve.recovered[1])


def test_auroc_matches_a_pairwise_count() -> None:
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert area_under_roc(labels, scores) == pytest.approx(1.0)


def test_auroc_is_half_for_a_constant_predictor() -> None:
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    assert area_under_roc(labels, np.ones(4)) == pytest.approx(0.5)


def test_auprc_equals_prevalence_for_a_constant_predictor() -> None:
    labels = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    scores = np.full(5, 0.3)
    assert area_under_precision_recall(labels, scores) == pytest.approx(0.4)
    assert prevalence_baseline(labels) == pytest.approx(0.4)


def test_expected_calibration_error_is_zero_for_a_perfect_calibration() -> None:
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(probabilities, labels) < 1e-9


def test_calibration_curve_weights_bins_by_counts() -> None:
    probabilities = np.array([0.05, 0.15, 0.95])
    labels = np.array([0.0, 1.0, 1.0])
    curve = calibration_curve(probabilities, labels, bins=5)
    assert curve.counts.sum() == 3
    assert 0.0 <= curve.expected_error() <= 1.0


def test_tractability_filter_only_reorders() -> None:
    flag = np.array([False, True, False, True])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    filtered = tractability_filter_ranking(flag, scores)
    assert ranking_order(filtered).tolist() == [1, 3, 0, 2]
    assert len(set(np.round(filtered, 9).tolist())) == 4


def test_kl_and_js_are_zero_for_identical_distributions() -> None:
    distribution = np.array([[0.5, 0.3, 0.2]])
    assert float(kl_divergence(distribution, distribution)[0]) < 1e-9
    assert float(jensen_shannon_divergence(distribution, distribution)[0]) < 1e-9


def test_js_is_bounded_by_log_two() -> None:
    left = np.array([[1.0, 0.0]])
    right = np.array([[0.0, 1.0]])
    assert float(jensen_shannon_divergence(left, right)[0]) <= math.log(2.0) + 1e-12


def test_kl_is_asymmetric() -> None:
    left = np.array([[0.9, 0.1]])
    right = np.array([[0.5, 0.5]])
    assert float(kl_divergence(left, right)[0]) != pytest.approx(float(kl_divergence(right, left)[0]))


def test_mean_divergence_averages_over_rows() -> None:
    predicted = np.array([[0.9, 0.1], [0.2, 0.8]])
    target = np.array([[1.0, 0.0], [1.0, 0.0]])
    per_row = kl_divergence(predicted, target)
    assert mean_kl_divergence(predicted, target) == pytest.approx(float(per_row.mean()))
    assert mean_jensen_shannon_divergence(predicted, target) > 0.0


def test_deletion_curve_separates_high_and_low_attribution() -> None:
    attribution = np.array([0.7, 0.2, 0.07, 0.03])
    weight = {0: 1.0, 1: 0.4, 2: 0.1, 3: 0.05}

    def score_from_mask(mask: np.ndarray) -> np.ndarray:
        return np.array([sum(weight[index] for index in np.flatnonzero(mask))])

    curve = deletion_curve(attribution, score_from_mask)
    assert curve.area_between() < 0.0
    assert len(curve.fractions) == len(DELETION_GRID)


def test_attribution_faithfulness_needs_two_axes() -> None:
    try:
        attribution_faithfulness(np.array([1.0]), lambda mask: np.array([0.0]))
    except ValueError:
        return
    raise AssertionError("a single axis must be rejected")


def test_kendall_tau_is_one_for_identical_rankings() -> None:
    values = np.array([3.0, 1.0, 2.0, 0.5])
    assert kendall_tau_b(values, values) == pytest.approx(1.0)


def test_kendall_tau_is_minus_one_for_reversed_rankings() -> None:
    left = np.array([1.0, 2.0, 3.0, 4.0])
    right = np.array([4.0, 3.0, 2.0, 1.0])
    assert kendall_tau_b(left, right) == pytest.approx(-1.0)


def test_kendall_tau_handles_ties() -> None:
    left = np.array([1.0, 1.0, 2.0, 3.0])
    right = np.array([1.0, 2.0, 2.0, 3.0])
    value = kendall_tau_b(left, right)
    assert -1.0 <= value <= 1.0


def test_kendall_tau_rejects_mismatched_lengths() -> None:
    try:
        kendall_tau_b(np.zeros(3), np.zeros(4))
    except ValueError:
        return
    raise AssertionError("mismatched rankings must be rejected")


def test_rank_positions_start_at_zero() -> None:
    positions = rank_positions(np.array([0.2, 0.9, 0.5]))
    assert positions.tolist() == [2.0, 0.0, 1.0]


def test_pearson_of_ranks_is_the_spearman_coefficient() -> None:
    left = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    right = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert pearson_of_ranks(left, right) == pytest.approx(-1.0)


def test_primary_cutoff_is_twenty() -> None:
    assert PRIMARY_CUTOFF == 20
