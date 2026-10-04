"""Statistics: Holm step-down, blocked bootstrap and the equivalence test."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from immunotargetnet.stats.bootstrap import (
    block_resample,
    blocked_bootstrap_difference,
    bootstrap_metric,
    gene_blocks,
    percentile_interval,
)
from immunotargetnet.stats.equivalence import (
    MINIMUM_MEANINGFUL_KL_DROP,
    PARITY_MARGIN,
    exceeds_minimum_meaningful,
    margin_multiple,
    paired_difference_interval,
    primary_criterion_met,
    two_one_sided_test,
)
from immunotargetnet.stats.holm import holm_step_down


def test_holm_adjusted_values_are_monotone_in_p_order() -> None:
    p_values = np.array([0.001, 0.02, 0.03, 0.2])
    result = holm_step_down(p_values, alpha=0.05)
    order = np.argsort(p_values, kind="stable")
    adjusted = result.adjusted[order]
    assert all(earlier <= later for earlier, later in itertools.pairwise(adjusted))


def test_holm_leaves_the_smallest_p_value_unchanged_up_to_scaling() -> None:
    p_values = np.array([0.01, 0.5, 0.6])
    result = holm_step_down(p_values, alpha=0.05)
    assert result.adjusted[0] == pytest.approx(0.03)


def test_holm_never_shrinks_a_p_value() -> None:
    p_values = np.array([0.04, 0.05, 0.2])
    result = holm_step_down(p_values, alpha=0.05)
    assert (result.adjusted >= p_values - 1e-12).all()


def test_holm_handles_an_empty_family() -> None:
    result = holm_step_down(np.zeros(0), alpha=0.05)
    assert result.family_size == 0


def test_holm_stops_at_the_first_insignificant_step() -> None:
    p_values = np.array([0.001, 0.008, 0.039, 0.041])
    result = holm_step_down(p_values, alpha=0.05)
    # scaled in ascending p order: 0.004, 0.024, 0.078, 0.078, so the step-down rejects two
    # and stops at the third rather than rejecting everything below alpha.
    assert int(result.rejected.sum()) == 2
    assert result.adjusted[0] == pytest.approx(0.004)
    assert result.adjusted[3] == pytest.approx(0.078)


def test_holm_rejects_a_running_minimum() -> None:
    p_values = np.array([0.001, 0.008, 0.039, 0.041])
    result = holm_step_down(p_values, alpha=0.05)
    bonferroni = p_values * p_values.size
    assert not np.allclose(result.adjusted, bonferroni)


def test_percentile_interval_brackets_its_point() -> None:
    samples = np.linspace(-1.0, 1.0, 1001)
    interval = percentile_interval(samples, level=0.95, point=0.0)
    assert interval.lower < 0.0 < interval.upper
    assert interval.contains(0.0)


def test_bootstrap_metric_reproduces_the_point_estimate() -> None:
    units = np.linspace(0.0, 1.0, 64)
    interval = bootstrap_metric(units, lambda block: float(block.mean()), resamples=200, seed=3)
    assert interval.point == pytest.approx(0.5, abs=0.02)


def test_blocked_difference_brackets_zero_for_identical_arms() -> None:
    units = np.linspace(0.0, 1.0, 64)
    interval = blocked_bootstrap_difference(
        units, units.copy(), lambda block: float(block.mean()), resamples=200, seed=5
    )
    assert interval.lower <= 0.0 <= interval.upper


def test_blocked_difference_rejects_mismatched_lengths() -> None:
    try:
        blocked_bootstrap_difference(np.zeros(3), np.zeros(4), lambda block: float(block.mean()))
    except ValueError:
        return
    raise AssertionError("mismatched arms must be rejected")


def test_gene_blocks_group_every_cohort_of_a_gene() -> None:
    blocks = gene_blocks([("A", "c1"), ("B", "c1"), ("A", "c2")])
    assert set(blocks) == {"A", "B"}
    assert blocks["A"].tolist() == [0, 2]


def test_block_resample_keeps_whole_blocks_together() -> None:
    per_unit = np.arange(6, dtype=np.float64)
    blocks = {"A": np.array([0, 1]), "B": np.array([2, 3]), "C": np.array([4, 5])}
    drawn = block_resample(per_unit, blocks, seed=7)
    assert drawn.size % 2 == 0
    for position in range(0, drawn.size, 2):
        assert drawn[position + 1] - drawn[position] == 1.0


def test_tost_declares_parity_inside_the_band() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.53, 0.02, size=128)
    verdict = two_one_sided_test(base, base + 0.003, resamples=300, seed=11)
    assert verdict.equivalent
    assert verdict.lower >= -PARITY_MARGIN and verdict.upper <= PARITY_MARGIN


def test_tost_rejects_a_difference_outside_the_band() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(0.53, 0.02, size=128)
    verdict = two_one_sided_test(base, base + 0.05, resamples=300, seed=11)
    assert not verdict.equivalent
    assert "above" in verdict.detail or "below" in verdict.detail


def test_paired_interval_matches_its_point_estimate() -> None:
    rng = np.random.default_rng(5)
    left = rng.normal(size=64)
    right = left - 0.02
    point, lower, upper = paired_difference_interval(left, right, resamples=200, seed=13)
    assert lower <= point <= upper
    assert point == pytest.approx(0.02, abs=0.01)


def test_minimum_meaningful_difference_thresholds() -> None:
    exceeded, multiple = exceeds_minimum_meaningful(0.521, axis="immune_relevance_divergence")
    assert exceeded
    assert multiple == pytest.approx(0.521 / MINIMUM_MEANINGFUL_KL_DROP)
    try:
        exceeds_minimum_meaningful(0.1, axis="unknown_axis")
    except KeyError:
        return
    raise AssertionError("an unknown axis must be rejected")


def test_margin_multiple_is_an_absolute_ratio() -> None:
    assert margin_multiple(-0.137) == pytest.approx(13.7)


def test_primary_criterion_reports_its_margin() -> None:
    met, margin = primary_criterion_met(0.539)
    assert met
    assert margin == pytest.approx(0.039)
