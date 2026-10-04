"""The manuscript's own arithmetic, recomputed and pinned.

The failing checks are deliberate: they record statements inside the article that cannot
both hold. Pinning them means a later revision that changes either side fails loudly.
"""

from __future__ import annotations

import pytest

from immunotargetnet.verification.checks import FAIL, NOT_RUN, PASS
from immunotargetnet.verification.manuscript import (
    check_cellstate_withholding_value,
    check_cohort_means_match_loco,
    check_constraint_axis_gaps,
    check_context_token_duplication,
    check_figure1_baselines,
    check_gain_claims,
    check_interaction_ratios,
    check_mask_index_space,
    check_mask_logit_notation,
    check_programme_count,
    check_resource_count,
    check_shortlist_spearman,
    check_table6_column_reference,
    check_table6_ratio,
    check_three_layer_counterpart,
    run_manuscript_checks,
)

EXPECTED_FAILURES = {
    "manuscript-cellstate-withholding-present",
    "manuscript-mask-logit-notation",
    "manuscript-mask-index-space",
    "manuscript-context-token-duplication",
    "manuscript-table6-column-reference",
    "manuscript-figure1-baseline-caption",
    "manuscript-abstract-resource-count",
}


def test_every_manuscript_check_returns_a_known_status() -> None:
    results = run_manuscript_checks()
    assert len(results) >= 40
    assert all(result.status in {PASS, FAIL, NOT_RUN} for result in results)
    assert all(result.category == "manuscript" for result in results)
    assert all(result.paper_anchor for result in results)


def test_the_failing_family_is_exactly_the_recorded_discrepancies() -> None:
    failing = {result.name for result in run_manuscript_checks() if result.status == FAIL}
    assert failing == EXPECTED_FAILURES


def test_supplementary_only_values_are_not_run() -> None:
    outstanding = {result.name for result in run_manuscript_checks() if result.status == NOT_RUN}
    assert "manuscript-degree-control-values" in outstanding
    assert "manuscript-calibration-triple" in outstanding
    assert len(outstanding) == 6


def test_interaction_ratios_reproduce_the_paper_estimator() -> None:
    result = check_interaction_ratios()
    assert result.status == PASS
    computed = result.evidence["computed"]
    assert computed["graph_x_bottleneck"]["ndcg20"] == pytest.approx(1.33, abs=0.01)
    assert computed["bottleneck_x_perturbation"]["kl"] == pytest.approx(1.09, abs=0.01)


def test_gain_claims_close_on_the_tables_own_numbers() -> None:
    result = check_gain_claims()
    assert result.status == PASS
    assert result.evidence["stated"]["ndcg_gap_strongest"] == pytest.approx(8.8)


def test_constraint_axis_gaps_clear_their_thresholds() -> None:
    result = check_constraint_axis_gaps()
    assert result.status == PASS
    assert result.evidence["kl_drop"] == pytest.approx(0.521, abs=1e-3)
    assert min(result.evidence["parity_multiples"].values()) > 10.0


def test_cohort_column_means_reproduce_the_leave_one_cohort_out_row() -> None:
    result = check_cohort_means_match_loco()
    assert result.status == PASS
    assert result.evidence["gaps"]["immunotargetnet"] < 1e-3


def test_shortlist_spearman_is_point_six_six() -> None:
    result = check_shortlist_spearman()
    assert result.status == PASS
    assert result.evidence["computed"] == pytest.approx(0.66, abs=0.005)


def test_training_time_ratio_is_one_point_eight() -> None:
    result = check_table6_ratio()
    assert result.status == PASS
    assert result.evidence["computed"] == pytest.approx(1.8, abs=0.05)


def test_three_layer_counterpart_closes() -> None:
    assert check_three_layer_counterpart().status == PASS


def test_programme_count_is_twelve() -> None:
    result = check_programme_count()
    assert result.status == PASS
    assert result.evidence["named"] == 12


def test_cellstate_withholding_value_is_absent_from_the_tables() -> None:
    result = check_cellstate_withholding_value()
    assert result.status == FAIL
    assert result.evidence["appears_in_tables"] is False
    assert result.evidence["reported"] == pytest.approx(0.512)


def test_mask_notation_checks_fail_on_purpose() -> None:
    assert check_mask_logit_notation().status == FAIL
    assert check_mask_index_space().status == FAIL
    assert check_context_token_duplication().status == FAIL
    assert check_table6_column_reference().status == FAIL
    assert check_figure1_baselines().status == FAIL
    assert check_resource_count().status == FAIL


def test_the_discrepancies_are_distinct_records() -> None:
    results = {result.name: result for result in run_manuscript_checks()}
    assert len(results) == len(run_manuscript_checks())
    for name in EXPECTED_FAILURES:
        assert results[name].detail
