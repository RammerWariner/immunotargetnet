"""Protocols: splits, perturbation plans, baselines and the metric assembly."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from immunotargetnet.evidence.panel import PanelBundle
from immunotargetnet.evidence.schema import MODALITY_ORDER
from immunotargetnet.metrics.ranking import ndcg_at_k
from immunotargetnet.protocols import baselines as baseline_module, perturbation as perturbation_module
from immunotargetnet.protocols.evaluation import build_row, layer_mask_from_selection, layer_selection
from immunotargetnet.protocols.splits import (
    CohortFold,
    cohort_axis,
    label_split,
    leave_one_cohort_out,
    verify_no_sample_leakage,
)
from immunotargetnet.protocols.study import (
    StudyConfig,
    ablation_specs,
    interaction_ratio,
    run_study,
    spec_by_key,
)


def test_leave_one_cohort_out_holds_out_one_each() -> None:
    folds = leave_one_cohort_out(("A", "B", "C"))
    assert [fold.held_out for fold in folds] == ["A", "B", "C"]
    assert all(fold.held_out not in fold.training_cohorts for fold in folds)
    assert all(len(fold.training_cohorts) == 2 for fold in folds)


def test_cohort_axis_returns_positions() -> None:
    folds = leave_one_cohort_out(("A", "B", "C"))
    held, training = cohort_axis(folds[1], ("A", "B", "C"))
    assert held.tolist() == [1]
    assert training.tolist() == [0, 2]


def test_label_split_is_disjoint_and_complete() -> None:
    protected = np.zeros(200, dtype=bool)
    protected[::10] = True
    split = label_split(200, seed=3, protected=protected)
    covered = np.concatenate([split.training, split.validation, split.held_out])
    assert np.unique(covered).size == 200
    overlap = len(np.intersect1d(split.training, split.held_out))
    assert overlap == 0
    assert protected[split.held_out].all()


def test_label_split_rejects_shared_units() -> None:
    split = label_split(50, seed=1)
    split.assert_disjoint()
    broken = type(split)(training=split.training, validation=split.training, held_out=split.held_out)
    try:
        broken.assert_disjoint()
    except AssertionError:
        return
    raise AssertionError("a shared unit must be detected")


def test_sample_leakage_detection() -> None:
    fold = leave_one_cohort_out(("A", "B"))[0]
    verify_no_sample_leakage(fold, ["A", "B", "A", "B"])
    malformed = CohortFold(index=0, held_out="A", training_cohorts=("A", "B"))
    try:
        verify_no_sample_leakage(malformed, ["A", "B"])
    except AssertionError:
        return
    raise AssertionError("the held-out cohort cannot also be a training cohort")


def test_modality_dropout_plan_keeps_the_named_layers(bundle: PanelBundle) -> None:
    panel = bundle.panel
    plan = perturbation_module.modality_dropout_plan(
        panel.availability, layers_available=2, context_index=len(MODALITY_ORDER) - 1, seed=1
    )
    assert plan.layer_mask is not None
    assert plan.layer_mask[:, 2].sum() == 0
    assert plan.layer_mask[:, 0].sum() == panel.availability[:, 0].sum()
    assert np.array_equal(
        plan.layer_mask[:, len(MODALITY_ORDER) - 1],
        panel.availability[:, len(MODALITY_ORDER) - 1],
    )


def test_modality_dropout_plan_rejects_zero_layers(bundle: PanelBundle) -> None:
    try:
        perturbation_module.modality_dropout_plan(
            bundle.panel.availability, layers_available=0, context_index=3, seed=1
        )
    except ValueError:
        return
    raise AssertionError("a zero-layer plan must be rejected")


def test_random_subset_control_matches_the_withheld_count() -> None:
    plan = perturbation_module.random_subset_control(500, 40, seed=3)
    assert plan.n_genes_withheld == 40
    assert np.unique(plan.withheld_genes).size == 40


def test_cohort_resample_blocks_keeps_genes_together() -> None:
    genes = np.repeat(np.arange(10), 3)
    cohort_index = np.tile(np.arange(3), 10)
    draws = perturbation_module.cohort_resample_blocks(genes, cohort_index, seed=2, n_resamples=4)
    assert len(draws) == 4
    rows, cohorts = draws[0]
    assert rows.size == genes.size
    assert cohorts.size == genes.size


def test_layer_mask_from_selection_blanks_the_rest(bundle: PanelBundle) -> None:
    layers = bundle.panel.availability.shape[1]
    mask = layer_mask_from_selection(bundle.panel.availability, layer_selection(layers, (0, 3)))
    assert mask[:, 1].sum() == 0
    assert mask[:, 2].sum() == 0
    assert mask.any(axis=1).all()
    try:
        layer_mask_from_selection(bundle.panel.availability, np.array([0, 3]))
    except ValueError:
        return
    raise AssertionError("an index array must be rejected in place of a selection")


def test_every_baseline_produces_a_score_per_gene(bundle: PanelBundle) -> None:
    scores = baseline_module.baseline_scores(bundle.panel)
    for key in baseline_module.BASELINE_KEYS[:-1]:
        values = scores.get(key)
        assert values.shape == (bundle.panel.n_genes,)
        assert np.isfinite(values).all()


def test_baselines_are_distinct_from_each_other(bundle: PanelBundle) -> None:
    scores = baseline_module.baseline_scores(bundle.panel)
    reference = scores.get("differential_expression")
    for key in baseline_module.BASELINE_KEYS[1:-1]:
        assert not np.allclose(reference, scores.get(key))


def test_build_row_reports_the_expected_columns(bundle: PanelBundle) -> None:
    panel = bundle.panel
    row = build_row(
        method="check",
        panel=panel,
        scores=panel.development_labels,
        programmes=panel.programme_targets,
        readiness=np.clip(panel.development_labels / panel.development_labels.max(), 0.0, 1.0),
    )
    table = row.as_table()
    assert set(table) == {"NDCG@20", "MRR", "AUROC", "Top-50", "AUPRC", "KL", "JS", "AUCdel", "tau", "ECE"}
    assert 0.0 <= table["NDCG@20"] <= 1.0


def test_interaction_ratio_is_combined_over_individual() -> None:
    ratio = interaction_ratio(combined=0.052, first=-0.003, second=0.042, direction="higher")
    assert ratio == pytest.approx(0.052 / 0.039, rel=1e-3)


def test_interaction_ratio_for_a_lower_is_better_metric() -> None:
    ratio = interaction_ratio(combined=0.663, first=0.521, second=0.097, direction="lower")
    assert ratio == pytest.approx(0.663 / 0.618, rel=1e-3)


def test_interaction_ratio_handles_a_zero_denominator() -> None:
    assert np.isnan(interaction_ratio(0.1, 0.0, 0.0, direction="higher"))


def test_ablation_specs_cover_the_reported_rows() -> None:
    keys = {spec.key for spec in ablation_specs()}
    required = {
        "full",
        "minus_bottleneck",
        "minus_graph_tower",
        "minus_druggability_head",
        "minus_perturbation_objective",
        "minus_modality_dropout",
        "minus_niche_layer",
        "minus_cellstate_layer",
        "minus_graph_minus_bottleneck",
        "minus_bottleneck_minus_perturbation",
    }
    assert required <= keys
    assert spec_by_key("minus_druggability_head").readiness_from_filter


def test_spec_by_key_rejects_an_unknown_configuration() -> None:
    try:
        spec_by_key("not_a_configuration")
    except KeyError:
        return
    raise AssertionError("an unknown configuration must be rejected")


def test_small_study_run_produces_every_table(bundle: PanelBundle) -> None:
    config = StudyConfig(epochs=1, stability_draws=1, bootstrap_resamples=50, model_dim=8)
    outcome = run_study(bundle.panel, config=config, seed=7)
    assert len(outcome.table1) == len(baseline_module.BASELINE_KEYS) + 1
    assert len(outcome.shortlist) == 12
    assert outcome.interaction_ratios
    for row in outcome.table4_ablations:
        assert 0.0 <= float(row["NDCG@20"]) <= 1.0
    for row in outcome.table3_settings:
        assert "IT-Net" in row
    assert outcome.cost["framework_params"] > 0.0


def test_shortlist_relevance_is_monotone_in_a_study_run(bundle: PanelBundle) -> None:
    config = StudyConfig(epochs=1, stability_draws=1, bootstrap_resamples=50, model_dim=8)
    outcome = run_study(bundle.panel, config=config, seed=11)
    values = [float(entry["relevance"]) for entry in outcome.shortlist]
    assert all(earlier >= later for earlier, later in itertools.pairwise(values))


def test_study_scores_reproduce_under_a_repeat(bundle: PanelBundle) -> None:
    config = StudyConfig(epochs=1, stability_draws=1, bootstrap_resamples=50, model_dim=8)
    first = run_study(bundle.panel, config=config, seed=13)
    second = run_study(bundle.panel, config=config, seed=13)
    assert first.cost["framework_ndcg20"] == second.cost["framework_ndcg20"]
    assert [entry["gene"] for entry in first.shortlist] == [entry["gene"] for entry in second.shortlist]


def test_recovery_metrics_are_reported_on_the_target_labels(bundle: PanelBundle) -> None:
    panel = bundle.panel
    labels = panel.target_labels.astype(np.float64)
    perfect = ndcg_at_k(labels, labels.astype(np.float64), 20)
    inverted = ndcg_at_k(labels, -labels.astype(np.float64), 20)
    assert perfect == pytest.approx(1.0)
    assert inverted < perfect
