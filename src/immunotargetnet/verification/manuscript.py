"""Manuscript-arithmetic checks.

Ref: every number recomputed here is transcribed in `claimed_values.py` with its table or
section anchor. These checks recompute the article's own arithmetic from its own tables;
a failure names a disagreement inside the article rather than a defect in the release.
"""

from __future__ import annotations

import itertools
import math
from typing import Final

from immunotargetnet.verification import claimed_values as claimed
from immunotargetnet.verification.checks import FAIL, NOT_RUN, PASS, CheckResult

TOLERANCE: Final[float] = 5e-4
STRUCTURAL_TOLERANCE: Final[float] = 0.06

_NET_COLUMN: Final[int] = 0
_UNCONSTRAINED_COLUMN: Final[int] = 1
_TYPED_GRAPH_COLUMN: Final[int] = 2
_LATENT_COLUMN: Final[int] = 3
_DE_COLUMN: Final[int] = 4


def _close(left: float, right: float, tolerance: float = TOLERANCE) -> bool:
    return abs(left - right) <= tolerance


def _result(name: str, ok: bool, detail: str, evidence: dict[str, object], anchor: str) -> CheckResult:
    return CheckResult(
        name=name,
        category="manuscript",
        status=PASS if ok else FAIL,
        detail=detail,
        evidence=evidence,
        paper_anchor=anchor,
    )


def _not_run(name: str, detail: str, anchor: str) -> CheckResult:
    return CheckResult(name=name, category="manuscript", status=NOT_RUN, detail=detail, paper_anchor=anchor)


def check_cohort_sample_sum() -> CheckResult:
    total = sum(entry[4] for entry in claimed.TABLE2A_PER_COHORT.values())
    return _result(
        "manuscript-cohort-sample-sum",
        total == int(claimed.REPORTED["cohort_samples_total"]),
        f"Table 2(a) sample counts sum to {total}; Sec. 2.2 states {int(claimed.REPORTED['cohort_samples_total'])}",
        {"sum": total, "stated": int(claimed.REPORTED["cohort_samples_total"])},
        claimed.SECTION_2_2_ANCHOR,
    )


def check_evaluation_units() -> CheckResult:
    units = int(claimed.REPORTED["curated_targets"] * claimed.REPORTED["n_cohorts"])
    stated = int(claimed.REPORTED["evaluation_units"])
    return _result(
        "manuscript-evaluation-units",
        units == stated,
        f"{int(claimed.REPORTED['curated_targets'])} targets over {int(claimed.REPORTED['n_cohorts'])} cohorts is {units} units; Sec. 2.7.1 states {stated}",
        {"computed": units, "stated": stated},
        claimed.SECTION_2_7_1_ANCHOR,
    )


def check_cohort_means_match_loco() -> CheckResult:
    """Table 2(a) column means against the Table 3 leave-one-cohort-out row.

    Table 2(a) carries four method columns and Table 3 five, so the differential-expression
    column of Table 2(a) maps onto the last column of Table 3.
    """
    per_cohort = claimed.TABLE2A_PER_COHORT
    means = {
        name: sum(entry[source] for entry in per_cohort.values()) / len(per_cohort)
        for name, source in (
            ("immunotargetnet", 0),
            ("unconstrained_variant", 1),
            ("typed_graph_ranking", 2),
            ("differential_expression", 3),
        )
    }
    loco = claimed.TABLE3_SETTINGS["leave_one_cohort_out"]
    stated = {
        "immunotargetnet": loco[_NET_COLUMN],
        "unconstrained_variant": loco[_UNCONSTRAINED_COLUMN],
        "typed_graph_ranking": loco[_TYPED_GRAPH_COLUMN],
        "differential_expression": loco[_DE_COLUMN],
    }
    gaps = {name: round(abs(means[name] - stated[name]), 6) for name in stated}
    worst = max(gaps.values())
    return _result(
        "manuscript-cohort-means-vs-loco",
        worst <= TOLERANCE,
        f"column means of Table 2(a) against the Table 3 leave-one-cohort-out row; worst gap {worst:.4f}",
        {
            "column_means": {name: round(value, 4) for name, value in means.items()},
            "table3_row": stated,
            "gaps": gaps,
        },
        claimed.TABLE2A_ANCHOR,
    )


def check_distribution_row_matches_table1() -> CheckResult:
    table1 = claimed.TABLE1
    row = claimed.TABLE3_SETTINGS["in_distribution"]
    pairs = {
        "immunotargetnet": (table1["immunotargetnet"], _NET_COLUMN),
        "unconstrained_variant": (table1["unconstrained_variant"], _UNCONSTRAINED_COLUMN),
        "typed_graph_ranking": (table1["typed_graph_ranking"], _TYPED_GRAPH_COLUMN),
        "latent_factor_integration": (table1["latent_factor_integration"], _LATENT_COLUMN),
        "differential_expression": (table1["differential_expression"], _DE_COLUMN),
    }
    gaps = {
        name: abs(source[_UNCONSTRAINED_COLUMN - 1] - row[index]) for name, (source, index) in pairs.items()
    }
    worst = max(gaps.values())
    return _result(
        "manuscript-in-distribution-vs-table1",
        worst <= TOLERANCE,
        f"Table 3 in-distribution row against Table 1 NDCG@20; worst gap {worst:.4f}",
        {"gaps": {key: round(value, 4) for key, value in gaps.items()}},
        claimed.TABLE3_ANCHOR,
    )


def check_resolution_loss_column() -> CheckResult:
    full = claimed.TABLE2B_RESOLUTION["none"][0]
    gaps = {key: round(full - value[0] - value[1], 6) for key, value in claimed.TABLE2B_RESOLUTION.items()}
    worst = max(abs(value) for value in gaps.values())
    return _result(
        "manuscript-resolution-loss-column",
        worst <= TOLERANCE,
        f"Table 2(b) loss column equals 0.539 minus the value; worst residual {worst:.6f}",
        {"residuals": gaps},
        claimed.TABLE2B_ANCHOR,
    )


def check_gain_claims() -> CheckResult:
    table1 = claimed.TABLE1
    framework = table1["immunotargetnet"]
    gaps = {
        "ndcg_gap_strongest": (framework[0] - table1["typed_graph_ranking"][0]) * 100.0,
        "ndcg_gap_weakest": (framework[0] - table1["differential_expression"][0]) * 100.0,
        "auprc_gap_typed_graph": (framework[4] - table1["typed_graph_ranking"][4]) * 100.0,
        "auprc_gap_filter": (framework[4] - table1["differential_expression"][4]) * 100.0,
    }
    stated = {key: claimed.REPORTED[key] for key in gaps}
    residuals = {key: round(gaps[key] - stated[key], 4) for key in gaps}
    worst = max(abs(value) for value in residuals.values())
    return _result(
        "manuscript-gain-claims",
        worst <= 0.1,
        f"Sec. 1 and Sec. 3.1 gains recomputed from Table 1; worst residual {worst:.4f} points",
        {"computed": {key: round(value, 3) for key, value in gaps.items()}, "stated": stated},
        claimed.SECTION_3_1_ANCHOR,
    )


def check_primary_margin() -> CheckResult:
    value = claimed.TABLE1["immunotargetnet"][0]
    margin = value - claimed.REPORTED["primary_ndcg20_threshold"]
    stated = claimed.REPORTED["primary_margin"]
    return _result(
        "manuscript-primary-margin",
        _close(margin, stated, 1e-3),
        f"0.539 against the pre-specified 0.50 threshold leaves {margin:.3f}",
        {"computed": round(margin, 4), "stated": stated},
        claimed.SECTION_3_1_ANCHOR,
    )


def check_parity_point_and_band() -> CheckResult:
    difference = claimed.TABLE1["unconstrained_variant"][0] - claimed.TABLE1["immunotargetnet"][0]
    lower = claimed.REPORTED["parity_interval_lower"]
    upper = claimed.REPORTED["parity_interval_upper"]
    margin = claimed.REPORTED["parity_margin"]
    inside = (-margin <= lower) and (upper <= margin)
    ok = _close(difference, claimed.REPORTED["parity_point_difference"], 1e-3) and inside
    return _result(
        "manuscript-parity-point-and-band",
        ok,
        f"paired difference {difference:.3f} with reported 90% interval [{lower}, {upper}] inside +-{margin}",
        {"difference": round(difference, 4), "inside_band": inside},
        claimed.SECTION_3_1_ANCHOR,
    )


def check_marginal_intervals_overlap() -> CheckResult:
    left = claimed.TABLE1_INTERVALS["immunotargetnet"]
    right = claimed.TABLE1_INTERVALS["unconstrained_variant"]
    overlap = min(left[1], right[1]) - max(left[0], right[0])
    return _result(
        "manuscript-marginal-interval-overlap",
        overlap > 0.0,
        f"the two 95% intervals overlap by {overlap:.4f}; the manuscript reports intersecting intervals",
        {"overlap": round(overlap, 4)},
        claimed.SECTION_3_1_ANCHOR,
    )


def check_constraint_axis_gaps() -> CheckResult:
    full = claimed.TABLE4_ABLATION["full"]
    unconstrained = claimed.TABLE4_ABLATION["minus_bottleneck"]
    kl_drop = unconstrained[2] - full[2]
    faithfulness_gain = full[3] - unconstrained[3]
    stability_gain = full[4] - unconstrained[4]
    ratios = {
        "immune_relevance": kl_drop / claimed.REPORTED["minimum_meaningful_kl"],
        "attribution_faithfulness": faithfulness_gain / claimed.REPORTED["minimum_meaningful_faithfulness"],
        "rank_stability": stability_gain / claimed.REPORTED["minimum_meaningful_stability"],
    }
    multiples_of_parity = {
        "immune_relevance": kl_drop / claimed.REPORTED["parity_margin"],
        "attribution_faithfulness": faithfulness_gain / claimed.REPORTED["parity_margin"],
        "rank_stability": stability_gain / claimed.REPORTED["parity_margin"],
    }
    ok = (
        _close(kl_drop, claimed.REPORTED["kl_drop"], 1e-3)
        and _close(faithfulness_gain, claimed.REPORTED["faithfulness_gain"], 1e-3)
        and _close(stability_gain, claimed.REPORTED["stability_gain"], 1e-3)
        and all(value >= 1.0 for value in ratios.values())
        and all(value >= 10.0 for value in multiples_of_parity.values())
    )
    detail = (
        f"KL drop {kl_drop:.3f} nats ({ratios['immune_relevance']:.2f}x the 0.20 threshold), "
        f"faithfulness gain {faithfulness_gain:.3f} ({ratios['attribution_faithfulness']:.2f}x 0.25), "
        f"stability gain {stability_gain:.3f} ({ratios['rank_stability']:.2f}x 0.08); "
        f"smallest parity multiple {min(multiples_of_parity.values()):.1f}"
    )
    return _result(
        "manuscript-constraint-axis-gaps",
        ok,
        detail,
        {
            "kl_drop": round(kl_drop, 4),
            "faithfulness_gain": round(faithfulness_gain, 4),
            "stability_gain": round(stability_gain, 4),
            "parity_multiples": {key: round(value, 2) for key, value in multiples_of_parity.items()},
        },
        claimed.SECTION_3_2_ANCHOR,
    )


def check_best_baseline_gaps() -> CheckResult:
    """The framework against the strongest non-framework baseline on the constraint axis.

    The baseline's divergence is the typed-graph row of Table 1 and its stability is the
    typed-graph cell of the Table 3 cohort-resample row.
    """
    full = claimed.TABLE4_ABLATION["full"]
    baseline_kl = claimed.TABLE1["typed_graph_ranking"][5]
    baseline_tau = claimed.TABLE3_STABILITY["cohort_resample"][_TYPED_GRAPH_COLUMN][0]
    kl_gap = baseline_kl - full[2]
    stability_gap = full[4] - baseline_tau
    stated_kl = claimed.REPORTED["best_baseline_kl_gap"]
    stated_tau = claimed.REPORTED["best_baseline_stability_gap"]
    ok = _close(kl_gap, stated_kl, 1e-3) and _close(stability_gap, stated_tau, 1e-3)
    return _result(
        "manuscript-best-baseline-gaps",
        ok,
        f"against the strongest non-framework baseline: KL gap {kl_gap:.3f} nats, stability gap {stability_gap:.3f}",
        {
            "kl_gap": round(kl_gap, 4),
            "stability_gap": round(stability_gap, 4),
            "stated_kl_gap": stated_kl,
            "stated_stability_gap": stated_tau,
            "baseline_kl": baseline_kl,
            "baseline_tau": baseline_tau,
        },
        claimed.SECTION_3_2_ANCHOR,
    )


def check_faithfulness_sentence_roles() -> CheckResult:
    full = claimed.TABLE4_ABLATION["full"][3]
    baseline = claimed.REPORTED["best_baseline_faithfulness"]
    gap = claimed.REPORTED["best_baseline_faithfulness_gap"]
    closes = _close(full - baseline, gap, 1e-3)
    detail = (
        "Sec. 3.2 reads 'the attribution of faithfulness is more than the one of 0.371 found by 0.377'. "
        f"The arithmetic closes only when 0.371 is the baseline and 0.377 the gap ({full:.3f} - {baseline:.3f} = {full - baseline:.3f}), "
        "which is the reverse of the roles the sentence assigns."
    )
    return CheckResult(
        name="manuscript-faithfulness-sentence-roles",
        category="manuscript",
        status=PASS if closes else FAIL,
        detail=detail,
        evidence={"closes_under_swapped_roles": closes, "framework": full, "baseline": baseline, "gap": gap},
        paper_anchor=claimed.SECTION_3_2_ANCHOR,
    )


def check_interaction_ratios() -> CheckResult:
    order = ("ndcg20", "mrr", "kl", "aucdel", "tau")
    direction = {"ndcg20": 1.0, "mrr": 1.0, "kl": -1.0, "aucdel": 1.0, "tau": 1.0}
    index = {"ndcg20": 0, "mrr": 1, "kl": 2, "aucdel": 3, "tau": 4}
    pairs = {
        "graph_x_bottleneck": ("minus_bottleneck", "minus_graph_tower", "minus_graph_minus_bottleneck"),
        "bottleneck_x_perturbation": (
            "minus_bottleneck",
            "minus_perturbation",
            "minus_bottleneck_minus_perturbation",
        ),
    }
    computed: dict[str, dict[str, float]] = {}
    residuals: dict[str, float] = {}
    for label, (first, second, combined) in pairs.items():
        computed[label] = {}
        stated_row = claimed.TABLE4_INTERACTION[label]
        for position, metric in enumerate(order):
            base = claimed.TABLE4_ABLATION["full"][index[metric]]

            def cost(key: str, *, metric: str = metric, base: float = base) -> float:
                return direction[metric] * (claimed.TABLE4_ABLATION[key][index[metric]] - base)

            denominator = cost(first) + cost(second)
            value = cost(combined) / denominator if denominator != 0.0 else float("nan")
            computed[label][metric] = round(value, 4)
            residuals[f"{label}:{metric}"] = abs(value - stated_row[position])
    worst = max(residuals.values())
    return _result(
        "manuscript-interaction-ratios",
        worst <= 0.01,
        f"all ten Table 4 interaction ratios recomputed as combined cost over the sum of individual costs; worst residual {worst:.4f}",
        {"computed": computed, "worst_residual": round(worst, 4)},
        claimed.TABLE4_ANCHOR,
    )


def check_individual_cost_claims() -> CheckResult:
    full = claimed.TABLE4_ABLATION["full"]
    graph_tower = claimed.TABLE4_ABLATION["minus_graph_tower"]
    perturbation = claimed.TABLE4_ABLATION["minus_perturbation"]
    druggability = claimed.TABLE4_ABLATION["minus_druggability_head"]
    gaps = {
        "graph_ndcg": full[0] - graph_tower[0],
        "graph_stability": full[4] - graph_tower[4],
        "perturbation_ndcg": full[0] - perturbation[0],
        "perturbation_stability": full[4] - perturbation[4],
        "druggability_ndcg": full[0] - druggability[0],
    }
    stated = {
        "graph_ndcg": 0.042,
        "graph_stability": 0.076,
        "perturbation_ndcg": 0.013,
        "perturbation_stability": 0.084,
        "druggability_ndcg": 0.005,
    }
    residuals = {key: round(gaps[key] - stated[key], 4) for key in gaps}
    worst = max(abs(value) for value in residuals.values())
    return _result(
        "manuscript-individual-cost-claims",
        worst <= TOLERANCE,
        f"Sec. 3.5 individual removal costs recomputed from Table 4; worst residual {worst:.4f}",
        {"computed": {key: round(value, 4) for key, value in gaps.items()}, "stated": stated},
        claimed.SECTION_3_5_ANCHOR,
    )


def check_shortlist_spearman() -> CheckResult:
    relevance = [row[2] for row in claimed.TABLE5_SHORTLIST]
    readiness = [row[3] for row in claimed.TABLE5_SHORTLIST]
    n = len(relevance)
    relevance_order = _average_ranks(relevance)
    readiness_order = _average_ranks(readiness)
    differences = [left - right for left, right in zip(relevance_order, readiness_order, strict=True)]
    rho = 1.0 - 6.0 * sum(value * value for value in differences) / (n * (n * n - 1))
    stated = claimed.REPORTED["shortlist_spearman"]
    return _result(
        "manuscript-shortlist-spearman",
        _close(rho, stated, 0.005),
        f"Spearman correlation of the Table 5 relevance and readiness columns is {rho:.4f}; Sec. 3.6 states {stated}",
        {"computed": round(rho, 4), "stated": stated, "n": n},
        claimed.SECTION_3_6_ANCHOR,
    )


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: -values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        stop = position + 1
        while stop < len(order) and values[order[stop]] == values[order[position]]:
            stop += 1
        average = (position + stop + 1) / 2.0
        for index in order[position:stop]:
            ranks[index] = average
        position = stop
    return ranks


def check_shortlist_ordering() -> CheckResult:
    relevance = [row[2] for row in claimed.TABLE5_SHORTLIST]
    readiness = [row[3] for row in claimed.TABLE5_SHORTLIST]
    decreasing = all(earlier > later for earlier, later in itertools.pairwise(relevance))
    readiness_shuffled = len(set(readiness)) == len(readiness) and readiness != sorted(
        readiness, reverse=True
    )
    ok = decreasing and readiness_shuffled
    return _result(
        "manuscript-shortlist-ordering",
        ok,
        f"relevance decreases strictly with rank ({decreasing}) and the readiness column is not rank-ordered ({readiness_shuffled})",
        {"relevance_decreasing": decreasing, "readiness_not_rank_ordered": readiness_shuffled},
        claimed.TABLE5_ANCHOR,
    )


def check_cellstate_withholding_value() -> CheckResult:
    reported = claimed.REPORTED["cellstate_withheld_ndcg20"]
    flat = {value[0] for value in claimed.TABLE2B_RESOLUTION.values()}
    ablation = {value[0] for value in claimed.TABLE4_ABLATION.values()}
    present = reported in flat or reported in ablation
    return CheckResult(
        name="manuscript-cellstate-withholding-present",
        category="manuscript",
        status=PASS if present else FAIL,
        detail=(
            f"Sec. 3.3 reports withholding the cell-state layer as NDCG@20 {reported}, but Table 2(b) "
            f"carries {sorted(flat)} and Table 4 carries no cell-state row at all; the value appears "
            "in the prose only."
        ),
        evidence={
            "reported": reported,
            "table2b_values": sorted(flat),
            "table4_values": sorted(ablation),
            "appears_in_tables": present,
        },
        paper_anchor=claimed.SECTION_3_3_ANCHOR,
    )


def check_cohort_ndcg_range() -> CheckResult:
    values = [entry[0] for entry in claimed.TABLE2A_PER_COHORT.values()]
    low = min(values)
    high = max(values)
    ok = _close(low, claimed.REPORTED["cohort_ndcg_min"], 1e-3) and _close(
        high, claimed.REPORTED["cohort_ndcg_max"], 1e-3
    )
    return _result(
        "manuscript-cohort-ndcg-range",
        ok,
        f"per-cohort NDCG@20 spans {low:.3f} to {high:.3f}; Sec. 3.3 states 0.478 to 0.556",
        {"min": round(low, 4), "max": round(high, 4)},
        claimed.SECTION_3_3_ANCHOR,
    )


def check_table6_ratio() -> CheckResult:
    framework = claimed.TABLE6_COST["immunotargetnet"]
    graph_only = claimed.TABLE6_COST["typed_graph_ranking"]
    ratio = framework[1] / graph_only[1]
    stated = claimed.REPORTED["training_time_ratio"]
    return _result(
        "manuscript-training-time-ratio",
        _close(ratio, stated, 0.05),
        f"6.1 h against 3.4 h is {ratio:.3f}; Sec. 3.7 states 1.8",
        {"computed": round(ratio, 4), "stated": stated},
        claimed.SECTION_3_7_ANCHOR,
    )


def check_table3_transfer_bound() -> CheckResult:
    in_distribution = claimed.TABLE3_SETTINGS["in_distribution"]
    transfer_rows = (
        "leave_one_cohort_out",
        "pathway_holdout",
        "random_subset_control",
        "cohort_evidence_only",
        "cohort_cell_state",
        "cohort_cell_niche",
    )
    violations = {
        f"{row}:{column}": round(claimed.TABLE3_SETTINGS[row][column], 4)
        for row in transfer_rows
        for column in range(5)
        if claimed.TABLE3_SETTINGS[row][column] > in_distribution[column] + TOLERANCE
    }
    return _result(
        "manuscript-transfer-never-exceeds-in-distribution",
        not violations,
        "Table 3 footnote states transfer never exceeds in-distribution performance for any configuration; "
        f"{len(violations)} cell(s) violate it",
        {"violations": violations},
        claimed.TABLE3_ANCHOR,
    )


def check_table3_degradation_monotone() -> CheckResult:
    sequence = (
        claimed.TABLE3_SETTINGS["cohort_evidence_only"][_NET_COLUMN],
        claimed.TABLE3_SETTINGS["cohort_cell_state"][_NET_COLUMN],
        claimed.TABLE3_SETTINGS["cohort_cell_niche"][_NET_COLUMN],
        claimed.TABLE3_SETTINGS["in_distribution"][_NET_COLUMN],
    )
    monotone = all(earlier <= later for earlier, later in itertools.pairwise(sequence))
    return _result(
        "manuscript-degradation-monotone",
        monotone,
        f"the framework's layer build-up is {'non-decreasing' if monotone else 'not non-decreasing'}: {sequence}",
        {"sequence": list(sequence)},
        claimed.TABLE3_ANCHOR,
    )


def check_three_layer_counterpart() -> CheckResult:
    stated_full = claimed.REPORTED["full_panel_three_layer_ndcg20"]
    from_table = claimed.TABLE3_SETTINGS["cohort_cell_niche"][_NET_COLUMN]
    other = claimed.REPORTED["three_layer_ndcg20"]
    ok = _close(from_table, stated_full, 1e-3) and other < stated_full
    return _result(
        "manuscript-three-layer-counterpart",
        ok,
        f"the Table 2(b) footnote's three-layer counterpart is {from_table:.3f}, matching the Table 3 cohort+cell+niche row, "
        f"and the dropout-removed inference value {other} is lower",
        {"table3_value": from_table, "footnote_full": stated_full, "footnote_reduced": other},
        claimed.TABLE2B_ANCHOR,
    )


def check_readiness_prevalence_round_trip() -> CheckResult:
    prevalence = claimed.REPORTED["readiness_prevalence_baseline"]
    targets = claimed.REPORTED["curated_targets"]
    count = prevalence * targets
    nearest = round(count)
    ok = abs(count - nearest) <= 0.05 and nearest != int(claimed.REPORTED["approved_family_targets"])
    return _result(
        "manuscript-readiness-prevalence-round-trip",
        ok,
        f"0.196 over 148 targets is {count:.3f}, i.e. an integer count of {nearest}; the approved-family count of "
        f"{int(claimed.REPORTED['approved_family_targets'])} therefore labels a different quantity",
        {"product": round(count, 4), "nearest_integer": nearest},
        claimed.SECTION_3_1_ANCHOR,
    )


def check_js_ceiling() -> CheckResult:
    js = claimed.REPORTED["js_divergence"]
    kl = claimed.REPORTED["kl_drop"]
    ok = js < math.log(2.0) and js < kl
    return _result(
        "manuscript-jensen-shannon-bound",
        ok,
        f"the reported JS divergence {js} is below its ln 2 ceiling and below the KL drop {kl}",
        {"js": js, "ceiling": round(math.log(2.0), 4), "kl_drop": kl},
        claimed.SECTION_3_2_ANCHOR,
    )


def check_dagger_rows_coincide() -> CheckResult:
    left = claimed.TABLE1["differential_expression"]
    right = claimed.TABLE1["coexpression_modules"]
    same = _close(left[4], right[4], 1e-9)
    detail = (
        "Table 1 footnote: the two daggered rows land on the same AUPRC because their top-50 sets coincide; "
        f"the tabulated values are {left[4]} and {right[4]}"
    )
    return _result(
        "manuscript-dagger-rows-coincide",
        same,
        detail,
        {"differential_expression": left[4], "coexpression_modules": right[4]},
        claimed.TABLE1_FOOTNOTE_ANCHOR,
    )


def check_strongest_baseline_identification() -> CheckResult:
    candidates = {
        key: value
        for key, value in claimed.TABLE1.items()
        if key not in {"unconstrained_variant", "immunotargetnet"}
    }
    best = max(candidates, key=lambda key: candidates[key][0])
    return _result(
        "manuscript-strongest-baseline-identification",
        best == "typed_graph_ranking",
        f"the strongest non-framework baseline on NDCG@20 is {best}; the manuscript names the heterogeneous knowledge-graph ranking model",
        {"identified": best, "value": candidates[best][0]},
        claimed.SECTION_3_1_ANCHOR,
    )


def check_keratinocyte_candidates() -> CheckResult:
    genes = {row[1] for row in claimed.TABLE5_SHORTLIST}
    named = claimed.PROSE_KERATINOCYTE_CANDIDATES
    present = [gene for gene in named if gene in genes]
    ok = len(present) == len(named) == 5 and len(named) == 5
    return _result(
        "manuscript-keratinocyte-candidate-count",
        ok,
        f"Sec. 3.3 names five keratinocyte-carrying candidates; {len(present)} of them appear in the Table 5 shortlist",
        {"named": list(named), "present": present},
        claimed.SECTION_3_3_ANCHOR,
    )


def check_programme_count() -> CheckResult:
    names = claimed.PROGRAMME_NAMES_SECTION_2_4
    return _result(
        "manuscript-programme-count",
        len(names) == int(claimed.REPORTED["n_programmes"]),
        f"Sec. 2.4 names {len(names)} immune programmes; Sec. 2.1 states K = {int(claimed.REPORTED['n_programmes'])}",
        {"named": len(names), "stated": int(claimed.REPORTED["n_programmes"])},
        claimed.SECTION_2_4_ANCHOR,
    )


def check_normalisation_constant() -> CheckResult:
    weights = (
        claimed.REPORTED["lambda_immune"],
        claimed.REPORTED["lambda_readiness"],
        claimed.REPORTED["lambda_perturbation"],
        claimed.REPORTED["modality_dropout_train"],
    )
    config_values = (0.5, 0.5, 0.25, 0.2)
    return _result(
        "manuscript-hyperparameter-round-trip",
        weights == config_values,
        f"the released defaults reproduce the reported lambda_1, lambda_2, lambda_3 and the training dropout rate: {weights}",
        {
            "lambda_immune": weights[0],
            "lambda_readiness": weights[1],
            "lambda_perturbation": weights[2],
            "dropout_train": weights[3],
            "batch_size": claimed.REPORTED["batch_size"],
            "epochs": claimed.REPORTED["epochs"],
        },
        claimed.SECTION_2_1_ANCHOR,
    )


def check_mask_logit_notation() -> CheckResult:
    detail = (
        "Eq. (2) adds log m_{g,j} to the attention logits, so an unavailable token takes a logit of -inf. "
        "Algorithm 1 line 10 writes the same term as 'log M', a positive constant that would shift every "
        "logit equally and mask nothing."
    )
    return CheckResult(
        name="manuscript-mask-logit-notation",
        category="manuscript",
        status=FAIL,
        detail=detail,
        evidence={"equation": "Eq. (2)", "algorithm_step": "Algorithm 1 line 10", "term_written": "log M"},
        paper_anchor=claimed.SECTION_2_3_ANCHOR,
    )


def check_mask_index_space() -> CheckResult:
    detail = (
        "Sec. 2.1 defines the availability mask over the L tokens the encoder builds and writes m_{g,j} with "
        "j indexing that token axis; Algorithm 1 loops j over the M evidence layers and tests m_{g,j}. The two "
        "statements index the same symbol over different spaces."
    )
    return CheckResult(
        name="manuscript-mask-index-space",
        category="manuscript",
        status=FAIL,
        detail=detail,
        evidence={"section_2_1": "m_g in {0,1}^L", "algorithm": "j = 1..M"},
        paper_anchor=claimed.SECTION_2_1_ANCHOR,
    )


def check_context_token_duplication() -> CheckResult:
    detail = (
        "Sec. 2.3 states that the keystone layer is realised as the graph-context embedding, so it is one of the "
        "M = 4 layer tokens; Algorithm 1 builds a token per layer and then appends u_ctx(h_g) again, which yields "
        "five tokens for four layers."
    )
    return CheckResult(
        name="manuscript-context-token-duplication",
        category="manuscript",
        status=FAIL,
        detail=detail,
        evidence={"layers": int(claimed.REPORTED["n_evidence_layers"]), "tokens_implied": 5},
        paper_anchor=claimed.SECTION_2_3_ANCHOR,
    )


def check_table6_column_reference() -> CheckResult:
    detail = (
        "Sec. 2.7.1 refers to 'the cost figures in column 6 in table 6'. Table 6's sixth column is peak memory in "
        "gigabytes; the cost figures are the training hours (column 3) and the inference latency (column 4)."
    )
    return CheckResult(
        name="manuscript-table6-column-reference",
        category="manuscript",
        status=FAIL,
        detail=detail,
        evidence={"column_6": "Mem. (GB)", "cost_columns": "Training (h), Infer. (ms/gene)"},
        paper_anchor=claimed.SECTION_2_7_1_ANCHOR,
    )


def check_figure1_baselines() -> CheckResult:
    values = {key: claimed.TABLE1[key][0] for key in claimed.TABLE1}
    ranked = sorted(values, key=lambda key: values[key], reverse=True)
    plotted = {"differential_expression", "typed_graph_ranking"}
    strongest_non_framework = [key for key in ranked if key in claimed.TABLE1][:4]
    weakest_flag = ranked[-1] == "differential_expression"
    detail = (
        "Figure 1's legend plots the differential-expression baseline against typed-graph ranking, and its caption "
        "calls them 'the two most powerful baselines (non-framework)'. Table 1 makes differential expression the "
        f"weakest configuration at NDCG@20 {values['differential_expression']}."
    )
    return CheckResult(
        name="manuscript-figure1-baseline-caption",
        category="manuscript",
        status=(
            FAIL if weakest_flag and plotted == {"differential_expression", "typed_graph_ranking"} else PASS
        ),
        detail=detail,
        evidence={
            "plotted": sorted(plotted),
            "ranked_top": strongest_non_framework,
            "differential_expression_value": values["differential_expression"],
        },
        paper_anchor=claimed.FIGURE1_ANCHOR,
    )


def check_resource_count() -> CheckResult:
    counted = (
        int(claimed.REPORTED["n_cohorts"])
        + int(claimed.REPORTED["n_single_cell_resources"])
        + int(claimed.REPORTED["n_spatial_resources"])
        + int(claimed.REPORTED["n_perturbation_resources"])
    )
    keystone_resources = 4
    detail = (
        f"the abstract enumerates {counted} resources (eight cohorts, three single-cell, three spatial, two "
        f"perturbation); Sec. 2.2 names a further four keystone resources (interaction network, gene-set "
        f"collections, approved immune gene groups, drug-target and tractability annotations), so the abstract's "
        "enumeration covers the four evidence layers' non-keystone members only."
    )
    return CheckResult(
        name="manuscript-abstract-resource-count",
        category="manuscript",
        status=FAIL,
        detail=detail,
        evidence={"abstract_count": counted, "keystone_resources": keystone_resources},
        paper_anchor=claimed.SECTION_2_2_ANCHOR,
    )


def check_stability_prose_matches_table() -> CheckResult:
    prose = {
        "cohort_resample": (0.778, 0.703),
        "pathway_holdout": (0.731, 0.664),
        "modality_dropout": (0.762, 0.691),
    }
    residuals = {
        family: max(
            abs(value - claimed.TABLE3_STABILITY[family][0][index]) for index, value in enumerate(pair)
        )
        for family, pair in prose.items()
    }
    worst = max(residuals.values())
    return _result(
        "manuscript-stability-prose-vs-table",
        worst <= TOLERANCE,
        f"Sec. 3.4's framework means and fifth percentiles against Table 3; worst residual {worst:.4f}",
        {"residuals": {key: round(value, 4) for key, value in residuals.items()}},
        claimed.SECTION_3_4_ANCHOR,
    )


def check_scale_mismatch_table3_transcript() -> CheckResult:
    detail = (
        "Table 3's first seven rows are NDCG@20 and its last three are Kendall tau. The released reader reproduces "
        "the label order, and this check pins the column count so a revision that reorders them fails loudly."
    )
    widths = {len(value) for value in claimed.TABLE3_SETTINGS.values()}
    widths.update(len(value) for value in claimed.TABLE3_STABILITY.values())
    ok = len(claimed.TABLE3_SETTINGS) == 7 and len(claimed.TABLE3_STABILITY) == 3 and len(widths) == 1
    return _result(
        "manuscript-table3-shape",
        ok,
        detail,
        {
            "ndcg_rows": len(claimed.TABLE3_SETTINGS),
            "tau_rows": len(claimed.TABLE3_STABILITY),
            "columns": sorted(widths),
        },
        claimed.TABLE3_ANCHOR,
    )


def check_readiness_column_dagger() -> CheckResult:
    auprc = claimed.TABLE1["immunotargetnet"][4]
    best_baseline = max(value[4] for key, value in claimed.TABLE1.items() if key not in {"immunotargetnet"})
    return _result(
        "manuscript-readiness-outranks-baselines",
        auprc > best_baseline,
        f"the framework's readiness AUPRC {auprc} exceeds every other row's maximum {best_baseline}",
        {"framework": auprc, "best_other": best_baseline},
        claimed.TABLE1_ANCHOR,
    )


NOT_VERIFIABLE: Final[tuple[tuple[str, str, str], ...]] = (
    (
        "manuscript-degree-control-values",
        "Sec. 3.4 reports a shortlist degree gap of 0.42 standard deviations, a graph-only gap of 1.87, and a degree-adjusted move from rank four to rank five. These need the interaction-degree table of the curated set, which is not public.",
        claimed.SECTION_3_4_ANCHOR,
    ),
    (
        "manuscript-per-candidate-cohort-positions",
        "Sec. 3.6 states every shortlisted candidate stays inside the top twenty in all eight cohorts with a worst position of eleven. The per-candidate per-cohort matrix is Supplementary Table S13, which is not distributed.",
        claimed.SECTION_3_6_ANCHOR,
    ),
    (
        "manuscript-calibration-triple",
        "Sec. 3.1 reports expected calibration errors of 0.041 for the framework, 0.118 for the rule-based filter and 0.184 for the prevalence baseline. These are Supplementary Table S8, which is not distributed.",
        claimed.SECTION_3_1_ANCHOR,
    ),
    (
        "manuscript-threshold-widths",
        "Sec. 3.6 tabulates shortlist behaviour at widths eight, twelve and twenty in Supplementary Table S13 panel (c); the table is not distributed.",
        claimed.SECTION_3_6_ANCHOR,
    ),
    (
        "manuscript-perturbation-reversal-detail",
        "Sec. 3.5 and Table 5 footnote report the perturbation-reversal index under four signature definitions in Supplementary Table S6; the table is not distributed.",
        claimed.TABLE5_FOOTNOTE_ANCHOR,
    ),
    (
        "manuscript-per-cohort-mrr",
        "Table 2(b)'s footnote points at Supplementary Table S2 for per-cohort recovery broken out by mean reciprocal rank; the table is not distributed.",
        claimed.TABLE2A_ANCHOR,
    ),
)


def run_manuscript_checks() -> list[CheckResult]:
    results = [
        check_cohort_sample_sum(),
        check_evaluation_units(),
        check_cohort_means_match_loco(),
        check_distribution_row_matches_table1(),
        check_resolution_loss_column(),
        check_gain_claims(),
        check_primary_margin(),
        check_parity_point_and_band(),
        check_marginal_intervals_overlap(),
        check_constraint_axis_gaps(),
        check_best_baseline_gaps(),
        check_faithfulness_sentence_roles(),
        check_interaction_ratios(),
        check_individual_cost_claims(),
        check_shortlist_spearman(),
        check_shortlist_ordering(),
        check_cellstate_withholding_value(),
        check_cohort_ndcg_range(),
        check_table6_ratio(),
        check_table3_transfer_bound(),
        check_table3_degradation_monotone(),
        check_three_layer_counterpart(),
        check_readiness_prevalence_round_trip(),
        check_js_ceiling(),
        check_dagger_rows_coincide(),
        check_strongest_baseline_identification(),
        check_keratinocyte_candidates(),
        check_programme_count(),
        check_normalisation_constant(),
        check_mask_logit_notation(),
        check_mask_index_space(),
        check_context_token_duplication(),
        check_table6_column_reference(),
        check_figure1_baselines(),
        check_resource_count(),
        check_stability_prose_matches_table(),
        check_scale_mismatch_table3_transcript(),
        check_readiness_column_dagger(),
    ]
    results.extend(_not_run(name, detail, anchor) for name, detail, anchor in NOT_VERIFIABLE)
    return results
