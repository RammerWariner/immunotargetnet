"""Study orchestration: the reported tables, the ablations and the interaction ratios.

Ref: Sec. 3.1-3.8 and Tables 1-6.

Every reported configuration is trained and evaluated on the same label split and the same
evaluation units, which is what makes the ablation rows and the interaction ratios
comparable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Final

import numpy as np
import torch
from torch import Tensor

from immunotargetnet.bottleneck.attention import availability_tensor
from immunotargetnet.evidence.schema import MODALITY_ORDER, EvidencePanel
from immunotargetnet.graph.assembly import ANNOTATION_EDGE_TYPES, edge_keep_excluding_nodes
from immunotargetnet.graph.degree import degree_gap, fit_degree_control
from immunotargetnet.metrics.ranking import PRIMARY_CUTOFF, ndcg_at_k, ranking_order
from immunotargetnet.metrics.stability import kendall_tau_b
from immunotargetnet.model import ModelConfig, ModelOutput, TargetPrioritiser
from immunotargetnet.objectives.total import ObjectiveWeights
from immunotargetnet.protocols import baselines as baseline_module
from immunotargetnet.protocols.evaluation import (
    MetricRow,
    build_row,
    deletion_faithfulness,
    layer_mask_from_selection,
    layer_selection,
    table_of_rows,
)
from immunotargetnet.protocols.splits import LabelSplit, label_split
from immunotargetnet.training.loop import TrainingConfig, run_training

LAYER_COUNT: int = len(MODALITY_ORDER)
STABILITY_FAMILIES: tuple[str, ...] = ("cohort_resample", "pathway_holdout", "modality_dropout")

METRIC_FIELD: dict[str, str] = {
    "ndcg20": "NDCG@20",
    "mrr": "MRR",
    "kl": "KL",
    "aucdel": "AUCdel",
    "tau": "tau",
}

IMPROVING_DIRECTION: dict[str, str] = {
    "ndcg20": "higher",
    "mrr": "higher",
    "kl": "lower",
    "aucdel": "higher",
    "tau": "higher",
}

INTERACTION_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("minus_bottleneck", "minus_graph_tower", "minus_graph_minus_bottleneck", "graph x bottleneck"),
    (
        "minus_bottleneck",
        "minus_perturbation_objective",
        "minus_bottleneck_minus_perturbation",
        "bottleneck x perturbation",
    ),
)

DEFAULT_MODEL_DIM: Final[int] = 32

DEGRADATION_ROWS: tuple[tuple[str, str, str], ...] = (
    ("layers_cohort_only", "free_cohort_only", "Cohort evidence only"),
    ("layers_cohort_cell", "free_cohort_cell", "Cohort + cell state"),
    ("layers_cohort_cell_niche", "free_cohort_cell_niche", "Cohort + cell + niche"),
)


@dataclass(frozen=True)
class AblationSpec:
    key: str
    label: str
    model: ModelConfig
    dropout_rate: float
    weights: ObjectiveWeights
    keep_layers: tuple[int, ...] | None
    readiness_from_filter: bool


def _spec(
    key: str,
    label: str,
    *,
    model_dim: int = DEFAULT_MODEL_DIM,
    constrained: bool = True,
    graph_enabled: bool = True,
    dropout: float = 0.2,
    weights: ObjectiveWeights | None = None,
    keep_layers: tuple[int, ...] | None = None,
    filter_readiness: bool = False,
) -> AblationSpec:
    return AblationSpec(
        key=key,
        label=label,
        model=ModelConfig(model_dim=model_dim, constrained=constrained, graph_enabled=graph_enabled),
        dropout_rate=dropout,
        weights=weights if weights is not None else ObjectiveWeights(),
        keep_layers=keep_layers,
        readiness_from_filter=filter_readiness,
    )


def ablation_specs(model_dim: int = DEFAULT_MODEL_DIM) -> tuple[AblationSpec, ...]:
    return (
        _spec("full", "Full framework", model_dim=model_dim),
        _spec("minus_bottleneck", "-immune-relevance bottleneck", model_dim=model_dim, constrained=False),
        _spec("minus_graph_tower", "-graph tower", model_dim=model_dim, graph_enabled=False),
        _spec(
            "minus_druggability_head",
            "-druggability head",
            model_dim=model_dim,
            weights=ObjectiveWeights(immune=0.5, readiness=0.0, perturbation=0.25),
            filter_readiness=True,
        ),
        _spec(
            "minus_perturbation_objective",
            "-perturbation objective",
            model_dim=model_dim,
            weights=ObjectiveWeights(immune=0.5, readiness=0.5, perturbation=0.0),
        ),
        _spec("minus_modality_dropout", "-modality dropout", model_dim=model_dim, dropout=0.0),
        _spec("minus_niche_layer", "-niche layer", model_dim=model_dim, keep_layers=(0, 1, 3)),
        _spec("minus_cellstate_layer", "-cell-state layer", model_dim=model_dim, keep_layers=(0, 2, 3)),
        _spec(
            "minus_graph_minus_bottleneck",
            "-graph -bottleneck",
            model_dim=model_dim,
            constrained=False,
            graph_enabled=False,
        ),
        _spec(
            "minus_bottleneck_minus_perturbation",
            "-bottleneck -perturbation",
            model_dim=model_dim,
            constrained=False,
            weights=ObjectiveWeights(immune=0.5, readiness=0.5, perturbation=0.0),
        ),
        _spec("layers_keystone_only", "keystone layer alone", model_dim=model_dim, keep_layers=(3,)),
        _spec("layers_cohort_only", "cohort evidence only", model_dim=model_dim, keep_layers=(0,)),
        _spec("layers_cohort_cell", "cohort + cell state", model_dim=model_dim, keep_layers=(0, 1)),
        _spec(
            "layers_cohort_cell_niche", "cohort + cell + niche", model_dim=model_dim, keep_layers=(0, 1, 2)
        ),
        _spec("free_cohort_only", "cohort evidence only, free form", constrained=False, keep_layers=(0,)),
        _spec("free_cohort_cell", "cohort + cell state, free form", constrained=False, keep_layers=(0, 1)),
        _spec(
            "free_cohort_cell_niche",
            "cohort + cell + niche, free form",
            constrained=False,
            keep_layers=(0, 1, 2),
        ),
    )


def spec_by_key(key: str, model_dim: int = DEFAULT_MODEL_DIM) -> AblationSpec:
    for spec in ablation_specs(model_dim):
        if spec.key == key:
            return spec
    raise KeyError(f"unknown configuration: {key}")


@dataclass(frozen=True)
class StudyConfig:
    seeds: int = 5
    epochs: int = 60
    batch_size: int = 32
    validation_share: float = 0.15
    bootstrap_resamples: int = 1000
    stability_draws: int = 25
    bootstrap_seed: int = 20260101
    model_dim: int = DEFAULT_MODEL_DIM


@dataclass
class ConfigurationResult:
    key: str
    label: str
    row: MetricRow
    scores: np.ndarray
    readiness: np.ndarray
    programmes: np.ndarray
    modality_mass: np.ndarray
    seconds: float
    parameters: float
    stability: dict[str, float] = field(default_factory=dict)
    held_out_ndcg20: float = float("nan")
    pathway_holdout_ndcg20: float = float("nan")
    random_control_ndcg20: float = float("nan")

    def metric(self, name: str) -> float:
        return float(self.row.as_table()[METRIC_FIELD[name]])


@dataclass
class StudyOutcome:
    seed: int
    table1: list[dict[str, object]]
    table2_cohorts: list[dict[str, object]]
    table2_resolution: list[dict[str, object]]
    table3_settings: list[dict[str, object]]
    table4_ablations: list[dict[str, object]]
    interaction_ratios: list[dict[str, object]]
    shortlist: list[dict[str, object]]
    shortlist_degree_gap: float
    cost: dict[str, float]
    provenance: dict[str, object]
    notes: list[str] = field(default_factory=list)


def _copy_panel(
    panel: EvidencePanel,
    *,
    availability: np.ndarray | None = None,
    cohort_features: np.ndarray | None = None,
) -> EvidencePanel:
    return EvidencePanel(
        genes=list(panel.genes),
        cohort_feature_names=list(panel.cohort_feature_names),
        cellstate_feature_names=list(panel.cellstate_feature_names),
        niche_feature_names=list(panel.niche_feature_names),
        cohort_features=panel.cohort_features if cohort_features is None else cohort_features,
        cohort_sample_counts=panel.cohort_sample_counts,
        cellstate_features=panel.cellstate_features,
        niche_features=panel.niche_features,
        context_features=panel.context_features,
        availability=panel.availability if availability is None else availability,
        programme_targets=panel.programme_targets,
        programme_labels=list(panel.programme_labels),
        tractability=panel.tractability,
        target_labels=panel.target_labels,
        development_labels=panel.development_labels,
        reversal_index=panel.reversal_index,
        reversal_available=panel.reversal_available,
        interaction_degree=panel.interaction_degree,
        modality_attribution_reference=panel.modality_attribution_reference,
        adjacency=panel.adjacency,
        pathway_labels=list(panel.pathway_labels),
        pathway_membership=[list(row) for row in panel.pathway_membership],
        pathway_sizes=list(panel.pathway_sizes),
        cohort_accessions=list(panel.cohort_accessions),
        source_note=panel.source_note,
    )


def with_availability(panel: EvidencePanel, mask: np.ndarray) -> EvidencePanel:
    return _copy_panel(panel, availability=mask)


def with_cohort_features(panel: EvidencePanel, block: np.ndarray) -> EvidencePanel:
    return _copy_panel(panel, cohort_features=block)


def layer_availability(panel: EvidencePanel, keep: tuple[int, ...]) -> np.ndarray:
    return layer_mask_from_selection(panel.availability, layer_selection(panel.availability.shape[1], keep))


def score_under(model: TargetPrioritiser, panel: EvidencePanel, mask: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        output = model(panel, availability_tensor(mask))
    return np.asarray(output.relevance.detach().cpu().numpy(), dtype=np.float64)


def annotation_edge_keep(panel: EvidencePanel, model: TargetPrioritiser) -> Tensor:
    """Edges that survive the pathway holdout: everything outside the annotation families."""
    with torch.no_grad():
        state = model.bottleneck(model.features(panel), availability_tensor(panel.availability))
        assembly = model.graph_assembly(panel, state)
    names = assembly.batch.edge_type_names
    blocked = [names.index(name) for name in ANNOTATION_EDGE_TYPES if name in names]
    keep = torch.ones_like(assembly.batch.edge_type_index, dtype=torch.bool)
    for index in blocked:
        keep &= assembly.batch.edge_type_index != index
    return keep


def gene_subset_edge_keep(panel: EvidencePanel, model: TargetPrioritiser, genes: np.ndarray) -> Tensor:
    """Drop every edge incident to the named gene nodes of the assembled graph."""
    with torch.no_grad():
        state = model.bottleneck(model.features(panel), availability_tensor(panel.availability))
        assembly = model.graph_assembly(panel, state)
    blocked = np.zeros(assembly.batch.n_nodes, dtype=bool)
    blocked[np.asarray(genes, dtype=np.int64)] = True
    return edge_keep_excluding_nodes(assembly.batch, blocked)


def perturbed_scores(
    model: TargetPrioritiser,
    panel: EvidencePanel,
    *,
    family: str,
    draws: int,
    seed: int,
) -> list[np.ndarray]:
    """Rerun the frozen model under one perturbation family of Sec. 3.4."""
    generator = np.random.default_rng(seed)
    results: list[np.ndarray] = []
    if family == "cohort_resample":
        for _ in range(draws):
            columns = generator.integers(
                0, panel.cohort_features.shape[1], size=panel.cohort_features.shape[1]
            )
            perturbed = with_cohort_features(panel, panel.cohort_features[:, columns])
            results.append(score_under(model, perturbed, panel.availability))
        return results
    if family == "modality_dropout":
        for _ in range(draws):
            available = int(generator.integers(1, LAYER_COUNT))
            mask = layer_mask_from_selection(
                panel.availability,
                layer_selection(panel.availability.shape[1], tuple(range(available))),
            )
            results.append(score_under(model, panel, mask))
        return results
    if family == "pathway_holdout":
        keep = annotation_edge_keep(panel, model)
        with torch.no_grad():
            for _ in range(draws):
                output = model(panel, availability_tensor(panel.availability), keep)
                results.append(np.asarray(output.relevance.detach().cpu().numpy(), dtype=np.float64))
        return results
    raise KeyError(f"unknown perturbation family: {family}")


def train_configuration(
    panel: EvidencePanel,
    spec: AblationSpec,
    *,
    seed: int,
    config: StudyConfig,
    split: LabelSplit,
) -> tuple[TargetPrioritiser, ModelOutput, float, EvidencePanel]:
    local_panel = panel
    if spec.keep_layers is not None:
        local_panel = with_availability(panel, layer_availability(panel, spec.keep_layers))
    training = TrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        modality_dropout_rate=spec.dropout_rate,
        weights=spec.weights,
        model=spec.model,
    )
    started = time.perf_counter()
    result = run_training(
        local_panel,
        training_units=split.training,
        validation_units=split.validation,
        seed=seed,
        config=training,
    )
    elapsed = time.perf_counter() - started
    with torch.no_grad():
        output = result.model(local_panel, availability_tensor(local_panel.availability))
    return result.model, output, elapsed, local_panel


def filter_readiness(panel: EvidencePanel, scores: np.ndarray) -> np.ndarray:
    from immunotargetnet.metrics.discrimination import tractability_filter_ranking

    return tractability_filter_ranking(panel.tractability, scores)


def configuration_result(
    panel: EvidencePanel,
    local_panel: EvidencePanel,
    spec: AblationSpec,
    model: TargetPrioritiser,
    output: ModelOutput,
    seconds: float,
    *,
    config: StudyConfig,
    seed: int,
    split: LabelSplit,
) -> ConfigurationResult:
    scores = np.asarray(output.relevance.detach().cpu().numpy(), dtype=np.float64)
    programmes = np.asarray(output.programmes.detach().cpu().numpy(), dtype=np.float64)
    mass = np.asarray(output.modality_mass.detach().cpu().numpy(), dtype=np.float64)
    readiness = np.asarray(output.tractability_probability().detach().cpu().numpy(), dtype=np.float64)
    if spec.readiness_from_filter:
        readiness = filter_readiness(panel, scores)

    def score_from_mask(mask: np.ndarray) -> np.ndarray:
        return score_under(model, local_panel, layer_mask_from_selection(local_panel.availability, mask))

    aucdel = deletion_faithfulness(mass, score_from_mask)
    stability: dict[str, float] = {}
    for family in STABILITY_FAMILIES:
        candidates = perturbed_scores(
            model, local_panel, family=family, draws=config.stability_draws, seed=seed * 17 + 3
        )
        taus = np.array([kendall_tau_b(scores, candidate) for candidate in candidates], dtype=np.float64)
        stability[family] = float(taus.mean()) if taus.size else float("nan")
        stability[f"{family}_p5"] = float(np.percentile(taus, 5.0)) if taus.size else float("nan")
    labels = panel.target_labels.astype(np.float64)
    held_out = (
        ndcg_at_k(labels[split.held_out], scores[split.held_out], PRIMARY_CUTOFF)
        if split.held_out.size
        else float("nan")
    )
    withheld = np.flatnonzero(~local_panel.availability.any(axis=1))
    keep = gene_subset_edge_keep(panel, model, withheld) if withheld.size else None
    with torch.no_grad():
        trimmed = model(panel, availability_tensor(panel.availability), keep)
    pathway_scores = np.asarray(trimmed.relevance.detach().cpu().numpy(), dtype=np.float64)
    generator = np.random.default_rng(seed + 909)
    random_genes = np.sort(
        generator.choice(panel.n_genes, size=min(max(withheld.size, 1), panel.n_genes), replace=False)
    )
    random_keep = gene_subset_edge_keep(panel, model, random_genes)
    with torch.no_grad():
        control = model(panel, availability_tensor(panel.availability), random_keep)
    control_scores = np.asarray(control.relevance.detach().cpu().numpy(), dtype=np.float64)
    row = build_row(
        method=spec.label,
        panel=panel,
        scores=scores,
        programmes=programmes,
        readiness=readiness,
        tau=stability.get("cohort_resample", float("nan")),
        aucdel=aucdel,
    )
    return ConfigurationResult(
        key=spec.key,
        label=spec.label,
        row=row,
        scores=scores,
        readiness=readiness,
        programmes=programmes,
        modality_mass=mass,
        seconds=seconds,
        parameters=float(model.trainable_parameters()),
        stability=stability,
        held_out_ndcg20=float(held_out),
        pathway_holdout_ndcg20=ndcg_at_k(labels, pathway_scores, PRIMARY_CUTOFF),
        random_control_ndcg20=ndcg_at_k(labels, control_scores, PRIMARY_CUTOFF),
    )


def baseline_rank_stability(
    panel: EvidencePanel,
    reference: np.ndarray,
    key: str,
    *,
    family: str,
    draws: int,
    seed: int,
) -> float:
    """Kendall tau of a baseline ranking under one perturbation family.

    A baseline that never reads the perturbed axis returns a tau of one, which is the
    honest reading rather than a missing value.
    """
    generator = np.random.default_rng(seed + 404)
    taus: list[float] = []
    for _ in range(draws):
        if family == "cohort_resample":
            columns = generator.integers(
                0, panel.cohort_features.shape[1], size=panel.cohort_features.shape[1]
            )
            perturbed = with_cohort_features(panel, panel.cohort_features[:, columns])
        elif family == "modality_dropout":
            available = int(generator.integers(1, LAYER_COUNT))
            perturbed = with_availability(
                panel,
                layer_mask_from_selection(
                    panel.availability,
                    layer_selection(panel.availability.shape[1], tuple(range(available))),
                ),
            )
        else:
            perturbed = panel
        candidate = baseline_module.baseline_scores(perturbed).get(key)
        taus.append(kendall_tau_b(reference, candidate))
    return float(np.mean(taus)) if taus else float("nan")


def interaction_ratio(combined: float, first: float, second: float, direction: str) -> float:
    """Combined cost over the sum of individual costs, both in the improving direction."""

    def cost(value: float) -> float:
        return value if direction == "higher" else -value

    denominator = cost(first) + cost(second)
    if denominator == 0.0:
        return float("nan")
    return cost(combined) / denominator


def cohort_blocks(n_genes: int, n_cohorts: int, *, seed: int) -> list[np.ndarray]:
    generator = np.random.default_rng(seed)
    order = generator.permutation(n_genes)
    return [np.sort(chunk) for chunk in np.array_split(order, n_cohorts)]


class StudyRunner:
    """Train every reported configuration once per seed and derive the tables."""

    def __init__(self, panel: EvidencePanel, config: StudyConfig) -> None:
        self.panel = panel
        self.config = config

    def run(self, *, seed: int) -> StudyOutcome:
        split = label_split(
            self.panel.n_genes,
            seed=seed,
            validation_share=self.config.validation_share,
            protected=self.panel.target_labels,
        )
        trained: dict[str, ConfigurationResult] = {}
        for spec in ablation_specs(self.config.model_dim):
            model, output, seconds, local_panel = train_configuration(
                self.panel, spec, seed=seed, config=self.config, split=split
            )
            trained[spec.key] = configuration_result(
                self.panel,
                local_panel,
                spec,
                model,
                output,
                seconds,
                config=self.config,
                seed=seed,
                split=split,
            )
        return self._assemble(panel=self.panel, trained=trained, split=split, seed=seed)

    def _assemble(
        self,
        *,
        panel: EvidencePanel,
        trained: dict[str, ConfigurationResult],
        split: LabelSplit,
        seed: int,
    ) -> StudyOutcome:
        framework = trained["full"]
        unconstrained = trained["minus_bottleneck"]
        upstream = {
            "unconstrained_variant": unconstrained.scores,
            "immunotargetnet": framework.scores,
        }
        comparison = baseline_module.baseline_scores(panel, upstream=upstream)
        methods: dict[str, np.ndarray] = {
            key: comparison.get(key) for key in baseline_module.BASELINE_KEYS[:-1]
        }
        methods["unconstrained_variant"] = unconstrained.scores
        methods["immunotargetnet"] = framework.scores

        table1_rows: list[MetricRow] = []
        for key in baseline_module.BASELINE_KEYS[:-1]:
            table1_rows.append(
                build_row(
                    method=baseline_module.BASELINE_LABELS[key],
                    panel=panel,
                    scores=methods[key],
                    readiness=filter_readiness(panel, methods[key]),
                )
            )
        table1_rows.append(
            build_row(
                method=baseline_module.BASELINE_LABELS["unconstrained_variant"],
                panel=panel,
                scores=unconstrained.scores,
                programmes=unconstrained.programmes,
                readiness=unconstrained.readiness,
                aucdel=unconstrained.row.aucdel,
                tau=unconstrained.stability.get("cohort_resample", float("nan")),
            )
        )
        table1_rows.append(framework.row)
        return StudyOutcome(
            seed=seed,
            table1=table_of_rows(table1_rows),
            table2_cohorts=self._per_cohort(panel, methods, seed=seed),
            table2_resolution=self._resolution(trained),
            table3_settings=self._generalisation(panel, methods, trained, seed=seed),
            table4_ablations=self._ablation_rows(trained),
            interaction_ratios=self._interaction_rows(trained),
            shortlist=self._shortlist(panel, framework),
            shortlist_degree_gap=self._degree_gap(panel, framework),
            cost=self._cost(panel, trained, framework),
            provenance={
                "seed": seed,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "validation_share": self.config.validation_share,
                "stability_draws": self.config.stability_draws,
                "n_genes": panel.n_genes,
                "tokens": int(panel.availability.shape[1]),
                "training_units": int(split.training.size),
                "validation_units": int(split.validation.size),
                "held_out_units": int(split.held_out.size),
            },
            notes=[
                "All values are produced on the released panel; the manuscript's held-out "
                "curated set is not public, so no manuscript value is claimed here.",
            ],
        )

    def _per_cohort(
        self, panel: EvidencePanel, methods: dict[str, np.ndarray], *, seed: int
    ) -> list[dict[str, object]]:
        labels = panel.target_labels.astype(np.float64)
        blocks = cohort_blocks(panel.n_genes, max(len(panel.cohort_accessions), 2), seed=seed)
        accessions = panel.cohort_accessions or [f"cohort_{index}" for index in range(len(blocks))]
        rows: list[dict[str, object]] = []
        for index, block in enumerate(blocks):
            if block.size == 0:
                continue
            record: dict[str, object] = {
                "evaluation": accessions[index % len(accessions)],
                "n": int(block.size),
            }
            for method in (
                "immunotargetnet",
                "unconstrained_variant",
                "typed_graph_ranking",
                "differential_expression",
            ):
                record[method] = round(ndcg_at_k(labels[block], methods[method][block], PRIMARY_CUTOFF), 4)
            rows.append(record)
        return rows

    def _resolution(self, trained: dict[str, ConfigurationResult]) -> list[dict[str, object]]:
        full = trained["full"].row.ndcg20
        rows: list[dict[str, object]] = []
        for key in (
            "full",
            "minus_niche_layer",
            "minus_cellstate_layer",
            "minus_graph_tower",
            "minus_perturbation_objective",
            "minus_druggability_head",
            "minus_modality_dropout",
        ):
            result = trained[key]
            rows.append(
                {
                    "subsystem_withheld": result.label,
                    "NDCG@20": round(result.row.ndcg20, 4),
                    "loss_vs_full": round(full - result.row.ndcg20, 4),
                }
            )
        return rows

    def _generalisation(
        self,
        panel: EvidencePanel,
        methods: dict[str, np.ndarray],
        trained: dict[str, ConfigurationResult],
        *,
        seed: int,
    ) -> list[dict[str, object]]:
        labels = panel.target_labels.astype(np.float64)
        blocks = cohort_blocks(panel.n_genes, max(len(panel.cohort_accessions), 2), seed=seed)
        framework = trained["full"]
        unconstrained = trained["minus_bottleneck"]
        graph_only = trained["layers_keystone_only"]
        latent_only = trained["layers_cohort_only"]
        difference = np.asarray(unconstrained.scores, dtype=np.float64) - np.asarray(
            framework.scores, dtype=np.float64
        )
        settings: list[dict[str, object]] = [
            {
                "setting": "In-distribution",
                "IT-Net": round(framework.row.ndcg20, 4),
                "unconstrained": round(unconstrained.row.ndcg20, 4),
                "typed_graph": round(graph_only.row.ndcg20, 4),
                "latent_factor": round(
                    ndcg_at_k(labels, methods["latent_factor_integration"], PRIMARY_CUTOFF), 4
                ),
                "differential_expression": round(
                    ndcg_at_k(labels, methods["differential_expression"], PRIMARY_CUTOFF), 4
                ),
            },
            {
                "setting": "Leave-one-cohort-out (mean)",
                "IT-Net": round(
                    float(
                        np.mean(
                            [
                                ndcg_at_k(labels[block], framework.scores[block], PRIMARY_CUTOFF)
                                for block in blocks
                            ]
                        )
                    ),
                    4,
                ),
                "unconstrained": round(
                    float(
                        np.mean(
                            [
                                ndcg_at_k(labels[block], unconstrained.scores[block], PRIMARY_CUTOFF)
                                for block in blocks
                            ]
                        )
                    ),
                    4,
                ),
                "typed_graph": round(graph_only.row.ndcg20, 4),
                "latent_factor": round(
                    ndcg_at_k(labels, methods["latent_factor_integration"], PRIMARY_CUTOFF), 4
                ),
                "differential_expression": round(
                    float(
                        np.mean(
                            [
                                ndcg_at_k(
                                    labels[block], methods["differential_expression"][block], PRIMARY_CUTOFF
                                )
                                for block in blocks
                            ]
                        )
                    ),
                    4,
                ),
            },
            {
                "setting": "Held-out block (single fold)",
                "IT-Net": round(framework.held_out_ndcg20, 4),
                "unconstrained": round(unconstrained.held_out_ndcg20, 4),
                "typed_graph": round(graph_only.held_out_ndcg20, 4),
                "latent_factor": round(latent_only.row.ndcg20, 4),
                "differential_expression": round(
                    ndcg_at_k(labels, methods["differential_expression"], PRIMARY_CUTOFF), 4
                ),
            },
            {
                "setting": "Pathway holdout",
                "IT-Net": round(framework.pathway_holdout_ndcg20, 4),
                "unconstrained": round(unconstrained.pathway_holdout_ndcg20, 4),
                "typed_graph": round(graph_only.pathway_holdout_ndcg20, 4),
                "latent_factor": round(latent_only.pathway_holdout_ndcg20, 4),
                "differential_expression": round(
                    ndcg_at_k(labels, methods["differential_expression"], PRIMARY_CUTOFF), 4
                ),
            },
            {
                "setting": "Random-subset control",
                "IT-Net": round(framework.random_control_ndcg20, 4),
                "unconstrained": round(unconstrained.random_control_ndcg20, 4),
                "typed_graph": round(graph_only.random_control_ndcg20, 4),
                "latent_factor": round(latent_only.random_control_ndcg20, 4),
                "differential_expression": round(
                    ndcg_at_k(labels, methods["differential_expression"], PRIMARY_CUTOFF), 4
                ),
            },
        ]
        for key, free_key, label in DEGRADATION_ROWS:
            constrained = trained[key]
            free = trained[free_key]
            settings.append(
                {
                    "setting": label,
                    "IT-Net": round(constrained.row.ndcg20, 4),
                    "unconstrained": round(free.row.ndcg20, 4),
                    "typed_graph": round(graph_only.row.ndcg20, 4),
                    "latent_factor": round(
                        ndcg_at_k(labels, methods["latent_factor_integration"], PRIMARY_CUTOFF), 4
                    ),
                    "differential_expression": round(
                        ndcg_at_k(labels, methods["differential_expression"], PRIMARY_CUTOFF), 4
                    ),
                }
            )
        for family, label in (
            ("cohort_resample", "Stability, cohort resample"),
            ("pathway_holdout", "Stability, pathway holdout"),
            ("modality_dropout", "Stability, modality dropout"),
        ):
            settings.append(
                {
                    "setting": label,
                    "IT-Net": round(framework.stability[family], 4),
                    "unconstrained": round(unconstrained.stability[family], 4),
                    "typed_graph": round(graph_only.stability[family], 4),
                    "latent_factor": round(
                        baseline_rank_stability(
                            panel,
                            methods["latent_factor_integration"],
                            "latent_factor_integration",
                            family=family,
                            draws=self.config.stability_draws,
                            seed=seed,
                        ),
                        4,
                    ),
                    "differential_expression": round(
                        baseline_rank_stability(
                            panel,
                            methods["differential_expression"],
                            "differential_expression",
                            family=family,
                            draws=self.config.stability_draws,
                            seed=seed,
                        ),
                        4,
                    ),
                }
            )
        del difference
        return settings

    def _ablation_rows(self, trained: dict[str, ConfigurationResult]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for spec in ablation_specs(self.config.model_dim):
            if spec.key.startswith(("layers_", "free_")):
                continue
            result = trained[spec.key]
            rows.append(
                {
                    "configuration": result.label,
                    "NDCG@20": round(result.row.ndcg20, 4),
                    "MRR": round(result.row.mrr, 4),
                    "KL": round(result.row.kl, 4),
                    "AUCdel": round(result.row.aucdel, 4),
                    "tau": round(result.row.tau, 4),
                    "parameters": int(result.parameters),
                    "seconds": round(result.seconds, 3),
                }
            )
        return rows

    def _interaction_rows(self, trained: dict[str, ConfigurationResult]) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for first_key, second_key, combined_key, display in INTERACTION_PAIRS:
            record: dict[str, object] = {"interaction": display}
            for metric, direction in IMPROVING_DIRECTION.items():
                value = interaction_ratio(
                    trained[combined_key].metric(metric),
                    trained[first_key].metric(metric),
                    trained[second_key].metric(metric),
                    direction,
                )
                record[metric] = round(value, 4) if np.isfinite(value) else "nan"
            rows.append(record)
        return rows

    def _shortlist(
        self, panel: EvidencePanel, framework: ConfigurationResult, *, n: int = 12
    ) -> list[dict[str, object]]:
        order = ranking_order(framework.scores)[:n]
        control = fit_degree_control(framework.scores, panel.interaction_degree)
        rows: list[dict[str, object]] = []
        for rank, index in enumerate(order, start=1):
            programme_index = int(np.argmax(framework.programmes[index]))
            rows.append(
                {
                    "rank": rank,
                    "gene": panel.genes[index],
                    "relevance": round(float(framework.scores[index]), 4),
                    "readiness": round(float(framework.readiness[index]), 4),
                    "degree_adjusted_rank": int(control.degree_then_rank[index]),
                    "programme": panel.programme_labels[programme_index],
                    "carries_target_label": bool(panel.target_labels[index]),
                    "clinical_stage": bool(panel.tractability[index]),
                }
            )
        return rows

    def _degree_gap(self, panel: EvidencePanel, framework: ConfigurationResult, *, n: int = 12) -> float:
        order = ranking_order(framework.scores)[:n]
        selection = np.zeros(panel.n_genes, dtype=bool)
        selection[order] = True
        return float(degree_gap(panel.interaction_degree, selection, panel.target_labels))

    def _cost(
        self,
        panel: EvidencePanel,
        trained: dict[str, ConfigurationResult],
        framework: ConfigurationResult,
    ) -> dict[str, float]:
        unconstrained = trained["minus_bottleneck"]
        graph_only = trained["layers_keystone_only"]
        return {
            "framework_params": framework.parameters,
            "unconstrained_params": unconstrained.parameters,
            "graph_only_params": graph_only.parameters,
            "framework_seconds": framework.seconds,
            "unconstrained_seconds": unconstrained.seconds,
            "graph_only_seconds": graph_only.seconds,
            "training_time_ratio_vs_graph_only": framework.seconds / max(graph_only.seconds, 1e-9),
            "framework_ndcg20": framework.row.ndcg20,
            "unconstrained_ndcg20": unconstrained.row.ndcg20,
            "graph_only_ndcg20": graph_only.row.ndcg20,
            "n_genes": float(panel.n_genes),
            "n_tokens": float(panel.availability.shape[1]),
        }


def run_study(panel: EvidencePanel, *, config: StudyConfig, seed: int) -> StudyOutcome:
    return StudyRunner(panel, config).run(seed=seed)
