"""Assembly of the candidate-pool panel the model consumes.

Ref: Sec. 2.1 (candidate pool, modality tuple, availability mask) and Sec. 2.7.4
(candidate pool size before processing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

from immunotargetnet.evidence import fixture as fixture_builder
from immunotargetnet.evidence.cellstate import CELL_STATE_ACCESSIONS
from immunotargetnet.evidence.cohort import (
    COHORT_ACCESSIONS,
    COHORT_SPECS,
    CohortSpec,
    ContrastResult,
    cohort_feature_matrix,
)
from immunotargetnet.evidence.keystone import KeystoneSnapshot
from immunotargetnet.evidence.markers import CELL_POPULATIONS, NICHE_COMPARTMENTS
from immunotargetnet.evidence.msigdb import ProgrammeTable
from immunotargetnet.evidence.niche import NICHE_ACCESSIONS, NICHE_DATASETS
from immunotargetnet.evidence.pathways import PathwayTable
from immunotargetnet.evidence.schema import EvidencePanel

DEFAULT_CONTEXT_DIM: Final[int] = 16

MODALITY_SOURCE_KEYS: Final[dict[str, str]] = {
    "cohort": "cohort",
    "cell_state": "cell_state",
    "niche": "niche",
    "keystone": "keystone",
}


@dataclass(frozen=True)
class PanelProvenance:
    """Which parts of a panel came from a distributed resource and which from the fixture."""

    candidate_pool: str
    cohort_layer: str
    cell_state_layer: str
    niche_layer: str
    keystone_layer: str
    programme_table: str
    perturbation_layer: str
    notes: tuple[str, ...] = ()


@dataclass
class PanelBundle:
    panel: EvidencePanel
    provenance: PanelProvenance
    construct: fixture_builder.FixtureConstruct | None = None
    cohort_contrasts: list[ContrastResult] = field(default_factory=list)


def normalise_programme_targets(raw: np.ndarray, *, temperature: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax over the programme axis, the simplex label of Eq. (1)."""
    scaled = raw / max(temperature, 1e-6)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    normalised: np.ndarray = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    return normalised


def build_panel_from_fixture(
    snapshot: KeystoneSnapshot,
    table: ProgrammeTable,
    pathways: PathwayTable,
    *,
    seed: int,
    context_dim: int = DEFAULT_CONTEXT_DIM,
    programme_temperature: float = 0.35,
) -> PanelBundle:
    construct = fixture_builder.build_construct(snapshot, table, seed=seed)
    genes = construct.genes
    cohort = fixture_builder.cohort_features(construct, seed=seed)
    cell_state = fixture_builder.cellstate_features(construct, seed=seed)
    niche_sections = fixture_builder.niche_section_features(construct, seed=seed)
    niche_features = _niche_summary(niche_sections)
    context = fixture_builder.context_features(snapshot, genes, seed=seed, dim=context_dim)
    availability = fixture_builder.availability_mask(construct, snapshot, seed=seed)
    targets = normalise_programme_targets(construct.programme_targets, temperature=programme_temperature)
    membership, sizes = pathways.incidence(genes)
    panel = EvidencePanel(
        genes=list(genes),
        cohort_feature_names=list(COHORT_ACCESSIONS),
        cellstate_feature_names=list(CELL_POPULATIONS),
        niche_feature_names=list(NICHE_COMPARTMENTS),
        cohort_features=cohort,
        cohort_sample_counts=fixture_builder.cohort_sample_counts(),
        cellstate_features=cell_state,
        niche_features=niche_features,
        context_features=context,
        availability=availability,
        programme_targets=targets,
        programme_labels=list(table.display_names),
        tractability=fixture_builder.tractability_vector(snapshot, genes),
        target_labels=construct.target_labels,
        development_labels=construct.development_labels,
        reversal_index=construct.reversal_index,
        reversal_available=construct.reversal_available,
        interaction_degree=fixture_builder.interaction_degree(snapshot, genes, membership, sizes),
        modality_attribution_reference=np.abs(cohort).mean(axis=0),
        adjacency=fixture_builder.adjacency_lists(snapshot, genes),
        pathway_labels=pathways.labels(),
        pathway_membership=membership,
        pathway_sizes=sizes,
        cohort_accessions=list(COHORT_ACCESSIONS),
        source_note="candidate pool from the programme corpus; features from the declared construct",
    )
    provenance = PanelProvenance(
        candidate_pool="programme corpus intersected with the frozen annotation snapshot",
        cohort_layer="declared construct: eight public cohorts are not redistributed",
        cell_state_layer="declared construct over the canonical population marker panel",
        niche_layer="declared construct over the canonical compartment marker panel",
        keystone_layer="frozen snapshot: platform tractability annotations plus the pathway corpus",
        programme_table="MSigDB release 2023.2.Hs gene-set collections",
        perturbation_layer="declared construct: the compound perturbation reference is not redistributable",
        notes=(
            "Cohort-level manuscript values cannot be produced from this panel and are reported NOT_RUN.",
            "The programme table, the tractability labels and the typed edges are the distributed resources.",
        ),
    )
    return PanelBundle(panel=panel, provenance=provenance, construct=construct)


def build_panel_from_cohorts(
    snapshot: KeystoneSnapshot,
    table: ProgrammeTable,
    pathways: PathwayTable,
    contrasts: list[ContrastResult],
    *,
    seed: int,
    context_dim: int = DEFAULT_CONTEXT_DIM,
    programme_temperature: float = 0.35,
) -> PanelBundle:
    """Assemble a panel from real cohort contrasts plus the frozen keystone snapshot."""
    genes = fixture_builder.candidate_pool(snapshot, table)
    cohort = cohort_feature_matrix(contrasts, genes)
    if cohort.shape[1] == 0:
        raise ValueError("no cohort contrast available")
    construct = fixture_builder.build_construct(snapshot, table, seed=seed)
    cell_state = fixture_builder.cellstate_features(construct, seed=seed)
    niche = _niche_summary(fixture_builder.niche_section_features(construct, seed=seed))
    context = fixture_builder.context_features(snapshot, genes, seed=seed, dim=context_dim)
    availability = fixture_builder.availability_mask(construct, snapshot, seed=seed)
    targets = normalise_programme_targets(construct.programme_targets, temperature=programme_temperature)
    membership, sizes = pathways.incidence(genes)
    panel = EvidencePanel(
        genes=list(genes),
        cohort_feature_names=[item.accession for item in contrasts],
        cellstate_feature_names=list(CELL_POPULATIONS),
        niche_feature_names=list(NICHE_COMPARTMENTS),
        cohort_features=cohort,
        cohort_sample_counts=np.array(
            [item.n_positive + item.n_negative for item in contrasts], dtype=np.int64
        ),
        cellstate_features=cell_state,
        niche_features=niche,
        context_features=context,
        availability=availability,
        programme_targets=targets,
        programme_labels=list(table.display_names),
        tractability=fixture_builder.tractability_vector(snapshot, genes),
        target_labels=construct.target_labels,
        development_labels=construct.development_labels,
        reversal_index=construct.reversal_index,
        reversal_available=construct.reversal_available,
        interaction_degree=fixture_builder.interaction_degree(snapshot, genes, membership, sizes),
        modality_attribution_reference=np.abs(cohort).mean(axis=0),
        adjacency=fixture_builder.adjacency_lists(snapshot, genes),
        pathway_labels=pathways.labels(),
        pathway_membership=membership,
        pathway_sizes=sizes,
        cohort_accessions=[item.accession for item in contrasts],
        source_note="cohort contrasts read from distributed series matrices",
    )
    provenance = PanelProvenance(
        candidate_pool="programme corpus intersected with the frozen annotation snapshot",
        cohort_layer="series matrices on disk: " + ",".join(item.accession for item in contrasts),
        cell_state_layer="declared construct over the canonical population marker panel",
        niche_layer="declared construct over the canonical compartment marker panel",
        keystone_layer="frozen snapshot: platform tractability annotations plus the pathway corpus",
        programme_table="MSigDB release 2023.2.Hs gene-set collections",
        perturbation_layer="declared construct: the compound perturbation reference is not redistributable",
        notes=("Cohorts missing from disk are absent from the cohort axis rather than imputed.",),
    )
    return PanelBundle(panel=panel, provenance=provenance, construct=construct, cohort_contrasts=contrasts)


def _niche_summary(section_block: np.ndarray) -> np.ndarray:
    """Mean over the psoriasis-supported sections, then per gene across compartments."""
    psoriasis = [index for index, dataset in enumerate(NICHE_DATASETS) if dataset.psoriasis_cohort]
    slices: list[np.ndarray] = []
    for index in psoriasis:
        start = index * fixture_builder.NICHE_SECTIONS_PER_DATASET
        stop = start + fixture_builder.NICHE_SECTIONS_PER_DATASET
        slices.append(section_block[start:stop].mean(axis=0))
    stacked = np.stack(slices, axis=0)
    return np.asarray(stacked.mean(axis=0), dtype=np.float64)


def cohort_specs() -> tuple[CohortSpec, ...]:
    return COHORT_SPECS


def resource_inventory() -> dict[str, tuple[str, ...]]:
    return {
        "cohort": COHORT_ACCESSIONS,
        "cell_state": CELL_STATE_ACCESSIONS,
        "niche": NICHE_ACCESSIONS,
    }


def snapshot_path(assets_dir: Path) -> Path:
    from immunotargetnet.evidence.keystone import SNAPSHOT_NAME

    return assets_dir / SNAPSHOT_NAME
