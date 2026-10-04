"""Schema-compatible cohort fixture.

Ref: Sec. 2.1 (the panel the model consumes) and Sec. 2.7.1 (cohort-level splits).

The eight expression cohorts are public but not distributable with the release, so the
release executes the pipeline over a fixture whose per-gene latent structure is declared
here and whose programme table, tractability labels and typed edges come from the frozen
keystone snapshot. No manuscript table value is claimed from a fixture run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from immunotargetnet.evidence.cellstate import CELL_STATE_ACCESSIONS
from immunotargetnet.evidence.cohort import COHORT_ACCESSIONS, COHORT_SPECS
from immunotargetnet.evidence.keystone import KeystoneSnapshot
from immunotargetnet.evidence.markers import CELL_POPULATIONS, NICHE_COMPARTMENTS
from immunotargetnet.evidence.msigdb import ProgrammeTable
from immunotargetnet.evidence.niche import NICHE_ACCESSIONS
from immunotargetnet.evidence.pathways import co_membership_degree
from immunotargetnet.evidence.schema import MODALITY_ORDER
from immunotargetnet.runtime.numeric import zscore
from immunotargetnet.runtime.seed import spawn_generators

NICHE_SECTIONS_PER_DATASET: Final[int] = 3
COHORT_EFFECT_SCALE: Final[float] = 0.9
COHORT_NOISE_SCALE: Final[float] = 0.55
POPULATION_NOISE_SCALE: Final[float] = 0.35
# Share of the candidate pool that carries a curated-target label, and the declared
# relation between the development labels the ranking term sees and the held-out set.
TARGET_QUANTILE: Final[float] = 0.06
DEVELOPMENT_SIGNAL: Final[float] = 0.55
DEVELOPMENT_NOISE: Final[float] = 0.75
DEVELOPMENT_OFFSET: Final[float] = 1.2


@dataclass(frozen=True)
class FixtureConstruct:
    """The declared latent structure a fixture run is built from."""

    seed: int
    genes: tuple[str, ...]
    programme_targets: np.ndarray
    latent_relevance: np.ndarray
    cohort_loading: np.ndarray
    population_affinity: np.ndarray
    compartment_affinity: np.ndarray
    reversal_index: np.ndarray
    reversal_available: np.ndarray
    target_labels: np.ndarray
    development_labels: np.ndarray


def _programme_targets(table: ProgrammeTable, genes: tuple[str, ...]) -> np.ndarray:
    """Normalised programme incidence per gene, derived from the real gene-set corpus."""
    targets = np.zeros((len(genes), len(table.keys)), dtype=np.float64)
    for column, key in enumerate(table.keys):
        members = set(table.members[key])
        for row, gene in enumerate(genes):
            if gene in members:
                targets[row, column] = 1.0
    counts = targets.sum(axis=1, keepdims=True)
    uniform = np.full_like(targets, 1.0 / targets.shape[1])
    smoothed = np.where(counts > 0, targets, uniform)
    normalised: np.ndarray = smoothed / smoothed.sum(axis=1, keepdims=True)
    return normalised


def candidate_pool(snapshot: KeystoneSnapshot, table: ProgrammeTable) -> tuple[str, ...]:
    universe = set(table.universe())
    annotated = set(snapshot.annotations)
    return tuple(sorted(universe & annotated))


def build_construct(snapshot: KeystoneSnapshot, table: ProgrammeTable, *, seed: int) -> FixtureConstruct:
    genes = candidate_pool(snapshot, table)
    n_genes = len(genes)
    n_programmes = len(table.keys)
    rng = spawn_generators(seed, 8)
    targets = _programme_targets(table, genes)
    latent = rng[0].normal(0.0, 1.0, size=n_genes)
    cohort_loading = rng[1].normal(0.0, 1.0, size=(n_programmes, len(COHORT_ACCESSIONS)))
    population_affinity = rng[2].normal(0.0, 1.0, size=(n_programmes, len(CELL_POPULATIONS)))
    compartment_affinity = rng[3].normal(0.0, 1.0, size=(n_programmes, len(NICHE_COMPARTMENTS)))
    programme_component = targets @ rng[4].normal(0.0, 1.0, size=(n_programmes,))
    reversal_unavailable = rng[5].random(n_genes) < 0.18
    reversal = np.tanh(0.8 * latent + 0.6 * programme_component + rng[6].normal(0.0, 0.6, size=n_genes))
    in_programme = targets.max(axis=1) > (1.0 / n_programmes)
    ranked = latent.copy()
    ranked[~in_programme] = -np.inf
    cutoff = np.quantile(ranked[in_programme], 1.0 - TARGET_QUANTILE)
    targets_flag = in_programme & (ranked >= cutoff)
    development = np.clip(
        DEVELOPMENT_SIGNAL * latent
        + DEVELOPMENT_NOISE * rng[7].normal(0.0, 1.0, size=n_genes)
        + DEVELOPMENT_OFFSET,
        0.0,
        None,
    )
    return FixtureConstruct(
        seed=seed,
        genes=genes,
        programme_targets=targets,
        latent_relevance=latent,
        cohort_loading=cohort_loading,
        population_affinity=population_affinity,
        compartment_affinity=compartment_affinity,
        reversal_index=reversal,
        reversal_available=~reversal_unavailable,
        target_labels=targets_flag,
        development_labels=development,
    )


def cohort_features(construct: FixtureConstruct, *, seed: int) -> np.ndarray:
    """One column per cohort, driven by the programme loading of the gene's programmes."""
    rng = spawn_generators(seed + 11, 1)[0]
    structured = construct.programme_targets @ construct.cohort_loading
    noise = rng.normal(0.0, COHORT_NOISE_SCALE, size=structured.shape)
    features = COHORT_EFFECT_SCALE * structured + noise
    contrast_scale = np.array(
        [1.0 if spec.contrast != "treatment_timecourse" else 0.82 for spec in COHORT_SPECS]
    )
    scaled: np.ndarray = features * contrast_scale
    return zscore(scaled, axis=0)


def cohort_sample_counts() -> np.ndarray:
    return np.array([spec.n_samples_reported for spec in COHORT_SPECS], dtype=np.int64)


def cellstate_features(construct: FixtureConstruct, *, seed: int) -> np.ndarray:
    rng = spawn_generators(seed + 23, 1)[0]
    structured = construct.programme_targets @ construct.population_affinity
    noise = rng.normal(0.0, POPULATION_NOISE_SCALE, size=structured.shape)
    return zscore(structured + noise, axis=0)


def niche_section_features(construct: FixtureConstruct, *, seed: int) -> np.ndarray:
    """One row per spatial section, tagged by the resource it comes from."""
    rng = spawn_generators(seed + 31, 1)[0]
    base = construct.programme_targets @ construct.compartment_affinity
    scale = np.array(
        [
            0.55 if accession == "GSE299256" else 1.0
            for accession in NICHE_ACCESSIONS
            for _ in range(NICHE_SECTIONS_PER_DATASET)
        ]
    )
    noise = rng.normal(0.0, POPULATION_NOISE_SCALE, size=(scale.size, *base.shape))
    block = np.stack([base] * scale.size, axis=0) + noise
    return zscore(block * scale[:, None, None], axis=1)


def niche_section_names() -> tuple[str, ...]:
    names: list[str] = []
    for accession in NICHE_ACCESSIONS:
        prefix = "psoriasis" if accession in {"GSE206391", "GSE310145"} else "reference"
        for index in range(NICHE_SECTIONS_PER_DATASET):
            names.append(f"{prefix}:{accession}:section{index + 1}")
    return tuple(names)


def context_features(
    snapshot: KeystoneSnapshot, genes: tuple[str, ...], *, seed: int, dim: int
) -> np.ndarray:
    """Frozen graph-context embedding: a spectral sketch of the typed interaction graph."""
    rng = spawn_generators(seed + 41, 1)[0]
    position = {symbol: index for index, symbol in enumerate(genes)}
    edges = [
        (position[edge.source], position[edge.target], edge.weight)
        for edge in snapshot.edges
        if edge.source in position and edge.target in position
    ]
    adjacency = np.zeros((len(genes), len(genes)), dtype=np.float64)
    for left, right, weight in edges:
        adjacency[left, right] = max(adjacency[left, right], weight)
        adjacency[right, left] = max(adjacency[right, left], weight)
    sketch = rng.normal(0.0, 1.0, size=(len(genes), dim)) / np.sqrt(dim)
    propagated = adjacency @ sketch
    degree = adjacency.sum(axis=1, keepdims=True)
    normalised = np.divide(propagated, np.maximum(degree, 1e-6))
    return zscore(np.concatenate([sketch, normalised], axis=1), axis=0)


def interaction_degree(
    snapshot: KeystoneSnapshot,
    genes: tuple[str, ...],
    membership: list[list[int]],
    pathway_sizes: list[int],
) -> np.ndarray:
    """Distinct partner genes a candidate can reach, through either family of edges.

    The interaction edges of the snapshot contribute one partner each; a pathway of n
    members contributes the n - 1 genes it shares with the candidate.
    """
    position = {symbol: index for index, symbol in enumerate(genes)}
    degree = np.zeros(len(genes), dtype=np.float64)
    for edge in snapshot.edges:
        for symbol in (edge.source, edge.target):
            index = position.get(symbol)
            if index is not None:
                degree[index] += 1.0
    combined: np.ndarray = degree + co_membership_degree(membership, pathway_sizes)
    return combined


def adjacency_lists(snapshot: KeystoneSnapshot, genes: tuple[str, ...]) -> list[list[tuple[int, str, float]]]:
    position = {symbol: index for index, symbol in enumerate(genes)}
    neighbours: list[list[tuple[int, str, float]]] = [[] for _ in genes]
    for edge in snapshot.edges:
        left = position.get(edge.source)
        right = position.get(edge.target)
        if left is None or right is None or left == right:
            continue
        neighbours[left].append((right, edge.edge_type, edge.weight))
    for row in neighbours:
        row.sort(key=lambda item: (-item[2], item[0], item[1]))
    return neighbours


def tractability_vector(snapshot: KeystoneSnapshot, genes: tuple[str, ...]) -> np.ndarray:
    flags = np.zeros(len(genes), dtype=bool)
    for row, gene in enumerate(genes):
        annotation = snapshot.annotations.get(gene)
        if annotation is not None:
            flags[row] = annotation.tractable
    return flags


def availability_mask(construct: FixtureConstruct, snapshot: KeystoneSnapshot, *, seed: int) -> np.ndarray:
    """One column per evidence layer, with declared missingness.

    A gene absent from the interaction panel has no keystone token, and a declared
    fraction of genes carries no niche token, which is the condition the availability
    mechanism of Sec. 2.3 exists to absorb.
    """
    rng = spawn_generators(seed + 59, 1)[0]
    n_genes = len(construct.genes)
    mask = np.zeros((n_genes, len(MODALITY_ORDER)), dtype=bool)
    panel = {edge.source for edge in snapshot.edges} | {edge.target for edge in snapshot.edges}
    mask[:, 0] = True
    mask[:, 1] = True
    mask[:, 2] = rng.random(n_genes) > 0.12
    mask[:, 3] = np.array([gene in panel for gene in construct.genes])
    mask[~mask.any(axis=1), 0] = True
    return mask


def resource_labels() -> dict[str, tuple[str, ...]]:
    return {
        "cohort": COHORT_ACCESSIONS,
        "cell_state": CELL_STATE_ACCESSIONS,
        "niche": NICHE_ACCESSIONS,
    }
