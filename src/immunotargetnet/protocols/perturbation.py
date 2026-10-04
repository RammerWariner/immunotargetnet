"""Perturbation protocols for the generalisation and stability analyses.

Ref: Sec. 2.7.1 (pathway holdout masks curated-target graph edges) and Sec. 3.4
(three perturbation families: cohort resampling, pathway holdout, modality dropout,
with a random-subset control matched on the number of genes withheld).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from immunotargetnet.runtime.seed import spawn_generators

DROPOUT_GRID: Final[tuple[int, ...]] = (4, 3, 2, 1)
PATHWAY_FAMILIES: Final[tuple[str, ...]] = (
    "programme_membership",
    "population_attribution",
    "compartment_localisation",
    "tractability_annotation",
)


@dataclass(frozen=True)
class HoldoutPlan:
    name: str
    withheld_genes: np.ndarray
    edge_mask: np.ndarray | None
    layer_mask: np.ndarray | None
    n_genes_withheld: int


def pathway_holdout(
    edge_type_index: np.ndarray,
    edge_endpoints: np.ndarray,
    edge_types: tuple[str, ...],
    *,
    n_gene_nodes: int,
    families: tuple[str, ...] = PATHWAY_FAMILIES,
) -> HoldoutPlan:
    """Mask every edge whose type belongs to one of the curated-target pathway families.

    Holdout scores then measure transfer rather than recall of the graph, because the
    mechanism that carried the label during training is removed at inference.
    """
    indices = [edge_types.index(name) for name in families if name in edge_types]
    withheld_types = np.isin(edge_type_index, np.asarray(indices, dtype=np.int64))
    retained = ~withheld_types
    touched = np.unique(edge_endpoints[np.isin(edge_type_index, np.asarray(indices, dtype=np.int64))])
    withheld = np.setdiff1d(np.arange(n_gene_nodes), touched)
    return HoldoutPlan(
        name="pathway_holdout",
        withheld_genes=np.asarray(withheld, dtype=np.int64),
        edge_mask=retained,
        layer_mask=None,
        n_genes_withheld=int(withheld.size),
    )


def random_subset_control(n_genes: int, n_withheld: int, *, seed: int) -> HoldoutPlan:
    """Withhold as many genes at random as the pathway holdout withholds."""
    generator = spawn_generators(seed + 101, 1)[0]
    withheld = np.sort(generator.choice(n_genes, size=min(n_withheld, n_genes), replace=False))
    return HoldoutPlan(
        name="random_subset_control",
        withheld_genes=np.asarray(withheld, dtype=np.int64),
        edge_mask=None,
        layer_mask=None,
        n_genes_withheld=int(withheld.size),
    )


def modality_dropout_plan(
    availability: np.ndarray, *, layers_available: int, context_index: int, seed: int
) -> HoldoutPlan:
    """Keep the first `layers_available` evidence layers plus the keystone slot.

    Ref: Sec. 3.4 - the degradation curve is the out-of-distribution counterpart of the
    in-distribution gain that removing modality dropout produces.
    """
    if layers_available < 1:
        raise ValueError("at least one evidence layer must remain")
    layer_columns = [index for index in range(availability.shape[1]) if index != context_index]
    keep_layers = layer_columns[:layers_available]
    mask = np.zeros_like(availability, dtype=bool)
    mask[:, keep_layers] = availability[:, keep_layers]
    mask[:, context_index] = availability[:, context_index]
    empty = ~mask.any(axis=1)
    mask[empty, context_index] = True
    return HoldoutPlan(
        name=f"modality_dropout_{layers_available}",
        withheld_genes=np.zeros(0, dtype=np.int64),
        edge_mask=None,
        layer_mask=mask,
        n_genes_withheld=0,
    )


def cohort_resample_blocks(
    genes: np.ndarray, cohort_index: np.ndarray, *, seed: int, n_resamples: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Resample whole genes, reordering the cohort columns each draw.

    The unit of resampling is the gene in a cohort (Sec. 2.7.3), so a draw keeps every
    cohort observation of a selected gene together.
    """
    generator = spawn_generators(seed + 211, 1)[0]
    unique = np.unique(genes)
    draws: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_resamples):
        chosen = generator.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([np.flatnonzero(genes == gene) for gene in chosen])
        draws.append((rows, cohort_index[rows]))
    return draws


def cohort_holdout_curve(folds: int) -> np.ndarray:
    return np.arange(0, folds, dtype=np.float64)
