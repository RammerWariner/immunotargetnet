"""Assembly of the typed graph into the flat batch the tower consumes.

Ref: Algorithm 2, step 1 (gene nodes initialised with the bottleneck state, annotation
nodes with frozen embeddings).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from torch import Tensor

from immunotargetnet.evidence.schema import EvidencePanel
from immunotargetnet.graph.message_passing import TypedGraphBatch

MEMBERSHIP_EDGE: Final[str] = "programme_membership"
POPULATION_EDGE: Final[str] = "population_attribution"
COMPARTMENT_EDGE: Final[str] = "compartment_localisation"
TRACTABILITY_EDGE: Final[str] = "tractability_annotation"
PATHWAY_EDGE: Final[str] = "pathway_membership"

# The annotation families carry the curated gene-set and tractability knowledge; the
# pathway holdout of Sec. 2.7.1 removes exactly these.
ANNOTATION_EDGE_TYPES: Final[tuple[str, ...]] = (
    MEMBERSHIP_EDGE,
    PATHWAY_EDGE,
    POPULATION_EDGE,
    COMPARTMENT_EDGE,
    TRACTABILITY_EDGE,
)


@dataclass(frozen=True)
class GraphAssembly:
    batch: TypedGraphBatch
    n_gene_nodes: int
    n_annotation_nodes: int
    node_labels: tuple[str, ...]


def _frozen_annotation_state(annotation_index: int, node_dim: int, *, seed: int = 20260101) -> np.ndarray:
    generator = np.random.default_rng(seed + annotation_index)
    vector = generator.normal(0.0, 1.0, size=node_dim)
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float64)


def build_graph_batch(
    panel: EvidencePanel,
    node_state: Tensor,
    *,
    interaction_edge_types: tuple[str, ...],
    membership_threshold: float = 0.05,
    attribution_threshold: float = 0.75,
    include_annotation_nodes: bool = True,
) -> GraphAssembly:
    """Build the flat graph: gene-gene interactions plus gene-to-annotation edges."""
    n_genes = panel.n_genes
    n_programmes = panel.n_programmes
    population_count = len(panel.cellstate_feature_names)
    compartment_count = len(panel.niche_feature_names)
    modality_count = panel.availability.shape[1]

    annotation_labels: list[str] = []
    if include_annotation_nodes:
        annotation_labels.extend(f"programme::{name}" for name in panel.programme_labels)
        annotation_labels.extend(panel.pathway_labels)
        annotation_labels.extend(f"population::{name}" for name in panel.cellstate_feature_names)
        annotation_labels.extend(f"compartment::{name}" for name in panel.niche_feature_names)
        annotation_labels.extend(f"tractability::{index}" for index in range(modality_count))
    n_annotation = len(annotation_labels)
    n_nodes = n_genes + n_annotation

    edge_names: list[str] = list(interaction_edge_types)
    for name in (
        MEMBERSHIP_EDGE,
        PATHWAY_EDGE,
        POPULATION_EDGE,
        COMPARTMENT_EDGE,
        TRACTABILITY_EDGE,
    ):
        if name not in edge_names:
            edge_names.append(name)

    sources: list[int] = []
    targets: list[int] = []
    type_indices: list[int] = []
    log_weights: list[float] = []

    def add(source: int, target: int, name: str, weight: float) -> None:
        sources.append(source)
        targets.append(target)
        type_indices.append(edge_names.index(name))
        log_weights.append(float(np.log(max(weight, 1e-3))))

    for index, neighbours in enumerate(panel.adjacency):
        for neighbour, edge_type, weight in neighbours:
            add(index, neighbour, edge_type, weight)

    if include_annotation_nodes:
        programme_base = n_genes
        pathway_base = programme_base + n_programmes
        population_base = pathway_base + len(panel.pathway_labels)
        compartment_base = population_base + population_count
        modality_base = compartment_base + compartment_count
        for row in range(n_genes):
            for column in range(n_programmes):
                weight = float(panel.programme_targets[row, column])
                if weight >= membership_threshold:
                    add(row, programme_base + column, MEMBERSHIP_EDGE, weight)
            for pathway_index in panel.pathway_membership[row]:
                add(row, pathway_base + pathway_index, PATHWAY_EDGE, 0.5)
            for column in range(population_count):
                score = float(panel.cellstate_features[row, column]) if panel.cellstate_features.size else 0.0
                if score >= attribution_threshold:
                    add(row, population_base + column, POPULATION_EDGE, 0.5 + 0.1 * score)
            for column in range(compartment_count):
                score = float(panel.niche_features[row, column]) if panel.niche_features.size else 0.0
                if score >= attribution_threshold:
                    add(row, compartment_base + column, COMPARTMENT_EDGE, 0.5 + 0.1 * score)
            if panel.tractability[row]:
                add(row, modality_base, TRACTABILITY_EDGE, 1.0)

    state = np.asarray(node_state.detach().cpu().numpy(), dtype=np.float64)
    node_dim = state.shape[1]
    block = np.zeros((n_nodes, node_dim), dtype=np.float64)
    block[:n_genes] = state
    for offset in range(n_annotation):
        block[n_genes + offset] = _frozen_annotation_state(offset, node_dim)

    batch = TypedGraphBatch(
        node_state=torch.as_tensor(block, dtype=torch.float32),
        edge_source=torch.as_tensor(np.asarray(sources, dtype=np.int64), dtype=torch.long),
        edge_target=torch.as_tensor(np.asarray(targets, dtype=np.int64), dtype=torch.long),
        edge_type_index=torch.as_tensor(np.asarray(type_indices, dtype=np.int64), dtype=torch.long),
        edge_log_weight=torch.as_tensor(np.asarray(log_weights, dtype=np.float64), dtype=torch.float32),
        edge_type_names=tuple(edge_names),
        n_nodes=n_nodes,
    )
    labels = tuple([f"gene::{gene}" for gene in panel.genes] + annotation_labels)
    return GraphAssembly(
        batch=batch,
        n_gene_nodes=n_genes,
        n_annotation_nodes=n_annotation,
        node_labels=labels,
    )


def with_gene_states(assembly: GraphAssembly, panel: EvidencePanel, state: Tensor) -> TypedGraphBatch:
    """Replace the gene-node block, keeping annotation embeddings frozen."""
    block = assembly.batch.node_state.clone()
    block[: panel.n_genes] = state
    return TypedGraphBatch(
        node_state=block,
        edge_source=assembly.batch.edge_source,
        edge_target=assembly.batch.edge_target,
        edge_type_index=assembly.batch.edge_type_index,
        edge_log_weight=assembly.batch.edge_log_weight,
        edge_type_names=assembly.batch.edge_type_names,
        n_nodes=assembly.batch.n_nodes,
    )


def interaction_edge_types(panel: EvidencePanel) -> tuple[str, ...]:
    names = sorted({edge_type for neighbours in panel.adjacency for _, edge_type, _ in neighbours})
    return tuple(names)


def edge_keep_excluding_nodes(batch: TypedGraphBatch, blocked: np.ndarray) -> Tensor:
    """Keep every edge that touches no blocked node.

    The mask is built over the assembled edge list, so it stays aligned when the graph
    carries several annotation families.
    """
    blocked_flags = torch.as_tensor(np.asarray(blocked, dtype=bool), dtype=torch.bool)
    if blocked_flags.numel() != batch.n_nodes:
        raise ValueError("the blocked array must cover every node of the batch")
    if not bool(blocked_flags.any()):
        return torch.ones(batch.edge_source.numel(), dtype=torch.bool)
    touched = blocked_flags[batch.edge_source] | blocked_flags[batch.edge_target]
    return ~touched


def node_degrees(assembly: GraphAssembly) -> np.ndarray:
    counted = np.bincount(
        np.asarray(assembly.batch.edge_target.detach().cpu().numpy(), dtype=np.int64),
        minlength=assembly.batch.n_nodes,
    )
    return np.asarray(counted, dtype=np.float64)
