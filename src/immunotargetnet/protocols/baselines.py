"""The seven comparison configurations of Table 1.

Ref: Sec. 2.4 (baseline families) and Table 1 (B1-B7).

B7 is the unconstrained optimum of the same model family: the simplex projection is
replaced by a free-form linear projection while the encoders, the graph tower and the two
heads are kept, which is the reference the parity claim is made against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from immunotargetnet.evidence.schema import EvidencePanel
from immunotargetnet.metrics.ranking import ranking_order
from immunotargetnet.runtime.numeric import cosine_similarity_matrix, zscore

BASELINE_KEYS: Final[tuple[str, ...]] = (
    "differential_expression",
    "coexpression_modules",
    "network_propagation",
    "latent_factor_integration",
    "foundation_encoder_similarity",
    "typed_graph_ranking",
    "unconstrained_variant",
)

BASELINE_LABELS: Final[dict[str, str]] = {
    "differential_expression": "Differential expression",
    "coexpression_modules": "Co-expression modules",
    "network_propagation": "Network propagation",
    "latent_factor_integration": "Latent-factor integration",
    "foundation_encoder_similarity": "Foundation-encoder similarity",
    "typed_graph_ranking": "Typed graph ranking",
    "unconstrained_variant": "Unconstrained variant",
}

FRAMEWORK_LABEL: Final[str] = "ImmunoTargetNet (constrained)"
FRAMEWORK_ROW: Final[str] = "immunotargetnet"


def differential_expression_scores(panel: EvidencePanel) -> np.ndarray:
    """B1: absolute pooled contrast magnitude on the cohort axis."""
    score: np.ndarray = np.abs(panel.cohort_features).mean(axis=1)
    return score


def coexpression_module_scores(panel: EvidencePanel) -> np.ndarray:
    """B2: module eigengene agreement between the cohort and cell-state blocks.

    The module structure is taken from the correlation of the cohort block with itself and
    the eigengene is scored by its agreement with the strongest contrast.
    """
    block = np.asarray(panel.cohort_features, dtype=np.float64)
    if block.shape[1] < 2:
        degenerate: np.ndarray = np.abs(block).mean(axis=1)
        return degenerate
    correlation = np.corrcoef(block.T)
    _eigenvalues, eigenvectors = np.linalg.eigh(np.nan_to_num(correlation))
    leading = eigenvectors[:, -1]
    eigengene = zscore(block @ leading)
    contrast = np.abs(panel.cohort_features).sum(axis=1)
    module_score: np.ndarray = np.abs(eigengene) * (0.5 + 0.5 * zscore(contrast))
    return module_score


def network_propagation_scores(panel: EvidencePanel, *, restart: float = 0.5, steps: int = 8) -> np.ndarray:
    """B3: random walk with restart over the interaction graph, seeded by the contrast."""
    n_genes = panel.n_genes
    adjacency = np.zeros((n_genes, n_genes), dtype=np.float64)
    for row, neighbours in enumerate(panel.adjacency):
        for column, _edge, weight in neighbours:
            adjacency[row, column] = max(adjacency[row, column], weight)
    degree = adjacency.sum(axis=1, keepdims=True)
    transition = np.divide(adjacency, np.maximum(degree, 1e-12))
    seed = np.abs(panel.cohort_features).mean(axis=1)
    seed = seed / max(seed.sum(), 1e-12)
    state = seed.copy()
    for _ in range(steps):
        state = restart * seed + (1.0 - restart) * (transition.T @ state)
    propagated: np.ndarray = state
    return propagated


def latent_factor_scores(panel: EvidencePanel, *, components: int = 8) -> np.ndarray:
    """B4: multi-view factorisation over the concatenated evidence blocks."""
    blocks = [
        zscore(np.asarray(panel.cohort_features, dtype=np.float64), axis=0),
        zscore(np.asarray(panel.cellstate_features, dtype=np.float64), axis=0),
        zscore(np.asarray(panel.niche_features, dtype=np.float64), axis=0),
    ]
    stacked = np.concatenate(blocks, axis=1)
    stacked = np.nan_to_num(stacked)
    n_components = min(components, min(stacked.shape) - 1)
    if n_components < 1:
        degenerate: np.ndarray = np.abs(stacked).mean(axis=1)
        return degenerate
    centred = stacked - stacked.mean(axis=0, keepdims=True)
    _u, singular, components_matrix = np.linalg.svd(centred, full_matrices=False)
    loadings = components_matrix[:n_components]
    factors = centred @ loadings.T
    weights = singular[:n_components] / max(float(singular[:n_components].sum()), 1e-12)
    factor_score: np.ndarray = np.abs(factors * weights).sum(axis=1)
    return factor_score


def foundation_encoder_scores(panel: EvidencePanel) -> np.ndarray:
    """B5: agreement between the frozen encoder features and the programme vocabulary.

    The features are mapped onto the programme axis by least squares against the
    programme table, and a gene scores by the cosine agreement of that projection with the
    nearest programme direction.
    """
    features = np.nan_to_num(
        np.concatenate(
            [
                zscore(np.asarray(panel.cohort_features, dtype=np.float64), axis=0),
                zscore(np.asarray(panel.cellstate_features, dtype=np.float64), axis=0),
                zscore(np.asarray(panel.niche_features, dtype=np.float64), axis=0),
            ],
            axis=1,
        )
    )
    design = np.column_stack([features, np.ones(features.shape[0])])
    coefficients, *_ = np.linalg.lstsq(design, panel.programme_targets, rcond=None)
    projected = design @ coefficients
    similarity = cosine_similarity_matrix(projected, np.eye(panel.n_programmes))
    encoder_score: np.ndarray = np.asarray(similarity.max(axis=1), dtype=np.float64)
    return encoder_score


def typed_graph_only_scores(panel: EvidencePanel, *, steps: int = 6) -> np.ndarray:
    """B6: the keystone layer alone, aggregated over the typed graph."""
    n_genes = panel.n_genes
    degree = np.maximum(panel.interaction_degree, 1.0)
    seed = zscore(np.log1p(panel.interaction_degree))
    adjacency = np.zeros((n_genes, n_genes), dtype=np.float64)
    for row, neighbours in enumerate(panel.adjacency):
        for column, _edge, weight in neighbours:
            adjacency[row, column] = max(adjacency[row, column], weight)
    transition = np.divide(adjacency, np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-12))
    state = seed.copy()
    for _ in range(steps):
        state = 0.3 * seed + 0.7 * (transition.T @ state)
    graph_score: np.ndarray = np.asarray(state + 0.1 * np.log1p(degree), dtype=np.float64)
    return graph_score


@dataclass(frozen=True)
class BaselineScores:
    ordering: tuple[str, ...]
    scores: dict[str, np.ndarray]

    def get(self, key: str) -> np.ndarray:
        return self.scores[key]

    def rankings(self) -> dict[str, np.ndarray]:
        return {key: ranking_order(value) for key, value in self.scores.items()}


def baseline_scores(panel: EvidencePanel, *, upstream: dict[str, np.ndarray] | None = None) -> BaselineScores:
    """Score every baseline; B7 and the framework come from the trained model."""
    values: dict[str, np.ndarray] = {
        "differential_expression": differential_expression_scores(panel),
        "coexpression_modules": coexpression_module_scores(panel),
        "network_propagation": network_propagation_scores(panel),
        "latent_factor_integration": latent_factor_scores(panel),
        "foundation_encoder_similarity": foundation_encoder_scores(panel),
        "typed_graph_ranking": typed_graph_only_scores(panel),
    }
    if upstream:
        for key in ("unconstrained_variant", FRAMEWORK_ROW):
            if key in upstream:
                values[key] = np.asarray(upstream[key], dtype=np.float64)
    return BaselineScores(ordering=BASELINE_KEYS, scores=values)
