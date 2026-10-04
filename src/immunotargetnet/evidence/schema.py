"""Evidence-layer schema shared by the readers, the encoders and the model.

Ref: Sec. 2.1 (gene tuple, modality set, availability mask, programme simplex) and
Sec. 2.2 (resource provenance).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

MODALITY_ORDER: Final[tuple[str, ...]] = ("cohort", "cell_state", "niche", "keystone")
# The keystone layer is realised as the graph-context token (Sec. 2.3), so the
# contextual token is the last of the four layer slots rather than a fifth token.
CONTEXT_TOKEN_INDEX: Final[int] = len(MODALITY_ORDER) - 1

SCHEMA_VERSION: Final[str] = "itn-keystone-1"

# Programme keys use the twelve names of Sec. 2.4; the display strings are the ones a
# reader sees in the shortlist dossier.
PROGRAMME_KEYS: Final[tuple[str, ...]] = (
    "myeloid_antigen_presentation",
    "type17_effector_output",
    "type1_effector_output",
    "type2_effector_output",
    "type17_associated_epithelial_response",
    "interleukin_36_alarm_signalling",
    "innate_pattern_recognition_priming",
    "cytokine_signal_transduction",
    "neutrophil_activation",
    "keratinocyte_myeloid_chemotaxis",
    "epidermal_stress_response",
    "interferon_response",
)

PROGRAMME_DISPLAY: Final[dict[str, str]] = {
    "myeloid_antigen_presentation": "Myeloid antigen presentation",
    "type17_effector_output": "Type 17 effector output",
    "type1_effector_output": "Type 1 effector output",
    "type2_effector_output": "Type 2 effector output",
    "type17_associated_epithelial_response": "Type 17-associated epithelial response",
    "interleukin_36_alarm_signalling": "Interleukin-36 alarm signalling",
    "innate_pattern_recognition_priming": "Innate pattern recognition",
    "cytokine_signal_transduction": "Cytokine signal transduction",
    "neutrophil_activation": "Neutrophil activation",
    "keratinocyte_myeloid_chemotaxis": "Keratinocyte-myeloid chemotaxis",
    "epidermal_stress_response": "Epidermal stress response",
    "interferon_response": "Interferon response",
}


# Clinical-stage buckets of the platform's tractability schema; a gene is readiness-positive
# when at least one of its modalities reaches one of these.
CLINICAL_TRACTABILITY_LABELS: Final[frozenset[str]] = frozenset(
    {"Approved Drug", "Advanced Clinical", "Phase 1 Clinical"}
)


@dataclass(frozen=True)
class TractabilityBucket:
    modality: str
    label: str
    value: bool


@dataclass(frozen=True)
class GeneAnnotation:
    symbol: str
    ensembl_id: str
    biotype: str
    tractability: tuple[TractabilityBucket, ...] = ()
    is_essential: bool = False

    @property
    def tractable(self) -> bool:
        return any(bucket.value for bucket in self.tractability)

    @property
    def clinical_stage(self) -> bool:
        """Readiness label of Sec. 2.1: a clinical-stage tractability annotation exists."""
        return any(
            bucket.value and bucket.label in CLINICAL_TRACTABILITY_LABELS for bucket in self.tractability
        )

    @property
    def tractable_buckets(self) -> int:
        return sum(1 for bucket in self.tractability if bucket.value)


@dataclass(frozen=True)
class TypedEdge:
    source: str
    target: str
    edge_type: str
    weight: float
    source_database: str


@dataclass(frozen=True)
class ResourceProvenance:
    role: str
    identifier: str
    version: str
    licence: str
    access_url: str


@dataclass(frozen=True)
class ProgrammeDefinition:
    key: str
    display_name: str
    collections: tuple[str, ...]


@dataclass
class EvidencePanel:
    """Everything the model reads for one candidate pool."""

    genes: list[str]
    cohort_feature_names: list[str]
    cellstate_feature_names: list[str]
    niche_feature_names: list[str]
    cohort_features: NDArray[np.float64]
    cohort_sample_counts: NDArray[np.int64]
    cellstate_features: NDArray[np.float64]
    niche_features: NDArray[np.float64]
    context_features: NDArray[np.float64]
    availability: NDArray[np.bool_]
    programme_targets: NDArray[np.float64]
    programme_labels: list[str]
    tractability: NDArray[np.bool_]
    target_labels: NDArray[np.bool_]
    development_labels: NDArray[np.float64]
    reversal_index: NDArray[np.float64]
    reversal_available: NDArray[np.bool_]
    interaction_degree: NDArray[np.float64]
    modality_attribution_reference: NDArray[np.float64]
    adjacency: list[list[tuple[int, str, float]]] = field(default_factory=list)
    pathway_labels: list[str] = field(default_factory=list)
    pathway_membership: list[list[int]] = field(default_factory=list)
    pathway_sizes: list[int] = field(default_factory=list)
    cohort_accessions: list[str] = field(default_factory=list)
    source_note: str = ""

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    @property
    def n_programmes(self) -> int:
        return len(self.programme_labels)

    @property
    def n_tokens(self) -> int:
        return int(self.availability.shape[1])


@dataclass(frozen=True)
class CohortLayer:
    accession: str
    contrast: str
    platform: str
    release_year: int
    n_samples: int
    samples: NDArray[np.float64]
    gene_symbols: tuple[str, ...]


@dataclass(frozen=True)
class EdgeTypeVocabulary:
    names: tuple[str, ...]

    def index(self, name: str) -> int:
        return self.names.index(name)
