"""Niche layer: three spatial resources of Sec. 2.2, of which two profile psoriasis.

GSE299256 profiles neutrophilic dermatosis and enters the panel only as a compartment
reference for the dermal neutrophil axis, never as a psoriasis cohort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from immunotargetnet.evidence.markers import NICHE_COMPARTMENTS
from immunotargetnet.evidence.scoring import collapse_populations, signature_score
from immunotargetnet.runtime.numeric import ScalePair, zscore


@dataclass(frozen=True)
class NicheDataset:
    accession: str
    role: str
    platform: str
    release_year: int
    psoriasis_cohort: bool
    licence: str


NICHE_DATASETS: Final[tuple[NicheDataset, ...]] = (
    NicheDataset(
        accession="GSE206391",
        role="spatially resolved landscape of non-communicable inflammatory skin disease",
        platform="GPL24676",
        release_year=2022,
        psoriasis_cohort=True,
        licence="GEO public access, no redistribution restriction",
    ),
    NicheDataset(
        accession="GSE310145",
        role="body-site-resolved spatial profile of psoriasis lesions",
        platform="GPL24676",
        release_year=2026,
        psoriasis_cohort=True,
        licence="GEO public access, no redistribution restriction",
    ),
    NicheDataset(
        accession="GSE299256",
        role="neutrophilic dermatosis spatial reference for the dermal neutrophil axis",
        platform="GPL24676",
        release_year=2025,
        psoriasis_cohort=False,
        licence="GEO public access, no redistribution restriction",
    ),
)

NICHE_ACCESSIONS: Final[tuple[str, ...]] = tuple(dataset.accession for dataset in NICHE_DATASETS)
PSORIASIS_NICHE_ACCESSIONS: Final[tuple[str, ...]] = tuple(
    dataset.accession for dataset in NICHE_DATASETS if dataset.psoriasis_cohort
)


@dataclass(frozen=True)
class NicheLayer:
    compartments: tuple[str, ...]
    genes: tuple[str, ...]
    scores: np.ndarray
    sections: tuple[str, ...]
    datasets_used: tuple[str, ...]

    def collapsed(self) -> np.ndarray:
        return collapse_populations(self.scores)

    def psoriasis_only(self) -> np.ndarray:
        """Compartments supported by the two psoriasis sections, in section order."""
        keep = [index for index, name in enumerate(self.sections) if name.startswith("psoriasis:")]
        if not keep:
            return self.scores
        return self.scores[keep]


def compartment_signatures(genes: tuple[str, ...]) -> dict[str, np.ndarray]:
    position = {symbol: index for index, symbol in enumerate(genes)}
    signatures: dict[str, np.ndarray] = {}
    for compartment, markers in NICHE_COMPARTMENTS.items():
        mask = np.zeros(len(genes), dtype=bool)
        for marker in markers:
            index = position.get(marker)
            if index is not None:
                mask[index] = True
        if mask.any():
            signatures[compartment] = mask
    return signatures


def score_sections(
    sections: np.ndarray,
    genes: tuple[str, ...],
    *,
    section_names: tuple[str, ...],
    datasets_used: tuple[str, ...],
    scale: ScalePair | None = None,
) -> NicheLayer:
    """Score one row per spatial section, keeping the section axis for the holdout."""
    signatures = compartment_signatures(genes)
    blocks = [signature_score(sections, mask) for mask in signatures.values()]
    scored = np.column_stack(blocks) if blocks else np.zeros((sections.shape[0], 0))
    scaled = scale.apply(scored) if scale is not None else scored
    return NicheLayer(
        compartments=tuple(signatures),
        genes=genes,
        scores=np.asarray(scaled, dtype=np.float64),
        sections=section_names,
        datasets_used=datasets_used,
    )


def to_gene_features(layer: NicheLayer, *, psoriasis_only: bool = True) -> np.ndarray:
    """Transpose to one row per gene over the psoriasis-supported sections."""
    block = layer.psoriasis_only() if psoriasis_only else layer.scores
    if block.shape[0] == 0:
        raise ValueError("niche layer carries no section to summarise")
    feature = zscore(block, axis=1).mean(axis=0)
    return np.asarray(feature, dtype=np.float64)
