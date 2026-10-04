"""Cell-state layer: three single-cell skin resources of Sec. 2.2.

The layer answers whether a gene is carried by a population rather than by the average,
so it is stored as a (populations, genes) score block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from immunotargetnet.evidence.markers import CELL_POPULATIONS
from immunotargetnet.evidence.scoring import collapse_populations, signature_score
from immunotargetnet.runtime.numeric import ScalePair, zscore


@dataclass(frozen=True)
class CellStateDataset:
    accession: str
    role: str
    platform: str
    release_year: int
    licence: str


CELL_STATE_DATASETS: Final[tuple[CellStateDataset, ...]] = (
    CellStateDataset(
        accession="GSE254707",
        role="cutaneous dendritic-cell and macrophage populations, lesion and non-lesion skin",
        platform="GPL24676",
        release_year=2023,
        licence="GEO public access, no redistribution restriction",
    ),
    CellStateDataset(
        accession="GSE176509",
        role="epidermal single-cell resource separating keratinocytes from immune cells",
        platform="GPL18573",
        release_year=2024,
        licence="GEO public access, no redistribution restriction",
    ),
    CellStateDataset(
        accession="GSE173706",
        role="paired single-cell and spatial profiling of psoriatic skin",
        platform="GPL24676",
        release_year=2021,
        licence="GEO public access, no redistribution restriction",
    ),
)

CELL_STATE_ACCESSIONS: Final[tuple[str, ...]] = tuple(dataset.accession for dataset in CELL_STATE_DATASETS)


@dataclass(frozen=True)
class CellStateLayer:
    populations: tuple[str, ...]
    genes: tuple[str, ...]
    scores: np.ndarray
    datasets_used: tuple[str, ...]

    def collapsed(self) -> np.ndarray:
        return collapse_populations(self.scores)


def population_signatures(genes: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Map each population onto a boolean mask over the shared gene panel."""
    position = {symbol: index for index, symbol in enumerate(genes)}
    signatures: dict[str, np.ndarray] = {}
    for population, markers in CELL_POPULATIONS.items():
        mask = np.zeros(len(genes), dtype=bool)
        for marker in markers:
            index = position.get(marker)
            if index is not None:
                mask[index] = True
        if mask.any():
            signatures[population] = mask
    return signatures


def score_expression_block(
    expression: np.ndarray, genes: tuple[str, ...], *, scale: ScalePair | None = None
) -> CellStateLayer:
    """Score one expression block per population and standardise across genes."""
    signatures = population_signatures(genes)
    scored = np.column_stack([signature_score(expression, mask) for mask in signatures.values()])
    scaled = scale.apply(scored) if scale is not None else scored
    return CellStateLayer(
        populations=tuple(signatures),
        genes=genes,
        scores=np.asarray(scaled, dtype=np.float64),
        datasets_used=(),
    )


def merge_layers(layers: list[CellStateLayer]) -> CellStateLayer:
    """Average the per-resource score blocks on the union of their populations."""
    if not layers:
        raise ValueError("no cell-state layers to merge")
    genes = layers[0].genes
    populations = sorted({name for layer in layers for name in layer.populations})
    merged = np.zeros((len(populations), len(genes)), dtype=np.float64)
    counts = np.zeros((len(populations), len(genes)), dtype=np.float64)
    for layer in layers:
        rows = {name: index for index, name in enumerate(layer.populations)}
        for target, name in enumerate(populations):
            if name in rows:
                merged[target] += layer.scores[rows[name]]
                counts[target] += 1.0
    merged = np.divide(merged, np.maximum(counts, 1.0))
    return CellStateLayer(
        populations=tuple(populations),
        genes=genes,
        scores=zscore(merged, axis=1),
        datasets_used=tuple(sorted({accession for layer in layers for accession in layer.datasets_used})),
    )


def to_gene_features(layer: CellStateLayer) -> np.ndarray:
    """Transpose the population block to one row per gene, in the layer's own order."""
    return np.asarray(layer.scores.T, dtype=np.float64)
