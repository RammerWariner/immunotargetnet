"""Cohort evidence layer: the eight public expression cohorts of Sec. 2.2.

Contrast assignment follows the manuscript: three lesional versus non-lesional cohorts,
four treatment time-course cohorts and one blood pharmacological series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from immunotargetnet.evidence.geo import SeriesMatrix, collate_samples, gene_index
from immunotargetnet.runtime.numeric import (
    EPS,
    moderated_contrast,
    variance_stabilising_transform,
)

LESIONAL_VS_NONLESIONAL: Final[str] = "lesional_vs_nonlesional"
TREATMENT_TIMECOURSE: Final[str] = "treatment_timecourse"
BLOOD_PHARMACOLOGICAL: Final[str] = "blood_pharmacological"


@dataclass(frozen=True)
class CohortSpec:
    accession: str
    contrast: str
    platform: str
    release_year: int
    n_samples_reported: int
    positive_labels: frozenset[str]
    negative_labels: frozenset[str]
    characteristic_key: str


COHORT_SPECS: Final[tuple[CohortSpec, ...]] = (
    CohortSpec(
        accession="GSE53552",
        contrast=LESIONAL_VS_NONLESIONAL,
        platform="GPL570",
        release_year=2013,
        n_samples_reported=99,
        positive_labels=frozenset({"lesional", "lesional skin", "pp"}),
        negative_labels=frozenset({"non-lesional", "non lesional", "nonlesional", "nn"}),
        characteristic_key="disease state",
    ),
    CohortSpec(
        accession="GSE55201",
        contrast=BLOOD_PHARMACOLOGICAL,
        platform="GPL10558",
        release_year=2014,
        n_samples_reported=81,
        positive_labels=frozenset({"baseline", "day 0", "before treatment", "week 0"}),
        negative_labels=frozenset({"week 12", "day 84", "after treatment", "week 4"}),
        characteristic_key="time point",
    ),
    CohortSpec(
        accession="GSE69967",
        contrast=TREATMENT_TIMECOURSE,
        platform="GPL10558",
        release_year=2015,
        n_samples_reported=95,
        positive_labels=frozenset({"baseline", "day 0", "week 0"}),
        negative_labels=frozenset({"week 12", "day 84", "week 4"}),
        characteristic_key="time point",
    ),
    CohortSpec(
        accession="GSE85034",
        contrast=TREATMENT_TIMECOURSE,
        platform="GPL570",
        release_year=2016,
        n_samples_reported=179,
        positive_labels=frozenset({"baseline", "day 0", "week 0"}),
        negative_labels=frozenset({"week 12", "day 84", "week 4"}),
        characteristic_key="time point",
    ),
    CohortSpec(
        accession="GSE30999",
        contrast=LESIONAL_VS_NONLESIONAL,
        platform="GPL570",
        release_year=2012,
        n_samples_reported=170,
        positive_labels=frozenset({"lesional", "lesional skin", "pp"}),
        negative_labels=frozenset({"non-lesional", "non lesional", "nonlesional", "nn"}),
        characteristic_key="disease state",
    ),
    CohortSpec(
        accession="GSE182740",
        contrast=LESIONAL_VS_NONLESIONAL,
        platform="GPL20301",
        release_year=2022,
        n_samples_reported=75,
        positive_labels=frozenset({"lesional", "lesional skin", "pp"}),
        negative_labels=frozenset({"non-lesional", "non lesional", "nonlesional", "nn"}),
        characteristic_key="disease state",
    ),
    CohortSpec(
        accession="GSE201827",
        contrast=TREATMENT_TIMECOURSE,
        platform="GPL24676",
        release_year=2023,
        n_samples_reported=434,
        positive_labels=frozenset({"baseline", "day 0", "week 0", "pre-treatment"}),
        negative_labels=frozenset({"week 12", "day 84", "week 4", "post-treatment"}),
        characteristic_key="time point",
    ),
    CohortSpec(
        accession="GSE212836",
        contrast=TREATMENT_TIMECOURSE,
        platform="GPL24676",
        release_year=2024,
        n_samples_reported=199,
        positive_labels=frozenset({"baseline", "day 0", "week 0", "pre-treatment"}),
        negative_labels=frozenset({"week 12", "day 84", "week 4", "post-treatment"}),
        characteristic_key="time point",
    ),
)

COHORT_ACCESSIONS: Final[tuple[str, ...]] = tuple(spec.accession for spec in COHORT_SPECS)


@dataclass(frozen=True)
class ContrastResult:
    accession: str
    contrast: str
    genes: tuple[str, ...]
    effect: np.ndarray
    standard_error: np.ndarray
    n_positive: int
    n_negative: int


def contrast_from_matrix(matrix: SeriesMatrix, spec: CohortSpec, genes: tuple[str, ...]) -> ContrastResult:
    """Moderated lesional-versus-control contrast for a shared gene panel."""
    positive, negative = collate_samples(
        matrix, spec.characteristic_key, set(spec.positive_labels), set(spec.negative_labels)
    )
    if not positive or not negative:
        raise ValueError(f"{spec.accession}: contrast sides are not populated")
    index = gene_index(matrix.gene_symbols)
    columns = [index[symbol] for symbol in genes if symbol in index]
    present = [symbol for symbol in genes if symbol in index]
    block = np.array(matrix.values, dtype=np.float64)
    block = variance_stabilising_transform(block)
    if not columns:
        return ContrastResult(
            accession=spec.accession,
            contrast=spec.contrast,
            genes=(),
            effect=np.zeros(0, dtype=np.float64),
            standard_error=np.zeros(0, dtype=np.float64),
            n_positive=len(positive),
            n_negative=len(negative),
        )
    selected = block[columns, :]
    group_a = selected[:, positive].T
    group_b = selected[:, negative].T
    effect, standard_error = moderated_contrast(group_a, group_b)
    return ContrastResult(
        accession=spec.accession,
        contrast=spec.contrast,
        genes=tuple(present),
        effect=effect,
        standard_error=standard_error,
        n_positive=len(positive),
        n_negative=len(negative),
    )


def cohort_feature_matrix(contrasts: list[ContrastResult], genes: tuple[str, ...]) -> np.ndarray:
    """Pack one column per cohort, zero-filled where a gene is absent from a platform."""
    features = np.zeros((len(genes), len(contrasts)), dtype=np.float64)
    for column, result in enumerate(contrasts):
        position = {symbol: row for row, symbol in enumerate(genes)}
        for row, symbol in enumerate(result.genes):
            target = position.get(symbol)
            if target is not None:
                features[target, column] = result.effect[row]
    return features


def pooled_effect(contrasts: list[ContrastResult], genes: tuple[str, ...]) -> np.ndarray:
    """Inverse-variance pooled effect per gene; the differential-expression baseline."""
    pooled = np.zeros(len(genes), dtype=np.float64)
    weight_sum = np.zeros(len(genes), dtype=np.float64)
    position = {symbol: row for row, symbol in enumerate(genes)}
    for result in contrasts:
        for row, symbol in enumerate(result.genes):
            target = position.get(symbol)
            if target is None:
                continue
            variance = max(float(result.standard_error[row]) ** 2, EPS)
            pooled[target] += result.effect[row] / variance
            weight_sum[target] += 1.0 / variance
    return np.divide(pooled, np.maximum(weight_sum, EPS))


def contrast_direction(accession: str) -> str:
    for spec in COHORT_SPECS:
        if spec.accession == accession:
            return spec.contrast
    raise KeyError(f"unknown cohort accession: {accession}")
