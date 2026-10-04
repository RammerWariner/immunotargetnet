"""Degree-matched control for the shortlist.

Ref: Sec. 3.4 (the degree-matched control) and Sec. 3.6 (hub-effect check).

A high-interaction-degree gene reaches the top of a graph ranking without any
disease-specific evidence, so the control re-scores the ranking after removing the
degree-explained share of each score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from immunotargetnet.runtime.numeric import EPS


@dataclass(frozen=True)
class DegreeControl:
    standardised_degree: np.ndarray
    slope: float
    intercept: float
    residual_score: np.ndarray
    degree_then_rank: np.ndarray

    def rank_shift(self, gene_index: int) -> int:
        """Position of a gene under the degree-adjusted ranking, one-based."""
        order = np.argsort(-self.residual_score, kind="stable")
        return int(np.flatnonzero(order == gene_index)[0]) + 1


def standardise_degree(degree: np.ndarray) -> np.ndarray:
    mean = float(degree.mean())
    spread = max(float(degree.std()), EPS)
    return (degree - mean) / spread


def fit_degree_control(scores: np.ndarray, degree: np.ndarray) -> DegreeControl:
    """Regress the score on degree and keep the residual as the degree-adjusted score."""
    standardised = standardise_degree(degree)
    design = np.column_stack([np.ones_like(standardised), standardised])
    coefficients, *_ = np.linalg.lstsq(design, scores, rcond=None)
    intercept = float(coefficients[0])
    slope = float(coefficients[1])
    residual = scores - (intercept + slope * standardised)
    order = np.argsort(-residual, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    return DegreeControl(
        standardised_degree=standardised,
        slope=slope,
        intercept=intercept,
        residual_score=residual,
        degree_then_rank=ranks + 1,
    )


def mean_standardised_degree(degree: np.ndarray, selection: np.ndarray) -> float:
    standardised = standardise_degree(degree)
    return float(standardised[selection].mean())


def degree_gap(degree: np.ndarray, shortlist: np.ndarray, curated: np.ndarray) -> float:
    """Shortlist mean standardised degree minus the curated set's, in standard deviations."""
    standardised = standardise_degree(degree)
    return float(standardised[shortlist].mean() - standardised[curated].mean())


def top_decile_mask(degree: np.ndarray, *, fraction: float = 0.1) -> np.ndarray:
    cutoff = np.quantile(degree, 1.0 - fraction)
    return np.asarray(degree >= cutoff, dtype=bool)
