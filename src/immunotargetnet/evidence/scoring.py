"""Signature scoring for the cell-state and niche layers.

Ref: Sec. 2.4 (the programmes are scored by the same gene-based method used for
single-cell annotation) and Sec. 2.2 (annotation transfer and compartment localisation).
"""

from __future__ import annotations

import numpy as np

from immunotargetnet.runtime.numeric import EPS

# The published default of the rank-based scorer: the top 2.5% of the ranked
# transcriptome, with a hard cap so a deep library cannot inflate the background.
MAX_RANK_FRACTION: float = 0.025
MAX_RANK: int = 1500


def rank_axis(size: int) -> float:
    """Uncapped rank threshold for a transcriptome of `size` genes."""
    return float(np.ceil(MAX_RANK_FRACTION * size))


def signature_score(expression: np.ndarray, signature: np.ndarray) -> np.ndarray:
    """Score one gene set against a (cells, genes) expression block.

    Ranks are descending, so a highly expressed signature gene carries a small rank and
    drives the U statistic towards zero; the returned score is 1 - U / (n * n_max).
    """
    cells, genes = expression.shape
    if signature.shape != (genes,):
        raise ValueError("signature length does not match the expression width")
    n_signature = int(np.count_nonzero(signature))
    if n_signature == 0:
        return np.zeros(cells, dtype=np.float64)
    descending = (genes + 1) - (
        np.argsort(np.argsort(expression, axis=1, kind="stable"), axis=1, kind="stable") + 1
    )
    capped = np.minimum(descending, min(MAX_RANK, genes)).astype(np.float64)
    rank_sum = capped[:, signature].sum(axis=1)
    u_statistic = rank_sum - n_signature * (n_signature + 1.0) / 2.0
    n_max = max(n_signature, genes - n_signature)
    score = 1.0 - u_statistic / (n_signature * float(n_max))
    clipped: np.ndarray = np.clip(score, 0.0, 1.0)
    return clipped


def signature_scores(expression: np.ndarray, signatures: dict[str, np.ndarray]) -> np.ndarray:
    """Score every signature, returning a (cells, signatures) block in insertion order."""
    if not signatures:
        return np.zeros((expression.shape[0], 0), dtype=np.float64)
    return np.column_stack([signature_score(expression, mask) for mask in signatures.values()])


def presence_score(expression: np.ndarray, signature: np.ndarray) -> np.ndarray:
    """Fraction of a signature's genes detected above the detection floor."""
    n_signature = int(np.count_nonzero(signature))
    if n_signature == 0:
        return np.zeros(expression.shape[0], dtype=np.float64)
    detected = expression[:, signature] > EPS
    fractions: np.ndarray = detected.sum(axis=1) / float(n_signature)
    return fractions


def collapse_populations(per_population: np.ndarray, *, top_fraction: float = 0.25) -> np.ndarray:
    """Reduce a (populations, genes) score block to one value per gene.

    The layer exists to say whether a gene is carried by a population, so the summary is
    the mean of the strongest populations rather than a population average that would
    dilute a gene restricted to a minority compartment.
    """
    populations = per_population.shape[0]
    if populations == 0:
        return np.zeros(per_population.shape[1], dtype=np.float64)
    keep = max(1, int(np.ceil(top_fraction * populations)))
    ordered = np.sort(per_population, axis=0)[::-1]
    collapsed: np.ndarray = ordered[:keep].mean(axis=0)
    return collapsed
