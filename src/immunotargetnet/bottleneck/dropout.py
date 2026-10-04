"""Modality dropout over the availability mask.

Ref: Sec. 2.3 (a chosen non-empty section of the mask is zeroed at every training step).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from immunotargetnet.runtime.seed import spawn_generators


@dataclass(frozen=True)
class DropoutPlan:
    mask: np.ndarray
    dropped_fraction: float
    kept_columns: tuple[int, ...]


def modality_dropout(availability: np.ndarray, rate: float, *, seed: int, step: int) -> DropoutPlan:
    """Zero a non-empty subset of each gene's available tokens.

    The mask is guaranteed to keep at least one token per gene: the mechanism exists to
    test one-layer inference, and an all-zero row would leave the gene undefined rather
    than sparse.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError("dropout rate must lie in [0, 1)")
    mask = availability.copy()
    if rate == 0.0:
        return DropoutPlan(mask=mask, dropped_fraction=0.0, kept_columns=_kept(mask))
    rng = spawn_generators(seed + step, 1)[0]
    n_genes, n_tokens = mask.shape
    draw = rng.random((n_genes, n_tokens))
    drop = (draw < rate) & mask
    surviving = mask & ~drop
    emptied = ~surviving.any(axis=1) & mask.any(axis=1)
    if emptied.any():
        survivors = np.argwhere(mask[emptied])
        first = survivors[np.unique(survivors[:, 0], return_index=True)[1]]
        surviving[emptied, first[:, 1]] = True
    dropped = int((mask & ~surviving).sum())
    total = int(mask.sum())
    return DropoutPlan(
        mask=surviving,
        dropped_fraction=0.0 if total == 0 else dropped / total,
        kept_columns=_kept(surviving),
    )


def _kept(mask: np.ndarray) -> tuple[int, ...]:
    counts = mask.sum(axis=0)
    return tuple(int(index) for index in np.flatnonzero(counts))


def available_layer_count(mask: np.ndarray, *, context_index: int) -> np.ndarray:
    """Number of evidence layers available per gene, excluding the keystone slot."""
    layer_columns = [index for index in range(mask.shape[1]) if index != context_index]
    counts: np.ndarray = mask[:, layer_columns].sum(axis=1)
    return counts
