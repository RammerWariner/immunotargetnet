"""Cohort-level splits and leakage control.

Ref: Sec. 2.7.1 (splits are made by cohort, never by sample; leave-one-cohort-out;
every gene is present in either the training labels or the finalised target set).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from immunotargetnet.runtime.seed import spawn_generators

DEFAULT_VALIDATION_SHARE: Final[float] = 0.15


@dataclass(frozen=True)
class CohortFold:
    index: int
    held_out: str
    training_cohorts: tuple[str, ...]

    @property
    def is_leave_one_out(self) -> bool:
        return len(self.training_cohorts) == len(set(self.training_cohorts))


@dataclass(frozen=True)
class LabelSplit:
    training: np.ndarray
    validation: np.ndarray
    held_out: np.ndarray

    def assert_disjoint(self) -> None:
        overlap = (
            len(np.intersect1d(self.training, self.validation))
            + len(np.intersect1d(self.training, self.held_out))
            + len(np.intersect1d(self.validation, self.held_out))
        )
        if overlap:
            raise AssertionError("label splits share an evaluation unit")


def leave_one_cohort_out(accessions: tuple[str, ...]) -> tuple[CohortFold, ...]:
    """One fold per cohort; the held-out cohort is never part of the training block."""
    folds: list[CohortFold] = []
    for index, accession in enumerate(accessions):
        training = tuple(name for name in accessions if name != accession)
        folds.append(CohortFold(index=index, held_out=accession, training_cohorts=training))
    return tuple(folds)


def label_split(
    n_units: int,
    *,
    seed: int,
    validation_share: float = DEFAULT_VALIDATION_SHARE,
    protected: np.ndarray | None = None,
) -> LabelSplit:
    """Partition evaluation units into training, validation and held-out blocks.

    A protected unit is one that carries a curated-target label; the leakage rule of
    Sec. 2.7.1 requires it to appear in the training labels or in the held-out block, so
    protected units are never dropped from both.
    """
    generator = spawn_generators(seed, 1)[0]
    order = generator.permutation(n_units)
    n_validation = round(validation_share * n_units)
    validation = np.sort(order[:n_validation])
    remainder = order[n_validation:]
    held_out_mask = protected[remainder] if protected is not None else np.zeros(remainder.size, dtype=bool)
    held_out = np.sort(remainder[held_out_mask])
    training = np.sort(remainder[~held_out_mask])
    split = LabelSplit(training=training, validation=validation, held_out=held_out)
    split.assert_disjoint()
    return split


def cohort_axis(fold: CohortFold, accessions: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the held-out and training columns on the cohort axis."""
    held = np.array([accessions.index(fold.held_out)], dtype=np.int64)
    training = np.array([accessions.index(name) for name in fold.training_cohorts], dtype=np.int64)
    return held, training


def verify_no_sample_leakage(fold: CohortFold, sample_cohort: list[str]) -> None:
    """A sample may not appear on both sides of a fold."""
    held = {index for index, name in enumerate(sample_cohort) if name == fold.held_out}
    training = {index for index, name in enumerate(sample_cohort) if name in fold.training_cohorts}
    if held & training:
        raise AssertionError("a sample appears in both the held-out and training cohort")
