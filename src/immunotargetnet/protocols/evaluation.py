"""Metric assembly for one configuration.

Ref: Sec. 2.7.2 (recovery, readiness and interpretability metrics) and Table 1 columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from immunotargetnet.evidence.schema import EvidencePanel
from immunotargetnet.metrics.attribution import ScoreFromMask, attribution_faithfulness
from immunotargetnet.metrics.discrimination import (
    area_under_precision_recall,
    area_under_roc,
    expected_calibration_error,
    prevalence_baseline,
)
from immunotargetnet.metrics.divergence import mean_jensen_shannon_divergence, mean_kl_divergence
from immunotargetnet.metrics.ranking import (
    PRIMARY_CUTOFF,
    RECALL_CUTOFF,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from immunotargetnet.metrics.stability import StabilitySummary, kendall_tau_b, stability_over_resamples

METRIC_COLUMNS: tuple[str, ...] = ("NDCG@20", "MRR", "AUROC", "Top-50", "AUPRC", "KL")


@dataclass
class MetricRow:
    method: str
    ndcg20: float
    mrr: float
    auroc: float
    top50: float
    auprc: float
    kl: float
    js: float
    aucdel: float
    tau: float
    ece: float
    prevalence: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def as_table(self) -> dict[str, float]:
        return {
            "NDCG@20": self.ndcg20,
            "MRR": self.mrr,
            "AUROC": self.auroc,
            "Top-50": self.top50,
            "AUPRC": self.auprc,
            "KL": self.kl,
            "JS": self.js,
            "AUCdel": self.aucdel,
            "tau": self.tau,
            "ECE": self.ece,
        }

    def ordering_key(self) -> float:
        return self.ndcg20


@dataclass(frozen=True)
class RankedOutput:
    scores: np.ndarray
    programmes: np.ndarray | None
    readiness: np.ndarray | None
    modality_mass: np.ndarray | None


def readiness_metrics(readiness: np.ndarray, tractability: np.ndarray) -> tuple[float, float, float, float]:
    auprc = area_under_precision_recall(tractability.astype(np.float64), readiness)
    auroc = area_under_roc(tractability.astype(np.float64), readiness)
    ece = expected_calibration_error(readiness, tractability.astype(np.float64))
    return auprc, auroc, ece, prevalence_baseline(tractability.astype(np.float64))


def deletion_faithfulness(modality_mass: np.ndarray | None, score_from_mask: ScoreFromMask) -> float:
    """Drop the layers the model attributed the most mass to, and symmetrically the least.

    Ref: Sec. 2.7.2 - attribution faithfulness under deletion. `score_from_mask` re-runs
    the model with the named layers withheld, so the curve is measured on the model.
    """
    if modality_mass is None:
        return float("nan")
    attribution = np.asarray(modality_mass, dtype=np.float64).mean(axis=0)
    return attribution_faithfulness(attribution, score_from_mask)


def rank_stability(reference_scores: np.ndarray, resampled: list[np.ndarray]) -> StabilitySummary:
    return stability_over_resamples(reference_scores, resampled)


def layer_selection(n_layers: int, indices: tuple[int, ...]) -> np.ndarray:
    """Boolean selection over the layer axis."""
    selection = np.zeros(n_layers, dtype=bool)
    for index in indices:
        selection[index] = True
    if not selection.any():
        raise ValueError("at least one layer must be selected")
    return selection


def layer_mask_from_selection(availability: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Keep the selected layers of an availability mask and nothing else.

    `keep` is a boolean selection over the layer axis; an index array would silently be
    read as a Boolean mask by NumPy's advanced indexing.
    """
    selection = np.asarray(keep, dtype=bool)
    if selection.shape != (availability.shape[1],):
        raise ValueError("the selection must cover every layer exactly once")
    if not selection.any():
        raise ValueError("at least one layer must be kept")
    mask = np.zeros_like(availability, dtype=bool)
    mask[:, selection] = availability[:, selection]
    first = int(np.flatnonzero(selection)[0])
    mask[~mask.any(axis=1), first] = True
    return mask


def build_row(
    *,
    method: str,
    panel: EvidencePanel,
    scores: np.ndarray,
    programmes: np.ndarray | None = None,
    readiness: np.ndarray | None = None,
    tau: float = float("nan"),
    aucdel: float = float("nan"),
) -> MetricRow:
    labels = panel.target_labels.astype(np.float64)
    auprc = float("nan")
    auroc = float("nan")
    ece = float("nan")
    prevalence = float("nan")
    if readiness is not None:
        auprc, auroc, ece, prevalence = readiness_metrics(readiness, panel.tractability)
    kl = float("nan")
    js = float("nan")
    if programmes is not None:
        kl = mean_kl_divergence(programmes, panel.programme_targets)
        js = mean_jensen_shannon_divergence(programmes, panel.programme_targets)
    return MetricRow(
        method=method,
        ndcg20=ndcg_at_k(labels, scores, PRIMARY_CUTOFF),
        mrr=mean_reciprocal_rank(labels, scores),
        auroc=auroc,
        top50=recall_at_k(labels, scores, RECALL_CUTOFF),
        auprc=auprc,
        kl=kl,
        js=js,
        aucdel=aucdel,
        tau=tau,
        ece=ece,
        prevalence=prevalence,
    )


def pairwise_tau(reference: np.ndarray, candidate: np.ndarray) -> float:
    return kendall_tau_b(reference, candidate)


def table_of_rows(rows: list[MetricRow]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        record: dict[str, object] = {"method": row.method}
        record.update({key: round(value, 4) for key, value in row.as_table().items()})
        output.append(record)
    return output
