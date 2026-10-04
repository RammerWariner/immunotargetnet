"""Frozen per-layer encoders.

Ref: Sec. 2.3 (one pre-trained encoder per evidence layer, frozen as an intentional
decision) and Algorithm 1 line 4.

Each encoder is a parameter-free feature map over a modality; the trainable projection
that follows it lives in the bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from torch import Tensor, nn

from immunotargetnet.evidence.schema import EvidencePanel


class FrozenEncoder(Protocol):
    name: str
    feature_names: list[str]
    context_encoder: bool

    def width(self) -> int: ...

    def encode(self, panel: EvidencePanel) -> Tensor: ...


def _as_tensor(block: np.ndarray) -> Tensor:
    return torch.as_tensor(np.asarray(block, dtype=np.float64), dtype=torch.float32)


@dataclass
class CohortContrastEncoder:
    """Per-cohort moderated contrast, weighted by the precision of each cohort.

    Ref: Sec. 2.2 (variance-stabilising within-cohort contrast, inverse-variance pooling).
    """

    name: str = "cohort"
    context_encoder: bool = False
    feature_names: list[str] = field(default_factory=list)

    def width(self) -> int:
        return len(self.feature_names or [])

    def encode(self, panel: EvidencePanel) -> Tensor:
        block = _as_tensor(panel.cohort_features)
        counts = _as_tensor(panel.cohort_sample_counts.astype(np.float64)).clamp(min=1.0)
        precision = counts / counts.sum()
        return block * torch.sqrt(precision).unsqueeze(0)


@dataclass
class CellStateEncoder:
    """Population-resolved signature scores; the score itself is rank-based and frozen.

    Ref: Sec. 2.2 (three single-cell resources) and Sec. 2.4 (gene-based scoring).
    """

    name: str = "cell_state"
    context_encoder: bool = False
    feature_names: list[str] = field(default_factory=list)

    def width(self) -> int:
        return len(self.feature_names or [])

    def encode(self, panel: EvidencePanel) -> Tensor:
        return _as_tensor(panel.cellstate_features)


@dataclass
class NicheEncoder:
    """Compartment localisation over the psoriasis-supported spatial sections.

    Ref: Sec. 2.2 (three spatial resources with associated histology).
    """

    name: str = "niche"
    context_encoder: bool = False
    feature_names: list[str] = field(default_factory=list)

    def width(self) -> int:
        return len(self.feature_names or [])

    def encode(self, panel: EvidencePanel) -> Tensor:
        return _as_tensor(panel.niche_features)


@dataclass
class GraphContextEncoder:
    """Graph-context embedding for the keystone layer; it also seeds the graph tower.

    Ref: Sec. 2.5 (typed graph context) and Algorithm 1 line 9.
    """

    name: str = "keystone"
    context_encoder: bool = True
    feature_names: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    declared_width: int = 0

    def width(self) -> int:
        if self.embedding is not None:
            return int(self.embedding.shape[1])
        return self.declared_width

    def encode(self, panel: EvidencePanel) -> Tensor:
        source = panel.context_features if self.embedding is None else self.embedding
        return _as_tensor(source)


@dataclass
class FrozenEncoderSuite:
    encoders: tuple[FrozenEncoder, ...]

    @classmethod
    def from_panel(cls, panel: EvidencePanel, *, context_dim: int) -> FrozenEncoderSuite:
        encoders: tuple[FrozenEncoder, ...] = (
            CohortContrastEncoder(feature_names=list(panel.cohort_feature_names)),
            CellStateEncoder(feature_names=list(panel.cellstate_feature_names)),
            NicheEncoder(feature_names=list(panel.niche_feature_names)),
            GraphContextEncoder(
                feature_names=[f"context_{index}" for index in range(context_dim)],
                embedding=None,
                declared_width=context_dim,
            ),
        )
        return cls(encoders=encoders)

    def modality_encoders(self) -> tuple[FrozenEncoder, ...]:
        return tuple(encoder for encoder in self.encoders if not encoder.context_encoder)

    def context(self) -> FrozenEncoder:
        for encoder in self.encoders:
            if encoder.context_encoder:
                return encoder
        raise LookupError("suite carries no graph-context encoder")

    def widths(self) -> tuple[int, ...]:
        return tuple(encoder.width() for encoder in self.encoders)


class ModalityProjection(nn.Module):
    """Trainable projection of one frozen modality feature map.

    Ref: Algorithm 1 lines 4-5 (W_j E_j(x) + b_j, layer norm, modality-type token).
    """

    def __init__(self, in_width: int, model_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_width, model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.type_token = nn.Parameter(torch.zeros(model_dim))

    def forward(self, features: Tensor) -> Tensor:
        projected: Tensor = self.linear(features)
        normalised: Tensor = self.norm(projected)
        return normalised + self.type_token
