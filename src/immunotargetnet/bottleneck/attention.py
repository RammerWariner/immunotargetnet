"""Availability-masked cross-modal encoding into the immune-relevance bottleneck.

Ref: Eq. (2)-(3) and Algorithm 1.

The bottleneck is the single route from the encoders to the ranking head, so the rank of
the ranking expression is bounded by the number of immune programmes.

Token layout: one token per evidence layer, in the layer order of Sec. 2.1. Sec. 2.3
states that the keystone layer is realised as the graph-context embedding, so the fourth
token is the graph-context token and Algorithm 1's extra append would duplicate it; the
release keeps four tokens and records the departure in the deviation register.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from immunotargetnet.bottleneck.fusion import FreeFormFusion
from immunotargetnet.encoders.frozen import FrozenEncoderSuite, ModalityProjection

NEGATIVE_INFINITY: float = -1e9


@dataclass(frozen=True)
class BottleneckConfig:
    """Structural constants of the bottleneck.

    `n_programmes` is fixed at twelve by Sec. 2.1; the remaining values are engineering
    defaults exposed in config.
    """

    model_dim: int
    n_programmes: int
    n_heads: int = 4
    n_graph_rounds: int = 2
    n_readiness_tiers: int = 2
    constrained: bool = True


@dataclass(frozen=True)
class BottleneckState:
    programmes: Tensor
    pooled: Tensor
    token_stack: Tensor
    attention: Tensor
    token_valid: Tensor
    modality_mass: Tensor


class AvailabilityMaskedBottleneck(nn.Module):
    """Tokens, masked attention, mean pooling and the simplex projection."""

    def __init__(self, widths: tuple[int, ...], config: BottleneckConfig) -> None:
        super().__init__()
        self.config = config
        self.modality_projections = nn.ModuleList(
            ModalityProjection(width, config.model_dim) for width in widths[:-1]
        )
        self.context_projection = ModalityProjection(widths[-1], config.model_dim)
        self.query = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.key = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.value = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.programme_projection = nn.Linear(config.model_dim, config.n_programmes)
        self.free_form = None if config.constrained else FreeFormFusion(config.model_dim, config.n_programmes)
        self.scale = float(config.model_dim) ** 0.5

    def token_stack(self, features: tuple[Tensor, ...]) -> Tensor:
        columns = [
            projection(feature)
            for projection, feature in zip(self.modality_projections, features[:-1], strict=True)
        ]
        columns.append(self.context_projection(features[-1]))
        return torch.stack(columns, dim=1)

    def forward(self, features: tuple[Tensor, ...], availability: Tensor) -> BottleneckState:
        tokens = self.token_stack(features)
        valid = availability.to(dtype=torch.bool)
        queries = self.query(tokens)
        keys = self.key(tokens)
        values = self.value(tokens)
        logits = torch.matmul(queries, keys.transpose(1, 2)) / self.scale
        # Eq. (2) adds log m to the logits. An unavailable token must contribute neither to
        # the softmax nor to the pooled state, so its logit is driven to -inf.
        mask_term = torch.where(
            valid.unsqueeze(1), torch.zeros_like(logits), torch.full_like(logits, NEGATIVE_INFINITY)
        )
        attention = torch.softmax(logits + mask_term, dim=-1)
        # An unavailable query has no maskable output, so its row is dropped before pooling
        # rather than averaged in with a uniform attention row.
        attended = torch.matmul(attention, values) * valid.unsqueeze(-1).to(dtype=attention.dtype)
        counts = valid.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = attended.sum(dim=1) / counts.to(dtype=attended.dtype)
        if self.free_form is None:
            programmes = torch.softmax(self.programme_projection(pooled), dim=-1)
        else:
            programmes = self.free_form(pooled)
        modality_mass = attention.mean(dim=1) * valid.to(dtype=attention.dtype)
        return BottleneckState(
            programmes=programmes,
            pooled=pooled,
            token_stack=tokens,
            attention=attention,
            token_valid=valid,
            modality_mass=modality_mass,
        )


def availability_tensor(mask: np.ndarray) -> Tensor:
    return torch.as_tensor(np.asarray(mask, dtype=bool), dtype=torch.bool)


def build_bottleneck(widths: tuple[int, ...], config: BottleneckConfig) -> AvailabilityMaskedBottleneck:
    return AvailabilityMaskedBottleneck(widths, config)


def suite_widths(panel: object) -> tuple[int, ...]:
    from immunotargetnet.evidence.schema import EvidencePanel

    if not isinstance(panel, EvidencePanel):
        raise TypeError("panel must be an EvidencePanel")
    suite = FrozenEncoderSuite.from_panel(panel, context_dim=panel.context_features.shape[1])
    return suite.widths()


def layer_mass(state: BottleneckState, layer_index: int) -> Tensor:
    """Total attention mass a gene routes to one evidence layer."""
    return state.modality_mass[:, layer_index]
