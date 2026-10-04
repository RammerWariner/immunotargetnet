"""Dual decision heads over the shared bottleneck state.

Ref: Algorithm 2, steps 8-9 (shared layer, relevance head, readiness head) and Sec. 2.5.

The two heads share their first layer, which is what makes an unsupported relevance claim
propagate into the readiness output instead of being averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from immunotargetnet.bottleneck.attention import BottleneckConfig


@dataclass(frozen=True)
class HeadOutput:
    relevance: Tensor
    readiness: Tensor
    embedding: Tensor

    def tractability_probability(self) -> Tensor:
        """Probability that the readiness head places a gene in the tractable tier."""
        return self.readiness[:, -1]


class DualHeads(nn.Module):
    def __init__(self, node_dim: int, config: BottleneckConfig, hidden_dim: int = 64) -> None:
        super().__init__()
        self.config = config
        self.shared = nn.Linear(2 * node_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.relevance = nn.Linear(hidden_dim, 1)
        self.readiness = nn.Linear(hidden_dim, config.n_readiness_tiers)

    def forward(self, programmes: Tensor, graph_state: Tensor) -> HeadOutput:
        joint = _concat(programmes, graph_state)
        embedding = self.activation(self.shared(joint))
        relevance = torch.sigmoid(self.relevance(embedding)).squeeze(-1)
        readiness = torch.softmax(self.readiness(embedding), dim=-1)
        return HeadOutput(relevance=relevance, readiness=readiness, embedding=embedding)


def _concat(left: Tensor, right: Tensor) -> Tensor:
    return torch.cat([left, right], dim=-1)
