"""Edge-type-aware message passing over the typed knowledge graph.

Ref: Eq. (4) and Algorithm 2, steps 2-7.

Weights across edge types live on the simplex and the two rounds are the whole depth the
manuscript reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from immunotargetnet.bottleneck.attention import BottleneckConfig


def segment_softmax(logits: Tensor, segment: Tensor, n_segments: int) -> Tensor:
    """Softmax of `logits` within each segment index."""
    if logits.numel() == 0:
        return logits
    maxima = torch.full(
        (n_segments,), float("-inf"), dtype=logits.dtype, device=logits.device
    ).scatter_reduce(0, segment, logits, reduce="amax", include_self=True)
    shifted = logits - maxima[segment]
    weights = torch.exp(shifted)
    totals = torch.zeros(n_segments, dtype=logits.dtype, device=logits.device)
    totals = totals.index_add(0, segment, weights)
    normalised: Tensor = weights / totals[segment].clamp(min=1e-12)
    return normalised


@dataclass(frozen=True)
class TypedGraphBatch:
    """Flat representation of a typed multi-relational graph."""

    node_state: Tensor
    edge_source: Tensor
    edge_target: Tensor
    edge_type_index: Tensor
    edge_log_weight: Tensor
    edge_type_names: tuple[str, ...]
    n_nodes: int

    @property
    def n_edge_types(self) -> int:
        return len(self.edge_type_names)


class TypedGraphTower(nn.Module):
    """Two rounds of edge-type-aware aggregation with simplex type weights."""

    def __init__(self, node_dim: int, n_edge_types: int, config: BottleneckConfig) -> None:
        super().__init__()
        self.config = config
        self.n_edge_types = n_edge_types
        self.type_weight_logits = nn.Parameter(torch.zeros(n_edge_types))
        self.transform = nn.ModuleList(nn.Linear(node_dim, node_dim, bias=False) for _ in range(n_edge_types))
        self.attention = nn.ModuleList(nn.Linear(2 * node_dim, 1, bias=False) for _ in range(n_edge_types))
        self.activation = nn.Sigmoid()
        self.depth = config.n_graph_rounds

    def type_weights(self) -> Tensor:
        return torch.softmax(self.type_weight_logits, dim=-1)

    def _message(self, batch: TypedGraphBatch) -> Tensor:
        weights = self.type_weights()
        aggregated = torch.zeros_like(batch.node_state)
        for type_index in range(self.n_edge_types):
            selected = batch.edge_type_index == type_index
            if not bool(selected.any()):
                continue
            source = batch.edge_source[selected]
            target = batch.edge_target[selected]
            projected = self.transform[type_index](batch.node_state)
            pair = torch.cat([projected[target], projected[source]], dim=-1)
            logit = self.attention[type_index](pair).squeeze(-1)
            logit = torch.nn.functional.leaky_relu(logit, negative_slope=0.2)
            logit = logit + batch.edge_log_weight[selected]
            coefficient = segment_softmax(logit, target, batch.n_nodes)
            contribution = coefficient.unsqueeze(-1) * projected[source]
            aggregated = aggregated.index_add(0, target, contribution * weights[type_index])
        activated: Tensor = self.activation(aggregated)
        return activated

    def forward(self, batch: TypedGraphBatch) -> Tensor:
        state = batch.node_state
        for _ in range(self.depth):
            updated = self._message(
                TypedGraphBatch(
                    node_state=state,
                    edge_source=batch.edge_source,
                    edge_target=batch.edge_target,
                    edge_type_index=batch.edge_type_index,
                    edge_log_weight=batch.edge_log_weight,
                    edge_type_names=batch.edge_type_names,
                    n_nodes=batch.n_nodes,
                )
            )
            state = updated
        return state


def masked_graph_edges(batch: TypedGraphBatch, mask: Tensor) -> TypedGraphBatch:
    """Remove the edges a holdout perturbation withholds, leaving node count untouched."""
    keep = mask.to(dtype=torch.bool)
    return TypedGraphBatch(
        node_state=batch.node_state,
        edge_source=batch.edge_source[keep],
        edge_target=batch.edge_target[keep],
        edge_type_index=batch.edge_type_index[keep],
        edge_log_weight=batch.edge_log_weight[keep],
        edge_type_names=batch.edge_type_names,
        n_nodes=batch.n_nodes,
    )
