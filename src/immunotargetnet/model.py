"""The prioritisation model: frozen encoders, bottleneck, typed graph tower, dual heads.

Ref: Algorithm 1 (encoding), Algorithm 2 (graph tower, heads, objective) and Sec. 2.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from immunotargetnet.bottleneck.attention import (
    AvailabilityMaskedBottleneck,
    BottleneckConfig,
    BottleneckState,
    availability_tensor,
    suite_widths,
)
from immunotargetnet.encoders.frozen import FrozenEncoderSuite
from immunotargetnet.evidence.schema import EvidencePanel
from immunotargetnet.graph.assembly import (
    GraphAssembly,
    build_graph_batch,
    interaction_edge_types,
    with_gene_states,
)
from immunotargetnet.graph.heads import DualHeads
from immunotargetnet.graph.message_passing import TypedGraphBatch, TypedGraphTower, masked_graph_edges


@dataclass(frozen=True)
class ModelOutput:
    relevance: Tensor
    readiness: Tensor
    programmes: Tensor
    pooled: Tensor
    graph_state: Tensor
    modality_mass: Tensor
    token_valid: Tensor
    embedding: Tensor

    def tractability_probability(self) -> Tensor:
        return self.readiness[:, -1]


@dataclass(frozen=True)
class ModelConfig:
    model_dim: int = 32
    hidden_dim: int = 64
    graph_rounds: int = 2
    n_readiness_tiers: int = 2
    include_annotation_nodes: bool = True
    constrained: bool = True
    graph_enabled: bool = True


class TargetPrioritiser(nn.Module):
    """Composition of the four frozen encoders, the bottleneck, the tower and the heads."""

    def __init__(self, panel: EvidencePanel, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.widths = suite_widths(panel)
        self.bottleneck_config = BottleneckConfig(
            model_dim=config.model_dim,
            n_programmes=panel.n_programmes,
            n_graph_rounds=config.graph_rounds,
            n_readiness_tiers=config.n_readiness_tiers,
            constrained=config.constrained,
        )
        self.bottleneck = AvailabilityMaskedBottleneck(self.widths, self.bottleneck_config)
        self.edge_types = _graph_edge_types(panel)
        self.tower = TypedGraphTower(panel.n_programmes, len(self.edge_types), self.bottleneck_config)
        self.heads = DualHeads(panel.n_programmes, self.bottleneck_config, hidden_dim=config.hidden_dim)

    def encode(
        self,
        panel: EvidencePanel,
        availability: Tensor | None = None,
        edge_keep: Tensor | None = None,
    ) -> ModelOutput:
        features = self.features(panel)
        mask = availability_tensor(panel.availability) if availability is None else availability
        state = self.bottleneck(features, mask)
        if self.config.graph_enabled:
            assembly = self.graph_assembly(panel, state)
            edges = assembly.batch if edge_keep is None else masked_graph_edges(assembly.batch, edge_keep)
            graph_state = self.tower(edges)[: panel.n_genes]
        else:
            graph_state = torch.zeros_like(state.programmes)
        heads = self.heads(state.programmes, graph_state)
        return ModelOutput(
            relevance=heads.relevance,
            readiness=heads.readiness,
            programmes=state.programmes,
            pooled=state.pooled,
            graph_state=graph_state,
            modality_mass=state.modality_mass,
            token_valid=state.token_valid,
            embedding=heads.embedding,
        )

    def features(self, panel: EvidencePanel) -> tuple[Tensor, ...]:
        suite = FrozenEncoderSuite.from_panel(panel, context_dim=panel.context_features.shape[1])
        return tuple(encoder.encode(panel) for encoder in suite.encoders)

    def graph_assembly(self, panel: EvidencePanel, state: BottleneckState) -> GraphAssembly:
        return build_graph_batch(
            panel,
            state.programmes,
            interaction_edge_types=self.edge_types,
            include_annotation_nodes=self.config.include_annotation_nodes,
        )

    def graph_batch(self, panel: EvidencePanel, state: BottleneckState) -> TypedGraphBatch:
        assembly = self.graph_assembly(panel, state)
        return with_gene_states(assembly, panel, state.programmes)

    def forward(
        self,
        panel: EvidencePanel,
        availability: Tensor | None = None,
        edge_keep: Tensor | None = None,
    ) -> ModelOutput:
        return self.encode(panel, availability, edge_keep)

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def _graph_edge_types(panel: EvidencePanel) -> tuple[str, ...]:
    names = list(interaction_edge_types(panel))
    for name in (
        "programme_membership",
        "population_attribution",
        "compartment_localisation",
        "tractability_annotation",
    ):
        if name not in names:
            names.append(name)
    if not names:
        raise ValueError("the typed graph carries no edge type")
    return tuple(names)


def scores_from_output(output: ModelOutput) -> np.ndarray:
    return np.asarray(output.relevance.detach().cpu().numpy(), dtype=np.float64)


def readiness_from_output(output: ModelOutput) -> np.ndarray:
    return np.asarray(output.tractability_probability().detach().cpu().numpy(), dtype=np.float64)


def programme_from_output(output: ModelOutput) -> np.ndarray:
    return np.asarray(output.programmes.detach().cpu().numpy(), dtype=np.float64)


def modality_attribution(output: ModelOutput, layer_index: int) -> np.ndarray:
    return np.asarray(output.modality_mass[:, layer_index].detach().cpu().numpy(), dtype=np.float64)
