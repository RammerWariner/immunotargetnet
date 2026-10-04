"""Bottleneck, graph tower, heads and the composed model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from immunotargetnet.bottleneck.attention import (
    NEGATIVE_INFINITY,
    AvailabilityMaskedBottleneck,
    BottleneckConfig,
    availability_tensor,
    suite_widths,
)
from immunotargetnet.bottleneck.dropout import available_layer_count, modality_dropout
from immunotargetnet.bottleneck.fusion import FreeFormFusion
from immunotargetnet.encoders.frozen import FrozenEncoderSuite, ModalityProjection
from immunotargetnet.evidence.panel import PanelBundle
from immunotargetnet.evidence.schema import MODALITY_ORDER
from immunotargetnet.graph.assembly import ANNOTATION_EDGE_TYPES, build_graph_batch, interaction_edge_types
from immunotargetnet.graph.degree import degree_gap, fit_degree_control, standardise_degree, top_decile_mask
from immunotargetnet.graph.heads import DualHeads
from immunotargetnet.graph.message_passing import TypedGraphBatch, TypedGraphTower, segment_softmax
from immunotargetnet.model import ModelConfig, TargetPrioritiser


def test_masked_attention_matches_a_reference(bundle: PanelBundle) -> None:
    panel = bundle.panel
    config = BottleneckConfig(model_dim=6, n_programmes=panel.n_programmes)
    model = AvailabilityMaskedBottleneck(suite_widths(panel), config)
    model.eval()
    suite = FrozenEncoderSuite.from_panel(panel, context_dim=panel.context_features.shape[1])
    features = tuple(encoder.encode(panel)[:16] for encoder in suite.encoders)
    mask = availability_tensor(panel.availability[:16])
    with torch.no_grad():
        state = model(features, mask)
        stack = model.token_stack(features)
        queries = model.query(stack).numpy().astype(np.float64)
        keys = model.key(stack).numpy().astype(np.float64)
        values = model.value(stack).numpy().astype(np.float64)
    mask_array = np.asarray(panel.availability[:16], dtype=bool)
    reference = np.zeros_like(state.pooled.numpy().astype(np.float64))
    for row in range(16):
        logits = queries[row] @ keys[row].T / float(model.scale)
        logits = np.where(mask_array[row][None, :], logits, NEGATIVE_INFINITY)
        block = logits - logits.max(axis=1, keepdims=True)
        weights = np.exp(block) / np.exp(block).sum(axis=1, keepdims=True)
        reference[row] = (weights @ values[row])[mask_array[row]].mean(axis=0)
    assert np.abs(reference - state.pooled.numpy().astype(np.float64)).max() < 1e-5
    assert torch.isfinite(state.programmes).all()


def test_programmes_land_on_the_simplex(bundle: PanelBundle) -> None:
    panel = bundle.panel
    config = BottleneckConfig(model_dim=6, n_programmes=panel.n_programmes)
    model = AvailabilityMaskedBottleneck(suite_widths(panel), config)
    suite = FrozenEncoderSuite.from_panel(panel, context_dim=panel.context_features.shape[1])
    features = tuple(encoder.encode(panel)[:8] for encoder in suite.encoders)
    with torch.no_grad():
        state = model(features, availability_tensor(panel.availability[:8]))
    sums = state.programmes.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert (state.programmes >= 0).all()


def test_free_form_variant_is_not_a_simplex(bundle: PanelBundle) -> None:
    panel = bundle.panel
    config = BottleneckConfig(model_dim=6, n_programmes=panel.n_programmes, constrained=False)
    model = AvailabilityMaskedBottleneck(suite_widths(panel), config)
    suite = FrozenEncoderSuite.from_panel(panel, context_dim=panel.context_features.shape[1])
    features = tuple(encoder.encode(panel)[:8] for encoder in suite.encoders)
    with torch.no_grad():
        state = model(features, availability_tensor(panel.availability[:8]))
    sums = state.programmes.sum(dim=1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)
    assert model.free_form is not None
    assert isinstance(model.free_form, FreeFormFusion)


def test_modality_projection_adds_a_learnable_type_token() -> None:
    projection = ModalityProjection(4, 6)
    baseline = projection(torch.zeros(3, 4))
    assert baseline.shape == (3, 6)
    assert projection.type_token.requires_grad
    with torch.no_grad():
        projection.type_token.add_(1.0)
    shifted = projection(torch.zeros(3, 4))
    assert not torch.allclose(baseline, shifted)


def test_modality_dropout_never_empties_a_gene() -> None:
    rng = np.random.default_rng(3)
    availability = rng.random((200, len(MODALITY_ORDER))) > 0.4
    availability[:, 0] = True
    availability[0] = [True, False, False, False]
    plan = modality_dropout(availability, 0.9, seed=5, step=0)
    assert (~plan.mask.any(axis=1)).sum() == 0
    assert (plan.mask & availability).sum() < availability.sum()
    assert plan.mask[0, 0]


def test_available_layer_count_excludes_the_keystone_slot() -> None:
    mask = np.array([[True, False, False, True], [True, True, True, False]])
    counts = available_layer_count(mask, context_index=len(MODALITY_ORDER) - 1)
    assert counts.tolist() == [1, 3]


def test_segment_softmax_normalises_within_groups() -> None:
    logits = torch.tensor([1.0, 2.0, 3.0, 0.5, 0.5, 0.5])
    segment = torch.tensor([0, 0, 0, 1, 1, 1])
    weights = segment_softmax(logits, segment, 2)
    assert float(weights[:3].sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(weights[3:].sum()) == pytest.approx(1.0, abs=1e-6)
    assert weights[2] > weights[0]


def test_typed_graph_tower_keeps_the_node_count(bundle: PanelBundle) -> None:
    panel = bundle.panel
    set_seed = torch.manual_seed
    set_seed(11)
    state = torch.rand(panel.n_genes, panel.n_programmes)
    assembly = build_graph_batch(panel, state, interaction_edge_types=interaction_edge_types(panel))
    tower = TypedGraphTower(
        panel.n_programmes,
        assembly.batch.n_edge_types,
        BottleneckConfig(model_dim=panel.n_programmes, n_programmes=panel.n_programmes, n_graph_rounds=1),
    )
    with torch.no_grad():
        updated = tower(assembly.batch)
    assert updated.shape == (assembly.batch.n_nodes, panel.n_programmes)
    assert torch.isfinite(updated).all()
    assert torch.allclose(tower.type_weights().sum(), torch.tensor(1.0), atol=1e-6)
    assert all(name in assembly.batch.edge_type_names for name in ANNOTATION_EDGE_TYPES)
    expected_annotations = (
        panel.n_programmes
        + len(panel.pathway_labels)
        + len(panel.cellstate_feature_names)
        + len(panel.niche_feature_names)
        + panel.availability.shape[1]
    )
    assert assembly.n_annotation_nodes == expected_annotations


def test_graph_without_edges_still_returns_a_state() -> None:
    batch = TypedGraphBatch(
        node_state=torch.zeros(4, 3),
        edge_source=torch.zeros(0, dtype=torch.long),
        edge_target=torch.zeros(0, dtype=torch.long),
        edge_type_index=torch.zeros(0, dtype=torch.long),
        edge_log_weight=torch.zeros(0),
        edge_type_names=("string",),
        n_nodes=4,
    )
    tower = TypedGraphTower(3, 1, BottleneckConfig(model_dim=3, n_programmes=3))
    with torch.no_grad():
        updated = tower(batch)
    assert updated.shape == (4, 3)
    # Eq. (4) applies the activation to the empty aggregate, so the state is the
    # activation of zero rather than zero itself.
    assert torch.allclose(updated, torch.full((4, 3), 0.5), atol=1e-6)


def test_dual_heads_share_their_first_layer(bundle: PanelBundle) -> None:
    panel = bundle.panel
    heads = DualHeads(
        panel.n_programmes,
        BottleneckConfig(model_dim=8, n_programmes=panel.n_programmes, n_readiness_tiers=2),
    )
    programmes = torch.full((5, panel.n_programmes), 1.0 / panel.n_programmes)
    graph_state = torch.zeros(5, panel.n_programmes)
    output = heads(programmes, graph_state)
    assert output.relevance.shape == (5,)
    assert output.readiness.shape == (5, 2)
    assert torch.allclose(output.readiness.sum(dim=1), torch.ones(5), atol=1e-5)
    assert float(output.relevance.detach().min()) >= 0.0
    assert float(output.relevance.detach().max()) <= 1.0


def test_heads_agree_when_both_inputs_agree(bundle: PanelBundle) -> None:
    panel = bundle.panel
    heads = DualHeads(
        panel.n_programmes,
        BottleneckConfig(model_dim=8, n_programmes=panel.n_programmes, n_readiness_tiers=2),
    )
    programmes = torch.zeros(4, panel.n_programmes)
    programmes[:, 0] = 1.0
    graph_state = torch.zeros(4, panel.n_programmes)
    graph_state[:, 0] = 5.0
    output = heads(programmes, graph_state)
    assert float(output.tractability_probability().detach().std()) < 1e-5


def test_model_reports_its_trainable_parameter_count(bundle: PanelBundle) -> None:
    panel = bundle.panel
    model = TargetPrioritiser(panel, ModelConfig(model_dim=8))
    counted = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert model.trainable_parameters() == counted > 0


def test_degree_control_residualises_the_score(bundle: PanelBundle) -> None:
    panel = bundle.panel
    rng = np.random.default_rng(5)
    degree = rng.random(panel.n_genes) * 10.0
    scores = 2.0 * standardise_degree(degree) + rng.normal(scale=0.1, size=panel.n_genes)
    control = fit_degree_control(scores, degree)
    assert control.slope > 1.5
    assert abs(float(np.corrcoef(control.residual_score, standardise_degree(degree))[0, 1])) < 1e-6
    assert control.degree_then_rank.min() == 1


def test_top_decile_mask_selects_a_tenth(bundle: PanelBundle) -> None:
    degree = np.arange(1000, dtype=np.float64)
    mask = top_decile_mask(degree, fraction=0.1)
    assert 90 <= int(mask.sum()) <= 110


def test_degree_gap_is_signed(bundle: PanelBundle) -> None:
    degree = np.arange(100, dtype=np.float64)
    shortlist = np.zeros(100, dtype=bool)
    shortlist[-20:] = True
    curated = np.zeros(100, dtype=bool)
    curated[:20] = True
    assert degree_gap(degree, shortlist, curated) > 0.0
