"""Objective terms: each one is checked for the property the manuscript assigns it."""

from __future__ import annotations

import torch

from immunotargetnet.objectives.total import (
    ObjectiveWeights,
    immune_relevance_divergence,
    listwise_ranking_loss,
    perturbation_consistency,
    readiness_cross_entropy,
    total_objective,
)


def test_listwise_loss_prefers_the_correct_order() -> None:
    relevance = torch.tensor([1.0, 0.0, 0.0, 0.0])
    aligned = torch.tensor([3.0, 1.0, 0.0, -1.0])
    inverted = torch.tensor([-1.0, 0.0, 1.0, 3.0])
    assert float(listwise_ranking_loss(aligned, relevance)) < float(
        listwise_ranking_loss(inverted, relevance)
    )


def test_listwise_loss_is_zero_for_a_perfect_match() -> None:
    relevance = torch.tensor([1.0, 0.5, 0.0])
    loss = listwise_ranking_loss(relevance * 10.0, relevance, temperature=0.05)
    assert float(loss) < 0.05


def test_listwise_loss_ignores_excluded_units() -> None:
    scores = torch.tensor([1.0, 2.0, 3.0])
    relevance = torch.tensor([0.0, 1.0, 0.0])
    mask = torch.tensor([True, True, False])
    narrowed = listwise_ranking_loss(scores, relevance, mask=mask)
    direct = listwise_ranking_loss(scores[:2], relevance[:2])
    assert float(narrowed) == float(direct)


def test_immune_divergence_is_zero_at_the_label() -> None:
    target = torch.tensor([[0.6, 0.3, 0.1]])
    assert float(immune_relevance_divergence(target, target)) < 1e-5


def test_immune_divergence_grows_with_disagreement() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0]])
    close = torch.tensor([[0.9, 0.05, 0.05]])
    far = torch.tensor([[0.05, 0.05, 0.9]])
    assert float(immune_relevance_divergence(close, target)) < float(immune_relevance_divergence(far, target))


def test_immune_divergence_survives_a_zero_entry_in_the_label() -> None:
    target = torch.tensor([[1.0, 0.0, 0.0]])
    predicted = torch.tensor([[0.5, 0.3, 0.2]])
    value = float(immune_relevance_divergence(predicted, target))
    assert 0.0 <= value < 100.0


def test_readiness_cross_entropy_penalises_confident_mistakes() -> None:
    labels = torch.tensor([1.0, 0.0])
    confident = torch.tensor([[0.01, 0.99], [0.99, 0.01]])
    wrong = torch.tensor([[0.99, 0.01], [0.01, 0.99]])
    assert float(readiness_cross_entropy(confident, labels)) < float(readiness_cross_entropy(wrong, labels))


def test_readiness_cross_entropy_needs_two_tiers() -> None:
    try:
        readiness_cross_entropy(torch.full((2, 3), 1 / 3), torch.zeros(2))
    except ValueError:
        return
    raise AssertionError("a three-tier readiness head must be rejected")


def test_perturbation_consistency_punishes_disagreement_only() -> None:
    relevance = torch.tensor([0.9, 0.1])
    reversal = torch.tensor([1.0, 1.0])
    available = torch.tensor([True, True])
    value = float(perturbation_consistency(relevance, reversal, available))
    agreeing = float(perturbation_consistency(torch.tensor([0.9, 0.9]), reversal, available))
    assert value > agreeing


def test_perturbation_consistency_is_zero_without_labels() -> None:
    value = perturbation_consistency(torch.tensor([0.5]), torch.tensor([1.0]), torch.tensor([False]))
    assert float(value) == 0.0


def test_total_objective_is_the_weighted_combination() -> None:
    weights = ObjectiveWeights(immune=0.5, readiness=0.5, perturbation=0.25)
    breakdown = total_objective(
        scores=torch.tensor([2.0, 1.0, 0.0]),
        relevance=torch.tensor([1.0, 0.5, 0.0]),
        programmes=torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]),
        programme_targets=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.2, 0.3, 0.5]]),
        readiness=torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.5, 0.5]]),
        tractability=torch.tensor([1.0, 0.0, 1.0]),
        reversal=torch.tensor([1.0, -1.0, 0.5]),
        reversal_available=torch.tensor([True, True, False]),
        weights=weights,
    )
    expected = (
        breakdown.ranking
        + weights.immune * breakdown.immune
        + weights.readiness * breakdown.readiness
        + weights.perturbation * breakdown.perturbation
    )
    assert abs(float(breakdown.total) - float(expected)) < 1e-6
    assert all(
        torch.isfinite(tensor)
        for tensor in (
            breakdown.total,
            breakdown.ranking,
            breakdown.immune,
            breakdown.readiness,
            breakdown.perturbation,
        )
    )
