"""The four objective terms of Eq. (1).

Ref: Eq. (1) and Sec. 2.1.

    L = L_rank(y_dev, rho) + lambda_1 * L_imm + lambda_2 * L_read + lambda_3 * L_pert

lambda_1 = 0.5, lambda_2 = 0.5, lambda_3 = 0.25, and the weights are constrained so the
ranking term is never traded away for interpretability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

EPSILON: Final[float] = 1e-7


def listwise_ranking_loss(
    scores: Tensor, relevance: Tensor, *, temperature: float = 0.1, mask: Tensor | None = None
) -> Tensor:
    """Listwise cross-entropy between the score distribution and the label distribution.

    L_rank is a listwise loss over the development labels, so the two distributions are
    taken over the whole pool at once rather than over pairs.
    """
    if mask is not None:
        scores = scores[mask]
        relevance = relevance[mask]
    if scores.numel() == 0:
        return torch.zeros((), dtype=scores.dtype, device=scores.device)
    score_logprob = torch.log_softmax(scores / temperature, dim=0)
    target = torch.softmax(relevance / temperature, dim=0)
    return -(target * score_logprob).sum()


def immune_relevance_divergence(programmes: Tensor, target: Tensor) -> Tensor:
    """KL(s || s_tilde) in nats against the supervised programme label.

    The label is floored so a programme absent from a gene cannot drive the divergence to
    infinity; the floor is the only numerical guard the term needs.
    """
    floored = target.clamp(min=EPSILON)
    normalised = floored / floored.sum(dim=-1, keepdim=True)
    return (normalised * (normalised.log() - programmes.clamp(min=EPSILON).log())).sum(dim=-1).mean()


def readiness_cross_entropy(readiness: Tensor, labels: Tensor) -> Tensor:
    """Binary cross-entropy between the tier distribution and the tractability label."""
    if readiness.shape[-1] != 2:
        raise ValueError("the readiness head must carry exactly two tiers for this term")
    probability = readiness[:, -1].clamp(min=EPSILON, max=1.0 - EPSILON)
    target = labels.to(dtype=probability.dtype)
    return -(target * probability.log() + (1.0 - target) * (1.0 - probability).log()).mean()


def perturbation_consistency(
    relevance: Tensor, reversal: Tensor, available: Tensor, *, margin: float = 0.5
) -> Tensor:
    """Penalise a nomination whose relevance disagrees with the perturbation-reversal sign.

    Ref: Sec. 2.5 - the term constrains consistency and never creates a nomination, so it
    is a one-sided penalty: agreement costs nothing, disagreement costs at most `margin`.
    """
    if not bool(available.any()):
        return torch.zeros((), dtype=relevance.dtype, device=relevance.device)
    selected_relevance = relevance[available]
    selected_reversal = reversal[available]
    disagreement = margin - selected_relevance * torch.tanh(selected_reversal)
    return torch.nn.functional.relu(disagreement).mean()


@dataclass(frozen=True)
class ObjectiveWeights:
    immune: float = 0.5
    readiness: float = 0.5
    perturbation: float = 0.25


@dataclass(frozen=True)
class ObjectiveBreakdown:
    total: Tensor
    ranking: Tensor
    immune: Tensor
    readiness: Tensor
    perturbation: Tensor


def total_objective(
    *,
    scores: Tensor,
    relevance: Tensor,
    programmes: Tensor,
    programme_targets: Tensor,
    readiness: Tensor,
    tractability: Tensor,
    reversal: Tensor,
    reversal_available: Tensor,
    weights: ObjectiveWeights,
    temperature: float = 0.1,
    ranking_weight: float = 1.0,
) -> ObjectiveBreakdown:
    ranking = listwise_ranking_loss(scores, relevance, temperature=temperature)
    immune = immune_relevance_divergence(programmes, programme_targets)
    ready = readiness_cross_entropy(readiness, tractability)
    perturbed = perturbation_consistency(relevance, reversal, reversal_available)
    total = ranking_weight * ranking + weights.immune * immune + weights.readiness * ready
    total = total + weights.perturbation * perturbed
    return ObjectiveBreakdown(
        total=total,
        ranking=ranking,
        immune=immune,
        readiness=ready,
        perturbation=perturbed,
    )
