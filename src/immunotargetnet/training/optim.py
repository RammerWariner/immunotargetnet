"""Optimiser and learning-rate schedule.

Ref: Sec. 2.1 (AdamW with a variable learning rate of 1e-3, batch size 32, 60 epochs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

DEFAULT_WARMUP_FRACTION: Final[float] = 0.05
DEFAULT_MIN_FACTOR: Final[float] = 0.01


@dataclass(frozen=True)
class OptimiserSpec:
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    warmup_fraction: float = DEFAULT_WARMUP_FRACTION
    min_factor: float = DEFAULT_MIN_FACTOR
    grad_clip_norm: float = 1.0


def build_optimiser(model: nn.Module, spec: OptimiserSpec) -> AdamW:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(("type_token", "type_weight_logits")):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": spec.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(
        groups,
        lr=spec.learning_rate,
        betas=spec.betas,
        eps=spec.eps,
    )


def schedule_factor(
    step: int,
    total_steps: int,
    *,
    warmup_steps: int,
    min_factor: float = DEFAULT_MIN_FACTOR,
) -> float:
    """Linear warm-up into a cosine decay, floored so the last steps still move."""
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps and warmup_steps > 0:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(min_factor, cosine)


def build_scheduler(optimiser: AdamW, total_steps: int, spec: OptimiserSpec) -> LambdaLR:
    warmup_steps = round(spec.warmup_fraction * total_steps)
    return LambdaLR(
        optimiser,
        lr_lambda=lambda step: schedule_factor(
            step, total_steps, warmup_steps=warmup_steps, min_factor=spec.min_factor
        ),
    )


def backpropagate(loss: torch.Tensor) -> None:
    """Typed wrapper: `Tensor.backward` is unannotated in this torch build."""
    torch.autograd.backward(loss)


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm)
    return float(norm)


def steps_per_epoch(n_genes: int, batch_size: int) -> int:
    return max(1, math.ceil(n_genes / batch_size))
