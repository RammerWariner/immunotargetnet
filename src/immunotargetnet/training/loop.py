"""Training loop.

Ref: Sec. 2.1 (AdamW, learning rate 1e-3, batch size 32, 60 epochs, availability-mask
dropout 0.2 during training and 0 at inference, early stopping on validation fidelity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from immunotargetnet.bottleneck.attention import availability_tensor
from immunotargetnet.bottleneck.dropout import modality_dropout
from immunotargetnet.evidence.schema import EvidencePanel
from immunotargetnet.metrics.ranking import PRIMARY_CUTOFF, ndcg_at_k
from immunotargetnet.model import ModelConfig, ModelOutput, TargetPrioritiser
from immunotargetnet.objectives.total import ObjectiveBreakdown, ObjectiveWeights, total_objective
from immunotargetnet.runtime.logging_setup import get_logger
from immunotargetnet.runtime.seed import set_seed
from immunotargetnet.training.checkpoint import CheckpointPayload, save_checkpoint, state_dict_digest
from immunotargetnet.training.earlystop import EarlyStopping
from immunotargetnet.training.optim import (
    OptimiserSpec,
    backpropagate,
    build_optimiser,
    build_scheduler,
    clip_gradients,
    steps_per_epoch,
)

LOGGER = get_logger("training.loop")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 60
    batch_size: int = 32
    modality_dropout_rate: float = 0.2
    inference_dropout_rate: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    patience: int = 10
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    model: ModelConfig = field(default_factory=ModelConfig)

    def optimiser_spec(self) -> OptimiserSpec:
        return OptimiserSpec(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            grad_clip_norm=self.grad_clip_norm,
        )


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    ranking: float
    immune: float
    readiness: float
    perturbation: float
    validation_fidelity: float


@dataclass
class TrainingResult:
    model: TargetPrioritiser
    history: list[EpochRecord]
    best_epoch: int
    best_validation_fidelity: float
    seed: int
    digest: str
    n_parameters: int


def _batch_indices(pool: np.ndarray, batch_size: int, generator: np.random.Generator) -> np.ndarray:
    if pool.size == 0:
        return np.zeros(0, dtype=np.int64)
    size = min(batch_size, pool.size)
    return np.asarray(generator.choice(pool, size=size, replace=False), dtype=np.int64)


def run_training(
    panel: EvidencePanel,
    *,
    training_units: np.ndarray,
    validation_units: np.ndarray,
    seed: int,
    config: TrainingConfig,
    checkpoint_path: Any = None,
) -> TrainingResult:
    """Train the model on the development labels and select on validation fidelity."""
    set_seed(seed)
    model = TargetPrioritiser(panel, config.model)
    optimiser = build_optimiser(model, config.optimiser_spec())
    per_epoch = steps_per_epoch(int(training_units.size), config.batch_size)
    total_steps = max(1, per_epoch * config.epochs)
    scheduler = build_scheduler(optimiser, total_steps, config.optimiser_spec())
    stopper = EarlyStopping(patience=config.patience)
    generator = np.random.default_rng(seed)
    base_mask = availability_tensor(panel.availability)
    development = torch.as_tensor(panel.development_labels, dtype=torch.float32)
    programme_targets = torch.as_tensor(panel.programme_targets, dtype=torch.float32)
    tractability = torch.as_tensor(panel.tractability.astype(np.float64), dtype=torch.float32)
    reversal = torch.as_tensor(panel.reversal_index, dtype=torch.float32)
    reversal_available = torch.as_tensor(panel.reversal_available, dtype=torch.bool)
    history: list[EpochRecord] = []
    step_index = 0
    best_state: dict[str, Any] | None = None
    best_fidelity = float("-inf")
    best_epoch = -1

    for epoch in range(config.epochs):
        model.train()
        totals = np.zeros(4, dtype=np.float64)
        for _ in range(per_epoch):
            plan = modality_dropout(
                panel.availability, config.modality_dropout_rate, seed=seed, step=step_index
            )
            mask = availability_tensor(plan.mask)
            batch = _batch_indices(training_units, config.batch_size, generator)
            output = model(panel, mask)
            breakdown = _loss_on_batch(
                output,
                batch,
                development=development,
                programme_targets=programme_targets,
                tractability=tractability,
                reversal=reversal,
                reversal_available=reversal_available,
                weights=config.weights,
            )
            optimiser.zero_grad(set_to_none=True)
            backpropagate(breakdown.total)
            clip_gradients(model, config.grad_clip_norm)
            optimiser.step()
            scheduler.step()
            step_index += 1
            totals += np.array(
                [
                    float(breakdown.total.detach().cpu()),
                    float(breakdown.ranking.detach().cpu()),
                    float(breakdown.immune.detach().cpu()),
                    float(breakdown.perturbation.detach().cpu()),
                ],
                dtype=np.float64,
            )
        fidelity = validation_fidelity(model, panel, validation_units, base_mask)
        record = EpochRecord(
            epoch=epoch,
            loss=float(totals[0] / max(per_epoch, 1)),
            ranking=float(totals[1] / max(per_epoch, 1)),
            immune=float(totals[2] / max(per_epoch, 1)),
            readiness=0.0,
            perturbation=float(totals[3] / max(per_epoch, 1)),
            validation_fidelity=fidelity,
        )
        history.append(record)
        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if stopper.update(fidelity, epoch):
            LOGGER.info("early stop at epoch %d, best fidelity %.4f", epoch, stopper.best)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    digest = state_dict_digest(model)
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            CheckpointPayload(
                epoch=best_epoch,
                seed=seed,
                model_state=model.state_dict(),
                optimiser_state=None,
                scheduler_state=None,
                metrics={"validation_fidelity": best_fidelity},
                config={
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "modality_dropout_rate": config.modality_dropout_rate,
                },
            ),
        )
    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_validation_fidelity=best_fidelity,
        seed=seed,
        digest=digest,
        n_parameters=model.trainable_parameters(),
    )


def _loss_on_batch(
    output: ModelOutput,
    batch: np.ndarray,
    *,
    development: torch.Tensor,
    programme_targets: torch.Tensor,
    tractability: torch.Tensor,
    reversal: torch.Tensor,
    reversal_available: torch.Tensor,
    weights: ObjectiveWeights,
) -> ObjectiveBreakdown:
    if batch.size == 0:
        raise ValueError("training batch is empty")
    selection = torch.as_tensor(batch, dtype=torch.long)
    return total_objective(
        scores=output.relevance[selection],
        relevance=development[selection],
        programmes=output.programmes[selection],
        programme_targets=programme_targets[selection],
        readiness=output.readiness[selection],
        tractability=tractability[selection],
        reversal=reversal[selection],
        reversal_available=reversal_available[selection],
        weights=weights,
    )


@torch.no_grad()
def validation_fidelity(
    model: TargetPrioritiser,
    panel: EvidencePanel,
    validation_units: np.ndarray,
    mask: torch.Tensor | None = None,
) -> float:
    """NDCG@20 of the validation block against the development labels."""
    model.eval()
    inference_mask = availability_tensor(panel.availability) if mask is None else mask
    output = model(panel, inference_mask)
    scores = np.asarray(output.relevance.detach().cpu().numpy(), dtype=np.float64)
    labels = np.asarray(panel.development_labels, dtype=np.float64)
    if validation_units.size == 0:
        return 0.0
    selected = np.asarray(validation_units, dtype=np.int64)
    return ndcg_at_k(labels[selected], scores[selected], PRIMARY_CUTOFF)
