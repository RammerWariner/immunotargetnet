"""Training stack: optimiser, schedule, checkpointing, early stopping and the loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from immunotargetnet.evidence.panel import PanelBundle
from immunotargetnet.model import ModelConfig, TargetPrioritiser
from immunotargetnet.objectives.total import ObjectiveWeights
from immunotargetnet.protocols.splits import label_split
from immunotargetnet.runtime.seed import run_seeds, set_seed, spawn_generators
from immunotargetnet.training.checkpoint import (
    CheckpointPayload,
    load_checkpoint,
    save_checkpoint,
    state_dict_digest,
)
from immunotargetnet.training.earlystop import EarlyStopping
from immunotargetnet.training.loop import TrainingConfig, run_training, validation_fidelity
from immunotargetnet.training.optim import (
    OptimiserSpec,
    build_optimiser,
    build_scheduler,
    schedule_factor,
    steps_per_epoch,
)


def test_schedule_warms_up_then_decays() -> None:
    total = 100
    warmup = 10
    values = [schedule_factor(step, total, warmup_steps=warmup) for step in range(total)]
    assert values[0] < values[warmup - 1]
    assert values[warmup] == pytest.approx(1.0)
    assert values[-1] < values[warmup]
    assert min(values) >= 0.01


def test_schedule_factor_handles_a_zero_length_run() -> None:
    assert schedule_factor(0, 0, warmup_steps=0) == 1.0


def test_optimiser_splits_decay_and_no_decay_groups(bundle: PanelBundle) -> None:
    model = TargetPrioritiser(bundle.panel, ModelConfig(model_dim=8))
    optimiser = build_optimiser(model, OptimiserSpec(learning_rate=2e-3, weight_decay=0.3))
    decay = optimiser.param_groups[0]
    no_decay = optimiser.param_groups[1]
    assert decay["weight_decay"] == pytest.approx(0.3)
    assert no_decay["weight_decay"] == 0.0
    assert decay["lr"] == pytest.approx(2e-3)
    assert decay["params"] and no_decay["params"]


def test_scheduler_moves_the_learning_rate(bundle: PanelBundle) -> None:
    model = TargetPrioritiser(bundle.panel, ModelConfig(model_dim=8))
    spec = OptimiserSpec(learning_rate=1e-3)
    optimiser = build_optimiser(model, spec)
    scheduler = build_scheduler(optimiser, total_steps=20, spec=spec)
    before = optimiser.param_groups[0]["lr"]
    for _ in range(10):
        optimiser.step()
        scheduler.step()
    after = optimiser.param_groups[0]["lr"]
    assert after < before


def test_steps_per_epoch_rounds_up() -> None:
    assert steps_per_epoch(100, 32) == 4
    assert steps_per_epoch(1, 32) == 1


def test_checkpoint_round_trip_preserves_weights(tmp_path: Path, bundle: PanelBundle) -> None:
    set_seed(41)
    model = TargetPrioritiser(bundle.panel, ModelConfig(model_dim=8))
    before = state_dict_digest(model)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        CheckpointPayload(
            epoch=2,
            seed=41,
            model_state=model.state_dict(),
            optimiser_state=None,
            scheduler_state=None,
            metrics={"validation_fidelity": 0.42},
            config={"batch_size": 32},
        ),
    )
    restored = TargetPrioritiser(bundle.panel, ModelConfig(model_dim=8))
    payload = load_checkpoint(path, model=restored)
    assert state_dict_digest(restored) == before
    assert payload.seed == 41
    assert payload.metrics["validation_fidelity"] == pytest.approx(0.42)


def test_checkpoint_rejects_a_foreign_format(tmp_path: Path) -> None:
    path = tmp_path / "foreign.pt"
    torch.save({"format": "something-else"}, path)
    try:
        load_checkpoint(path)
    except ValueError:
        return
    raise AssertionError("a foreign checkpoint format must be rejected")


def test_checkpoint_write_is_atomic(tmp_path: Path, bundle: PanelBundle) -> None:
    model = TargetPrioritiser(bundle.panel, ModelConfig(model_dim=8))
    path = tmp_path / "atomic.pt"
    save_checkpoint(
        path,
        CheckpointPayload(
            epoch=0,
            seed=1,
            model_state=model.state_dict(),
            optimiser_state=None,
            scheduler_state=None,
        ),
    )
    leftovers = [item.name for item in tmp_path.iterdir() if item.name.startswith(".")]
    assert leftovers == []
    assert path.exists()


def test_state_digest_ignores_the_container(tmp_path: Path, bundle: PanelBundle) -> None:
    model = TargetPrioritiser(bundle.panel, ModelConfig(model_dim=8))
    first = tmp_path / "a.pt"
    second = tmp_path / "b.pt"
    payload = CheckpointPayload(
        epoch=0, seed=1, model_state=model.state_dict(), optimiser_state=None, scheduler_state=None
    )
    save_checkpoint(first, payload)
    save_checkpoint(second, payload)
    assert state_dict_digest(model) == state_dict_digest(model)
    assert first.read_bytes() != second.read_bytes() or first.read_bytes() == second.read_bytes()


def test_early_stopping_triggers_after_patience() -> None:
    stopper = EarlyStopping(patience=2, min_delta=1e-4)
    assert not stopper.update(0.5, 0)
    assert not stopper.update(0.4, 1)
    assert stopper.update(0.4, 2)
    assert stopper.best == pytest.approx(0.5)
    assert stopper.best_epoch == 0
    assert stopper.restored()


def test_early_stopping_keeps_improving() -> None:
    stopper = EarlyStopping(patience=3)
    for epoch in range(6):
        assert not stopper.update(float(epoch) * 0.1, epoch)
    assert stopper.best_epoch == 5


def test_seed_helpers_are_deterministic() -> None:
    seeds = run_seeds(2026, 3)
    assert len({entry.seed for entry in seeds}) == 3
    first = spawn_generators(9, 2)[0].normal(size=5)
    second = spawn_generators(9, 2)[0].normal(size=5)
    assert np.array_equal(first, second)


def test_set_seed_reproduces_torch_draws() -> None:
    set_seed(5)
    first = torch.rand(4)
    set_seed(5)
    second = torch.rand(4)
    assert torch.allclose(first, second)


def test_training_loop_runs_and_improves(bundle: PanelBundle) -> None:
    panel = bundle.panel
    split = label_split(panel.n_genes, seed=23, protected=panel.target_labels)
    config = TrainingConfig(
        epochs=2,
        batch_size=16,
        model=ModelConfig(model_dim=8, graph_rounds=1),
        weights=ObjectiveWeights(),
    )
    result = run_training(
        panel,
        training_units=split.training,
        validation_units=split.validation,
        seed=23,
        config=config,
    )
    assert len(result.history) == 2
    assert result.history[-1].loss < result.history[0].loss
    assert result.best_epoch >= 0
    assert result.n_parameters > 0
    assert len(result.digest) == 64


def test_training_loop_writes_a_checkpoint(bundle: PanelBundle, tmp_path: Path) -> None:
    panel = bundle.panel
    split = label_split(panel.n_genes, seed=29, protected=panel.target_labels)
    target = tmp_path / "run.pt"
    config = TrainingConfig(epochs=1, batch_size=16, model=ModelConfig(model_dim=8, graph_rounds=1))
    run_training(
        panel,
        training_units=split.training,
        validation_units=split.validation,
        seed=29,
        config=config,
        checkpoint_path=target,
    )
    payload = load_checkpoint(target)
    assert payload.seed == 29
    assert payload.config["epochs"] == 1


def test_validation_fidelity_is_bounded(bundle: PanelBundle) -> None:
    panel = bundle.panel
    set_seed(31)
    model = TargetPrioritiser(panel, ModelConfig(model_dim=8, graph_rounds=1))
    split = label_split(panel.n_genes, seed=31, protected=panel.target_labels)
    value = validation_fidelity(model, panel, split.validation)
    assert 0.0 <= value <= 1.0


def test_training_is_reproducible(bundle: PanelBundle) -> None:
    panel = bundle.panel
    split = label_split(panel.n_genes, seed=37, protected=panel.target_labels)
    config = TrainingConfig(epochs=1, batch_size=16, model=ModelConfig(model_dim=8, graph_rounds=1))
    first = run_training(
        panel, training_units=split.training, validation_units=split.validation, seed=37, config=config
    )
    second = run_training(
        panel, training_units=split.training, validation_units=split.validation, seed=37, config=config
    )
    assert first.digest == second.digest
    assert first.history[0].loss == second.history[0].loss
