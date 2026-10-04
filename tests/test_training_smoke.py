"""Two steps on the smoke configuration, driven by the released file."""

from __future__ import annotations

from pathlib import Path

import yaml

from immunotargetnet.evidence.panel import PanelBundle
from immunotargetnet.objectives.total import ObjectiveWeights
from immunotargetnet.protocols.splits import label_split
from immunotargetnet.training.loop import TrainingConfig, run_training


def load_smoke(release_root: Path) -> dict[str, object]:
    with (release_root / "configs" / "study" / "_smoke.yaml").open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    assert isinstance(payload, dict)
    return payload


def test_smoke_config_is_labelled_and_reduced(release_root: Path) -> None:
    text = (release_root / "configs" / "study" / "_smoke.yaml").read_text(encoding="utf-8")
    assert "smoke only" in text
    payload = load_smoke(release_root)
    assert payload["epochs"] == 2
    assert payload["seeds"] == 1
    primary = yaml.safe_load(
        (release_root / "configs" / "study" / "primary.yaml").read_text(encoding="utf-8")
    )
    assert primary["epochs"] > payload["epochs"]
    assert primary["batch_size"] > payload["batch_size"]


def test_primary_config_carries_the_reported_schedule(release_root: Path) -> None:
    primary = yaml.safe_load(
        (release_root / "configs" / "study" / "primary.yaml").read_text(encoding="utf-8")
    )
    assert primary["batch_size"] == 32
    assert primary["epochs"] == 60
    assert primary["learning_rate"] == 1e-3
    assert primary["modality_dropout_rate"] == 0.2
    assert primary["inference_dropout_rate"] == 0.0
    assert primary["lambda_immune"] == 0.5
    assert primary["lambda_readiness"] == 0.5
    assert primary["lambda_perturbation"] == 0.25
    assert primary["seeds"] == 5


def test_reporting_config_holds_no_manuscript_result(release_root: Path) -> None:
    import re

    from immunotargetnet.verification.claimed_values import TABLE1, TABLE5_SHORTLIST

    reported = {value for row in TABLE1.values() for value in row}
    reported.update(entry[2] for entry in TABLE5_SHORTLIST)
    reported.update(entry[3] for entry in TABLE5_SHORTLIST)
    text = "\n".join(
        (release_root / "configs" / name).read_text(encoding="utf-8")
        for name in ("study/primary.yaml", "bottleneck/attention.yaml", "training/optimiser.yaml")
    )
    collisions = [
        value
        for value in reported
        if re.search(rf"(^|[^0-9.]){re.escape(str(round(value, 4)))}($|[^0-9])", text)
    ]
    assert collisions == []


def test_two_smoke_steps_lower_the_loss(bundle: PanelBundle, release_root: Path) -> None:
    payload = load_smoke(release_root)
    panel = bundle.panel
    split = label_split(panel.n_genes, seed=int(payload["seed"]), protected=panel.target_labels)
    config = TrainingConfig(
        epochs=int(payload["epochs"]),
        batch_size=int(payload["batch_size"]),
        modality_dropout_rate=float(payload["modality_dropout_rate"]),
        learning_rate=float(payload["learning_rate"]),
        weight_decay=float(payload["weight_decay"]),
        model=__import__("immunotargetnet.model", fromlist=["ModelConfig"]).ModelConfig(
            model_dim=int(payload["model_dim"]), graph_rounds=int(payload["n_graph_rounds"])
        ),
        weights=ObjectiveWeights(
            immune=float(payload["lambda_immune"]),
            readiness=float(payload["lambda_readiness"]),
            perturbation=float(payload["lambda_perturbation"]),
        ),
    )
    result = run_training(
        panel,
        training_units=split.training,
        validation_units=split.validation,
        seed=int(payload["seed"]),
        config=config,
    )
    assert len(result.history) == int(payload["epochs"])
    assert result.history[-1].loss < result.history[0].loss
    assert result.history[-1].validation_fidelity >= 0.0
