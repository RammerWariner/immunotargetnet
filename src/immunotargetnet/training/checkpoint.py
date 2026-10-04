"""Atomic checkpointing with the run seed recorded beside the weights.

Ref: Sec. 2.7.4 (five independent runs, each reproducible from its own seed).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FORMAT: str = "immunotargetnet-checkpoint-1"


@dataclass
class CheckpointPayload:
    epoch: int
    seed: int
    model_state: dict[str, Any]
    optimiser_state: dict[str, Any] | None
    scheduler_state: dict[str, Any] | None
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    format: str = CHECKPOINT_FORMAT


def save_checkpoint(path: Path, payload: CheckpointPayload) -> Path:
    """Write through a temporary file so an interrupted run cannot leave a torn checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".pt")
    os.close(handle)
    temporary = Path(name)
    try:
        torch.save(
            {
                "format": payload.format,
                "epoch": payload.epoch,
                "seed": payload.seed,
                "model_state": payload.model_state,
                "optimiser_state": payload.optimiser_state,
                "scheduler_state": payload.scheduler_state,
                "metrics": payload.metrics,
                "config": payload.config,
            },
            temporary,
        )
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_checkpoint(path: Path, *, model: torch.nn.Module | None = None) -> CheckpointPayload:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unexpected checkpoint format: {payload.get('format')}")
    if model is not None:
        model.load_state_dict(payload["model_state"])
    return CheckpointPayload(
        epoch=int(payload["epoch"]),
        seed=int(payload["seed"]),
        model_state=payload["model_state"],
        optimiser_state=payload.get("optimiser_state"),
        scheduler_state=payload.get("scheduler_state"),
        metrics={str(key): float(value) for key, value in payload.get("metrics", {}).items()},
        config=payload.get("config", {}),
        format=str(payload["format"]),
    )


def state_dict_digest(model: torch.nn.Module) -> str:
    """Content digest of the tensor payloads.

    Container metadata differs between two byte-identical writes of the same weights, so
    the digest is taken over the tensors rather than over the file.
    """
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
