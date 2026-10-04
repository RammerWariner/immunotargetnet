"""Train one configuration and write its checkpoint and run report.

Ref: Sec. 2.1 (training schedule) and Sec. 2.7.4 (reproducibility record).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from immunotargetnet.evidence.io import AssetPaths, build_panel
from immunotargetnet.protocols.splits import label_split
from immunotargetnet.protocols.study import spec_by_key
from immunotargetnet.runtime.atomic import write_text_atomic
from immunotargetnet.runtime.logging_setup import configure_logging, get_logger
from immunotargetnet.training.loop import EpochRecord, TrainingConfig, run_training

LOGGER = get_logger("cli.train")


def _format_history(history: list[EpochRecord]) -> list[str]:
    lines = ["epoch\tloss\tranking\timmune\tperturbation\tvalidation_fidelity"]
    for record in history:
        lines.append(
            f"{record.epoch}\t{record.loss:.6f}\t{record.ranking:.6f}\t{record.immune:.6f}\t{record.perturbation:.6f}\t{record.validation_fidelity:.6f}"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one reported configuration.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--configuration", default="full")
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    paths = AssetPaths(root=args.root)
    bundle = build_panel(paths, seed=args.seed)
    spec = spec_by_key(args.configuration)
    split = label_split(bundle.panel.n_genes, seed=args.seed, protected=bundle.panel.target_labels)
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        modality_dropout_rate=spec.dropout_rate,
        weights=spec.weights,
        model=spec.model,
    )
    checkpoint = args.out / f"{args.configuration}_seed{args.seed}.pt"
    result = run_training(
        bundle.panel,
        training_units=split.training,
        validation_units=split.validation,
        seed=args.seed,
        config=config,
        checkpoint_path=checkpoint,
    )
    report = [
        f"configuration: {spec.label}",
        f"seed: {args.seed}",
        f"epochs_requested: {args.epochs}",
        f"epochs_run: {len(result.history)}",
        f"batch_size: {args.batch_size}",
        f"best_epoch: {result.best_epoch}",
        f"best_validation_fidelity: {result.best_validation_fidelity:.6f}",
        f"trainable_parameters: {result.n_parameters}",
        f"checkpoint: {checkpoint.name}",
        f"state_digest: {result.digest}",
        "",
        *_format_history(result.history),
    ]
    target = args.out / f"train_{args.configuration}_seed{args.seed}.txt"
    write_text_atomic(target, "\n".join(report) + "\n")
    LOGGER.info("wrote %s", target.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
