"""Run the study and write every reported table as plain text.

Ref: Sec. 3.1-3.8 and Tables 1-6. Tables are written as tab-separated blocks under
`runs/`; nothing is written as HTML or markdown.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from immunotargetnet.evidence.io import AssetPaths, build_panel
from immunotargetnet.protocols.study import StudyConfig, StudyOutcome, run_study
from immunotargetnet.runtime.atomic import write_text_atomic
from immunotargetnet.runtime.logging_setup import configure_logging, get_logger

LOGGER = get_logger("cli.evaluate")


def _render_table(title: str, rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return [title, "(no rows)", ""]
    columns = list(rows[0])
    lines = [title, "\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row.get(column, "")) for column in columns))
    lines.append("")
    return lines


def _render_scalars(title: str, values: dict[str, Any]) -> list[str]:
    lines = [title]
    for key, value in values.items():
        rendered = f"{value:.6g}" if isinstance(value, float) else str(value)
        lines.append(f"{key}\t{rendered}")
    lines.append("")
    return lines


def render_outcome(outcome: StudyOutcome) -> str:
    lines: list[str] = ["ImmunoTargetNet study output", ""]
    lines += _render_scalars("run provenance", dict(outcome.provenance))
    lines += _render_table("Table 1. Recovery, readiness and constraint axis", outcome.table1)
    lines += _render_table("Table 2a. Per-cohort recovery", outcome.table2_cohorts)
    lines += _render_table("Table 2b. Layer resolution", outcome.table2_resolution)
    lines += _render_table("Table 3. Generalisation, degradation and stability", outcome.table3_settings)
    lines += _render_table("Table 4. Ablation", outcome.table4_ablations)
    lines += _render_table("Table 4. Interaction ratios", outcome.interaction_ratios)
    lines += _render_table("Table 5. Shortlist", outcome.shortlist)
    lines += _render_scalars(
        "Table 5. Degree control", {"shortlist_degree_gap": outcome.shortlist_degree_gap}
    )
    lines += _render_scalars("Table 6. Cost", dict(outcome.cost))
    lines += ["notes"] + [f"- {note}" for note in outcome.notes] + [""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the study and write the reported tables.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--stability-draws", type=int, default=None)
    parser.add_argument("--use-cohorts", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    paths = AssetPaths(root=args.root)
    bundle = build_panel(paths, seed=args.seed, use_cohorts=args.use_cohorts)
    overrides: dict[str, int] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.seeds is not None:
        overrides["seeds"] = args.seeds
    if args.stability_draws is not None:
        overrides["stability_draws"] = args.stability_draws
    config = replace(StudyConfig(), **overrides)
    outcome = run_study(bundle.panel, config=config, seed=args.seed)
    body = render_outcome(outcome)
    target = args.out / f"study_seed{args.seed}_epochs{config.epochs}.txt"
    write_text_atomic(target, body)
    LOGGER.info("wrote %s", target.name)
    LOGGER.info("panel provenance: %s", asdict(bundle.provenance)["cohort_layer"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
