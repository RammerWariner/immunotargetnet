"""Render the shortlist dossier: one evidence record per nomination.

Ref: Sec. 3.6, Table 5 and Figure 4.

Every nomination carries the programme that claims its score, the per-layer attribution,
and the evidence that is missing from the public resources, because a list that does not
state what is absent is not a usable list.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from immunotargetnet.bottleneck.attention import availability_tensor
from immunotargetnet.evidence.io import AssetPaths, build_panel
from immunotargetnet.evidence.schema import MODALITY_ORDER
from immunotargetnet.graph.degree import fit_degree_control, top_decile_mask
from immunotargetnet.metrics.ranking import ranking_order
from immunotargetnet.protocols.splits import label_split
from immunotargetnet.protocols.study import spec_by_key
from immunotargetnet.runtime.atomic import write_json_atomic, write_text_atomic
from immunotargetnet.runtime.logging_setup import configure_logging, get_logger
from immunotargetnet.training.loop import TrainingConfig, run_training

LOGGER = get_logger("cli.prioritise")


def missing_evidence_note(panel: object, index: int) -> str:
    from immunotargetnet.evidence.schema import EvidencePanel

    if not isinstance(panel, EvidencePanel):
        raise TypeError("panel must be an EvidencePanel")
    absent = [
        MODALITY_ORDER[column]
        for column in range(panel.availability.shape[1])
        if not panel.availability[index, column]
    ]
    if not absent:
        return "none"
    return "no " + ", ".join(absent) + " evidence for this gene"


def dossier(panel: object, *, top_n: int, seed: int, epochs: int) -> dict[str, object]:
    from immunotargetnet.evidence.schema import EvidencePanel

    if not isinstance(panel, EvidencePanel):
        raise TypeError("panel must be an EvidencePanel")
    split = label_split(panel.n_genes, seed=seed, protected=panel.target_labels)
    spec = spec_by_key("full")
    result = run_training(
        panel,
        training_units=split.training,
        validation_units=split.validation,
        seed=seed,
        config=TrainingConfig(epochs=epochs, batch_size=32, model=spec.model),
    )
    with torch.no_grad():
        output = result.model(panel, availability_tensor(panel.availability))
    scores = np.asarray(output.relevance.detach().cpu().numpy(), dtype=np.float64)
    readiness = np.asarray(output.tractability_probability().detach().cpu().numpy(), dtype=np.float64)
    programmes = np.asarray(output.programmes.detach().cpu().numpy(), dtype=np.float64)
    mass = np.asarray(output.modality_mass.detach().cpu().numpy(), dtype=np.float64)
    control = fit_degree_control(scores, panel.interaction_degree)
    hub = top_decile_mask(panel.interaction_degree)
    order = ranking_order(scores)[:top_n]
    entries: list[dict[str, object]] = []
    for rank, index in enumerate(order, start=1):
        entries.append(
            {
                "rank": rank,
                "gene": panel.genes[index],
                "relevance": round(float(scores[index]), 4),
                "readiness": round(float(readiness[index]), 4),
                "carrying_programme": panel.programme_labels[int(np.argmax(programmes[index]))],
                "programme_weights": {
                    label: round(float(programmes[index, position]), 4)
                    for position, label in enumerate(panel.programme_labels)
                },
                "layer_attribution": {
                    name: round(float(mass[index, column]), 4) for column, name in enumerate(MODALITY_ORDER)
                },
                "degree_adjusted_rank": int(control.degree_then_rank[index]),
                "in_top_degree_decile": bool(hub[index]),
                "carries_target_label": bool(panel.target_labels[index]),
                "clinical_stage_tractability": bool(panel.tractability[index]),
                "missing_evidence": missing_evidence_note(panel, index),
            }
        )
    return {
        "seed": seed,
        "epochs": epochs,
        "n_candidates": panel.n_genes,
        "control_slope": round(control.slope, 6),
        "control_intercept": round(control.intercept, 6),
        "entries": entries,
    }


def render_dossier(payload: dict[str, object]) -> str:
    entries = payload["entries"]
    if not isinstance(entries, list):
        raise TypeError("dossier entries must be a list")
    lines = [
        "ImmunoTargetNet shortlist dossier",
        f"seed\t{payload['seed']}",
        f"epochs\t{payload['epochs']}",
        f"candidate_pool\t{payload['n_candidates']}",
        f"degree_control_slope\t{payload['control_slope']}",
        "",
        "rank\tgene\trelevance\treadiness\tprogramme\tdegree_adjusted_rank\tclinical_stage\tmissing_evidence",
    ]
    for entry in entries:
        lines.append(
            "\t".join(
                str(entry[key])
                for key in (
                    "rank",
                    "gene",
                    "relevance",
                    "readiness",
                    "carrying_programme",
                    "degree_adjusted_rank",
                    "clinical_stage_tractability",
                    "missing_evidence",
                )
            )
        )
    lines.append("")
    lines.append("layer attribution (mean attention mass per layer)")
    header = "gene\t" + "\t".join(MODALITY_ORDER)
    lines.append(header)
    for entry in entries:
        attribution = entry["layer_attribution"]
        if not isinstance(attribution, dict):
            raise TypeError("layer attribution must be a mapping")
        lines.append("\t".join([str(entry["gene"])] + [str(attribution[name]) for name in MODALITY_ORDER]))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render the shortlist dossier.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--use-cohorts", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    paths = AssetPaths(root=args.root)
    bundle = build_panel(paths, seed=args.seed, use_cohorts=args.use_cohorts)
    payload = dossier(bundle.panel, top_n=args.top, seed=args.seed, epochs=args.epochs)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        args.out / f"shortlist_seed{args.seed}.json",
        payload,
    )
    write_text_atomic(args.out / f"shortlist_seed{args.seed}.txt", render_dossier(payload))
    entries = payload["entries"]
    count = len(entries) if isinstance(entries, list) else 0
    LOGGER.info("wrote shortlist for %d nominations", count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
