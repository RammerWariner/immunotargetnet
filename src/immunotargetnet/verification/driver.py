"""Verification driver: run both passes and write every root artefact.

Pass one is the paper-claim to code mapping; pass two executes the pipeline and checks the
manuscript's own arithmetic. The integrity manifest is written last and excludes only
itself.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from immunotargetnet.runtime.atomic import write_json_atomic, write_text_atomic
from immunotargetnet.runtime.logging_setup import configure_logging, get_logger
from immunotargetnet.verification import claimed_values as claimed
from immunotargetnet.verification.checks import (
    CODE_CATEGORIES,
    MANUSCRIPT_CATEGORY,
    NOT_RUN,
    CheckResult,
    overall_status,
    summarise,
    verdict,
)
from immunotargetnet.verification.claims import MAPPINGS
from immunotargetnet.verification.execution import run_execution_checks
from immunotargetnet.verification.hygiene import run_hygiene_checks
from immunotargetnet.verification.manifest import MANIFEST_NAME, compare_with_tree, write_manifest
from immunotargetnet.verification.manuscript import run_manuscript_checks

LOGGER = get_logger("verification.driver")

PACKAGE_NAME: Final[str] = "immunotargetnet"
REPORT_NAME: Final[str] = "verification_report.json"
CLAIM_MAP_NAME: Final[str] = "claim_to_code.json"
SUMMARY_NAME: Final[str] = "verification_summary.txt"
BOUNDED_STUDY_EPOCHS: Final[int] = 1
BOUNDED_STABILITY_DRAWS: Final[int] = 1

DEVIATIONS: Final[tuple[dict[str, str], ...]] = (
    {
        "paper_anchor": claimed.SECTION_2_3_ANCHOR + " / Algorithm 1 line 10",
        "departure": "Eq. (2) writes the additive mask as log m_{g,j} and Algorithm 1 line 10 writes the same term as log M. The release implements the Eq. (2) form, because a positive constant cannot mask anything.",
        "justification": "the two statements cannot both hold; the equation is the one the ranking head depends on.",
    },
    {
        "paper_anchor": claimed.SECTION_2_1_ANCHOR + " / Sec. 2.3",
        "departure": "Sec. 2.1 gives the availability mask over the L encoder tokens while Algorithm 1 loops j over the M evidence layers. The release indexes the mask over the four evidence layers and derives per-token availability from it.",
        "justification": "a four-layer index keeps the mask and the layer count of Sec. 2.1 consistent, and it is the index space Algorithm 1 actually reads.",
    },
    {
        "paper_anchor": claimed.SECTION_2_3_ANCHOR + " / Algorithm 1 line 9",
        "departure": "Sec. 2.3 states that the keystone layer is the graph-context embedding, so Algorithm 1's extra append of a context token would duplicate it. The release builds four layer tokens and does not append a fifth.",
        "justification": "four tokens preserve the M = 4 layer budget of Sec. 2.1 and the token count the attention of Eq. (2) is defined over.",
    },
    {
        "paper_anchor": claimed.SECTION_2_1_ANCHOR,
        "departure": "The readiness head is a softmax over readiness tiers while L_read is a binary cross-entropy against tractability annotations. The release fixes the tier count at two, which makes the softmax and the binary cross-entropy the same object.",
        "justification": "the paper does not state the tier count; two is the value that removes the contradiction without changing the printed architecture.",
    },
    {
        "paper_anchor": claimed.SECTION_2_1_ANCHOR,
        "departure": "The curated non-coding list on the IL23/IL17 axis is not published. The release ships a named twelve-gene curation in evidence/programmes.py.",
        "justification": "the pool needs a non-coding member to be the object Sec. 2.1 describes; the curation is declared rather than inferred.",
    },
    {
        "paper_anchor": claimed.SECTION_2_4_ANCHOR,
        "departure": "The twelve programmes are assembled from named gene-set collections of the public immune corpora (GO biological process, Reactome, Hallmark) instead of the unpublished curated list.",
        "justification": "Sec. 2.4 states the programmes were assembled from public immune gene-set collections; the collection identifiers are pinned so the table is rebuildable.",
    },
    {
        "paper_anchor": claimed.SECTION_2_2_ANCHOR,
        "departure": "The eight expression cohorts, the single-cell resources and the spatial resources are public but are not redistributed. The release ships a schema-compatible fixture whose declared construct generates the panel, and takes the programme table, the tractability labels and the typed edges from the frozen keystone snapshot.",
        "justification": "no manuscript table value is claimed from the fixture; every cohort-level value is reported NOT_RUN with the reason recorded.",
    },
    {
        "paper_anchor": claimed.SECTION_2_2_ANCHOR,
        "departure": "The compound perturbation reference of the perturbation layer is not redistributable, so the perturbation-reversal index is part of the declared construct rather than a read of that resource.",
        "justification": "the consistency term keeps its structure and its sign semantics; only the index source departs.",
    },
    {
        "paper_anchor": claimed.SECTION_2_2_ANCHOR,
        "departure": "Interaction edges are read through the platform's interactions schema, whose sourceDatabase field names the originating resource, rather than through a direct download of the interaction network.",
        "justification": "the frozen snapshot records the originating database of every edge, so the edge typing is preserved and auditable.",
    },
    {
        "paper_anchor": claimed.SECTION_2_5_ANCHOR,
        "departure": "The interaction resolver returned no edge during the capture sweep, so the frozen snapshot's interaction family is empty and the typed graph's gene-to-gene structure is absent; the graph keeps its annotation families, which include the pathway nodes of the pathway corpus.",
        "justification": "the empty sweep is recorded in the snapshot itself with the query count and the budget that bounded it, so a reader sees what was collected rather than an assumption; the refresh path repopulates the family.",
    },
    {
        "paper_anchor": claimed.SECTION_3_4_ANCHOR,
        "departure": "The degree control reads a candidate's distinct partner count over all typed families, not interaction edges alone, because the frozen snapshot carries no interaction edge.",
        "justification": "the quantity stays the same kind of thing the manuscript controls for; the definition is recorded here rather than left implicit.",
    },
    {
        "paper_anchor": claimed.SECTION_2_1_ANCHOR,
        "departure": "The manuscript does not report the optimiser weight decay, the warm-up schedule, gradient clipping, the readiness tier count, the model width, the graph-context dimension or the number of attention heads. The release exposes each of them in config with a declared default.",
        "justification": "these are engineering defaults, labelled as such, not reported values.",
    },
    {
        "paper_anchor": claimed.TABLE1_ANCHOR,
        "departure": "The held-out curated target set of 148 targets is not public. The released panel carries a declared target label derived from its own construct.",
        "justification": "the evaluation stack, the statistics and the ranking metrics are exercised end to end; the manuscript's recovery values remain NOT_RUN.",
    },
)


@dataclass(frozen=True)
class GateOutcome:
    label: str
    exit_code: int

    def record(self) -> dict[str, object]:
        return {"gate": self.label, "exit_code": self.exit_code, "passed": self.exit_code == 0}


def _gate_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["COLUMNS"] = "120"
    environment["LINES"] = "40"
    environment["NO_COLOR"] = "1"
    environment["PY_COLORS"] = "0"
    # Bytecode carries the compiling source's absolute path, so a gate must not leave one.
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def run_gate(label: str, command: list[str], root: Path, *, timeout: int = 1800) -> GateOutcome:
    completed = subprocess.run(
        command,
        cwd=str(root),
        env=_gate_environment(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    LOGGER.info("gate %s exited %d", label, completed.returncode)
    return GateOutcome(label=label, exit_code=int(completed.returncode))


def gate_checks(outcomes: list[GateOutcome]) -> list[CheckResult]:
    results: list[CheckResult] = []
    for outcome in outcomes:
        results.append(
            CheckResult(
                name=f"gate-{outcome.label}",
                category="gate",
                status="PASS" if outcome.exit_code == 0 else "FAIL",
                detail=f"{outcome.label} exited with code {outcome.exit_code}",
                evidence={"exit_code": outcome.exit_code},
            )
        )
    return results


def default_gates(root: Path) -> list[tuple[str, list[str]]]:
    python = sys.executable
    # Every gate runs with its own cache disabled: a cache directory records the tree it was
    # built from, and a release must not ship a record of its build host.
    return [
        ("ruff", [python, "-m", "ruff", "check", "--no-cache", "."]),
        ("black", [python, "-m", "black", "--check", "."]),
        ("isort", [python, "-m", "isort", "--check-only", "."]),
        (
            "mypy",
            [python, "-m", "mypy", "--strict", "--cache-dir=/dev/null", f"src/{PACKAGE_NAME}"],
        ),
        ("pytest", [python, "-m", "pytest", "-q", "-p", "no:cacheprovider"]),
    ]


def manifest_freshness(root: Path) -> CheckResult:
    path = root / MANIFEST_NAME
    if not path.exists():
        return CheckResult(
            name="integrity-manifest-freshness",
            category="provenance",
            status=NOT_RUN,
            detail="the manifest is written at the end of this run, so a clean tree has none to compare against",
        )
    import json

    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    comparison = compare_with_tree(root, payload)
    return CheckResult(
        name="integrity-manifest-freshness",
        category="provenance",
        status="PASS" if comparison["digest_matches"] else "FAIL",
        detail=(
            "the shipped manifest matches the live tree entry for entry"
            if comparison["digest_matches"]
            else "the shipped manifest disagrees with the live tree"
        ),
        evidence=comparison,
    )


def build_claim_map() -> dict[str, Any]:
    return {
        "package": PACKAGE_NAME,
        "manuscript": {
            "title": "ImmunoTargetNet: A Multimodal Multi-Omics Framework for Immune-Related Target Prioritisation in Psoriasis",
            "venue": "Frontiers (structured abstract, Frontiers citation style and running footer)",
            "figures": ["Figure 1", "Figure 2", "Figure 3", "Figure 4"],
            "tables": ["Table 1", "Table 2", "Table 3", "Table 4", "Table 5", "Table 6"],
            "equations": ["Eq. (1)", "Eq. (2)", "Eq. (3)", "Eq. (4)"],
            "algorithms": ["Algorithm 1", "Algorithm 2"],
            "supplementary": "Tables S1-S13 are referenced but not distributed with the manuscript",
        },
        "claims": [
            {
                "claim_id": item.claim_id,
                "statement": item.statement,
                "paper_anchor": item.paper_anchor,
                "code_locations": list(item.code_locations),
                "implemented_as": item.implemented_as,
                "verified_by": list(item.verified_by),
                "status": item.status,
            }
            for item in MAPPINGS
        ],
        "deviations": list(DEVIATIONS),
        "transcribed_values": dict(claimed.REPORTED),
    }


def build_report(
    *,
    root: Path,
    results: list[CheckResult],
    gates: list[GateOutcome],
) -> dict[str, Any]:
    code_status = verdict(results, CODE_CATEGORIES)
    manuscript_status = verdict(results, (MANUSCRIPT_CATEGORY,))
    overall = overall_status(code_status, manuscript_status)
    code_failing = [
        result.record()
        for result in results
        if result.category in CODE_CATEGORIES and result.status == "FAIL"
    ]
    manuscript_failing = [
        result.record()
        for result in results
        if result.category == MANUSCRIPT_CATEGORY and result.status == "FAIL"
    ]
    outstanding = [result.record() for result in results if result.status in {NOT_RUN, "BLOCKED"}]
    return {
        "overall_status": overall,
        "code_status": code_status,
        "manuscript_status": manuscript_status,
        "tally": summarise(results),
        "checks": [result.record() for result in results],
        "gates": [gate.record() for gate in gates],
        "code_failures": code_failing,
        "manuscript_discrepancies": manuscript_failing,
        "outstanding": outstanding,
        "interpretation": (
            "A failing check in the manuscript family records a disagreement inside the article's own "
            "tables or text, not a defect in this release. A failing check in a code family is a defect "
            "in the release. Every table value that needs the manuscript's cohorts stays NOT_RUN because "
            "those series matrices are not redistributed."
        ),
    }


def render_summary(report: dict[str, Any], root: Path) -> str:
    lines = [
        "ImmunoTargetNet verification summary",
        f"root\t{root.resolve().name}",
        f"overall_status\t{report['overall_status']}",
        f"code_status\t{report['code_status']}",
        f"manuscript_status\t{report['manuscript_status']}",
        "",
        "tally",
    ]
    for status, count in sorted(report["tally"].items()):
        lines.append(f"{status}\t{count}")
    lines.append("")
    lines.append("gates")
    for gate in report["gates"]:
        lines.append(f"{gate['gate']}\texit_code={gate['exit_code']}")
    lines.append("")
    lines.append(f"code_failures\t{len(report['code_failures'])}")
    for record in report["code_failures"]:
        lines.append(f"{record['name']}\t{record['detail']}")
    lines.append("")
    lines.append(f"manuscript_discrepancies\t{len(report['manuscript_discrepancies'])}")
    for record in report["manuscript_discrepancies"]:
        lines.append(f"{record['name']}\t{record['detail']}")
    lines.append("")
    lines.append(
        "A discrepancy inside the manuscript's own tables is not a defect in this release; it is kept "
        "as FAIL so the disagreement is visible in the record."
    )
    lines.append("")
    lines.append(f"outstanding\t{len(report['outstanding'])}")
    return "\n".join(lines) + "\n"


def run_verification(
    root: Path,
    *,
    run_tools: bool = True,
    run_study: bool = True,
    epochs: int = BOUNDED_STUDY_EPOCHS,
    stability_draws: int = BOUNDED_STABILITY_DRAWS,
) -> dict[str, Any]:
    gates: list[GateOutcome] = []
    results: list[CheckResult] = []
    if run_tools:
        for label, command in default_gates(root):
            gates.append(run_gate(label, command, root))
        results.extend(gate_checks(gates))
    else:
        results.append(
            CheckResult(
                name="gate-suite",
                category="gate",
                status=NOT_RUN,
                detail="the tooling gates were skipped for this pass",
            )
        )
    results.extend(
        run_execution_checks(root, epochs=epochs, stability_draws=stability_draws, include_study=run_study)
    )
    results.extend(run_hygiene_checks(root))
    results.extend(run_manuscript_checks())
    results.append(manifest_freshness(root))
    return build_report(root=root, results=results, gates=gates)


def write_artefacts(root: Path, report: dict[str, Any]) -> dict[str, Path]:
    """Write the claim map, the report, the summary and finally the manifest."""
    claim_path = write_json_atomic(root / CLAIM_MAP_NAME, build_claim_map())
    report_path = write_json_atomic(root / REPORT_NAME, report)
    summary = render_summary(report, root)
    summary_path = write_text_atomic(root / SUMMARY_NAME, summary)
    manifest_path = write_manifest(root)
    return {
        "claim_map": claim_path,
        "report": report_path,
        "summary": summary_path,
        "manifest": manifest_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the release and write the root artefacts.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--skip-tools", action="store_true")
    parser.add_argument("--skip-study", action="store_true", help="iterate without the study pipeline")
    parser.add_argument("--epochs", type=int, default=BOUNDED_STUDY_EPOCHS)
    parser.add_argument("--stability-draws", type=int, default=BOUNDED_STABILITY_DRAWS)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    report = run_verification(
        args.root,
        run_tools=not args.skip_tools,
        run_study=not args.skip_study,
        epochs=args.epochs,
        stability_draws=args.stability_draws,
    )
    written = write_artefacts(args.root, report)
    LOGGER.info(
        "overall %s, code %s, manuscript %s",
        report["overall_status"],
        report["code_status"],
        report["manuscript_status"],
    )
    for name, path in written.items():
        LOGGER.info("wrote %s (%s)", path.name, name)
    return 0 if report["code_status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
