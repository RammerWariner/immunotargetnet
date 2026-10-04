"""Asset loading and panel assembly for the entry points.

Ref: Sec. 2.1 (the panel the model consumes) and Sec. 2.7.1 (cohort-level splits).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from immunotargetnet.evidence.cohort import COHORT_SPECS, CohortSpec, ContrastResult, contrast_from_matrix
from immunotargetnet.evidence.geo import read_series_matrix
from immunotargetnet.evidence.keystone import SNAPSHOT_NAME, KeystoneSnapshot
from immunotargetnet.evidence.msigdb import ProgrammeTable
from immunotargetnet.evidence.panel import (
    DEFAULT_CONTEXT_DIM,
    PanelBundle,
    build_panel_from_cohorts,
    build_panel_from_fixture,
)
from immunotargetnet.evidence.pathways import PATHWAY_TABLE_NAME, PathwayTable
from immunotargetnet.runtime.logging_setup import get_logger

PROGRAMME_TABLE_NAME: Final[str] = "programme_table.json"
COHORT_MATRIX_SUFFIX: Final[str] = "_series_matrix.txt.gz"

LOGGER = get_logger("evidence.io")


@dataclass(frozen=True)
class AssetPaths:
    root: Path

    @property
    def assets(self) -> Path:
        return self.root / "assets"

    @property
    def programme_table(self) -> Path:
        return self.assets / PROGRAMME_TABLE_NAME

    @property
    def snapshot(self) -> Path:
        return self.assets / SNAPSHOT_NAME

    @property
    def pathway_table(self) -> Path:
        return self.assets / PATHWAY_TABLE_NAME

    @property
    def cohort_root(self) -> Path:
        return self.root / "data" / "cohorts"

    def require(self) -> None:
        missing = [
            str(path)
            for path in (self.programme_table, self.snapshot, self.pathway_table)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("missing release assets: " + ", ".join(missing))


def load_programme_table(path: Path) -> ProgrammeTable:
    with path.open("r", encoding="utf-8") as stream:
        payload: dict[str, Any] = json.load(stream)
    keys: list[str] = []
    names: list[str] = []
    members: dict[str, tuple[str, ...]] = {}
    for record in payload["programmes"]:
        keys.append(str(record["key"]))
        names.append(str(record["display_name"]))
        members[str(record["key"])] = tuple(str(gene) for gene in record["members"])
    if len(keys) != 12:
        raise ValueError(f"the programme table must carry twelve programmes, found {len(keys)}")
    return ProgrammeTable(keys=tuple(keys), display_names=tuple(names), members=members)


def noncoding_axis_genes(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8") as stream:
        payload: dict[str, Any] = json.load(stream)
    return tuple(str(gene) for gene in payload.get("noncoding_axis_genes", []))


def discover_cohort_matrices(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not root.exists():
        return found
    for spec in COHORT_SPECS:
        candidate = root / spec.accession / f"{spec.accession}{COHORT_MATRIX_SUFFIX}"
        if candidate.exists():
            found[spec.accession] = candidate
    return found


def read_cohort_contrasts(root: Path, genes: tuple[str, ...]) -> tuple[list[ContrastResult], list[str]]:
    """Read whatever series matrices are on disk; absent cohorts are reported, not imputed."""
    contrasts: list[ContrastResult] = []
    absent: list[str] = []
    for accession, path in discover_cohort_matrices(root).items():
        spec = next(item for item in COHORT_SPECS if item.accession == accession)
        matrix = read_series_matrix(path, accession=accession)
        contrasts.append(contrast_from_matrix(matrix, spec, genes))
        LOGGER.info("read cohort %s with %d samples", accession, matrix.n_samples)
    for spec in COHORT_SPECS:
        if spec.accession not in discover_cohort_matrices(root):
            absent.append(spec.accession)
    return contrasts, absent


def load_pathway_table(path: Path) -> PathwayTable:
    return PathwayTable.read(path)


def build_panel(
    paths: AssetPaths,
    *,
    seed: int,
    context_dim: int = DEFAULT_CONTEXT_DIM,
    use_cohorts: bool = False,
) -> PanelBundle:
    """Assemble the panel from the shipped assets, preferring distributed cohorts when present."""
    paths.require()
    table = load_programme_table(paths.programme_table)
    snapshot = KeystoneSnapshot.read(paths.snapshot)
    pathways = load_pathway_table(paths.pathway_table)
    if use_cohorts:
        contrasts, absent = read_cohort_contrasts(paths.cohort_root, table.universe())
        if contrasts:
            if absent:
                LOGGER.warning("cohorts absent from disk: %s", ",".join(absent))
            return build_panel_from_cohorts(
                snapshot, table, pathways, contrasts, seed=seed, context_dim=context_dim
            )
        LOGGER.warning("no series matrix on disk; falling back to the panel builder")
    return build_panel_from_fixture(snapshot, table, pathways, seed=seed, context_dim=context_dim)


def cohort_specs() -> tuple[CohortSpec, ...]:
    return COHORT_SPECS
