"""Refresh the frozen programme table and keystone snapshot.

Ref: Sec. 2.2 and Sec. 2.4. Run with --refresh to reach the public sources; the shipped
snapshot is what every other entry point reads.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Final

from immunotargetnet.evidence.keystone import (
    SNAPSHOT_NAME,
    KeystoneSnapshot,
    msigdb_provenance,
    open_targets_provenance,
)
from immunotargetnet.evidence.msigdb import GeneSetCorpus, ProgrammeTable
from immunotargetnet.evidence.opentargets import OpenTargetsClient
from immunotargetnet.evidence.pathways import PATHWAY_TABLE_NAME, PathwayTable
from immunotargetnet.evidence.programmes import (
    MSIGDB_COLLECTIONS,
    MSIGDB_RELEASE,
    NONCODING_AXIS_GENES,
)
from immunotargetnet.evidence.schema import SCHEMA_VERSION
from immunotargetnet.runtime.atomic import write_json_atomic
from immunotargetnet.runtime.logging_setup import configure_logging, get_logger

PROGRAMME_TABLE_NAME: Final[str] = "programme_table.json"
DEFAULT_MIN_PROGRAMMES: Final[int] = 3
DEFAULT_BUDGET_SECONDS: Final[float] = 900.0
DEFAULT_MAX_QUERIES: Final[int] = 400


def corpus_from_roots(roots: dict[str, Path]) -> GeneSetCorpus:
    directories = {tag: roots[tag] for tag in MSIGDB_COLLECTIONS}
    missing = [str(path) for path in directories.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("gene-set corpus missing: " + ", ".join(missing))
    return GeneSetCorpus.read(directories, MSIGDB_RELEASE)


def write_programme_table(table: ProgrammeTable, path: Path) -> Path:
    payload = {
        "release": MSIGDB_RELEASE,
        "collections": MSIGDB_COLLECTIONS,
        "programmes": [
            {
                "key": key,
                "display_name": display,
                "members": list(table.members[key]),
            }
            for key, display in zip(table.keys, table.display_names, strict=True)
        ],
        "noncoding_axis_genes": list(NONCODING_AXIS_GENES),
    }
    return write_json_atomic(path, payload)


def build(
    *,
    assets_dir: Path,
    corpus_roots: dict[str, Path],
    min_programmes: int = DEFAULT_MIN_PROGRAMMES,
    refresh: bool = False,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    max_queries: int = DEFAULT_MAX_QUERIES,
) -> tuple[Path, Path | None]:
    logger = get_logger("build_evidence")
    corpus = corpus_from_roots(corpus_roots)
    table = ProgrammeTable.resolve(corpus)
    table_path = write_programme_table(table, assets_dir / PROGRAMME_TABLE_NAME)
    pathway_table = PathwayTable.from_corpus(dict(corpus.members_of("REACTOME")), table.universe())
    pathway_table.write(assets_dir / PATHWAY_TABLE_NAME)
    logger.info(
        "pathway table written: %d pathways over %d genes",
        len(pathway_table.members),
        len(pathway_table.universe()),
    )
    logger.info("programme table written: %d programmes, %d genes", len(table.keys), len(table.universe()))
    if not refresh:
        logger.info("snapshot refresh not requested; programme table only")
        return table_path, None

    client = OpenTargetsClient()
    symbols = sorted(set(table.universe()) | set(NONCODING_AXIS_GENES))
    logger.info("resolving %d symbols through the platform", len(symbols))
    resolved = client.map_symbols(symbols)
    unresolved = sorted(set(symbols) - set(resolved))
    logger.info("resolved %d symbols, %d unresolved", len(resolved), len(unresolved))

    identifiers = sorted(resolved.values())
    annotations = client.annotations(identifiers)
    incidence = table.incidence()
    panel_ids = sorted(
        annotation.ensembl_id
        for annotation in annotations.values()
        if incidence.get(annotation.symbol, 0) >= min_programmes
    )
    universe_ids = set(identifiers)
    logger.info("collecting typed edges for %d panel genes", len(panel_ids))
    edges, failed, skipped = client.typed_edges(
        panel_ids,
        partner_universe=universe_ids,
        budget_seconds=budget_seconds,
        max_queries=max_queries,
    )
    logger.info(
        "collected %d typed edges across %d edge types", len(edges), len({e.edge_type for e in edges})
    )

    snapshot = KeystoneSnapshot(
        schema_version=SCHEMA_VERSION,
        retrieved_on=date.today().isoformat(),
        provenance=[open_targets_provenance(date.today().isoformat()), msigdb_provenance(MSIGDB_RELEASE)],
        annotations=annotations,
        noncoding=list(NONCODING_AXIS_GENES),
        edges=edges,
        unresolved_symbols=unresolved,
        failed_interaction_queries=failed,
        unqueried_interaction_genes=skipped,
        notes=[
            "Interaction rows are served by the platform's interactions schema and carry the "
            "originating database in sourceDatabase, which is the edge type of the typed graph.",
            f"The interaction panel is the subset of the candidate pool present in at least {min_programmes} "
            "programmes; the paper does not publish its panel rule.",
            f"The interaction sweep is bounded at {max_queries} queries and {budget_seconds:.0f} s; "
            f"{len(skipped)} panel genes were left unqueried and {len(failed)} queries failed.",
            (
                "The interaction family of the typed graph is populated with the edges this sweep "
                f"returned ({len(edges)} edges)."
                if edges
                else "The interaction resolver returned no edge during the sweep, so the typed graph "
                "carries the annotation families only (programme membership, pathway membership, "
                "population attribution, compartment localisation, tractability annotation). "
                "Re-run the refresh path to populate the interaction family."
            ),
        ],
    )
    snapshot_path = snapshot.write(assets_dir / SNAPSHOT_NAME)
    logger.info("snapshot written: %s", snapshot_path.name)
    return table_path, snapshot_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh the programme table and keystone snapshot.")
    root = Path(__file__).resolve().parents[3]
    parser.add_argument("--assets", type=Path, default=root / "assets")
    parser.add_argument("--corpus-root", type=Path, default=root / "data" / "msigdb", dest="corpus_root")
    parser.add_argument("--min-programmes", type=int, default=DEFAULT_MIN_PROGRAMMES)
    parser.add_argument("--budget-seconds", type=float, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--max-queries", type=int, default=DEFAULT_MAX_QUERIES)
    parser.add_argument("--refresh", action="store_true", help="reach the public sources for a new snapshot")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    roots = {tag: args.corpus_root / tag for tag in MSIGDB_COLLECTIONS}
    build(
        assets_dir=args.assets,
        corpus_roots=roots,
        min_programmes=args.min_programmes,
        refresh=args.refresh,
        budget_seconds=args.budget_seconds,
        max_queries=args.max_queries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
