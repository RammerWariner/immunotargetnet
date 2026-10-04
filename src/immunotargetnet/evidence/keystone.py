"""Frozen keystone-layer snapshot: annotation, tractability and typed interaction edges.

Ref: Sec. 2.2 (keystone evidence layer) and Sec. 2.5 (typed knowledge graph).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from immunotargetnet.evidence.schema import (
    SCHEMA_VERSION,
    GeneAnnotation,
    ResourceProvenance,
    TractabilityBucket,
    TypedEdge,
)

SNAPSHOT_NAME: Final[str] = "keystone_snapshot.json"


@dataclass
class KeystoneSnapshot:
    schema_version: str
    retrieved_on: str
    provenance: list[ResourceProvenance]
    annotations: dict[str, GeneAnnotation]
    noncoding: list[str]
    edges: list[TypedEdge]
    unresolved_symbols: list[str] = field(default_factory=list)
    failed_interaction_queries: list[str] = field(default_factory=list)
    unqueried_interaction_genes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "retrieved_on": self.retrieved_on,
            "provenance": [
                {
                    "role": item.role,
                    "identifier": item.identifier,
                    "version": item.version,
                    "licence": item.licence,
                    "access_url": item.access_url,
                }
                for item in self.provenance
            ],
            "annotations": [
                {
                    "symbol": annotation.symbol,
                    "ensembl_id": annotation.ensembl_id,
                    "biotype": annotation.biotype,
                    "is_essential": annotation.is_essential,
                    "tractability": [
                        [bucket.modality, bucket.label, int(bucket.value)]
                        for bucket in annotation.tractability
                    ],
                    "clinical_stage": annotation.clinical_stage,
                }
                for annotation in sorted(self.annotations.values(), key=lambda a: a.symbol)
            ],
            "noncoding": list(self.noncoding),
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "edge_type": edge.edge_type,
                    "weight": round(edge.weight, 4),
                    "source_database": edge.source_database,
                }
                for edge in sorted(self.edges, key=lambda e: (e.edge_type, e.source, e.target))
            ],
            "unresolved_symbols": list(self.unresolved_symbols),
            "failed_interaction_queries": list(self.failed_interaction_queries),
            "unqueried_interaction_genes": list(self.unqueried_interaction_genes),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> KeystoneSnapshot:
        annotations: dict[str, GeneAnnotation] = {}
        for record in payload["annotations"]:
            buckets = tuple(
                TractabilityBucket(modality=str(bucket[0]), label=str(bucket[1]), value=bool(bucket[2]))
                for bucket in record["tractability"]
            )
            annotation = GeneAnnotation(
                symbol=str(record["symbol"]),
                ensembl_id=str(record["ensembl_id"]),
                biotype=str(record["biotype"]),
                tractability=buckets,
                is_essential=bool(record["is_essential"]),
            )
            annotations[annotation.symbol] = annotation
        provenance = [
            ResourceProvenance(
                role=str(item["role"]),
                identifier=str(item["identifier"]),
                version=str(item["version"]),
                licence=str(item["licence"]),
                access_url=str(item["access_url"]),
            )
            for item in payload["provenance"]
        ]
        edges = [
            TypedEdge(
                source=str(item["source"]),
                target=str(item["target"]),
                edge_type=str(item["edge_type"]),
                weight=float(item["weight"]),
                source_database=str(item["source_database"]),
            )
            for item in payload["edges"]
        ]
        return cls(
            schema_version=str(payload["schema_version"]),
            retrieved_on=str(payload["retrieved_on"]),
            provenance=provenance,
            annotations=annotations,
            noncoding=[str(item) for item in payload["noncoding"]],
            edges=edges,
            unresolved_symbols=[str(item) for item in payload["unresolved_symbols"]],
            failed_interaction_queries=[str(item) for item in payload["failed_interaction_queries"]],
            unqueried_interaction_genes=[
                str(item) for item in payload.get("unqueried_interaction_genes", [])
            ],
            notes=[str(item) for item in payload["notes"]],
        )

    def write(self, path: Path) -> Path:
        from immunotargetnet.runtime.atomic import write_json_atomic

        return write_json_atomic(path, self.to_dict())

    @classmethod
    def read(cls, path: Path) -> KeystoneSnapshot:
        with path.open("r", encoding="utf-8") as stream:
            payload: dict[str, Any] = json.load(stream)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"snapshot schema {payload.get('schema_version')} does not match {SCHEMA_VERSION}"
            )
        return cls.from_dict(payload)

    def edge_types(self) -> tuple[str, ...]:
        names = {edge.edge_type for edge in self.edges}
        return tuple(sorted(names))


def open_targets_provenance(retrieved_on: str) -> ResourceProvenance:
    return ResourceProvenance(
        role="keystone:interaction, pathway and tractability annotations",
        identifier="Open Targets Platform GraphQL API, v4",
        version="24.09",
        licence="CC0 1.0 for platform data; third-party sources retain their own terms",
        access_url="https://api.platform.opentargets.org/api/v4/graphql",
    )


def msigdb_provenance(release: str) -> ResourceProvenance:
    return ResourceProvenance(
        role="keystone: immune gene-set collections behind the twelve programmes",
        identifier=f"MSigDB {release} (c5.go.bp, c2.cp.reactome, h.all)",
        version=release,
        licence="MSigDB terms of use; registration required for the full corpus",
        access_url="https://www.gsea-msigdb.org/gsea/msigdb/",
    )
