"""Pathway resource behind the graph's annotation nodes.

Ref: Sec. 2.2 (pathway descriptions among the keystone knowledge resources) and
Sec. 2.5 (nodes that are not genes carry frozen annotation embeddings).

The pathway families come from the Reactome collection of the gene-set corpus, which is
the same release the programme table is resolved from, so the graph's annotation structure
is rebuildable offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from immunotargetnet.runtime.atomic import write_json_atomic

PATHWAY_TABLE_NAME: Final[str] = "pathway_table.json"
MIN_PATHWAY_MEMBERS: Final[int] = 3
MAX_PATHWAY_MEMBERS: Final[int] = 400


@dataclass(frozen=True)
class PathwayTable:
    """Pathway identifier to the members of the candidate pool it contains."""

    members: dict[str, tuple[str, ...]]

    @classmethod
    def from_corpus(cls, pathway_sets: dict[str, tuple[str, ...]], universe: tuple[str, ...]) -> PathwayTable:
        allowed = set(universe)
        members: dict[str, tuple[str, ...]] = {}
        for name, entries in pathway_sets.items():
            kept = tuple(sorted(set(entries) & allowed))
            if MIN_PATHWAY_MEMBERS <= len(kept) <= MAX_PATHWAY_MEMBERS:
                members[name] = kept
        if not members:
            raise ValueError("no pathway survived the size window")
        return cls(members=members)

    def universe(self) -> tuple[str, ...]:
        union: set[str] = set()
        for entries in self.members.values():
            union.update(entries)
        return tuple(sorted(union))

    def write(self, path: Path) -> Path:
        payload = {
            "min_members": MIN_PATHWAY_MEMBERS,
            "max_members": MAX_PATHWAY_MEMBERS,
            "pathways": [
                {"id": name, "members": list(entries)} for name, entries in sorted(self.members.items())
            ],
        }
        return write_json_atomic(path, payload)

    @classmethod
    def read(cls, path: Path) -> PathwayTable:
        with path.open("r", encoding="utf-8") as stream:
            payload: dict[str, Any] = json.load(stream)
        members = {
            str(record["id"]): tuple(str(gene) for gene in record["members"])
            for record in payload["pathways"]
        }
        return cls(members=members)

    def incidence(self, genes: tuple[str, ...]) -> tuple[list[list[int]], list[int]]:
        """Per gene, the indices of the pathways it belongs to, plus each pathway's size."""
        names = sorted(self.members)
        position = {name: index for index, name in enumerate(names)}
        membership: list[list[int]] = [[] for _ in genes]
        gene_index = {gene: index for index, gene in enumerate(genes)}
        for name in names:
            index = position[name]
            for gene in self.members[name]:
                row = gene_index.get(gene)
                if row is not None:
                    membership[row].append(index)
        sizes = [len(self.members[name]) for name in names]
        return membership, sizes

    def labels(self) -> list[str]:
        return [f"pathway::{name}" for name in sorted(self.members)]


def co_membership_degree(membership: list[list[int]], sizes: list[int]) -> np.ndarray:
    """Distinct partner genes reachable through a shared pathway, per gene.

    A gene in a pathway of n members shares that pathway with n - 1 other genes, so the
    count is exact without materialising the quadratic edge set.
    """
    degree = np.zeros(len(membership), dtype=np.float64)
    for row, pathways in enumerate(membership):
        degree[row] = sum(max(sizes[index] - 1, 0) for index in pathways)
    return degree
