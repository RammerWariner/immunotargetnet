"""Reader for the MSigDB gene-set corpus and resolver for the programme table.

Ref: Sec. 2.4 (programmes assembled from public immune gene-set collections).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from immunotargetnet.evidence.programmes import corpus_for_collection, programme_definitions


@dataclass(frozen=True)
class GeneSetCorpus:
    release: str
    sets: dict[str, tuple[str, ...]]

    @classmethod
    def read(cls, directories: dict[str, Path], release: str) -> GeneSetCorpus:
        sets: dict[str, tuple[str, ...]] = {}
        for tag, directory in directories.items():
            for name, members in _read_gmt(directory).items():
                sets[f"{tag}:{name}"] = members
        return cls(release=release, sets=sets)

    def members_of(self, tag: str) -> dict[str, tuple[str, ...]]:
        """Every collection of one corpus, keyed by its own identifier."""
        prefix = f"{tag}:"
        return {
            name[len(prefix) :]: entries for name, entries in self.sets.items() if name.startswith(prefix)
        }

    def members(self, collection: str) -> tuple[str, ...]:
        key = f"{corpus_for_collection(collection)}:{collection}"
        try:
            return self.sets[key]
        except KeyError as error:
            raise KeyError(f"collection absent from the corpus: {collection}") from error


def _read_gmt(path: Path) -> dict[str, tuple[str, ...]]:
    if path.is_dir():
        merged: dict[str, tuple[str, ...]] = {}
        for child in sorted(path.iterdir()):
            if child.is_file():
                merged.update(_read_gmt(child))
        return merged
    sets: dict[str, tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            members = tuple(member for member in fields[2:] if member)
            sets[fields[0]] = members
    return sets


@dataclass(frozen=True)
class ProgrammeTable:
    keys: tuple[str, ...]
    display_names: tuple[str, ...]
    members: dict[str, tuple[str, ...]]

    @classmethod
    def resolve(cls, corpus: GeneSetCorpus) -> ProgrammeTable:
        keys: list[str] = []
        names: list[str] = []
        membership: dict[str, tuple[str, ...]] = {}
        for definition in programme_definitions():
            union: set[str] = set()
            for collection in definition.collections:
                union.update(corpus.members(collection))
            if not union:
                raise ValueError(f"programme resolved to an empty set: {definition.key}")
            keys.append(definition.key)
            names.append(definition.display_name)
            membership[definition.key] = tuple(sorted(union))
        return cls(keys=tuple(keys), display_names=tuple(names), members=membership)

    def universe(self) -> tuple[str, ...]:
        union: set[str] = set()
        for members in self.members.values():
            union.update(members)
        return tuple(sorted(union))

    def incidence(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for members in self.members.values():
            for gene in members:
                counts[gene] = counts.get(gene, 0) + 1
        return counts
