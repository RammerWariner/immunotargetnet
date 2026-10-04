"""Reader for the GEO series-matrix text format.

Ref: Sec. 2.2 (cohort evidence layer, eight public expression cohorts).
"""

from __future__ import annotations

import gzip
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# GEO writes these markers in lower case; the comparison against a stripped, lowered line
# keeps both spellings working.
TABLE_BEGIN: Final[str] = "!series_matrix_table_begin"
TABLE_END: Final[str] = "!series_matrix_table_end"


class SeriesMatrixError(ValueError):
    """Raised when a series matrix is malformed or truncated."""


@dataclass(frozen=True)
class SeriesMatrix:
    accession: str
    platform: str
    sample_characteristics: dict[str, tuple[str, ...]]
    gene_symbols: tuple[str, ...]
    values: list[list[float]]
    n_samples: int
    n_genes: int

    def column_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self.sample_characteristics))


def _open_text(path: Path) -> Iterable[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
            yield from stream
    else:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            yield from stream


def _strip_quotes(field: str) -> str:
    return field.strip().strip('"')


def _split_table_line(line: str) -> list[str]:
    return [_strip_quotes(field) for field in line.rstrip("\n").split("\t")]


def read_series_matrix(path: Path, *, accession: str | None = None) -> SeriesMatrix:
    header_fields: dict[str, list[str]] = {}
    characteristic_rows: list[list[str]] = []
    rows: list[list[str]] = []
    in_table = False
    for raw in _open_text(path):
        marker = raw.strip().lower()
        if marker.startswith(TABLE_BEGIN):
            in_table = True
            continue
        if marker.startswith(TABLE_END):
            in_table = False
            continue
        if in_table:
            rows.append(_split_table_line(raw))
            continue
        if not raw.startswith("!"):
            continue
        fields = _split_table_line(raw)
        key = fields[0]
        if key.lower().startswith("!sample_characteristics"):
            characteristic_rows.append(fields[1:])
            continue
        header_fields.setdefault(key, []).extend(fields[1:])

    if not rows:
        raise SeriesMatrixError(f"no series-matrix table block in {path.name}")

    columns = rows[0]
    symbols: list[str] = []
    values: list[list[float]] = []
    symbol_column = 0
    for index, name in enumerate(columns):
        if name.lower().replace(" ", "") in {"genesymbol", "symbol", "gene_assignmentsymbol"}:
            symbol_column = index
    for row in rows[1:]:
        if len(row) < len(columns):
            continue
        symbols.append(row[symbol_column] if symbol_column else row[0])
        numeric: list[float] = []
        for field in row[1:]:
            try:
                numeric.append(float(field))
            except ValueError:
                numeric.append(float("nan"))
        values.append(numeric)

    if not values:
        raise SeriesMatrixError(f"series matrix {path.name} carries no data rows")

    n_samples = len(values[0])
    characteristics = _sample_characteristics(characteristic_rows)
    resolved_accession = accession or _first_accession(header_fields) or path.stem
    return SeriesMatrix(
        accession=resolved_accession,
        platform=",".join(header_fields.get("!Series_platform_id", ["unknown"])),
        sample_characteristics=characteristics,
        gene_symbols=tuple(symbols),
        values=values,
        n_samples=n_samples,
        n_genes=len(values),
    )


def _sample_characteristics(rows: list[list[str]]) -> dict[str, tuple[str, ...]]:
    """One entry per characteristic line, keyed by the label its own values carry.

    GEO repeats the characteristic field once per characteristic, so each line is a
    separate annotating column and merging them would concatenate unrelated labels.
    """
    characteristics: dict[str, tuple[str, ...]] = {}
    for ordinal, fields in enumerate(rows, start=1):
        values = tuple(_characteristic_value(field) for field in fields)
        labels = {field.split(":", 1)[0].strip() for field in fields if ":" in field}
        key = labels.pop() if len(labels) == 1 else f"characteristic_{ordinal}"
        if not key:
            key = f"characteristic_{ordinal}"
        if key in characteristics:
            key = f"{key}_{ordinal}"
        characteristics[key] = values
    return characteristics


def _characteristic_value(field: str) -> str:
    if ":" in field:
        return field.split(":", 1)[1].strip()
    return field.strip()


def _first_accession(header_fields: dict[str, list[str]]) -> str | None:
    for field in header_fields.get("!Series_geo_accession", []):
        if field:
            return field
    return None


def collate_samples(
    matrix: SeriesMatrix, contrast_key: str, positive_labels: set[str], negative_labels: set[str]
) -> tuple[list[int], list[int]]:
    """Return the sample indices on each side of a two-group contrast."""
    labels = matrix.sample_characteristics.get(contrast_key)
    if labels is None:
        available = ", ".join(matrix.column_keys())
        raise SeriesMatrixError(f"contrast key {contrast_key!r} absent; available: {available}")
    positive = [i for i, label in enumerate(labels) if label.strip().lower() in positive_labels]
    negative = [i for i, label in enumerate(labels) if label.strip().lower() in negative_labels]
    return positive, negative


def gene_index(symbols: tuple[str, ...]) -> dict[str, int]:
    """First occurrence wins; series matrices repeat a symbol across probe sets."""
    index: dict[str, int] = {}
    for position, symbol in enumerate(symbols):
        if symbol and symbol not in index:
            index[symbol] = position
    return index
