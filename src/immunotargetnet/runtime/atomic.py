"""Atomic writes for checkpoints and JSON artefacts.

A truncated checkpoint is unrecoverable and a half-written manifest is silently
misleading, so every persistent write goes through a temporary sibling and
os.replace.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _prepare(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_bytes_atomic(path: Path, payload: bytes, *, mode: int = 0o644) -> Path:
    target = _prepare(path)
    handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def write_text_atomic(path: Path, text: str, *, mode: int = 0o644) -> Path:
    return write_bytes_atomic(path, text.encode("utf-8"), mode=mode)


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2, mode: int = 0o644) -> Path:
    body = json.dumps(payload, indent=indent, sort_keys=False, ensure_ascii=False)
    return write_text_atomic(path, body + "\n", mode=mode)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_delimited_atomic(
    path: Path, rows: list[Mapping[str, object]], columns: list[str], *, delimiter: str = "\t"
) -> Path:
    lines = [delimiter.join(columns)]
    for row in rows:
        lines.append(delimiter.join(str(row.get(column, "")) for column in columns))
    return write_text_atomic(path, "\n".join(lines) + "\n")
