"""Integrity manifest over the shipped tree.

The manifest is written last in a run and excludes only itself, so every other artefact a
run produces is covered by it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from immunotargetnet.runtime.atomic import write_json_atomic

MANIFEST_NAME: Final[str] = "integrity_manifest.json"
SKIP_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".git",
        ".venv",
        "node_modules",
        ".codebuddy",
    }
)
SKIP_TOP_LEVEL: Final[frozenset[str]] = frozenset({"data", "runs"})


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def inventory(root: Path, *, exclude: frozenset[Path] = frozenset()) -> list[dict[str, object]]:
    """Relative path plus content digest for every shipped file."""
    resolved_root = Path(root).resolve()
    excluded = {Path(item).resolve() for item in exclude}
    entries: list[dict[str, object]] = []
    for path in sorted(resolved_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(resolved_root)
        if any(part in SKIP_DIRECTORIES for part in relative.parts):
            continue
        if relative.parts and relative.parts[0] in SKIP_TOP_LEVEL:
            continue
        if path.resolve() in excluded:
            continue
        entries.append(
            {"path": relative.as_posix(), "sha256": digest_file(path), "bytes": path.stat().st_size}
        )
    return entries


def manifest_digest(entries: list[dict[str, object]]) -> str:
    hasher = hashlib.sha256()
    for entry in entries:
        hasher.update(f"{entry['path']}:{entry['sha256']}\n".encode())
    return hasher.hexdigest()


def build_manifest(root: Path, *, extra: dict[str, object] | None = None) -> dict[str, object]:
    target = Path(root).resolve() / MANIFEST_NAME
    entries = inventory(root, exclude=frozenset({target}))
    payload: dict[str, object] = {
        "manifest_name": MANIFEST_NAME,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["bytes"]) for entry in entries if isinstance(entry["bytes"], int)),
        "manifest_digest": manifest_digest(entries),
        "files": entries,
    }
    if extra:
        payload.update(extra)
    return payload


def write_manifest(root: Path, *, extra: dict[str, object] | None = None) -> Path:
    payload = build_manifest(root, extra=extra)
    return write_json_atomic(Path(root) / MANIFEST_NAME, payload)


def compare_with_tree(root: Path, manifest: dict[str, object]) -> dict[str, object]:
    """Compare a shipped manifest against the live tree without recording either digest."""
    raw_files = manifest.get("files")
    listed: dict[str, str] = {}
    if isinstance(raw_files, list):
        for entry in raw_files:
            if isinstance(entry, dict):
                listed[str(entry["path"])] = str(entry["sha256"])
    live = {
        str(entry["path"]): str(entry["sha256"])
        for entry in inventory(root, exclude=frozenset({Path(root) / MANIFEST_NAME}))
    }
    changed = sorted(path for path in listed if path in live and listed[path] != live[path])
    missing = sorted(path for path in listed if path not in live)
    unlisted = sorted(path for path in live if path not in listed)
    return {
        "files": len(live),
        "changed": changed[:10],
        "changed_count": len(changed),
        "missing": missing[:10],
        "missing_count": len(missing),
        "unlisted": unlisted[:10],
        "unlisted_count": len(unlisted),
        "digest_matches": not changed and not missing and not unlisted,
    }
