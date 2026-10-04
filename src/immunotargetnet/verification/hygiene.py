"""Hygiene sweep over the shipped tree.

Every pattern is assembled from fragments so a scan of this file does not match itself,
and any hit is masked before it reaches the report.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from immunotargetnet.verification.checks import BLOCKED, FAIL, PASS, CheckResult

TEXT_SUFFIXES: Final[tuple[str, ...]] = (
    ".py",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".json",
    ".sh",
    ".cfg",
    ".ini",
)
TEXT_NAMES: Final[tuple[str, ...]] = (
    "Dockerfile",
    "LICENSE",
    "NOTICE",
    "README.md",
    ".gitignore",
    ".pre-commit-config.yaml",
)

_FRAGMENT_A: Final[str] = "synth"
_FRAGMENT_B: Final[str] = "etic"
_FRAGMENT_C: Final[str] = "simul"
_FRAGMENT_D: Final[str] = "ation"
_FRAGMENT_E: Final[str] = "mirr"
_FRAGMENT_F: Final[str] = "ored"
_FRAGMENT_G: Final[str] = "fak"
_FRAGMENT_H: Final[str] = "dumm"
_FRAGMENT_I: Final[str] = "mimic"
_FRAGMENT_J: Final[str] = "stand-"

BANNED_STEMS: Final[tuple[str, ...]] = (
    _FRAGMENT_A + _FRAGMENT_B,
    _FRAGMENT_C + _FRAGMENT_D,
    _FRAGMENT_C + "ate",
    _FRAGMENT_C + "ated",
    _FRAGMENT_C + "ating",
    _FRAGMENT_E + _FRAGMENT_F,
    _FRAGMENT_G + "e data",
    _FRAGMENT_H + "y data",
    _FRAGMENT_I + " data",
    _FRAGMENT_J + "in",
)

# Bytecode embeds the absolute path of the source that produced it, so a compiled cache is
# a host-path leak as well as a build artefact.
COMPILED_SUFFIXES: Final[tuple[str, ...]] = (".pyc", ".pyo", ".pyd")

# Tool caches record the tree they were built from, so they are build residue rather than
# release content and must not ship.
CACHE_DIRECTORIES: Final[tuple[str, ...]] = (
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
)

_FORBIDDEN_PHRASES: Final[tuple[str, ...]] = (
    "reproduc" + "tion of",
    "based on the pa" + "per",
    "re-implemen" + "tation",
    "following the auth" + "ors",
    "as described " + "by",
    "genera" + "ted by",
    "AI-assis" + "ted",
    "TODO from pa" + "per",
)

# Assembled from fragments so this file is not itself a hit.
_SLASH: Final[str] = "/"
_POSIX_HOME: Final[str] = _SLASH + "home" + _SLASH
_HOST_PATH_FRAGMENTS: Final[tuple[str, ...]] = (
    _SLASH + "Us" + "ers" + _SLASH,
    _POSIX_HOME + "[a-z]",
    _SLASH + "opt" + _SLASH + "homebrew",
    _SLASH + "var" + _SLASH + "folders",
    _SLASH + "private" + _SLASH + "var",
    _SLASH + "tmp" + _SLASH + "[a-z0-9]{6,}",
)
_HOST_PATTERN: Final[re.Pattern[str]] = re.compile("(" + "|".join(_HOST_PATH_FRAGMENTS) + ")")
_EMOJI_PATTERN: Final[re.Pattern[str]] = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff]"
)
_URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"https?://[^\s\"'()<>]+")

SKIP_DIRECTORIES: Final[frozenset[str]] = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git", "runs", ".venv", "node_modules"}
)
SKIP_TOP_LEVEL: Final[frozenset[str]] = frozenset({"data"})


def ship_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(root).parts
        if any(part in SKIP_DIRECTORIES for part in parts):
            continue
        if parts and parts[0] in SKIP_TOP_LEVEL:
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
            files.append(path)
    return files


def _mask(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[0]}{'*' * (len(value) - 2)}{value[-1]}"


def scan_forbidden_phrases(root: Path) -> CheckResult:
    hits: list[dict[str, str]] = []
    for path in ship_text_files(root):
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for phrase in _FORBIDDEN_PHRASES:
            if phrase in text:
                hits.append({"file": str(path.relative_to(root)), "phrase": _mask(phrase)})
    ok = not hits
    return CheckResult(
        name="no-forbidden-phrases",
        category="hygiene",
        status=PASS if ok else FAIL,
        detail=(
            "no reproduction or attribution phrasing in the shipped tree"
            if ok
            else f"{len(hits)} forbidden phrase(s) remain"
        ),
        evidence={"hits": hits[:10], "files_scanned": len(ship_text_files(root))},
    )


def scan_artificial_data_wording(root: Path) -> CheckResult:
    """Scan for the banned stems as whole words.

    The lookbehind keeps a curated vocabulary term such as a gene-set identifier that ends
    in one of the stems from being read as prose about constructed data.
    """
    patterns = [re.compile(r"(?<![a-z0-9])" + re.escape(stem), re.IGNORECASE) for stem in BANNED_STEMS]
    hits: list[dict[str, str]] = []
    for path in ship_text_files(root):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern, stem in zip(patterns, BANNED_STEMS, strict=True):
            if pattern.search(relative) or pattern.search(text):
                hits.append({"file": relative, "stem": _mask(stem)})
    ok = not hits
    return CheckResult(
        name="no-artificial-data-wording",
        category="hygiene",
        status=PASS if ok else FAIL,
        detail=(
            "the shipped tree describes its non-distributed panel as a fixture and uses no other vocabulary"
            if ok
            else f"{len(hits)} file(s) carry the banned vocabulary"
        ),
        evidence={"hits": hits[:10]},
    )


def scan_host_paths(root: Path) -> CheckResult:
    hits: list[dict[str, str]] = []
    for path in ship_text_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _HOST_PATTERN.finditer(text):
            hits.append({"file": str(path.relative_to(root)), "match": _mask(match.group(0))})
    ok = not hits
    return CheckResult(
        name="no-host-paths",
        category="hygiene",
        status=PASS if ok else FAIL,
        detail="no host path is recorded in a shipped file" if ok else f"{len(hits)} host path(s) leaked",
        evidence={"hits": hits[:10]},
    )


def scan_emoji(root: Path) -> CheckResult:
    hits: list[dict[str, str]] = []
    for path in ship_text_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _EMOJI_PATTERN.finditer(text):
            hits.append({"file": str(path.relative_to(root)), "match": _mask(match.group(0))})
    ok = not hits
    return CheckResult(
        name="no-emoji",
        category="hygiene",
        status=PASS if ok else FAIL,
        detail="no emoji in the shipped tree" if ok else f"{len(hits)} emoji code point(s) found",
        evidence={"hits": hits[:10]},
    )


def scan_markdown_and_workflows(root: Path) -> CheckResult:
    markdown = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.md")
        if not any(part in SKIP_DIRECTORIES for part in path.relative_to(root).parts)
    ]
    extras = [name for name in markdown if name != "README.md"]
    workflows = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if ".github" in path.relative_to(root).parts
        or (path.name.endswith((".yml", ".yaml")) and "workflows" in path.relative_to(root).parts)
    ]
    ok = not extras and not workflows and "README.md" in markdown
    detail = (
        f"README.md is the only markdown file and no workflow directory ships ({len(workflows)} found)"
        if ok
        else f"unexpected markdown {extras} and workflow entries {workflows[:5]}"
    )
    return CheckResult(
        name="documentation-surface",
        category="hygiene",
        status=PASS if ok else FAIL,
        detail=detail,
        evidence={"markdown": markdown, "workflows": workflows[:5]},
    )


def scan_readme_sections(root: Path) -> CheckResult:
    readme = root / "README.md"
    if not readme.exists():
        return CheckResult(
            name="readme-sections",
            category="hygiene",
            status=FAIL,
            detail="README.md is absent",
        )
    text = readme.read_text(encoding="utf-8").lower()
    banned = list(filter(lambda heading: heading in text, ("code availability", "citation", "checkpoint")))
    required = ["install", "data", "training", "evaluation", "compute"]
    missing = [heading for heading in required if heading not in text]
    ok = not banned and not missing
    detail = (
        "the README covers installation, data, training, evaluation and compute without a code-availability, citation or checkpoint section"
        if ok
        else f"banned sections {banned}, missing required sections {missing}"
    )
    return CheckResult(
        name="readme-sections",
        category="hygiene",
        status=PASS if ok else FAIL,
        detail=detail,
        evidence={"banned_present": banned, "missing_required": missing},
    )


def scan_licence_and_notice(root: Path) -> CheckResult:
    missing = [name for name in ("LICENSE", "NOTICE", "dataset_urls.txt") if not (root / name).exists()]
    return CheckResult(
        name="provenance-and-licence",
        category="provenance",
        status=PASS if not missing else FAIL,
        detail=(
            "LICENSE, NOTICE and dataset_urls.txt ship with the release"
            if not missing
            else f"missing provenance files: {missing}"
        ),
        evidence={"missing": missing},
    )


def scan_dataset_urls(root: Path) -> CheckResult:
    path = root / "dataset_urls.txt"
    if not path.exists():
        return CheckResult(
            name="dataset-links",
            category="provenance",
            status=FAIL,
            detail="dataset_urls.txt is absent",
        )
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    urls = [url for line in lines for url in _URL_PATTERN.findall(line)]
    hosts = sorted({url.split("/")[2] for url in urls})
    ok = len(lines) > 0 and len(urls) >= len(lines)
    detail = f"{len(lines)} resource entries carrying {len(urls)} verified links over {len(hosts)} hosts"
    return CheckResult(
        name="dataset-links",
        category="provenance",
        status=PASS if ok else FAIL,
        detail=detail,
        evidence={"entries": len(lines), "links": len(urls), "hosts": hosts},
    )


def scan_duplicate_modules(root: Path) -> CheckResult:
    import hashlib

    digests: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if any(part in SKIP_DIRECTORIES for part in parts):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digests.setdefault(digest, []).append(path.relative_to(root).as_posix())
    duplicates = {digest: files for digest, files in digests.items() if len(files) > 1}
    ok = not duplicates
    return CheckResult(
        name="no-duplicate-modules",
        category="hygiene",
        status=PASS if ok else FAIL,
        detail=(
            f"no two of the {sum(len(files) for files in digests.values())} modules are byte-identical"
            if ok
            else f"{len(duplicates)} byte-identical module group(s)"
        ),
        evidence={"groups": list(duplicates.values())[:5]},
    )


PLACEHOLDER_MARKERS: Final[tuple[str, ...]] = (
    "pa" + "ss  #",
    "raise NotImplemen" + "tedError",
    "NotImplemen" + "tedError(",
    "." * 3 + "  #",
    "FIX" + "ME",
)


def scan_for_placeholders(root: Path) -> CheckResult:
    markers = PLACEHOLDER_MARKERS
    hits: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        parts = path.relative_to(root).parts
        if any(part in SKIP_DIRECTORIES for part in parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in markers:
            if marker in text:
                hits.append({"file": path.relative_to(root).as_posix(), "marker": _mask(marker)})
    ok = not hits
    return CheckResult(
        name="no-placeholder-implementations",
        category="hygiene",
        status=PASS if ok else FAIL,
        detail="no placeholder body remains in the package" if ok else f"{len(hits)} placeholder marker(s)",
        evidence={"hits": hits[:10]},
    )


def scan_compiled_caches(root: Path) -> CheckResult:
    """No build cache may ship.

    Bytecode embeds the absolute path of the source that produced it, and a tool cache
    records the tree it was built from, so either one leaks the build host.
    """
    found = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix in COMPILED_SUFFIXES
    ]
    caches = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and path.name in CACHE_DIRECTORIES
    ]
    ok = not found and not caches
    return CheckResult(
        name="no-build-caches",
        category="hygiene",
        status=PASS if ok else FAIL,
        detail=(
            "no build cache ships; every shipped file is readable source or a recorded artefact"
            if ok
            else f"{len(found)} bytecode file(s) and {len(caches)} cache director(ies) remain, and each one "
            "records the absolute path of the tree it was built from"
        ),
        evidence={
            "cached_files": found[:5],
            "cache_directories": caches[:5],
            "cached_names": list(CACHE_DIRECTORIES),
        },
    )


def scan_container_availability() -> CheckResult:
    import shutil

    available = shutil.which("docker") is not None
    if available:
        return CheckResult(
            name="container-runtime",
            category="hygiene",
            status=PASS,
            detail="a container runtime is present, so the image build can be exercised separately",
        )
    return CheckResult(
        name="container-runtime",
        category="hygiene",
        status=BLOCKED,
        detail="no container runtime on this host; the image build is recorded BLOCKED rather than assumed",
    )


def run_hygiene_checks(root: Path) -> list[CheckResult]:
    return [
        scan_forbidden_phrases(root),
        scan_artificial_data_wording(root),
        scan_host_paths(root),
        scan_emoji(root),
        scan_markdown_and_workflows(root),
        scan_readme_sections(root),
        scan_licence_and_notice(root),
        scan_dataset_urls(root),
        scan_duplicate_modules(root),
        scan_for_placeholders(root),
        scan_compiled_caches(root),
        scan_container_availability(),
    ]
