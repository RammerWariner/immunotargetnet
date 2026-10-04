"""The verification driver and its artefact writers.

The writers take a root, so a test can run them against a scratch directory and leave the
shipped artefacts untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

from immunotargetnet.verification import driver
from immunotargetnet.verification.checks import PASS, CheckResult, overall_status, verdict
from immunotargetnet.verification.hygiene import (
    scan_artificial_data_wording,
    scan_duplicate_modules,
    scan_emoji,
    scan_forbidden_phrases,
    scan_host_paths,
    scan_readme_sections,
    ship_text_files,
)
from immunotargetnet.verification.manifest import build_manifest, compare_with_tree, inventory


def test_status_arithmetic_separates_code_from_manuscript() -> None:
    code = [CheckResult("a", "execution", PASS, "ok")]
    manuscript = [CheckResult("b", "manuscript", "FAIL", "disagreement")]
    assert verdict(code, ("execution",)) == PASS
    assert verdict(manuscript, ("manuscript",)) == "FAIL"
    assert overall_status(PASS, "FAIL") == "PARTIALLY_VERIFIED"
    assert overall_status("FAIL", PASS) == "UNVERIFIED"
    assert overall_status(PASS, PASS) == "VERIFIED"
    assert overall_status("NOT_RUN", PASS) == "PARTIALLY_VERIFIED"


def test_check_result_rejects_an_unknown_status() -> None:
    try:
        CheckResult("x", "execution", "MAYBE", "nope")
    except ValueError:
        return
    raise AssertionError("an unknown status must be rejected")


def test_check_result_rejects_an_unknown_category() -> None:
    try:
        CheckResult("x", "vibes", PASS, "nope")
    except ValueError:
        return
    raise AssertionError("an unknown category must be rejected")


def test_claim_map_covers_every_mapping() -> None:
    payload = driver.build_claim_map()
    assert len(payload["claims"]) == len(driver.MAPPINGS)
    assert payload["deviations"]
    assert all(entry["paper_anchor"] for entry in payload["deviations"])
    assert payload["manuscript"]["title"].startswith("ImmunoTargetNet")


def test_report_keeps_the_three_verdicts(release_root: Path) -> None:
    report = driver.build_report(
        root=release_root, results=[CheckResult("a", "execution", PASS, "ok")], gates=[]
    )
    assert set(report) >= {
        "overall_status",
        "code_status",
        "manuscript_status",
        "code_failures",
        "manuscript_discrepancies",
        "outstanding",
        "tally",
    }
    assert "not a defect in this release" in report["interpretation"]


def test_summary_renders_the_verdicts(release_root: Path) -> None:
    report = driver.build_report(
        root=release_root, results=[CheckResult("a", "execution", PASS, "ok")], gates=[]
    )
    text = driver.render_summary(report, release_root)
    assert "overall_status" in text
    assert "code_status" in text
    assert "manuscript_status" in text


def test_artefact_writers_target_the_given_root(release_root: Path, tmp_path: Path) -> None:
    report = driver.build_report(root=tmp_path, results=[CheckResult("a", "execution", PASS, "ok")], gates=[])
    written = driver.write_artefacts(tmp_path, report)
    assert (tmp_path / "claim_to_code.json").exists()
    assert (tmp_path / "verification_report.json").exists()
    assert (tmp_path / "verification_summary.txt").exists()
    assert (tmp_path / "integrity_manifest.json").exists()
    assert written["manifest"].parent == tmp_path
    payload = json.loads((tmp_path / "integrity_manifest.json").read_text(encoding="utf-8"))
    assert payload["manifest_name"] == "integrity_manifest.json"
    listed = {entry["path"] for entry in payload["files"]}
    assert "integrity_manifest.json" not in listed
    assert "verification_report.json" in listed


def test_manifest_excludes_itself_and_caches(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    cached = tmp_path / "__pycache__"
    cached.mkdir()
    (cached / "keep.cpython-311.pyc").write_bytes(b"cache")
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "cohort.tsv").write_text("a\tb\n", encoding="utf-8")
    manifest = build_manifest(tmp_path)
    listed = {entry["path"] for entry in manifest["files"]}
    assert listed == {"keep.py"}
    assert manifest["file_count"] == 1


def test_manifest_comparison_reports_an_outcome_only(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    manifest = build_manifest(tmp_path)
    comparison = compare_with_tree(tmp_path, manifest)
    assert comparison["digest_matches"] is True
    assert "live_digest" not in comparison
    assert "shipped_digest" not in comparison
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    changed = compare_with_tree(tmp_path, manifest)
    assert changed["changed"] == ["a.py"]


def test_shipped_manifest_matches_the_tree(release_root: Path) -> None:
    path = release_root / "integrity_manifest.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    comparison = compare_with_tree(release_root, payload)
    assert comparison["digest_matches"], comparison


def test_shipped_report_is_the_complete_one(release_root: Path) -> None:
    path = release_root / "verification_report.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["checks"]) > 80
    assert payload["overall_status"] in {"VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED"}
    assert payload["code_status"] in {"PASS", "FAIL", "NOT_RUN"}


def test_hygiene_scanners_run_over_the_shipped_tree(release_root: Path) -> None:
    assert ship_text_files(release_root)
    for scanner in (
        scan_forbidden_phrases,
        scan_artificial_data_wording,
        scan_emoji,
        scan_host_paths,
        scan_readme_sections,
        scan_duplicate_modules,
    ):
        result = scanner(release_root)
        assert result.status == PASS, (result.name, result.detail, result.evidence)


def test_inventory_lists_relative_paths_only(release_root: Path) -> None:
    entries = inventory(release_root)
    assert entries
    assert all(not str(entry["path"]).startswith("/") for entry in entries)
