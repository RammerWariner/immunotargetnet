"""Check result types and the status vocabulary.

Statuses are exactly PASS, FAIL, NOT_RUN and BLOCKED, and each is decided by something the
run actually produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

PASS: Final[str] = "PASS"
FAIL: Final[str] = "FAIL"
NOT_RUN: Final[str] = "NOT_RUN"
BLOCKED: Final[str] = "BLOCKED"

STATUSES: Final[tuple[str, ...]] = (PASS, FAIL, NOT_RUN, BLOCKED)

CODE_CATEGORIES: Final[tuple[str, ...]] = (
    "execution",
    "unit",
    "regression",
    "shape",
    "statistics",
    "provenance",
    "hygiene",
    "gate",
)
MANUSCRIPT_CATEGORY: Final[str] = "manuscript"
ALL_CATEGORIES: Final[tuple[str, ...]] = (*CODE_CATEGORIES, MANUSCRIPT_CATEGORY)


@dataclass(frozen=True)
class CheckResult:
    name: str
    category: str
    status: str
    detail: str
    evidence: dict[str, object] = field(default_factory=dict)
    paper_anchor: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status: {self.status}")
        if self.category not in ALL_CATEGORIES:
            raise ValueError(f"unknown category: {self.category}")

    def record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "detail": self.detail,
        }
        if self.paper_anchor:
            record["paper_anchor"] = self.paper_anchor
        if self.evidence:
            record["evidence"] = self.evidence
        return record


def summarise(results: list[CheckResult]) -> dict[str, int]:
    tally = dict.fromkeys(STATUSES, 0)
    for result in results:
        tally[result.status] += 1
    return tally


def verdict(results: list[CheckResult], categories: tuple[str, ...]) -> str:
    """A family fails only when a check actually failed.

    A blocked check means the host could not exercise it, so it counts as unrun rather than
    as a defect; otherwise a missing container runtime would read as a broken release.
    """
    selected = [result for result in results if result.category in categories]
    if not selected:
        return NOT_RUN
    if any(result.status == FAIL for result in selected):
        return FAIL
    if any(result.status == NOT_RUN for result in selected):
        return NOT_RUN
    if any(result.status == BLOCKED for result in selected):
        return NOT_RUN
    return PASS


def overall_status(code_status: str, manuscript_status: str) -> str:
    """Three verdicts collapse to one: a manuscript inconsistency keeps the release unverified."""
    if code_status == FAIL:
        return "UNVERIFIED"
    if manuscript_status == FAIL:
        return "PARTIALLY_VERIFIED"
    if code_status == NOT_RUN or manuscript_status == NOT_RUN:
        return "PARTIALLY_VERIFIED"
    return "VERIFIED"
