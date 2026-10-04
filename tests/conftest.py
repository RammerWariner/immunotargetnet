"""Shared fixtures: the release root and one assembled panel."""

from __future__ import annotations

from pathlib import Path

import pytest

from immunotargetnet.evidence.io import AssetPaths, build_panel
from immunotargetnet.evidence.panel import PanelBundle

RELEASE_ROOT = Path(__file__).resolve().parents[1]
PANEL_SEED = 20260101


@pytest.fixture(scope="session")
def release_root() -> Path:
    return RELEASE_ROOT


@pytest.fixture(scope="session")
def assets_root() -> Path:
    paths = AssetPaths(root=RELEASE_ROOT)
    paths.require()
    return RELEASE_ROOT


@pytest.fixture(scope="session")
def bundle(assets_root: Path) -> PanelBundle:
    return build_panel(AssetPaths(root=assets_root), seed=PANEL_SEED, context_dim=8)
