"""Deterministic seeding for every stochastic component.

Ref: Sec. 2.7.4 (five independent runs) and Table 6 (medians of five runs).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


def set_seed(seed: int, *, deterministic_algorithms: bool = False) -> None:
    """Seed the Python, NumPy and torch generators in one call."""
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True


def spawn_generators(seed: int, count: int) -> list[np.random.Generator]:
    """Return `count` independent generators drawn from one root seed."""
    root = np.random.default_rng(seed)
    return [np.random.default_rng(int(child)) for child in root.integers(0, 2**31, size=count)]


@dataclass(frozen=True)
class RunSeed:
    """A single reported run index and the seed that realises it."""

    run_index: int
    seed: int


def run_seeds(root_seed: int, n_runs: int) -> tuple[RunSeed, ...]:
    children = spawn_generators(root_seed, n_runs)
    return tuple(RunSeed(run_index=i, seed=int(g.integers(0, 2**31))) for i, g in enumerate(children))
