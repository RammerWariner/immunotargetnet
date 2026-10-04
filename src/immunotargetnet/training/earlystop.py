"""Early stopping on validation fidelity.

Ref: Sec. 2.1 (early stopping by validation fidelity) and Sec. 2.7.2 (fidelity is the
agreement between the simplex state and the held-out programme labels).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 1e-4
    best: float = float("-inf")
    best_epoch: int = -1
    stale_epochs: int = 0
    history: list[float] = field(default_factory=list)

    def update(self, value: float, epoch: int) -> bool:
        """Record a validation value; returns True when training should stop."""
        self.history.append(value)
        if value > self.best + self.min_delta:
            self.best = value
            self.best_epoch = epoch
            self.stale_epochs = 0
            return False
        self.stale_epochs += 1
        return self.stale_epochs >= self.patience

    def restored(self) -> bool:
        return self.best_epoch >= 0

    def summary(self) -> dict[str, float]:
        return {
            "best": self.best,
            "best_epoch": float(self.best_epoch),
            "stale_epochs": float(self.stale_epochs),
            "evaluations": float(len(self.history)),
        }
