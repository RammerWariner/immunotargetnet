"""Free-form fusion layer of the unconstrained variant.

Ref: Sec. 2.4 and Table 1 (B7 is the unrestricted optimum of the same model family: the
simplex is replaced by a free-form linear projection while the encoders, the graph tower
and the heads are kept).
"""

from __future__ import annotations

from torch import Tensor, nn


class FreeFormFusion(nn.Module):
    """A wider linear fusion in place of the twelve-programme simplex projection."""

    def __init__(self, model_dim: int, n_outputs: int, expansion: int = 4) -> None:
        super().__init__()
        self.expansion = expansion
        self.first = nn.Linear(model_dim, expansion * model_dim)
        self.activation = nn.GELU()
        self.second = nn.Linear(expansion * model_dim, n_outputs)

    def forward(self, pooled: Tensor) -> Tensor:
        hidden: Tensor = self.activation(self.first(pooled))
        output: Tensor = self.second(hidden)
        return output
