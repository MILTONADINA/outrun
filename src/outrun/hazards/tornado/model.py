"""TorNet-style CNN (TORN-04) and focal loss (TORN-05)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .features import N_CHANNELS


class TornadoCNN(nn.Module):
    """VGG blocks (2x conv3x3+ReLU, maxpool 2), CoordConv inputs, global max-pool, MLP head (TORN-04).

    ``forward(x (B, in_channels, A, R)) -> logits (B,)``; ValueError on a wrong channel count.
    """

    def __init__(
        self, in_channels: int = N_CHANNELS, widths: Sequence[int] = (32, 64, 128, 256), hidden: int = 128
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def predict_proba(model: TornadoCNN, x: torch.Tensor) -> torch.Tensor:
    """Eval mode, no grad: ``sigmoid(logits)`` ``(B,)`` in [0, 1] (TORN-04)."""
    raise NotImplementedError


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float | None = None
) -> torch.Tensor:
    """mean(-a_t (1 - p_t)^gamma log p_t), stable from logits; gamma=0, alpha=None == BCE (TORN-05)."""
    raise NotImplementedError
