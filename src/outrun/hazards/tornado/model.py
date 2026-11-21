"""TorNet-style CNN (TORN-04) and focal loss (TORN-05)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .features import N_CHANNELS


class TornadoCNN(nn.Module):
    """VGG blocks (2x conv3x3+ReLU, maxpool 2), CoordConv inputs, global max-pool, MLP head (TORN-04).

    ``forward(x (B, in_channels, A, R)) -> logits (B,)``; ValueError on a wrong channel count.
    """

    def __init__(
        self, in_channels: int = N_CHANNELS, widths: Sequence[int] = (32, 64, 128, 256), hidden: int = 128
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        blocks: list[nn.Module] = []
        c_in = in_channels
        for width in widths:
            blocks += [
                nn.Conv2d(c_in, width, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = width
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(c_in, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (B, {self.in_channels}, A, R) input, got {tuple(x.shape)}")
        h = self.features(x)
        h = torch.amax(h, dim=(2, 3))  # global max-pool: a tornado signature anywhere in the chip counts
        return self.head(h).squeeze(-1)


def predict_proba(model: TornadoCNN, x: torch.Tensor) -> torch.Tensor:
    """Eval mode, no grad: ``sigmoid(logits)`` ``(B,)`` in [0, 1] (TORN-04)."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x))


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float | None = None
) -> torch.Tensor:
    """mean(-a_t (1 - p_t)^gamma log p_t), stable from logits; gamma=0, alpha=None == BCE (TORN-05)."""
    targets = targets.to(logits.dtype)
    # -log p_t is the per-sample BCE, computed stably from logits
    nll = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-nll)
    loss = (1.0 - p_t) ** gamma * nll
    if alpha is not None:
        loss = (alpha * targets + (1.0 - alpha) * (1.0 - targets)) * loss
    return loss.mean()
