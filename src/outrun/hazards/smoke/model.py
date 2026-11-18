"""Smoke CNN + MLP quantile model and pinball loss (SMOKE-04)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


class SmokeNet(nn.Module):
    """SMOKE-04. CNN branch (2 conv + pool) on the 8x9x9 HRRR patch, MLP on tabular features, 3 dense layers.

    forward(patch[B, 8, 9, 9], tab[B, n_tabular]) -> [B, len(quantiles)] in log1p-PM2.5 space. ~0.1 M params.
    """

    def __init__(
        self,
        in_channels: int = 8,
        patch: int = 9,
        n_tabular: int = 11,
        quantiles: tuple[float, ...] = (0.5, 0.9),
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, patch: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:  # noqa: D102
        raise NotImplementedError


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: Sequence[float]) -> torch.Tensor:
    """SMOKE-04. mean over b, q of max(q * (y - pred), (q - 1) * (y - pred))."""
    raise NotImplementedError


def to_pm(pred_log: np.ndarray) -> np.ndarray:
    """SMOKE-04. expm1, clipped at 0, running max across the quantile axis (non-crossing)."""
    raise NotImplementedError
