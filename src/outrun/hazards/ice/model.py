"""Ice MIL-CNN: U-Net cell logits, LogSumExp zone pooling, zone-level isotonic calibration (ICE-04, ICE-06)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class IceMILNet(nn.Module):
    """ICE-04. U-Net encoder-decoder (default about 2 M params, 0.5-5 M) giving per-cell logits.

    forward(x[B, in_channels, H, W]) -> logits[B, 1, H, W] for H, W multiples of 16.
    """

    def __init__(self, in_channels: int = 20, base: int = 32) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        raise NotImplementedError


def mil_pool(cell_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ICE-04. Zone logit[B] = LogSumExp of cell logits over in-zone cells only. Empty mask -> ValueError."""
    raise NotImplementedError


def mil_loss(cell_logits: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ICE-04. Mean BCE-with-logits of mil_pool(cell_logits, mask) against zone labels in {0, 1}."""
    raise NotImplementedError


def cell_probabilities(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """ICE-04. sigmoid(cell logits) as a numpy (B, H, W) array, eval mode, no grad. Values in [0, 1]."""
    raise NotImplementedError


class ZoneCalibrator:
    """ICE-06. Isotonic calibration fitted on validation zone probabilities, applied to any-shape arrays."""

    def fit(self, zone_probs: np.ndarray, labels: np.ndarray) -> ZoneCalibrator:
        """Fit isotonic regression (out-of-bounds clipped). Returns self."""
        raise NotImplementedError

    def transform(self, p: np.ndarray) -> np.ndarray:
        """Monotone map to [0, 1], same shape as ``p``. Before fit -> RuntimeError."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Persist the fitted calibrator."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> ZoneCalibrator:
        """Load a calibrator saved with ``save``; outputs identical to the original."""
        raise NotImplementedError
