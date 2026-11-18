"""Flood U-Net and focal loss with the uncertain ring masked (specs/10-flood.md: FLOOD-04, FLOOD-05)."""

from __future__ import annotations

import torch
from torch import nn


class FloodUNet(nn.Module):
    """U-Net: (B, in_channels, H, W) -> logits (B, 1, H, W); H, W divisible by 2**levels (FLOOD-04).

    Defaults (11 channels, base 32, 3 levels) give 0.5-5 M parameters. Wrong channel count -> ValueError.
    ``in_channels=8`` is the no-FLASH ablation (FLOOD-09).
    """

    def __init__(self, in_channels: int = 11, base: int = 32, levels: int = 3) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = 0.25,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Mean focal loss over pixels with target != ignore_index; 0.0 (finite, backward-able) if none (FLOOD-05)."""
    raise NotImplementedError
