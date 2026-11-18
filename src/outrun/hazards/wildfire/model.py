"""Fire U-Net and BCE + Dice loss with uncertain pixels ignored (specs/11-wildfire.md: FIRE-04, FIRE-05)."""

from __future__ import annotations

import torch
from torch import nn


class FireUNet(nn.Module):
    """U-Net: (B, in_channels, H, W) -> logits (B, 1, H, W); H, W divisible by 2**levels (FIRE-04).

    Defaults (12 NDWS channels incl. the satellite channels, base 32, 3 levels) give 0.5-5 M parameters.
    Wrong channel count -> ValueError.
    """

    def __init__(self, in_channels: int = 12, base: int = 32, levels: int = 3) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def fire_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 3.0,
    dice_weight: float = 1.0,
    ignore_index: int = -1,
    smooth: float = 1.0,
) -> torch.Tensor:
    """BCE(pos_weight) mean over valid pixels + dice_weight * soft Dice over valid pixels; 0.0 if none (FIRE-05)."""
    raise NotImplementedError
