"""Storm U-Net (STORM-04), exceedance targets / intensity weights (STORM-03), loss and training step (STORM-05)."""

from __future__ import annotations

import torch
from torch import nn

VIL_THRESHOLDS: tuple[int, ...] = (74, 133, 181)
RISK_THRESHOLD = 133
RISK_THRESHOLD_INDEX = 1


def exceedance_targets(y: torch.Tensor) -> torch.Tensor:
    """``(..., 12, H, W)`` VIL units -> float32 ``(..., 12, 3, H, W)`` of ``y >= VIL_THRESHOLDS[k]`` (STORM-03)."""
    raise NotImplementedError


def intensity_weights(y: torch.Tensor) -> torch.Tensor:
    """SEVIR-style weights 1/2/5/10/20/30 at VIL breaks 16/74/133/160/181 (STORM-03)."""
    raise NotImplementedError


class StormUNet(nn.Module):
    """2D U-Net, time as channels (STORM-04).

    ``forward(x (B, in_channels, H, W)) -> (vil (B, n_out, H, W), logits (B, n_out, n_thresholds, H, W))``.
    ValueError if H/W not divisible by ``2**levels`` or the channel count is wrong.
    """

    def __init__(
        self, in_channels: int = 26, n_out: int = 12, n_thresholds: int = 3, base: int = 32, levels: int = 4
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def predict_proba(model: StormUNet, x: torch.Tensor) -> torch.Tensor:
    """Eval mode, no grad: ``sigmoid(logits)`` ``(B, n_out, n_thresholds, H, W)`` in [0, 1] (STORM-04)."""
    raise NotImplementedError


def storm_loss(
    vil_pred: torch.Tensor, logits: torch.Tensor, y: torch.Tensor, bce_weight: float = 1.0
) -> torch.Tensor:
    """Weighted MSE + MAE on ``y/255`` plus BCE on exceedance targets (STORM-05)."""
    raise NotImplementedError


def train_step(
    model: StormUNet, batch: tuple[torch.Tensor, torch.Tensor], optimizer: torch.optim.Optimizer
) -> float:
    """One optimisation step; returns the loss before the step (STORM-05)."""
    raise NotImplementedError
