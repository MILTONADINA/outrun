"""Fire U-Net and BCE + Dice loss with uncertain pixels ignored (specs/11-wildfire.md: FIRE-04, FIRE-05)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


def _block(c_in: int, c_out: int) -> nn.Sequential:
    """Two (3x3 conv, batch norm, ReLU) layers."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class FireUNet(nn.Module):
    """U-Net: (B, in_channels, H, W) -> logits (B, 1, H, W); H, W divisible by 2**levels (FIRE-04).

    Defaults (12 NDWS channels incl. the satellite channels, base 32, 3 levels) give about 1.9 M parameters.
    Wrong channel count -> ValueError.
    """

    def __init__(self, in_channels: int = 12, base: int = 32, levels: int = 3) -> None:
        super().__init__()
        self.in_channels, self.base, self.levels = in_channels, base, levels
        widths = [base * 2**i for i in range(levels + 1)]
        self.inc = _block(in_channels, widths[0])
        self.down = nn.ModuleList(_block(widths[i], widths[i + 1]) for i in range(levels))
        self.up = nn.ModuleList(
            nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in range(levels)
        )
        self.dec = nn.ModuleList(_block(2 * widths[i], widths[i]) for i in range(levels))
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (B, {self.in_channels}, H, W), got {tuple(x.shape)}")
        skips = [self.inc(x)]
        for block in self.down:
            skips.append(block(F.max_pool2d(skips[-1], 2)))
        h = skips.pop()
        for i in reversed(range(self.levels)):
            h = self.dec[i](torch.cat([self.up[i](h), skips[i]], dim=1))
        return self.head(h)

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> FireUNet:
        """Rebuild the architecture from a checkpoint's weight shapes (in_channels, base, levels)."""
        w = state["inc.0.weight"]
        levels = sum(1 for k in state if k.startswith("down.") and k.endswith(".0.weight"))
        net = cls(in_channels=int(w.shape[1]), base=int(w.shape[0]), levels=levels)
        net.load_state_dict(state)
        return net


def fire_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 3.0,
    dice_weight: float = 1.0,
    ignore_index: int = -1,
    smooth: float = 1.0,
) -> torch.Tensor:
    """BCE(pos_weight) mean over valid pixels + dice_weight * soft Dice over valid pixels; 0.0 if none (FIRE-05)."""
    if logits.shape != target.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and target {tuple(target.shape)} must match")
    valid = target != ignore_index
    n_valid = valid.sum()
    if n_valid == 0:
        return logits.sum() * 0.0  # finite zero that still supports .backward()
    v = valid.to(logits.dtype)
    t = torch.where(valid, target, torch.zeros_like(target)).to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        t,
        pos_weight=torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device),
        reduction="none",
    )
    bce = (bce * v).sum() / n_valid
    p = torch.sigmoid(logits) * v
    dice = 1.0 - (2.0 * (p * t).sum() + smooth) / (p.sum() + t.sum() + smooth)
    return bce + dice_weight * dice


class TemperatureScaler:
    """Calibration fitted on val: p -> sigmoid((logit(p) - bias) / T). ``transform`` takes and returns probabilities.

    Temperature alone cannot undo the upward shift caused by ``pos_weight`` (the loss over-weights fire pixels), so a
    bias is fitted with it (a two-parameter Platt fit on the logits); T = 1, bias = 0 is the identity.
    """

    def __init__(self, temperature: float = 1.0, bias: float = 0.0) -> None:
        self.temperature, self.bias = float(temperature), float(bias)

    def fit(self, logits: np.ndarray, label: np.ndarray) -> TemperatureScaler:
        """Minimise BCE over label != -1 with LBFGS on (1/T, bias / T)."""
        keep = label != -1
        z = torch.as_tensor(np.asarray(logits)[keep], dtype=torch.float64)
        y = torch.as_tensor(np.asarray(label)[keep], dtype=torch.float64)
        a = torch.ones((), dtype=torch.float64, requires_grad=True)  # 1 / T
        b = torch.zeros((), dtype=torch.float64, requires_grad=True)  # -bias / T
        opt = torch.optim.LBFGS([a, b], max_iter=100, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(a * z + b, y)
            loss.backward()
            return loss

        opt.step(closure)
        a_v = float(a.detach())
        self.temperature, self.bias = 1.0 / a_v, -float(b.detach()) / a_v
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
        z = (np.log(p / (1 - p)) - self.bias) / self.temperature
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"temperature": self.temperature, "bias": self.bias}))

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        return cls(**json.loads(Path(path).read_text()))
