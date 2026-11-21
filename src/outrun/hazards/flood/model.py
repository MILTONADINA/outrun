"""Flood U-Net and focal loss with the uncertain ring masked (specs/10-flood.md: FLOOD-04, FLOOD-05)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dataset import FLASH_CHANNELS, FLOOD_CHANNELS

# Fixed (not fitted) input scaling so the network can take the raw FLOOD-01 stack: precipitation, FFG ratio and ARI
# are heavy-tailed, so they go through a signed log1p; terrain is divided by a typical magnitude.
_LOG_CHANNELS = ("qpe_1h", "qpe_3h", "qpe_6h", "qpe_24h", "ffg_ratio_1h", "ffg_ratio_3h", "ari_max")
_DIVISORS = {"elevation": 1000.0, "slope": 5.0, "tpi": 20.0}


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class FloodUNet(nn.Module):
    """U-Net: (B, in_channels, H, W) -> logits (B, 1, H, W); H, W divisible by 2**levels (FLOOD-04).

    Defaults (11 channels, base 32, 3 levels) give 0.5-5 M parameters. Wrong channel count -> ValueError.
    ``in_channels=8`` is the no-FLASH ablation (FLOOD-09). The first step is a fixed per-channel input scaling.
    """

    def __init__(self, in_channels: int = 11, base: int = 32, levels: int = 3) -> None:
        super().__init__()
        self.in_channels, self.levels = in_channels, levels
        names = (
            FLOOD_CHANNELS
            if in_channels == len(FLOOD_CHANNELS)
            else tuple(c for c in FLOOD_CHANNELS if c not in FLASH_CHANNELS)
        )
        if in_channels not in (len(FLOOD_CHANNELS), len(names)):
            names = ()  # unknown layout: no scaling
        self.register_buffer("is_log", torch.tensor([c in _LOG_CHANNELS for c in names], dtype=torch.bool))
        self.register_buffer("div", torch.tensor([_DIVISORS.get(c, 1.0) for c in names], dtype=torch.float32))
        widths = [base * 2**i for i in range(levels + 1)]
        self.enc = nn.ModuleList([_conv_block(in_channels, widths[0])])
        for i in range(levels):
            self.enc.append(_conv_block(widths[i], widths[i + 1]))
        self.up = nn.ModuleList(
            [nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in range(levels)]
        )
        self.dec = nn.ModuleList([_conv_block(2 * widths[i], widths[i]) for i in range(levels)])
        self.head = nn.Conv2d(widths[0], 1, 1)

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_log.numel() == 0:
            return x
        c = self.is_log.view(1, -1, 1, 1)
        logged = torch.sign(x) * torch.log1p(x.abs())
        return torch.where(c, logged, x / self.div.view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (B, {self.in_channels}, H, W), got {tuple(x.shape)}")
        h = self._scale(x)
        skips = []
        for i, block in enumerate(self.enc):
            h = block(h)
            if i < self.levels:
                skips.append(h)
                h = F.max_pool2d(h, 2)
        for i in reversed(range(self.levels)):
            h = self.dec[i](torch.cat([self.up[i](h), skips[i]], dim=1))
        return self.head(h)


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float | None = 0.25,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Mean focal loss over pixels with target != ignore_index; 0.0 (finite, backward-able) if none (FLOOD-05)."""
    if logits.shape != target.shape:
        raise ValueError(f"logits {tuple(logits.shape)} and target {tuple(target.shape)} shapes differ")
    keep = target != ignore_index
    t = torch.where(keep, target, torch.zeros_like(target)).to(logits.dtype)
    z = torch.where(keep, logits, torch.zeros_like(logits))  # ignored logits never enter the graph
    log_pt = -F.binary_cross_entropy_with_logits(z, t, reduction="none")  # log p_t, stable
    p_t = log_pt.exp()
    loss = -((1 - p_t) ** gamma) * log_pt
    if alpha is not None:
        loss = loss * torch.where(t == 1, alpha, 1 - alpha)
    n = keep.sum()
    return (loss * keep).sum() / n.clamp(min=1)
