"""Ice MIL-CNN: U-Net cell logits, LogSumExp zone pooling, zone-level isotonic calibration (ICE-04, ICE-06)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from torch import nn


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class IceMILNet(nn.Module):
    """ICE-04. U-Net encoder-decoder (default about 2 M params, 0.5-5 M) giving per-cell logits.

    forward(x[B, in_channels, H, W]) -> logits[B, 1, H, W] for H, W multiples of 8.
    """

    def __init__(self, in_channels: int = 20, base: int = 32) -> None:
        super().__init__()
        self.in_channels = in_channels
        chans = [base, 2 * base, 4 * base, 8 * base]
        self.enc = nn.ModuleList([_block(in_channels, chans[0])])
        self.enc.extend(_block(chans[i], chans[i + 1]) for i in range(3))
        self.up = nn.ModuleList([nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in range(3)])
        self.dec = nn.ModuleList([_block(2 * chans[i], chans[i]) for i in range(3)])
        self.head = nn.Conv2d(chans[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x.shape[1]}")
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips[-1]
        for i in (2, 1, 0):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], dim=1))
        return self.head(x)


def mil_pool(cell_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ICE-04. Zone logit[B] = LogSumExp of cell logits over in-zone cells only. Empty mask -> ValueError."""
    m = mask.bool()
    if not bool(m.flatten(1).any(dim=1).all()):
        raise ValueError("every bag needs at least one in-zone cell")
    logits = cell_logits[:, 0].masked_fill(~m, float("-inf"))
    return torch.logsumexp(logits.flatten(1), dim=1)


def mil_loss(cell_logits: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ICE-04. Mean BCE-with-logits of mil_pool(cell_logits, mask) against zone labels in {0, 1}."""
    return F.binary_cross_entropy_with_logits(mil_pool(cell_logits, mask), labels.float())


def cell_probabilities(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """ICE-04. sigmoid(cell logits) as a numpy (B, H, W) array, eval mode, no grad. Values in [0, 1]."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x))[:, 0].cpu().numpy()


class ZoneCalibrator:
    """ICE-06. Isotonic calibration fitted on validation zone probabilities, applied to any-shape arrays."""

    _iso: IsotonicRegression | None = None

    def fit(self, zone_probs: np.ndarray, labels: np.ndarray) -> ZoneCalibrator:
        """Fit isotonic regression (out-of-bounds clipped). Returns self."""
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self._iso = iso.fit(np.asarray(zone_probs, float).ravel(), np.asarray(labels, float).ravel())
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        """Monotone map to [0, 1], same shape as ``p``. Before fit -> RuntimeError."""
        if self._iso is None:
            raise RuntimeError("ZoneCalibrator.transform called before fit")
        p = np.asarray(p, float)
        return np.clip(self._iso.predict(p.ravel()), 0.0, 1.0).reshape(p.shape)

    def save(self, path: str | Path) -> None:
        """Persist the fitted calibrator."""
        if self._iso is None:
            raise RuntimeError("cannot save an unfitted ZoneCalibrator")
        with open(path, "wb") as f:
            pickle.dump(self._iso, f)

    @classmethod
    def load(cls, path: str | Path) -> ZoneCalibrator:
        """Load a calibrator saved with ``save``; outputs identical to the original."""
        with open(path, "rb") as f:
            iso = pickle.load(f)  # noqa: S301 - our own checkpoint file
        cal = cls()
        cal._iso = iso
        return cal
