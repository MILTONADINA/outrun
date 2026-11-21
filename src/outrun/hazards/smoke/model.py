"""Smoke CNN + MLP quantile model and pinball loss (SMOKE-04)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


class SmokeNet(nn.Module):
    """SMOKE-04. CNN branch (2 conv + pool) on the 8x9x9 HRRR patch, MLP on tabular features, 3 dense layers.

    forward(patch[B, 8, 9, 9], tab[B, n_tabular]) -> [B, len(quantiles)] in log1p-PM2.5 space. ~0.1 M params.

    The model standardises its own inputs with four buffers (per-channel patch mean/std, per-feature tab mean/std).
    They start as identity and are set from the TRAIN statistics (``set_scaler``), so ``best.pt`` is self-contained
    and training, validation and the adapter all feed the same physical (already transformed) features.
    """

    def __init__(
        self,
        in_channels: int = 8,
        patch: int = 9,
        n_tabular: int = 11,
        quantiles: tuple[float, ...] = (0.5, 0.9),
    ) -> None:
        super().__init__()
        self.in_channels, self.patch, self.n_tabular = in_channels, patch, n_tabular
        self.quantiles = tuple(quantiles)
        self.register_buffer("patch_mean", torch.zeros(in_channels, 1, 1))
        self.register_buffer("patch_std", torch.ones(in_channels, 1, 1))
        self.register_buffer("tab_mean", torch.zeros(n_tabular))
        self.register_buffer("tab_std", torch.ones(n_tabular))
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        flat = 48 * (patch // 2) ** 2
        self.mlp = nn.Sequential(nn.Linear(n_tabular, 32), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(flat + 32, 96),
            nn.ReLU(),
            nn.Linear(96, 48),
            nn.ReLU(),
            nn.Linear(48, len(self.quantiles)),
        )

    def set_scaler(
        self, patch_mean: np.ndarray, patch_std: np.ndarray, tab_mean: np.ndarray, tab_std: np.ndarray
    ) -> None:
        """Store train statistics (stds floored so a constant channel does not blow up)."""
        self.patch_mean.copy_(torch.as_tensor(patch_mean, dtype=torch.float32).reshape(-1, 1, 1))
        self.patch_std.copy_(
            torch.as_tensor(np.maximum(patch_std, 1e-6), dtype=torch.float32).reshape(-1, 1, 1)
        )
        self.tab_mean.copy_(torch.as_tensor(tab_mean, dtype=torch.float32))
        self.tab_std.copy_(torch.as_tensor(np.maximum(tab_std, 1e-6), dtype=torch.float32))

    def forward(self, patch: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:  # noqa: D102
        if patch.shape[1:] != (self.in_channels, self.patch, self.patch):
            raise ValueError(
                f"patch must be (B, {self.in_channels}, {self.patch}, {self.patch}), got {tuple(patch.shape)}"
            )
        if tab.shape[1:] != (self.n_tabular,) or tab.shape[0] != patch.shape[0]:
            raise ValueError(f"tab must be (B, {self.n_tabular}), got {tuple(tab.shape)}")
        p = (patch - self.patch_mean) / self.patch_std
        t = (tab - self.tab_mean) / self.tab_std
        return self.head(torch.cat([self.cnn(p), self.mlp(t)], dim=1))


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: Sequence[float]) -> torch.Tensor:
    """SMOKE-04. mean over b, q of max(q * (y - pred), (q - 1) * (y - pred))."""
    q = torch.as_tensor(list(quantiles), dtype=pred.dtype, device=pred.device)
    err = target.unsqueeze(1) - pred
    return torch.maximum(q * err, (q - 1.0) * err).mean()


def to_pm(pred_log: np.ndarray) -> np.ndarray:
    """SMOKE-04. expm1, clipped at 0, running max across the quantile axis (non-crossing)."""
    pm = np.maximum(np.expm1(np.asarray(pred_log, dtype=np.float64)), 0.0)
    return np.maximum.accumulate(pm, axis=1)
