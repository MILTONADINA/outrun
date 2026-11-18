"""Fire baselines: persistence, downwind-dilated persistence, per-pixel logistic regression (FIRE-06, FIRE-07)."""

from __future__ import annotations

import numpy as np


def persistence(x: np.ndarray) -> np.ndarray:
    """PrevFireMask with values < 0 set to 0: (12,H,W) -> (H,W), (N,12,H,W) -> (N,H,W), float32 (FIRE-06)."""
    raise NotImplementedError


def downwind_persistence(x: np.ndarray) -> np.ndarray:
    """Persistence OR persistence shifted one pixel downwind (circular-mean th, bearing th+180 rounded to 45 deg,
    step (-round(cos b), round(sin b)), no wrap-around) (FIRE-06)."""
    raise NotImplementedError


class PixelLogistic:
    """Per-pixel logistic regression on the 12 normalised channels; ignores y == -1 (FIRE-07)."""

    def __init__(self, C: float = 1.0, seed: int = 0) -> None:
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray) -> PixelLogistic:
        """x (N, 12, H, W), y (N, H, W); fewer than two classes -> ValueError."""
        raise NotImplementedError

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """float32 (N, H, W) in [0, 1]; before fit -> RuntimeError."""
        raise NotImplementedError
