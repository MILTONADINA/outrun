"""Fire baselines: persistence, downwind-dilated persistence, per-pixel logistic regression (FIRE-06, FIRE-07)."""

from __future__ import annotations

import numpy as np

PREV = 11  # PrevFireMask channel
TH = 1  # wind direction channel


def persistence(x: np.ndarray) -> np.ndarray:
    """PrevFireMask with values < 0 set to 0: (12,H,W) -> (H,W), (N,12,H,W) -> (N,H,W), float32 (FIRE-06)."""
    return np.maximum(np.asarray(x)[..., PREV, :, :], 0.0).astype(np.float32)


def _downwind_step(th: np.ndarray, burning: np.ndarray) -> tuple[int, int] | None:
    """(d_row, d_col) one pixel toward the circular-mean wind (burning pixels, else all), 45 degree steps."""
    sel = burning if burning.any() else np.ones_like(burning)
    rad = np.radians(th[sel].astype(np.float64))
    rad = rad[np.isfinite(rad)]
    if rad.size == 0:
        return None
    mean = np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean()))
    bearing = np.radians(np.round((mean + 180.0) / 45.0) * 45.0)
    return int(-np.rint(np.cos(bearing))), int(np.rint(np.sin(bearing)))


def _shift(mask: np.ndarray, d_row: int, d_col: int) -> np.ndarray:
    """``mask`` moved by (d_row, d_col); pixels pushed off the grid are dropped (no wrap-around)."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    src = mask[max(0, -d_row) : h - max(0, d_row), max(0, -d_col) : w - max(0, d_col)]
    out[max(0, d_row) : h - max(0, -d_row), max(0, d_col) : w - max(0, -d_col)] = src
    return out


def downwind_persistence(x: np.ndarray) -> np.ndarray:
    """Persistence OR persistence shifted one pixel downwind (circular-mean th, bearing th+180 rounded to 45 deg,
    step (-round(cos b), round(sin b)), no wrap-around) (FIRE-06)."""
    x = np.asarray(x)
    if x.ndim == 4:
        return np.stack([downwind_persistence(s) for s in x])
    base = persistence(x)
    step = _downwind_step(x[TH], base > 0)
    if step is None or step == (0, 0):
        return base
    return np.maximum(base, _shift(base, *step))


class PixelLogistic:
    """Per-pixel logistic regression on the 12 normalised channels; ignores y == -1 (FIRE-07).

    ``max_pixels`` random labelled pixels (seeded) are used to fit, because 15 k NDWS train samples hold 61 M pixels.
    """

    def __init__(self, C: float = 1.0, seed: int = 0, max_pixels: int = 2_000_000) -> None:
        self.C, self.seed, self.max_pixels = C, seed, max_pixels
        self._model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> PixelLogistic:
        """x (N, 12, H, W), y (N, H, W); fewer than two classes -> ValueError."""
        from sklearn.linear_model import LogisticRegression

        feats = np.moveaxis(np.asarray(x), 1, -1).reshape(-1, x.shape[1])
        lab = np.asarray(y).reshape(-1)
        keep = np.flatnonzero(lab != -1)
        if np.unique(lab[keep]).size < 2:
            raise ValueError("need both fire and no-fire pixels to fit")
        if keep.size > self.max_pixels:
            keep = np.sort(np.random.default_rng(self.seed).choice(keep, self.max_pixels, replace=False))
        self._model = LogisticRegression(C=self.C, max_iter=300, random_state=self.seed)
        self._model.fit(np.nan_to_num(feats[keep]), lab[keep].astype(int))
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """float32 (N, H, W) in [0, 1]; before fit -> RuntimeError."""
        if self._model is None:
            raise RuntimeError("PixelLogistic.fit has not been called")
        feats = np.moveaxis(np.asarray(x), 1, -1).reshape(-1, x.shape[1])
        p = self._model.predict_proba(np.nan_to_num(feats))[:, 1]
        return p.reshape(x.shape[0], *x.shape[2:]).astype(np.float32)
