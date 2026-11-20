"""CSI / FSS / MSE nowcast scores per lead (STORM-10). The formulas here are the contract (plain numpy/scipy)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.ndimage import uniform_filter

CSI_THRESHOLDS: tuple[int, ...] = (16, 74, 133, 160, 181, 219)


def csi(pred: np.ndarray, truth: np.ndarray, threshold: float) -> float:
    """hits / (hits + misses + false alarms) with events ``>= threshold``; nan if 0/0 (STORM-10)."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {truth.shape}")
    p, t = pred >= threshold, truth >= threshold
    hits = np.count_nonzero(p & t)
    denom = hits + np.count_nonzero(~p & t) + np.count_nonzero(p & ~t)
    return float(hits / denom) if denom else float("nan")


def fss(pred: np.ndarray, truth: np.ndarray, threshold: float, window: int) -> float:
    """Fractions skill score over a ``window x window`` neighbourhood (zero padded); nan if 0/0 (STORM-10)."""
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window must be odd and >= 1, got {window}")
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.shape != truth.shape or pred.ndim < 2:
        raise ValueError(f"need matching arrays with at least 2 dims, got {pred.shape} vs {truth.shape}")
    size = (1,) * (pred.ndim - 2) + (window, window)  # smooth only over the last two axes

    def fractions(field: np.ndarray) -> np.ndarray:
        return uniform_filter((field >= threshold).astype(np.float64), size=size, mode="constant", cval=0.0)

    pf, po = fractions(pred), fractions(truth)
    denom = np.sum(pf**2) + np.sum(po**2)
    return float(1.0 - np.sum((pf - po) ** 2) / denom) if denom else float("nan")


def storm_scores(
    pred: np.ndarray,
    truth: np.ndarray,
    thresholds: Sequence[int] = CSI_THRESHOLDS,
    fss_threshold: float = 133,
    fss_windows: Sequence[int] = (3, 7),
    crop: int | None = None,
) -> dict:
    """``{"csi": {thr: (12,)}, "fss": {window: (12,)}, "mse": (12,)}`` per lead, pooled over N (STORM-10)."""
    pred, truth = np.asarray(pred, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 4:
        raise ValueError(f"expected matching (N, 12, H, W) arrays, got {pred.shape} vs {truth.shape}")
    if crop is not None:
        h, w = pred.shape[2:]
        if crop > h or crop > w or crop < 1:
            raise ValueError(f"crop {crop} does not fit a {h}x{w} field")
        r0, c0 = (h - crop) // 2, (w - crop) // 2
        pred = pred[:, :, r0 : r0 + crop, c0 : c0 + crop]
        truth = truth[:, :, r0 : r0 + crop, c0 : c0 + crop]
    n_leads = pred.shape[1]
    return {
        "csi": {
            thr: np.array([csi(pred[:, k], truth[:, k], thr) for k in range(n_leads)]) for thr in thresholds
        },
        "fss": {
            win: np.array([fss(pred[:, k], truth[:, k], fss_threshold, win) for k in range(n_leads)])
            for win in fss_windows
        },
        "mse": ((pred - truth) ** 2).mean(axis=(0, 2, 3)),
    }
