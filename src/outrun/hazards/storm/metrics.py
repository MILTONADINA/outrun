"""CSI / FSS / MSE nowcast scores per lead (STORM-10). May delegate to ``outrun.eval.metrics``."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

CSI_THRESHOLDS: tuple[int, ...] = (16, 74, 133, 160, 181, 219)


def csi(pred: np.ndarray, truth: np.ndarray, threshold: float) -> float:
    """hits / (hits + misses + false alarms) with events ``>= threshold``; nan if 0/0 (STORM-10)."""
    raise NotImplementedError


def fss(pred: np.ndarray, truth: np.ndarray, threshold: float, window: int) -> float:
    """Fractions skill score over a ``window x window`` neighbourhood (zero padded); nan if 0/0 (STORM-10)."""
    raise NotImplementedError


def storm_scores(
    pred: np.ndarray,
    truth: np.ndarray,
    thresholds: Sequence[int] = CSI_THRESHOLDS,
    fss_threshold: float = 133,
    fss_windows: Sequence[int] = (3, 7),
    crop: int | None = None,
) -> dict:
    """``{"csi": {thr: (12,)}, "fss": {window: (12,)}, "mse": (12,)}`` per lead, pooled over N (STORM-10)."""
    raise NotImplementedError
