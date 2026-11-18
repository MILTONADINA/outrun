"""Smoke baselines (persistence, calibrated raw HRRR-Smoke, linear regression) and metrics (SMOKE-05, SMOKE-09)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np


def persistence(pm_hist: np.ndarray) -> np.ndarray:
    """SMOKE-05. Most recent non-NaN history value per row; NaN if the row is all NaN."""
    raise NotImplementedError


def background_median(times: Sequence[datetime], pm: np.ndarray, at: datetime, days: int = 30) -> float:
    """SMOKE-05. NaN-ignoring median of pm over [at - days, at); NaN if empty."""
    raise NotImplementedError


class MassdenCalibrated:
    """SMOKE-05. pm_target - background = a * massden_f06 + b (OLS); predict max(0, a*m + b + background)."""

    a: float
    b: float

    def fit(
        self, massden_f06: np.ndarray, background: np.ndarray, pm_target: np.ndarray
    ) -> MassdenCalibrated:
        """Fit a, b. Returns self."""
        raise NotImplementedError

    def predict(self, massden_f06: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Before fit -> RuntimeError."""
        raise NotImplementedError


class LinearSmokeBaseline:
    """SMOKE-05. OLS on the model's flattened inputs, target in log1p space."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearSmokeBaseline:
        """Fit. Returns self."""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predictions in log1p space."""
        raise NotImplementedError


def smoke_metrics(
    pred_q50: np.ndarray,
    pred_q90: np.ndarray,
    obs: np.ndarray,
    episode: np.ndarray,
    threshold: float = 35.4,
) -> dict[str, float]:
    """SMOKE-09. rmse, mae, rmse_episode, mae_episode, csi, pod, far (obs/pred > threshold), q90_coverage."""
    raise NotImplementedError
