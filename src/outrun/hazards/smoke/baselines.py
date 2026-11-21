"""Smoke baselines (persistence, calibrated raw HRRR-Smoke, linear regression) and metrics (SMOKE-05, SMOKE-09)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import numpy as np


def persistence(pm_hist: np.ndarray) -> np.ndarray:
    """SMOKE-05. Most recent non-NaN history value per row; NaN if the row is all NaN."""
    h = np.asarray(pm_hist, dtype=np.float64)
    valid = ~np.isnan(h)
    last = h.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)  # index of the newest valid value
    out = h[np.arange(len(h)), last]
    return np.where(valid.any(axis=1), out, np.nan)


def background_median(times: Sequence[datetime], pm: np.ndarray, at: datetime, days: int = 30) -> float:
    """SMOKE-05. NaN-ignoring median of pm over [at - days, at); NaN if empty."""
    t = (
        np.asarray([_naive_utc(x) for x in times], dtype="datetime64[us]")
        if not isinstance(times, np.ndarray)
        else times
    )
    lo = np.datetime64(_naive_utc(at - timedelta(days=days)))
    hi = np.datetime64(_naive_utc(at))
    vals = np.asarray(pm, dtype=np.float64)[(t >= lo) & (t < hi)]
    vals = vals[~np.isnan(vals)]
    return float(np.median(vals)) if len(vals) else float("nan")


def _naive_utc(t: datetime) -> datetime:
    return t.astimezone(UTC).replace(tzinfo=None)


class MassdenCalibrated:
    """SMOKE-05. pm_target - background = a * massden_f06 + b (OLS); predict max(0, a*m + b + background)."""

    a: float
    b: float

    def fit(
        self, massden_f06: np.ndarray, background: np.ndarray, pm_target: np.ndarray
    ) -> MassdenCalibrated:
        """Fit a, b. Returns self."""
        m = np.asarray(massden_f06, dtype=np.float64)
        y = np.asarray(pm_target, dtype=np.float64) - np.asarray(background, dtype=np.float64)
        ok = ~(np.isnan(m) | np.isnan(y))
        design = np.stack([m[ok], np.ones(int(ok.sum()))], axis=1)
        (a, b), *_ = np.linalg.lstsq(design, y[ok], rcond=None)
        self.a, self.b = float(a), float(b)
        return self

    def predict(self, massden_f06: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Before fit -> RuntimeError."""
        if not hasattr(self, "a"):
            raise RuntimeError("MassdenCalibrated.predict called before fit")
        out = (
            self.a * np.asarray(massden_f06, dtype=np.float64)
            + self.b
            + np.asarray(background, dtype=np.float64)
        )
        return np.maximum(out, 0.0)


class LinearSmokeBaseline:
    """SMOKE-05. OLS on the model's flattened inputs, target in log1p space."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearSmokeBaseline:
        """Fit. Returns self."""
        X = np.asarray(X, dtype=np.float64)
        design = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        w, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=np.float64), rcond=None)
        self.coef_, self.intercept_ = w[:-1], float(w[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predictions in log1p space."""
        if not hasattr(self, "coef_"):
            raise RuntimeError("LinearSmokeBaseline.predict called before fit")
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_


def smoke_metrics(
    pred_q50: np.ndarray,
    pred_q90: np.ndarray,
    obs: np.ndarray,
    episode: np.ndarray,
    threshold: float = 35.4,
) -> dict[str, float]:
    """SMOKE-09. rmse, mae, rmse_episode, mae_episode, csi, pod, far (obs/pred > threshold), q90_coverage."""
    q50, q90, o, ep = (np.asarray(a) for a in (pred_q50, pred_q90, obs, episode))
    if not (len(q50) == len(q90) == len(o) == len(ep)) or len(o) == 0:
        raise ValueError("smoke_metrics needs equal-length, non-empty inputs")
    q50, q90, o, ep = q50.astype(np.float64), q90.astype(np.float64), o.astype(np.float64), ep.astype(bool)
    err = q50 - o
    hit = (o > threshold) & (q50 > threshold)
    miss = (o > threshold) & ~(q50 > threshold)
    fa = ~(o > threshold) & (q50 > threshold)
    n_h, n_m, n_f = int(hit.sum()), int(miss.sum()), int(fa.sum())

    def ratio(a: int, b: int) -> float:
        return a / b if b else 0.0

    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "rmse_episode": float(np.sqrt(np.mean(err[ep] ** 2))) if ep.any() else float("nan"),
        "mae_episode": float(np.mean(np.abs(err[ep]))) if ep.any() else float("nan"),
        "csi": ratio(n_h, n_h + n_m + n_f),
        "pod": ratio(n_h, n_h + n_m),
        "far": ratio(n_f, n_h + n_f),
        "q90_coverage": float(np.mean(o <= q90)),
    }
