"""Calibration and probabilistic scores (specs/02-grid-risk.md, CAL-01..07).

Metric functions take probabilities ``p`` and 0/1 labels ``y`` of equal shape (flattened) and an optional boolean
``mask`` (True = use). Bad input (shape mismatch, nothing usable, NaN/out-of-range p, non-binary y) -> ValueError.
Calibrators are fitted on validation data only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReliabilityBins:
    """Equal-width reliability bins (CAL-02). ``mean_pred``/``frac_pos`` are NaN for empty bins."""

    edges: np.ndarray
    counts: np.ndarray
    mean_pred: np.ndarray
    frac_pos: np.ndarray


P_CLIP = 1e-6  # temperature scaling clips p to [P_CLIP, 1 - P_CLIP] before the logit


def _prepare(p: Any, y: Any, mask: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Validate and flatten ``p`` and ``y`` (keeping only ``mask`` elements); returns float64 arrays."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"p {p.shape} and y {y.shape} must have the same shape")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != p.shape:
            raise ValueError(f"mask {mask.shape} must have the shape of p {p.shape}")
        p, y = p[mask], y[mask]
    p, y = p.ravel(), y.ravel()
    if p.size == 0:
        raise ValueError("no usable elements")
    if np.isnan(p).any() or p.min() < 0 or p.max() > 1:
        raise ValueError("p must be in [0, 1] with no NaN")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("labels must be 0 or 1")
    return p, y


def brier_score(p: Any, y: Any, mask: Any = None) -> float:
    """``mean((p - y)^2)`` (CAL-01)."""
    p, y = _prepare(p, y, mask)
    return float(np.mean((p - y) ** 2))


def brier_skill_score(p: Any, y: Any, climatology: float | None = None, mask: Any = None) -> float:
    """``1 - BS / BS_ref`` with reference ``climatology`` or ``mean(y)``; ``BS_ref == 0`` -> ValueError (CAL-01)."""
    p, y = _prepare(p, y, mask)
    c = float(np.mean(y)) if climatology is None else float(climatology)
    bs_ref = float(np.mean((c - y) ** 2))
    if bs_ref == 0:
        raise ValueError("reference Brier score is 0 (climatology is perfect); skill is undefined")
    return 1.0 - float(np.mean((p - y) ** 2)) / bs_ref


def reliability_bins(p: Any, y: Any, n_bins: int = 10, mask: Any = None) -> ReliabilityBins:
    """Bin = ``min(floor(p * n_bins), n_bins - 1)`` (CAL-02)."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    p, y = _prepare(p, y, mask)
    idx = np.minimum(np.floor(p * n_bins).astype(np.int64), n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    sum_p = np.bincount(idx, weights=p, minlength=n_bins)
    sum_y = np.bincount(idx, weights=y, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):  # empty bins become NaN
        mean_pred = np.where(counts > 0, sum_p / counts, np.nan)
        frac_pos = np.where(counts > 0, sum_y / counts, np.nan)
    return ReliabilityBins(
        edges=np.linspace(0.0, 1.0, n_bins + 1), counts=counts, mean_pred=mean_pred, frac_pos=frac_pos
    )


def expected_calibration_error(p: Any, y: Any, n_bins: int = 10, mask: Any = None) -> float:
    """``sum_b (n_b / N) |frac_pos_b - mean_pred_b|`` over non-empty bins (CAL-03)."""
    b = reliability_bins(p, y, n_bins=n_bins, mask=mask)
    used = b.counts > 0
    weights = b.counts[used] / b.counts.sum()
    return float(np.sum(weights * np.abs(b.frac_pos[used] - b.mean_pred[used])))


class IsotonicCalibrator:
    """Monotone non-decreasing map fitted on val; out-of-range inputs clipped (CAL-04).

    Fitting is the pool-adjacent-violators algorithm (scikit-learn ``IsotonicRegression``): sort by ``p`` and
    average neighbouring groups of labels until the averages never go down. Equal ``p`` values share one output.
    The result is a step-like curve stored as knots ``(x_knots, y_knots)``; ``transform`` interpolates linearly
    between knots and holds the end values outside them, so the calibrator is just two arrays (easy to save).
    """

    def __init__(self) -> None:
        self.x_knots: np.ndarray | None = None
        self.y_knots: np.ndarray | None = None

    def fit(self, p: Any, y: Any, mask: Any = None) -> IsotonicCalibrator:
        from sklearn.isotonic import IsotonicRegression

        p, y = _prepare(p, y, mask)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(p, y)
        self.x_knots = np.asarray(iso.X_thresholds_, dtype=np.float64)
        self.y_knots = np.asarray(iso.y_thresholds_, dtype=np.float64)
        return self

    def transform(self, p: Any) -> np.ndarray:
        """float32, input shape kept; RuntimeError before fit."""
        if self.x_knots is None or self.y_knots is None:
            raise RuntimeError("IsotonicCalibrator.transform called before fit")
        p = np.asarray(p, dtype=np.float64)
        # np.interp holds y_knots[0] / y_knots[-1] outside the fitted range (= clipping the input).
        return np.interp(p, self.x_knots, self.y_knots).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """``{"type": "isotonic", ...}`` (CAL-06)."""
        if self.x_knots is None or self.y_knots is None:
            raise RuntimeError("IsotonicCalibrator.to_dict called before fit")
        return {"type": "isotonic", "x": self.x_knots.tolist(), "y": self.y_knots.tolist()}


class TemperatureCalibrator:
    """``sigmoid(logit(p) / T)`` with one fitted ``temperature`` T > 0 (NLL on val) (CAL-05).

    T > 1 softens overconfident scores towards 0.5, T < 1 sharpens them. T is found by a bounded 1-D search
    over ``log T`` in [log 0.01, log 100] minimising the mean negative log-likelihood.
    """

    temperature: float

    def __init__(self) -> None:
        self.temperature = float("nan")
        self._fitted = False

    @staticmethod
    def _logit(p: Any) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=np.float64), P_CLIP, 1.0 - P_CLIP)
        return np.log(p) - np.log1p(-p)

    def fit(self, p: Any, y: Any, mask: Any = None) -> TemperatureCalibrator:
        from scipy.optimize import minimize_scalar

        p, y = _prepare(p, y, mask)
        z = self._logit(p)

        def nll(log_t: float) -> float:
            s = z / np.exp(log_t)
            # -[y log sigmoid(s) + (1 - y) log(1 - sigmoid(s))], written with logaddexp so it never overflows
            return float(np.mean(y * np.logaddexp(0.0, -s) + (1.0 - y) * np.logaddexp(0.0, s)))

        res = minimize_scalar(nll, bounds=(np.log(1e-2), np.log(1e2)), method="bounded")
        self.temperature = float(np.exp(res.x))
        self._fitted = True
        return self

    def transform(self, p: Any) -> np.ndarray:
        """float32, input shape kept; RuntimeError before fit."""
        if not self._fitted:
            raise RuntimeError("TemperatureCalibrator.transform called before fit")
        from scipy.special import expit

        return expit(self._logit(p) / self.temperature).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        """``{"type": "temperature", ...}`` (CAL-06)."""
        if not self._fitted:
            raise RuntimeError("TemperatureCalibrator.to_dict called before fit")
        return {"type": "temperature", "temperature": self.temperature}


def calibrator_from_dict(d: dict[str, Any]) -> IsotonicCalibrator | TemperatureCalibrator:
    """Rebuild a calibrator from ``to_dict`` output; unknown type -> ValueError (CAL-06)."""
    kind = d.get("type")
    if kind == "isotonic":
        iso = IsotonicCalibrator()
        iso.x_knots = np.asarray(d["x"], dtype=np.float64)
        iso.y_knots = np.asarray(d["y"], dtype=np.float64)
        return iso
    if kind == "temperature":
        temp = TemperatureCalibrator()
        temp.temperature = float(d["temperature"])
        temp._fitted = True
        return temp
    raise ValueError(f"unknown calibrator type {kind!r}")


def plot_reliability(bins: ReliabilityBins, path: str | Path, title: str = "") -> Path:
    """Write a reliability-diagram PNG (Agg backend) and return ``path`` (CAL-07)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # A bare Figure + Agg canvas: no pyplot global state, works headless.
    fig = Figure(figsize=(5, 6))
    FigureCanvasAgg(fig)
    ax, ax_counts = fig.subplots(2, 1, height_ratios=[3, 1], sharex=True)
    used = bins.counts > 0
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(bins.mean_pred[used], bins.frac_pos[used], "o-", label="model")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    ax.set_title(title)
    centres = (bins.edges[:-1] + bins.edges[1:]) / 2
    ax_counts.bar(centres, bins.counts, width=np.diff(bins.edges) * 0.9)
    ax_counts.set_yscale("log" if bins.counts.max() > 0 else "linear")
    ax_counts.set_xlabel("predicted probability")
    ax_counts.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(str(path), format="png")
    return Path(path)
