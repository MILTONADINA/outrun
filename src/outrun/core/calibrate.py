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


def brier_score(p: Any, y: Any, mask: Any = None) -> float:
    """``mean((p - y)^2)`` (CAL-01)."""
    raise NotImplementedError


def brier_skill_score(p: Any, y: Any, climatology: float | None = None, mask: Any = None) -> float:
    """``1 - BS / BS_ref`` with reference ``climatology`` or ``mean(y)``; ``BS_ref == 0`` -> ValueError (CAL-01)."""
    raise NotImplementedError


def reliability_bins(p: Any, y: Any, n_bins: int = 10, mask: Any = None) -> ReliabilityBins:
    """Bin = ``min(floor(p * n_bins), n_bins - 1)`` (CAL-02)."""
    raise NotImplementedError


def expected_calibration_error(p: Any, y: Any, n_bins: int = 10, mask: Any = None) -> float:
    """``sum_b (n_b / N) |frac_pos_b - mean_pred_b|`` over non-empty bins (CAL-03)."""
    raise NotImplementedError


class IsotonicCalibrator:
    """Monotone non-decreasing map fitted on val; out-of-range inputs clipped (CAL-04)."""

    def fit(self, p: Any, y: Any, mask: Any = None) -> IsotonicCalibrator:
        raise NotImplementedError

    def transform(self, p: Any) -> np.ndarray:
        """float32, input shape kept; RuntimeError before fit."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """``{"type": "isotonic", ...}`` (CAL-06)."""
        raise NotImplementedError


class TemperatureCalibrator:
    """``sigmoid(logit(p) / T)`` with one fitted ``temperature`` T > 0 (NLL on val) (CAL-05)."""

    temperature: float

    def fit(self, p: Any, y: Any, mask: Any = None) -> TemperatureCalibrator:
        raise NotImplementedError

    def transform(self, p: Any) -> np.ndarray:
        """float32, input shape kept; RuntimeError before fit."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """``{"type": "temperature", ...}`` (CAL-06)."""
        raise NotImplementedError


def calibrator_from_dict(d: dict[str, Any]) -> IsotonicCalibrator | TemperatureCalibrator:
    """Rebuild a calibrator from ``to_dict`` output; unknown type -> ValueError (CAL-06)."""
    raise NotImplementedError


def plot_reliability(bins: ReliabilityBins, path: str | Path, title: str = "") -> Path:
    """Write a reliability-diagram PNG (Agg backend) and return ``path`` (CAL-07)."""
    raise NotImplementedError
