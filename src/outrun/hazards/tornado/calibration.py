"""Isotonic calibration of detector scores (TORN-05). May delegate to ``outrun.core.calibrate``."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Non-decreasing map to [0, 1], clipped outside the fitted range; RuntimeError before ``fit`` (TORN-05)."""

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64))
        self._iso = iso
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("IsotonicCalibrator.transform called before fit")
        return np.clip(self._iso.predict(np.asarray(p, dtype=np.float64)), 0.0, 1.0)

    def to_dict(self) -> dict:
        """Fitted step function as plain lists (for ``calibrator.json``)."""
        if self._iso is None:
            raise RuntimeError("IsotonicCalibrator.to_dict called before fit")
        return {"x": self._iso.X_thresholds_.tolist(), "y": self._iso.y_thresholds_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> IsotonicCalibrator:
        cal = cls()
        x, y = np.asarray(d["x"], dtype=np.float64), np.asarray(d["y"], dtype=np.float64)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(x, y)  # re-fitting a monotone step function reproduces it
        cal._iso = iso
        return cal
