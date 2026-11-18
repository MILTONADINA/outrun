"""Isotonic calibration of detector scores (TORN-05). May delegate to ``outrun.core.calibrate``."""

from __future__ import annotations

import numpy as np


class IsotonicCalibrator:
    """Non-decreasing map to [0, 1], clipped outside the fitted range; RuntimeError before ``fit`` (TORN-05)."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        raise NotImplementedError

    def transform(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError
