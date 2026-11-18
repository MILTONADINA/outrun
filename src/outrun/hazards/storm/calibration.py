"""Per-lead isotonic calibration and 15-minute slotting (STORM-07).

May delegate the isotonic fit to ``outrun.core.calibrate``.
"""

from __future__ import annotations

import numpy as np


class PerLeadCalibrator:
    """One non-decreasing isotonic map per lead (axis 1), output in [0, 1] (STORM-07).

    ``transform`` before ``fit`` -> RuntimeError; lead count mismatch -> ValueError.
    """

    def fit(self, probs: np.ndarray, truth: np.ndarray) -> PerLeadCalibrator:
        raise NotImplementedError

    def transform(self, probs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def frames_to_slots(p: np.ndarray, frames_per_slot: int = 3) -> np.ndarray:
    """Element-wise max over consecutive groups of frames along axis 0 (STORM-07)."""
    raise NotImplementedError
