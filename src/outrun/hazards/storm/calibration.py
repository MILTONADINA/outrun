"""Per-lead isotonic calibration and 15-minute slotting (STORM-07).

The isotonic fit itself is ``outrun.core.calibrate.IsotonicCalibrator``; this module only runs one per lead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from outrun.core.calibrate import IsotonicCalibrator


class PerLeadCalibrator:
    """One non-decreasing isotonic map per lead (axis 1), output in [0, 1] (STORM-07).

    ``transform`` before ``fit`` -> RuntimeError; lead count mismatch -> ValueError.
    """

    def __init__(self) -> None:
        self._maps: list[IsotonicCalibrator] | None = None

    def fit(self, probs: np.ndarray, truth: np.ndarray) -> PerLeadCalibrator:
        probs = np.asarray(probs)
        truth = np.asarray(truth)
        if probs.ndim < 2 or probs.shape != truth.shape:
            raise ValueError(f"probs {probs.shape} and truth {truth.shape} must match and have a lead axis")
        self._maps = [
            IsotonicCalibrator().fit(probs[:, lead].ravel(), truth[:, lead].ravel().astype(np.float64))
            for lead in range(probs.shape[1])
        ]
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self._maps is None:
            raise RuntimeError("PerLeadCalibrator.transform called before fit")
        probs = np.asarray(probs)
        if probs.ndim < 2 or probs.shape[1] != len(self._maps):
            raise ValueError(f"expected {len(self._maps)} leads on axis 1, got shape {probs.shape}")
        out = np.empty(probs.shape, dtype=np.float32)
        for lead, iso in enumerate(self._maps):
            out[:, lead] = iso.transform(probs[:, lead])
        return np.clip(out, 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Plain lists (one knot pair per lead) so the calibrator can sit next to the checkpoint as JSON."""
        if self._maps is None:
            raise RuntimeError("PerLeadCalibrator.to_dict called before fit")
        return {"leads": [{"x": m.x_knots.tolist(), "y": m.y_knots.tolist()} for m in self._maps]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PerLeadCalibrator:
        cal = cls()
        cal._maps = []
        for knots in d["leads"]:
            iso = IsotonicCalibrator()
            iso.x_knots = np.asarray(knots["x"], dtype=np.float64)
            iso.y_knots = np.asarray(knots["y"], dtype=np.float64)
            cal._maps.append(iso)
        return cal


def frames_to_slots(p: np.ndarray, frames_per_slot: int = 3) -> np.ndarray:
    """Element-wise max over consecutive groups of frames along axis 0 (STORM-07)."""
    p = np.asarray(p)
    if frames_per_slot < 1 or p.shape[0] % frames_per_slot:
        raise ValueError(f"{p.shape[0]} frames cannot be split into slots of {frames_per_slot}")
    n_slots = p.shape[0] // frames_per_slot
    return p.reshape(n_slots, frames_per_slot, *p.shape[1:]).max(axis=1)
