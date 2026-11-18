"""Chip tensors (TORN-01) and hand-crafted radar features (TORN-03).

A chip maps each variable in ``VARIABLES`` to a float array ``(T, 2, A, R)`` (frame, tilt, azimuth, range gate);
missing gates are NaN. Tilt 0 is the lowest tilt.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

VARIABLES: tuple[str, ...] = ("DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH")
N_TILTS = 2
FILL_VALUES: dict[str, float] = {"DBZ": -20.0, "VEL": 0.0, "KDP": 0.0, "RHOHV": 0.0, "ZDR": 0.0, "WIDTH": 0.0}
N_CHANNELS = 27
N_CHANNELS_NO_RANGE = 26
N_NORMALIZED = 24
HAND_FEATURES: tuple[str, ...] = (
    "max_dbz", "max_dv", "min_rhohv_dbz30", "max_zdr", "max_kdp", "max_width", "range_km",
)  # fmt: skip


def build_chip_tensor(
    chip: Mapping[str, np.ndarray],
    range_km: np.ndarray,
    azimuth_deg: np.ndarray,
    include_range: bool = True,
) -> np.ndarray:
    """float32 ``(27|26, A, R)``: 12 last-frame, 12 last-minus-first, missing mask, range/300, azimuth in [-1, 1].

    KeyError on a missing variable, ValueError on inconsistent shapes (TORN-01). Does not modify ``chip``.
    """
    raise NotImplementedError


def fit_channel_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over (N, A, R); std 0 -> 1 (TORN-01)."""
    raise NotImplementedError


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray, n_norm: int = N_NORMALIZED) -> np.ndarray:
    """float32 copy with the first ``n_norm`` channels standardised, the rest unchanged (TORN-01)."""
    raise NotImplementedError


def max_dv(chip: Mapping[str, np.ndarray], tilt: int = 0) -> float:
    """Max |VEL[a+1, r] - VEL[a, r]| in the last frame at ``tilt``, NaN pairs skipped, 0 if none (TORN-03)."""
    raise NotImplementedError


def hand_features(chip: Mapping[str, np.ndarray], range_km_center: float) -> np.ndarray:
    """float64 ``(7,)`` in ``HAND_FEATURES`` order from the last frame, tilt 0 (TORN-03)."""
    raise NotImplementedError
