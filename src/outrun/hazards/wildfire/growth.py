"""Hourly linear growth between daily fire masks and the observed layer (specs/11-wildfire.md: FIRE-09, FIRE-11)."""

from __future__ import annotations

from datetime import datetime

import numpy as np


def hourly_growth(
    day0_mask: np.ndarray,
    day1_prob: np.ndarray,
    day0_time: datetime,
    issued: datetime,
    n_slots: int = 12,
    slot_minutes: int = 15,
) -> np.ndarray:
    """float32 (n_slots, H, W): P_s = (1 - a_s) m0 + a_s p1, a_s = clip(floor((t_s - day0)/1 h)/24, 0, 1),
    t_s = issued + s * slot_minutes; m0 = day0 mask with < 0 / NaN -> 0. issued < day0_time, naive datetimes or
    shape mismatch -> ValueError (FIRE-09)."""
    raise NotImplementedError


def observed_from_mask(day1_mask: np.ndarray, n_slots: int = 12) -> np.ndarray:
    """float32 (n_slots, H, W), each slot 1 where day1_mask == 1 else 0 (FIRE-11)."""
    raise NotImplementedError
