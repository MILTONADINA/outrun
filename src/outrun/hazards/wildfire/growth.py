"""Hourly linear growth between daily fire masks and the observed layer (specs/11-wildfire.md: FIRE-09, FIRE-11)."""

from __future__ import annotations

from datetime import datetime, timedelta

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
    if day0_time.tzinfo is None or issued.tzinfo is None:
        raise ValueError("day0_time and issued must be timezone-aware (UTC)")
    if issued < day0_time:
        raise ValueError(f"issued {issued} is before the day-0 time {day0_time}")
    m0 = np.nan_to_num(np.asarray(day0_mask, dtype=np.float32), nan=0.0)
    m0 = np.where(m0 < 0, 0.0, m0).astype(np.float32)
    p1 = np.asarray(day1_prob, dtype=np.float32)
    if m0.shape != p1.shape:
        raise ValueError(f"day0_mask {m0.shape} and day1_prob {p1.shape} must have the same shape")
    out = np.empty((n_slots, *m0.shape), np.float32)
    for s in range(n_slots):
        hours = np.floor((issued + timedelta(minutes=s * slot_minutes) - day0_time) / timedelta(hours=1))
        a = float(np.clip(hours / 24.0, 0.0, 1.0))
        out[s] = (1.0 - a) * m0 + a * p1
    return out


def observed_from_mask(day1_mask: np.ndarray, n_slots: int = 12) -> np.ndarray:
    """float32 (n_slots, H, W), each slot 1 where day1_mask == 1 else 0 (FIRE-11)."""
    m = (np.asarray(day1_mask) == 1).astype(np.float32)  # NaN and -1 are not 1
    return np.repeat(m[None], n_slots, axis=0)
