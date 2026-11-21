"""Swath projection of a detection (TORN-08) and the storm-motion fallback order (TORN-09).

Motion is ``(u east, v north)`` in m/s; grid coordinates are GRID metres.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def project_swath(
    p: float,
    x0: float,
    y0: float,
    motion: tuple[float, float],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    n_slots: int = 12,
    slot_minutes: float = 15,
    horizon_min: float = 30,
    sigma_along_km: float = 2.0,
    sigma_across0_km: float = 4.0,
    sigma_across_end_km: float = 8.0,
    step_min: float = 1.0,
) -> np.ndarray:
    """float32 ``(n_slots, len(grid_y), len(grid_x))``: max of advected anisotropic Gaussian kernels per slot.

    ValueError if ``p`` is outside [0, 1] (TORN-08).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    u, v = motion
    speed = math.hypot(u, v)
    ex, ey = (
        (u / speed, v / speed) if speed > 0 else (1.0, 0.0)
    )  # unit vector along the track (east if still)
    gx = np.asarray(grid_x, dtype=np.float64)[None, :]
    gy = np.asarray(grid_y, dtype=np.float64)[:, None]
    out = np.zeros((n_slots, gy.shape[0], gx.shape[1]), dtype=np.float32)
    n_steps = int(math.ceil(horizon_min / step_min - 1e-9))
    for k in range(n_steps):
        tau = k * step_min
        if tau >= horizon_min:
            break
        slot = int(tau // slot_minutes)
        if slot >= n_slots:
            continue
        dx = gx - (x0 + u * 60.0 * tau)
        dy = gy - (y0 + v * 60.0 * tau)
        along = (dx * ex + dy * ey) / 1000.0
        across = (-dx * ey + dy * ex) / 1000.0
        sigma_c = sigma_across0_km + (sigma_across_end_km - sigma_across0_km) * tau / horizon_min
        kernel = p * np.exp(-0.5 * ((along / sigma_along_km) ** 2 + (across / sigma_c) ** 2))
        np.maximum(out[slot], kernel.astype(np.float32), out=out[slot])
    return out


def combine_swaths(swaths: Sequence[np.ndarray]) -> np.ndarray:
    """Element-wise max of a non-empty sequence of equal-shape arrays (TORN-08)."""
    if len(swaths) == 0:
        raise ValueError("combine_swaths needs at least one array")
    out = np.array(swaths[0], copy=True)
    for s in swaths[1:]:
        np.maximum(out, s, out=out)
    return out


def resolve_motion(
    storm_motion: tuple[float, float] | None, hrrr_wind: tuple[float, float] | None
) -> tuple[float, float, str]:
    """(u, v, source): storm motion -> HRRR wind at 75 % rotated 30 deg clockwise -> 15 m/s from 240 deg (TORN-09)."""

    def ok(w: tuple[float, float] | None) -> bool:
        return w is not None and math.isfinite(w[0]) and math.isfinite(w[1])

    if storm_motion is not None and ok(storm_motion):
        return float(storm_motion[0]), float(storm_motion[1]), "storm"
    if hrrr_wind is not None and ok(hrrr_wind):
        u, v = hrrr_wind
        c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
        return 0.75 * (u * c + v * s), 0.75 * (-u * s + v * c), "hrrr"
    toward = math.radians(60.0)  # wind from 240 degrees blows toward 60 degrees (compass)
    return 15.0 * math.sin(toward), 15.0 * math.cos(toward), "default"
