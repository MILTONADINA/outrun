"""Swath projection of a detection (TORN-08) and the storm-motion fallback order (TORN-09).

Motion is ``(u east, v north)`` in m/s; grid coordinates are GRID metres.
"""

from __future__ import annotations

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
    raise NotImplementedError


def combine_swaths(swaths: Sequence[np.ndarray]) -> np.ndarray:
    """Element-wise max of a non-empty sequence of equal-shape arrays (TORN-08)."""
    raise NotImplementedError


def resolve_motion(
    storm_motion: tuple[float, float] | None, hrrr_wind: tuple[float, float] | None
) -> tuple[float, float, str]:
    """(u, v, source): storm motion -> HRRR wind at 75 % rotated 30 deg clockwise -> 15 m/s from 240 deg (TORN-09)."""
    raise NotImplementedError
