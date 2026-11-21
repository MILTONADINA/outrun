"""HRRR grid-relative winds to earth-relative (east, north) (carry-forward from T4/T5 NOTES).

The smoke store keeps UGRD/VGRD exactly as HRRR ships them: components along the Lambert grid axes. The HRRR
projection is Lambert conformal with standard parallel 38.5 N and central meridian 97.5 W, so the grid axes are
turned by ``a = sin(38.5 deg) * (lon + 97.5 deg)`` from east/north (zero on the central meridian).
"""

from __future__ import annotations

import numpy as np

HRRR_STD_PARALLEL_DEG = 38.5
HRRR_CENTRAL_MERIDIAN_DEG = -97.5


def grid_rotation_angle(lon_deg: np.ndarray) -> np.ndarray:
    """Angle (radians) of true east counter-clockwise from the grid +x axis, at longitudes ``lon_deg``."""
    return np.sin(np.radians(HRRR_STD_PARALLEL_DEG)) * np.radians(
        np.asarray(lon_deg, dtype=np.float64) + 97.5
    )


def rotate_winds(
    u_grid: np.ndarray, v_grid: np.ndarray, lon_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(u, v) along the grid axes -> (east, north) components. Speed is preserved."""
    a = grid_rotation_angle(lon_deg)
    c, s = np.cos(a), np.sin(a)
    # true east is (cos a, sin a) and true north (-sin a, cos a) in grid coordinates (checked against pyproj)
    return u_grid * c + v_grid * s, -u_grid * s + v_grid * c
