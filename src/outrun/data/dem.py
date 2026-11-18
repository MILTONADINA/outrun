"""DEM derivatives for flood statics: slope and TPI (specs/01-data.md DATA-18). ETOPO/3DEP IO is acceptance-tested."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def slope_deg(elev: np.ndarray, cell_m: float) -> np.ndarray:
    """DATA-18. degrees(arctan(hypot(dz/dx, dz/dy))) via np.gradient; float32."""
    # np.gradient returns (d/drow, d/dcol): central differences inside, one-sided at the edges.
    dz_dy, dz_dx = np.gradient(np.asarray(elev, dtype=np.float64), cell_m)
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))).astype(np.float32)


def tpi(elev: np.ndarray, radius_cells: int) -> np.ndarray:
    """DATA-18. elev minus the in-bounds finite mean of the (2r+1)^2 window; radius < 1 -> ValueError."""
    if radius_cells < 1:
        raise ValueError(f"radius_cells must be >= 1, got {radius_cells}")
    elev = np.asarray(elev, dtype=np.float64)
    valid = np.isfinite(elev)
    size = 2 * radius_cells + 1
    # uniform_filter gives window means with zeros outside the array (mode="constant"). The ratio of the mean of the
    # values (NaN set to 0) to the mean of the valid flags is the mean over in-bounds, finite cells only.
    sum_like = uniform_filter(np.where(valid, elev, 0.0), size=size, mode="constant", cval=0.0)
    count_like = uniform_filter(valid.astype(np.float64), size=size, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        window_mean = np.where(count_like > 1e-12, sum_like / count_like, np.nan)
    return (elev - window_mean).astype(np.float32)
