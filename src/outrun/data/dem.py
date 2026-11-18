"""DEM derivatives for flood statics: slope and TPI (specs/01-data.md DATA-18). ETOPO/3DEP IO is acceptance-tested."""

from __future__ import annotations

import numpy as np


def slope_deg(elev: np.ndarray, cell_m: float) -> np.ndarray:
    """DATA-18. degrees(arctan(hypot(dz/dx, dz/dy))) via np.gradient; float32."""
    raise NotImplementedError


def tpi(elev: np.ndarray, radius_cells: int) -> np.ndarray:
    """DATA-18. elev minus the in-bounds finite mean of the (2r+1)^2 window; radius < 1 -> ValueError."""
    raise NotImplementedError
