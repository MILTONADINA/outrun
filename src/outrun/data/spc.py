"""SPC tornado tracks: CSV parsing, swath polygons, rasterization (specs/01-data.md DATA-07)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

YARD_M = 0.9144
MILE_M = 1609.344


def read_spc_tornadoes(path: Path | str) -> pd.DataFrame:
    """DATA-07. SPC actual_tornadoes CSV -> om, begin (UTC), st, mag, slat, slon, elat, elon, len_m, wid_m."""
    raise NotImplementedError


def swath_polygons(df: pd.DataFrame, crs: str = "EPSG:5070", min_width_m: float = 50.0):
    """DATA-07. GeoDataFrame: start->end segment projected to ``crs`` buffered by max(wid_m, min)/2 (round caps)."""
    raise NotImplementedError


def rasterize_mask(geoms: Sequence, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """DATA-07. (len(y), len(x)) bool: cell centre inside or on the boundary of any geometry."""
    raise NotImplementedError
