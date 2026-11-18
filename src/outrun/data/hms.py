"""NOAA HMS smoke polygons (specs/01-data.md DATA-07). Output feeds SMOKE-02 ``rasterize_hms``."""

from __future__ import annotations

from datetime import date
from pathlib import Path

DENSITIES: tuple[str, ...] = ("Light", "Medium", "Heavy")


def read_hms(path: Path | str, crs: str = "EPSG:5070") -> list:
    """DATA-07. Shapefile -> list[(shapely.Polygon in ``crs``, "Light"|"Medium"|"Heavy")], multipolygons exploded."""
    raise NotImplementedError


def hms_date(name: str) -> date:
    """DATA-07. The YYYYMMDD date in an HMS file name; none -> ValueError."""
    raise NotImplementedError
