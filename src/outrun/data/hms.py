"""NOAA HMS smoke polygons (specs/01-data.md DATA-07). Output feeds SMOKE-02 ``rasterize_hms``."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

DENSITIES: tuple[str, ...] = ("Light", "Medium", "Heavy")


def read_hms(path: Path | str, crs: str = "EPSG:5070") -> list:
    """DATA-07. Shapefile -> list[(shapely.Polygon in ``crs``, "Light"|"Medium"|"Heavy")], multipolygons exploded."""
    import geopandas as gpd
    from shapely.geometry import Polygon

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")  # HMS polygons are lon/lat
    gdf = gdf.to_crs(crs).explode(index_parts=False)

    by_lower = {d.lower(): d for d in DENSITIES}
    out = []
    for geom, density in zip(gdf.geometry, gdf["Density"], strict=True):
        key = str(density).strip().lower()
        if key not in by_lower:
            raise ValueError(f"unknown HMS density {density!r}")
        if isinstance(geom, Polygon) and not geom.is_empty:
            out.append((geom, by_lower[key]))
    return out


def hms_date(name: str) -> date:
    """DATA-07. The YYYYMMDD date in an HMS file name; none -> ValueError."""
    match = re.search(r"(?<!\d)(\d{8})(?!\d)", Path(name).name)
    if match is None:
        raise ValueError(f"no YYYYMMDD date in HMS file name {name!r}")
    return datetime.strptime(match.group(1), "%Y%m%d").date()
