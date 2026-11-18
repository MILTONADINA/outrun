"""SPC tornado tracks: CSV parsing, swath polygons, rasterization (specs/01-data.md DATA-07)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

YARD_M = 0.9144
MILE_M = 1609.344
# SPC ``tz`` code -> hours to add to the local time to get UTC.
_TZ_TO_UTC_HOURS: dict[int, int] = {3: 6, 9: 0}


def read_spc_tornadoes(path: Path | str) -> pd.DataFrame:
    """DATA-07. SPC actual_tornadoes CSV -> om, begin (UTC), st, mag, slat, slon, elat, elon, len_m, wid_m."""
    df = pd.read_csv(path, low_memory=False)
    if "sg" in df.columns:
        df = df[df["sg"] == 1]  # 1 = whole track; 2/-9 are state segments of the same tornado
    df = df[df["slat"] != 0].reset_index(drop=True)  # no start point recorded

    bad_tz = sorted(set(df["tz"]) - set(_TZ_TO_UTC_HOURS))
    if bad_tz:
        raise ValueError(f"unexpected SPC tz codes {bad_tz} (3 = CST, 9 = GMT)")
    local = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
    begin = (local + pd.to_timedelta(df["tz"].map(_TZ_TO_UTC_HOURS), unit="h")).dt.tz_localize("UTC")

    no_end = (df["elat"] == 0) & (df["elon"] == 0)  # end point 0,0 means "same as start"
    return pd.DataFrame(
        {
            "om": df["om"],
            "begin": begin,
            "st": df["st"],
            "mag": df["mag"],
            "slat": df["slat"],
            "slon": df["slon"],
            "elat": df["elat"].where(~no_end, df["slat"]),
            "elon": df["elon"].where(~no_end, df["slon"]),
            "len_m": df["len"] * MILE_M,
            "wid_m": df["wid"] * YARD_M,
        }
    )


def swath_polygons(df: pd.DataFrame, crs: str = "EPSG:5070", min_width_m: float = 50.0):
    """DATA-07. GeoDataFrame: start->end segment projected to ``crs`` buffered by max(wid_m, min)/2 (round caps)."""
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import LineString, Point

    to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x0, y0 = to_crs.transform(df["slon"].to_numpy(), df["slat"].to_numpy())
    x1, y1 = to_crs.transform(df["elon"].to_numpy(), df["elat"].to_numpy())
    radius = np.maximum(df["wid_m"].to_numpy(dtype=float), min_width_m) / 2.0

    geoms = []
    for xa, ya, xb, yb, r in zip(x0, y0, x1, y1, radius, strict=True):
        # buffer() gives round caps; a zero-length track is a point, whose buffer is a disc.
        track = Point(xa, ya) if (xa, ya) == (xb, yb) else LineString([(xa, ya), (xb, yb)])
        geoms.append(track.buffer(r))
    return gpd.GeoDataFrame(df.copy(), geometry=geoms, crs=crs)


def rasterize_mask(geoms: Sequence, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """DATA-07. (len(y), len(x)) bool: cell centre inside or on the boundary of any geometry."""
    import shapely

    xx, yy = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float))  # (len(y), len(x))
    mask = np.zeros(xx.shape, dtype=bool)
    for geom in geoms:
        # intersects_xy is True inside and on the boundary ("covers"), unlike contains_xy.
        mask |= shapely.intersects_xy(geom, xx, yy)
    return mask
