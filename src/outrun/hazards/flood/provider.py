"""Real flood provider (FLOOD-10): MRMS QPE + FLASH + ETOPO statics + Storm Events -> ``FloodInputs`` for one scenario.

Reads the same RAW MRMS hours (``mrms_qpe_01h_pass2``, ``mrms_flash``) and the ``PROC/flood/statics.npz`` terrain the
training patches were cut from, and applies the identical feature stack (``dataset.build_features``; accumulations
over slots T..T-h, NaN propagating). The model grid is the native MRMS 0.01 degree grid cut to the scenario bbox
plus a margin and padded to a multiple of 8 (the U-Net is fully convolutional). ``regrid`` samples the nearest MRMS
cell for every 1 km GRID cell. The issue hour is ``scn.t0`` (UTC, on the hour).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer

from outrun.config import get_paths
from outrun.hazards.flood import dataset as D
from outrun.hazards.flood import prep
from outrun.hazards.flood.adapter import FloodInputs

MARGIN_DEG = 0.15
REPORT_PAD_DEG = 0.3


def issue_hour(scn) -> object:
    t = scn.t0
    if (t.minute, t.second, t.microsecond) != (0, 0, 0):
        raise ValueError(f"flood issue time must be on the hour, got {t.isoformat()}")
    return t


def real_provider(raw=None, proc=None):
    """Zero-argument factory returning ``provider(scn) -> FloodInputs``; statics and Storm Events load lazily."""
    raw = get_paths().raw if raw is None else raw
    proc = get_paths().proc if proc is None else proc
    cache: dict = {}

    def statics():
        if "st" not in cache:
            z = np.load(proc / "flood" / "statics.npz")
            cache["st"] = {k: z[k] for k in ("elevation", "slope", "tpi")}
        return cache["st"]

    def reports():
        if "rep" not in cache:
            df = prep.load_reports(raw)
            df["begin"], df["end"] = (
                pd.to_datetime(df["begin"], utc=True),
                pd.to_datetime(df["end"], utc=True),
            )
            cache["rep"] = df
        return cache["rep"]

    def provider(scn) -> FloodInputs:
        t = issue_hour(scn)
        lon0, lat0, lon1, lat1 = scn.bbox
        top = int(np.floor((prep.LAT0 - (lat1 + MARGIN_DEG)) / prep.STEP))
        left = int(np.floor(((lon0 - MARGIN_DEG) - prep.LON0) / prep.STEP))
        bottom = int(np.ceil((prep.LAT0 - (lat0 - MARGIN_DEG)) / prep.STEP)) + 1
        right = int(np.ceil(((lon1 + MARGIN_DEG) - prep.LON0) / prep.STEP)) + 1
        h, w = -(-(bottom - top) // 8) * 8, -(-(right - left) // 8) * 8
        sl = (slice(top, top + h), slice(left, left + w))

        def crop(product: str, hour) -> np.ndarray:
            path = prep.hour_path(raw, product, hour)
            return prep.clean(prep.read_raw(path), product)[sl]

        from datetime import timedelta

        q = np.stack([crop(prep.QPE_PRODUCT, t - timedelta(hours=k)) for k in range(24)])
        flash = [crop(p, t) for p in prep.FLASH_PRODUCTS]
        acc = {n: q[:n].sum(axis=0) for n in (1, 3, 6, 24)}
        st = statics()
        features = D.build_features(
            acc[1], acc[3], acc[6], acc[24], flash[0], flash[1], flash[2],
            st["elevation"][sl], st["slope"][sl], st["tpi"][sl],
        )  # fmt: skip

        win = scn.grid_window()
        gx, gy = np.meshgrid(win.x_centers(), win.y_centers())
        to_ll = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
        lon, lat = to_ll.transform(gx, gy)
        rows = np.clip(np.rint((prep.LAT0 - lat) / prep.STEP).astype(int) - top, 0, h - 1)
        cols = np.clip(np.rint((lon - prep.LON0) / prep.STEP).astype(int) - left, 0, w - 1)

        def regrid(a: np.ndarray) -> np.ndarray:
            return np.asarray(a)[:, rows, cols]

        df = reports()
        near = df[
            (df["begin"] - timedelta(hours=1) <= t + timedelta(hours=6))
            & (df["end"] + timedelta(hours=1) >= t)
            & df["begin_lat"].between(lat0 - REPORT_PAD_DEG, lat1 + REPORT_PAD_DEG)
            & df["begin_lon"].between(lon0 - REPORT_PAD_DEG, lon1 + REPORT_PAD_DEG)
        ]
        to_xy = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
        frep = []
        for r in near.itertuples():
            x0, y0 = to_xy.transform(r.begin_lon, r.begin_lat)
            x1, y1 = to_xy.transform(r.end_lon, r.end_lat)
            frep.append(
                D.FloodReport(
                    r.begin.to_pydatetime(), r.end.to_pydatetime(), float(x0), float(y0), float(x1), float(y1)
                )
            )
        return FloodInputs(
            features=features, reports=frep, x=win.x_centers(), y=win.y_centers(), regrid=regrid,
            held_out=t.year >= 2025,  # the flood test year is 2025 (splits/flood.json)
        )  # fmt: skip

    return provider
