"""Real smoke provider (SMOKE-08): HRRR-Smoke store + AQS + HMS -> ``SmokeInputs`` for one scenario.

Needs ``OUTRUN_PROC`` (HRRR store, ``PROC/smoke`` tables) and ``OUTRUN_RAW`` (HMS zips). Only cycles 2022-2024 exist, so
a scenario outside that range raises. The model grid is the HRRR 9 km grid cells covering the scenario window (the
store is 9 km, an approved deviation from the 3 km in the spec text); patch/feature transforms are the same functions
the training data used (``build.transform_fields``, ``tabular_features_batch``). Checkpoint and
``massden_calibration.json`` are read from ``models/smoke``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from outrun.config import get_paths
from outrun.core.grid import regrid_nearest
from outrun.hazards.smoke.adapter import SmokeInputs, idw
from outrun.hazards.smoke.baselines import MassdenCalibrated
from outrun.hazards.smoke.build import (
    CYCLE0,
    N_CYCLES,
    PATCH,
    ZARR_ARRAYS,
    cycle_rows,
    open_store,
    smoke_dir,
    transform_fields,
)
from outrun.hazards.smoke.dataset import SmokeFeatureConfig, tabular_features_batch


def real_provider():
    """Zero-argument factory used lazily by ``SmokeAdapter``: loads the store handles once, returns ``provider(scn)``."""
    from pyproj import Transformer

    root = open_store()
    rows = cycle_rows(root)
    lat, lon = np.asarray(root["latitude"][:]), np.asarray(root["longitude"][:])
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    gx, gy = tr.transform(lon.astype(np.float64), lat.astype(np.float64))
    sites = pd.read_parquet(smoke_dir() / "sites.parquet")
    aqs = pd.read_parquet(smoke_dir() / "aqs.parquet")
    cal_path = get_paths().models / "smoke" / "massden_calibration.json"
    a_b = json.loads(cal_path.read_text()) if cal_path.exists() else {"a": 0.0, "b": 0.0}
    cal = MassdenCalibrated()
    cal.a, cal.b = float(a_b["a"]), float(a_b["b"])
    cfg = SmokeFeatureConfig()
    wide = aqs.pivot_table(index="time", columns="site_id", values="pm25", aggfunc="mean").reindex(
        columns=sites["site_id"]
    )
    sxy = sites[["x_m", "y_m"]].to_numpy()

    def provider(scn) -> SmokeInputs:
        t0 = scn.t0.astimezone(UTC)
        c = int((t0 - CYCLE0) // timedelta(hours=6))
        if not 0 <= c < N_CYCLES:
            raise RuntimeError(f"no HRRR-Smoke cycle for {t0.isoformat()} (store covers 2022-2024)")
        t = CYCLE0 + timedelta(hours=6 * c)
        win = scn.grid_window()
        xmin, ymin, xmax, ymax = win.bounds
        m = PATCH // 2
        inside = (gx >= xmin - 9000) & (gx <= xmax + 9000) & (gy >= ymin - 9000) & (gy <= ymax + 9000)
        ii, jj = np.nonzero(inside)
        i0, i1 = max(ii.min(), m), min(ii.max(), lat.shape[0] - 1 - m)
        j0, j1 = max(jj.min(), m), min(jj.max(), lat.shape[1] - 1 - m)
        gi, gj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        shape = gi.shape
        fields = transform_fields([np.asarray(root[n][rows[c]], dtype=np.float32) for n in ZARR_ARRAYS], lon)
        pi = gi.ravel()[:, None, None] + np.arange(-m, m + 1)[None, :, None]
        pj = gj.ravel()[:, None, None] + np.arange(-m, m + 1)[None, None, :]
        patches = np.stack([f[pi, pj] for f in fields], axis=1).astype(np.float32)  # (cells, 8, 9, 9)
        cx, cy = gx[gi, gj], gy[gi, gj]
        # monitors: last 6 h of PM2.5 and the 30-day background, IDW'd to the cell centres
        hist_t = [t - timedelta(hours=h) for h in range(5, -1, -1)]
        hist = np.stack([_idw_cells(sxy, wide, ht, cx, cy) for ht in hist_t], axis=1)
        bg_sites = wide.loc[(wide.index >= t - timedelta(days=30)) & (wide.index < t)].median()
        background = _idw_vals(sxy, bg_sites.to_numpy(), cx, cy)
        hms = _hms_at(t, cx, cy, lon[gi, gj], lat[gi, gj])
        tabs = tabular_features_batch(hist, hms, pd.DatetimeIndex([t] * hist.shape[0]), cfg)
        hrrr_pm = cal.predict(np.expm1(patches[:, 1, m, m]), background).reshape(shape).astype(np.float32)
        # truth at t0 + 6 h from monitors inside the window
        tt = t + timedelta(hours=6)
        in_win = (sxy[:, 0] >= xmin) & (sxy[:, 0] <= xmax) & (sxy[:, 1] >= ymin) & (sxy[:, 1] <= ymax)
        vals = wide.reindex([tt]).to_numpy()[0] if tt in wide.index else np.full(len(sxy), np.nan)
        obs = None
        if np.any(in_win & ~np.isnan(vals)):
            obs = idw(sxy, vals, win.x_centers(), win.y_centers())
        lon_c, lat_c = lon[gi, gj], lat[gi, gj]

        def regrid(a: np.ndarray) -> np.ndarray:
            return np.stack(
                [regrid_nearest(a[k], lon_c, lat_c, win, 9000.0 * 1.5, fill=0.0) for k in range(len(a))]
            )

        held = t.year >= 2024 or (t.date().isoformat() in _quarantine_dates())
        return SmokeInputs(
            patches, tabs, shape, hrrr_pm, obs, win.x_centers(), win.y_centers(), regrid, bool(held)
        )

    return provider


def _idw_vals(sxy: np.ndarray, vals: np.ndarray, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    return _idw_points(sxy, vals, cx.ravel(), cy.ravel())


def _idw_cells(
    sxy: np.ndarray, wide: pd.DataFrame, t: datetime, cx: np.ndarray, cy: np.ndarray
) -> np.ndarray:
    vals = wide.reindex([t]).to_numpy()[0]
    return (
        _idw_points(sxy, vals, cx.ravel(), cy.ravel())
        if not np.all(np.isnan(vals))
        else np.full(cx.size, np.nan)
    )


def _idw_points(sxy: np.ndarray, vals: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """IDW to scattered points (same weights as ``adapter.idw``)."""
    ok = ~np.isnan(vals)
    if not ok.any():
        return np.full(len(px), np.nan)
    d = np.hypot(px[:, None] - sxy[ok, 0][None], py[:, None] - sxy[ok, 1][None])
    w = 1.0 / np.maximum(d, 1.0) ** 2
    return ((w * vals[ok][None]).sum(1) / w.sum(1)).astype(np.float32)


def _quarantine_dates() -> set[str]:
    from outrun.eval.splits import load_quarantine

    return {d.isoformat() for d in load_quarantine(get_paths().splits / "quarantine.json")}


def _hms_at(t: datetime, cx: np.ndarray, cy: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Causal HMS density at cell centres: polygons whose END is in (t - 24 h, t], from the files of days D-1, D."""
    import geopandas as gpd
    import shapely

    from outrun.hazards.smoke.build import _parse_hms_time

    raw = get_paths().raw / "hms_smoke_polygons"
    pts = shapely.points(lon.ravel(), lat.ravel())
    out = np.zeros(lon.size, dtype=np.float32)
    for day in (t - timedelta(days=1), t):
        z = raw / f"{day:%Y}" / f"{day:%m}" / f"hms_smoke{day:%Y%m%d}.zip"
        if not z.exists():
            continue
        g = gpd.read_file(f"zip://{z}").to_crs("EPSG:4326")
        end = np.array([_parse_hms_time(e).timestamp() for e in g["End"]])
        dens = (
            g["Density"]
            .astype(str)
            .str.strip()
            .str.title()
            .map({"Light": 1, "Medium": 2, "Heavy": 3})
            .to_numpy()
        )
        for k in np.flatnonzero((end > t.timestamp() - 86400) & (end <= t.timestamp())):
            hit = shapely.intersects(g.geometry.iloc[k], pts)
            out[hit] = np.maximum(out[hit], dens[k])
    return out
