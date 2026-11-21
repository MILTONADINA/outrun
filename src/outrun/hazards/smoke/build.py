"""Smoke data prep (T18): AQS monitors, HRRR-Smoke site patches, HMS density, sample table.

``OUTRUN_PROC=... OUTRUN_RAW=... .venv/bin/python -m outrun.hazards.smoke.build [stage ...]``; stages run in this order:

``aqs``      EPA AQS hourly PM2.5 2022-2024 -> ``PROC/smoke/aqs.parquet`` (site_id, time, pm25) and ``sites.parquet``
             (site_id, lat, lon, x_m, y_m, gi, gj = nearest HRRR 9 km cell). CONUS only (HRRR crop with a 4-cell margin
             for the 9x9 patch; AK, HI, PR, Canada and Mexico codes are dropped). Gate counts -> ``gate_aqs.json``.
``patches``  one 8 x 9 x 9 float16 patch per (cycle, site) -> ``patches.npy`` (cycle, site, 8, 9, 9) plus
             ``patch_ok.npy``. Channel order is ``PATCH_CHANNELS``; MASSDEN is log1p(ug/m3), winds are rotated
             from grid-relative to east/north. Gate counts (non-finite, implausible) -> ``gate_hrrr.json``.
``samples``  SMOKE-01 sample table with the 30-day background median -> ``samples.parquet``. Rows whose patch is not
             finite are dropped here, so the sample universe is final before ``freeze_smoke``.
``hms``      HMS smoke density at every site for every cycle (causal, see ``build_hms``) -> ``hms.npy``.

Cycle index c is the HRRR cycle ``2022-01-01 00Z + 6 h * c`` (4,384 cycles, plus one extra HMS row for t + 6 h).
Everything is a deterministic function of RAW + the HRRR store; nothing here reads a split.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from outrun.config import get_paths
from outrun.hazards.smoke.dataset import build_station_samples, hourly_matrix
from outrun.hazards.smoke.winds import rotate_winds

N_CYCLES = 4384  # 1,096 days x 4 cycles, 2022-01-01 00Z .. 2024-12-31 18Z
CYCLE0 = datetime(2022, 1, 1, tzinfo=UTC)
PATCH = 9
PATCH_CHANNELS: tuple[str, ...] = (
    "MASSDEN_f01", "MASSDEN_f06", "HPBL_f01", "HPBL_f06", "UGRD_f01", "UGRD_f06", "VGRD_f01", "VGRD_f06",
)  # fmt: skip
ZARR_ARRAYS: tuple[str, ...] = (
    "MASSDEN_8m_f01", "MASSDEN_8m_f06", "HPBL_surface_f01", "HPBL_surface_f06",
    "UGRD_10m_f01", "UGRD_10m_f06", "VGRD_10m_f01", "VGRD_10m_f06",
)  # fmt: skip
EXCLUDE_STATES = {"02", "15", "72", "78", "80", "CC"}  # AK, HI, PR, VI, Mexico, Canada
WINDOW_START = datetime(2021, 12, 31, 18, tzinfo=UTC)
WINDOW_END = datetime(2025, 1, 1, 6, tzinfo=UTC)
BACKGROUND_DAYS = 30


def smoke_dir() -> Path:
    d = get_paths().proc / "smoke"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cycle_time(c: int) -> datetime:
    return CYCLE0 + timedelta(hours=6 * c)


def all_cycles(n: int = N_CYCLES) -> list[datetime]:
    return [cycle_time(c) for c in range(n)]


def open_store():
    import zarr

    return zarr.open_group(get_paths().proc / "hrrr" / "smoke" / "data.zarr", mode="r")


def cycle_rows(root) -> np.ndarray:
    """Zarr row of every cycle index 0..N_CYCLES-1 (rows are in completion order, so go through the time array)."""
    t = np.asarray(root["time"][:])
    row = {int(v): i for i, v in enumerate(t)}
    want = [int(cycle_time(c).timestamp()) for c in range(N_CYCLES)]
    missing = [c for c, w in enumerate(want) if w not in row]
    if missing:
        raise RuntimeError(
            f"{len(missing)} cycles missing from the smoke store, first {cycle_time(missing[0])}"
        )
    return np.array([row[w] for w in want], dtype=np.int64)


def _grid_xy(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x, y = tr.transform(lon.ravel().astype(np.float64), lat.ravel().astype(np.float64))
    return np.stack([x, y], axis=1)


# --------------------------------------------------------------------------------------------------- stage aqs
def aqs_raw_gate(path: Path) -> dict:
    """Raw-value gate on one yearly AQS CSV (before the reader clips at 0)."""
    v = pd.read_csv(path, usecols=["Sample Measurement"])["Sample Measurement"].astype(float)
    return {
        "rows": int(len(v)),
        "nan": int(v.isna().sum()),
        "negative": int((v < 0).sum()),
        "below_minus_10": int((v < -10).sum()),
        "min": float(v.min()),
        "max": float(v.max()),
        "above_500": int((v > 500).sum()),
        "above_1000": int((v > 1000).sum()),
        "sentinel_999": int(v.isin([-999.0, -9999.0, 999.0, 9999.0]).sum()),
    }


def build_aqs(raw: Path | None = None, years: tuple[int, ...] = (2022, 2023, 2024)) -> dict:
    from scipy.spatial import cKDTree

    from outrun.data.aqs import read_aqs_hourly

    raw = raw or get_paths().raw
    out = smoke_dir()
    gate: dict = {"years": {}}
    frames = []
    for y in years:
        path = raw / "aqs_pm25_hourly" / f"hourly_88101_{y}.zip"
        gate["years"][str(y)] = aqs_raw_gate(path)
        print("aqs", y, gate["years"][str(y)], flush=True)
        frames.append(read_aqs_hourly(path))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["time"] >= WINDOW_START) & (df["time"] <= WINDOW_END)]
    df = df.groupby(["site_id", "time"], as_index=False).agg(
        pm25=("pm25", "mean"), lat=("lat", "median"), lon=("lon", "median")
    )
    gate["rows_after_merge_window"] = int(len(df))
    gate["sites_all"] = int(df["site_id"].nunique())

    sites = df.groupby("site_id", as_index=False).agg(lat=("lat", "median"), lon=("lon", "median"))
    sites = sites[~sites["site_id"].str[:2].isin(EXCLUDE_STATES)].reset_index(drop=True)
    gate["sites_after_state_filter"] = int(len(sites))
    root = open_store()
    lat, lon = np.asarray(root["latitude"][:]), np.asarray(root["longitude"][:])
    h, w = lat.shape
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    sites["x_m"], sites["y_m"] = tr.transform(sites["lon"].to_numpy(), sites["lat"].to_numpy())
    dist, flat = cKDTree(_grid_xy(lat, lon)).query(sites[["x_m", "y_m"]].to_numpy())
    sites["gi"], sites["gj"] = np.divmod(flat, w)
    m = PATCH // 2
    inside = (
        (dist <= 12_000)
        & (sites["gi"] >= m)
        & (sites["gi"] < h - m)
        & (sites["gj"] >= m)
        & (sites["gj"] < w - m)
    )
    gate["sites_dropped_outside_crop_or_margin"] = int((~inside).sum())
    sites = sites[inside].reset_index(drop=True)
    gate["sites_kept"] = int(len(sites))
    df = df[df["site_id"].isin(set(sites["site_id"]))][["site_id", "time", "pm25"]].reset_index(drop=True)
    gate["rows_kept"] = int(len(df))
    gate["pm25_clipped_at_0_rows"] = int((df["pm25"] == 0.0).sum())
    df.to_parquet(out / "aqs.parquet")
    sites.to_parquet(out / "sites.parquet")
    (out / "gate_aqs.json").write_text(json.dumps(gate, indent=2))
    print("aqs done:", {k: v for k, v in gate.items() if k != "years"}, flush=True)
    return gate


# ------------------------------------------------------------------------------------------------- stage patches
def transform_fields(fields: list[np.ndarray], lon: np.ndarray) -> list[np.ndarray]:
    """Raw store fields (ZARR_ARRAYS order, float32 (H, W)) -> model channels: log1p MASSDEN, earth-relative winds.

    The one place the patch transform lives; ``build_patches`` and the real provider both call it."""
    f = list(fields)
    for lead in (0, 1):  # f01, f06
        f[4 + lead], f[6 + lead] = rotate_winds(f[4 + lead], f[6 + lead], lon)
    for k in (0, 1):
        f[k] = np.log1p(np.maximum(f[k], 0.0))
    return f


def build_patches() -> dict:
    out = smoke_dir()
    sites = pd.read_parquet(out / "sites.parquet")
    root = open_store()
    rows = cycle_rows(root)
    lon = np.asarray(root["longitude"][:], dtype=np.float32)
    h, w = lon.shape
    gi, gj = sites["gi"].to_numpy(), sites["gj"].to_numpy()
    m = PATCH // 2
    ii = (gi[:, None, None] + np.arange(-m, m + 1)[None, :, None]) * np.ones((1, 1, PATCH), dtype=int)
    jj = (gj[:, None, None] + np.arange(-m, m + 1)[None, None, :]) * np.ones((1, PATCH, 1), dtype=int)
    arrays = [np.asarray(root[n][:]) for n in ZARR_ARRAYS]  # 8 x (4383+1, H, W) float16, ~15 GB in RAM
    n_sites = len(sites)
    patches = np.lib.format.open_memmap(
        out / "patches.npy",
        mode="w+",
        dtype=np.float16,
        shape=(N_CYCLES, n_sites, len(PATCH_CHANNELS), PATCH, PATCH),
    )
    ok = np.zeros((N_CYCLES, n_sites), dtype=bool)
    gate = {"cycles": N_CYCLES, "sites": n_sites, "non_finite_cells_in_grid": 0, "cycles_with_non_finite_grid": 0,
            "patches_non_finite": 0, "massden_max": 0.0, "winds_max_abs": 0.0, "hpbl_max": 0.0,
            "cycles_all_zero_field": []}  # fmt: skip
    for c in range(N_CYCLES):
        r = rows[c]
        fields = [a[r].astype(np.float32) for a in arrays]
        bad = sum(int((~np.isfinite(f)).sum()) for f in fields)
        if bad:
            gate["non_finite_cells_in_grid"] += bad
            gate["cycles_with_non_finite_grid"] += 1
        for k, f in enumerate(fields):
            if not np.any(f):
                gate["cycles_all_zero_field"].append([c, PATCH_CHANNELS[k]])
        fields = transform_fields(fields, lon)
        gate["massden_max"] = max(gate["massden_max"], float(np.expm1(np.nanmax(fields[0:2]))))
        gate["hpbl_max"] = max(gate["hpbl_max"], float(np.nanmax(fields[2:4])))
        gate["winds_max_abs"] = max(gate["winds_max_abs"], float(np.nanmax(np.abs(fields[4:8]))))
        p = np.stack([f[ii, jj] for f in fields], axis=1)  # (S, 8, 9, 9)
        finite = np.isfinite(p).all(axis=(1, 2, 3))
        gate["patches_non_finite"] += int((~finite).sum())
        ok[c] = finite
        patches[c] = np.where(np.isfinite(p), p, 0.0).astype(np.float16)
        if c % 500 == 0:
            print("patches cycle", c, flush=True)
    patches.flush()
    np.save(out / "patch_ok.npy", ok)
    (out / "gate_hrrr.json").write_text(json.dumps(gate, indent=2))
    print("patches done:", gate, flush=True)
    return gate


# ------------------------------------------------------------------------------------------------- stage samples
def rolling_background(
    mat: np.ndarray, lo_hour: int, issue_hours: np.ndarray, days: int = BACKGROUND_DAYS
) -> np.ndarray:
    """30-day median PM2.5 per site over [t - 30 d, t) at every issue hour: (S, n_issue), NaN with < 24 values.

    ``mat`` is the dense hourly matrix from ``hourly_matrix`` (columns are hours from ``lo_hour``). The window is the
    ``24 * days`` hours before ``t`` (hour t itself excluded), the same as ``baselines.background_median``.
    """
    win = 24 * days
    roll = pd.DataFrame(mat.T).rolling(win, min_periods=24, closed="left").median()
    return roll.to_numpy()[issue_hours - lo_hour, :].T.astype(np.float32)


def build_samples() -> dict:
    out = smoke_dir()
    aqs = pd.read_parquet(out / "aqs.parquet")
    sites = pd.read_parquet(out / "sites.parquet")
    ok = np.load(out / "patch_ok.npy")
    cycles = all_cycles()
    smp = build_station_samples(aqs, cycles)
    n0 = len(smp)
    hist = np.stack(smp["pm_hist"].to_numpy()).astype(np.float32)
    site_idx = pd.Index(sites["site_id"]).get_indexer(smp["site_id"])
    t_ns = smp["issue_time"].dt.as_unit("ns").astype("int64").to_numpy()
    cyc = ((t_ns // 3_600_000_000_000 - int(CYCLE0.timestamp()) // 3600) // 6).astype(np.int64)
    assert (site_idx >= 0).all() and cyc.min() >= 0 and cyc.max() < N_CYCLES
    keep = ok[cyc, site_idx]
    gate = {"samples_before_patch_filter": int(n0), "dropped_nonfinite_patch": int((~keep).sum())}
    # 30-day background median (causal: window ends at the issue time)
    lo = int(CYCLE0.timestamp()) // 3600 - 24 * BACKGROUND_DAYS
    hi_h = int(CYCLE0.timestamp()) // 3600 + 6 * (N_CYCLES - 1)
    s_ids, mat = hourly_matrix(aqs, lo, hi_h)
    assert list(s_ids) == list(sites["site_id"])
    issue_hours = np.array([int(t.timestamp()) // 3600 for t in cycles])
    bg = rolling_background(mat, lo, issue_hours)  # (S, n_cycles)
    table = pd.DataFrame(
        {
            "site_id": smp["site_id"].to_numpy(),
            "issue_time": smp["issue_time"].to_numpy(),
            "site_idx": site_idx.astype(np.int32),
            "cycle_idx": cyc.astype(np.int32),
            **{f"h{k}": hist[:, k] for k in range(hist.shape[1])},
            "target": smp["target"].to_numpy(),
            "pm_target": smp["pm_target"].to_numpy(),
            "background": bg[site_idx, cyc],
        }
    )
    table = table[keep].reset_index(drop=True)
    gate["samples"] = int(len(table))
    gate["sites_with_samples"] = int(table["site_idx"].nunique())
    gate["per_year"] = {
        str(k): int(v) for k, v in table["issue_time"].dt.year.value_counts().sort_index().items()
    }
    gate["background_nan"] = int(table["background"].isna().sum())
    table.to_parquet(out / "samples.parquet")
    (out / "gate_samples.json").write_text(json.dumps(gate, indent=2))
    print("samples done:", gate, flush=True)
    return gate


# ----------------------------------------------------------------------------------------------------- stage hms
def _parse_hms_time(s: str) -> datetime:
    """HMS 'YYYYDDD HHMM' (UTC) -> datetime."""
    day, hhmm = str(s).split()
    return datetime.strptime(day, "%Y%j").replace(hour=int(hhmm[:2]), minute=int(hhmm[2:]), tzinfo=UTC)


def read_hms_polygons(raw: Path | None = None) -> pd.DataFrame:
    """All daily HMS polygons 2022-2024: columns end (tz-aware UTC), density (1..3), geometry (lon/lat)."""
    import geopandas as gpd

    raw = raw or get_paths().raw
    parts = []
    for z in sorted((raw / "hms_smoke_polygons").glob("*/*/hms_smoke*.zip")):
        g = gpd.read_file(f"zip://{z}")
        if g.empty:
            continue
        g = g.set_crs("EPSG:4326") if g.crs is None else g.to_crs("EPSG:4326")
        dens = g["Density"].astype(str).str.strip().str.title().map({"Light": 1, "Medium": 2, "Heavy": 3})
        if dens.isna().any():
            raise ValueError(f"{z.name}: unknown HMS density {sorted(set(g['Density']))}")
        parts.append(
            pd.DataFrame(
                {
                    "end": [_parse_hms_time(e) for e in g["End"]],
                    "density": dens.to_numpy(),
                    "geometry": g.geometry.to_numpy(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def build_hms(raw: Path | None = None) -> dict:
    """HMS density 0-3 at each site for each cycle c = 0..N_CYCLES (one extra row: the target time of the last cycle).

    Causal: the value for cycle time t is the maximum density of the polygons whose analysis END lies in
    (t - 24 h, t], so nothing drawn after t is used. Row c + 1 is the HMS state at the +6 h target time of cycle c;
    the episode flag of a sample is computed from that row (evaluation only; the model gets row c).
    """
    import shapely
    from shapely import STRtree

    out = smoke_dir()
    sites = pd.read_parquet(out / "sites.parquet")
    poly = read_hms_polygons(raw).sort_values("end", kind="stable").reset_index(drop=True)
    pts = shapely.points(sites["lon"].to_numpy(), sites["lat"].to_numpy())
    pi, si = STRtree(pts).query(
        poly["geometry"].to_numpy(), predicate="intersects"
    )  # polygon i covers site s
    order = np.argsort(pi, kind="stable")
    pi, si = pi[order], si[order]
    starts = np.searchsorted(pi, np.arange(len(poly) + 1))
    ends = np.array([e.timestamp() for e in poly["end"]])
    dens = poly["density"].to_numpy().astype(np.uint8)
    hms = np.zeros((N_CYCLES + 1, len(sites)), dtype=np.uint8)
    for c in range(N_CYCLES + 1):
        t = cycle_time(c).timestamp()
        lo, hi = np.searchsorted(ends, t - 86400, side="right"), np.searchsorted(ends, t, side="right")
        for p in range(lo, hi):
            cover = si[starts[p] : starts[p + 1]]
            hms[c, cover] = np.maximum(hms[c, cover], dens[p])
    np.save(out / "hms.npy", hms)
    gate = {"polygons": int(len(poly)), "share_cycle_site_hms_ge1": float((hms >= 1).mean()), "share_ge2": float((hms >= 2).mean()),
            "share_ge3": float((hms >= 3).mean())}  # fmt: skip
    (out / "gate_hms.json").write_text(json.dumps(gate, indent=2))
    print("hms done:", gate, flush=True)
    return gate


STAGES = {"aqs": build_aqs, "patches": build_patches, "samples": build_samples, "hms": build_hms}


def main(argv: list[str] | None = None) -> None:
    stages = list(argv if argv is not None else sys.argv[1:]) or list(STAGES)
    for s in stages:
        STAGES[s]()


if __name__ == "__main__":
    main()
