"""Ice zone bags from the HRRR stream crop (T15): zone masks, per-cycle zone statistics, labelled bags.

Run: ``OUTRUN_PROC=... OUTRUN_RAW=... .venv/bin/python -m outrun.hazards.ice.build [--out DIR]``.
Everything here is a deterministic function of the HRRR zarr, the NWS zone shapefile and Storm Events. It uses no
split and fits nothing; the split is frozen from the bag table (``freeze_ice_split``) before any baseline is fit.

Bag = (NWS zone, 6-h window starting 00/06/12/18 UTC). HRRR f01 cycles are 3-hourly, so the window starting at ``ws``
uses cycle ``ws + 3 h`` as time t and cycle ``ws`` as t-3 h (valid ws+4 h and ws+1 h, both inside the window).
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from outrun.config import get_paths
from outrun.hazards.ice.baselines import physics_rule
from outrun.hazards.ice.dataset import ICE_FIELDS, assign_bag_labels, label_storm_events

ARRAY = {
    f: (f if ":" in f else f + ":surface").replace(":", "_").replace(" ", "") + "_f01" for f in ICE_FIELDS
}  # "TMP:2 m" -> "TMP_2m_f01"
MIN_ZONE_CELLS = 4
MIN_ZONE_COVERAGE = 0.8  # in-grid cell area / polygon area; drops zones cut by the domain edge


def zarr_path() -> Path:
    return get_paths().proc / "hrrr" / "ice" / "data.zarr"


def load_zone_cells(zip_path: Path, lat: np.ndarray, lon: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    """NWS land zones fully inside the HRRR crop -> (zone ids, flat cell indices of each zone's mask)."""
    import geopandas as gpd
    import shapely

    from outrun.data.nwszones import STATE_FIPS_TO_POSTAL, zones_from_frame

    with zipfile.ZipFile(zip_path) as z:
        shp = next(n for n in z.namelist() if n.endswith(".shp"))
    gdf = zones_from_frame(gpd.read_file(f"zip://{zip_path}!{shp}"))
    gdf = gdf[gdf["STATE"].isin(set(STATE_FIPS_TO_POSTAL.values()))].to_crs(4326)
    south, north, west, east = lat.min(), lat.max(), lon.min(), lon.max()
    gdf = gdf.cx[west:east, south:north]
    area_m2 = gdf.to_crs(5070).area.to_numpy()
    flat_lat, flat_lon = lat.ravel().astype(np.float64), lon.ravel().astype(np.float64)
    ids, cells = [], []
    for zid, geom, area in zip(gdf["zone_id"], gdf.geometry, area_m2, strict=True):
        w, s, e, n = geom.bounds
        cand = np.flatnonzero((flat_lon >= w) & (flat_lon <= e) & (flat_lat >= s) & (flat_lat <= n))
        inside = cand[shapely.intersects_xy(geom, flat_lon[cand], flat_lat[cand])]
        if len(inside) < MIN_ZONE_CELLS or len(inside) * 9e6 / area < MIN_ZONE_COVERAGE:
            continue
        ids.append(str(zid))
        cells.append(inside.astype(np.int64))
    return ids, cells


def elevation_on_grid(etopo_tif: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """ETOPO 30 arc-second surface elevation (m) nearest-sampled at the HRRR cell centres."""
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(etopo_tif) as ds:
        win = from_bounds(lon.min() - 0.1, lat.min() - 0.1, lon.max() + 0.1, lat.max() + 0.1, ds.transform)
        win = win.round_offsets().round_lengths()
        arr = ds.read(1, window=win).astype(np.float32)
        tf = ds.window_transform(win)
    col = np.clip(((lon - tf.c) / tf.a).astype(int), 0, arr.shape[1] - 1)
    row = np.clip(((lat - tf.f) / tf.e).astype(int), 0, arr.shape[0] - 1)
    return arr[row, col]


def scan_cycles(zarr_dir: Path, cells: list[np.ndarray]) -> dict[str, np.ndarray]:
    """One pass over every HRRR cycle: per (cycle, zone) statistics. Returns arrays indexed by time-sorted cycle.

    ``mean``/``max``: (n_cycles, n_zones, 9) in-zone mean and max of the 9 ICE_FIELDS (float32, NaN-safe sums
    over finite cells only); ``rule``: (n_cycles, n_zones) physics_rule_zone; ``t2m_min``: (n_cycles, n_zones);
    ``nonfinite``: (n_cycles, n_zones) count of non-finite in-zone cell values over the 9 fields.
    Gate arrays: ``frame_nan`` (n_cycles, 9) NaN fraction of the full frame, ``frame_min/max`` (n_cycles, 9).
    """
    import zarr

    root = zarr.open_group(zarr_dir, mode="r")
    times = root["time"][:]
    order = np.argsort(times)
    flat = np.concatenate(cells)
    starts = np.cumsum([0] + [len(c) for c in cells[:-1]])
    counts = np.array([len(c) for c in cells], float)
    n, nz, nf = len(order), len(cells), len(ICE_FIELDS)
    out = {
        "time": times[order].astype(np.int64),
        "mean": np.zeros((n, nz, nf), np.float32),
        "max": np.zeros((n, nz, nf), np.float32),
        "rule": np.zeros((n, nz), np.float32),
        "t2m_min": np.zeros((n, nz), np.float32),
        "nonfinite": np.zeros((n, nz), np.int32),
        "frame_nan": np.zeros((n, nf)),
        "frame_min": np.zeros((n, nf), np.float32),
        "frame_max": np.zeros((n, nf), np.float32),
    }
    arrays = [root[ARRAY[f]] for f in ICE_FIELDS]
    for k, src in enumerate(order):
        frames = {
            f: np.asarray(a[int(src)], dtype=np.float32) for f, a in zip(ICE_FIELDS, arrays, strict=True)
        }
        for j, f in enumerate(ICE_FIELDS):
            fr = frames[f]
            fin = np.isfinite(fr)
            out["frame_nan"][k, j] = 1.0 - fin.mean()
            out["frame_min"][k, j] = np.nanmin(fr) if fin.any() else np.nan
            out["frame_max"][k, j] = np.nanmax(fr) if fin.any() else np.nan
            vals = fr.ravel()[flat]
            ok = np.isfinite(vals)
            out["nonfinite"][k] += np.add.reduceat((~ok).astype(np.int32), starts)
            clean = np.where(ok, vals, 0.0)
            cnt = np.maximum(np.add.reduceat(ok.astype(np.float64), starts), 1.0)
            out["mean"][k, :, j] = np.add.reduceat(clean.astype(np.float64), starts) / cnt
            out["max"][k, :, j] = np.maximum.reduceat(np.where(ok, vals, -np.inf), starts)
            if f == "TMP:2 m":
                out["t2m_min"][k] = np.minimum.reduceat(np.where(ok, vals, np.inf), starts)
        # physics rule on the full frame (finite cells only), then any-in-zone per zone
        hit = physics_rule({f: np.nan_to_num(v, nan=999.0) for f, v in frames.items()})
        out["rule"][k] = np.maximum.reduceat(hit.ravel()[flat].astype(np.int8), starts)
        if k % 500 == 0:
            print(f"  scanned {k}/{n} cycles", flush=True)
    del counts
    return out


def make_windows(
    scan: dict[str, np.ndarray], zone_ids: list[str]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """(zone, window) table for windows whose cycles ws and ws+3 h both exist and are fully finite in the zone.

    Returns the table (zone_id, window_start, t2m_min_k, rule) and the cycle indices ``i_t`` (ws+3) / ``i_tm3`` (ws)
    plus the zone index of each row, used to rebuild feature rows.
    """
    t = scan["time"]
    pos = {int(s): i for i, s in enumerate(t)}
    ws_list = [int(s) for s in t if (int(s) // 3600) % 6 == 0 and int(s) + 3 * 3600 in pos]
    i_tm3 = np.array([pos[s] for s in ws_list])
    i_t = np.array([pos[s + 3 * 3600] for s in ws_list])
    nz = len(zone_ids)
    finite = (scan["nonfinite"][i_t] == 0) & (scan["nonfinite"][i_tm3] == 0)
    t2m = np.minimum(scan["t2m_min"][i_t], scan["t2m_min"][i_tm3])
    rule = np.maximum(scan["rule"][i_t], scan["rule"][i_tm3])
    ws_dt = pd.to_datetime(np.array(ws_list), unit="s", utc=True)
    zi, wi = np.meshgrid(np.arange(nz), np.arange(len(ws_list)))  # (W, Z)
    keep = finite.ravel()
    table = pd.DataFrame(
        {
            "zone_id": np.array(zone_ids)[zi.ravel()][keep],
            "window_start": ws_dt[wi.ravel()][keep],
            "t2m_min_k": t2m.ravel()[keep],
            "rule": rule.ravel()[keep],
        }
    )
    idx = {"i_t": i_t[wi.ravel()][keep], "i_tm3": i_tm3[wi.ravel()][keep], "zone_index": zi.ravel()[keep]}
    return table, idx


def read_events(raw: Path, first: datetime, last: datetime) -> pd.DataFrame:
    from outrun.data.stormevents import read_storm_events

    files = sorted((raw / "storm_events").glob("StormEvents_details-ftp_v1.0_d202[0-5]_*.csv.gz"))
    ev = read_storm_events(files)
    ev = ev[ev["zone_id"].notna()]
    ev = ev[(ev["end"] >= first - timedelta(days=2)) & (ev["begin"] <= last + timedelta(days=2))]
    return label_storm_events(ev[["zone_id", "event_type", "begin", "end", "narrative"]])


def build(out_dir: Path) -> dict:
    import zarr

    paths = get_paths()
    out_dir.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(zarr_path(), mode="r")
    lat, lon = root["latitude"][:], root["longitude"][:]
    print("zone masks ...", flush=True)
    zone_ids, cells = load_zone_cells(paths.raw / "nws_zones" / "z_16ap26.zip", lat, lon)
    print(f"  {len(zone_ids)} zones kept", flush=True)
    elev = elevation_on_grid(paths.raw / "etopo_2022_30s" / "ETOPO_2022_v1_30s_N90W180_surface.tif", lat, lon)
    np.save(out_dir / "elevation.npy", elev)
    offs = np.cumsum([0] + [len(c) for c in cells])
    np.savez(
        out_dir / "zones.npz",
        zone_ids=np.array(zone_ids),
        cells=np.concatenate(cells),
        offsets=offs,
        grid_shape=np.array(lat.shape),
    )
    print("scan cycles ...", flush=True)
    scan = scan_cycles(zarr_path(), cells)
    np.savez(out_dir / "scan.npz", **scan)
    table, idx = make_windows(scan, zone_ids)
    first = datetime.fromtimestamp(int(scan["time"][0]), UTC)
    last = datetime.fromtimestamp(int(scan["time"][-1]), UTC)
    events = read_events(paths.raw, first, last)
    bags = assign_bag_labels(table, events)
    # carry rule + feature-row indices through the ambiguous-window drop
    key = table["zone_id"] + "|" + table["window_start"].astype(str)
    bkey = bags["zone_id"] + "|" + bags["window_start"].astype(str)
    sel = np.flatnonzero(key.isin(set(bkey)).to_numpy())
    assert len(sel) == len(bags)
    bags["rule"] = table["rule"].to_numpy()[sel]
    bags["t2m_min_k"] = table["t2m_min_k"].to_numpy()[sel]
    for k, v in idx.items():
        bags[k] = v[sel]
    bags.to_parquet(out_dir / "bags.parquet")
    events.to_parquet(out_dir / "events.parquet")
    gate = {
        "n_zones": len(zone_ids),
        "zone_cells_min_median_max": [
            int(np.min(np.diff(offs))),
            float(np.median(np.diff(offs))),
            int(np.max(np.diff(offs))),
        ],
        "n_cycles": int(len(scan["time"])),
        "frame_nan_max": float(np.nanmax(scan["frame_nan"])),
        "frames_with_nan": int((scan["frame_nan"].max(axis=1) > 0).sum()),
        "field_min": {f: float(np.nanmin(scan["frame_min"][:, j])) for j, f in enumerate(ICE_FIELDS)},
        "field_max": {f: float(np.nanmax(scan["frame_max"][:, j])) for j, f in enumerate(ICE_FIELDS)},
        "windows_total_before_finite_filter": int(len(table) + int((scan["nonfinite"] > 0).sum())),
        "windows_non_finite_dropped_est": "see windows_after_finite_filter",
        "windows_after_finite_filter": int(len(table)),
        "bags_after_ambiguous_drop": int(len(bags)),
        "events_labelled": {
            "positive": int((events.label == 1).sum()),
            "ambiguous": int((events.label == -1).sum()),
        },
    }
    (out_dir / "gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))
    return gate


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None, help="default: <OUTRUN_PROC>/hrrr/ice/bags")
    a = ap.parse_args(argv)
    build(a.out or get_paths().proc / "hrrr" / "ice" / "bags")


if __name__ == "__main__":
    main()
