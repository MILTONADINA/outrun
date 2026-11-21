"""Flood data prep: fast MRMS reader, hour scan, sample table, feature/label arrays (specs/10-flood.md, DESIGN 3.3).

Stages (``python -m outrun.hazards.flood.prep <stage>``; every stage is deterministic: seeded, sorted, order-free):
  scan     per-hour 64x64-block maxima of QPE_01H (to find hard negatives / wet hours) + raw-value gate counts
  select   sample table (positives from Storm Events, hard negatives, wet hours): ids + times + patch origins only
  (freeze) ``outrun.eval.freeze_splits.freeze_flood`` writes splits/flood.json from the table, before anything is fit
  build    features (11, 64, 64), ring labels, kinds for every table row; per-split stats for train and val only

A sample is an *issue hour* T (UTC) plus a 64x64 patch origin on the native MRMS 0.01 degree grid. The target is a
flash flood within [T, T+6 h] (ring labels, FLOOD-02). Missing MRMS values (-999 / -3) are masked to NaN by
``data.mrms.clean_mrms``; FFG percent -> ratio by ``data.mrms.ffg_percent_to_ratio`` (the one place for that).
"""

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import os
import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from outrun.config import get_paths
from outrun.data.mrms import clean_mrms, ffg_percent_to_ratio
from outrun.hazards.flood import dataset as D

ROWS, COLS, STEP = 3500, 7000, 0.01
LAT0, LON0 = 54.995, -129.995  # centre of cell (0, 0): MRMS CONUS grid, north to south, west to east
PATCH = 64
QPE_DIR = "mrms_qpe_01h_pass2"
QPE_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"
FLASH_PRODUCTS = ("FLASH_QPE_FFG01H_00.00", "FLASH_QPE_FFG03H_00.00", "FLASH_QPE_ARIMAX_00.00")
EPOCH = datetime(2021, 1, 1, tzinfo=UTC)
R_EARTH = 6_371_008.8
SEED = 0


# ------------------------------------------------------------------ reading
def hour_path(raw: Path, product: str, t: datetime) -> Path:
    day = t.strftime("%Y%m%d")
    if product == QPE_PRODUCT:
        return raw / QPE_DIR / day / f"MRMS_{product}_{day}-{t:%H}0000.grib2.gz"
    return raw / "mrms_flash" / product / day / f"MRMS_{product}_{day}-{t:%H}0000.grib2.gz"


def read_raw(path: Path) -> np.ndarray:
    """Fast eccodes decode of one gzipped MRMS GRIB2 message: raw float32 (3500, 7000), north first."""
    import eccodes

    with gzip.open(path, "rb") as f:
        message = f.read()
    h = eccodes.codes_new_from_message(message)
    try:
        v = eccodes.codes_get_values(h).astype(np.float32).reshape(ROWS, COLS)
    finally:
        eccodes.codes_release(h)
    return v


def clean(raw_values: np.ndarray, product: str) -> np.ndarray:
    """Same cleaning as ``data.mrms.read_mrms`` (negatives -> NaN; FFG percent -> ratio)."""
    return ffg_percent_to_ratio(raw_values) if product.startswith("FLASH_QPE_FFG") else clean_mrms(raw_values)


def gate_counts(raw_values: np.ndarray, cleaned: np.ndarray) -> dict[str, float]:
    return {
        "n": float(raw_values.size),
        "neg999": float((raw_values == -999).sum()),
        "neg3": float((raw_values == -3).sum()),
        "other_neg": float(((raw_values < 0) & (raw_values != -999) & (raw_values != -3)).sum()),
        "v9999": float((raw_values >= 9999).sum()),
        "nan_after": float(np.isnan(cleaned).sum()),
        "neg_after": float((cleaned < 0).sum()),
        "max_after": float(np.nanmax(cleaned)) if np.isfinite(cleaned).any() else float("nan"),
    }


def merge_gate(acc: dict[str, float], g: dict[str, float]) -> None:
    for k, v in g.items():
        if k == "max_after":
            acc[k] = max(acc.get(k, -np.inf), v) if np.isfinite(v) else acc.get(k, -np.inf)
        else:
            acc[k] = acc.get(k, 0.0) + v


def hour_index(t: datetime) -> int:
    return int((t - EPOCH) / timedelta(hours=1))


def hour_from_index(i: int) -> datetime:
    return EPOCH + timedelta(hours=int(i))


def list_hours(root: Path) -> list[datetime]:
    """UTC hours of the on-the-hour files under ``root/YYYYMMDD/``."""
    out = []
    for day in sorted(os.listdir(root)):
        if len(day) == 8 and day.isdigit():
            for f in sorted(os.listdir(root / day)):
                m = re.search(r"-(\d{2})0000\.grib2\.gz$", f)
                if m:
                    out.append(datetime.strptime(day + m.group(1), "%Y%m%d%H").replace(tzinfo=UTC))
    return out


def proc_dir() -> Path:
    d = get_paths().proc / "flood"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ geometry
def cell_lonlat(row0: int, col0: int, n: int = PATCH) -> tuple[np.ndarray, np.ndarray]:
    lat = LAT0 - (row0 + np.arange(n)) * STEP
    lon = LON0 + (col0 + np.arange(n)) * STEP
    return np.meshgrid(lon, lat)  # (n, n) each, (row, col) layout


def to_xy(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular frame in metres around (lon0, lat0); fine for a 64 km patch plus a 25 km halo."""
    x = R_EARTH * np.cos(np.radians(lat0)) * np.radians(np.asarray(lon) - lon0)
    y = R_EARTH * np.radians(np.asarray(lat) - lat0)
    return x, y


def patch_xy(row0: int, col0: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    lon, lat = cell_lonlat(row0, col0)
    lon0, lat0 = float(lon.mean()), float(lat.mean())
    x, y = to_xy(lon, lat, lon0, lat0)
    return x, y, lon0, lat0


# ------------------------------------------------------------------ reports
def load_reports(raw: Path) -> pd.DataFrame:
    """Storm Events 'Flash Flood' 2021-2025 with a begin location (end location defaults to begin), UTC times."""
    from outrun.data.stormevents import filter_event_types, read_storm_events

    files = sorted((raw / "storm_events").glob("StormEvents_details-ftp_v1.0_d202[1-5]_*.csv.gz"))
    df = filter_event_types(read_storm_events(files), ["Flash Flood"])
    df = df.dropna(subset=["begin_lat", "begin_lon", "begin"]).copy()
    df["end_lat"] = df["end_lat"].fillna(df["begin_lat"])
    df["end_lon"] = df["end_lon"].fillna(df["begin_lon"])
    df["end"] = df["end"].fillna(df["begin"])
    df = df[df["end"] >= df["begin"]]
    inside = df["begin_lat"].between(20.5, 54.5) & df["begin_lon"].between(-129.5, -60.5)
    return df[inside].sort_values("event_id").reset_index(drop=True)


def reports_near(df: pd.DataFrame, t: datetime, pad_h: float, lat0: float, lon0: float, deg: float = 1.0):
    """Reports with [begin - pad_h, end + pad_h] containing t and located within ``deg`` degrees of the patch."""
    ok = (
        (df["begin"] - timedelta(hours=pad_h) <= t)
        & (df["end"] + timedelta(hours=pad_h) >= t)
        & ((df["begin_lat"] - lat0).abs() <= deg)
        & ((df["begin_lon"] - lon0).abs() <= deg)
    )
    return df[ok]


def to_flood_reports(sub: pd.DataFrame, lon0: float, lat0: float) -> list[D.FloodReport]:
    out = []
    for r in sub.itertuples():
        x0, y0 = to_xy(r.begin_lon, r.begin_lat, lon0, lat0)
        x1, y1 = to_xy(r.end_lon, r.end_lat, lon0, lat0)
        out.append(
            D.FloodReport(
                r.begin.to_pydatetime(), r.end.to_pydatetime(), float(x0), float(y0), float(x1), float(y1)
            )
        )
    return out


# ------------------------------------------------------------------ stage: scan
def _scan_one(args):
    raw, t = args
    p = hour_path(raw, QPE_PRODUCT, t)
    v = read_raw(p)
    c = clean(v, QPE_PRODUCT)
    g = gate_counts(v, c)
    nb_r, nb_c = ROWS // PATCH, COLS // PATCH
    block = np.where(np.isnan(c), -1.0, c)[: nb_r * PATCH, : nb_c * PATCH].reshape(nb_r, PATCH, nb_c, PATCH)
    return hour_index(t), block.max(axis=(1, 3)).astype(np.float16), g


def stage_scan(raw: Path, workers: int, limit: int | None = None) -> None:
    hours = list_hours(raw / QPE_DIR)
    if limit:
        hours = hours[:limit]
    acc: dict[str, float] = {}
    idx, blocks = [], []
    with mp.Pool(workers) as pool:
        for i, (hi, b, g) in enumerate(pool.imap(_scan_one, [(raw, t) for t in hours], chunksize=8)):
            idx.append(hi)
            blocks.append(b)
            merge_gate(acc, g)
            if i % 2000 == 0:
                print(f"scan {i}/{len(hours)}", flush=True)
    order = np.argsort(idx)
    np.savez(proc_dir() / "scan.npz", hour_idx=np.array(idx)[order], block_max=np.stack(blocks)[order])
    (proc_dir() / "gate_qpe.json").write_text(json.dumps(acc, indent=1))
    print("qpe gate:", acc, flush=True)


# ------------------------------------------------------------------ stage: select
def stage_select(raw: Path, quota_cap: int | None = None) -> None:
    reps = load_reports(raw)
    scan = np.load(proc_dir() / "scan.npz")
    hidx, bmax = scan["hour_idx"], scan["block_max"].astype(np.float32)
    n_all = hour_index(datetime(2026, 1, 1, tzinfo=UTC))
    has_qpe = np.zeros(n_all + 1, bool)
    has_qpe[hidx] = True
    has_flash = np.ones(n_all + 1, bool)
    for p in FLASH_PRODUCTS:
        m = np.zeros(n_all + 1, bool)
        m[[hour_index(t) for t in list_hours(raw / "mrms_flash" / p)]] = True
        has_flash &= m
    # a chain is usable when all 24 trailing QPE hours and the FLASH hour exist
    run = np.zeros(n_all + 1, int)
    for k in range(24):
        run += np.roll(has_qpe, k)
    usable = (run == 24) & has_flash
    usable[:24] = False
    print(f"reports {len(reps)}; usable issue hours {int(usable.sum())} of {len(hidx)} QPE hours", flush=True)

    chosen: dict[int, list[tuple[int, int]]] = defaultdict(list)
    rows: list[dict] = []

    def free(hi: int, r0: int, c0: int) -> bool:
        return all(abs(r0 - a) >= 32 or abs(c0 - b) >= 32 for a, b in chosen[hi])

    def add(hi: int, r0: int, c0: int, basis: str, ev: int) -> None:
        chosen[hi].append((r0, c0))
        t = hour_from_index(hi)
        rows.append(
            {
                "id": f"flood:{t:%Y%m%d%H}:{r0:04d}_{c0:04d}",
                "hour": hi,
                "row0": r0,
                "col0": c0,
                "basis": basis,
                "event_id": ev,
            }
        )

    # positives: one issue hour 0-5 h before the report hour, jittered patch so the report is not centred
    n_skip = 0
    for r in reps.itertuples():
        rng = np.random.default_rng([SEED, int(r.event_id)])
        t_rep = r.begin.to_pydatetime().replace(minute=0, second=0, microsecond=0)
        lead = rng.permutation(6)
        dr, dc = rng.integers(-20, 21, 2)
        mid_lat, mid_lon = (r.begin_lat + r.end_lat) / 2, (r.begin_lon + r.end_lon) / 2
        row_c, col_c = round((LAT0 - mid_lat) / STEP), round((mid_lon - LON0) / STEP)
        r0 = int(np.clip(row_c + dr - PATCH // 2, 0, ROWS - PATCH))
        c0 = int(np.clip(col_c + dc - PATCH // 2, 0, COLS - PATCH))
        placed = False
        for k in lead:
            hi = hour_index(t_rep - timedelta(hours=int(k)))
            if 0 <= hi <= n_all and usable[hi] and free(hi, r0, c0):
                add(hi, r0, c0, "positive", int(r.event_id))
                placed = True
                break
        n_skip += not placed
    n_pos = len(rows)
    quota = n_pos if quota_cap is None else min(n_pos, quota_cap)
    print(f"positives {n_pos} (skipped {n_skip}: no usable hour or overlapping patch)", flush=True)

    # hard negatives (block max qpe_1h >= 25 mm, no report within 25 km, +-9 h) and wet blocks (1-25 mm)
    for basis, lo, hi_mm in (("hard_negative", 25.0, np.inf), ("wet", 1.0, 25.0)):
        cand = np.argwhere(usable[hidx][:, None, None] & (bmax >= lo) & (bmax < hi_mm))
        rng = np.random.default_rng([SEED, 1 if basis == "wet" else 2])
        cand = cand[rng.permutation(len(cand))]
        taken, per_day = 0, set()
        for ci, bi, bj in cand:
            hi = int(hidx[ci])
            r0, c0 = int(bi) * PATCH, int(bj) * PATCH
            day_key = (hi // 24, bi, bj)
            if day_key in per_day or not free(hi, r0, c0):
                continue
            t = hour_from_index(hi)
            x, y, lon0, lat0 = patch_xy(r0, c0)
            if basis == "hard_negative":
                near = reports_near(reps, t, 9.0, lat0, lon0, deg=1.0)
                if D.near_report(x, y, to_flood_reports(near, lon0, lat0), t, radius_m=25000.0, window_h=9.0):
                    continue
            per_day.add(day_key)
            add(hi, r0, c0, basis, -1)
            taken += 1
            if taken >= quota:
                break
        print(f"{basis}: {taken} of {len(cand)} candidates (quota {quota})", flush=True)

    tab = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    assert tab["id"].is_unique
    tab.to_parquet(proc_dir() / "samples.parquet")
    reps.to_parquet(proc_dir() / "reports.parquet")
    print(tab.groupby("basis").size().to_dict(), "total", len(tab), flush=True)


# ------------------------------------------------------------------ statics
def build_statics(raw: Path) -> Path:
    """ETOPO 2022 30 arc-second surface -> MRMS 0.01 degree grid: elevation (m), slope (deg), TPI (m, 5-cell)."""
    import rasterio
    from rasterio.windows import Window
    from scipy.ndimage import map_coordinates

    from outrun.data.dem import slope_deg, tpi

    out = proc_dir() / "statics.npz"
    if out.exists():
        return out
    tif = next((raw / "etopo_2022_30s").glob("*.tif"))
    with rasterio.open(tif) as src:
        res = src.transform.a  # degrees per pixel (1/120)
        top, left = src.transform.f, src.transform.c
        r0, r1 = int((top - 55.2) / res), int((top - 19.8) / res) + 1
        c0, c1 = int((-130.2 - left) / res), int((-59.8 - left) / res) + 1
        z = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
    lat = LAT0 - np.arange(ROWS) * STEP
    lon = LON0 + np.arange(COLS) * STEP
    fr = ((top - lat) / res - 0.5 - r0)[:, None] * np.ones((1, COLS))
    fc = ((lon - left) / res - 0.5 - c0)[None, :] * np.ones((ROWS, 1))
    elev = map_coordinates(z, [fr, fc], order=1, mode="nearest").astype(np.float32)
    np.savez(out, elevation=elev, slope=slope_deg(elev, 1000.0), tpi=tpi(elev, 5))
    return out


# ------------------------------------------------------------------ stage: build
_STAGE: dict = {}


def _init_worker(paths: dict, shapes: dict) -> None:
    for name, (path, shape, dtype) in paths.items():
        _STAGE[name] = np.memmap(path, dtype=dtype, mode="r+", shape=shape)


def _extract(task):
    """Decode one MRMS hour once and copy the 64x64 crops of every sample that needs it."""
    kind, path, product, entries = task
    v = read_raw(Path(path))
    c = clean(v, product)
    g = gate_counts(v, c)
    arr = _STAGE["qpe"] if kind == "qpe" else _STAGE["flash"]
    for ent in entries:
        s, slot, r0, c0 = ent
        arr[s, slot] = c[r0 : r0 + PATCH, c0 : c0 + PATCH]
    return product, g


def stage_build(raw: Path, workers: int) -> None:
    out = proc_dir()
    tab = pd.read_parquet(out / "samples.parquet")
    reps = pd.read_parquet(out / "reports.parquet")
    reps["begin"], reps["end"] = (
        pd.to_datetime(reps["begin"], utc=True),
        pd.to_datetime(reps["end"], utc=True),
    )
    S = len(tab)
    tmp_q, tmp_f = out / "stage_qpe.f32", out / "stage_flash.f32"
    shape_q, shape_f = (S, 24, PATCH, PATCH), (S, 3, PATCH, PATCH)
    np.memmap(tmp_q, dtype=np.float32, mode="w+", shape=shape_q).fill(np.nan)
    np.memmap(tmp_f, dtype=np.float32, mode="w+", shape=shape_f).fill(np.nan)

    qtasks: dict[int, list] = defaultdict(list)
    ftasks: dict[tuple[int, int], list] = defaultdict(list)
    for s, r in enumerate(tab.itertuples()):
        for k in range(24):  # slot k = hour (T - k)
            qtasks[r.hour - k].append((s, k, r.row0, r.col0))
        for j in range(3):
            ftasks[(r.hour, j)].append((s, j, r.row0, r.col0))
    tasks = [
        ("qpe", str(hour_path(raw, QPE_PRODUCT, hour_from_index(h))), QPE_PRODUCT, e)
        for h, e in sorted(qtasks.items())
    ]
    tasks += [
        ("flash", str(hour_path(raw, FLASH_PRODUCTS[j], hour_from_index(h))), FLASH_PRODUCTS[j], e)
        for (h, j), e in sorted(ftasks.items())
    ]
    print(f"{S} samples, {len(tasks)} hour files to decode", flush=True)
    gates: dict[str, dict] = defaultdict(dict)
    paths = {"qpe": (tmp_q, shape_q, np.float32), "flash": (tmp_f, shape_f, np.float32)}
    with mp.Pool(workers, initializer=_init_worker, initargs=(paths, None)) as pool:
        for i, (product, g) in enumerate(pool.imap_unordered(_extract, tasks, chunksize=4)):
            merge_gate(gates[product], g)
            if i % 5000 == 0:
                print(f"extract {i}/{len(tasks)}", flush=True)
    (out / "gate_extracted.json").write_text(json.dumps(gates, indent=1))
    print("gate (decoded files):", json.dumps(gates), flush=True)

    st = np.load(build_statics(raw))
    qpe = np.memmap(tmp_q, dtype=np.float32, mode="r", shape=shape_q)
    fl = np.memmap(tmp_f, dtype=np.float32, mode="r", shape=shape_f)
    X = np.lib.format.open_memmap(out / "x.npy", mode="w+", dtype=np.float32, shape=(S, 11, PATCH, PATCH))
    Y = np.lib.format.open_memmap(out / "y.npy", mode="w+", dtype=np.int8, shape=(S, PATCH, PATCH))
    kinds, qpe_max, skipped = [], np.zeros(S, np.float32), []
    for s, r in enumerate(tab.itertuples()):
        t = hour_from_index(r.hour)
        q = qpe[s]  # (24, H, W); slot k = hour T-k; NaN where the hour is missing at that pixel
        sl = (slice(r.row0, r.row0 + PATCH), slice(r.col0, r.col0 + PATCH))
        acc = {
            h: q[:h].sum(axis=0) for h in (1, 3, 6, 24)
        }  # NaN propagates: partial windows are invalid pixels
        try:
            X[s] = D.build_features(
                acc[1],
                acc[3],
                acc[6],
                acc[24],
                fl[s, 0],
                fl[s, 1],
                fl[s, 2],
                st["elevation"][sl],
                st["slope"][sl],
                st["tpi"][sl],
            )
        except D.MissingHoursError:
            skipped.append(r.id)
            X[s] = 0.0
            Y[s] = 0
            kinds.append(D.SampleKind.DRY.value)
            continue
        x, y, lon0, lat0 = patch_xy(r.row0, r.col0)
        sub = reports_near(reps, t, 9.0, lat0, lon0, 1.5)
        frep = to_flood_reports(sub, lon0, lat0)
        lab = D.ring_labels(x, y, frep, t)
        Y[s] = lab
        kinds.append(D.sample_kind(lab, acc[1], D.near_report(x, y, frep, t)).value)
        qpe_max[s] = float(np.nanmax(acc[1])) if np.isfinite(acc[1]).any() else np.nan
    X.flush()
    Y.flush()
    tab["kind"] = kinds
    tab.to_parquet(out / "samples_built.parquet")
    (out / "ids.json").write_text(json.dumps(list(tab["id"])))
    (out / "kinds.json").write_text(json.dumps(kinds))
    (out / "skipped.json").write_text(json.dumps(skipped))
    tmp_q.unlink()
    tmp_f.unlink()
    print(f"built x {X.shape} y {Y.shape}; skipped (missing hours) {len(skipped)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("scan", "select", "build"))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quota-cap", type=int, default=None)
    a = ap.parse_args()
    raw = get_paths().raw
    if a.stage == "scan":
        stage_scan(raw, a.workers, a.limit)
    elif a.stage == "select":
        stage_select(raw, a.quota_cap)
    else:
        stage_build(raw, a.workers)


if __name__ == "__main__":
    main()
