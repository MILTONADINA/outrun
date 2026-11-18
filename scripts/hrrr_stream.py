"""Stream HRRR fields from AWS by .idx byte ranges, crop them, and append them to an fp16 zarr store.

Two products (DESIGN 3.5 / 3.6, specs 01 DATA-03, 12, 13):

  ice    9 wrfsfc fields, f01 only, 3-hourly cycles, winters Nov 15 - Mar 15 2020-21 .. 2024-25,
         cropped to ICE_BELT (native 3 km).
  smoke  MASSDEN 8 m, HPBL, UGRD/VGRD 10 m, f01 + f06, 6-hourly cycles 2022-2024, cropped to CONUS and
         block-averaged 3x3 to 9 km (keeps the store small). MASSDEN is converted to ug/m3.

How one cycle works (all in ``process_cycle``):
  1. download the small .idx text file for each lead,
  2. ``fetch_fields`` downloads only the wanted GRIB messages (contiguous byte ranges are merged),
  3. every message is decoded on its own with cfgrib, so it is named by its .idx "VAR:LEVEL" string,
  4. ``crop_bbox`` cuts the domain out, values become float16.

Output: $OUTRUN_PROC/hrrr/<product>/data.zarr (zarr v3). Arrays:
  time       int64 (T,)        cycle init time, seconds since 1970 UTC (written LAST for each row)
  <name>     float16 (T, H, W) one per field and lead, e.g. "TMP_surface_f01", "MASSDEN_8m_f06"
  latitude, longitude  float32 (H, W)   (longitude in -180..180)
Rows are in completion order, not time order: sort by ``time`` when reading.

Winds: UGRD/VGRD are stored exactly as HRRR ships them, GRID-RELATIVE (along the Lambert grid's x/y axes,
GRIB uvRelativeToGrid=1), not east/north. Near the CONUS edges the grid axes are turned up to ~20 degrees from
east/north, so a reader must rotate them before using them as earth-relative winds. The store records this in
``root.attrs["winds"]``.

Resume: the cycles already in the ``time`` array are skipped. A small state.json (flushed every 30 s) keeps
the streamed-bytes counter, the per-cycle byte counts and the cycles missing on AWS. The run stops when the
streamed bytes reach the cap.

Usage:
  OUTRUN_PROC=... .venv/bin/python scripts/hrrr_stream.py ice   [--start 2021-01-01] [--end 2021-02-01] [--limit N]
  OUTRUN_PROC=... .venv/bin/python scripts/hrrr_stream.py smoke [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import requests
import zarr

from outrun.config import get_paths, load_config
from outrun.data import hrrr

BUCKET_URL = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
N_THREADS = 8
STATE_FLUSH_S = 30.0

SMOKE_FIELDS = ("MASSDEN:8 m above ground", "HPBL:surface", "UGRD:10 m above ground", "VGRD:10 m above ground")

# Everything that differs between the two products.
PRODUCTS = {
    "ice": dict(fields=hrrr.HRRR_ICE_FIELDS, leads=(1,), domain="ICE_BELT", coarsen=1, cap_gb=80.0),
    "smoke": dict(fields=SMOKE_FIELDS, leads=(1, 6), domain="CONUS", coarsen=3, cap_gb=80.0),
}


# ---------------------------------------------------------------------------------------------------------- cycles
def cycle_times(product: str) -> list[datetime]:
    """All cycle init times of a product, sorted."""
    if product == "ice":
        return hrrr.ice_cycle_times(2020, 2024, hrrr.ICE_CYCLES_3H)
    times, t = [], datetime(2022, 1, 1, tzinfo=UTC)
    while t < datetime(2025, 1, 1, tzinfo=UTC):  # 1,096 days x 00/06/12/18Z
        times.append(t)
        t += timedelta(hours=6)
    return times


def grib_url(init: datetime, lead_h: int) -> str:
    return f"{BUCKET_URL}/hrrr.{init:%Y%m%d}/conus/hrrr.t{init:%H}z.wrfsfcf{lead_h:02d}.grib2"


def array_name(field: str, lead_h: int) -> str:
    """'TMP:2 m above ground', 1 -> 'TMP_2m_f01' (a safe zarr array name)."""
    var, level = field.split(":")
    level = level.replace(" above ground", "").replace(" ", "")
    return f"{var}_{level}_f{lead_h:02d}"


# ---------------------------------------------------------------------------------------------------------- network
_local = threading.local()


def session() -> requests.Session:
    """One HTTP session per worker thread (keeps connections open)."""
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def http_get(url: str, headers: dict | None = None, tries: int = 4) -> bytes | None:
    """GET with retries. Returns None for 404 (file not on AWS)."""
    for attempt in range(tries):
        try:
            r = session().get(url, headers=headers, timeout=60)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2**attempt)
    return None


def range_get(url: str, start: int, end: int | None) -> bytes:
    data = http_get(url, headers={"Range": hrrr.range_header(start, end)})
    if data is None:
        raise FileNotFoundError(url)
    return data


# ---------------------------------------------------------------------------------------------------------- one cycle
_decode_lock = threading.Lock()  # eccodes is not guaranteed thread-safe: decode one message at a time


def coarsen(a: np.ndarray, k: int) -> np.ndarray:
    """k x k block mean (edges that do not fill a block are dropped). k = 1 returns the input."""
    if k == 1:
        return a
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def process_cycle(init: datetime, cfg: dict, bbox: tuple) -> dict | None:
    """Fetch, decode and crop every field and lead of one cycle. None if a file is missing on AWS."""
    out: dict[str, np.ndarray] = {}
    lat = lon = None
    n_bytes = 0
    for lead in cfg["leads"]:
        for field in cfg["fields"]:
            hrrr.check_lead(field.split(":")[0], lead)  # f01 rule
        url = grib_url(init, lead)
        idx_bytes = http_get(url + ".idx")
        if idx_bytes is None:
            return None
        idx_text = idx_bytes.decode()
        data = hrrr.fetch_fields(url, idx_text, cfg["fields"], range_get)
        n_bytes += len(data)

        # fetch_fields concatenates the selected messages in ascending byte order: slice them back apart.
        entries = sorted(hrrr.select_fields(hrrr.parse_idx(idx_text), cfg["fields"]), key=lambda e: e.start)
        offset = 0
        for e in entries:
            size = (e.end - e.start + 1) if e.end is not None else len(data) - offset
            message, offset = data[offset : offset + size], offset + size
            name = array_name(f"{e.var}:{e.level}", lead)
            if name in out:  # a second message with the same VAR:LEVEL (e.g. another time range): keep the first
                continue
            with _decode_lock:
                (ds,) = hrrr.decode_grib_bytes(message)
            da = hrrr.crop_bbox(ds[next(iter(ds.data_vars))], bbox)
            values = da.values.astype(np.float64)
            if e.var == "MASSDEN":
                hrrr.check_massden_units(str(da.attrs.get("units", "unknown")), init + timedelta(hours=lead))
                values = hrrr.massden_to_ugm3(values, init + timedelta(hours=lead))
            out[name] = coarsen(values, cfg["coarsen"]).astype(np.float16)
            if lat is None:
                lat = coarsen(da["latitude"].values, cfg["coarsen"]).astype(np.float32)
                lon = coarsen((da["longitude"].values + 180.0) % 360.0 - 180.0, cfg["coarsen"]).astype(np.float32)
    return {"init": init, "arrays": out, "lat": lat, "lon": lon, "bytes": n_bytes}


# ---------------------------------------------------------------------------------------------------------- storage
class Store:
    """The zarr store plus state.json. Only the main thread touches it."""

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = out_dir / "state.json"
        self.root = zarr.open_group(out_dir / "data.zarr", mode="a", zarr_format=3)
        self.state = {"streamed_bytes": 0, "bytes_per_cycle": {}, "missing": []}
        if self.state_path.exists():
            self.state.update(json.loads(self.state_path.read_text()))
        self.last_flush = time.time()
        self._repair()

    def _repair(self) -> None:
        """After a crash, cut every array back to the rows whose time was really written.

        A crash can leave field arrays one row longer than `time`, or `time` resized but its last entry never
        set. An unset entry reads as the fill value 0 (1970-01-01), which is never a real HRRR cycle, so the
        finished rows are the leading entries with time > 0.
        """
        if "time" not in self.root:
            return
        times = self.root["time"][:]
        unset = np.flatnonzero(times <= 0)
        n = int(unset[0]) if unset.size else len(times)
        if n != len(times):
            self.root["time"].resize((n,))
        for _name, arr in self.root.arrays():
            if arr.ndim == 3 and arr.shape[0] != n:
                arr.resize((n, *arr.shape[1:]))

    def done_times(self) -> set[datetime]:
        if "time" not in self.root:
            return set()
        return {datetime.fromtimestamp(int(s), UTC) for s in self.root["time"][:]}

    def append(self, result: dict) -> None:
        if "time" not in self.root:  # first cycle: create every array from its shapes
            self.root.create_array("latitude", data=result["lat"])
            self.root.create_array("longitude", data=result["lon"])
            for name, a in result["arrays"].items():
                self.root.create_array(name, shape=(0, *a.shape), chunks=(1, *a.shape), dtype="float16")
            self.root.create_array("time", shape=(0,), chunks=(4096,), dtype="int64")
        if any(name.startswith("UGRD") for name in result["arrays"]) and "winds" not in self.root.attrs:
            self.root.attrs["winds"] = "grid-relative (HRRR Lambert, uvRelativeToGrid=1); not rotated to east/north"
        n = self.root["time"].shape[0]
        for name, a in result["arrays"].items():
            arr = self.root[name]
            arr.resize((n + 1, *arr.shape[1:]))
            arr[n] = a
        t = self.root["time"]
        t.resize((n + 1,))
        t[n] = int(result["init"].timestamp())  # written last: a row counts only once its time is set
        self.state["streamed_bytes"] += result["bytes"]
        self.state["bytes_per_cycle"][result["init"].isoformat()] = result["bytes"]

    def flush(self, force: bool = False) -> None:
        if force or time.time() - self.last_flush > STATE_FLUSH_S:
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state))
            tmp.replace(self.state_path)  # atomic
            self.last_flush = time.time()


# ---------------------------------------------------------------------------------------------------------- main
def parse_day(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("product", choices=sorted(PRODUCTS))
    ap.add_argument("--start", type=parse_day, help="first cycle day, YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", type=parse_day, help="last day, YYYY-MM-DD (exclusive)")
    ap.add_argument("--limit", type=int, help="process at most N new cycles")
    ap.add_argument("--cap-gb", type=float, help="stop when this many GB were streamed (default 80)")
    ap.add_argument("--dry-run", action="store_true", help="list what would run, download nothing")
    args = ap.parse_args(argv)

    cfg = PRODUCTS[args.product]
    cap = (args.cap_gb or cfg["cap_gb"]) * 1e9
    dom = load_config("domains")["domains"][cfg["domain"]]
    bbox = (dom["lon"][0], dom["lat"][0], dom["lon"][1], dom["lat"][1])
    out_dir = get_paths().proc / "hrrr" / args.product

    todo = cycle_times(args.product)
    if args.start:
        todo = [t for t in todo if t >= args.start]
    if args.end:
        todo = [t for t in todo if t < args.end]
    n_total = len(todo)

    if args.dry_run:
        print(f"== dry run {args.product}: {n_total} cycles {todo[0]:%Y-%m-%d %HZ} .. {todo[-1]:%Y-%m-%d %HZ}, "
              f"leads {cfg['leads']}, {len(cfg['fields'])} fields, bbox {bbox}, coarsen {cfg['coarsen']}, "
              f"cap {cap / 1e9:.0f} GB, out {out_dir}")
        return 0

    store = Store(out_dir)
    skip = store.done_times() | {datetime.fromisoformat(s) for s in store.state["missing"]}
    todo = [t for t in todo if t not in skip]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"== {args.product}: {n_total} cycles in range, {n_total - len(todo)} already done or missing "
          f"(or beyond --limit), {len(todo)} to do; streamed so far {store.state['streamed_bytes'] / 1e9:.2f} GB",
          flush=True)

    t0, run_bytes, n_done = time.time(), 0, 0
    pending: dict = {}
    queue = iter(todo)
    stopped = False
    with ThreadPoolExecutor(N_THREADS) as pool:
        while True:
            # Keep the pool full unless the cap is reached.
            while not stopped and len(pending) < 2 * N_THREADS:
                if store.state["streamed_bytes"] >= cap:
                    print(f"== cap reached ({cap / 1e9:.0f} GB): not starting new cycles", flush=True)
                    stopped = True
                    break
                t = next(queue, None)
                if t is None:
                    break
                pending[pool.submit(process_cycle, t, cfg, bbox)] = t
            if not pending:
                break
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                t = pending.pop(fut)
                try:
                    result = fut.result()
                except Exception as exc:  # network or decode failure: log it, the next run retries it
                    print(f"== FAILED {t:%Y-%m-%d %HZ}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                if result is None:
                    print(f"== missing on AWS {t:%Y-%m-%d %HZ}", flush=True)
                    store.state["missing"].append(t.isoformat())
                    continue
                store.append(result)
                n_done += 1
                run_bytes += result["bytes"]
                dt = time.time() - t0
                print(f"== {t:%Y-%m-%d %HZ} {result['bytes'] / 1e6:.1f} MB | {n_done}/{len(todo)} | "
                      f"{run_bytes / 1e6 / dt:.1f} MB/s | total {store.state['streamed_bytes'] / 1e9:.2f} GB",
                      flush=True)
            store.flush()
    store.flush(force=True)
    dt = time.time() - t0
    print(f"== done {n_done} cycles, {run_bytes / 1e9:.3f} GB in {dt:.0f} s "
          f"({run_bytes / 1e6 / max(dt, 1e-9):.1f} MB/s); store has {len(store.done_times())} cycles", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
