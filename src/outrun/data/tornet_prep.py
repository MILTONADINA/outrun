"""TorNet preprocessing: stream the yearly tarballs once and write compact fp16 arrays per split.

Run: ``python -m outrun.data.tornet_prep`` (OUTRUN_RAW / OUTRUN_PROC set). Reads ``splits/tornet.json`` (ids are
catalog ``filename`` values, also the tar member names) and the catalog. Output in ``$OUTRUN_PROC/tornet/``:

- ``{split}_x.npy``    float16 (N, 25, A, R): the first 25 channels of ``read_chip`` (last frame, temporal
  difference, missing mask). The two coordinate channels are redundant, so ``{split}_range_km.npy`` (N, R) and
  ``{split}_azimuth.npy`` (N, A) are stored instead; rebuild them with ``features.build_chip_tensor`` or broadcast.
- ``{split}_meta.csv`` one row per chip, same order: id, label (1 = TOR), category, ef_number, lat, lon, start_time,
  site_name, and QC columns (``miss_any``, ``dbz0_missing``, ``n_sentinel``, ``all_missing``).
- ``qc_summary.json``  per-variable raw min/max and NaN fractions, so fill/sentinel issues are visible.

split in train, val, test (official TorNet test chips, preprocessed only) and case_holdout (case-study quarantine).
The test array is never used for training, tuning or scoring before T24.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from outrun.data.tornet import VARIABLES, chip_from_dataset

N_STORED = 25
A, R = 120, 240
SPLITS = ("train", "val", "test", "case_holdout")


def _ids(split_json: dict) -> dict[str, list[str]]:
    out = {s: [x["id"] for x in split_json[s]] for s in ("train", "val", "test")}
    out["case_holdout"] = [x["id"] for x in split_json["extra"]["case_holdout"]]
    return out


def _open_bytes(data: bytes) -> xr.Dataset:
    import netCDF4

    return xr.open_dataset(xr.backends.NetCDF4DataStore(netCDF4.Dataset("chip.nc", memory=data)))


def _worker(year: int, tar_path: str, out: str, ids: dict[str, list[str]], q: mp.Queue) -> None:
    out_p = Path(out)
    where: dict[str, list[tuple[str, int]]] = {}
    for s in SPLITS:
        for i, name in enumerate(ids[s]):
            if name.split("/")[1] == str(year):
                where.setdefault(name, []).append((s, i))
    x = {s: np.load(out_p / f"{s}_x.npy", mmap_mode="r+") for s in SPLITS}
    rk = {s: np.load(out_p / f"{s}_range_km.npy", mmap_mode="r+") for s in SPLITS}
    az = {s: np.load(out_p / f"{s}_azimuth.npy", mmap_mode="r+") for s in SPLITS}
    qc = {s: np.load(out_p / f"{s}_qc.npy", mmap_mode="r+") for s in SPLITS}
    vmin = dict.fromkeys(VARIABLES, np.inf)
    vmax = dict.fromkeys(VARIABLES, -np.inf)
    nan_sum = dict.fromkeys(VARIABLES, 0.0)
    n_chips = 0
    seen: set[str] = set()
    t0 = time.time()
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if m.name not in where:
                continue
            ds = _open_bytes(tf.extractfile(m).read())
            c, rkm, azd = chip_from_dataset(ds)
            for s, i in where[m.name]:
                x[s][i] = c[:N_STORED]
                rk[s][i] = rkm
                az[s][i] = azd
                miss = c[24] > 0
                dbz_miss = np.isnan(ds["DBZ"].values[-1, :, :, 0]).mean()
                sent = sum(int((np.abs(np.nan_to_num(ds[v].values)) >= 900).sum()) for v in VARIABLES)
                qc[s][i] = (miss.mean(), dbz_miss, sent, float(miss.all()))
            for v in VARIABLES:
                a = ds[v].values
                vmin[v] = min(vmin[v], float(np.nanmin(a)) if not np.isnan(a).all() else np.inf)
                vmax[v] = max(vmax[v], float(np.nanmax(a)) if not np.isnan(a).all() else -np.inf)
                nan_sum[v] += float(np.isnan(a).mean())
            ds.close()
            seen.add(m.name)
            n_chips += 1
            if n_chips % 1000 == 0:
                print(f"[{year}] {n_chips}/{len(where)} chips, {time.time() - t0:.0f}s", flush=True)
    for arr in (*x.values(), *rk.values(), *az.values(), *qc.values()):
        arr.flush()
    q.put({"year": year, "n": n_chips, "missing_in_tar": sorted(set(where) - seen)[:20],
           "n_missing_in_tar": len(set(where) - seen), "vmin": vmin, "vmax": vmax,
           "mean_nan_frac": {v: nan_sum[v] / max(n_chips, 1) for v in VARIABLES}, "seconds": time.time() - t0})


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    raw = Path(os.environ["OUTRUN_RAW"])
    out = Path(os.environ["OUTRUN_PROC"]) / "tornet"
    out.mkdir(parents=True, exist_ok=True)
    sj = json.loads((root / "splits" / "tornet.json").read_text())
    ids = _ids(sj)
    cat = pd.read_csv(raw / "tornet_catalog" / "catalog.csv").set_index("filename")
    tars = {2015: "12655151/files/tornet_2015.tar.gz", 2016: "12655179/files/tornet_2016.tar.gz",
            2019: "12655716/files/tornet_2019.tar.gz"}
    for s in SPLITS:
        n = len(ids[s])
        np.lib.format.open_memmap(out / f"{s}_x.npy", "w+", np.float16, (n, N_STORED, A, R)).flush()
        np.lib.format.open_memmap(out / f"{s}_range_km.npy", "w+", np.float32, (n, R)).flush()
        np.lib.format.open_memmap(out / f"{s}_azimuth.npy", "w+", np.float32, (n, A)).flush()
        np.lib.format.open_memmap(out / f"{s}_qc.npy", "w+", np.float32, (n, 4)).flush()
        print(s, n, flush=True)
    q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=_worker, args=(y, str(raw / "tornet_2019_2015_2016" / t), str(out), ids, q))
             for y, t in tars.items()]
    for p in procs:
        p.start()
    res = [q.get() for _ in procs]
    for p in procs:
        p.join()
    for s in SPLITS:
        meta = cat.loc[ids[s]].reset_index()
        meta["label"] = (meta["category"] == "TOR").astype(int)
        qcs = np.load(out / f"{s}_qc.npy")
        meta["miss_any"], meta["dbz0_missing"], meta["n_sentinel"], meta["all_missing"] = qcs.T
        meta.to_csv(out / f"{s}_meta.csv", index=False)
    (out / "qc_summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
