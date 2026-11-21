"""WSTS channel-mapper validation and 2021 fire catalog (T13).

``python -m outrun.hazards.wildfire.wsts_shift ks``      KS table of the mapped WSTS 2018-2020 inputs vs NDWS train
                                                         (inputs only, no labels, no 2021 data).
``python -m outrun.hazards.wildfire.wsts_shift catalog`` one row per WSTS 2021 fire: place, days, burned pixels.
Neither command trains, tunes or scores anything.
"""

from __future__ import annotations

import argparse
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from outrun.config import get_paths
from outrun.data.wsts import channel_shift_stats, map_to_ndws, read_wsts_day

TIF = re.compile(r"^(?P<year>\d{4})/(?P<fire>fire_\d+)/(?P<day>\d{4}-\d{2}-\d{2})\.tif$")


def zip_index(zip_path: Path) -> dict[tuple[str, str], list[str]]:
    """{(year, fire id): sorted day names} from the archive directory."""
    out: dict[tuple[str, str], list[str]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            m = TIF.match(name)
            if m:
                out.setdefault((m["year"], m["fire"]), []).append(name)
    return {k: sorted(v) for k, v in out.items()}


def read_member(z: zipfile.ZipFile, name: str, tmp: Path) -> tuple[dict[str, np.ndarray], float, Path]:
    path = Path(z.extract(name, tmp))
    bands, res = read_wsts_day(path)
    return bands, res, path


def ks_table(zip_path: Path, n_fires: int = 40, days_per_fire: int = 3, seed: int = 0) -> pd.DataFrame:
    """Map ``days_per_fire`` days of ``n_fires`` random 2018-2020 fires and compare with NDWS train inputs."""
    idx = {k: v for k, v in zip_index(zip_path).items() if k[0] in ("2018", "2019", "2020")}
    rng = np.random.default_rng(seed)
    keys = sorted(idx)
    pick = [keys[i] for i in rng.choice(len(keys), min(n_fires, len(keys)), replace=False)]
    mapped = []
    with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory() as tmp:
        for k in pick:
            days = idx[k]
            for name in [
                days[i] for i in rng.choice(len(days), min(days_per_fire, len(days)), replace=False)
            ]:
                bands, res, path = read_member(z, name, Path(tmp))
                mapped.append(map_to_ndws(bands, res))
                path.unlink()
    size = max(m.shape[-1] for m in mapped)
    ws = np.full((len(mapped), 12, size, size), np.nan, np.float32)
    for i, m in enumerate(mapped):
        ws[i, :, : m.shape[1], : m.shape[2]] = m[:, :size, :size]
    ids = np.load(get_paths().proc / "ndws" / "ndws_train.npy", mmap_mode="r")
    nd = np.asarray(ids[np.sort(rng.choice(len(ids), 2000, replace=False))])
    print(
        f"mapped {len(mapped)} WSTS days from {len(pick)} fires of 2018-2020 vs {len(nd)} NDWS train samples"
    )
    return channel_shift_stats(ws, nd)


def fire_catalog(zip_path: Path) -> pd.DataFrame:
    """One row per 2021 fire: first tif centre (lon, lat), days, active-fire pixels summed over days, max/day."""
    import rasterio
    from pyproj import Transformer

    rows = []
    idx = {k: v for k, v in zip_index(zip_path).items() if k[0] == "2021"}
    with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory() as tmp:
        for (_, fire), names in sorted(idx.items()):
            counts = []
            for i, name in enumerate(names):
                path = Path(z.extract(name, tmp))
                with rasterio.open(path) as src:
                    if i == 0:
                        cx, cy = src.xy(src.height // 2, src.width // 2)
                        lon, lat = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True).transform(
                            cx, cy
                        )
                        h, w, crs = src.height, src.width, str(src.crs)
                    desc = [(d or "").strip().lower() for d in src.descriptions]
                    af = src.read(desc.index("active fire") + 1)
                counts.append(int((np.nan_to_num(af) > 0).sum()))
                path.unlink()
            rows.append({"fire": fire, "first_day": names[0][-14:-4], "last_day": names[-1][-14:-4],
                         "n_days": len(names), "lon": lon, "lat": lat, "h": h, "w": w, "crs": crs,
                         "fire_px_total": sum(counts), "fire_px_max_day": max(counts)})  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("ks", "catalog"))
    a = ap.parse_args()
    zip_path = get_paths().raw / "wsts" / "WildfireSpreadTS.zip"
    out = get_paths().proc / "wsts"
    out.mkdir(parents=True, exist_ok=True)
    if a.what == "ks":
        df = ks_table(zip_path)
        df.to_csv(out / "shift_table.csv")
        print(df[["ks", "shifted", "wsts_q05", "wsts_q50", "wsts_q95", "ndws_q05", "ndws_q50", "ndws_q95"]]
              .round(4).to_string())  # fmt: skip
    else:
        df = fire_catalog(zip_path)
        df.to_csv(out / "fires_2021.csv", index=False)
        print(df.sort_values("fire_px_total", ascending=False).head(25).to_string())


if __name__ == "__main__":
    main()
