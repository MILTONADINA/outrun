"""SEVIR reader: CATALOG filter, VIL (NEXRAD) + GOES-16 IR107 h5 to 2 km uint8 memmaps (specs/01-data.md DATA-01,
DATA-08)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

GB = 10**9

# [CANDIDATE] h5 layout and IR encoding, verified in T1 against real SEVIR files.
VIL_KEY = "vil"
IR107_KEY = "ir107"
ID_KEY = "id"
IR_RAW_SCALE = 0.01
IR_RAW_OFFSET = 273.15
STORM_ID_PREFIX = "S"

IR_TB_MIN_K = 180.0
IR_TB_MAX_K = 320.0

CATALOG_REQUIRED = (
    "id", "file_name", "file_index", "img_type", "time_utc", "proj",
    "llcrnrlat", "llcrnrlon", "urcrnrlat", "urcrnrlon",
)  # fmt: skip


@dataclass(frozen=True)
class CapResult:
    kept: list[str]
    dropped: list[str]
    total_bytes: int
    over_cap: bool


def read_catalog(path: Path | str) -> pd.DataFrame:
    """DATA-01. CATALOG.csv with tz-aware UTC ``time_utc``; missing required column -> KeyError."""
    cat = pd.read_csv(path, low_memory=False)
    missing = [c for c in CATALOG_REQUIRED if c not in cat.columns]
    if missing:
        raise KeyError(f"CATALOG is missing columns {missing}")
    cat["time_utc"] = pd.to_datetime(cat["time_utc"], utc=True)
    return cat


def filter_catalog(
    catalog: pd.DataFrame,
    years: Sequence[int] = (2018, 2019),
    img_types: Sequence[str] = ("vil", "ir107"),
    bbox: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """DATA-01. Year/type/storm-event/bbox-intersection filter, then ids having every img_type; sorted (id, img_type)."""
    t = pd.to_datetime(catalog["time_utc"], utc=True)
    keep = t.dt.year.isin(list(years)) & catalog["img_type"].isin(list(img_types))
    keep &= catalog["id"].astype(str).str.startswith(STORM_ID_PREFIX)
    if bbox is not None:
        w, s, e, n = bbox
        keep &= (
            (catalog["llcrnrlon"] <= e) & (catalog["urcrnrlon"] >= w)
            & (catalog["llcrnrlat"] <= n) & (catalog["urcrnrlat"] >= s)
        )  # fmt: skip
    out = catalog[keep]
    n_types = out.groupby("id")["img_type"].transform("nunique")
    out = out[n_types == len(set(img_types))]
    return out.sort_values(["id", "img_type"], kind="stable").reset_index(drop=True)


def pool_vil_2x2(vil: np.ndarray) -> np.ndarray:
    """DATA-01. 2x2 average pool of the last two axes, rounded half up, uint8; odd sizes -> ValueError."""
    h, w = vil.shape[-2:]
    if h % 2 or w % 2:
        raise ValueError(f"last two axes must be even, got {(h, w)}")
    v = vil.astype(np.uint16)  # 4 * 255 + 2 fits
    s = v[..., 0::2, 0::2] + v[..., 0::2, 1::2] + v[..., 1::2, 0::2] + v[..., 1::2, 1::2] + 2
    return (s // 4).astype(np.uint8)


def decode_ir(raw: np.ndarray, scale: float = IR_RAW_SCALE, offset: float = IR_RAW_OFFSET) -> np.ndarray:
    """DATA-01. Raw IR -> brightness temperature in kelvin (float32)."""
    return (np.asarray(raw).astype(np.float64) * scale + offset).astype(np.float32)


def ir_to_uint8(tb_k: np.ndarray) -> np.ndarray:
    """DATA-01. Clip to [180, 320] K, floor((tb-180)/140*255 + 0.5); NaN -> 0."""
    tb = np.nan_to_num(np.asarray(tb_k, dtype=np.float64), nan=IR_TB_MIN_K)
    tb = np.clip(tb, IR_TB_MIN_K, IR_TB_MAX_K)
    return np.floor((tb - IR_TB_MIN_K) / (IR_TB_MAX_K - IR_TB_MIN_K) * 255 + 0.5).astype(np.uint8)


def _event_row(h5: h5py.File, key: str, index: int, eid: str, path: Path) -> h5py.Dataset:
    got = h5[ID_KEY][index]
    got = got.decode() if isinstance(got, bytes) else str(got)
    if got != eid:
        raise ValueError(f"{path}: row {index} holds id {got!r}, CATALOG says {eid!r}")
    return h5[key]


def build_sevir_memmap(
    rows: pd.DataFrame,
    raw_dir: Path | str,
    out_path: Path | str,
    ir_scale: float = IR_RAW_SCALE,
    ir_offset: float = IR_RAW_OFFSET,
) -> pd.DataFrame:
    """DATA-01. Write (N, T, 2, S, S) uint8 .npy (ch0 pooled VIL, ch1 IR) + ``.meta.parquet``; return metadata."""
    raw_dir, out_path = Path(raw_dir), Path(out_path)
    vil_rows = rows[rows["img_type"] == "vil"].set_index("id")
    ir_rows = rows[rows["img_type"] == "ir107"].set_index("id")
    ids = sorted(set(vil_rows.index) & set(ir_rows.index))
    handles: dict[Path, h5py.File] = {}

    def open_h5(name: str) -> tuple[h5py.File, Path]:
        p = raw_dir / name
        if p not in handles:
            if not p.is_file():
                raise FileNotFoundError(p)
            handles[p] = h5py.File(p, "r")
        return handles[p], p

    def read_one(eid: str):
        v, i = vil_rows.loc[eid], ir_rows.loc[eid]
        fv, pv = open_h5(v["file_name"])
        fi, pi = open_h5(i["file_name"])
        vil = _event_row(fv, VIL_KEY, int(v["file_index"]), eid, pv)[int(v["file_index"])]
        ir = _event_row(fi, IR107_KEY, int(i["file_index"]), eid, pi)[int(i["file_index"])]
        if vil.shape[0] != 2 * ir.shape[0] or vil.shape[1] != 2 * ir.shape[1]:
            raise ValueError(f"{eid}: VIL {vil.shape} is not 2x IR {ir.shape} spatially")
        if vil.shape[2] != ir.shape[2]:
            raise ValueError(f"{eid}: VIL T={vil.shape[2]} != IR T={ir.shape[2]}")
        return vil, ir

    mm = None
    try:
        for r, eid in enumerate(ids):
            vil, ir = read_one(eid)
            if mm is None:
                s, t = ir.shape[0], ir.shape[2]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                mm = np.lib.format.open_memmap(
                    out_path, mode="w+", dtype=np.uint8, shape=(len(ids), t, 2, s, s)
                )
            elif ir.shape[2] != mm.shape[1] or ir.shape[0] != mm.shape[3]:
                raise ValueError(f"{eid}: shape {ir.shape} differs from the first event")
            mm[r, :, 0] = pool_vil_2x2(np.moveaxis(vil, -1, 0))
            mm[r, :, 1] = ir_to_uint8(decode_ir(np.moveaxis(ir, -1, 0), ir_scale, ir_offset))
        if mm is not None:
            mm.flush()
    finally:
        for f in handles.values():
            f.close()
        del mm
    cols = ["time_utc", "proj", "llcrnrlat", "llcrnrlon", "urcrnrlat", "urcrnrlon"]
    meta = vil_rows.loc[ids, cols].reset_index()
    meta.insert(0, "row", np.arange(len(ids)))
    meta.to_parquet(out_path.with_suffix(".meta.parquet"))
    return meta


_FNAME = re.compile(r"^SEVIR_([A-Za-z0-9]+)_STORMEVENTS_(\d{4})_(\d{2})(\d{2})_(\d{2})(\d{2})\.h5$")


def parse_sevir_filename(name: str) -> tuple[str, int, date, date]:
    """DATA-08. SEVIR_{TYPE}_STORMEVENTS_{YYYY}_{MMDD}_{MMDD}.h5 -> (type, year, start, end)."""
    m = _FNAME.match(Path(name).name)
    if not m:
        raise ValueError(f"not a SEVIR STORMEVENTS file name: {name!r}")
    typ, y, m1, d1, m2, d2 = m.groups()
    year = int(y)
    try:
        return typ.lower(), year, date(year, int(m1), int(d1)), date(year, int(m2), int(d2))
    except ValueError as exc:
        raise ValueError(f"bad date in SEVIR file name {name!r}") from exc


def apply_ir107_cap(files: Sequence[tuple[str, int]], cap_bytes: int = 60 * GB) -> CapResult:
    """DATA-08. Drop 2018 Jan-Jun files (earliest first) until total <= cap; 2019 never dropped."""
    total = sum(size for _, size in files)
    candidates = []
    for name, size in files:
        _, year, start, end = parse_sevir_filename(name)
        if year == 2018 and (end.month, end.day) <= (6, 30):
            candidates.append((start, name, size))
    dropped: list[str] = []
    for _, name, size in sorted(candidates):
        if total <= cap_bytes:
            break
        dropped.append(name)
        total -= size
    gone = set(dropped)
    return CapResult([n for n, _ in files if n not in gone], dropped, total, total > cap_bytes)


def prepare_sevir(raw: Path, proc: Path, split_path: Path, catalog_path: Path | None = None) -> pd.DataFrame:
    """Preprocess every event id in a frozen ``splits/sevir.json`` (train, val, test, case_holdout) into
    ``proc/sevir/sevir.npy`` (+ ``.meta.parquet``). The raw layout is ``raw/sevir_{vil,ir107}_2018_2019/<year>/``."""
    cat = filter_catalog(read_catalog(catalog_path or raw / "sevir" / "CATALOG.csv"))
    split = json.loads(Path(split_path).read_text())
    wanted = {i["id"] for part in ("train", "val", "test") for i in split[part]}
    wanted |= {i["id"] for part in split["extra"].values() for i in part}
    rows = cat[cat["id"].isin(wanted)].drop_duplicates(["id", "img_type"]).copy()
    rows["file_name"] = [
        f"sevir_{t}_2018_2019/" + f.split("/", 1)[1]
        for t, f in zip(rows["img_type"], rows["file_name"], strict=True)
    ]
    missing = wanted - set(rows["id"])
    if missing:
        raise ValueError(
            f"{len(missing)} split ids have no VIL+IR107 catalog rows, e.g. {sorted(missing)[:3]}"
        )
    return build_sevir_memmap(rows, raw, proc / "sevir" / "sevir.npy")


if __name__ == "__main__":  # python -m outrun.data.sevir  (run with nohup, see CLAUDE.md)
    import time

    from outrun.config import get_paths

    t0 = time.time()
    paths = get_paths()
    meta = prepare_sevir(Path(paths.raw), Path(paths.proc), Path("splits/sevir.json"))
    print(f"wrote {len(meta)} events in {time.time() - t0:.0f} s", flush=True)
