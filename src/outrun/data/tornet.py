"""TorNet reader: catalog.csv, NetCDF NEXRAD Level-II chips to fp16 arrays, official split (specs/01-data.md DATA-02)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

# [CANDIDATE] verified in T1/T10 against mit-ll/tornet and real chips.
VARIABLES: tuple[str, ...] = ("DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH")
N_TILTS = 2
FILL_VALUES: dict[str, float] = {"DBZ": -20.0, "VEL": 0.0, "KDP": 0.0, "RHOHV": 0.0, "ZDR": 0.0, "WIDTH": 0.0}
N_CHANNELS = 27  # 12 last frame + 12 diff + missing mask + range km + azimuth deg


@dataclass(frozen=True)
class SplitRule:
    """[CANDIDATE] default; T1 records the verified rule and only these defaults change."""

    mode: Literal["column", "julian"] = "column"
    column: str = "type"
    modulus: int = 20
    train_below: int = 17


def read_tornet_catalog(path: Path | str) -> pd.DataFrame:
    """DATA-02. Requires filename/category/start_time; adds year, julian_day and (if absent) label."""
    df = pd.read_csv(path)
    for col in ("filename", "category", "start_time"):
        if col not in df.columns:
            raise KeyError(col)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["year"] = df["start_time"].dt.year
    df["julian_day"] = df["start_time"].dt.dayofyear
    if "label" not in df.columns:
        df["label"] = (df["category"] == "TOR").astype(int)
    return df


def official_split(
    catalog: pd.DataFrame, rule: SplitRule = SplitRule(), years: Sequence[int] = (2015, 2016, 2019)
) -> pd.DataFrame:
    """DATA-02. Rows of the given years plus a ``split`` column in {train, test} per ``rule``."""
    out = catalog[catalog["year"].isin(list(years))].reset_index(drop=True).copy()
    if rule.mode == "column":
        if rule.column not in out.columns:
            raise KeyError(rule.column)
        split = out[rule.column].astype(str).str.strip().str.lower()
        bad = sorted(set(split) - {"train", "test"})
        if bad:
            raise ValueError(f"unexpected official split values: {bad}")
        out["split"] = split
    else:
        out["split"] = np.where(out["julian_day"] % rule.modulus < rule.train_below, "train", "test")
    return out


def chip_from_dataset(
    ds: xr.Dataset, fill_values: dict[str, float] = FILL_VALUES
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One opened chip -> (float16 (27, A, R), range_km (R,), azimuth_deg (A,)). KeyError on a missing variable."""
    last: list[np.ndarray] = []
    diff: list[np.ndarray] = []
    mask = None
    for var in VARIABLES:
        arr = ds[var].transpose("time", "azimuth", "range", "sweep").values.astype(np.float32)
        for tilt in range(N_TILTS):
            a_last = arr[-1, :, :, tilt]
            a_first = arr[0, :, :, tilt]
            miss = np.isnan(a_last)
            mask = miss if mask is None else (mask | miss)
            fill = fill_values[var]
            f_last = np.where(np.isnan(a_last), fill, a_last)
            f_first = np.where(np.isnan(a_first), fill, a_first)
            last.append(f_last)
            diff.append(f_last - f_first if arr.shape[0] > 1 else np.zeros_like(f_last))
    rng_km = ds["range"].values.astype(np.float32) / 1000.0
    az = ds["azimuth"].values.astype(np.float32)
    n_az, n_rng = last[0].shape
    x = np.empty((N_CHANNELS, n_az, n_rng), dtype=np.float16)
    x[:12] = np.stack(last)
    x[12:24] = np.stack(diff)
    x[24] = mask
    x[25] = np.broadcast_to(rng_km[None, :], (n_az, n_rng))
    x[26] = np.broadcast_to(az[:, None], (n_az, n_rng))
    return x, rng_km, az


def read_chip(path: Path | str, fill_values: dict[str, float] = FILL_VALUES) -> np.ndarray:
    """DATA-02. One NetCDF chip -> float16 (27, A, R) (layout in the spec)."""
    with xr.open_dataset(path) as ds:
        return chip_from_dataset(ds, fill_values)[0]


def convert_chips(
    paths: Sequence[Path | str], out_path: Path | str, fill_values: dict[str, float] = FILL_VALUES
) -> np.memmap:
    """DATA-02. Stack ``read_chip`` results into a float16 .npy memmap (N, 27, A, R)."""
    out = None
    for i, p in enumerate(paths):
        x = read_chip(p, fill_values)
        if out is None:
            out = np.lib.format.open_memmap(
                out_path, mode="w+", dtype=np.float16, shape=(len(paths), *x.shape)
            )
        elif x.shape != out.shape[1:]:
            raise ValueError(f"chip {p} has shape {x.shape}, expected {out.shape[1:]}")
        out[i] = x
    if out is None:
        raise ValueError("no chips to convert")
    out.flush()
    return out
