"""WildfireSpreadTS GeoTIFF reader + 12-channel NDWS mapper + shift statistics (specs/01-data.md DATA-06)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from outrun.data.ndws import NDWS_INPUT_FEATURES

# Band descriptions as written in the 2021 GeoTIFFs (23 bands, verified in T13); matching is case-insensitive.
# Units already agree with NDWS (GRIDMET m/s, K, kg/kg, mm, degrees; NDVI on the MODIS x10000 scale), so the
# conversions are identity (UNIT_CONVERSIONS stays empty); the KS table in NOTES.md checks that claim.
WSTS_TO_NDWS: dict[str, str] = {
    "elevation": "elevation",
    "th": "wind direction",
    "vs": "wind speed",
    "tmmn": "minimum temperature",
    "tmmx": "maximum temperature",
    "sph": "specific humidity",
    "pr": "total precipitation",
    "pdsi": "pdsi",
    "NDVI": "NDVI_last",
    "erc": "energy release component",
    "PrevFireMask": "active fire",
}
UNIT_CONVERSIONS: dict[str, tuple[float, float]] = {}  # channel -> (scale, offset); identity when absent
WIND_DIR_CHANNEL = "th"
FIRE_CHANNEL = "PrevFireMask"


def read_wsts_day(path: Path | str) -> tuple[dict[str, np.ndarray], float]:
    """DATA-06. {band description: float32 array}, source pixel size in metres; undescribed band -> ValueError."""
    import rasterio

    with rasterio.open(path) as src:
        descs = [(d or "").strip() for d in src.descriptions]
        if not all(descs):
            raise ValueError(f"{path}: every band needs a description, got {src.descriptions}")
        data = src.read().astype(np.float32)
        res = float(abs(src.transform.a))
    return {d: data[i] for i, d in enumerate(descs)}, res


def _overlap_matrix(n_src: int, n_dst: int, ratio: float) -> np.ndarray:
    """(n_dst, n_src) overlap lengths (in source pixels) between target cell i = [i*ratio, (i+1)*ratio) and pixel j."""
    lo = np.arange(n_dst)[:, None] * ratio
    j = np.arange(n_src)[None, :]
    return np.clip(np.minimum(lo + ratio, j + 1) - np.maximum(lo, j), 0.0, None)


def _find(bands: Mapping[str, np.ndarray], desc: str) -> np.ndarray:
    for k, v in bands.items():
        if k.strip().lower() == desc.lower():
            return np.asarray(v, dtype=np.float64)
    raise KeyError(f"WSTS band {desc!r} not found; have {sorted(bands)}")


def map_to_ndws(
    bands: Mapping[str, np.ndarray],
    src_res_m: float,
    population: np.ndarray | None = None,
    target_res_m: float = 1000.0,
) -> np.ndarray:
    """DATA-06. (12, H', W') float32 in NDWS_INPUT_FEATURES order: area mean, circular wind mean, any-fire mask."""
    arrays = {
        name: _find(bands, WSTS_TO_NDWS[name]) for name in WSTS_TO_NDWS
    }  # KeyError if a band is missing
    h, w = next(iter(arrays.values())).shape
    ratio = target_res_m / src_res_m
    hh, ww = round(h * src_res_m / target_res_m), round(w * src_res_m / target_res_m)
    if population is not None and np.shape(population) != (hh, ww):
        raise ValueError(f"population must have shape {(hh, ww)}, got {np.shape(population)}")
    wr, wc = _overlap_matrix(h, hh, ratio), _overlap_matrix(w, ww, ratio)

    def wmean(v: np.ndarray) -> np.ndarray:
        """Area-weighted mean ignoring NaN source pixels; NaN where every overlapping pixel is NaN."""
        ok = np.isfinite(v)
        num = wr @ np.where(ok, v, 0.0) @ wc.T
        den = wr @ ok.astype(np.float64) @ wc.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 1e-9, num / den, np.nan)

    out = np.full((len(NDWS_INPUT_FEATURES), hh, ww), np.nan, np.float32)
    for c, name in enumerate(NDWS_INPUT_FEATURES):
        if name == "population":
            if population is not None:
                out[c] = population
        elif name == WIND_DIR_CHANNEL:
            rad = np.radians(arrays[name])
            s, co = wmean(np.sin(rad)), wmean(np.cos(rad))
            out[c] = np.degrees(np.arctan2(s, co)) % 360.0
        elif name == FIRE_CHANNEL:
            burning = (np.nan_to_num(arrays[name], nan=0.0) > 0).astype(np.float64)
            out[c] = (((wr > 1e-9) @ burning) @ (wc > 1e-9).T > 0).astype(np.float32)
        else:
            scale, offset = UNIT_CONVERSIONS.get(name, (1.0, 0.0))
            out[c] = wmean(arrays[name]) * scale + offset
    return out


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import ks_2samp

    return float(ks_2samp(a, b).statistic)


def channel_shift_stats(
    wsts: np.ndarray,
    ndws: np.ndarray,
    threshold: float = 0.2,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> pd.DataFrame:
    """DATA-06. Per-channel KS statistic, quantiles (wsts_qNN, ndws_qNN) and ``shifted`` flag."""
    rows = {}
    for c, name in enumerate(NDWS_INPUT_FEATURES):
        a = np.asarray(wsts[:, c], dtype=np.float64).ravel()
        b = np.asarray(ndws[:, c], dtype=np.float64).ravel()
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        row: dict[str, float | bool] = {}
        if a.size and b.size:
            row["ks"] = _ks(a, b)
            row["shifted"] = bool(row["ks"] > threshold)
        else:
            row["ks"], row["shifted"] = float("nan"), True
        for q in quantiles:
            tag = f"q{int(round(q * 100)):02d}"
            row[f"wsts_{tag}"] = float(np.quantile(a, q)) if a.size else float("nan")
            row[f"ndws_{tag}"] = float(np.quantile(b, q)) if b.size else float("nan")
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")
