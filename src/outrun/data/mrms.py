"""MRMS QPE 01H Pass2 + FLASH GRIB2 decode, crop, accumulations, FLASH event hours (specs/01-data.md DATA-04, DATA-09)."""

from __future__ import annotations

import gzip
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# [CANDIDATE] product names and key layout; T3 lists s3://noaa-mrms-pds/CONUS/.
QPE_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"
FLASH_PRODUCTS: tuple[str, ...] = (
    "FLASH_QPE_FFG01H_00.00",
    "FLASH_QPE_FFG03H_00.00",
    "FLASH_QPE_ARIMAX_00.00",  # max QPE ARI over all durations, in years: the flood 'ari_max' channel
)

# FLASH QPE-to-FFG products are coded in percent (100 = QPE equals flash-flood guidance, 5 % steps); read_mrms turns
# them into a plain ratio. ARI and QPE products share no part of this prefix and are never rescaled.
FFG_PRODUCT_PREFIX = "FLASH_QPE_FFG"

_NAME_RE = re.compile(r"^MRMS_(?P<product>.+)_(?P<date>\d{8})-(?P<time>\d{6})\.grib2(\.gz)?$")


def _as_utc(t: datetime) -> datetime:
    """Aware datetime in UTC; naive datetimes are ambiguous and rejected."""
    if t.tzinfo is None:
        raise ValueError(f"naive datetime {t}; use a timezone-aware UTC time")
    return t.astimezone(UTC)


def _decode_cfgrib(path: Path) -> xr.DataArray:
    """Default MRMS decoder: the single variable of a one-message GRIB2 file, via cfgrib (no .idx side file)."""
    with xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""}) as ds:
        (name,) = ds.data_vars
        return ds[name].load()


def mrms_key(product: str, valid_time: datetime) -> str:
    """DATA-04. CONUS/{product}/{YYYYMMDD}/MRMS_{product}_{YYYYMMDD}-{HHMMSS}.grib2.gz"""
    t = _as_utc(valid_time)
    day = t.strftime("%Y%m%d")
    return f"CONUS/{product}/{day}/MRMS_{product}_{day}-{t.strftime('%H%M%S')}.grib2.gz"


def parse_mrms_name(name: str) -> tuple[str, datetime]:
    """DATA-04. Key or file name -> (product, UTC valid time); bad name -> ValueError."""
    match = _NAME_RE.match(name.rsplit("/", 1)[-1])
    if match is None:
        raise ValueError(f"not an MRMS file name: {name!r}")
    t = datetime.strptime(match["date"] + match["time"], "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return match["product"], t


def clean_mrms(values: np.ndarray | xr.DataArray) -> np.ndarray | xr.DataArray:
    """DATA-04. Negative (missing/no-coverage) values -> NaN, float32."""
    if isinstance(values, xr.DataArray):
        out = values.astype(np.float32)
        return out.where(~(out < 0))  # where() puts NaN where the condition is False
    arr = np.asarray(values, dtype=np.float32)
    return np.where(arr < 0, np.float32(np.nan), arr)


def is_ffg_product(product: str) -> bool:
    """DATA-04. True for the FLASH QPE-to-FFG ratio products (coded in percent), False for ARI, QPE and the rest."""
    return product.startswith(FFG_PRODUCT_PREFIX)


def ffg_percent_to_ratio(values: np.ndarray | xr.DataArray) -> np.ndarray | xr.DataArray:
    """DATA-04. FLASH FFG percent -> plain ratio (100 % -> 1.0). Missing codes are masked to NaN first."""
    return clean_mrms(values) / np.float32(100.0)


def read_mrms(path: Path | str, decoder: Callable[[Path], xr.DataArray] | None = None) -> xr.DataArray:
    """DATA-04. Gunzip .gz to a temp file (raw untouched), decode (default cfgrib), clean_mrms, remove temp.

    The product comes from the file name: FLASH FFG products (``is_ffg_product``) are also turned from percent into a
    plain ratio. This is the only place that conversion happens. Names that are not MRMS names are only cleaned,
    except names containing ``FLASH_QPE_FFG``: those raise ValueError, because without the product we can't tell
    whether to convert, and unconverted percent values (0-255) would look like huge ratios downstream.
    """
    path = Path(path)
    decode = decoder if decoder is not None else _decode_cfgrib
    try:
        product, _ = parse_mrms_name(path.name)
    except ValueError:
        if FFG_PRODUCT_PREFIX in path.name:
            raise ValueError(
                f"{path.name}: looks like a FLASH FFG file but is not named MRMS_<product>_<YYYYMMDD>-<HHMMSS>"
                ".grib2[.gz], so read_mrms can't convert its percent values; keep the S3 file name"
            ) from None
        product = ""
    convert = ffg_percent_to_ratio if is_ffg_product(product) else clean_mrms
    if path.suffix != ".gz":
        return convert(decode(path))

    fd, tmp_name = tempfile.mkstemp(suffix=".grib2")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, gzip.open(path, "rb") as src:
            shutil.copyfileobj(src, out)
        return convert(decode(tmp).load())  # load before the temp file disappears
    finally:
        tmp.unlink(missing_ok=True)


def crop_latlon(da: xr.DataArray, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    """DATA-04. Cells inside the bbox on 1-D latitude/longitude; longitudes -> [-180, 180); lat order kept."""
    lon_min, lat_min, lon_max, lat_max = bbox
    da = da.assign_coords(longitude=(da["longitude"] + 180.0) % 360.0 - 180.0)
    lat, lon = da["latitude"].values, da["longitude"].values
    rows = np.flatnonzero((lat >= lat_min) & (lat <= lat_max))
    cols = np.flatnonzero((lon >= lon_min) & (lon <= lon_max))
    if rows.size == 0 or cols.size == 0:
        raise ValueError(f"no grid cell inside bbox {bbox}")
    # Latitude order is kept; longitudes are sorted in case the 0..360 -> -180..180 shift wrapped them.
    return da.isel(latitude=rows, longitude=cols).sortby("longitude")


def accumulate(hourly: xr.DataArray, windows_h: Sequence[int] = (1, 3, 6, 24)) -> xr.Dataset:
    """DATA-04. qpe_{h}h = trailing h-hour sums; NaN when any hour is missing or NaN."""
    hourly = hourly.transpose("time", ...)
    times = pd.DatetimeIndex(hourly["time"].values)
    if (times != times.floor("h")).any():
        raise ValueError("accumulate: time values must be on exact hours")
    if len(times) > 1 and not (np.diff(times.asi8) > 0).all():
        raise ValueError("accumulate: time values must be strictly increasing")

    values = np.asarray(hourly.values, dtype=np.float32)  # (time, ...)
    out = {}
    for h in windows_h:
        total = np.zeros_like(values)
        complete = np.ones(len(times), dtype=bool)  # every hour of the window exists in `time`
        for k in range(h):
            pos = times.get_indexer(times - pd.Timedelta(hours=k))  # -1 where that hour is absent
            complete &= pos >= 0
            total += values[np.where(pos >= 0, pos, 0)]  # NaN at a cell propagates through the sum
        total[~complete] = np.nan
        out[f"qpe_{h}h"] = hourly.copy(data=total)
    return xr.Dataset(out)


def flash_hours(
    report_times: Iterable[datetime],
    rainy_hours: Sequence[datetime],
    pad_h: int = 6,
    random_frac: float = 0.2,
    seed: int = 0,
    max_hours: int | None = None,
) -> list[datetime]:
    """DATA-09. +-pad_h event hours around reports, plus seeded random rainy hours, capped; sorted."""
    reports = [_as_utc(t) for t in report_times]
    rainy = sorted({_as_utc(t) for t in rainy_hours})

    event: set[datetime] = set()
    for t in reports:
        hour = t.replace(minute=0, second=0, microsecond=0)
        event.update(hour + k * timedelta(hours=1) for k in range(-pad_h, pad_h + 1))
    event_sorted = sorted(event)

    available = [t for t in rainy if t not in event]
    k = min(round(random_frac * len(event_sorted)), len(available))
    drawn_idx = np.random.default_rng(seed).choice(len(available), k, replace=False) if k > 0 else []
    random_hours = [available[i] for i in drawn_idx]  # in draw order

    if max_hours is not None:
        if len(event_sorted) >= max_hours:
            return event_sorted[:max_hours]
        random_hours = random_hours[: max_hours - len(event_sorted)]  # drop the latest-drawn first
    return sorted(event | set(random_hours))
