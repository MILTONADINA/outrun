"""MRMS QPE 01H Pass2 + FLASH GRIB2 decode, crop, accumulations, FLASH event hours (specs/01-data.md DATA-04, DATA-09)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

# [CANDIDATE] product names and key layout; T3 lists s3://noaa-mrms-pds/CONUS/.
QPE_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"
FLASH_PRODUCTS: tuple[str, ...] = (
    "FLASH_QPE_FFG01H_00.00",
    "FLASH_QPE_FFG03H_00.00",
    "FLASH_QPE_ARI01H_00.00",
)


def mrms_key(product: str, valid_time: datetime) -> str:
    """DATA-04. CONUS/{product}/{YYYYMMDD}/MRMS_{product}_{YYYYMMDD}-{HHMMSS}.grib2.gz"""
    raise NotImplementedError


def parse_mrms_name(name: str) -> tuple[str, datetime]:
    """DATA-04. Key or file name -> (product, UTC valid time); bad name -> ValueError."""
    raise NotImplementedError


def clean_mrms(values: np.ndarray | xr.DataArray) -> np.ndarray | xr.DataArray:
    """DATA-04. Negative (missing/no-coverage) values -> NaN, float32."""
    raise NotImplementedError


def read_mrms(path: Path | str, decoder: Callable[[Path], xr.DataArray] | None = None) -> xr.DataArray:
    """DATA-04. Gunzip .gz to a temp file (raw untouched), decode (default cfgrib), clean_mrms, remove temp."""
    raise NotImplementedError


def crop_latlon(da: xr.DataArray, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    """DATA-04. Cells inside the bbox on 1-D latitude/longitude; longitudes -> [-180, 180); lat order kept."""
    raise NotImplementedError


def accumulate(hourly: xr.DataArray, windows_h: Sequence[int] = (1, 3, 6, 24)) -> xr.Dataset:
    """DATA-04. qpe_{h}h = trailing h-hour sums; NaN when any hour is missing or NaN."""
    raise NotImplementedError


def flash_hours(
    report_times: Iterable[datetime],
    rainy_hours: Sequence[datetime],
    pad_h: int = 6,
    random_frac: float = 0.2,
    seed: int = 0,
    max_hours: int | None = None,
) -> list[datetime]:
    """DATA-09. +-pad_h event hours around reports, plus seeded random rainy hours, capped; sorted."""
    raise NotImplementedError
