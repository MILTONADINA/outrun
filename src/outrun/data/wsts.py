"""WildfireSpreadTS GeoTIFF reader + 12-channel NDWS mapper + shift statistics (specs/01-data.md DATA-06)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# [CANDIDATE] WSTS band descriptions and units, verified in T13.
WSTS_TO_NDWS: dict[str, str] = {
    "elevation": "Elevation",
    "th": "Wind direction",
    "vs": "Wind speed",
    "tmmn": "Min temperature",
    "tmmx": "Max temperature",
    "sph": "Specific humidity",
    "pr": "Total precipitation",
    "pdsi": "PDSI",
    "NDVI": "NDVI",
    "erc": "ERC",
    "PrevFireMask": "Active fire",
}
UNIT_CONVERSIONS: dict[str, tuple[float, float]] = {}  # channel -> (scale, offset); identity when absent
WIND_DIR_CHANNEL = "th"
FIRE_CHANNEL = "PrevFireMask"


def read_wsts_day(path: Path | str) -> tuple[dict[str, np.ndarray], float]:
    """DATA-06. {band description: float32 array}, source pixel size in metres; undescribed band -> ValueError."""
    raise NotImplementedError


def map_to_ndws(
    bands: Mapping[str, np.ndarray],
    src_res_m: float,
    population: np.ndarray | None = None,
    target_res_m: float = 1000.0,
) -> np.ndarray:
    """DATA-06. (12, H', W') float32 in NDWS_INPUT_FEATURES order: area mean, circular wind mean, any-fire mask."""
    raise NotImplementedError


def channel_shift_stats(
    wsts: np.ndarray,
    ndws: np.ndarray,
    threshold: float = 0.2,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> pd.DataFrame:
    """DATA-06. Per-channel KS statistic, quantiles (wsts_qNN, ndws_qNN) and ``shifted`` flag."""
    raise NotImplementedError
