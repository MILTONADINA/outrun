"""SEVIR reader: CATALOG filter, VIL (NEXRAD) + GOES-16 IR107 h5 to 2 km uint8 memmaps (specs/01-data.md DATA-01,
DATA-08)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

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
    raise NotImplementedError


def filter_catalog(
    catalog: pd.DataFrame,
    years: Sequence[int] = (2018, 2019),
    img_types: Sequence[str] = ("vil", "ir107"),
    bbox: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """DATA-01. Year/type/storm-event/bbox-intersection filter, then ids having every img_type; sorted (id, img_type)."""
    raise NotImplementedError


def pool_vil_2x2(vil: np.ndarray) -> np.ndarray:
    """DATA-01. 2x2 average pool of the last two axes, rounded half up, uint8; odd sizes -> ValueError."""
    raise NotImplementedError


def decode_ir(raw: np.ndarray, scale: float = IR_RAW_SCALE, offset: float = IR_RAW_OFFSET) -> np.ndarray:
    """DATA-01. Raw IR -> brightness temperature in kelvin (float32)."""
    raise NotImplementedError


def ir_to_uint8(tb_k: np.ndarray) -> np.ndarray:
    """DATA-01. Clip to [180, 320] K, floor((tb-180)/140*255 + 0.5); NaN -> 0."""
    raise NotImplementedError


def build_sevir_memmap(
    rows: pd.DataFrame,
    raw_dir: Path | str,
    out_path: Path | str,
    ir_scale: float = IR_RAW_SCALE,
    ir_offset: float = IR_RAW_OFFSET,
) -> pd.DataFrame:
    """DATA-01. Write (N, T, 2, S, S) uint8 .npy (ch0 pooled VIL, ch1 IR) + ``.meta.parquet``; return metadata."""
    raise NotImplementedError


def parse_sevir_filename(name: str) -> tuple[str, int, date, date]:
    """DATA-08. SEVIR_{TYPE}_STORMEVENTS_{YYYY}_{MMDD}_{MMDD}.h5 -> (type, year, start, end)."""
    raise NotImplementedError


def apply_ir107_cap(files: Sequence[tuple[str, int]], cap_bytes: int = 60 * GB) -> CapResult:
    """DATA-08. Drop 2018 Jan-Jun files (earliest first) until total <= cap; 2019 never dropped."""
    raise NotImplementedError
