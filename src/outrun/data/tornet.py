"""TorNet reader: catalog.csv, NetCDF NEXRAD Level-II chips to fp16 arrays, official split (specs/01-data.md DATA-02)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

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
    raise NotImplementedError


def official_split(
    catalog: pd.DataFrame, rule: SplitRule = SplitRule(), years: Sequence[int] = (2015, 2016, 2019)
) -> pd.DataFrame:
    """DATA-02. Rows of the given years plus a ``split`` column in {train, test} per ``rule``."""
    raise NotImplementedError


def read_chip(path: Path | str, fill_values: dict[str, float] = FILL_VALUES) -> np.ndarray:
    """DATA-02. One NetCDF chip -> float16 (27, A, R) (layout in the spec)."""
    raise NotImplementedError


def convert_chips(
    paths: Sequence[Path | str], out_path: Path | str, fill_values: dict[str, float] = FILL_VALUES
) -> np.memmap:
    """DATA-02. Stack ``read_chip`` results into a float16 .npy memmap (N, 27, A, R)."""
    raise NotImplementedError
