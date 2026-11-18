"""HRRR: Herbie-style .idx byte-range field fetch, in-memory crop, f01 rule, MASSDEN units (specs/01-data.md DATA-03).

Network access is an injected ``range_get(url, start, end) -> bytes``; GRIB decoding (cfgrib) is acceptance-tested.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import xarray as xr

ACCUM_FIELDS: frozenset[str] = frozenset({"PRATE", "FRZR", "APCP", "CFRZR", "CICEP", "CSNOW", "CRAIN"})
# [CANDIDATE level strings; T3 checks a real wrfsfc .idx]
HRRR_ICE_FIELDS: tuple[str, ...] = (
    "TMP:surface", "TMP:2 m above ground", "DPT:2 m above ground", "CFRZR:surface", "CICEP:surface",
    "CSNOW:surface", "CRAIN:surface", "PRATE:surface", "FRZR:surface",
)  # fmt: skip
HRRR_V4_START = datetime(2020, 12, 3, tzinfo=UTC)
ICE_CYCLES_3H: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21)
ICE_CYCLES_6H: tuple[int, ...] = (0, 6, 12, 18)
MASSDEN_UNIT_CHANGE = datetime(2021, 12, 21, tzinfo=UTC)


@dataclass(frozen=True)
class IdxEntry:
    num: int
    start: int
    end: int | None  # inclusive; None for the last message
    init: datetime
    var: str
    level: str
    fcst: str


def parse_idx(text: str) -> list[IdxEntry]:
    """DATA-03. Parse ``num:start:d=YYYYMMDDHH:VAR:LEVEL:FCST:`` lines; end = next start - 1."""
    raise NotImplementedError


def select_fields(entries: Sequence[IdxEntry], fields: Sequence[str]) -> list[IdxEntry]:
    """DATA-03. Entries whose "VAR:LEVEL" equals a requested field; unmatched field -> KeyError."""
    raise NotImplementedError


def byte_ranges(entries: Sequence[IdxEntry]) -> list[tuple[int, int | None]]:
    """DATA-03. Sorted (start, end) ranges with contiguous ones merged."""
    raise NotImplementedError


def range_header(start: int, end: int | None) -> str:
    """DATA-03. "bytes=start-end" or "bytes=start-"."""
    raise NotImplementedError


def fetch_fields(
    grib_url: str,
    idx_text: str,
    fields: Sequence[str],
    range_get: Callable[[str, int, int | None], bytes],
) -> bytes:
    """DATA-03. One range_get per merged range, ascending; concatenated message bytes."""
    raise NotImplementedError


def crop_bbox(da: xr.DataArray, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    """DATA-03. Smallest (y, x) index rectangle containing every cell inside the bbox (2-D lat/lon coords)."""
    raise NotImplementedError


def check_lead(var: str, lead_h: int) -> None:
    """DATA-03. f01 rule: ValueError for lead < 0 or lead 0 on an accumulated/precip field."""
    raise NotImplementedError


def ice_cycle_times(
    first_winter: int = 2020, last_winter: int = 2024, cycles: Sequence[int] = ICE_CYCLES_3H
) -> list[datetime]:
    """DATA-03. Nov 15..Mar 15 (inclusive) of each winter x cycles, from HRRR_V4_START on; sorted."""
    raise NotImplementedError


def choose_ice_cycles(mean_mb_per_message: float, threshold_mb: float = 1.9) -> tuple[int, ...]:
    """DATA-03. 6-hourly cycles if mean MB/message > threshold, else 3-hourly."""
    raise NotImplementedError


def choose_smoke_leads(
    mean_mb_per_message: float, n_days: int = 1096, cap_gb: float = 60.0
) -> tuple[int, ...]:
    """DATA-03. (1, 6) if n_days*4*2*4*mb/1000 <= cap_gb, else (6,)."""
    raise NotImplementedError


def massden_to_ugm3(values: np.ndarray, valid_time: datetime) -> np.ndarray:
    """DATA-03 (same semantics as SMOKE-03). kg/m3 -> ug/m3 (x1e9) at/after 2021-12-21; asserted plausibility."""
    raise NotImplementedError


def check_massden_units(units: str, valid_time: datetime) -> None:
    """DATA-03. Units must contain "kg" at/after MASSDEN_UNIT_CHANGE and must not before; else ValueError."""
    raise NotImplementedError
