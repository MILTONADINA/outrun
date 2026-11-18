"""HRRR: Herbie-style .idx byte-range field fetch, in-memory crop, f01 rule, MASSDEN units (specs/01-data.md DATA-03).

Network access is an injected ``range_get(url, start, end) -> bytes``; GRIB decoding (cfgrib) is acceptance-tested.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

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
    entries: list[tuple[int, int, datetime, str, str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 6 or not parts[2].startswith("d="):
            raise ValueError(f"malformed .idx line: {line!r}")
        try:
            num, start = int(parts[0]), int(parts[1])
            init = datetime.strptime(parts[2][2:], "%Y%m%d%H").replace(tzinfo=UTC)
        except ValueError as exc:
            raise ValueError(f"malformed .idx line: {line!r}") from exc
        # parts[6:] are optional trailing fields (ignored).
        entries.append((num, start, init, parts[3], parts[4], parts[5]))

    out: list[IdxEntry] = []
    for i, (num, start, init, var, level, fcst) in enumerate(entries):
        # A message ends one byte before the next one starts; the last one runs to end of file.
        end = entries[i + 1][1] - 1 if i + 1 < len(entries) else None
        out.append(IdxEntry(num, start, end, init, var, level, fcst))
    return out


def select_fields(entries: Sequence[IdxEntry], fields: Sequence[str]) -> list[IdxEntry]:
    """DATA-03. Entries whose "VAR:LEVEL" equals a requested field; unmatched field -> KeyError."""
    wanted = list(dict.fromkeys(fields))  # unique, order kept
    for field in wanted:
        if not any(f"{e.var}:{e.level}" == field for e in entries):
            raise KeyError(f"HRRR field not in .idx: {field}")
    return [e for e in entries if f"{e.var}:{e.level}" in wanted]


def byte_ranges(entries: Sequence[IdxEntry]) -> list[tuple[int, int | None]]:
    """DATA-03. Sorted (start, end) ranges with contiguous ones merged."""
    ranges: list[tuple[int, int | None]] = []
    for e in sorted(entries, key=lambda e: e.start):
        if ranges:
            prev_start, prev_end = ranges[-1]
            if prev_end is not None and prev_end + 1 == e.start:
                ranges[-1] = (prev_start, e.end)  # contiguous: extend the previous range
                continue
        ranges.append((e.start, e.end))
    return ranges


def range_header(start: int, end: int | None) -> str:
    """DATA-03. "bytes=start-end" or "bytes=start-"."""
    return f"bytes={start}-" if end is None else f"bytes={start}-{end}"


def fetch_fields(
    grib_url: str,
    idx_text: str,
    fields: Sequence[str],
    range_get: Callable[[str, int, int | None], bytes],
) -> bytes:
    """DATA-03. One range_get per merged range, ascending; concatenated message bytes."""
    entries = select_fields(parse_idx(idx_text), fields)
    return b"".join(range_get(grib_url, start, end) for start, end in byte_ranges(entries))


def crop_bbox(da: xr.DataArray, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    """DATA-03. Smallest (y, x) index rectangle containing every cell inside the bbox (2-D lat/lon coords).

    The crop keeps HRRR's own conventions: row 0 is the southern edge (MRMS crops are north-up), and the
    ``longitude`` coordinate stays in 0..360 (only the inside-mask uses -180..180). A regrid onto GRID must
    convert longitudes with ``(lon + 180) % 360 - 180`` and read the y orientation from the coordinates.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    lat = np.asarray(da["latitude"].values)
    lon = (np.asarray(da["longitude"].values) + 180.0) % 360.0 - 180.0  # 0..360 -> -180..180
    inside = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    if not inside.any():
        raise ValueError(f"no grid cell inside bbox {bbox}")
    rows = np.flatnonzero(inside.any(axis=1))
    cols = np.flatnonzero(inside.any(axis=0))
    ydim, xdim = da["latitude"].dims
    return da.isel({ydim: slice(rows[0], rows[-1] + 1), xdim: slice(cols[0], cols[-1] + 1)})


def check_lead(var: str, lead_h: int) -> None:
    """DATA-03. f01 rule: ValueError for lead < 0 or lead 0 on an accumulated/precip field."""
    if lead_h < 0:
        raise ValueError(f"negative lead {lead_h}")
    if lead_h == 0 and var in ACCUM_FIELDS:
        raise ValueError(f"{var} is accumulated/precip: zero or degenerate at f00, use f01 or later")
    return None


def ice_cycle_times(
    first_winter: int = 2020, last_winter: int = 2024, cycles: Sequence[int] = ICE_CYCLES_3H
) -> list[datetime]:
    """DATA-03. Nov 15..Mar 15 (inclusive) of each winter x cycles, from HRRR_V4_START on; sorted."""
    times: set[datetime] = set()
    for winter in range(first_winter, last_winter + 1):
        day = datetime(winter, 11, 15, tzinfo=UTC)
        last_day = datetime(winter + 1, 3, 15, tzinfo=UTC)
        while day <= last_day:
            for hour in cycles:
                t = day + timedelta(hours=hour)
                if t >= HRRR_V4_START:
                    times.add(t)
            day += timedelta(days=1)
    return sorted(times)


def choose_ice_cycles(mean_mb_per_message: float, threshold_mb: float = 1.9) -> tuple[int, ...]:
    """DATA-03. 6-hourly cycles if mean MB/message > threshold, else 3-hourly."""
    return ICE_CYCLES_6H if mean_mb_per_message > threshold_mb else ICE_CYCLES_3H


def choose_smoke_leads(
    mean_mb_per_message: float, n_days: int = 1096, cap_gb: float = 80.0
) -> tuple[int, ...]:
    """DATA-03. (1, 6) if n_days*4*2*4*mb/1000 <= cap_gb, else (6,). Cap 80 GB (T3 measured 73.5 GB for f01+f06)."""
    cycles_per_day, n_fields = 4, 4
    both_leads_gb = n_days * cycles_per_day * 2 * n_fields * mean_mb_per_message / 1000.0
    return (1, 6) if both_leads_gb <= cap_gb else (6,)


def massden_to_ugm3(values: np.ndarray, valid_time: datetime) -> np.ndarray:
    """DATA-03 (same semantics as SMOKE-03). kg/m3 -> ug/m3 (x1e9) at/after 2021-12-21; asserted plausibility."""
    _require_aware(valid_time)
    raw = np.asarray(values, dtype=np.float64)
    finite = raw[~np.isnan(raw)]
    if (finite < 0).any():
        raise ValueError("negative MASSDEN value")
    if valid_time < MASSDEN_UNIT_CHANGE:
        return raw.astype(np.float32)  # already ug/m3
    if (finite > 1e-3).any():
        raise ValueError("MASSDEN after 2021-12-21 should be kg/m3 (<= 1e-3); got a larger value")
    return (raw * 1e9).astype(np.float32)


def check_massden_units(units: str, valid_time: datetime) -> None:
    """DATA-03. Units must contain "kg" at/after MASSDEN_UNIT_CHANGE and must not before; else ValueError.

    cfgrib reports MASSDEN with units "unknown" (its GRIB tables lack the parameter). That string says nothing, so it
    is accepted: the unit is then decided by the valid-time date and the magnitude guard in ``massden_to_ugm3``.
    """
    _require_aware(valid_time)
    if units.strip().lower() == "unknown":
        return None
    has_kg = "kg" in units.lower()
    if valid_time >= MASSDEN_UNIT_CHANGE and not has_kg:
        raise ValueError(f"MASSDEN units {units!r} at {valid_time}: expected kg/m3")
    if valid_time < MASSDEN_UNIT_CHANGE and has_kg:
        raise ValueError(f"MASSDEN units {units!r} at {valid_time}: expected ug/m3")
    return None


def _require_aware(t: datetime) -> None:
    if t.tzinfo is None:
        raise ValueError(f"naive datetime {t}; use a timezone-aware UTC time")


def decode_grib_bytes(data: bytes) -> list[xr.Dataset]:
    """Decode concatenated GRIB2 messages (the output of ``fetch_fields``) with cfgrib.

    cfgrib reads files, not bytes, so the messages go to a temporary file that is removed afterwards. HRRR fields sit
    on different level types (surface, 2 m, 8 m), which cfgrib cannot put in one Dataset, so ``cfgrib.open_datasets``
    returns one Dataset per level type. Everything is loaded into memory before the temp file is deleted.
    """
    import os
    import tempfile

    import cfgrib

    fd, tmp = tempfile.mkstemp(suffix=".grib2")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        datasets = cfgrib.open_datasets(tmp, backend_kwargs={"indexpath": ""})
        return [ds.load() for ds in datasets]
    finally:
        os.remove(tmp)
