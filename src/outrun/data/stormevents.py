"""NCEI Storm Events CSV parsing (specs/01-data.md DATA-07). Output feeds ICE-01 and flood labels."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from outrun.data.nwszones import STATE_FIPS_TO_POSTAL, zone_id

TZ_OFFSETS: dict[str, int] = {"EST": -5, "CST": -6, "MST": -7, "PST": -8, "AKST": -9, "HST": -10}
YARD_M = 0.9144
MILE_M = 1609.344


def read_storm_events(paths: Path | str | Sequence[Path | str]) -> pd.DataFrame:
    """DATA-07. Storm Events details CSV(.gz) -> normalized columns with UTC begin/end and zone_id."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    # pandas infers gzip from the .gz suffix; low_memory=False avoids mixed-type chunk warnings on national files.
    raw = pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True)

    offsets = raw["CZ_TIMEZONE"].map(_utc_offset_hours)
    state_fips = pd.to_numeric(raw["STATE_FIPS"], errors="coerce")
    cz_fips = pd.to_numeric(raw["CZ_FIPS"], errors="coerce")
    cz_type = raw["CZ_TYPE"].astype(str).str.strip().str.upper()
    is_land_zone = (cz_type == "Z") & state_fips.isin(list(STATE_FIPS_TO_POSTAL)) & cz_fips.notna()
    zone_ids = [
        zone_id(int(s), int(c)) if ok else None
        for s, c, ok in zip(state_fips, cz_fips, is_land_zone, strict=True)
    ]
    narrative = (
        raw["EVENT_NARRATIVE"].fillna("").astype(str) + " " + raw["EPISODE_NARRATIVE"].fillna("").astype(str)
    )

    def num(col: str) -> pd.Series:
        return pd.to_numeric(raw[col], errors="coerce").astype(float)

    return pd.DataFrame(
        {
            "event_id": raw["EVENT_ID"],
            "episode_id": raw["EPISODE_ID"],
            "event_type": raw["EVENT_TYPE"].astype(str).str.strip(),
            "state_fips": state_fips,
            "cz_type": cz_type,
            "cz_fips": cz_fips,
            "zone_id": pd.Series(zone_ids, dtype=object),
            "begin": _local_to_utc(raw["BEGIN_YEARMONTH"], raw["BEGIN_DAY"], raw["BEGIN_TIME"], offsets),
            "end": _local_to_utc(raw["END_YEARMONTH"], raw["END_DAY"], raw["END_TIME"], offsets),
            "begin_lat": num("BEGIN_LAT"),
            "begin_lon": num("BEGIN_LON"),
            "end_lat": num("END_LAT"),
            "end_lon": num("END_LON"),
            "narrative": narrative.str.strip(),
            "tor_width_m": num("TOR_WIDTH") * YARD_M,
            "tor_length_m": num("TOR_LENGTH") * MILE_M,
        }
    )


def _utc_offset_hours(tz: object) -> int:
    """'CST-6' -> -6, 'AST-4' -> -4; a bare name such as 'CST' is looked up in TZ_OFFSETS; else ValueError."""
    text = str(tz).strip().upper()
    match = re.fullmatch(r"([A-Z]+)(-?\d+)?", text)
    if match is None:
        raise ValueError(f"unknown Storm Events timezone {tz!r}")
    if match.group(2) is not None:
        return int(match.group(2))
    if text in TZ_OFFSETS:
        return TZ_OFFSETS[text]
    raise ValueError(f"unknown Storm Events timezone {tz!r}")


def _local_to_utc(yearmonth: pd.Series, day: pd.Series, hhmm: pd.Series, offset_h: pd.Series) -> pd.Series:
    """YYYYMM + day + HHMM in local standard time -> UTC. UTC = local - offset (CST-6 -> local + 6 h)."""
    ym, hm = yearmonth.astype(int), hhmm.astype(int)
    local = pd.to_datetime(
        pd.DataFrame(
            {
                "year": ym // 100,
                "month": ym % 100,
                "day": day.astype(int),
                "hour": hm // 100,
                "minute": hm % 100,
            }
        )
    )
    return (local - pd.to_timedelta(offset_h, unit="h")).dt.tz_localize("UTC")


def filter_event_types(df: pd.DataFrame, types: Sequence[str]) -> pd.DataFrame:
    """DATA-07. Rows whose event_type matches one of ``types`` (case-insensitive, stripped)."""
    wanted = {t.strip().lower() for t in types}
    keep = df["event_type"].astype(str).str.strip().str.lower().isin(wanted)
    return df[keep].reset_index(drop=True)
