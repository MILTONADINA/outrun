"""NCEI Storm Events CSV parsing (specs/01-data.md DATA-07). Output feeds ICE-01 and flood labels."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

TZ_OFFSETS: dict[str, int] = {"EST": -5, "CST": -6, "MST": -7, "PST": -8, "AKST": -9, "HST": -10}
YARD_M = 0.9144
MILE_M = 1609.344


def read_storm_events(paths: Path | str | Sequence[Path | str]) -> pd.DataFrame:
    """DATA-07. Storm Events details CSV(.gz) -> normalized columns with UTC begin/end and zone_id."""
    raise NotImplementedError


def filter_event_types(df: pd.DataFrame, types: Sequence[str]) -> pd.DataFrame:
    """DATA-07. Rows whose event_type matches one of ``types`` (case-insensitive, stripped)."""
    raise NotImplementedError
