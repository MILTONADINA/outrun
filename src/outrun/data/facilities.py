"""Shelters, hospitals, fire/EMS, schools: normalized facility table (specs/01-data.md DATA-19)."""

from __future__ import annotations

import pandas as pd

FACILITY_KINDS: tuple[str, ...] = ("shelter", "hospital", "fire_station", "school")


def normalize_facilities(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """DATA-19. -> facility_id, kind, name, lon, lat, capacity (NaN if unknown); rows without coords dropped."""
    raise NotImplementedError
