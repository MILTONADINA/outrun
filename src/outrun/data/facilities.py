"""Shelters, hospitals, fire/EMS, schools: normalized facility table (specs/01-data.md DATA-19)."""

from __future__ import annotations

import numpy as np
import pandas as pd

FACILITY_KINDS: tuple[str, ...] = ("shelter", "hospital", "fire_station", "school")


def normalize_facilities(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """DATA-19. -> facility_id, kind, name, lon, lat, capacity (NaN if unknown); rows without coords dropped."""
    if kind not in FACILITY_KINDS:
        raise ValueError(f"unknown facility kind {kind!r}; expected one of {FACILITY_KINDS}")
    by_lower = {str(c).lower(): c for c in df.columns}

    def find(*names: str) -> str | None:
        return next((by_lower[n] for n in names if n in by_lower), None)

    lon_col, lat_col = find("lon", "longitude"), find("lat", "latitude")
    if lon_col is None or lat_col is None:
        raise KeyError("facility table needs lon/lat or longitude/latitude columns")
    name_col, cap_col = find("name"), find("capacity")

    out = pd.DataFrame(
        {
            "name": df[name_col].fillna("").astype(str) if name_col else "",
            "lon": pd.to_numeric(df[lon_col], errors="coerce"),
            "lat": pd.to_numeric(df[lat_col], errors="coerce"),
            "capacity": pd.to_numeric(df[cap_col], errors="coerce").astype(float) if cap_col else np.nan,
        },
        index=df.index,
    )
    out = out.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    out.insert(0, "kind", kind)
    out.insert(0, "facility_id", [f"{kind}-{i}" for i in range(len(out))])
    return out
