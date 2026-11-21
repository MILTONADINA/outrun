"""Shelters, hospitals, fire/EMS, schools: normalized facility table (specs/01-data.md DATA-19)."""

from __future__ import annotations

from pathlib import Path

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


# HIFLD layer directory -> (facility kind, columns renamed to the DATA-19 names).
_LAYERS = {
    "hifld_shelters": ("shelter", {"shelter_name": "name", "evacuation_capacity": "capacity"}),
    "hifld_hospitals": ("hospital", {"BEDS": "capacity"}),
    "hifld_fire_ems_stations": ("fire_station", {}),
    "hifld_public_schools": ("school", {"ENROLLMENT": "capacity"}),
    "hifld_private_schools": ("school", {"ENROLLMENT": "capacity"}),
}


def load_hifld_facilities(raw_dir: str | Path, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    """Read the HIFLD parquet layers under ``raw_dir``, keep points inside ``bbox`` (lon0, lat0, lon1, lat1), and
    normalise each kind. Capacity: shelters ``evacuation_capacity``, hospitals ``BEDS``, schools ``ENROLLMENT``
    (non-positive values and the -999 sentinel become NaN); fire stations have none. -> DATA-19 table."""
    import glob

    lon0, lat0, lon1, lat1 = bbox
    frames = []
    for layer, (kind, rename) in _LAYERS.items():
        files = sorted(glob.glob(str(Path(raw_dir) / layer / "*.parquet")))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        lower = {c.lower() for c in df.columns}
        if not {"lat", "latitude"} & lower:  # fire/EMS layer has only a point geometry: use its bbox corner
            df = df.assign(
                longitude=df["bbox"].map(lambda b: b["xmin"]), latitude=df["bbox"].map(lambda b: b["ymin"])
            )
        lat_col = next(c for c in df.columns if c.lower() in ("lat", "latitude"))
        lon_col = next(c for c in df.columns if c.lower() in ("lon", "longitude"))
        df = df[df[lon_col].between(lon0, lon1) & df[lat_col].between(lat0, lat1)]
        keep = [
            c for c in df.columns if c.lower() in ("name", "lon", "lat", "longitude", "latitude", "capacity")
        ]
        norm = normalize_facilities(df[keep], kind)
        norm.loc[norm["capacity"] <= 0, "capacity"] = np.nan
        frames.append(norm)
    out = pd.concat(frames, ignore_index=True)
    out["facility_id"] = [f"{k}-{i}" for i, k in enumerate(out["kind"])]  # unique across layers
    return out
