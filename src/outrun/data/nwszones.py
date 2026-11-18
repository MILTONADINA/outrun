"""NWS public forecast zones: ids, polygons, masks (specs/01-data.md DATA-17)."""

from __future__ import annotations

import numpy as np

STATE_FIPS_TO_POSTAL: dict[int, str] = {
    1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC", 12: "FL", 13: "GA",
    15: "HI", 16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD",
    25: "MA", 26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV", 33: "NH", 34: "NJ",
    35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC",
    46: "SD", 47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI", 56: "WY",
}  # fmt: skip


def zone_id(state_fips: int, cz_fips: int) -> str:
    """DATA-17. e.g. (40, 25) -> "OKZ025"; unknown state FIPS -> ValueError."""
    postal = STATE_FIPS_TO_POSTAL.get(int(state_fips))
    if postal is None:
        raise ValueError(f"unknown state FIPS {state_fips}")
    return f"{postal}Z{int(cz_fips):03d}"


def zones_from_frame(gdf):
    """DATA-17. Add zone_id = STATE + "Z" + ZONE (3 digits); dissolve duplicates."""
    gdf = gdf.copy()
    gdf["zone_id"] = (
        gdf["STATE"].astype(str).str.strip() + "Z" + gdf["ZONE"].astype(str).str.strip().str.zfill(3)
    )
    # A zone split into several rows (islands, shared ids) becomes one (multi)polygon.
    return gdf.dissolve(by="zone_id", as_index=False)


def zone_masks(gdf, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """DATA-17. {zone_id: (len(y), len(x)) bool mask of cell centres inside/on the zone geometry}."""
    from outrun.data.spc import rasterize_mask  # same cell-centre "inside or on boundary" rule

    return {zid: rasterize_mask([geom], x, y) for zid, geom in zip(gdf["zone_id"], gdf.geometry, strict=True)}
