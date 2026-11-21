"""2020 census block-group population (specs/01-data.md DATA-19)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def block_group_population(table: pd.DataFrame, pop_col: str = "P1_001N") -> pd.DataFrame:
    """DATA-19. -> geoid (12-char str), population (int64); bad/duplicate rows -> ValueError."""
    for col in ("GEOID", pop_col):
        if col not in table.columns:
            raise KeyError(col)

    geoid = table["GEOID"]
    if pd.api.types.is_numeric_dtype(geoid):
        # Integer GEOIDs lost their leading zero (re-padded below); Int64 avoids "4.01e+11" from float columns.
        geoid = geoid.astype("Int64")
    # data.census.gov writes GEO_IDs like "1500000US401090001001": keep the part after "US".
    geoid = geoid.astype(str).str.strip().str.split("US").str[-1].str.zfill(12)
    if not geoid.str.fullmatch(r"\d{12}").all():
        raise ValueError("GEOID must be a 12-digit block-group code")
    if geoid.duplicated().any():
        raise ValueError(f"duplicate GEOIDs: {sorted(set(geoid[geoid.duplicated()]))[:5]}")

    pop = pd.to_numeric(table[pop_col], errors="coerce")
    if pop.isna().any() or (pop < 0).any() or (pop != pop.round()).any():
        raise ValueError(f"{pop_col} must be a non-negative whole number in every row")
    return pd.DataFrame({"geoid": geoid.to_numpy(), "population": pop.to_numpy().astype(np.int64)})


def read_pl94171_block_groups(zip_path: str | Path) -> pd.DataFrame:
    """Legacy pipe-delimited PL 94-171 state zip -> geoid, population (P0010001), lon, lat (internal point).

    Segment 1 (``*00001*.pl``) holds ``P0010001`` in field 6; the geo header (``*geo2020.pl``) joins by LOGRECNO and
    carries the 12-digit block-group code (summary level 150, whole-area component "00") and the internal point.
    """
    import io
    import zipfile

    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        seg1 = next(n for n in names if n.endswith("00012020.pl"))
        geo_name = next(n for n in names if n.endswith("geo2020.pl"))
        pop = pd.read_csv(
            io.BytesIO(z.read(seg1)), sep="|", header=None, dtype=str, encoding="latin1", usecols=[4, 5]
        )
        pop.columns = ["logrecno", "P1_001N"]
        rows = []
        for line in z.read(geo_name).decode("latin1").splitlines():
            f = line.split("|")
            if f[2] == "150" and f[4] == "00":  # block group, whole (non-split) geography
                rows.append((f[7], f[9], float(f[-5]), float(f[-4])))  # LOGRECNO, GEOCODE, INTPTLAT, INTPTLON
    geo = pd.DataFrame(rows, columns=["logrecno", "GEOID", "lat", "lon"])
    table = geo.merge(pop, on="logrecno", how="inner")
    out = block_group_population(table[["GEOID", "P1_001N"]])
    out["lon"] = table["lon"].to_numpy()
    out["lat"] = table["lat"].to_numpy()
    return out
