"""2020 census block-group population (specs/01-data.md DATA-19)."""

from __future__ import annotations

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
