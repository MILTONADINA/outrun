"""TorNet split: official train/test from the catalog ``type`` column, grouped calendar-day val, quarantine
(TORN-02)."""

from __future__ import annotations

from collections.abc import Collection
from datetime import date

import pandas as pd


def split_catalog(
    catalog: pd.DataFrame,
    val_frac: float = 0.1,
    seed: int = 0,
    quarantine: Collection[date] = (),
    column: str = "type",
) -> pd.Series:
    """Label rows "train" / "val" / "test" / "excluded" (TORN-02). KeyError without ``date`` or ``column``.

    The official split is ``catalog[column]`` (stripped, lower-case; anything but "train"/"test" -> ValueError), as
    shipped in TorNet catalog.csv. Delegates to ``outrun.eval.splits.tornet_split`` (SPLIT-02) on samples
    ``(str(index), date at 00:00 UTC, official)``; rows that land in no set (quarantine or the 1-day gap) are
    "excluded". Never runs a separate split algorithm.
    """
    raise NotImplementedError
