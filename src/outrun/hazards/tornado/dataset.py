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
    from datetime import UTC, datetime

    from outrun.eval.splits import tornet_split

    if "date" not in catalog.columns:
        raise KeyError("date")
    if column not in catalog.columns:
        raise KeyError(column)
    official = catalog[column].astype(str).str.strip().str.lower()
    days = pd.to_datetime(catalog["date"]).dt.date
    samples = [
        (str(i), datetime(d.year, d.month, d.day, tzinfo=UTC), o)
        for i, d, o in zip(catalog.index, days, official, strict=True)
    ]
    split = tornet_split(samples, quarantine=quarantine, val_frac=val_frac, seed=seed)
    part = {it.id: name for name in ("train", "val", "test") for it in getattr(split, name)}
    return pd.Series([part.get(str(i), "excluded") for i in catalog.index], index=catalog.index, name="split")
