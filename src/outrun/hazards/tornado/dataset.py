"""TorNet split: injected official test rule, grouped Julian-day val, quarantine (TORN-02)."""

from __future__ import annotations

from collections.abc import Callable, Collection
from datetime import date

import pandas as pd


def split_catalog(
    catalog: pd.DataFrame,
    is_test: Callable[[date], bool],
    val_frac: float = 0.1,
    seed: int = 0,
    quarantine: Collection[date] = (),
) -> pd.Series:
    """Label rows "train" / "val" / "test" / "excluded" (TORN-02). KeyError without a ``date`` column.

    Delegates to ``outrun.eval.splits.tornet_split`` (SPLIT-02) on samples ``(str(index), date at 00:00 UTC,
    "test" if is_test(date) else "train")``; rows that land in no set (quarantine or the 1-day gap) are
    "excluded". Never runs a separate split algorithm.
    """
    raise NotImplementedError
