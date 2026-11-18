"""Smoke dataset: station samples, HMS satellite input, MASSDEN units, spatial holdout (SMOKE-01..03, SMOKE-06)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from outrun.eval.splits import Split

HMS_DENSITY: dict[str, int] = {"Light": 1, "Medium": 2, "Heavy": 3}
MASSDEN_UNIT_CHANGE: datetime = datetime(2021, 12, 21, 0, 0, tzinfo=UTC)
SMOKE_ISSUE_HOURS: tuple[int, ...] = (0, 6, 12, 18)
HRRR_SMOKE_FIELDS: tuple[str, ...] = ("MASSDEN:8 m", "HPBL", "UGRD:10 m", "VGRD:10 m")
HRRR_SMOKE_LEADS_H: tuple[int, ...] = (1, 6)


@dataclass(frozen=True)
class SmokeFeatureConfig:
    """SMOKE-02. Feature switches; use_hms=False is the no-HMS (no satellite) ablation."""

    use_hms: bool = True


def build_station_samples(
    aqs: pd.DataFrame, issue_times: Sequence[datetime], horizon_h: int = 6, history_h: int = 6
) -> pd.DataFrame:
    """SMOKE-01. One row per (site, issue time): pm_hist (t-5 h..t), target log1p(clip0(PM at t+6 h)), pm_target.

    Drops rows with a missing target or all-NaN history. Issue time not at 00/06/12/18 UTC -> ValueError.
    """
    raise NotImplementedError


def samples_for_split(samples: pd.DataFrame, split: Split, part: str) -> pd.DataFrame:
    """SMOKE-01 / SPLIT-07. Rows whose id ``f"{site_id}|{issue_time.isoformat()}"`` is in ``split.<part>``.

    Uses ``eval.splits.split_mask``; input order kept, index reset.
    """
    raise NotImplementedError


def rasterize_hms(polygons: Sequence[tuple[object, str]], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """SMOKE-02. HMS polygons (shapely Polygon, density) -> uint8 (len(y), len(x)) grid, max on overlap, 0 outside."""
    raise NotImplementedError


def is_smoke_episode(density: int | np.ndarray) -> bool | np.ndarray:
    """SMOKE-02. HMS medium or heavy: density >= 2."""
    raise NotImplementedError


def n_tabular(cfg: SmokeFeatureConfig) -> int:
    """SMOKE-02. 11 tabular features with HMS, 10 without."""
    raise NotImplementedError


def tabular_features(
    pm_hist: np.ndarray, hms_density: float, issue_time: datetime, cfg: SmokeFeatureConfig
) -> np.ndarray:
    """SMOKE-02. [log1p(6 history, NaN -> history mean), HMS (if use_hms), sin/cos hour, sin/cos day-of-year]."""
    raise NotImplementedError


def massden_to_ugm3(values: np.ndarray, valid_time: datetime) -> np.ndarray:
    """SMOKE-03. Before 2021-12-21 00Z: already ug/m3; from then: kg/m3 x 1e9.

    Raises ValueError for post-change raw values > 1e-3, negatives, or a naive valid_time. NaN passes through.
    """
    raise NotImplementedError


def spatial_holdout(
    sites: pd.DataFrame, frac: float = 0.2, block_m: float = 200_000, seed: int = 0
) -> set[str]:
    """SMOKE-06. Hold out whole 200 km blocks; delegates to ``eval.splits.spatial_holdout_monitors`` (SPLIT-06).

    Equals ``set(smoke_split(...).meta["holdout_monitors"])`` for the same monitors and seed.
    """
    raise NotImplementedError
