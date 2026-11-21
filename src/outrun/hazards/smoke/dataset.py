"""Smoke dataset: station samples, HMS satellite input, MASSDEN units, spatial holdout (SMOKE-01..03, SMOKE-06)."""

from __future__ import annotations

import math
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


def hourly_matrix(aqs: pd.DataFrame, lo_hour: int, hi_hour: int) -> tuple[np.ndarray, np.ndarray]:
    """AQS long table -> (sorted site ids (S,), float32 matrix (S, hi - lo + 1)), NaN where missing.

    Columns of the matrix are hours since 1970-01-01 00 UTC from ``lo_hour`` to ``hi_hour`` inclusive; several rows
    for one (site, hour) are averaged (NaN values ignored).
    """
    hour_ns = 3_600_000_000_000
    stamps = pd.to_datetime(aqs["time"], utc=True).dt.as_unit("ns")
    hrs = stamps.astype("int64").to_numpy() // hour_ns
    sites, site_ix = np.unique(aqs["site_id"].to_numpy(), return_inverse=True)
    width = hi_hour - lo_hour + 1
    pm = aqs["pm25"].to_numpy(dtype=np.float64)
    ok = (hrs >= lo_hour) & (hrs <= hi_hour) & ~np.isnan(pm)
    flat = site_ix[ok].astype(np.int64) * width + (hrs[ok] - lo_hour)
    total = np.bincount(flat, weights=pm[ok], minlength=len(sites) * width)
    count = np.bincount(flat, minlength=len(sites) * width)
    with np.errstate(invalid="ignore", divide="ignore"):
        mat = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return sites, mat.astype(np.float32).reshape(len(sites), width)


def build_station_samples(
    aqs: pd.DataFrame, issue_times: Sequence[datetime], horizon_h: int = 6, history_h: int = 6
) -> pd.DataFrame:
    """SMOKE-01. One row per (site, issue time): pm_hist (t-5 h..t), target log1p(clip0(PM at t+6 h)), pm_target.

    Drops rows with a missing target or all-NaN history. Issue time not at 00/06/12/18 UTC -> ValueError.
    Vectorised: the AQS table becomes a dense (site x hour) matrix once, then every row is an index lookup.
    """
    times = []
    for t in issue_times:
        if t.tzinfo is None or t.utcoffset() is None:
            raise ValueError("issue times must be timezone-aware")
        t = t.astimezone(UTC)
        if t.hour not in SMOKE_ISSUE_HOURS or t.minute or t.second or t.microsecond:
            raise ValueError(f"issue time {t.isoformat()} is not 00/06/12/18 UTC")
        times.append(t)
    cols = ["site_id", "issue_time", "pm_hist", "target", "pm_target"]
    if not times or aqs.empty:
        return pd.DataFrame({c: [] for c in cols})
    hour_ns = 3_600_000_000_000
    issue_h = np.array([pd.Timestamp(t).as_unit("ns").value // hour_ns for t in times], dtype=np.int64)
    lo, hi = int(issue_h.min()) - (history_h - 1), int(issue_h.max()) + horizon_h
    sites, mat = hourly_matrix(aqs, lo, hi)
    ti = issue_h - lo  # (n_issue,)
    hist = mat[:, ti[:, None] + np.arange(-(history_h - 1), 1)[None, :]]  # (S, T, history)
    tgt = mat[:, ti + horizon_h]  # (S, T)
    keep = ~np.isnan(tgt) & ~np.all(np.isnan(hist), axis=2)
    si, tj = np.nonzero(keep)
    pm_target = np.maximum(tgt[si, tj], 0.0).astype(np.float32)
    out = pd.DataFrame(
        {
            "site_id": sites[si],
            "issue_time": pd.to_datetime([times[j] for j in tj], utc=True),
            "pm_hist": list(hist[si, tj]),
            "target": np.log1p(pm_target).astype(np.float32),
            "pm_target": pm_target,
        }
    )
    out = out.sort_values(["issue_time", "site_id"], kind="stable")
    return out.reset_index(drop=True)


def samples_for_split(samples: pd.DataFrame, split: Split, part: str) -> pd.DataFrame:
    """SMOKE-01 / SPLIT-07. Rows whose id ``f"{site_id}|{issue_time.isoformat()}"`` is in ``split.<part>``.

    Uses ``eval.splits.split_mask``; input order kept, index reset.
    """
    from outrun.eval.splits import split_mask

    ids = [
        f"{z}|{pd.Timestamp(t).isoformat()}"
        for z, t in zip(samples["site_id"], samples["issue_time"], strict=True)
    ]
    return samples[split_mask(ids, split, part)].reset_index(drop=True)


def rasterize_hms(polygons: Sequence[tuple[object, str]], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """SMOKE-02. HMS polygons (shapely Polygon, density) -> uint8 (len(y), len(x)) grid, max on overlap, 0 outside."""
    import shapely

    levels = []
    for poly, dens in polygons:
        key = str(dens).strip().title()
        if key not in HMS_DENSITY:
            raise ValueError(f"unknown HMS density {dens!r}")
        levels.append((poly, HMS_DENSITY[key]))
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    grid = np.zeros((len(y), len(x)), dtype=np.uint8)
    if not levels:
        return grid
    xx, yy = np.meshgrid(x, y)
    for poly, level in levels:
        inside = shapely.intersects_xy(poly, xx, yy)  # inside or on the boundary
        grid[inside] = np.maximum(grid[inside], level)
    return grid


def is_smoke_episode(density: int | np.ndarray) -> bool | np.ndarray:
    """SMOKE-02. HMS medium or heavy: density >= 2."""
    if np.ndim(density) == 0:
        return bool(density >= 2)
    return np.asarray(density) >= 2


def n_tabular(cfg: SmokeFeatureConfig) -> int:
    """SMOKE-02. 11 tabular features with HMS, 10 without."""
    return 11 if cfg.use_hms else 10


def tabular_features(
    pm_hist: np.ndarray, hms_density: float, issue_time: datetime, cfg: SmokeFeatureConfig
) -> np.ndarray:
    """SMOKE-02. [log1p(6 history, NaN -> history mean), HMS (if use_hms), sin/cos hour, sin/cos day-of-year]."""
    t = issue_time.astimezone(UTC) if issue_time.tzinfo is not None else issue_time
    hist = np.clip(np.asarray(pm_hist, dtype=np.float64), 0.0, None)
    if np.isnan(hist).any():
        hist = np.where(np.isnan(hist), np.nanmean(hist) if not np.isnan(hist).all() else 0.0, hist)
    hour_angle = 2.0 * math.pi * (t.hour + t.minute / 60.0) / 24.0
    doy_angle = 2.0 * math.pi * t.timetuple().tm_yday / 365.25
    feats = list(np.log1p(hist))
    if cfg.use_hms:
        feats.append(float(hms_density))
    feats += [math.sin(hour_angle), math.cos(hour_angle), math.sin(doy_angle), math.cos(doy_angle)]
    return np.asarray(feats, dtype=np.float32)


def massden_to_ugm3(values: np.ndarray, valid_time: datetime) -> np.ndarray:
    """SMOKE-03. Before 2021-12-21 00Z: already ug/m3; from then: kg/m3 x 1e9.

    Raises ValueError for post-change raw values > 1e-3, negatives, or a naive valid_time. NaN passes through.
    """
    if valid_time.tzinfo is None or valid_time.utcoffset() is None:
        raise ValueError("valid_time must be timezone-aware")
    v = np.asarray(values, dtype=np.float64)
    if np.nanmin(v, initial=0.0) < 0.0:
        raise ValueError("negative MASSDEN values")
    if valid_time >= MASSDEN_UNIT_CHANGE:
        if np.nanmax(v, initial=0.0) > 1e-3:
            raise ValueError("MASSDEN > 1e-3 kg/m3 after 2021-12-21: the values look like ug/m3 already")
        v = v * 1e9
    return v.astype(np.float32)


def spatial_holdout(
    sites: pd.DataFrame, frac: float = 0.2, block_m: float = 200_000, seed: int = 0
) -> set[str]:
    """SMOKE-06. Hold out whole 200 km blocks; delegates to ``eval.splits.spatial_holdout_monitors`` (SPLIT-06).

    Equals ``set(smoke_split(...).meta["holdout_monitors"])`` for the same monitors and seed.
    """
    from outrun.eval.splits import spatial_holdout_monitors

    monitors = {
        str(s): (float(x), float(y))
        for s, x, y in zip(sites["site_id"], sites["x_m"], sites["y_m"], strict=True)
    }
    return set(spatial_holdout_monitors(monitors, frac, block_m, seed))


def tabular_features_batch(
    pm_hist: np.ndarray,
    hms_density: np.ndarray,
    issue_times: Sequence[datetime] | pd.DatetimeIndex,
    cfg: SmokeFeatureConfig,
) -> np.ndarray:
    """Vectorised ``tabular_features`` for N rows: (N, 6), (N,), N times -> (N, n_tabular) float32 (same numbers)."""
    hist = np.clip(np.asarray(pm_hist, dtype=np.float64), 0.0, None)
    n_ok = (~np.isnan(hist)).sum(axis=1)
    mean = np.where(n_ok > 0, np.nansum(hist, axis=1) / np.maximum(n_ok, 1), 0.0)
    hist = np.where(np.isnan(hist), mean[:, None], hist)
    idx = pd.DatetimeIndex(issue_times)
    hour_angle = 2.0 * np.pi * (idx.hour.to_numpy() + idx.minute.to_numpy() / 60.0) / 24.0
    doy_angle = 2.0 * np.pi * idx.dayofyear.to_numpy() / 365.25
    cols = [np.log1p(hist)]
    if cfg.use_hms:
        cols.append(np.asarray(hms_density, dtype=np.float64)[:, None])
    cols.append(
        np.stack([np.sin(hour_angle), np.cos(hour_angle), np.sin(doy_angle), np.cos(doy_angle)], axis=1)
    )
    return np.concatenate(cols, axis=1).astype(np.float32)
