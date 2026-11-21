"""Ice dataset: weak labels, (zone, 6-h window) bags and 20-channel bag inputs (specs/12-ice.md ICE-01..ICE-03)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from outrun.eval.splits import Split

ICE_EVENT_TYPES: tuple[str, ...] = ("Ice Storm", "Sleet", "Freezing Fog")
WINTER_EVENT_TYPES: tuple[str, ...] = ("Winter Storm", "Winter Weather")
ICE_KEYWORDS: str = r"freezing rain|freezing drizzle|ice accum|glaze|sleet"  # matched case-insensitively

ICE_FIELDS: tuple[str, ...] = (
    "TMP:surface",
    "TMP:2 m",
    "DPT:2 m",
    "CFRZR",
    "CICEP",
    "CSNOW",
    "CRAIN",
    "PRATE",
    "FRZR",
)
ICE_LEAD_H: int = 1
ICE_CROP: int = 64
N_ICE_CHANNELS: int = 2 * len(ICE_FIELDS) + 2  # 9 fields x (t, t-3 h) + elevation + zone mask = 20
KELVIN_0C: float = 273.15


def label_storm_events(events: pd.DataFrame) -> pd.DataFrame:
    """ICE-01. Keyword weak-label filter on NCEI Storm Events rows.

    Ice types -> label 1; Winter Storm/Weather with an ICE_KEYWORDS narrative match -> 1; without -> -1
    (ambiguous, excluded later, never negative); other types dropped. Missing columns -> KeyError.
    """
    missing = [c for c in ("zone_id", "event_type", "begin", "end", "narrative") if c not in events.columns]
    if missing:
        raise KeyError(f"events is missing columns {missing}")
    etype = events["event_type"].astype(str).str.strip().str.lower()
    is_ice = etype.isin({t.lower() for t in ICE_EVENT_TYPES})
    is_winter = etype.isin({t.lower() for t in WINTER_EVENT_TYPES})
    hit = events["narrative"].fillna("").astype(str).str.contains(ICE_KEYWORDS, case=False, regex=True)
    label = np.where(is_ice | (is_winter & hit), 1, -1)
    out = events[is_ice | is_winter].copy()
    out["label"] = label[(is_ice | is_winter).to_numpy()].astype(int)
    return out.reset_index(drop=True)


def window_start(t: datetime) -> datetime:
    """ICE-02. Floor a tz-aware UTC time to its 6-h window start (00/06/12/18 UTC). Naive -> ValueError."""
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"expected a timezone-aware datetime, got {t!r}")
    t = t.astimezone(UTC)
    return t.replace(hour=t.hour - t.hour % 6, minute=0, second=0, microsecond=0)


def assign_bag_labels(
    windows: pd.DataFrame, events: pd.DataFrame, t2m_hard_neg_c: float = 3.0
) -> pd.DataFrame:
    """ICE-02. Label (zone, 6-h window) bags from ICE-01 events.

    Positive overlap -> 1; else ambiguous overlap -> dropped; else 0 with hard_negative = t2m_min_k < 273.15 + thr.
    Returns columns zone_id, window_start, label, hard_negative.
    """
    ws_all = pd.to_datetime(windows["window_start"])
    if ws_all.dt.tz is None:
        raise ValueError("window_start must be timezone-aware")
    ws_all = ws_all.dt.tz_convert("UTC")
    if not ((ws_all.dt.hour % 6 == 0) & (ws_all.dt.minute == 0) & (ws_all.dt.second == 0)).all():
        raise ValueError("window_start must be a 6-h window start (00/06/12/18 UTC)")
    ws_ns = (ws_all - pd.Timestamp(0, tz="UTC")).to_numpy().astype("timedelta64[ns]").astype(np.int64)
    six_h = np.int64(6 * 3600 * 10**9)

    def ns(col: pd.Series) -> np.ndarray:
        c = pd.to_datetime(col)
        c = c.dt.tz_localize("UTC") if c.dt.tz is None else c.dt.tz_convert("UTC")
        return (c - pd.Timestamp(0, tz="UTC")).to_numpy().astype("timedelta64[ns]").astype(np.int64)

    n = len(windows)
    pos = np.zeros(n, bool)
    amb = np.zeros(n, bool)
    zones = windows["zone_id"].to_numpy()
    ev_begin, ev_end = ns(events["begin"]), ns(events["end"])
    ev_label = events["label"].to_numpy()
    ev_by_zone: dict[str, np.ndarray] = {}
    for z, idx in pd.Series(np.arange(len(events))).groupby(events["zone_id"].to_numpy()):
        ev_by_zone[z] = idx.to_numpy()
    win_by_zone = pd.Series(np.arange(n)).groupby(zones)
    for z, widx in win_by_zone:
        eidx = ev_by_zone.get(z)
        if eidx is None:
            continue
        w = ws_ns[widx.to_numpy()][:, None]
        overlap = (ev_begin[eidx][None, :] < w + six_h) & (ev_end[eidx][None, :] >= w)
        pos[widx.to_numpy()] = (overlap & (ev_label[eidx][None, :] == 1)).any(axis=1)
        amb[widx.to_numpy()] = (overlap & (ev_label[eidx][None, :] == -1)).any(axis=1)
    drop = amb & ~pos
    hard = (~pos) & (windows["t2m_min_k"].to_numpy(dtype=float) < KELVIN_0C + t2m_hard_neg_c)
    out = pd.DataFrame(
        {
            "zone_id": windows["zone_id"].to_numpy(),
            "window_start": windows["window_start"].reset_index(drop=True),
            "label": pos.astype(int),
            "hard_negative": hard,
        }
    )
    return out[~drop].reset_index(drop=True)


def bags_for_split(bags: pd.DataFrame, split: Split, part: str) -> pd.DataFrame:
    """ICE-02 / SPLIT-07. Rows whose id ``f"{zone_id}|{window_start.isoformat()}"`` is in ``split.<part>``.

    Uses ``eval.splits.split_mask``; input order kept, index reset.
    """
    from outrun.eval.splits import split_mask

    ids = [
        f"{z}|{pd.Timestamp(t).isoformat()}"
        for z, t in zip(bags["zone_id"], bags["window_start"], strict=True)
    ]
    return bags[split_mask(ids, split, part)].reset_index(drop=True)


def check_lead(lead_h: int) -> None:
    """ICE-03. Only HRRR f01 is allowed (f00 precipitation is degenerate); any other lead -> ValueError."""
    if lead_h != ICE_LEAD_H:
        raise ValueError(f"ice uses HRRR f{ICE_LEAD_H:02d} only, got f{lead_h:02d}")


def build_bag_input(
    fields_t: np.ndarray,
    fields_tm3: np.ndarray,
    elevation: np.ndarray,
    zone_mask: np.ndarray,
    crop: int = ICE_CROP,
) -> tuple[np.ndarray, np.ndarray]:
    """ICE-03. Build the (20, crop, crop) float32 bag input and its bool zone-mask crop.

    Channels: fields_t (9), fields_tm3 (9), elevation, zone mask. Crop centred on the mask centroid, shifted inside
    the grid, zero-padded bottom/right when the grid is smaller. Empty mask / bad shapes -> ValueError.
    """
    n = len(ICE_FIELDS)
    if fields_t.ndim != 3 or fields_tm3.ndim != 3 or fields_t.shape[0] != n or fields_tm3.shape[0] != n:
        raise ValueError(f"fields must have shape ({n}, H, W)")
    h, w = fields_t.shape[1:]
    if fields_tm3.shape != fields_t.shape or elevation.shape != (h, w) or zone_mask.shape != (h, w):
        raise ValueError("fields_t, fields_tm3, elevation and zone_mask must share one (H, W)")
    zone_mask = np.asarray(zone_mask, bool)
    if not zone_mask.any():
        raise ValueError("empty zone mask")
    rows, cols = np.nonzero(zone_mask)
    cr, cc = round(float(rows.mean())), round(float(cols.mean()))
    r0 = min(max(cr - crop // 2, 0), max(0, h - crop))
    c0 = min(max(cc - crop // 2, 0), max(0, w - crop))
    r1, c1 = min(r0 + crop, h), min(c0 + crop, w)
    x = np.zeros((N_ICE_CHANNELS, crop, crop), np.float32)
    x[:n, : r1 - r0, : c1 - c0] = fields_t[:, r0:r1, c0:c1]
    x[n : 2 * n, : r1 - r0, : c1 - c0] = fields_tm3[:, r0:r1, c0:c1]
    x[2 * n, : r1 - r0, : c1 - c0] = elevation[r0:r1, c0:c1]
    x[2 * n + 1, : r1 - r0, : c1 - c0] = zone_mask[r0:r1, c0:c1]
    return x, x[2 * n + 1] > 0.5


def normalize_bag_input(x: np.ndarray) -> np.ndarray:
    """Fixed unit scaling of a raw ICE-03 input ``(..., 20, H, W)`` for the CNN (no fitted statistics).

    Temperatures: (K - 273.15) / 15. Categorical fields stay 0/1. PRATE (kg m-2 s-1) -> log1p(mm/h), FRZR -> log1p(x).
    Elevation / 1000 m. The zone-mask channel (19) is unchanged, so ``x[..., 19, :, :] > 0.5`` is still the mask.
    """
    out = np.array(x, dtype=np.float32, copy=True)
    for base in (0, len(ICE_FIELDS)):
        out[..., base : base + 3, :, :] = (out[..., base : base + 3, :, :] - 273.15) / 15.0
        out[..., base + 7, :, :] = np.log1p(np.maximum(out[..., base + 7, :, :], 0.0) * 3600.0)
        out[..., base + 8, :, :] = np.log1p(np.maximum(out[..., base + 8, :, :], 0.0))
    out[..., 2 * len(ICE_FIELDS), :, :] /= 1000.0
    return out
