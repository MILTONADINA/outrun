"""Ice dataset: weak labels, (zone, 6-h window) bags and 20-channel bag inputs (specs/12-ice.md ICE-01..ICE-03)."""

from __future__ import annotations

from datetime import datetime
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
    raise NotImplementedError


def window_start(t: datetime) -> datetime:
    """ICE-02. Floor a tz-aware UTC time to its 6-h window start (00/06/12/18 UTC). Naive -> ValueError."""
    raise NotImplementedError


def assign_bag_labels(
    windows: pd.DataFrame, events: pd.DataFrame, t2m_hard_neg_c: float = 3.0
) -> pd.DataFrame:
    """ICE-02. Label (zone, 6-h window) bags from ICE-01 events.

    Positive overlap -> 1; else ambiguous overlap -> dropped; else 0 with hard_negative = t2m_min_k < 273.15 + thr.
    Returns columns zone_id, window_start, label, hard_negative.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
