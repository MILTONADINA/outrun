"""Frozen train/val/test splits, quarantine of case-study days and split hashes (specs/15-evaluation.md, SPLIT-*).

Every hazard dataset builder (``hazards/<h>/dataset.py``) takes its sample IDs from the ``Split`` built here and
frozen in ``splits/<name>.json`` (names fixed by DESIGN 2.1: sevir, tornet, flood, ice, smoke, ndws, wsts,
quarantine, hashes). Official test splits are never modified: gap days and quarantine only remove items from
train and val.

Builders (one per hazard, SPLIT-01..SPLIT-06) all return a ``Split`` whose sets are sorted by ``SplitItem.id``,
that passes ``validate_split(split, gap_days=1)``, and whose ``extra["case_holdout"]`` holds the input items whose
day is quarantined (SPLIT-08).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SPLIT_NAMES",
    "Split",
    "SplitHashMismatch",
    "SplitItem",
    "SplitLeakageError",
    "assert_valid_split",
    "case_holdout",
    "flood_split",
    "ice_season",
    "ice_split",
    "load_quarantine",
    "quarantine_dates",
    "read_split",
    "sevir_split",
    "smoke_split",
    "split_mask",
    "spatial_holdout_monitors",
    "split_hash",
    "tornet_split",
    "validate_split",
    "verify_hashes",
    "wildfire_split",
    "write_hashes",
    "write_quarantine",
    "write_split",
]

SPLIT_NAMES: tuple[str, ...] = ("sevir", "tornet", "flood", "ice", "smoke", "ndws", "wsts")


class SplitLeakageError(ValueError):
    """A split violates disjointness or the gap-day rule (SPLIT-09)."""


class SplitHashMismatch(RuntimeError):
    """A frozen split file no longer matches ``splits/hashes.json`` (SPLIT-10)."""


@dataclass(frozen=True, order=True)
class SplitItem:
    """One sample, event, chip, bag or fire.

    ``day`` is its UTC calendar day (``None`` when the dataset has no dates, e.g. NDWS). Items that share a
    non-None ``group`` (e.g. a SEVIR event id) must all be in the same set.
    """

    id: str
    day: date | None = None
    group: str | None = None


@dataclass(frozen=True)
class Split:
    """A frozen split. ``extra`` holds named additional evaluation sets (``case_holdout``, ``test_wsts2021``,
    ``test_spatial``); ``meta`` holds JSON-serialisable builder facts (lists, not tuples)."""

    dataset: str
    train: tuple[SplitItem, ...]
    val: tuple[SplitItem, ...]
    test: tuple[SplitItem, ...]
    extra: Mapping[str, tuple[SplitItem, ...]] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ quarantine (SPLIT-07)
def quarantine_dates(case_dates: Iterable[date], radius_days: int = 3) -> frozenset[date]:
    """SPLIT-07. Every case date +- ``radius_days`` (inclusive)."""
    raise NotImplementedError


def write_quarantine(path: str | Path, cases: Mapping[str, date], radius_days: int = 3) -> Path:
    """SPLIT-07. Write ``splits/quarantine.json``: ``{"radius_days", "cases": {id: "YYYY-MM-DD"}, "dates": [...]}``."""
    raise NotImplementedError


def load_quarantine(path: str | Path) -> frozenset[date]:
    """SPLIT-07. Read ``quarantine.json``; ``ValueError`` if ``dates`` disagrees with ``cases`` +- ``radius_days``."""
    raise NotImplementedError


def split_mask(ids: Sequence[str], split: Split, part: str) -> np.ndarray:
    """SPLIT-07. Bool array, True where ``ids[i]`` is an item id of ``split.<part>``.

    ``part`` is "train", "val", "test" or a key of ``split.extra``; anything else -> ValueError. Every hazard dataset
    split factory selects its rows with this helper, so quarantined and test ids never reach train or val.
    """
    raise NotImplementedError


# ------------------------------------------------------------------ case hold-out (SPLIT-08)
def case_holdout(items: Iterable[SplitItem], quarantine: Collection[date]) -> tuple[SplitItem, ...]:
    """SPLIT-08. Items whose day is quarantined, sorted by id (the separately reported case hold-out set)."""
    raise NotImplementedError


# ------------------------------------------------------------------ validation (SPLIT-09)
def validate_split(split: Split, gap_days: int = 1) -> list[str]:
    """SPLIT-09. Human-readable violations; an empty list means the split is clean."""
    raise NotImplementedError


def assert_valid_split(split: Split, gap_days: int = 1) -> None:
    """SPLIT-09. Raise ``SplitLeakageError`` listing the violations, if any."""
    raise NotImplementedError


# ------------------------------------------------------------------ per-hazard builders (SPLIT-01..06)
def sevir_split(events: Iterable[tuple[str, datetime]], quarantine: Collection[date] = ()) -> Split:
    """SPLIT-01. SEVIR storm events (event_id, start time UTC) to the official-cut split."""
    raise NotImplementedError


def tornet_split(
    samples: Iterable[tuple[str, datetime, str]],
    quarantine: Collection[date] = (),
    val_frac: float = 0.1,
    seed: int = 0,
) -> Split:
    """SPLIT-02. TorNet chips (chip_id, time UTC, official split "train"|"test") to train/val/test."""
    raise NotImplementedError


def flood_split(
    samples: Iterable[tuple[str, datetime]],
    quarantine: Collection[date] = (),
    test_year: int = 2025,
    first_year: int = 2021,
) -> Split:
    """SPLIT-03. Flood samples (sample_id, valid time UTC) to a forward-in-time year split."""
    raise NotImplementedError


def wildfire_split(
    ndws: Mapping[str, Iterable[str]],
    wsts_fires: Iterable[tuple[str, date]],
    quarantine: Collection[date] = (),
) -> Split:
    """SPLIT-04. NDWS official files (train/eval/test) plus WSTS 2021 as the out-of-time set."""
    raise NotImplementedError


def ice_season(t: datetime | date) -> int:
    """SPLIT-05. Winter season key: the year the season starts (Jul 1 .. Jun 30); 2024-01-15 -> 2023."""
    raise NotImplementedError


def ice_split(
    bags: Iterable[tuple[str, datetime]],
    quarantine: Collection[date] = (),
    train_seasons: tuple[int, ...] = (2020, 2021, 2022),
    val_season: int = 2023,
    test_seasons: tuple[int, ...] = (2024,),
) -> Split:
    """SPLIT-05. Ice bags (zone_id, 6-h window start UTC) to a forward-in-time winter split."""
    raise NotImplementedError


def spatial_holdout_monitors(
    monitors: Mapping[str, tuple[float, float]],
    frac: float = 0.2,
    block_m: float = 200_000.0,
    seed: int = 0,
) -> frozenset[str]:
    """SPLIT-06 / SMOKE-06. The one block-holdout algorithm: sorted ``(floor(x/block_m), floor(y/block_m))`` blocks,
    reordered by ``default_rng(seed).permutation(n_blocks)``; the shortest prefix holding at least
    ``ceil(frac * len(monitors))`` monitors is held out. ``frac`` outside (0, 1) -> ValueError."""
    raise NotImplementedError


def smoke_split(
    samples: Iterable[tuple[str, datetime]],
    monitors: Mapping[str, tuple[float, float]],
    quarantine: Collection[date] = (),
    holdout_frac: float = 0.2,
    block_m: float = 200_000.0,
    seed: int = 0,
) -> Split:
    """SPLIT-06. Smoke samples (monitor_id, issue time UTC) to a time split plus spatially held-out monitors."""
    raise NotImplementedError


# ------------------------------------------------------------------ freeze + hashes (SPLIT-10)
def write_split(split: Split, path: str | Path) -> Path:
    """SPLIT-10. Canonical JSON (sorted keys, items sorted by id, 2-space indent, trailing newline)."""
    raise NotImplementedError


def read_split(path: str | Path) -> Split:
    """SPLIT-10. Inverse of ``write_split``: ``read_split(write_split(s, p)) == s``."""
    raise NotImplementedError


def split_hash(split: Split) -> str:
    """SPLIT-10. sha256 hex digest of the canonical JSON bytes ``write_split`` would write."""
    raise NotImplementedError


def write_hashes(splits_dir: str | Path) -> dict[str, dict[str, str]]:
    """SPLIT-10. Write ``hashes.json`` for every other ``*.json`` in ``splits_dir`` and return its content."""
    raise NotImplementedError


def verify_hashes(splits_dir: str | Path, strict: bool = False) -> list[str]:
    """SPLIT-10. Names (file stems) whose file is changed, missing or unlisted; ``strict`` raises instead."""
    raise NotImplementedError
