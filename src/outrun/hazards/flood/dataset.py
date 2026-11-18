"""Flood features, uncertain-ring labels, sampling and the training dataset (specs/10-flood.md).

Req IDs: FLOOD-01, FLOOD-02, FLOOD-03, FLOOD-09, FLOOD-11, FLOOD-13, FLOOD-14.
Pure numpy/torch: no imports from other outrun areas, except ``eval.splits.split_mask`` in ``from_split``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from outrun.eval.splits import Split

FLOOD_CHANNELS: tuple[str, ...] = (
    "qpe_1h",
    "qpe_3h",
    "qpe_6h",
    "qpe_24h",
    "ffg_ratio_1h",
    "ffg_ratio_3h",
    "ari_max",
    "elevation",
    "slope",
    "tpi",
    "valid",
)
FLASH_CHANNELS: tuple[str, ...] = ("ffg_ratio_1h", "ffg_ratio_3h", "ari_max")


class MissingHoursError(ValueError):
    """A QPE accumulation window has a missing MRMS hour: the window is skipped (FLOOD-01)."""


@dataclass(frozen=True)
class FloodReport:
    """A Storm Events "Flash Flood" report: begin/end time and begin/end location in planar metres (FLOOD-02)."""

    begin: datetime
    end: datetime
    x0: float
    y0: float
    x1: float
    y1: float


class SampleKind(StrEnum):
    """Patch classes for the balanced sampler (FLOOD-03, FLOOD-13)."""

    POSITIVE = "positive"
    HARD_NEGATIVE = "hard_negative"
    WET = "wet"
    DRY = "dry"


def build_features(
    qpe_1h: np.ndarray,
    qpe_3h: np.ndarray,
    qpe_6h: np.ndarray,
    qpe_24h: np.ndarray,
    ffg_ratio_1h: np.ndarray,
    ffg_ratio_3h: np.ndarray,
    ari_max: np.ndarray,
    elevation: np.ndarray,
    slope: np.ndarray,
    tpi: np.ndarray,
) -> np.ndarray:
    """Stack ten (H, W) inputs into a float32 (11, H, W) array in FLOOD_CHANNELS order (FLOOD-01).

    ``valid`` = 1 where all inputs are finite; non-finite inputs are written as 0. An all-non-finite QPE
    accumulation raises MissingHoursError; shape mismatch or non-2-D raises ValueError. Inputs are not modified.
    """
    raise NotImplementedError


def distance_to_reports(x: np.ndarray, y: np.ndarray, reports: Sequence[FloodReport]) -> np.ndarray:
    """Min Euclidean distance (m) from each (x, y) cell centre to any report segment; inf if none (FLOOD-02)."""
    raise NotImplementedError


def ring_labels(
    x: np.ndarray,
    y: np.ndarray,
    reports: Sequence[FloodReport],
    issued: datetime,
    horizon_h: float = 6.0,
    pre_h: float = 1.0,
    pos_m: float = 5000.0,
    ring_m: float = 15000.0,
) -> np.ndarray:
    """int8 labels: 1 within pos_m of an active report, -1 in (pos_m, ring_m], 0 elsewhere (FLOOD-02).

    Active: [begin - pre_h, end] intersects [issued, issued + horizon_h]. Naive issued or shape mismatch -> ValueError.
    """
    raise NotImplementedError


def near_report(
    x: np.ndarray,
    y: np.ndarray,
    reports: Sequence[FloodReport],
    issued: datetime,
    radius_m: float = 25000.0,
    window_h: float = 3.0,
) -> bool:
    """True iff a report with [begin - window_h, end + window_h] containing issued lies within radius_m (FLOOD-03)."""
    raise NotImplementedError


def sample_kind(
    labels: np.ndarray,
    qpe_1h: np.ndarray,
    report_near: bool,
    hard_neg_mm: float = 25.0,
    wet_mm: float = 1.0,
) -> SampleKind:
    """POSITIVE > HARD_NEGATIVE (max qpe_1h >= 25 mm, no report near) > WET (>= wet_mm) > DRY (FLOOD-03)."""
    raise NotImplementedError


def balanced_indices(kinds: Sequence[SampleKind], n: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted unique int64 indices: n//3 each of POSITIVE/HARD_NEGATIVE/WET (remainder to POSITIVE then
    HARD_NEGATIVE), min(quota, available) per kind, never DRY; n < 0 -> ValueError (FLOOD-13)."""
    raise NotImplementedError


def channel_names(use_flash: bool = True) -> tuple[str, ...]:
    """FLOOD_CHANNELS, or the 8 channels without FLASH_CHANNELS when use_flash is False (FLOOD-09)."""
    raise NotImplementedError


def select_channels(x: np.ndarray, use_flash: bool = True) -> np.ndarray:
    """(11,H,W)/(N,11,H,W) -> same, or the 8 non-FLASH channels in order; channel axis != 11 -> ValueError (FLOOD-09)."""
    raise NotImplementedError


def observed_from_reports(
    reports: Sequence[FloodReport],
    x: np.ndarray,
    y: np.ndarray,
    issued: datetime,
    n_slots: int = 12,
    slot_minutes: int = 15,
    radius_m: float = 5000.0,
) -> np.ndarray:
    """float32 (n_slots, ny, nx): 1 within radius_m of a report active in slot s
    (begin < slot_end and end >= slot_start), else 0 (FLOOD-11)."""
    raise NotImplementedError


class FloodPatchDataset(Dataset):
    """Memory-mapped flood patches: ds[i] -> (x float32 (11|8, H, W), y float32 (1, H, W), -1 kept) (FLOOD-14)."""

    def __init__(self, x_path: str | Path, y_path: str | Path, use_flash: bool = True) -> None:
        raise NotImplementedError

    @classmethod
    def from_split(
        cls,
        x_path: str | Path,
        y_path: str | Path,
        sample_ids: Sequence[str],
        split: Split,
        part: str,
        use_flash: bool = True,
    ) -> FloodPatchDataset:
        """FLOOD-14 / SPLIT-07. Only rows whose id is in ``split.<part>`` (``eval.splits.split_mask``), row order.

        Sets ``ds.ids``. ``len(sample_ids) != N`` -> ValueError.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
