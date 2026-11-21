"""Flood features, uncertain-ring labels, sampling and the training dataset (specs/10-flood.md).

Req IDs: FLOOD-01, FLOOD-02, FLOOD-03, FLOOD-09, FLOOD-11, FLOOD-13, FLOOD-14.
Pure numpy/torch: no imports from other outrun areas, except ``eval.splits.split_mask`` in ``from_split``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
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


def _require_aware(t: datetime) -> None:
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError(f"expected a timezone-aware datetime, got naive {t!r}")


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
    arrays = [qpe_1h, qpe_3h, qpe_6h, qpe_24h, ffg_ratio_1h, ffg_ratio_3h, ari_max, elevation, slope, tpi]
    arrays = [np.asarray(a) for a in arrays]
    shape = arrays[0].shape
    if len(shape) != 2 or any(a.shape != shape for a in arrays):
        raise ValueError(f"build_features needs ten 2-D arrays of one shape, got {[a.shape for a in arrays]}")
    for name, a in zip(FLOOD_CHANNELS[:4], arrays[:4], strict=True):
        if not np.isfinite(a).any():
            raise MissingHoursError(f"{name} is entirely missing: an MRMS hour of the window is absent")
    finite = np.stack([np.isfinite(a) for a in arrays])
    out = np.zeros((len(FLOOD_CHANNELS), *shape), dtype=np.float32)
    for k, (a, ok) in enumerate(zip(arrays, finite, strict=True)):
        out[k] = np.where(ok, a, 0.0)
    out[-1] = finite.all(axis=0)
    return out


def distance_to_reports(x: np.ndarray, y: np.ndarray, reports: Sequence[FloodReport]) -> np.ndarray:
    """Min Euclidean distance (m) from each (x, y) cell centre to any report segment; inf if none (FLOOD-02)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x and y shapes differ: {x.shape} vs {y.shape}")
    best = np.full(x.shape, np.inf)
    for r in reports:
        dx, dy = r.x1 - r.x0, r.y1 - r.y0
        seg2 = dx * dx + dy * dy
        if seg2 == 0.0:
            d = np.hypot(x - r.x0, y - r.y0)
        else:
            t = np.clip(((x - r.x0) * dx + (y - r.y0) * dy) / seg2, 0.0, 1.0)
            d = np.hypot(x - (r.x0 + t * dx), y - (r.y0 + t * dy))
        best = np.minimum(best, d)
    return best


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
    _require_aware(issued)
    x = np.asarray(x)
    if x.shape != np.asarray(y).shape:
        raise ValueError(f"x and y shapes differ: {x.shape} vs {np.asarray(y).shape}")
    lo, hi = issued, issued + timedelta(hours=horizon_h)
    active = [r for r in reports if r.begin - timedelta(hours=pre_h) <= hi and r.end >= lo]
    out = np.zeros(x.shape, dtype=np.int8)
    if not active:
        return out
    d = distance_to_reports(x, y, active)
    out[d <= ring_m] = -1
    out[d <= pos_m] = 1
    return out


def near_report(
    x: np.ndarray,
    y: np.ndarray,
    reports: Sequence[FloodReport],
    issued: datetime,
    radius_m: float = 25000.0,
    window_h: float = 3.0,
) -> bool:
    """True iff a report with [begin - window_h, end + window_h] containing issued lies within radius_m (FLOOD-03)."""
    _require_aware(issued)
    w = timedelta(hours=window_h)
    close = [r for r in reports if r.begin - w <= issued <= r.end + w]
    if not close:
        return False
    return bool((distance_to_reports(x, y, close) <= radius_m).any())


def sample_kind(
    labels: np.ndarray,
    qpe_1h: np.ndarray,
    report_near: bool,
    hard_neg_mm: float = 25.0,
    wet_mm: float = 1.0,
) -> SampleKind:
    """POSITIVE > HARD_NEGATIVE (max qpe_1h >= 25 mm, no report near) > WET (>= wet_mm) > DRY (FLOOD-03)."""
    labels = np.asarray(labels)
    if (labels == 1).any():
        return SampleKind.POSITIVE
    q = np.asarray(qpe_1h, dtype=np.float64)
    peak = float(np.nanmax(q)) if np.isfinite(q).any() else -np.inf
    if peak >= hard_neg_mm and not report_near:
        return SampleKind.HARD_NEGATIVE
    if peak >= wet_mm:
        return SampleKind.WET
    return SampleKind.DRY


def balanced_indices(kinds: Sequence[SampleKind], n: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted unique int64 indices: n//3 each of POSITIVE/HARD_NEGATIVE/WET (remainder to POSITIVE then
    HARD_NEGATIVE), min(quota, available) per kind, never DRY; n < 0 -> ValueError (FLOOD-13)."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    kinds = list(kinds)
    base, extra = divmod(n, 3)
    quota = {
        SampleKind.POSITIVE: base + (extra >= 1),
        SampleKind.HARD_NEGATIVE: base + (extra >= 2),
        SampleKind.WET: base,
    }
    picked: list[np.ndarray] = []
    for kind, q in quota.items():
        pool = np.array([i for i, k in enumerate(kinds) if k == kind], dtype=np.int64)
        take = min(q, pool.size)
        if take:
            picked.append(rng.choice(pool, size=take, replace=False))
    if not picked:
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.concatenate(picked)).astype(np.int64)


def channel_names(use_flash: bool = True) -> tuple[str, ...]:
    """FLOOD_CHANNELS, or the 8 channels without FLASH_CHANNELS when use_flash is False (FLOOD-09)."""
    if use_flash:
        return FLOOD_CHANNELS
    return tuple(c for c in FLOOD_CHANNELS if c not in FLASH_CHANNELS)


def select_channels(x: np.ndarray, use_flash: bool = True) -> np.ndarray:
    """(11,H,W)/(N,11,H,W) -> same, or the 8 non-FLASH channels in order; channel axis != 11 -> ValueError (FLOOD-09)."""
    x = np.asarray(x)
    if x.ndim not in (3, 4) or x.shape[-3] != len(FLOOD_CHANNELS):
        raise ValueError(f"expected (11, H, W) or (N, 11, H, W), got {x.shape}")
    if use_flash:
        return x
    keep = [i for i, c in enumerate(FLOOD_CHANNELS) if c not in FLASH_CHANNELS]
    return x[..., keep, :, :]


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
    _require_aware(issued)
    xx, yy = np.meshgrid(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    out = np.zeros((n_slots, len(y), len(x)), dtype=np.float32)
    step = timedelta(minutes=slot_minutes)
    for s in range(n_slots):
        start = issued + s * step
        end = start + step
        active = [r for r in reports if r.begin < end and r.end >= start]
        if active:
            out[s] = distance_to_reports(xx, yy, active) <= radius_m
    return out


class FloodPatchDataset(Dataset):
    """Memory-mapped flood patches: ds[i] -> (x float32 (11|8, H, W), y float32 (1, H, W), -1 kept) (FLOOD-14)."""

    def __init__(self, x_path: str | Path, y_path: str | Path, use_flash: bool = True) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError(f"x has {self.x.shape[0]} patches but y has {self.y.shape[0]}")
        self.use_flash = use_flash
        self.rows = np.arange(self.x.shape[0])
        self.ids: list[str] = []

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
        from outrun.eval.splits import split_mask

        ds = cls(x_path, y_path, use_flash)
        if len(sample_ids) != len(ds.x):
            raise ValueError(f"{len(sample_ids)} sample ids for {len(ds.x)} patches")
        ds.rows = np.flatnonzero(split_mask(sample_ids, split, part))
        ds.ids = [sample_ids[r] for r in ds.rows]
        return ds

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = int(self.rows[i])
        x = select_channels(np.asarray(self.x[r], dtype=np.float32), self.use_flash)
        y = np.asarray(self.y[r], dtype=np.float32)[None]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)
