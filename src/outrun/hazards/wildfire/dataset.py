"""NDWS channels, normalisation, satellite channels, wind-consistent augmentation, dataset (specs/11-wildfire.md).

Req IDs: FIRE-01, FIRE-02, FIRE-03, FIRE-12. Orientation: row 0 north, columns eastward. ``th`` is the wind
direction in degrees, meteorological "from" convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from outrun.eval.splits import Split

NDWS_CHANNELS: tuple[str, ...] = (
    "elevation",
    "th",
    "vs",
    "tmmn",
    "tmmx",
    "sph",
    "pr",
    "pdsi",
    "NDVI",
    "population",
    "erc",
    "PrevFireMask",
)
SATELLITE_CHANNELS: tuple[str, ...] = ("NDVI", "PrevFireMask")


@dataclass(frozen=True)
class NormStats:
    """Per-channel float32 (12,) clip bounds and standardisation stats from NDWS train (FIRE-01)."""

    lo: np.ndarray
    hi: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def fit_norm_stats(x_train: np.ndarray, lo_pct: float = 0.1, hi_pct: float = 99.9) -> NormStats:
    """Percentile clip bounds, mean/std of clipped values (std 0 -> 1); PrevFireMask identity (FIRE-01)."""
    raise NotImplementedError


def normalize(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """New float32 (clip(x, lo, hi) - mean) / std per channel; non-finite -> 0; channel axis != 12 -> ValueError."""
    raise NotImplementedError


def channel_indices(names: Sequence[str]) -> list[int]:
    """Positions of ``names`` in NDWS_CHANNELS; unknown name -> KeyError (FIRE-02)."""
    raise NotImplementedError


def check_channel_order(names: Sequence[str]) -> None:
    """None if tuple(names) == NDWS_CHANNELS else ValueError (guard for WSTS-mapped arrays, FIRE-02)."""
    raise NotImplementedError


def ablate_channels(x: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Copy of x with the named channels set to 0, others unchanged (no-satellite ablation, FIRE-02)."""
    raise NotImplementedError


def augment(
    x: np.ndarray, y: np.ndarray, hflip: bool = False, vflip: bool = False, k: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """hflip, then vflip, then rot90 CCW k times on x (12,H,W) and y (H,W); th rotated consistently (FIRE-03).

    hflip: th -> 360 - th; vflip: th -> 180 - th; rot90 k: th -> th - 90k; all mod 360. Inputs not modified.
    """
    raise NotImplementedError


def random_augment(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """augment() with hflip, vflip fair coins and k in {0..3} drawn from rng (FIRE-03)."""
    raise NotImplementedError


class NdwsDataset(Dataset):
    """Memory-mapped NDWS split: ds[i] -> (normalised x float32 (12,H,W), y float32 (1,H,W) with -1 kept) (FIRE-12)."""

    def __init__(
        self, x_path: str | Path, y_path: str | Path, stats: NormStats, augment: bool = False, seed: int = 0
    ) -> None:
        raise NotImplementedError

    @classmethod
    def from_split(
        cls,
        x_path: str | Path,
        y_path: str | Path,
        sample_ids: Sequence[str],
        split: Split,
        part: str,
        stats: NormStats,
        augment: bool = False,
        seed: int = 0,
    ) -> NdwsDataset:
        """FIRE-12 / SPLIT-07. Only rows whose id is in ``split.<part>`` (``eval.splits.split_mask``), row order.

        Sets ``ds.ids``. ``len(sample_ids) != N`` -> ValueError.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
