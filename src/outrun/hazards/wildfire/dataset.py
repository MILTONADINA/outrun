"""NDWS channels, normalisation, satellite channels, wind-consistent augmentation, dataset (specs/11-wildfire.md).

Req IDs: FIRE-01, FIRE-02, FIRE-03, FIRE-12. Orientation: row 0 north, columns eastward. ``th`` is the wind
direction in degrees, meteorological "from" convention.
"""

from __future__ import annotations

import os
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
    n_ch = len(NDWS_CHANNELS)
    lo, hi = np.zeros(n_ch, np.float32), np.zeros(n_ch, np.float32)
    mean, std = np.zeros(n_ch, np.float32), np.ones(n_ch, np.float32)
    for c, name in enumerate(NDWS_CHANNELS):
        if name == "PrevFireMask":
            lo[c], hi[c], mean[c], std[c] = -1.0, 1.0, 0.0, 1.0
            continue
        v = np.asarray(x_train[:, c], dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        lo_c, hi_c = np.percentile(v, [lo_pct, hi_pct])
        clipped = np.clip(v, lo_c, hi_c)
        s = clipped.std()
        lo[c], hi[c], mean[c], std[c] = lo_c, hi_c, clipped.mean(), s if s > 0 else 1.0
    return NormStats(lo=lo, hi=hi, mean=mean, std=std)


def normalize(x: np.ndarray, stats: NormStats) -> np.ndarray:
    """New float32 (clip(x, lo, hi) - mean) / std per channel; non-finite -> 0; channel axis != 12 -> ValueError."""
    x = np.asarray(x)
    if x.ndim not in (3, 4) or x.shape[-3] != len(NDWS_CHANNELS):
        raise ValueError(f"expected (12, H, W) or (N, 12, H, W), got {x.shape}")
    shape = (-1, 1, 1)
    lo, hi = stats.lo.reshape(shape), stats.hi.reshape(shape)
    mean, std = stats.mean.reshape(shape), stats.std.reshape(shape)
    with np.errstate(invalid="ignore"):
        z = (np.clip(x.astype(np.float32), lo, hi) - mean) / std
    z[~np.isfinite(z)] = 0.0
    return z.astype(np.float32)


def channel_indices(names: Sequence[str]) -> list[int]:
    """Positions of ``names`` in NDWS_CHANNELS; unknown name -> KeyError (FIRE-02)."""
    return [NDWS_CHANNELS.index(n) if n in NDWS_CHANNELS else _raise_key(n) for n in names]


def _raise_key(name: str) -> int:
    raise KeyError(f"unknown NDWS channel {name!r}")


def check_channel_order(names: Sequence[str]) -> None:
    """None if tuple(names) == NDWS_CHANNELS else ValueError (guard for WSTS-mapped arrays, FIRE-02)."""
    if tuple(names) != NDWS_CHANNELS:
        raise ValueError(f"channel order {tuple(names)} != NDWS_CHANNELS {NDWS_CHANNELS}")


def ablate_channels(x: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """Copy of x with the named channels set to 0, others unchanged (no-satellite ablation, FIRE-02)."""
    out = np.array(x, copy=True)
    out[..., channel_indices(names), :, :] = 0
    return out


_TH = NDWS_CHANNELS.index("th")


def augment(
    x: np.ndarray, y: np.ndarray, hflip: bool = False, vflip: bool = False, k: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """hflip, then vflip, then rot90 CCW k times on x (12,H,W) and y (H,W); th rotated consistently (FIRE-03).

    hflip: th -> 360 - th; vflip: th -> 180 - th; rot90 k: th -> th - 90k; all mod 360. Inputs not modified.
    """
    k %= 4
    x, y = np.array(x, copy=True), np.array(y, copy=True)
    th_offset, th_sign = 0.0, 1.0  # th_new = th_sign * th + th_offset, composed step by step
    if hflip:
        x, y = x[..., ::-1], y[..., ::-1]
        th_sign, th_offset = -th_sign, 360.0 - th_offset
    if vflip:
        x, y = x[..., ::-1, :], y[..., ::-1, :]
        th_sign, th_offset = -th_sign, 180.0 - th_offset
    if k:
        x, y = np.rot90(x, k, axes=(-2, -1)), np.rot90(y, k, axes=(-2, -1))
        th_offset -= 90.0 * k
    x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
    x[_TH] = np.mod(th_sign * x[_TH].astype(np.float64) + th_offset, 360.0)
    return x, y


def random_augment(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """augment() with hflip, vflip fair coins and k in {0..3} drawn from rng (FIRE-03)."""
    hflip, vflip = bool(rng.integers(2)), bool(rng.integers(2))
    return augment(x, y, hflip=hflip, vflip=vflip, k=int(rng.integers(4)))


class NdwsDataset(Dataset):
    """Memory-mapped NDWS split: ds[i] -> (normalised x float32 (12,H,W), y float32 (1,H,W) with -1 kept) (FIRE-12)."""

    def __init__(
        self, x_path: str | Path, y_path: str | Path, stats: NormStats, augment: bool = False, seed: int = 0
    ) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if len(self.x) != len(self.y):
            raise ValueError(f"x has {len(self.x)} samples but y has {len(self.y)}")
        self.stats, self.augment, self.seed = stats, augment, seed
        self.rows = np.arange(len(self.x))
        self.ids: list[str] = []
        self._rng: np.random.Generator | None = None
        self._rng_pid = -1

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
        from outrun.eval.splits import split_mask

        ds = cls(x_path, y_path, stats, augment=augment, seed=seed)
        if len(sample_ids) != len(ds.x):
            raise ValueError(f"{len(sample_ids)} sample ids for {len(ds.x)} rows")
        ds.rows = np.flatnonzero(split_mask(sample_ids, split, part))
        ds.ids = [sample_ids[i] for i in ds.rows]
        return ds

    def __len__(self) -> int:
        return len(self.rows)

    def _generator(self) -> np.random.Generator:
        """One generator per process, so DataLoader workers do not replay the same flips."""
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            self._rng, self._rng_pid = np.random.default_rng((self.seed, pid)), pid
        return self._rng

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = int(self.rows[i])
        x, y = np.array(self.x[r], dtype=np.float32), np.array(self.y[r])
        if self.augment:  # on raw values, because the wind-direction rule needs degrees (FIRE-03)
            x, y = random_augment(x, y, self._generator())
        return torch.from_numpy(normalize(x, self.stats)), torch.from_numpy(y.astype(np.float32))[None]
