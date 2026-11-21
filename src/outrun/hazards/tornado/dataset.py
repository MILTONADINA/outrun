"""TorNet split: official train/test from the catalog ``type`` column, grouped calendar-day val, quarantine
(TORN-02)."""

from __future__ import annotations

from collections.abc import Collection
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import NORM_CLIP, coord_channels, normalize


def split_catalog(
    catalog: pd.DataFrame,
    val_frac: float = 0.1,
    seed: int = 0,
    quarantine: Collection[date] = (),
    column: str = "type",
) -> pd.Series:
    """Label rows "train" / "val" / "test" / "excluded" (TORN-02). KeyError without ``date`` or ``column``.

    The official split is ``catalog[column]`` (stripped, lower-case; anything but "train"/"test" -> ValueError), as
    shipped in TorNet catalog.csv. Delegates to ``outrun.eval.splits.tornet_split`` (SPLIT-02) on samples
    ``(str(index), date at 00:00 UTC, official)``; rows that land in no set (quarantine or the 1-day gap) are
    "excluded". Never runs a separate split algorithm.
    """
    from datetime import UTC, datetime

    from outrun.eval.splits import tornet_split

    if "date" not in catalog.columns:
        raise KeyError("date")
    if column not in catalog.columns:
        raise KeyError(column)
    official = catalog[column].astype(str).str.strip().str.lower()
    days = pd.to_datetime(catalog["date"]).dt.date
    samples = [
        (str(i), datetime(d.year, d.month, d.day, tzinfo=UTC), o)
        for i, d, o in zip(catalog.index, days, official, strict=True)
    ]
    split = tornet_split(samples, quarantine=quarantine, val_frac=val_frac, seed=seed)
    part = {it.id: name for name in ("train", "val", "test") for it in getattr(split, name)}
    return pd.Series([part.get(str(i), "excluded") for i in catalog.index], index=catalog.index, name="split")


# ---------------------------------------------------------------------------------------------------------------
# Processed TorNet arrays (data/tornet_prep.py): ``{split}_x.npy`` fp16 (N, 25, A, R) = 12 last-frame channels,
# 12 temporal-difference channels and the missing mask (missing gates already hold the FILL_VALUES), plus
# ``{split}_range_km.npy`` (N, R), ``{split}_azimuth.npy`` (N, A) and ``{split}_meta.csv``.
# ---------------------------------------------------------------------------------------------------------------
SPLITS = ("train", "val", "test", "case_holdout")


def tornet_root() -> Path:
    from outrun.config import get_paths

    return get_paths().proc / "tornet"


def load_meta(split: str, root: Path | None = None) -> pd.DataFrame:
    """The prep meta table for one processed split (``label`` = category TOR, QC columns)."""
    return pd.read_csv((root or tornet_root()) / f"{split}_meta.csv")


def keep_indices(split: str, meta: pd.DataFrame) -> np.ndarray:
    """Rows to use. Train drops chips whose whole last frame is missing (no signal at all); val/test keep every
    chip so scores are not flattered."""
    if split == "train":
        return np.flatnonzero(meta["all_missing"].to_numpy() == 0)
    return np.arange(len(meta))


class TornetChips(Dataset):
    """``(x (27|26, A, R) float32, y float)``: normalised stored channels + CoordConv channels, memmap-backed."""

    def __init__(
        self,
        split: str,
        mean: np.ndarray,
        std: np.ndarray,
        indices: np.ndarray | None = None,
        include_range: bool = True,
        root: Path | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else tornet_root()
        self.split = split
        self.mean, self.std, self.include_range = mean, std, include_range
        self.meta = load_meta(split, self.root)
        self.indices = keep_indices(split, self.meta) if indices is None else np.asarray(indices)
        self.labels = self.meta["label"].to_numpy().astype(np.float32)
        self._x: np.ndarray | None = None
        self._rng = np.load(self.root / f"{split}_range_km.npy")
        self._az = np.load(self.root / f"{split}_azimuth.npy")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._x is None:  # opened lazily so DataLoader workers each get their own handle
            self._x = np.load(self.root / f"{self.split}_x.npy", mmap_mode="r")
        i = int(self.indices[k])
        x = normalize(np.asarray(self._x[i], dtype=np.float32)[None], self.mean, self.std, clip=NORM_CLIP)[0]
        x = np.concatenate([x, coord_channels(self._rng[i], self._az[i], self.include_range)])
        return torch.from_numpy(x), torch.tensor(self.labels[i])
