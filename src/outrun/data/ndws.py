"""Next Day Wildfire Spread TFRecords to npy (specs/01-data.md DATA-05). Uses the pure-Python ``tfrecord`` package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

# [CANDIDATE] feature names / size, verified in T1.
NDWS_INPUT_FEATURES: tuple[str, ...] = (
    "elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "NDVI", "population", "erc", "PrevFireMask",
)  # fmt: skip
NDWS_LABEL = "FireMask"
NDWS_SIZE = 64
NDWS_SPLITS: tuple[str, ...] = ("train", "eval", "test")


def parse_ndws_example(
    features: Mapping[str, Sequence[float] | np.ndarray], size: int = NDWS_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """DATA-05. -> x float32 (12, size, size), y int8 (size, size) with -1 kept."""
    raise NotImplementedError


def read_ndws_tfrecord(path: Path | str, size: int = NDWS_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """DATA-05. All examples in file order: X (N, 12, s, s) float32, y (N, s, s) int8."""
    raise NotImplementedError


def convert_ndws(
    split_files: Mapping[str, Sequence[Path | str]], out_dir: Path | str, size: int = NDWS_SIZE
) -> dict[str, tuple[Path, Path]]:
    """DATA-05. Write ndws_{split}.npy and ndws_{split}_labels.npy; return {split: (x_path, y_path)}."""
    raise NotImplementedError


def fit_clip_stats(
    X_train: np.ndarray,
    lo_pct: float = 0.1,
    hi_pct: float = 99.9,
    skip: Sequence[str] = ("th", "PrevFireMask"),
) -> dict[str, dict[str, float]]:
    """DATA-05. Per non-skipped channel: lo, hi percentiles, then mean/std of the clipped values."""
    raise NotImplementedError


def normalize(X: np.ndarray, stats: Mapping[str, Mapping[str, float]]) -> np.ndarray:
    """DATA-05. Clip + standardize channels with stats; others unchanged; float32."""
    raise NotImplementedError
