"""Next Day Wildfire Spread TFRecords to npy (specs/01-data.md DATA-05). Uses the pure-Python ``tfrecord`` package."""

from __future__ import annotations

import json
import re
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

    def field(name: str) -> np.ndarray:
        a = np.asarray(features[name], dtype=np.float32).ravel()  # KeyError if the feature is missing
        if a.size != size * size:
            raise ValueError(f"feature {name!r} has {a.size} values, expected {size * size}")
        return a.reshape(size, size)

    x = np.stack([field(name) for name in NDWS_INPUT_FEATURES]).astype(np.float32)
    lab = field(NDWS_LABEL)
    if not np.isin(lab, (-1.0, 0.0, 1.0)).all():
        raise ValueError(f"{NDWS_LABEL} values must be in {{-1, 0, 1}}")
    return x, lab.astype(np.int8)


def read_ndws_tfrecord(path: Path | str, size: int = NDWS_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """DATA-05. All examples in file order: X (N, 12, s, s) float32, y (N, s, s) int8."""
    from tfrecord.reader import tfrecord_loader  # pure Python; never TensorFlow

    xs, ys = [], []
    for ex in tfrecord_loader(str(path), None):
        x, y = parse_ndws_example(ex, size)
        xs.append(x)
        ys.append(y)
    if not xs:
        return np.zeros((0, len(NDWS_INPUT_FEATURES), size, size), np.float32), np.zeros(
            (0, size, size), np.int8
        )
    return np.stack(xs), np.stack(ys)


# GRIDMET weather channels that NDWS fills with exact zeros where the weather grid has no data (sea / outside CONUS).
WEATHER_CHANNELS: tuple[str, ...] = ("th", "vs", "tmmn", "tmmx", "sph", "pr", "erc")


def mask_weather_fill(X: np.ndarray) -> np.ndarray:
    """Set the GRIDMET weather channels to NaN (in place) at pixels where tmmn == tmmx == 0 K, the fill value.

    About 0.4% of NDWS train pixels (3.8% of samples) carry it. Left as zeros it would stretch the tmmn/tmmx
    mean and std (0 K is far below any real temperature); as NaN it is skipped by the percentile and moment fits
    and becomes 0 (the train mean) after normalisation. Returns X.
    """
    fill = (X[:, NDWS_INPUT_FEATURES.index("tmmn")] == 0) & (X[:, NDWS_INPUT_FEATURES.index("tmmx")] == 0)
    for name in WEATHER_CHANNELS:
        X[:, NDWS_INPUT_FEATURES.index(name)][fill] = np.nan
    return X


def _file_tag(path: Path) -> str:
    """``next_day_wildfire_spread_train_03.tfrecord`` -> ``train_03`` (the id stem of the frozen NDWS split)."""
    m = re.search(r"((?:train|eval|test)_\d+)", Path(path).stem)
    return m.group(1) if m else Path(path).stem


def convert_ndws(
    split_files: Mapping[str, Sequence[Path | str]], out_dir: Path | str, size: int = NDWS_SIZE
) -> dict[str, tuple[Path, Path]]:
    """DATA-05. Write ndws_{split}.npy and ndws_{split}_labels.npy; return {split: (x_path, y_path)}.

    Also writes ``ndws_{split}_ids.json``: row k's split id ``"ndws:<file tag>:<row in file, 5 digits>"``
    (the ids frozen in ``splits/ndws.json``).
    """
    bad = [s for s in split_files if s not in NDWS_SPLITS]
    if bad:
        raise ValueError(f"unknown NDWS split(s) {bad}; expected {NDWS_SPLITS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, tuple[Path, Path]] = {}
    for split, files in split_files.items():
        xs, ys, ids = [], [], []
        for f in sorted(Path(p) for p in files):
            x, y = read_ndws_tfrecord(f, size)
            mask_weather_fill(x)
            xs.append(x)
            ys.append(y)
            ids += [f"ndws:{_file_tag(f)}:{i:05d}" for i in range(len(x))]
        x_path, y_path = out_dir / f"ndws_{split}.npy", out_dir / f"ndws_{split}_labels.npy"
        np.save(x_path, np.concatenate(xs))
        np.save(y_path, np.concatenate(ys))
        (out_dir / f"ndws_{split}_ids.json").write_text(json.dumps(ids))
        result[split] = (x_path, y_path)
    return result


def fit_clip_stats(
    X_train: np.ndarray,
    lo_pct: float = 0.1,
    hi_pct: float = 99.9,
    skip: Sequence[str] = ("th", "PrevFireMask"),
) -> dict[str, dict[str, float]]:
    """DATA-05. Per non-skipped channel: lo, hi percentiles, then mean/std of the clipped values."""
    stats: dict[str, dict[str, float]] = {}
    for c, name in enumerate(NDWS_INPUT_FEATURES):
        if name in skip:
            continue
        v = np.asarray(X_train[:, c], dtype=np.float64).ravel()
        v = v[np.isfinite(v)]
        lo, hi = (float(q) for q in np.percentile(v, [lo_pct, hi_pct]))
        clipped = np.clip(v, lo, hi)
        std = float(clipped.std())
        stats[name] = {"lo": lo, "hi": hi, "mean": float(clipped.mean()), "std": std if std > 0 else 1.0}
    return stats


def normalize(X: np.ndarray, stats: Mapping[str, Mapping[str, float]]) -> np.ndarray:
    """DATA-05. Clip + standardize channels with stats; others unchanged; float32."""
    out = np.array(X, dtype=np.float32)  # copy
    for c, name in enumerate(NDWS_INPUT_FEATURES):
        if name not in stats:
            continue
        s = stats[name]
        out[:, c] = (np.clip(out[:, c], s["lo"], s["hi"]) - s["mean"]) / (s["std"] or 1.0)
    return out
