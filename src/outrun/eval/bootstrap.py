"""Bootstrap confidence intervals (specs/15-evaluation.md, EVAL-10..EVAL-12).

* Day-block bootstrap for per-model test metrics (DESIGN 6.1): whole calendar days are resampled, so samples
  from one weather day never get treated as independent.
* Paired bootstrap over OD pairs for the routing headline (DESIGN 6.2): "DL planning (c) closes X% of the
  observed-exposure gap between (a) and (d)".

Percentile intervals; every function takes a ``seed`` and is deterministic for a given seed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CI",
    "block_resample_indices",
    "day_block_ci",
    "gap_closed",
    "gap_closed_ci",
    "paired_bootstrap_ci",
]


@dataclass(frozen=True)
class CI:
    """A point estimate with a percentile confidence interval.

    ``n_resamples`` counts the resamples whose statistic was finite (the ones used for the interval);
    ``n_blocks`` is the number of resampling units (days or OD pairs).
    """

    estimate: float
    low: float
    high: float
    n_resamples: int
    n_blocks: int

    def to_dict(self) -> dict:
        """``{"estimate", "low", "high", "n_resamples", "n_blocks"}`` with plain Python numbers (NaN kept)."""
        return {
            "estimate": float(self.estimate),
            "low": float(self.low),
            "high": float(self.high),
            "n_resamples": int(self.n_resamples),
            "n_blocks": int(self.n_blocks),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CI:
        """Inverse of ``to_dict``."""
        return cls(
            float(d["estimate"]), float(d["low"]), float(d["high"]), int(d["n_resamples"]), int(d["n_blocks"])
        )


def _percentile_ci(estimate: float, stats: list[float], alpha: float, n_blocks: int) -> CI:
    """Percentile interval over the finite resample statistics."""
    finite = np.array([x for x in stats if np.isfinite(x)], dtype=np.float64)
    if finite.size == 0:
        return CI(estimate, float("nan"), float("nan"), 0, n_blocks)
    low, high = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    return CI(estimate, float(low), float(high), int(finite.size), n_blocks)


def block_resample_indices(days: Sequence | np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """EVAL-10. One block-bootstrap draw: sample ``n_days`` unique day labels with replacement and return the
    concatenated sample indices of every drawn day (a day drawn k times contributes all its indices k times)."""
    days_arr = np.asarray(days)
    uniq, inverse = np.unique(days_arr, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(uniq) + 1))
    blocks = [order[bounds[i] : bounds[i + 1]] for i in range(len(uniq))]
    drawn = rng.integers(0, len(uniq), size=len(uniq))
    if len(drawn) == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([blocks[i] for i in drawn]).astype(np.int64)


def day_block_ci(
    metric_fn: Callable[..., float],
    arrays: Sequence[np.ndarray],
    days: Sequence | np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """EVAL-11. Day-block bootstrap CI of ``metric_fn(*arrays)``; every array is indexed on its first axis."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    arrays = [np.asarray(a) for a in arrays]
    days_arr = np.asarray(days)
    n_blocks = len(np.unique(days_arr))
    if n_blocks < 2:
        raise ValueError("need at least 2 unique days")
    if any(len(a) != len(days_arr) for a in arrays):
        raise ValueError("days and every array must have the same first-axis length")
    estimate = float(metric_fn(*arrays))
    rng = np.random.default_rng(seed)
    stats = [
        float(metric_fn(*(a[idx] for a in arrays)))
        for idx in (block_resample_indices(days_arr, rng) for _ in range(n_resamples))
    ]
    return _percentile_ci(estimate, stats, alpha, n_blocks)


def paired_bootstrap_ci(
    stat_fn: Callable[..., float],
    arrays: Sequence[np.ndarray],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """EVAL-12. Paired bootstrap: resample the shared first-axis index of all arrays jointly."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays must have the same first-axis length")
    if n < 2:
        raise ValueError("need at least 2 samples")
    estimate = float(stat_fn(*arrays))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats.append(float(stat_fn(*(a[idx] for a in arrays))))
    return _percentile_ci(estimate, stats, alpha, n)


def gap_closed(exp_a: np.ndarray, exp_c: np.ndarray, exp_d: np.ndarray) -> float:
    """EVAL-12. (mean(a) - mean(c)) / (mean(a) - mean(d)); ``nan`` when mean(a) == mean(d)."""
    a, c, d = (float(np.mean(x)) for x in (exp_a, exp_c, exp_d))
    if a == d:
        return float("nan")
    return (a - c) / (a - d)


def gap_closed_ci(
    exp_a: np.ndarray,
    exp_c: np.ndarray,
    exp_d: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """EVAL-12. Paired-bootstrap CI of ``gap_closed`` over OD pairs."""
    return paired_bootstrap_ci(
        gap_closed, (exp_a, exp_c, exp_d), n_resamples=n_resamples, alpha=alpha, seed=seed
    )
