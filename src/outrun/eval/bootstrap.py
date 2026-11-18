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


def block_resample_indices(days: Sequence | np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """EVAL-10. One block-bootstrap draw: sample ``n_days`` unique day labels with replacement and return the
    concatenated sample indices of every drawn day (a day drawn k times contributes all its indices k times)."""
    raise NotImplementedError


def day_block_ci(
    metric_fn: Callable[..., float],
    arrays: Sequence[np.ndarray],
    days: Sequence | np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """EVAL-11. Day-block bootstrap CI of ``metric_fn(*arrays)``; every array is indexed on its first axis."""
    raise NotImplementedError


def paired_bootstrap_ci(
    stat_fn: Callable[..., float],
    arrays: Sequence[np.ndarray],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """EVAL-12. Paired bootstrap: resample the shared first-axis index of all arrays jointly."""
    raise NotImplementedError


def gap_closed(exp_a: np.ndarray, exp_c: np.ndarray, exp_d: np.ndarray) -> float:
    """EVAL-12. (mean(a) - mean(c)) / (mean(a) - mean(d)); ``nan`` when mean(a) == mean(d)."""
    raise NotImplementedError


def gap_closed_ci(
    exp_a: np.ndarray,
    exp_c: np.ndarray,
    exp_d: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """EVAL-12. Paired-bootstrap CI of ``gap_closed`` over OD pairs."""
    raise NotImplementedError
