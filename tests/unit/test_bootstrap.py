"""specs/15-evaluation.md: bootstrap CIs (EVAL-10..EVAL-12). Synthetic arrays only."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from outrun.eval.bootstrap import (
    CI,
    block_resample_indices,
    day_block_ci,
    gap_closed,
    gap_closed_ci,
    paired_bootstrap_ci,
)


def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))


# ---------------------------------------------------------------- EVAL-10
@pytest.mark.req("EVAL-10")
def test_block_resample_takes_whole_days():
    days = np.array(["d1"] * 3 + ["d2"] * 5 + ["d3"] * 2 + ["d4"] * 4)
    sizes = Counter(days.tolist())
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = block_resample_indices(days, rng)
        assert idx.dtype.kind in "iu"
        counts = Counter(idx.tolist())
        drawn_blocks = Counter()
        for d in sizes:
            members = np.flatnonzero(days == d)
            k = {counts.get(int(i), 0) for i in members}
            assert len(k) == 1, f"day {d} was split: per-index counts {k}"
            drawn_blocks[d] = k.pop()
        assert sum(drawn_blocks.values()) == len(sizes)  # n_days blocks drawn
        assert len(idx) == sum(sizes[d] * n for d, n in drawn_blocks.items())


@pytest.mark.req("EVAL-10")
def test_block_resample_accepts_dates_and_is_seeded():
    from datetime import date

    days = [date(2019, 5, 1 + i // 4) for i in range(20)]
    a = block_resample_indices(days, np.random.default_rng(3))
    b = block_resample_indices(days, np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert set(a.tolist()) <= set(range(20))


# ---------------------------------------------------------------- EVAL-11
@pytest.mark.req("EVAL-11")
def test_day_block_ci_estimate_determinism_and_order(rng):
    y = rng.normal(size=300)
    days = np.repeat(np.arange(30), 10)
    ci1 = day_block_ci(_mean, (y,), days, n_resamples=300, seed=7)
    ci2 = day_block_ci(_mean, (y,), days, n_resamples=300, seed=7)
    assert isinstance(ci1, CI)
    assert ci1 == ci2
    assert ci1.estimate == pytest.approx(float(np.mean(y)))
    assert ci1.low <= ci1.estimate <= ci1.high
    assert ci1.n_blocks == 30 and ci1.n_resamples == 300


@pytest.mark.req("EVAL-11")
def test_day_block_ci_width_reflects_days_not_samples():
    # 10 days, 50 identical samples each, values 0..9: the CI must be as wide as a bootstrap over 10 values
    # (about +-1.8), not over 500 samples (about +-0.25)
    y = np.repeat(np.arange(10, dtype=float), 50)
    days = np.repeat(np.arange(10), 50)
    ci = day_block_ci(_mean, (y,), days, n_resamples=1000, seed=0)
    assert ci.estimate == pytest.approx(4.5)
    assert ci.high - ci.low > 2.0


@pytest.mark.req("EVAL-11")
def test_day_block_ci_constant_data_collapses_and_multi_array():
    y = np.ones(40)
    p = np.full(40, 0.25)
    days = np.repeat(np.arange(8), 5)
    ci = day_block_ci(lambda a, b: float(np.mean((a - b) ** 2)), (y, p), days, n_resamples=200, seed=1)
    assert ci.estimate == pytest.approx(0.5625)
    assert ci.low == pytest.approx(0.5625) and ci.high == pytest.approx(0.5625)


@pytest.mark.req("EVAL-11")
def test_day_block_ci_drops_nan_resamples():
    # metric is nan unless the resample contains day 0
    y = np.arange(20, dtype=float)
    days = np.repeat(np.arange(4), 5)

    def metric(a):
        return float(np.mean(a)) if (a < 5).any() else float("nan")

    ci = day_block_ci(metric, (y,), days, n_resamples=400, seed=2)
    assert 0 < ci.n_resamples < 400
    assert math.isfinite(ci.low) and math.isfinite(ci.high)


@pytest.mark.req("EVAL-11")
def test_day_block_ci_errors():
    y = np.arange(10.0)
    with pytest.raises(ValueError):
        day_block_ci(_mean, (y,), np.zeros(10), n_resamples=10)  # one day only
    with pytest.raises(ValueError):
        day_block_ci(_mean, (y,), np.arange(9), n_resamples=10)  # length mismatch
    with pytest.raises(ValueError):
        day_block_ci(_mean, (y,), np.arange(10), n_resamples=0)
    with pytest.raises(ValueError):
        day_block_ci(_mean, (y,), np.arange(10), n_resamples=10, alpha=1.5)


# ---------------------------------------------------------------- EVAL-12
@pytest.mark.req("EVAL-12")
def test_gap_closed_point_values():
    a = np.array([10.0, 12.0])
    d = np.array([2.0, 4.0])
    assert gap_closed(a, np.array([6.0, 8.0]), d) == pytest.approx(0.5)
    assert gap_closed(a, a, d) == pytest.approx(0.0)
    assert gap_closed(a, d, d) == pytest.approx(1.0)
    assert math.isnan(gap_closed(a, d, a))


@pytest.mark.req("EVAL-12")
def test_gap_closed_ci_paired_and_deterministic(rng):
    n = 200
    a = 10 + rng.random(n)
    d = a - 8.0  # oracle always 8 lower
    c = a - 4.0  # DL always 4 lower -> exactly half the gap on every resample
    ci = gap_closed_ci(a, c, d, n_resamples=300, seed=0)
    assert ci.estimate == pytest.approx(0.5)
    assert ci.low == pytest.approx(0.5) and ci.high == pytest.approx(0.5)
    assert ci.n_blocks == n
    noisy_c = a - 4.0 + rng.normal(0, 1.0, n)
    c1 = gap_closed_ci(a, noisy_c, d, n_resamples=300, seed=5)
    c2 = gap_closed_ci(a, noisy_c, d, n_resamples=300, seed=5)
    assert c1 == c2 and c1.low < c1.estimate < c1.high


@pytest.mark.req("EVAL-12")
def test_paired_bootstrap_resamples_jointly(rng):
    x = rng.normal(size=100)
    # b == a + 1 elementwise: a joint (paired) resample of the difference is always exactly 1
    ci = paired_bootstrap_ci(lambda a, b: float(np.mean(b - a)), (x, x + 1.0), n_resamples=200, seed=0)
    assert ci.estimate == pytest.approx(1.0)
    assert ci.low == pytest.approx(1.0) and ci.high == pytest.approx(1.0)
    with pytest.raises(ValueError):
        paired_bootstrap_ci(lambda a, b: 0.0, (x, x[:-1]), n_resamples=10)
    with pytest.raises(ValueError):
        paired_bootstrap_ci(lambda a: 0.0, (x[:1],), n_resamples=10)
