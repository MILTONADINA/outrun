"""specs/02-grid-risk.md: CAL-01..07 (Brier, BSS, reliability bins, ECE, isotonic/temperature calibration)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from outrun.core.calibrate import (
    IsotonicCalibrator,
    ReliabilityBins,
    TemperatureCalibrator,
    brier_score,
    brier_skill_score,
    calibrator_from_dict,
    expected_calibration_error,
    plot_reliability,
    reliability_bins,
)

P = np.array([0.0, 1.0, 0.5, 0.25])
Y = np.array([0, 1, 1, 0])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- CAL-01
@pytest.mark.req("CAL-01")
def test_brier_and_skill_values():
    assert brier_score(P, Y) == pytest.approx(0.078125)
    assert brier_score(P.reshape(2, 2), Y.reshape(2, 2)) == pytest.approx(0.078125)
    assert brier_score(P, Y, mask=np.array([True, True, False, True])) == pytest.approx(0.0625 / 3)
    assert brier_skill_score(P, Y) == pytest.approx(1 - 0.078125 / 0.25)
    # explicit climatology 0.25: BS_ref = mean((0.25 - y)^2) = (0.0625*2 + 0.5625*2) / 4 = 0.3125
    assert brier_skill_score(P, Y, climatology=0.25) == pytest.approx(1 - 0.078125 / 0.3125)


@pytest.mark.req("CAL-01")
@pytest.mark.parametrize(
    "p,y,mask",
    [
        (np.array([0.1, 0.2]), np.array([0, 1, 1]), None),  # shape
        (np.array([0.1, np.nan]), np.array([0, 1]), None),  # NaN
        (np.array([0.1, 1.2]), np.array([0, 1]), None),  # out of range
        (np.array([0.1, 0.2]), np.array([0, 2]), None),  # non-binary
        (np.array([]), np.array([]), None),  # empty
        (np.array([0.1, 0.2]), np.array([0, 1]), np.array([False, False])),  # nothing usable
    ],
)
def test_metric_input_validation(p, y, mask):
    with pytest.raises(ValueError):
        brier_score(p, y, mask=mask)
    with pytest.raises(ValueError):
        expected_calibration_error(p, y, mask=mask)


@pytest.mark.req("CAL-01")
def test_bss_zero_reference_raises():
    with pytest.raises(ValueError):
        brier_skill_score(np.array([0.2, 0.3]), np.array([1, 1]))


# ---------------------------------------------------------------- CAL-02
@pytest.mark.req("CAL-02")
def test_reliability_bins():
    p = np.array([0.0, 0.05, 0.15, 0.15, 0.95, 1.0])
    y = np.array([0, 0, 1, 0, 1, 1])
    b = reliability_bins(p, y, n_bins=10)
    assert isinstance(b, ReliabilityBins)
    np.testing.assert_allclose(b.edges, np.linspace(0, 1, 11))
    np.testing.assert_array_equal(b.counts, [2, 2, 0, 0, 0, 0, 0, 0, 0, 2])
    assert b.mean_pred[0] == pytest.approx(0.025) and b.frac_pos[0] == 0.0
    assert b.mean_pred[1] == pytest.approx(0.15) and b.frac_pos[1] == pytest.approx(0.5)
    assert b.mean_pred[9] == pytest.approx(0.975) and b.frac_pos[9] == 1.0
    assert np.isnan(b.mean_pred[2]) and np.isnan(b.frac_pos[5])
    b4 = reliability_bins(p, y, n_bins=4)
    np.testing.assert_array_equal(b4.counts, [4, 0, 0, 2])
    with pytest.raises(ValueError):
        reliability_bins(p, y, n_bins=0)


# ---------------------------------------------------------------- CAL-03
@pytest.mark.req("CAL-03")
def test_ece_hand_example():
    p = np.array([0.05, 0.15, 0.15, 0.95])
    y = np.array([0, 1, 0, 1])
    assert expected_calibration_error(p, y, n_bins=10) == pytest.approx(0.2)
    assert expected_calibration_error(
        np.array([0.05, 0.95]), np.array([0, 1]), mask=np.array([True, False])
    ) == pytest.approx(0.05)


@pytest.mark.req("CAL-03")
def test_ece_near_zero_when_calibrated(rng):
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)
    assert expected_calibration_error(p, y) < 0.01


# ---------------------------------------------------------------- CAL-04
@pytest.mark.req("CAL-04")
def test_isotonic_recovers_square_and_improves_brier(rng):
    p = rng.random(20_000)
    y = (rng.random(20_000) < p**2).astype(int)
    cal = IsotonicCalibrator().fit(p, y)
    assert cal is not None
    out = cal.transform(p)
    assert out.dtype == np.float32 and out.shape == p.shape
    assert brier_score(out, y) <= brier_score(p, y) + 1e-12
    grid = np.linspace(0, 1, 101)
    g = cal.transform(grid)
    assert np.all(np.diff(g) >= 0) and g.min() >= 0 and g.max() <= 1
    assert float(cal.transform(np.array([0.5]))[0]) == pytest.approx(0.25, abs=0.07)
    assert float(cal.transform(np.array([0.9]))[0]) == pytest.approx(0.81, abs=0.07)
    two_d = cal.transform(grid[:100].reshape(10, 10))
    assert two_d.shape == (10, 10)


@pytest.mark.req("CAL-04")
def test_isotonic_clips_out_of_range_and_requires_fit():
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError):
        cal.transform(np.array([0.5]))
    p = np.array([0.2, 0.3, 0.6, 0.8])
    y = np.array([0, 0, 1, 1])
    cal.fit(p, y)
    lo, hi = cal.transform(np.array([0.0, 1.0]))
    assert lo == pytest.approx(float(cal.transform(np.array([0.2]))[0]))
    assert hi == pytest.approx(float(cal.transform(np.array([0.8]))[0]))
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit(p, np.array([0, 1, 2, 1]))


# ---------------------------------------------------------------- CAL-05
@pytest.mark.req("CAL-05")
def test_temperature_recovers_overconfidence(rng):
    z = rng.normal(0.0, 1.5, 20_000)
    y = (rng.random(20_000) < _sigmoid(z)).astype(int)
    p = _sigmoid(2.0 * z)
    cal = TemperatureCalibrator().fit(p, y)
    assert cal.temperature == pytest.approx(2.0, abs=0.25)
    out = cal.transform(p)
    assert out.dtype == np.float32 and out.shape == p.shape
    assert float(cal.transform(np.array([0.5]))[0]) == pytest.approx(0.5, abs=1e-6)
    g = cal.transform(np.linspace(0.01, 0.99, 50).reshape(5, 10))
    assert g.shape == (5, 10) and np.all(np.diff(g.ravel()) > 0)
    assert brier_score(out, y) < brier_score(p, y)


@pytest.mark.req("CAL-05")
def test_temperature_requires_fit_and_handles_extremes():
    with pytest.raises(RuntimeError):
        TemperatureCalibrator().transform(np.array([0.3]))
    cal = TemperatureCalibrator().fit(np.array([0.0, 1.0, 0.2, 0.9, 0.6, 0.3]), np.array([0, 1, 0, 1, 1, 0]))
    assert cal.temperature > 0
    out = cal.transform(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(out)) and out.min() >= 0 and out.max() <= 1


# ---------------------------------------------------------------- CAL-06
@pytest.mark.req("CAL-06")
@pytest.mark.parametrize(
    "cls,typ", [(IsotonicCalibrator, "isotonic"), (TemperatureCalibrator, "temperature")]
)
def test_calibrator_round_trip(cls, typ, rng):
    p = rng.random(2000)
    y = (rng.random(2000) < p**1.5).astype(int)
    cal = cls().fit(p, y)
    d = json.loads(json.dumps(cal.to_dict()))
    assert d["type"] == typ
    back = calibrator_from_dict(d)
    probe = rng.random(500)
    np.testing.assert_allclose(back.transform(probe), cal.transform(probe), rtol=0, atol=1e-6)
    with pytest.raises(ValueError):
        calibrator_from_dict({"type": "platt"})


# ---------------------------------------------------------------- CAL-07
@pytest.mark.req("CAL-07")
def test_plot_reliability_writes_png(rng, tmp_path):
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    path = tmp_path / "rel.png"
    out = plot_reliability(reliability_bins(p, y), path, title="synthetic")
    assert path.exists() and str(out) == str(path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
