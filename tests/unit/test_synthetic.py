"""specs/02-grid-risk.md: RISK-11 (moving Gaussian) and RISK-12 (synthetic window, scenario, layers, adapters, stack)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from tests.conftest import make_gaussian_cube

from outrun.core.grid import GridWindow, xy_to_lonlat
from outrun.core.policy import load_policies
from outrun.core.risk import Hazard, HazardAdapter, RiskLayer, RiskStack
from outrun.core.synthetic import (
    SYNTHETIC_STARTS,
    SyntheticAdapter,
    moving_gaussian,
    synthetic_layer,
    synthetic_scenario,
    synthetic_stack,
    synthetic_window,
)


# ---------------------------------------------------------------- RISK-11
@pytest.mark.req("RISK-11")
def test_moving_gaussian_formula():
    g = moving_gaussian()
    assert g.shape == (12, 32, 32) and g.dtype == np.float32
    np.testing.assert_allclose(g, make_gaussian_cube(), rtol=1e-6, atol=1e-7)
    # centre at slot 4 = (8 + 6, 8 + 4) = (14, 12) carries the peak
    assert np.unravel_index(np.argmax(g[4]), g[4].shape) == (14, 12)
    assert g[4, 14, 12] == pytest.approx(0.9)


@pytest.mark.req("RISK-11")
def test_moving_gaussian_params_and_determinism():
    a = moving_gaussian(n_slots=3, shape=(10, 20), start=(2.0, 3.0), velocity=(0.0, 2.0), sigma=1.5, peak=0.5)
    b = moving_gaussian(n_slots=3, shape=(10, 20), start=(2.0, 3.0), velocity=(0.0, 2.0), sigma=1.5, peak=0.5)
    assert a.shape == (3, 10, 20)
    np.testing.assert_array_equal(a, b)
    assert a[2, 2, 7] == pytest.approx(0.5)
    rr, cc = np.mgrid[0:10, 0:20]
    np.testing.assert_allclose(
        a[1], 0.5 * np.exp(-((rr - 2.0) ** 2 + (cc - 5.0) ** 2) / (2 * 1.5**2)), rtol=1e-6
    )


@pytest.mark.req("RISK-11")
@pytest.mark.parametrize("kw", [{"sigma": 0.0}, {"peak": 0.0}, {"peak": 1.5}, {"n_slots": 0}])
def test_moving_gaussian_rejects_bad_params(kw):
    with pytest.raises(ValueError):
        moving_gaussian(**kw)


# ---------------------------------------------------------------- RISK-12
@pytest.mark.req("RISK-12")
def test_synthetic_window_matches_fixture_coords(grid_coords):
    w = synthetic_window()
    assert w == GridWindow(x0=-150_000.0, y0=1_300_000.0, nx=32, ny=32, res=1000.0)
    x, y = grid_coords
    np.testing.assert_allclose(w.x_centers(), x)
    np.testing.assert_allclose(w.y_centers(), y)
    assert synthetic_window(8).shape == (8, 8)


@pytest.mark.req("RISK-12")
def test_synthetic_scenario():
    s = synthetic_scenario()
    assert s.id == "synthetic" and s.synthetic is True
    assert s.hazards == tuple(Hazard)
    assert not any(s.is_held_out(h) for h in Hazard)
    assert s.t0 == datetime(2019, 5, 26, 3, 0, tzinfo=UTC)
    assert s.grid_window() == synthetic_window()
    lon, lat = xy_to_lonlat(np.array([-150_000.0, -118_000.0]), np.array([1_268_000.0, 1_300_000.0]))
    lon_min, lat_min, lon_max, lat_max = s.bbox
    assert lon_min <= float(lon.min()) + 1e-6 and lon_max >= float(lon.max()) - 1e-6
    assert lat_min <= float(lat.min()) + 1e-6 and lat_max >= float(lat.max()) - 1e-6
    t = datetime(2020, 1, 1, tzinfo=UTC)
    assert synthetic_scenario(t, n=16).t0 == t and synthetic_scenario(t, n=16).grid_window().nx == 16


@pytest.mark.req("RISK-12")
def test_synthetic_layer(t0):
    w = synthetic_window()
    lay = synthetic_layer(Hazard.FLOOD, w, t0, kind="observed", horizon_min=90)
    assert isinstance(lay, RiskLayer)
    assert lay.prob.dims == ("slot", "y", "x") and lay.prob.dtype == np.float16
    np.testing.assert_allclose(lay.prob.coords["x"].values, w.x_centers())
    np.testing.assert_allclose(lay.prob.coords["y"].values, w.y_centers())
    expected = make_gaussian_cube(start=SYNTHETIC_STARTS[Hazard.FLOOD])
    np.testing.assert_allclose(lay.prob.values.astype(np.float32), expected, atol=1e-3)
    assert lay.kind == "observed" and lay.issued == t0 and lay.valid_until == t0 + timedelta(minutes=90)
    assert lay.source == "synthetic.flood.observed" and lay.held_out is False and lay.edge_overrides is None
    assert SYNTHETIC_STARTS[Hazard.STORM] == (4, 4) and SYNTHETIC_STARTS[Hazard.SMOKE] == (28, 8)


@pytest.mark.req("RISK-12")
def test_synthetic_adapter(t0):
    scn = synthetic_scenario(t0)
    ad = SyntheticAdapter(Hazard.ICE)
    assert isinstance(ad, HazardAdapter)
    assert ad.hazard is Hazard.ICE
    assert ad.policy.beyond_horizon == load_policies()[Hazard.ICE].beyond_horizon
    f, b, o = ad.predict(scn), ad.baseline(scn), ad.observed(scn)
    assert (f.kind, b.kind, o.kind) == ("forecast", "baseline", "observed")
    start = SYNTHETIC_STARTS[Hazard.ICE]
    np.testing.assert_allclose(f.prob.values.astype(np.float32), make_gaussian_cube(start=start), atol=1e-3)
    for k in range(12):
        np.testing.assert_array_equal(b.prob.values[k], f.prob.values[0])
    shifted = make_gaussian_cube(start=(start[0] + 1, start[1] + 1))
    np.testing.assert_allclose(o.prob.values.astype(np.float32), shifted, atol=1e-3)
    np.testing.assert_array_equal(ad.predict(scn).prob.values, f.prob.values)  # deterministic


@pytest.mark.req("RISK-12")
def test_synthetic_stack_equals_fixture(risk_stack_array, t0):
    st = synthetic_stack(t0)
    assert isinstance(st, RiskStack)
    assert st.data.shape == (6, 12, 32, 32) and st.t0 == t0
    np.testing.assert_allclose(st.data.values.astype(np.float32), risk_stack_array, atol=1e-3)
    assert set(st.meta) == set(Hazard)
