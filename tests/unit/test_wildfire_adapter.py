"""specs/11-wildfire.md: WildfireAdapter policy and RiskLayer construction (FIRE-10).

outrun.core and the adapter are imported inside the tests so that collection never depends on them.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest


def _core():
    from outrun.core.policy import Profile
    from outrun.core.risk import Hazard, RiskLayer
    from outrun.hazards.wildfire import adapter

    return adapter, Hazard, RiskLayer, Profile


@pytest.mark.req("FIRE-10")
def test_wildfire_adapter_hazard_and_policy():
    adapter, Hazard, _, Profile = _core()
    a = adapter.WildfireAdapter()
    assert a.hazard is Hazard.WILDFIRE
    for name in ("baseline", "predict", "observed"):
        assert callable(getattr(a, name))
    pol = a.policy
    assert pol.block_at[Profile.CIVILIAN] == pytest.approx(0.5)
    assert pol.block_at[Profile.SENSITIVE] == pytest.approx(0.5)
    assert pol.block_at[Profile.EV] == pytest.approx(0.7)
    for p in (0.0, 0.5, 1.0):
        assert pol.speed_factor(p) == pytest.approx(1.0)
    assert pol.exposure_weight[Profile.CIVILIAN] == pytest.approx(20)
    assert pol.exposure_weight[Profile.SENSITIVE] == pytest.approx(20)
    assert pol.exposure_weight[Profile.EV] == pytest.approx(5)
    assert pol.beyond_horizon == "hold"
    assert pol.decay_tau_min is None


@pytest.mark.req("FIRE-10")
def test_wildfire_to_risk_layer(t0):
    adapter, Hazard, RiskLayer, _ = _core()
    prob = np.full((12, 3, 4), 0.25, dtype=np.float32)
    prob[3, 1, 1] = 2.0
    prob[3, 1, 2] = -1.0
    prob[3, 2, 2] = np.nan
    x = np.arange(4) * 1000.0
    y = np.arange(3) * -1000.0
    layer = adapter.to_risk_layer(
        prob, kind="forecast", issued=t0, source="wildfire.unet@abc", held_out=True, x=x, y=y
    )
    assert isinstance(layer, RiskLayer)
    assert layer.hazard is Hazard.WILDFIRE
    assert layer.edge_overrides is None
    assert layer.prob.dims == ("slot", "y", "x") and layer.prob.dtype == np.float16
    v = layer.prob.values
    assert v[3, 1, 1] == 1.0 and v[3, 1, 2] == 0.0 and v[3, 2, 2] == 0.0
    assert float(v[0, 0, 0]) == pytest.approx(0.25)
    np.testing.assert_array_equal(layer.prob["x"].values, x)
    np.testing.assert_array_equal(layer.prob["y"].values, y)
    assert layer.valid_until == t0 + timedelta(hours=24)
    assert layer.kind == "forecast" and layer.source == "wildfire.unet@abc" and layer.held_out is True
    obs = adapter.to_risk_layer(
        prob,
        kind="observed",
        issued=t0,
        source="observed.wsts",
        held_out=True,
        valid_until=t0 + timedelta(hours=3),
    )
    assert obs.valid_until == t0 + timedelta(hours=3) and obs.kind == "observed"


@pytest.mark.req("FIRE-10")
def test_wildfire_to_risk_layer_errors(t0):
    adapter, *_ = _core()
    ok = np.zeros((12, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        adapter.to_risk_layer(np.zeros((6, 3, 3)), kind="forecast", issued=t0, source="s", held_out=False)
    with pytest.raises(ValueError):
        adapter.to_risk_layer(ok, kind="nowcast", issued=t0, source="s", held_out=False)
    with pytest.raises(ValueError):
        adapter.to_risk_layer(ok, kind="forecast", issued=datetime(2021, 8, 1), source="s", held_out=False)


# ---------------------------------------------------------------- FIRE-10 data-backed methods (synthetic provider)
def _fire_inputs(t0, day1=None, held_out=False, x=None):
    from tests.fixtures.adapters import X_CENTERS, Y_CENTERS, identity_regrid

    from outrun.hazards.wildfire.adapter import WildfireInputs

    rng = np.random.default_rng(0)
    x_raw = rng.random((12, 8, 8)).astype(np.float32)
    x_raw[1] = 90.0  # wind direction (th) in degrees
    x_raw[11] = 0.0
    x_raw[11, 3:5, 3:5] = 1.0  # PrevFireMask
    day0 = x_raw[11].copy()
    return WildfireInputs(
        x_model=rng.normal(size=(12, 8, 8)).astype(np.float32),
        x_raw=x_raw,
        day0_mask=day0,
        day0_time=t0,
        day1_mask=day1,
        x=X_CENTERS if x is None else x,
        y=Y_CENTERS,
        regrid=lambda a: identity_regrid(a),
        held_out=held_out,
    )


@pytest.mark.req("FIRE-10")
def test_wildfire_adapter_constructs_without_arguments_and_default_factory():
    from outrun.config import get_paths
    from outrun.hazards.wildfire import adapter

    assert isinstance(adapter.WildfireAdapter(), adapter.WildfireAdapter)
    a = adapter.default_adapter()
    assert a.checkpoint == get_paths().models / "wildfire" / "best.pt"


@pytest.mark.req("FIRE-10")
def test_wildfire_adapter_baseline_predict_observed(t0):
    import torch
    from tests.fixtures.adapters import check_layer, scenario

    from outrun.core.risk import Hazard
    from outrun.hazards.wildfire import adapter
    from outrun.hazards.wildfire.model import FireUNet

    scn = scenario(t0, Hazard.WILDFIRE)
    inp = _fire_inputs(t0, held_out=True)
    ad = adapter.WildfireAdapter(provider=lambda s: inp)
    b = ad.baseline(scn)
    vb = check_layer(b, Hazard.WILDFIRE, "baseline", t0, "wildfire.persistence", True)
    np.testing.assert_allclose(vb[0], inp.day0_mask, atol=1e-3)  # day0 == t0: first hour is the day-0 mask
    assert b.edge_overrides is None and b.valid_until == t0 + timedelta(hours=24)
    with pytest.raises(RuntimeError):
        ad.predict(scn)
    torch.manual_seed(0)
    model = FireUNet(base=4, levels=2).eval()
    f = adapter.WildfireAdapter(provider=lambda s: _fire_inputs(t0), model=model).predict(scn)
    vf = check_layer(f, Hazard.WILDFIRE, "forecast", t0, "wildfire.", False)
    np.testing.assert_allclose(vf[0], inp.day0_mask, atol=1e-3)
    assert ad.observed(scn) is None
    day1 = np.zeros((8, 8), np.float32)
    day1[0, 0] = 1.0
    day1[1, 1] = -1.0
    o = adapter.WildfireAdapter(provider=lambda s: _fire_inputs(t0, day1=day1)).observed(scn)
    vo = check_layer(o, Hazard.WILDFIRE, "observed", t0, "wildfire.viirs", False)
    for s in range(12):
        np.testing.assert_array_equal(vo[s], (day1 == 1).astype(np.float32))
    with pytest.raises(ValueError):
        adapter.WildfireAdapter(provider=lambda s: _fire_inputs(t0, x=np.arange(8.0))).baseline(scn)
