"""specs/10-flood.md: FloodAdapter policy and RiskLayer construction (FLOOD-10).

outrun.core and the adapter are imported inside the tests so that collection never depends on them.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest


def _core():
    from outrun.core.policy import Profile
    from outrun.core.risk import Hazard, RiskLayer
    from outrun.hazards.flood import adapter

    return adapter, Hazard, RiskLayer, Profile


@pytest.mark.req("FLOOD-10")
def test_flood_adapter_hazard_and_policy():
    adapter, Hazard, _, Profile = _core()
    a = adapter.FloodAdapter()
    assert a.hazard is Hazard.FLOOD
    for name in ("baseline", "predict", "observed"):
        assert callable(getattr(a, name))
    pol = a.policy
    assert pol.block_at[Profile.CIVILIAN] == pytest.approx(0.4)
    assert pol.block_at[Profile.SENSITIVE] == pytest.approx(0.4)
    assert pol.block_at[Profile.EV] == pytest.approx(0.7)
    for p in (0.0, 0.25, 0.5, 1.0):
        assert pol.speed_factor(p) == pytest.approx(1 - 0.4 * p)
    assert pol.exposure_weight[Profile.CIVILIAN] == pytest.approx(10)
    assert pol.exposure_weight[Profile.SENSITIVE] == pytest.approx(10)
    assert pol.exposure_weight[Profile.EV] == pytest.approx(3)
    assert pol.beyond_horizon == "hold"
    assert pol.decay_tau_min is None


@pytest.mark.req("FLOOD-10")
def test_flood_to_risk_layer(t0):
    adapter, Hazard, RiskLayer, _ = _core()
    prob = np.full((12, 4, 5), 0.3, dtype=np.float32)
    prob[0, 0, 0] = -0.5
    prob[0, 0, 1] = 1.5
    prob[0, 0, 2] = np.nan
    x = np.arange(5) * 1000.0
    y = np.arange(4) * -1000.0
    ov = {7: np.linspace(0, 1, 12, dtype=np.float32)}
    layer = adapter.to_risk_layer(
        prob, kind="forecast", issued=t0, source="flood.unet@abc", held_out=True, edge_overrides=ov, x=x, y=y
    )
    assert isinstance(layer, RiskLayer)
    assert layer.hazard is Hazard.FLOOD
    assert layer.prob.dims == ("slot", "y", "x")
    assert layer.prob.dtype == np.float16
    v = layer.prob.values
    assert v[0, 0, 0] == 0.0 and v[0, 0, 1] == 1.0 and v[0, 0, 2] == 0.0
    assert float(v[5, 2, 2]) == pytest.approx(0.3, abs=1e-3)
    np.testing.assert_array_equal(layer.prob["x"].values, x)
    np.testing.assert_array_equal(layer.prob["y"].values, y)
    assert layer.valid_until == t0 + timedelta(hours=6)
    assert layer.kind == "forecast" and layer.issued == t0
    assert layer.source == "flood.unet@abc" and layer.held_out is True
    assert set(layer.edge_overrides) == {7}
    np.testing.assert_array_equal(layer.edge_overrides[7], ov[7])
    custom = adapter.to_risk_layer(
        prob,
        kind="baseline",
        issued=t0,
        source="flood.ffg",
        held_out=False,
        valid_until=t0 + timedelta(hours=1),
    )
    assert custom.valid_until == t0 + timedelta(hours=1)
    assert custom.edge_overrides is None


@pytest.mark.req("FLOOD-10")
def test_flood_to_risk_layer_errors(t0):
    adapter, *_ = _core()
    ok = np.zeros((12, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        adapter.to_risk_layer(np.zeros((11, 3, 3)), kind="forecast", issued=t0, source="s", held_out=False)
    with pytest.raises(ValueError):
        adapter.to_risk_layer(np.zeros((3, 3)), kind="forecast", issued=t0, source="s", held_out=False)
    with pytest.raises(ValueError):
        adapter.to_risk_layer(ok, kind="guess", issued=t0, source="s", held_out=False)
    with pytest.raises(ValueError):
        adapter.to_risk_layer(
            ok, kind="forecast", issued=datetime(2019, 5, 26, 3), source="s", held_out=False
        )


# ---------------------------------------------------------------- FLOOD-10 data-backed methods (synthetic provider)
def _flood_inputs(reports=None, held_out=False, x=None):
    from tests.fixtures.adapters import X_CENTERS, Y_CENTERS, identity_regrid

    from outrun.hazards.flood.adapter import FloodInputs

    feats = np.zeros((11, 8, 8), np.float32)
    feats[4, 2, 3] = 1.5  # ffg_ratio_1h >= 1 at one cell
    return FloodInputs(
        features=feats,
        reports=reports,
        x=X_CENTERS if x is None else x,
        y=Y_CENTERS,
        regrid=identity_regrid,
        held_out=held_out,
    )


@pytest.mark.req("FLOOD-10")
def test_flood_adapter_constructs_without_arguments_and_default_factory():
    from outrun.config import get_paths
    from outrun.hazards.flood.adapter import FloodAdapter, default_adapter

    a = default_adapter()
    assert isinstance(a, FloodAdapter) and isinstance(FloodAdapter(), FloodAdapter)
    assert a.checkpoint == get_paths().models / "flood" / "best.pt"


@pytest.mark.req("FLOOD-10")
def test_flood_adapter_baseline_predict_observed(t0):
    import torch
    from tests.fixtures.adapters import check_layer, scenario

    from outrun.core.risk import Hazard
    from outrun.hazards.flood.adapter import FloodAdapter
    from outrun.hazards.flood.dataset import FloodReport
    from outrun.hazards.flood.model import FloodUNet

    scn = scenario(t0, Hazard.FLOOD)
    ad = FloodAdapter(provider=lambda s: _flood_inputs(held_out=True))
    b = ad.baseline(scn)
    v = check_layer(b, Hazard.FLOOD, "baseline", t0, "flood.ffg", True)
    expect = np.zeros((8, 8), np.float32)
    expect[2, 3] = 1.0
    for s in range(12):
        np.testing.assert_array_equal(v[s], expect)
    assert b.valid_until == t0 + timedelta(hours=6) and b.edge_overrides is None  # no graph
    with pytest.raises(RuntimeError):
        ad.predict(scn)
    torch.manual_seed(0)
    model = FloodUNet(base=4, levels=2).eval()
    f = FloodAdapter(provider=lambda s: _flood_inputs(), model=model).predict(scn)
    vf = check_layer(f, Hazard.FLOOD, "forecast", t0, "flood.", False)
    np.testing.assert_allclose(vf[0], vf[11])  # one map held over the 12 slots
    assert ad.observed(scn) is None  # no reports -> no truth
    rep = FloodReport(begin=t0, end=t0 + timedelta(hours=1), x0=3300.0, y0=4500.0, x1=3700.0, y1=4500.0)
    o = FloodAdapter(provider=lambda s: _flood_inputs(reports=[rep])).observed(scn)
    vo = check_layer(o, Hazard.FLOOD, "observed", t0, "flood.stormevents", False)
    assert vo[0, 3, 3] == 1.0 and vo.max() == 1.0
    assert vo[0, 7, 7] == 0.0  # (7500, 500) is > 5 km from the segment (FLOOD-11 radius)
    assert vo[4].max() == 1.0 and vo[5].max() == 0.0  # active until t0 + 1 h (slot 4 starts at 60 min)
    with pytest.raises(ValueError):  # provider coords that are not the window centres
        FloodAdapter(provider=lambda s: _flood_inputs(x=np.arange(8.0))).baseline(scn)


@pytest.mark.req("FLOOD-10")
@pytest.mark.req("FLOOD-08")
def test_flood_adapter_crossing_overrides_keyed_by_graph_edges(t0):
    import torch
    from tests.fixtures.adapters import scenario, tiny_graph

    from outrun.core.risk import Hazard
    from outrun.hazards.flood.adapter import FloodAdapter
    from outrun.hazards.flood.model import FloodUNet

    g = tiny_graph()
    scn = scenario(t0, Hazard.FLOOD)
    torch.manual_seed(0)
    model = FloodUNet(base=4, levels=2).eval()
    for graph in (g, lambda s: g):  # a RoadGraph or a per-scenario callable
        ad = FloodAdapter(provider=lambda s: _flood_inputs(), model=model, graph=graph)
        for layer in (ad.predict(scn), ad.baseline(scn)):
            ov = layer.edge_overrides
            assert ov is not None
            flagged = set(np.flatnonzero(g.crossing | g.bridge).tolist())
            assert set(ov) == flagged and set(ov) <= set(range(g.n_edges))
            for e, vals in ov.items():
                assert np.asarray(vals).shape == (12,)
                if g.bridge[e]:
                    np.testing.assert_array_equal(vals, 0.0)  # bridges do not flood (FLOOD-08)
