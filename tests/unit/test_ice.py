"""Icy roads (specs/12-ice.md). Synthetic, offline, CPU-only. outrun imports are inside tests so collection never fails."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

UTC0 = datetime(2021, 1, 5, 0, 0, tzinfo=UTC)


def _ts(h: float, m: int = 0) -> pd.Timestamp:
    return pd.Timestamp(UTC0 + timedelta(hours=h, minutes=m))


# ---------------------------------------------------------------- ICE-01
def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_id": ["Z1", "Z2", "Z3", "Z4", "Z5", "Z6", "Z7"],
            "event_type": [
                "Ice Storm",
                "Winter Storm",
                "Winter Weather",
                "Winter Weather",
                "Thunderstorm Wind",
                " sleet ",
                "Heavy Snow",
            ],
            "begin": [_ts(1)] * 7,
            "end": [_ts(2)] * 7,
            "narrative": [
                "",
                "Freezing RAIN glazed the trees.",
                "Heavy snow fell overnight.",
                np.nan,
                "Sleet mixed in.",
                "",
                "Some sleet at the end.",
            ],
        }
    )


@pytest.mark.req("ICE-01")
def test_label_storm_events_keyword_filter():
    from outrun.hazards.ice.dataset import label_storm_events

    out = label_storm_events(_events())
    assert list(out["zone_id"]) == ["Z1", "Z2", "Z3", "Z4", "Z6"]
    assert list(out["label"]) == [1, 1, -1, -1, 1]
    assert list(out.index) == list(range(5))


@pytest.mark.req("ICE-01")
def test_label_storm_events_keywords_and_missing_column():
    from outrun.hazards.ice.dataset import ICE_EVENT_TYPES, ICE_KEYWORDS, label_storm_events

    assert set(ICE_EVENT_TYPES) == {"Ice Storm", "Sleet", "Freezing Fog"}
    for kw in ("freezing rain", "freezing drizzle", "ice accum", "glaze", "sleet"):
        assert kw in ICE_KEYWORDS
    ev = _events()
    ev.loc[2, "narrative"] = "a quarter inch of ICE ACCUMulation"
    assert label_storm_events(ev).set_index("zone_id").loc["Z3", "label"] == 1
    with pytest.raises(KeyError):
        label_storm_events(_events().drop(columns=["narrative"]))


# ---------------------------------------------------------------- ICE-02
@pytest.mark.req("ICE-02")
def test_window_start_floors_to_6h():
    from outrun.hazards.ice.dataset import window_start

    assert window_start(datetime(2021, 1, 5, 13, 30, tzinfo=UTC)) == datetime(2021, 1, 5, 12, tzinfo=UTC)
    assert window_start(datetime(2021, 1, 5, 18, 0, tzinfo=UTC)) == datetime(2021, 1, 5, 18, tzinfo=UTC)
    assert window_start(datetime(2021, 1, 5, 5, 59, tzinfo=UTC)) == datetime(2021, 1, 5, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        window_start(datetime(2021, 1, 5, 13, 30))


def _windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_id": ["A", "A", "A", "A", "B"],
            "window_start": [_ts(0), _ts(6), _ts(12), _ts(18), _ts(0)],
            "t2m_min_k": [270.0, 280.0, 270.0, 275.0, 280.0],
        }
    )


def _labelled(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["zone_id", "event_type", "begin", "end", "narrative", "label"])


@pytest.mark.req("ICE-02")
def test_assign_bag_labels_positive_ambiguous_hard_negative():
    from outrun.hazards.ice.dataset import assign_bag_labels

    ev = _labelled(
        [
            ["A", "Ice Storm", _ts(7), _ts(8), "", 1],
            ["A", "Winter Weather", _ts(13), _ts(13, 30), "snow", -1],
            ["C", "Ice Storm", _ts(0), _ts(23), "", 1],
        ]
    )
    out = assign_bag_labels(_windows(), ev)
    got = [
        (r.zone_id, pd.Timestamp(r.window_start), int(r.label), bool(r.hard_negative))
        for r in out.itertuples()
    ]
    assert got == [
        ("A", _ts(0), 0, True),
        ("A", _ts(6), 1, False),
        ("A", _ts(18), 0, True),
        ("B", _ts(0), 0, False),
    ]


@pytest.mark.req("ICE-02")
def test_assign_bag_labels_overlap_boundaries_and_bad_window():
    from outrun.hazards.ice.dataset import assign_bag_labels

    # end == window start overlaps; begin == window end does not
    ev = _labelled(
        [
            ["A", "Ice Storm", _ts(17), _ts(18), "", 1],
            ["B", "Ice Storm", _ts(6), _ts(7), "", 1],
        ]
    )
    out = assign_bag_labels(_windows(), ev).set_index(["zone_id", "window_start"])
    assert out.loc[("A", _ts(18)), "label"] == 1
    assert out.loc[("A", _ts(12)), "label"] == 1
    assert out.loc[("B", _ts(0)), "label"] == 0
    bad = _windows()
    bad.loc[0, "window_start"] = _ts(1)
    with pytest.raises(ValueError):
        assign_bag_labels(bad, ev)


# ---------------------------------------------------------------- ICE-03
@pytest.mark.req("ICE-03")
def test_fields_and_lead_rule():
    from outrun.hazards.ice.dataset import ICE_FIELDS, N_ICE_CHANNELS, check_lead

    assert ICE_FIELDS == (
        "TMP:surface",
        "TMP:2 m",
        "DPT:2 m",
        "CFRZR",
        "CICEP",
        "CSNOW",
        "CRAIN",
        "PRATE",
        "FRZR",
    )
    assert N_ICE_CHANNELS == 20
    check_lead(1)
    for bad in (0, 2, 6):
        with pytest.raises(ValueError):
            check_lead(bad)


def _fields(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ft = np.stack([np.full((h, w), k, np.float32) for k in range(9)])
    fm = np.stack([np.full((h, w), 10 + k, np.float32) for k in range(9)])
    elev = np.arange(h * w, dtype=np.float32).reshape(h, w)
    return ft, fm, elev


@pytest.mark.req("ICE-03")
def test_build_bag_input_crop_shift_and_channels():
    from outrun.hazards.ice.dataset import build_bag_input

    ft, fm, elev = _fields(100, 100)
    mask = np.zeros((100, 100), bool)
    mask[40:49, 70:79] = True  # centroid (44, 74): rows 12..75, cols 42..105 -> shifted to 36..99
    x, mcrop = build_bag_input(ft, fm, elev, mask)
    assert x.shape == (20, 64, 64) and x.dtype == np.float32
    for k in range(9):
        assert np.all(x[k] == k) and np.all(x[9 + k] == 10 + k)
    np.testing.assert_array_equal(x[18], elev[12:76, 36:100])
    np.testing.assert_array_equal(mcrop, mask[12:76, 36:100])
    assert mcrop.dtype == bool and mcrop.sum() == 81
    np.testing.assert_array_equal(x[19] > 0.5, mcrop)


@pytest.mark.req("ICE-03")
def test_build_bag_input_pads_small_grid_and_errors():
    from outrun.hazards.ice.dataset import build_bag_input

    ft, fm, elev = _fields(32, 48)
    mask = np.zeros((32, 48), bool)
    mask[5:8, 5:8] = True
    x, mcrop = build_bag_input(ft, fm, elev, mask)
    assert x.shape == (20, 64, 64)
    np.testing.assert_array_equal(x[18, :32, :48], elev)
    assert np.all(x[:, 32:, :] == 0) and np.all(x[:, :, 48:] == 0)
    assert mcrop.sum() == 9
    with pytest.raises(ValueError):
        build_bag_input(ft, fm, elev, np.zeros((32, 48), bool))
    with pytest.raises(ValueError):
        build_bag_input(ft[:8], fm, elev, mask)
    with pytest.raises(ValueError):
        build_bag_input(ft, fm, elev[:, :40], mask)


# ---------------------------------------------------------------- ICE-04
@pytest.mark.req("ICE-04")
def test_ice_milnet_shapes_and_size():
    import torch

    from outrun.hazards.ice.model import IceMILNet

    torch.manual_seed(0)
    net = IceMILNet()
    assert net(torch.zeros(2, 20, 64, 64)).shape == (2, 1, 64, 64)
    assert net(torch.zeros(1, 20, 32, 32)).shape == (1, 1, 32, 32)
    n_params = sum(p.numel() for p in net.parameters())
    assert 500_000 <= n_params <= 5_000_000
    with pytest.raises(Exception) as excinfo:  # noqa: B017 - spec allows any exception
        net(torch.zeros(1, 7, 32, 32))
    assert not isinstance(excinfo.value, NotImplementedError)


@pytest.mark.req("ICE-04")
def test_mil_pool_logsumexp_in_zone_only():
    import torch

    from outrun.hazards.ice.model import mil_pool

    logits = torch.full((1, 1, 8, 8), 100.0)
    mask = torch.zeros(1, 8, 8, dtype=torch.bool)
    mask[0, :2, :5] = True
    logits[0, 0, :2, :5] = 0.3
    assert torch.allclose(mil_pool(logits, mask), torch.tensor([0.3 + math.log(10)]), atol=1e-5)

    g = torch.Generator().manual_seed(0)
    lg = torch.randn(3, 1, 8, 8, generator=g) * 5
    m = torch.rand(3, 8, 8, generator=g) > 0.5
    z = mil_pool(lg, m)
    for b in range(3):
        inz = lg[b, 0][m[b]]
        assert inz.max() - 1e-5 <= z[b] <= inz.max() + math.log(inz.numel()) + 1e-5
    big = torch.full((1, 1, 4, 4), 1000.0)
    assert torch.isfinite(mil_pool(big, torch.ones(1, 4, 4, dtype=torch.bool))).all()
    empty = torch.zeros(2, 8, 8, dtype=torch.bool)
    empty[0, 0, 0] = True
    with pytest.raises(ValueError):
        mil_pool(lg[:2], empty)


@pytest.mark.req("ICE-04")
def test_mil_loss_matches_bce_and_gradient_stays_in_zone():
    import torch
    import torch.nn.functional as F

    from outrun.hazards.ice.model import mil_loss, mil_pool

    g = torch.Generator().manual_seed(1)
    lg = torch.randn(4, 1, 6, 6, generator=g).requires_grad_(True)
    mask = torch.zeros(4, 6, 6, dtype=torch.bool)
    mask[:, 1:4, 1:4] = True
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = mil_loss(lg, mask, y)
    ref = F.binary_cross_entropy_with_logits(mil_pool(lg, mask), y)
    assert torch.allclose(loss, ref, atol=1e-6)
    loss.backward()
    assert lg.grad[:, 0][mask].abs().sum() > 0
    assert lg.grad[:, 0][~mask].abs().sum() == 0


def _toy_bags(n: int = 16, size: int = 32, seed: int = 0):
    import torch

    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.1, (n, 20, size, size)).astype(np.float32)
    mask = np.zeros((n, size, size), bool)
    mask[:, 8:24, 8:24] = True
    y = np.array([i % 2 for i in range(n)], np.float32)
    for i in range(n):
        x[i, 19] = mask[i]
        if y[i] == 1:
            r, c = rng.integers(10, 20, size=2)
            x[i, 4, r : r + 3, c : c + 3] = 1.0  # CICEP hot spot inside the zone
    return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(y)


@pytest.mark.slow
@pytest.mark.req("ICE-04")
def test_mil_training_reduces_loss_and_cell_probabilities():
    import torch

    from outrun.hazards.ice.model import IceMILNet, cell_probabilities, mil_loss, mil_pool

    torch.manual_seed(0)
    x, mask, y = _toy_bags()
    net = IceMILNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    with torch.no_grad():
        first = float(mil_loss(net(x), mask, y))
    for _ in range(40):
        opt.zero_grad()
        loss = mil_loss(net(x), mask, y)
        loss.backward()
        opt.step()
    with torch.no_grad():  # same (train) mode as `first`
        last = float(mil_loss(net(x), mask, y))
        z = mil_pool(net(x), mask)
    assert last < first
    assert z[y == 1].mean() > z[y == 0].mean()
    p = cell_probabilities(net, x)
    assert isinstance(p, np.ndarray) and p.shape == (16, 32, 32)
    assert p.min() >= 0.0 and p.max() <= 1.0


# ---------------------------------------------------------------- ICE-05
def _rule_fields() -> dict[str, np.ndarray]:
    t = np.array([272.0, 274.0, 273.15, 273.0, 260.0, 275.0])
    f = {name: np.zeros(6) for name in ("TMP:2 m", "DPT:2 m", "CSNOW", "CRAIN", "FRZR")}
    f["TMP:surface"] = t
    f["CFRZR"] = np.array([1, 1, 0, 0, 0, 0], float)
    f["CICEP"] = np.array([0, 0, 1, 0, 0, 0], float)
    f["PRATE"] = np.array([0, 0, 0, 1e-4, 0, 1e-4])
    return f


@pytest.mark.req("ICE-05")
def test_physics_rule():
    from outrun.hazards.ice.baselines import physics_rule, physics_rule_zone

    f = _rule_fields()
    np.testing.assert_array_equal(physics_rule(f), [True, False, True, True, False, False])
    assert physics_rule_zone(f, np.array([0, 1, 0, 0, 1, 1], bool)) == 0.0
    assert physics_rule_zone(f, np.array([0, 1, 0, 1, 0, 0], bool)) == 1.0
    g = dict(f)
    del g["CICEP"]
    with pytest.raises(KeyError):
        physics_rule(g)


@pytest.mark.req("ICE-05")
def test_zone_month_climatology():
    from outrun.hazards.ice.baselines import ZoneMonthClimatology

    clim = ZoneMonthClimatology()
    with pytest.raises(RuntimeError):
        clim.predict(["A"], [1])
    clim.fit(["A", "A", "A", "B"], [1, 1, 2, 1], [1, 0, 1, 0])
    np.testing.assert_allclose(
        clim.predict(["A", "A", "B", "C", "A"], [1, 2, 1, 1, 3]), [0.5, 1.0, 0.0, 0.5, 0.5]
    )


@pytest.mark.req("ICE-05")
def test_zone_features_and_logistic_baseline(rng):
    from outrun.hazards.ice.baselines import LogisticZoneBaseline, zone_features

    x = rng.random((20, 8, 8)).astype(np.float32)
    mask = np.zeros((8, 8), bool)
    mask[2:5, 3:7] = True
    v = zone_features(x, mask)
    assert v.shape == (36,)
    np.testing.assert_allclose(v[:18], x[:18][:, mask].mean(axis=1), rtol=1e-5)
    np.testing.assert_allclose(v[18:], x[:18][:, mask].max(axis=1), rtol=1e-5)

    X = rng.normal(size=(80, 36))
    y = (X[:, 0] > 0).astype(int)
    X[:, 0] += np.where(y == 1, 2.0, -2.0)
    p = LogisticZoneBaseline(seed=0).fit(X, y).predict_proba(X)
    assert p.shape == (80,) and p.min() >= 0 and p.max() <= 1
    assert p[y == 1].min() > p[y == 0].max()


# ---------------------------------------------------------------- ICE-06
@pytest.mark.req("ICE-06")
def test_zone_calibrator(tmp_path):
    from outrun.hazards.ice.model import ZoneCalibrator

    cal = ZoneCalibrator()
    with pytest.raises(RuntimeError):
        cal.transform(np.array([0.5]))
    cal.fit(np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), np.array([0, 0, 0, 1, 1, 1]))
    assert cal.transform(np.array([0.05]))[0] == pytest.approx(0.0)
    assert cal.transform(np.array([0.95]))[0] == pytest.approx(1.0)
    grid = np.linspace(-0.5, 1.5, 101)
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= -1e-12) and out.min() >= 0 and out.max() <= 1
    cells = np.random.default_rng(0).random((3, 4))
    assert cal.transform(cells).shape == (3, 4)
    path = tmp_path / "cal.pkl"
    cal.save(path)
    np.testing.assert_array_equal(ZoneCalibrator.load(path).transform(grid), out)


# ---------------------------------------------------------------- ICE-07
@pytest.mark.req("ICE-07")
def test_steps_to_slots_hold():
    from outrun.hazards.ice.adapter import steps_to_slots

    steps = np.stack([np.full((4, 5), v, np.float32) for v in (0.1, 0.2, 0.3)])
    out = steps_to_slots(steps, step_hours=0.5)
    assert out.shape == (12, 4, 5)
    np.testing.assert_allclose(out[:, 0, 0], [0.1, 0.1, 0.2, 0.2] + [0.3] * 8, rtol=1e-6)
    one = steps_to_slots(steps[:1])
    assert one.shape == (12, 4, 5) and np.allclose(one, 0.1)
    np.testing.assert_allclose(steps_to_slots(steps)[:, 0, 0], [0.1] * 12, rtol=1e-6)
    with pytest.raises(ValueError):
        steps_to_slots(np.zeros((0, 4, 5)))


@pytest.mark.req("ICE-07")
def test_bridge_overrides():
    from outrun.hazards.ice.adapter import bridge_overrides

    probs = {0: np.array([0.2, 0.8]), 1: np.array([0.5, 0.5]), 2: np.array([0.1, 0.1])}
    before = {k: v.copy() for k, v in probs.items()}
    out = bridge_overrides(probs, {0: True, 1: False, 2: True})
    assert set(out) == {0, 2}
    np.testing.assert_allclose(out[0], [0.3, 1.0])
    np.testing.assert_allclose(out[2], [0.15, 0.15])
    for k in probs:
        np.testing.assert_array_equal(probs[k], before[k])
    same = bridge_overrides(probs, {0: True, 1: False, 2: True}, multiplier=1.0)
    np.testing.assert_allclose(same[0], probs[0])


@pytest.mark.req("ICE-07")
def test_ice_speed_factor():
    from outrun.hazards.ice.adapter import ice_speed_factor

    assert ice_speed_factor(0.0) == pytest.approx(1.0)
    assert ice_speed_factor(1.0) == pytest.approx(0.5)
    assert ice_speed_factor(0.5) == pytest.approx(0.75)
    assert ice_speed_factor(-1.0) == pytest.approx(1.0)
    assert ice_speed_factor(2.0) == pytest.approx(1 / 3)  # clipped like core make_speed_factor
    np.testing.assert_allclose(ice_speed_factor(np.array([0.0, 0.2, 1.0])), [1.0, 0.9, 0.5])


# ---------------------------------------------------------------- ICE-08
def _check_ice_policy(pol):
    from outrun.core.policy import Profile

    for prof in Profile:
        assert pol.block_at[prof] is None
    assert pol.exposure_weight[Profile.CIVILIAN] == pytest.approx(3)
    assert pol.exposure_weight[Profile.SENSITIVE] == pytest.approx(3)
    assert pol.exposure_weight[Profile.EV] == pytest.approx(1)
    assert pol.speed_factor(0.0) == pytest.approx(1.0)
    assert pol.speed_factor(1.0) == pytest.approx(0.5)
    assert pol.beyond_horizon == "hold"
    assert pol.decay_tau_min is None


@pytest.mark.req("ICE-08")
def test_ice_policy_from_config():
    from outrun.hazards.ice.adapter import ice_policy

    _check_ice_policy(ice_policy())


@pytest.mark.req("ICE-08")
def test_make_ice_layer(t0, grid_coords):
    from outrun.core.risk import Hazard
    from outrun.hazards.ice.adapter import make_ice_layer

    x, y = grid_coords
    prob = np.full((12, 32, 32), 0.4, np.float32)
    prob[0, 0, 0] = 1.2
    over = {7: np.full(12, 0.6)}
    lay = make_ice_layer(
        prob,
        x,
        y,
        kind="forecast",
        issued=t0,
        valid_until=t0 + timedelta(hours=3),
        source="ice.milcnn@test",
        held_out=True,
        edge_overrides=over,
    )
    assert lay.hazard == Hazard.ICE
    assert lay.prob.dims == ("slot", "y", "x")
    assert lay.prob.dtype == np.float16
    np.testing.assert_allclose(lay.prob["x"].values, x)
    np.testing.assert_allclose(lay.prob["y"].values, y)
    assert float(lay.prob.max()) <= 1.0 and float(lay.prob[0, 0, 0]) == pytest.approx(1.0)
    assert lay.kind == "forecast" and lay.source == "ice.milcnn@test" and lay.held_out is True
    assert lay.issued == t0 and set(lay.edge_overrides) == {7}
    kw = dict(issued=t0, valid_until=t0 + timedelta(hours=3), source="s", held_out=False)
    with pytest.raises(ValueError):
        make_ice_layer(prob, x, y, kind="guess", **kw)
    with pytest.raises(ValueError):
        make_ice_layer(prob[:, :, :10], x, y, kind="forecast", **kw)
    with pytest.raises(ValueError):
        make_ice_layer(
            prob,
            x,
            y,
            kind="forecast",
            issued=t0,
            valid_until=t0 - timedelta(hours=1),
            source="s",
            held_out=False,
        )


@pytest.mark.req("ICE-08")
def test_ice_adapter_attributes():
    from outrun.core.risk import Hazard
    from outrun.hazards.ice.adapter import IceAdapter

    ad = IceAdapter()
    assert ad.hazard == Hazard.ICE
    _check_ice_policy(ad.policy)
    for name in ("baseline", "predict", "observed"):
        assert callable(getattr(ad, name))


def _ice_inputs(truth=None, held_out=False, x=None):
    from tests.fixtures.adapters import X_CENTERS, Y_CENTERS

    from outrun.hazards.ice.adapter import IceInputs

    # model grid 16 x 16 (IceMILNet needs multiples of 8/16); regrid keeps every second cell -> the 8 x 8 window
    rng = np.random.default_rng(0)
    fields = {k: np.zeros((16, 16), np.float32) for k in ("CFRZR", "CICEP", "PRATE")}
    fields["TMP:surface"] = np.full((16, 16), 270.0, np.float32)
    fields["CFRZR"][4, 4] = 1.0  # freezing rain at one sub-freezing cell -> window cell (2, 2)
    return IceInputs(
        x_steps=rng.normal(size=(1, 20, 16, 16)).astype(np.float32),
        fields_steps=[fields],
        truth=truth,
        x=X_CENTERS if x is None else x,
        y=Y_CENTERS,
        regrid=lambda a: np.asarray(a, np.float32)[:, ::2, ::2],
        held_out=held_out,
    )


@pytest.mark.req("ICE-08")
def test_ice_adapter_default_factory():
    from outrun.config import get_paths
    from outrun.hazards.ice.adapter import IceAdapter, default_adapter

    a = default_adapter()
    assert isinstance(a, IceAdapter) and a.checkpoint == get_paths().models / "ice" / "best.pt"


@pytest.mark.req("ICE-08")
def test_ice_adapter_baseline_predict_observed(t0):
    import torch
    from tests.fixtures.adapters import check_layer, scenario, tiny_graph

    from outrun.core.risk import Hazard
    from outrun.hazards.ice.adapter import IceAdapter
    from outrun.hazards.ice.model import IceMILNet

    scn = scenario(t0, Hazard.ICE)
    ad = IceAdapter(provider=lambda s: _ice_inputs(held_out=True))
    b = ad.baseline(scn)
    vb = check_layer(b, Hazard.ICE, "baseline", t0, "ice.rule", True)
    expect = np.zeros((8, 8), np.float32)
    expect[2, 2] = 1.0
    for s in range(12):
        np.testing.assert_array_equal(vb[s], expect)
    assert b.edge_overrides is None and b.valid_until == t0 + timedelta(minutes=180)
    with pytest.raises(RuntimeError):
        ad.predict(scn)
    torch.manual_seed(0)
    model = IceMILNet(base=4).eval()
    f = IceAdapter(provider=lambda s: _ice_inputs(), model=model).predict(scn)
    check_layer(f, Hazard.ICE, "forecast", t0, "ice.milcnn", False)
    assert f.edge_overrides is None  # no graph
    g = tiny_graph()
    fg = IceAdapter(provider=lambda s: _ice_inputs(), model=model, graph=g).predict(scn)
    assert set(fg.edge_overrides) == set(np.flatnonzero(g.bridge).tolist())
    assert set(fg.edge_overrides) <= set(range(g.n_edges))
    for vals in fg.edge_overrides.values():
        assert np.asarray(vals).shape == (12,) and np.all((np.asarray(vals) >= 0) & (np.asarray(vals) <= 1))
    assert ad.observed(scn) is None
    truth = np.zeros((12, 8, 8), np.float32)
    truth[:, 5, 5] = 1.0
    o = IceAdapter(provider=lambda s: _ice_inputs(truth=truth)).observed(scn)
    vo = check_layer(o, Hazard.ICE, "observed", t0, "ice.stormevents", False)
    np.testing.assert_array_equal(vo, truth)
    with pytest.raises(ValueError):
        IceAdapter(provider=lambda s: _ice_inputs(x=np.arange(8.0))).baseline(scn)


# ---------------------------------------------------------------- ICE-09
@pytest.mark.req("ICE-09")
def test_zone_metrics_perfect_and_climatology():
    from outrun.hazards.ice.baselines import zone_metrics

    y = np.array([0, 1, 0, 1, 0, 0, 1, 0])
    clim = np.full(8, y.mean())
    m = zone_metrics(y.astype(float), y, clim, threshold=0.5)
    assert m["auprc"] == pytest.approx(1.0) and m["roc_auc"] == pytest.approx(1.0)
    assert m["brier"] == pytest.approx(0.0)
    assert (
        m["csi"] == pytest.approx(1.0) and m["pod"] == pytest.approx(1.0) and m["far"] == pytest.approx(0.0)
    )
    assert m["brier_skill"] == pytest.approx(1.0)
    assert zone_metrics(clim, y, clim, threshold=0.5)["brier_skill"] == pytest.approx(0.0)


@pytest.mark.req("ICE-09")
def test_zone_metrics_counts_and_errors():
    from outrun.hazards.ice.baselines import zone_metrics

    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1])  # hit, miss, false alarm, correct negative
    m = zone_metrics(p, y, np.full(4, 0.5), threshold=0.5)
    assert (
        m["csi"] == pytest.approx(1 / 3) and m["pod"] == pytest.approx(0.5) and m["far"] == pytest.approx(0.5)
    )
    none = zone_metrics(np.zeros(4), y, np.full(4, 0.5), threshold=0.5)
    assert none["pod"] == 0.0 and none["far"] == 0.0 and none["csi"] == 0.0
    with pytest.raises(ValueError):
        zone_metrics(np.array([]), np.array([]), np.array([]), threshold=0.5)
    with pytest.raises(ValueError):
        zone_metrics(np.array([0.1, 0.2]), np.array([1, 1]), np.array([0.5, 0.5]), threshold=0.5)


@pytest.mark.req("ICE-02")
@pytest.mark.req("SPLIT-07")
def test_bags_for_split_excludes_test_and_quarantined():
    from datetime import date

    from outrun.eval.splits import Split, SplitItem
    from outrun.hazards.ice.dataset import bags_for_split

    ws = [datetime(2024, 1, 5, h, tzinfo=UTC) for h in (0, 6, 12, 18)]
    bags = pd.DataFrame(
        {"zone_id": ["OKZ001", "OKZ001", "OKZ002", "OKZ002"], "window_start": ws, "label": [1, 0, 0, 1]}
    )
    bid = [f"{z}|{w.isoformat()}" for z, w in zip(bags.zone_id, bags.window_start, strict=True)]
    split = Split(
        "ice",
        train=(SplitItem(bid[0]), SplitItem(bid[3])),
        val=(SplitItem(bid[1]),),
        test=(SplitItem(bid[2]),),
        extra={"case_holdout": (SplitItem("OKZ009|2024-01-05T00:00:00+00:00", date(2024, 1, 5)),)},
    )
    tr = bags_for_split(bags, split, "train")
    va = bags_for_split(bags, split, "val")
    te = bags_for_split(bags, split, "test")
    assert list(tr.index) == [0, 1] and list(tr.label) == [1, 1]
    assert list(tr.zone_id) == ["OKZ001", "OKZ002"] and list(tr.window_start) == [ws[0], ws[3]]
    assert list(va.window_start) == [ws[1]] and list(te.window_start) == [ws[2]]
    assert ws[2] not in set(tr.window_start) | set(va.window_start)
    quarantined = pd.DataFrame({"zone_id": ["OKZ009"], "window_start": [ws[0]], "label": [1]})
    for part in ("train", "val", "test"):
        assert len(bags_for_split(quarantined, split, part)) == 0
