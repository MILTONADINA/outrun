"""Smoke (specs/13-smoke.md). Synthetic, offline, CPU-only. outrun imports are inside tests so collection never fails."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

D0 = datetime(2023, 7, 1, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------- SMOKE-01
def _aqs() -> pd.DataFrame:
    hours = [D0 + timedelta(hours=h) for h in range(25)]
    s1 = pd.DataFrame({"site_id": "S1", "time": hours, "pm25": np.arange(25, dtype=float)})
    pm2 = np.full(25, -2.0)
    pm2[18] = np.nan
    s2 = pd.DataFrame({"site_id": "S2", "time": hours, "pm25": pm2})
    pm3 = np.full(25, np.nan)
    pm3[12] = 9.0
    s3 = pd.DataFrame({"site_id": "S3", "time": hours, "pm25": pm3})  # no history before 06Z issue
    return pd.concat([s2, s3, s1], ignore_index=True)


@pytest.mark.req("SMOKE-01")
def test_build_station_samples_targets_and_drops():
    from outrun.hazards.smoke.dataset import build_station_samples

    out = build_station_samples(_aqs(), [D0 + timedelta(hours=6), D0 + timedelta(hours=12)])
    keys = [(pd.Timestamp(r.issue_time), r.site_id) for r in out.itertuples()]
    assert keys == [
        (pd.Timestamp(D0 + timedelta(hours=6)), "S1"),
        (pd.Timestamp(D0 + timedelta(hours=6)), "S2"),
        (pd.Timestamp(D0 + timedelta(hours=12)), "S1"),
    ]
    assert list(out.index) == [0, 1, 2]
    r0 = out.iloc[0]
    np.testing.assert_allclose(r0["pm_hist"], [1, 2, 3, 4, 5, 6])
    assert r0["target"] == pytest.approx(math.log1p(12.0))
    assert r0["pm_target"] == pytest.approx(12.0)
    r1 = out.iloc[1]  # negative PM clipped to 0 before log1p
    assert r1["target"] == pytest.approx(0.0) and r1["pm_target"] == pytest.approx(0.0)
    np.testing.assert_allclose(out.iloc[2]["pm_hist"], [7, 8, 9, 10, 11, 12])
    assert out.iloc[2]["target"] == pytest.approx(math.log1p(18.0))


@pytest.mark.req("SMOKE-01")
def test_build_station_samples_rejects_off_cycle_issue():
    from outrun.hazards.smoke.dataset import build_station_samples

    with pytest.raises(ValueError):
        build_station_samples(_aqs(), [D0 + timedelta(hours=7)])


# ---------------------------------------------------------------- SMOKE-02
@pytest.mark.req("SMOKE-02")
def test_rasterize_hms_max_overlap_and_zero_outside():
    from shapely.geometry import box

    from outrun.hazards.smoke.dataset import rasterize_hms

    x = np.array([0.5, 1.5, 2.5, 3.5])
    y = np.array([3.5, 2.5, 1.5, 0.5])  # north-up
    polys = [(box(0, 0, 2, 4), "Light"), (box(1, 0, 3, 1), "Heavy"), (box(3, 3, 4, 4), "medium")]
    grid = rasterize_hms(polys, x, y)
    assert grid.dtype == np.uint8 and grid.shape == (4, 4)
    np.testing.assert_array_equal(grid, [[1, 1, 0, 2], [1, 1, 0, 0], [1, 1, 0, 0], [1, 3, 3, 0]])
    assert np.all(rasterize_hms([], x, y) == 0)
    with pytest.raises(ValueError):
        rasterize_hms([(box(0, 0, 1, 1), "Extreme")], x, y)


@pytest.mark.req("SMOKE-02")
def test_smoke_episode_and_hms_ablation_switch():
    from outrun.hazards.smoke.dataset import (
        HMS_DENSITY,
        SmokeFeatureConfig,
        is_smoke_episode,
        n_tabular,
        tabular_features,
    )

    assert HMS_DENSITY == {"Light": 1, "Medium": 2, "Heavy": 3}
    np.testing.assert_array_equal(is_smoke_episode(np.array([0, 1, 2, 3])), [False, False, True, True])
    t = datetime(2024, 3, 1, 6, 0, tzinfo=UTC)  # hour 6, day-of-year 61 (leap year)
    hist = np.array([1, 2, np.nan, 4, 5, 6], float)
    with_hms = tabular_features(hist, 3.0, t, SmokeFeatureConfig())
    no_hms = tabular_features(hist, 3.0, t, SmokeFeatureConfig(use_hms=False))
    assert with_hms.dtype == np.float32 and len(with_hms) == 11 == n_tabular(SmokeFeatureConfig())
    assert len(no_hms) == 10 == n_tabular(SmokeFeatureConfig(use_hms=False))
    np.testing.assert_allclose(with_hms[:6], np.log1p([1, 2, 3.6, 4, 5, 6]), rtol=1e-5)
    assert with_hms[6] == pytest.approx(3.0)
    doy = 2 * math.pi * 61 / 365.25
    np.testing.assert_allclose(with_hms[7:], [1.0, 0.0, math.sin(doy), math.cos(doy)], atol=1e-5)
    np.testing.assert_allclose(no_hms, np.delete(with_hms, 6), atol=1e-6)
    neg = tabular_features(np.array([-2.0, 1, 1, 1, 1, 1]), 0.0, t, SmokeFeatureConfig())
    assert neg[0] == pytest.approx(0.0) and np.all(np.isfinite(neg))


# ---------------------------------------------------------------- SMOKE-03
@pytest.mark.req("SMOKE-03")
def test_massden_unit_change_converted():
    from outrun.hazards.smoke.dataset import MASSDEN_UNIT_CHANGE, massden_to_ugm3

    assert datetime(2021, 12, 21, 0, 0, tzinfo=UTC) == MASSDEN_UNIT_CHANGE
    before = massden_to_ugm3(np.array([5.0, 20.0]), datetime(2021, 6, 1, tzinfo=UTC))
    np.testing.assert_allclose(before, [5.0, 20.0])
    after = massden_to_ugm3(np.array([5e-9, 2e-8, np.nan]), datetime(2022, 1, 1, tzinfo=UTC))
    assert after.dtype == np.float32
    np.testing.assert_allclose(after[:2], [5.0, 20.0], rtol=1e-5)
    assert np.isnan(after[2])
    at = massden_to_ugm3(np.array([1e-9]), MASSDEN_UNIT_CHANGE)
    np.testing.assert_allclose(at, [1.0], rtol=1e-5)


@pytest.mark.req("SMOKE-03")
def test_massden_unit_assertions():
    from outrun.hazards.smoke.dataset import massden_to_ugm3

    with pytest.raises(ValueError):  # still ug/m3 after the change
        massden_to_ugm3(np.array([5.0]), datetime(2023, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError):
        massden_to_ugm3(np.array([-1.0]), datetime(2021, 6, 1, tzinfo=UTC))
    with pytest.raises(ValueError):
        massden_to_ugm3(np.array([1e-9]), datetime(2023, 1, 1))


# ---------------------------------------------------------------- SMOKE-04
@pytest.mark.req("SMOKE-04")
def test_smokenet_shapes_and_size():
    import torch

    from outrun.hazards.smoke.model import SmokeNet

    torch.manual_seed(0)
    net = SmokeNet()
    assert net(torch.zeros(4, 8, 9, 9), torch.zeros(4, 11)).shape == (4, 2)
    assert SmokeNet(n_tabular=10)(torch.zeros(2, 8, 9, 9), torch.zeros(2, 10)).shape == (2, 2)
    n_params = sum(p.numel() for p in net.parameters())
    assert 20_000 <= n_params <= 500_000
    with pytest.raises(Exception) as excinfo:  # noqa: B017 - spec allows any exception
        net(torch.zeros(4, 8, 9, 9), torch.zeros(4, 5))
    assert not isinstance(excinfo.value, NotImplementedError)


@pytest.mark.req("SMOKE-04")
def test_pinball_loss_values():
    import torch

    from outrun.hazards.smoke.model import pinball_loss

    q = (0.5, 0.9)
    assert float(pinball_loss(torch.zeros(1, 2), torch.tensor([1.0]), q)) == pytest.approx(0.7)
    assert float(pinball_loss(torch.zeros(1, 2), torch.tensor([-1.0]), q)) == pytest.approx(0.3)
    pred = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    assert float(pinball_loss(pred, torch.tensor([1.0, -1.0]), q)) == pytest.approx(0.5)
    assert float(pinball_loss(torch.full((3, 2), 2.0), torch.full((3,), 2.0), q)) == pytest.approx(0.0)


@pytest.mark.req("SMOKE-04")
def test_to_pm_expm1_clip_noncrossing():
    from outrun.hazards.smoke.model import to_pm

    out = to_pm(np.array([[math.log1p(10.0), math.log1p(5.0)], [-1.0, math.log1p(3.0)]]))
    np.testing.assert_allclose(out, [[10.0, 10.0], [0.0, 3.0]], rtol=1e-6)


@pytest.mark.slow
@pytest.mark.req("SMOKE-04")
def test_smokenet_training_reduces_pinball():
    import torch

    from outrun.hazards.smoke.model import SmokeNet, pinball_loss

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    patch = torch.randn(64, 8, 9, 9, generator=g)
    tab = torch.randn(64, 11, generator=g)
    y = 0.5 * tab[:, 0] + 1.0
    net = SmokeNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    with torch.no_grad():
        first = float(pinball_loss(net(patch, tab), y, (0.5, 0.9)))
    for _ in range(60):
        opt.zero_grad()
        loss = pinball_loss(net(patch, tab), y, (0.5, 0.9))
        loss.backward()
        opt.step()
    with torch.no_grad():
        last = float(pinball_loss(net(patch, tab), y, (0.5, 0.9)))
    assert last < 0.5 * first


# ---------------------------------------------------------------- SMOKE-05
@pytest.mark.req("SMOKE-05")
def test_persistence_and_background_median():
    from outrun.hazards.smoke.baselines import background_median, persistence

    nan = np.nan
    hist = np.array([[1, 2, nan, nan, nan, nan], [nan] * 6, [3, nan, nan, nan, nan, 7]])
    out = persistence(hist)
    assert out[0] == 2 and np.isnan(out[1]) and out[2] == 7

    times = [D0 + timedelta(hours=h) for h in range(40 * 24)]
    pm = np.array([h // 24 for h in range(40 * 24)], float)
    at = D0 + timedelta(days=35)
    pm[35 * 24] = 1e6  # the target hour itself must be excluded
    assert background_median(times, pm, at, days=30) == pytest.approx(19.5)
    assert np.isnan(background_median(times, pm, D0, days=30))


@pytest.mark.req("SMOKE-05")
def test_massden_calibrated_and_linear_baseline(rng):
    from outrun.hazards.smoke.baselines import LinearSmokeBaseline, MassdenCalibrated

    m = rng.random(50) * 40
    bg = rng.random(50) * 10
    target = 2.0 * m + 3.0 + bg
    cal = MassdenCalibrated()
    with pytest.raises(RuntimeError):
        cal.predict(m, bg)
    cal.fit(m, bg, target)
    assert cal.a == pytest.approx(2.0, abs=1e-6) and cal.b == pytest.approx(3.0, abs=1e-6)
    np.testing.assert_allclose(cal.predict(m, bg), target, rtol=1e-6)
    assert cal.predict(np.array([-100.0]), np.array([0.0]))[0] == 0.0

    X = rng.normal(size=(40, 5))
    w = np.array([0.5, -1.0, 2.0, 0.0, 0.3])
    y = X @ w + 0.7
    np.testing.assert_allclose(LinearSmokeBaseline().fit(X, y).predict(X), y, atol=1e-8)


# ---------------------------------------------------------------- SMOKE-06
def _sites(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(10):  # 10 blocks of 200 km in a row, 2-8 sites each
        for k in range(int(rng.integers(2, 9))):
            rows.append((f"b{b}s{k}", b * 200_000 + rng.uniform(1, 199_999), rng.uniform(1, 199_999)))
    return pd.DataFrame(rows, columns=["site_id", "x_m", "y_m"])


@pytest.mark.req("SMOKE-06")
def test_spatial_holdout_whole_blocks_and_share():
    from outrun.hazards.smoke.dataset import spatial_holdout

    sites = _sites()
    held = spatial_holdout(sites, frac=0.2, seed=0)
    block = {
        r.site_id: (math.floor(r.x_m / 200_000), math.floor(r.y_m / 200_000)) for r in sites.itertuples()
    }
    held_blocks = {block[s] for s in held}
    kept_blocks = {block[s] for s in set(sites.site_id) - held}
    assert held_blocks.isdisjoint(kept_blocks)
    n = len(sites)
    largest = sites.assign(b=sites.site_id.map(block)).groupby("b").size().max()
    assert 0.2 <= len(held) / n < 0.2 + largest / n
    assert len(held) < n
    assert spatial_holdout(sites, frac=0.2, seed=0) == held


@pytest.mark.req("SMOKE-06")
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_spatial_holdout_matches_frozen_smoke_split(seed):
    from outrun.eval.splits import smoke_split, spatial_holdout_monitors
    from outrun.hazards.smoke.dataset import spatial_holdout

    sites = _sites()
    monitors = {r.site_id: (float(r.x_m), float(r.y_m)) for r in sites.itertuples()}
    samples = [(m, datetime(2022, 3, 1, 12, tzinfo=UTC)) for m in monitors]
    samples += [(m, datetime(2024, 3, 1, 12, tzinfo=UTC)) for m in monitors]
    split = smoke_split(samples, monitors, holdout_frac=0.2, seed=seed)
    held = spatial_holdout(sites, frac=0.2, seed=seed)
    assert held == set(split.meta["holdout_monitors"])
    assert held == set(spatial_holdout_monitors(monitors, 0.2, 200_000, seed))


@pytest.mark.req("SMOKE-06")
def test_spatial_holdout_invalid_frac():
    from outrun.hazards.smoke.dataset import spatial_holdout

    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            spatial_holdout(_sites(), frac=bad)


# ---------------------------------------------------------------- SMOKE-07
@pytest.mark.req("SMOKE-07")
def test_idw_exact_bounded_nan():
    from outrun.hazards.smoke.adapter import idw

    x = np.array([500.0, 1500.0, 2500.0, 3500.0])
    y = np.array([3500.0, 2500.0, 1500.0, 500.0])
    st = np.array([[500.0, 3500.0], [3500.0, 500.0], [2000.0, 2000.0]])
    vals = np.array([10.0, 50.0, np.nan])
    g = idw(st, vals, x, y)
    assert g.shape == (4, 4)
    assert g[0, 0] == pytest.approx(10.0) and g[3, 3] == pytest.approx(50.0)
    assert g.min() >= 10.0 - 1e-9 and g.max() <= 50.0 + 1e-9
    assert np.all(np.isfinite(g))
    with pytest.raises(ValueError):
        idw(st, np.full(3, np.nan), x, y)


@pytest.mark.req("SMOKE-07")
def test_pm_to_risk_hold_and_shelter_rule():
    from outrun.hazards.smoke.adapter import hold_to_slots, pm_to_risk, shelter_deprioritized

    np.testing.assert_allclose(
        pm_to_risk(np.array([0.0, 35.4, 92.9, 150.4, 500.0])), [0, 0, 0.5, 1, 1], atol=1e-9
    )
    assert pm_to_risk(10.0) == pytest.approx(0.0)
    f = np.random.default_rng(0).random((5, 6))
    s = hold_to_slots(f)
    assert s.shape == (12, 5, 6) and all(np.array_equal(s[i], f) for i in range(12))
    np.testing.assert_array_equal(shelter_deprioritized(np.array([0.2, 0.5, 0.51])), [False, False, True])


# ---------------------------------------------------------------- SMOKE-08
def _check_smoke_policy(pol):
    from outrun.core.policy import Profile

    for prof in Profile:
        assert pol.block_at[prof] is None
    assert pol.exposure_weight[Profile.CIVILIAN] == pytest.approx(0.5)
    assert pol.exposure_weight[Profile.SENSITIVE] == pytest.approx(5)
    assert pol.exposure_weight[Profile.EV] == pytest.approx(0)
    for p in (0.0, 0.3, 1.0):
        assert pol.speed_factor(p) == pytest.approx(1.0)
    assert pol.beyond_horizon == "hold"
    assert pol.decay_tau_min is None


@pytest.mark.req("SMOKE-08")
def test_smoke_policy_from_config():
    from outrun.hazards.smoke.adapter import smoke_policy

    _check_smoke_policy(smoke_policy())


@pytest.mark.req("SMOKE-08")
def test_make_smoke_layer(t0, grid_coords):
    from outrun.core.risk import Hazard
    from outrun.hazards.smoke.adapter import make_smoke_layer

    x, y = grid_coords
    pm = np.full((12, 32, 32), 92.9, np.float32)
    pm[:, 0, 0] = 10.0
    lay = make_smoke_layer(
        pm,
        x,
        y,
        kind="observed",
        issued=t0,
        valid_until=t0 + timedelta(hours=6),
        source="observed.aqs",
        held_out=True,
    )
    assert lay.hazard == Hazard.SMOKE
    assert lay.prob.dims == ("slot", "y", "x") and lay.prob.dtype == np.float16
    assert float(lay.prob[0, 0, 0]) == 0.0
    assert float(lay.prob[0, 1, 1]) == pytest.approx(0.5, abs=1e-3)
    np.testing.assert_allclose(lay.prob["x"].values, x)
    assert lay.edge_overrides is None and lay.kind == "observed" and lay.source == "observed.aqs"
    kw = dict(issued=t0, valid_until=t0 + timedelta(hours=6), source="s", held_out=False)
    with pytest.raises(ValueError):
        make_smoke_layer(pm, x, y, kind="nowcast", **kw)
    with pytest.raises(ValueError):
        make_smoke_layer(pm[:, :5], x, y, kind="forecast", **kw)
    with pytest.raises(ValueError):
        make_smoke_layer(
            pm,
            x,
            y,
            kind="forecast",
            issued=t0,
            valid_until=t0 - timedelta(minutes=1),
            source="s",
            held_out=False,
        )


@pytest.mark.req("SMOKE-08")
def test_smoke_adapter_attributes():
    from outrun.core.risk import Hazard
    from outrun.hazards.smoke.adapter import SmokeAdapter

    ad = SmokeAdapter()
    assert ad.hazard == Hazard.SMOKE
    _check_smoke_policy(ad.policy)
    for name in ("baseline", "predict", "observed"):
        assert callable(getattr(ad, name))


def _smoke_inputs(obs=None, held_out=False, x=None):
    from tests.fixtures.adapters import X_CENTERS, Y_CENTERS, identity_regrid

    from outrun.hazards.smoke.adapter import SmokeInputs

    rng = np.random.default_rng(0)
    hrrr = np.zeros((8, 8), np.float32)
    hrrr[1, 2] = 150.4  # -> risk 1
    hrrr[4, 4] = 92.9  # -> risk 0.5
    return SmokeInputs(
        patches=rng.normal(size=(64, 8, 9, 9)).astype(np.float32),
        tabs=rng.normal(size=(64, 11)).astype(np.float32),
        grid_shape=(8, 8),
        hrrr_pm=hrrr,
        obs_pm=obs,
        x=X_CENTERS if x is None else x,
        y=Y_CENTERS,
        regrid=identity_regrid,
        held_out=held_out,
    )


@pytest.mark.req("SMOKE-08")
def test_smoke_adapter_default_factory():
    from outrun.config import get_paths
    from outrun.hazards.smoke.adapter import SmokeAdapter, default_adapter

    a = default_adapter()
    assert isinstance(a, SmokeAdapter) and a.checkpoint == get_paths().models / "smoke" / "best.pt"


@pytest.mark.req("SMOKE-08")
def test_smoke_adapter_baseline_predict_observed(t0):
    import torch
    from tests.fixtures.adapters import check_layer, scenario

    from outrun.core.risk import Hazard
    from outrun.hazards.smoke.adapter import SmokeAdapter
    from outrun.hazards.smoke.model import SmokeNet

    scn = scenario(t0, Hazard.SMOKE)
    ad = SmokeAdapter(provider=lambda s: _smoke_inputs(held_out=True))
    b = ad.baseline(scn)
    vb = check_layer(b, Hazard.SMOKE, "baseline", t0, "smoke.hrrr", True)
    for s in range(12):
        assert vb[s, 1, 2] == pytest.approx(1.0) and vb[s, 4, 4] == pytest.approx(0.5, abs=1e-3)
        assert vb[s, 0, 0] == 0.0
    assert b.edge_overrides is None and b.valid_until == t0 + timedelta(hours=6)
    with pytest.raises(RuntimeError):
        ad.predict(scn)
    torch.manual_seed(0)
    f = SmokeAdapter(provider=lambda s: _smoke_inputs(), model=SmokeNet().eval()).predict(scn)
    vf = check_layer(f, Hazard.SMOKE, "forecast", t0, "smoke.cnnmlp", False)
    np.testing.assert_allclose(vf[0], vf[11])  # held over the slots
    assert ad.observed(scn) is None
    obs = np.zeros((8, 8), np.float32)
    obs[7, 7] = 200.0
    o = SmokeAdapter(provider=lambda s: _smoke_inputs(obs=obs)).observed(scn)
    vo = check_layer(o, Hazard.SMOKE, "observed", t0, "smoke.aqs", False)
    assert vo[0, 7, 7] == 1.0 and vo[0].sum() == 1.0
    with pytest.raises(ValueError):
        SmokeAdapter(provider=lambda s: _smoke_inputs(x=np.arange(8.0))).baseline(scn)


# ---------------------------------------------------------------- SMOKE-09
@pytest.mark.req("SMOKE-09")
def test_smoke_metrics():
    from outrun.hazards.smoke.baselines import smoke_metrics

    obs = np.array([10.0, 50.0, 100.0, 20.0])
    ep = np.array([False, True, True, False])
    perfect = smoke_metrics(obs, obs, obs, ep)
    assert perfect["rmse"] == pytest.approx(0) and perfect["mae"] == pytest.approx(0)
    assert perfect["csi"] == pytest.approx(1) and perfect["q90_coverage"] == pytest.approx(1)

    q50 = np.array([40.0, 10.0, 50.0, 20.0])
    q90 = np.array([10.0, 40.0, 200.0, 30.0])
    m = smoke_metrics(q50, q90, obs, ep)
    assert (
        m["csi"] == pytest.approx(1 / 3) and m["pod"] == pytest.approx(0.5) and m["far"] == pytest.approx(0.5)
    )
    assert m["q90_coverage"] == pytest.approx(0.75)
    assert m["mae"] == pytest.approx((30 + 40 + 50 + 0) / 4)
    assert m["rmse"] == pytest.approx(math.sqrt((900 + 1600 + 2500) / 4))
    assert m["mae_episode"] == pytest.approx(45.0)
    assert m["rmse_episode"] == pytest.approx(math.sqrt((1600 + 2500) / 2))

    none = smoke_metrics(obs, obs, obs, np.zeros(4, bool))
    assert np.isnan(none["rmse_episode"]) and np.isnan(none["mae_episode"])
    zero = smoke_metrics(np.zeros(4), np.zeros(4), np.zeros(4), ep)
    assert zero["csi"] == 0.0 and zero["pod"] == 0.0 and zero["far"] == 0.0
    with pytest.raises(ValueError):
        smoke_metrics(obs[:3], obs, obs, ep)
    with pytest.raises(ValueError):
        smoke_metrics(np.array([]), np.array([]), np.array([]), np.array([], bool))


@pytest.mark.req("SMOKE-01")
@pytest.mark.req("SPLIT-07")
def test_samples_for_split_excludes_test_quarantined_and_holdout():
    from datetime import date

    from outrun.eval.splits import Split, SplitItem
    from outrun.hazards.smoke.dataset import samples_for_split

    t = [datetime(2022, 3, 1, h, tzinfo=UTC) for h in (0, 6, 12)]
    samples = pd.DataFrame(
        {"site_id": ["A", "A", "B", "C"], "issue_time": [t[0], t[1], t[2], t[0]], "target": range(4)}
    )
    sid = [f"{s}|{x.isoformat()}" for s, x in zip(samples.site_id, samples.issue_time, strict=True)]
    split = Split(
        "smoke",
        train=(SplitItem(sid[0]),),
        val=(SplitItem(sid[2]),),
        test=(SplitItem(sid[1]), SplitItem(sid[3])),
        extra={
            "case_holdout": (SplitItem("Q|2022-03-01T00:00:00+00:00", date(2022, 3, 1)),),
            "test_spatial": (SplitItem(sid[3]),),
        },
        meta={"holdout_monitors": ["C"]},
    )
    tr = samples_for_split(samples, split, "train")
    va = samples_for_split(samples, split, "val")
    assert list(tr.target) == [0] and list(tr.index) == [0]
    assert list(va.target) == [2]
    assert "C" not in set(tr.site_id) | set(va.site_id)  # spatially held-out monitor
    assert list(samples_for_split(samples, split, "test").target) == [1, 3]
    assert list(samples_for_split(samples, split, "test_spatial").site_id) == ["C"]
    q = pd.DataFrame({"site_id": ["Q"], "issue_time": [t[0]], "target": [9]})
    assert len(samples_for_split(q, split, "train")) == 0 and len(samples_for_split(q, split, "val")) == 0
