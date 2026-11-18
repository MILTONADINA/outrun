"""specs/02-grid-risk.md: RISK-09 (profiles, policies from config) and HORIZON-01..04 (beyond-horizon rules)."""

from __future__ import annotations

import math
from dataclasses import fields
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from outrun.core.policy import (
    HazardPolicy,
    Profile,
    apply_beyond_horizon,
    blocks,
    load_hazard_config,
    load_policies,
    make_speed_factor,
)
from outrun.core.risk import Hazard

ROOT = Path(__file__).resolve().parents[2]


def _policy(rule, tau=None, block=None):
    return HazardPolicy(
        block_at={p: block for p in Profile},
        speed_factor=lambda p: 1.0,
        exposure_weight={p: 1.0 for p in Profile},
        beyond_horizon=rule,
        decay_tau_min=tau,
    )


def _apply(prob, pol, t0, horizon_min, **kw):
    return apply_beyond_horizon(
        prob, pol, t0=t0, issued=t0, valid_until=t0 + timedelta(minutes=horizon_min), **kw
    )


# ---------------------------------------------------------------- RISK-09
@pytest.mark.req("RISK-09")
def test_profile_and_policy_fields():
    assert [p.value for p in Profile] == ["civilian", "sensitive", "ev"]
    assert [f.name for f in fields(HazardPolicy)] == [
        "block_at",
        "speed_factor",
        "exposure_weight",
        "beyond_horizon",
        "decay_tau_min",
    ]


@pytest.mark.req("RISK-09")
def test_speed_factor_clipped_and_vectorised():
    f = make_speed_factor(0.5)
    assert f(0.0) == pytest.approx(1.0)
    assert f(1.0) == pytest.approx(0.5)
    g = make_speed_factor(5.0)
    assert g(1.0) == pytest.approx(1 / 3)
    arr = np.array([[0.0, 0.2], [0.5, 1.0]])
    out = np.asarray(g(arr))
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, np.clip(1 - 5.0 * arr, 1 / 3, 1))
    assert make_speed_factor(0.0)(0.9) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        make_speed_factor(-0.1)


@pytest.mark.req("RISK-09")
def test_blocks():
    pol = HazardPolicy(
        block_at={Profile.CIVILIAN: 0.4, Profile.SENSITIVE: 0.4, Profile.EV: None},
        speed_factor=lambda p: 1.0,
        exposure_weight={p: 1.0 for p in Profile},
        beyond_horizon="hold",
        decay_tau_min=None,
    )
    assert bool(blocks(pol, Profile.CIVILIAN, 0.4)) is True
    assert bool(blocks(pol, Profile.CIVILIAN, 0.39)) is False
    np.testing.assert_array_equal(blocks(pol, Profile.CIVILIAN, np.array([0.1, 0.5])), [False, True])
    assert not np.any(blocks(pol, Profile.EV, np.array([0.99, 1.0])))


@pytest.mark.req("RISK-09")
def test_load_policies_matches_shipped_config():
    pols = load_policies()
    assert set(pols) == set(Hazard)
    t, f, s, st, ice = (
        pols[Hazard.TORNADO],
        pols[Hazard.FLOOD],
        pols[Hazard.SMOKE],
        pols[Hazard.STORM],
        pols[Hazard.ICE],
    )
    assert t.exposure_weight[Profile.CIVILIAN] == 20 and t.block_at[Profile.CIVILIAN] == 0.5
    assert f.block_at[Profile.CIVILIAN] == 0.4 and f.block_at[Profile.EV] == 0.7
    assert s.exposure_weight[Profile.SENSITIVE] == 5 and s.exposure_weight[Profile.CIVILIAN] == 0.5
    assert st.beyond_horizon == "decay" and st.decay_tau_min == 30
    assert st.block_at[Profile.CIVILIAN] is None
    assert t.beyond_horizon == "drop"
    for h in (Hazard.FLOOD, Hazard.WILDFIRE, Hazard.ICE, Hazard.SMOKE):
        assert pols[h].beyond_horizon == "hold"
    assert ice.speed_factor(1.0) == pytest.approx(0.5)
    assert pols[Hazard.WILDFIRE].exposure_weight[Profile.EV] == 5


@pytest.mark.req("RISK-09")
def test_load_policies_errors_and_raw_config(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "hazards.yaml").read_text())
    raw = load_hazard_config()
    assert raw["hazards"]["storm"]["decay_zero_after_min"] == 120
    assert raw["hazards"]["tornado"]["blocked_linger_min"] == 15
    missing = dict(cfg, hazards={k: v for k, v in cfg["hazards"].items() if k != "smoke"})
    p1 = tmp_path / "missing.yaml"
    p1.write_text(yaml.safe_dump(missing))
    with pytest.raises(ValueError):
        load_policies(p1)
    bad = yaml.safe_load(yaml.safe_dump(cfg))
    bad["hazards"]["ice"]["beyond_horizon"] = "freeze"
    p2 = tmp_path / "bad.yaml"
    p2.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):
        load_policies(p2)


@pytest.mark.req("RISK-09")
def test_default_config_folder_follows_outrun_config_dir(tmp_path, monkeypatch):
    """load_policies()/load_hazard_config() resolve their default through get_paths().configs (OUTRUN_CONFIG_DIR)."""
    cfg = yaml.safe_load((ROOT / "configs" / "hazards.yaml").read_text())
    cfg["hazards"]["smoke"]["exposure_weight"]["sensitive"] = 7
    (tmp_path / "hazards.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("OUTRUN_CONFIG_DIR", str(tmp_path))
    assert load_hazard_config()["hazards"]["smoke"]["exposure_weight"]["sensitive"] == 7
    assert load_policies()[Hazard.SMOKE].exposure_weight[Profile.SENSITIVE] == 7


# ---------------------------------------------------------------- HORIZON-01
@pytest.mark.req("HORIZON-01")
def test_decay_storm_example(t0, risk_cube):
    out = _apply(risk_cube, _policy("decay", tau=30.0), t0, 60)
    assert out.shape == (12, 32, 32) and out.dtype == np.float32
    np.testing.assert_array_equal(out[:4], risk_cube[:4])
    L = risk_cube[3]
    np.testing.assert_allclose(out[4], L, rtol=1e-6)
    np.testing.assert_allclose(out[5], L * math.exp(-0.5), rtol=1e-5)
    np.testing.assert_allclose(out[6], L * math.exp(-1.0), rtol=1e-5)
    np.testing.assert_allclose(out[7], L * math.exp(-1.5), rtol=1e-5)
    assert float(np.abs(out[8:]).max()) == 0.0


@pytest.mark.req("HORIZON-01")
def test_decay_zero_after_relative_to_issue(t0, risk_cube):
    # issued 30 min before t0: zero once t_k - issued >= 120 min, i.e. from slot 6 (t_k = t0 + 90 min)
    pol = _policy("decay", tau=30.0)
    out = apply_beyond_horizon(
        risk_cube[:2], pol, t0=t0, issued=t0 - timedelta(minutes=30), valid_until=t0 + timedelta(minutes=30)
    )
    np.testing.assert_allclose(out[2], risk_cube[1], rtol=1e-6)
    np.testing.assert_allclose(out[5], risk_cube[1] * math.exp(-45 / 30), rtol=1e-5)
    assert float(np.abs(out[6:]).max()) == 0.0
    # zero_after_min=None: pure exponential decay to the end
    out2 = _apply(risk_cube, pol, t0, 60, zero_after_min=None)
    np.testing.assert_allclose(out2[11], risk_cube[3] * math.exp(-105 / 30), rtol=1e-5)


@pytest.mark.req("HORIZON-01")
@pytest.mark.parametrize("tau", [None, 0.0, -5.0])
def test_decay_requires_positive_tau(t0, risk_cube, tau):
    with pytest.raises(ValueError):
        _apply(risk_cube, _policy("decay", tau=tau), t0, 60)


# ---------------------------------------------------------------- HORIZON-02
@pytest.mark.req("HORIZON-02")
def test_drop_tornado_with_linger(t0, risk_cube):
    pol = _policy("drop", block=0.5)
    out = _apply(risk_cube, pol, t0, 30)
    np.testing.assert_array_equal(out[:2], risk_cube[:2])
    L = risk_cube[1]
    expected = np.where(L >= 0.5, L, 0.0)
    assert (L >= 0.5).any() and (L < 0.5).any()
    np.testing.assert_array_equal(out[2], expected.astype(np.float32))
    assert float(np.abs(out[3:]).max()) == 0.0


@pytest.mark.req("HORIZON-02")
def test_drop_threshold_is_min_block_and_none_never_lingers(t0, risk_cube):
    pol = HazardPolicy(
        block_at={Profile.CIVILIAN: 0.6, Profile.SENSITIVE: None, Profile.EV: 0.3},
        speed_factor=lambda p: 1.0,
        exposure_weight={p: 1.0 for p in Profile},
        beyond_horizon="drop",
        decay_tau_min=None,
    )
    out = _apply(risk_cube, pol, t0, 30)
    L = risk_cube[1]
    np.testing.assert_array_equal(out[2], np.where(L >= 0.3, L, 0.0).astype(np.float32))
    out_none = _apply(risk_cube, _policy("drop", block=None), t0, 30)
    assert float(np.abs(out_none[2:]).max()) == 0.0
    out_long = _apply(
        risk_cube, pol, t0, 30, linger_min=31.0
    )  # slots 2 and 3 (dt 0, 15) linger, slot 4 (dt 30) too
    np.testing.assert_array_equal(out_long[4], np.where(L >= 0.3, L, 0.0).astype(np.float32))
    assert float(np.abs(out_long[5:]).max()) == 0.0


# ---------------------------------------------------------------- HORIZON-03
@pytest.mark.req("HORIZON-03")
def test_hold_repeats_last_valid_slot(t0, risk_cube):
    out = _apply(risk_cube, _policy("hold"), t0, 90)
    np.testing.assert_array_equal(out[:6], risk_cube[:6])
    for k in range(6, 12):
        np.testing.assert_array_equal(out[k], risk_cube[5])


@pytest.mark.req("HORIZON-03")
def test_hold_extends_short_layer(t0, risk_cube):
    out = _apply(risk_cube[:3], _policy("hold"), t0, 360)
    assert out.shape == (12, 32, 32)
    np.testing.assert_array_equal(out[:3], risk_cube[:3])
    for k in range(3, 12):
        np.testing.assert_array_equal(out[k], risk_cube[2])
    full = _apply(risk_cube, _policy("hold"), t0, 360)
    np.testing.assert_array_equal(full, risk_cube)


# ---------------------------------------------------------------- HORIZON-04
@pytest.mark.req("HORIZON-04")
@pytest.mark.parametrize("rule,tau", [("hold", None), ("decay", 30.0), ("drop", None)])
def test_edge_override_vectors_and_no_mutation(t0, rule, tau):
    vec = np.linspace(0.2, 0.9, 12).astype(np.float32)
    before = vec.copy()
    out = _apply(vec, _policy(rule, tau=tau, block=0.5), t0, 60)
    assert out.shape == (12,) and out.dtype == np.float32
    np.testing.assert_array_equal(vec, before)
    assert out is not vec
    np.testing.assert_array_equal(out[:4], vec[:4])
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.req("HORIZON-04")
def test_beyond_horizon_errors(t0, risk_cube):
    hold = _policy("hold")
    with pytest.raises(ValueError):
        _apply(risk_cube, hold, t0, 0)  # horizon ends at t0
    with pytest.raises(ValueError):
        apply_beyond_horizon(
            risk_cube, hold, t0=t0, issued=t0 - timedelta(hours=2), valid_until=t0 - timedelta(hours=1)
        )
    with pytest.raises(ValueError):
        _apply(risk_cube[:0], hold, t0, 60)
    with pytest.raises(ValueError):
        apply_beyond_horizon(
            risk_cube, hold, t0=t0.replace(tzinfo=None), issued=t0, valid_until=t0 + timedelta(hours=1)
        )
    with pytest.raises(ValueError):
        _apply(risk_cube, _policy("freeze"), t0, 60)
