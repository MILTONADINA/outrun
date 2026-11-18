"""specs/15-evaluation.md: metrics (EVAL-01..EVAL-09). Synthetic arrays only."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from outrun.eval.metrics import (
    RouteTrace,
    auprc,
    brier,
    brier_skill_score,
    contingency,
    csi,
    csi_over_leads,
    ece,
    fss,
    mae,
    neighbourhood_csi,
    quantile_coverage,
    reliability_curve,
    rmse,
    roc_auc,
    route_trace,
    score_route,
)


# ---------------------------------------------------------------- EVAL-01
@pytest.mark.req("EVAL-01")
def test_contingency_counts_and_ratios():
    f = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    o = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
    c = contingency(f, o)
    assert (c.hits, c.misses, c.false_alarms, c.correct_negatives) == (2, 1, 1, 2)
    assert c.csi == pytest.approx(2 / 4)
    assert c.pod == pytest.approx(2 / 3)
    assert c.far == pytest.approx(1 / 3)


@pytest.mark.req("EVAL-01")
def test_contingency_mask_and_shape_errors():
    f = np.array([1, 1, 0, 0], dtype=bool)
    o = np.array([1, 0, 1, 0], dtype=bool)
    m = np.array([True, False, False, True])
    c = contingency(f, o, mask=m)
    assert (c.hits, c.misses, c.false_alarms, c.correct_negatives) == (1, 0, 0, 1)
    with pytest.raises(ValueError):
        contingency(f, o[:3])


@pytest.mark.req("EVAL-01")
def test_csi_threshold_is_inclusive_and_empty_is_nan():
    assert csi(np.array([0.5, 0.0]), np.array([0.5, 0.0]), threshold=0.5) == pytest.approx(1.0)
    assert csi(np.array([0.49, 0.6]), np.array([0.5, 0.0]), threshold=0.5) == pytest.approx(0.0)
    assert math.isnan(csi(np.zeros((4, 4)), np.zeros((4, 4)), threshold=74))
    with pytest.raises(ValueError):
        csi(np.zeros((4, 4)), np.zeros((4, 5)), threshold=74)


# ---------------------------------------------------------------- EVAL-02
def _lead_fields() -> tuple[np.ndarray, np.ndarray]:
    """3 leads of 6x6: lead 0 perfect (CSI 1), lead 1 no events (nan), lead 2 hits 1 misses 1 (CSI 0.5)."""
    f = np.zeros((3, 6, 6))
    o = np.zeros((3, 6, 6))
    f[0, 1, 1] = o[0, 1, 1] = 200
    f[0, 4, 2] = o[0, 4, 2] = 200
    o[2, 2, 2] = o[2, 3, 3] = 200
    f[2, 2, 2] = 200
    return f, o


@pytest.mark.req("EVAL-02")
def test_csi_over_leads_per_lead_and_nanmean():
    f, o = _lead_fields()
    out = csi_over_leads(f, o, thresholds=(74, 133))
    for thr in (74, 133):
        per = np.asarray(out["per_lead"][thr])
        assert per.shape == (3,)
        assert per[0] == pytest.approx(1.0)
        assert math.isnan(per[1])
        assert per[2] == pytest.approx(0.5)
        assert out["mean"][thr] == pytest.approx(0.75)
    assert out["mean_all"] == pytest.approx(0.75)


@pytest.mark.req("EVAL-02")
def test_csi_over_leads_pools_non_lead_axes():
    f, o = _lead_fields()
    fb, ob = np.stack([f, f]), np.stack([o, o])  # (N=2, L=3, 6, 6)
    out = csi_over_leads(fb, ob, thresholds=(74,), lead_axis=1)
    per = np.asarray(out["per_lead"][74])
    assert per.shape == (3,)
    assert per[0] == pytest.approx(1.0) and per[2] == pytest.approx(0.5)
    # threshold matters: values of 100 are events at 74 but not at 133
    f2 = np.full((2, 4, 4), 100.0)
    o2 = np.full((2, 4, 4), 100.0)
    out2 = csi_over_leads(f2, o2, thresholds=(74, 133))
    assert out2["mean"][74] == pytest.approx(1.0)
    assert math.isnan(out2["mean"][133])
    assert out2["mean_all"] == pytest.approx(1.0)


# ---------------------------------------------------------------- EVAL-03
@pytest.mark.req("EVAL-03")
def test_neighbourhood_csi_forgives_small_displacement():
    f = np.zeros((32, 32))
    o = np.zeros((32, 32))
    f[10, 10] = 1.0
    o[10, 11] = 1.0
    assert neighbourhood_csi(f, o, threshold=0.5, radius=1) == pytest.approx(1.0)
    assert neighbourhood_csi(f, o, threshold=0.5, radius=0) == pytest.approx(0.0)
    o2 = np.zeros((32, 32))
    o2[10, 13] = 1.0  # 3 cells away: miss + false alarm at radius 1
    assert neighbourhood_csi(f, o2, threshold=0.5, radius=1) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        neighbourhood_csi(f, o, threshold=0.5, radius=-1)


@pytest.mark.req("EVAL-03")
def test_neighbourhood_csi_radius_zero_equals_csi(rng):
    f = rng.random((32, 32))
    o = rng.random((32, 32))
    assert neighbourhood_csi(f, o, 0.7, 0) == pytest.approx(csi(f, o, 0.7))


# ---------------------------------------------------------------- EVAL-04
@pytest.mark.req("EVAL-04")
def test_fss_hand_computed_shift():
    f = np.zeros((32, 32))
    o = np.zeros((32, 32))
    f[10, 10] = 1.0
    o[10, 11] = 1.0
    assert fss(f, o, threshold=0.5, window=1) == pytest.approx(0.0)
    # window 3: 6 of 9 neighbourhood cells overlap -> 1 - (6/81) / (18/81) = 2/3
    assert fss(f, o, threshold=0.5, window=3) == pytest.approx(2 / 3)


@pytest.mark.req("EVAL-04")
def test_fss_identical_empty_and_bad_window(grid_field):
    assert fss(grid_field, grid_field, threshold=0.5, window=5) == pytest.approx(1.0)
    assert math.isnan(fss(np.zeros((32, 32)), np.zeros((32, 32)), threshold=0.5, window=3))
    for bad in (0, 2, 4, -3):
        with pytest.raises(ValueError):
            fss(grid_field, grid_field, threshold=0.5, window=bad)


@pytest.mark.req("EVAL-04")
def test_fss_zero_padding_at_border():
    # an event in the corner: with zero padding its fraction is 1/9 for window 3 in the 4 cells around it
    f = np.zeros((8, 8))
    f[0, 0] = 1.0
    o = f.copy()
    assert fss(f, o, 0.5, 3) == pytest.approx(1.0)
    o2 = np.zeros((8, 8))
    o2[0, 1] = 1.0
    # Pf nonzero on rows 0-1, cols 0-1 (4 cells); Po on rows 0-1, cols 0-2 (6 cells); overlap 4 cells
    # sum (Pf-Po)^2 = 2 * (1/9)^2 ; sum Pf^2 + sum Po^2 = (4 + 6) / 81
    assert fss(f, o2, 0.5, 3) == pytest.approx(1 - 2 / 10)


# ---------------------------------------------------------------- EVAL-05
@pytest.mark.req("EVAL-05")
def test_auc_and_auprc_match_sklearn(rng):
    y = rng.integers(0, 2, 400)
    s = rng.random(400) + 0.3 * y
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert auprc(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)


@pytest.mark.req("EVAL-05")
def test_auc_drops_uncertain_labels_and_mask(rng):
    y = rng.integers(0, 2, 300)
    s = rng.random(300) + 0.2 * y
    y_unc = y.copy()
    y_unc[::7] = -1
    keep = y_unc != -1
    assert roc_auc(y_unc, s) == pytest.approx(roc_auc_score(y[keep], s[keep]), abs=1e-12)
    assert auprc(y_unc, s) == pytest.approx(average_precision_score(y[keep], s[keep]), abs=1e-12)
    m = np.arange(300) < 150
    assert roc_auc(y, s, mask=m) == pytest.approx(roc_auc_score(y[m], s[m]), abs=1e-12)


@pytest.mark.req("EVAL-05")
def test_auc_degenerate_and_invalid_inputs():
    assert math.isnan(roc_auc(np.ones(5, int), np.linspace(0, 1, 5)))
    assert math.isnan(auprc(np.zeros(5, int), np.linspace(0, 1, 5)))
    with pytest.raises(ValueError):
        roc_auc(np.array([0, 1, 0]), np.array([0.1, np.nan, 0.3]))
    with pytest.raises(ValueError):
        auprc(np.array([0, 2, 1]), np.array([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------- EVAL-06
@pytest.mark.req("EVAL-06")
def test_brier_and_skill_score():
    y = np.array([1, 0])
    p = np.array([0.8, 0.4])
    assert brier(y, p) == pytest.approx(0.1)
    assert brier_skill_score(y, p) == pytest.approx(1 - 0.1 / 0.25)  # climatology base rate 0.5
    assert brier_skill_score(y, p, ref_probs=0.5) == pytest.approx(0.6)
    assert math.isnan(
        brier_skill_score(y, p, ref_probs=np.array([1.0, 0.0]))
    )  # perfect reference: BS_ref = 0


@pytest.mark.req("EVAL-06")
def test_brier_drops_uncertain_and_validates():
    y = np.array([1, -1, 0])
    p = np.array([0.8, 0.9, 0.4])
    assert brier(y, p) == pytest.approx(0.1)
    assert math.isnan(brier_skill_score(np.ones(3, int), np.full(3, 0.7)))  # climatology 1 -> BS_ref 0
    with pytest.raises(ValueError):
        brier(np.array([1, 0]), np.array([1.2, 0.0]))
    with pytest.raises(ValueError):
        brier(np.array([1, 0]), np.array([0.5, -0.1]))


# ---------------------------------------------------------------- EVAL-07
@pytest.mark.req("EVAL-07")
def test_reliability_curve_bins():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.05, 0.15, 0.15, 1.0])
    rc = reliability_curve(y, p, n_bins=10)
    assert np.allclose(rc.bin_edges, np.linspace(0, 1, 11))
    assert rc.counts.tolist() == [1, 1 + 1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert rc.mean_pred[0] == pytest.approx(0.05) and rc.obs_freq[0] == pytest.approx(0.0)
    assert rc.mean_pred[1] == pytest.approx(0.15) and rc.obs_freq[1] == pytest.approx(0.5)
    assert rc.mean_pred[9] == pytest.approx(1.0) and rc.obs_freq[9] == pytest.approx(1.0)  # 1.0 in last bin
    assert np.isnan(rc.mean_pred[2]) and np.isnan(rc.obs_freq[5])


@pytest.mark.req("EVAL-07")
def test_ece_values():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.05, 0.15, 0.15, 1.0])
    assert ece(y, p, n_bins=10) == pytest.approx(0.25 * 0.05 + 0.5 * 0.35 + 0.25 * 0.0)
    assert ece(np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0]), np.full(10, 0.3)) == pytest.approx(0.0)
    assert ece(np.zeros(5, int), np.full(5, 0.92)) == pytest.approx(0.92)
    with pytest.raises(ValueError):
        ece(np.zeros(3, int), np.full(3, 0.5), n_bins=0)


# ---------------------------------------------------------------- EVAL-08
@pytest.mark.req("EVAL-08")
def test_rmse_mae_skip_nan_pairs():
    y = np.array([1.0, 2.0, np.nan, 4.0])
    p = np.array([1.0, 4.0, 3.0, np.nan])
    assert rmse(y, p) == pytest.approx(math.sqrt(2.0))
    assert mae(y, p) == pytest.approx(1.0)
    m = np.array([False, True, True, True])
    assert rmse(y, p, mask=m) == pytest.approx(2.0)
    assert math.isnan(rmse(np.array([np.nan]), np.array([1.0])))
    with pytest.raises(ValueError):
        rmse(np.zeros(3), np.zeros(4))


@pytest.mark.req("EVAL-08")
def test_quantile_coverage():
    y = np.array([1.0, 2.0, 3.0, np.nan])
    q = np.full(4, 2.0)
    assert quantile_coverage(y, q) == pytest.approx(2 / 3)
    assert math.isnan(quantile_coverage(np.array([np.nan]), np.array([1.0])))


# ---------------------------------------------------------------- EVAL-09
def _risk() -> np.ndarray:
    r = np.zeros((3, 4))
    r[0] = 0.2
    r[2] = [0.1, 0.4, 0.6, 0.0]
    return r


@pytest.mark.req("EVAL-09")
def test_score_route_exposure_includes_waits_and_slots():
    trace = RouteTrace(edges=(0, 2), t_enter_s=(0.0, 1000.0), t_exit_s=(600.0, 1900.0))
    s = score_route(trace, _risk(), slot_s=900)
    # edge 0: [0,600) at 0.2; edge 2 occupancy [600,1900): 300 s @0.1, 900 s @0.4, 100 s @0.6
    assert s.exposure_risk_min == pytest.approx((0.2 * 600 + 0.1 * 300 + 0.4 * 900 + 0.6 * 100) / 60)
    assert s.travel_time_s == pytest.approx(1900.0)
    assert s.intersects is True  # traversal touches slot 2 (0.6 >= 0.5)
    assert s.blocked_crossings == 0


@pytest.mark.req("EVAL-09")
def test_score_route_intersection_uses_traversal_not_wait():
    r = _risk()
    r[2] = [0.9, 0.4, 0.3, 0.0]  # high only in slot 0, where the route merely waits
    trace = RouteTrace(edges=(0, 2), t_enter_s=(0.0, 1000.0), t_exit_s=(600.0, 1900.0))
    s = score_route(trace, r, slot_s=900)
    assert s.intersects is False
    assert s.exposure_risk_min == pytest.approx((0.2 * 600 + 0.9 * 300 + 0.4 * 900 + 0.3 * 100) / 60)
    assert score_route(trace, r, slot_s=900, threshold=0.4).intersects is True  # inclusive threshold


@pytest.mark.req("EVAL-09")
def test_score_route_last_slot_holds_and_blocked_crossings():
    r = np.zeros((3, 4))
    r[1, 3] = 1.0
    long = RouteTrace(edges=(1,), t_enter_s=(0.0,), t_exit_s=(5400.0,))
    assert score_route(long, r, slot_s=900).exposure_risk_min == pytest.approx(2700 / 60)
    blocked = np.zeros((3, 4), dtype=bool)
    blocked[2, 2] = True
    trace = RouteTrace(edges=(0, 2), t_enter_s=(0.0, 1000.0), t_exit_s=(600.0, 1900.0))
    s = score_route(trace, np.zeros((3, 4)), blocked, slot_s=900, crossing_edges={2})
    assert s.intersects is True and s.blocked_crossings == 1
    assert score_route(trace, np.zeros((3, 4)), blocked, slot_s=900).blocked_crossings == 0
    empty = score_route(RouteTrace((), (), ()), r)
    assert empty.travel_time_s == 0.0 and empty.exposure_risk_min == 0.0 and empty.intersects is False


@pytest.mark.req("EVAL-09")
def test_score_route_validation():
    r = np.zeros((3, 4))
    with pytest.raises(ValueError):
        score_route(RouteTrace((0,), (100.0,), (50.0,)), r)  # exit before enter
    with pytest.raises(ValueError):
        score_route(RouteTrace((0, 1), (0.0, 50.0), (100.0, 200.0)), r)  # enters edge 1 before leaving edge 0
    with pytest.raises(ValueError):
        score_route(RouteTrace((7,), (0.0,), (10.0,)), r)  # edge out of range
    with pytest.raises(ValueError):
        score_route(RouteTrace((0, 1), (0.0,), (10.0,)), r)  # length mismatch


@pytest.mark.req("EVAL-09")
def test_route_trace_from_router_route():
    from types import SimpleNamespace

    # spec 04 Route: waits 100 s at the origin and 50 s at the middle node; t0 = 1800 s
    route = SimpleNamespace(
        nodes=(0, 1, 2), edges=(4, 7), depart_s=(1900.0, 2550.0), waits_s=(100.0, 50.0), start_s=1800.0,
        arrive_s=3000.0,
    )  # fmt: skip
    tr = route_trace(route, t0_s=1800.0)
    assert tr.edges == (4, 7)
    assert tr.t_enter_s == pytest.approx((100.0, 750.0))
    assert tr.t_exit_s == pytest.approx((700.0, 1200.0))
    assert tr.travel_time_s == pytest.approx(1200.0)
    assert tr.t_start_s == pytest.approx(0.0) and tr.wait_nodes == (0, 1)
    # default t0_s = 0: scenario clock, the start is kept
    tr0 = route_trace(route)
    assert tr0.t_start_s == pytest.approx(1800.0) and tr0.t_enter_s == pytest.approx((1900.0, 2550.0))
    assert tr0.travel_time_s == pytest.approx(1200.0)


@pytest.mark.req("EVAL-09")
def test_score_route_late_start_uses_scenario_slots():
    r = np.zeros((3, 4))
    r[0, 1] = 0.6  # edge 0 hazardous only in slot 1 = [900, 1800)
    late = RouteTrace(edges=(0,), t_enter_s=(900.0,), t_exit_s=(1500.0,), t_start_s=900.0)
    s = score_route(late, r, slot_s=900)
    assert s.exposure_risk_min == pytest.approx(0.6 * 600 / 60)  # no pre-start interval is charged
    assert s.travel_time_s == pytest.approx(600.0) and s.intersects is True
    with pytest.raises(ValueError):  # enters before it starts
        score_route(RouteTrace((0,), (800.0,), (1500.0,), t_start_s=900.0), r)


@pytest.mark.req("EVAL-09")
def test_score_route_waits_use_node_risk_when_given():
    r = np.zeros((3, 4))
    r[2, 0] = 0.9  # edge 2 hazardous in slot 0 only
    node = np.zeros((5, 4))
    node[3, 0] = 0.1  # tail node of edge 2 is nearly safe
    trace = RouteTrace(edges=(2,), t_enter_s=(900.0,), t_exit_s=(1200.0,), t_start_s=0.0, wait_nodes=(3,))
    assert score_route(trace, r, slot_s=900).exposure_risk_min == pytest.approx(
        0.9 * 900 / 60
    )  # edge fallback
    s = score_route(trace, r, slot_s=900, node_risk=node)
    assert s.exposure_risk_min == pytest.approx(0.1 * 900 / 60)
    assert s.intersects is False  # traversal in slot 1 only
    with pytest.raises(ValueError):  # wait node outside node_risk
        score_route(RouteTrace((2,), (0.0,), (10.0,), wait_nodes=(9,)), r, node_risk=node)
    with pytest.raises(ValueError):  # wait_nodes length mismatch
        score_route(RouteTrace((2,), (0.0,), (10.0,), wait_nodes=(3, 4)), r, node_risk=node)
