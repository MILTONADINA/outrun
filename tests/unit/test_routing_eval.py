"""specs/15-evaluation.md: routing evaluation (EVAL-13..EVAL-15). Synthetic routes; the router is a stub callable."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from outrun.eval.metrics import RouteTrace
from outrun.eval.routing_eval import (
    PLANNER_ORDER,
    RISK_SOURCE,
    PlannerKind,
    evaluate_planners,
    sample_origins,
)

K = PlannerKind

# One edge per planner. Observed risk is constant over the 4 slots of each edge.
EDGE = {K.NO_HAZARD: 0, K.CURRENT_ONLY: 4, K.BASELINE: 3, K.MODEL: 1, K.ORACLE: 2}
OBS_RISK = np.repeat(np.array([[0.8], [0.4], [0.0], [0.6], [0.7]]), 4, axis=1)  # (5 edges, 4 slots)
EXIT_S = {K.NO_HAZARD: 600.0, K.CURRENT_ONLY: 600.0, K.BASELINE: 600.0, K.MODEL: 600.0, K.ORACLE: 900.0}


def _make_plan(fail: set[tuple[PlannerKind, int]] = frozenset()):
    calls: Counter = Counter()

    def plan(kind: PlannerKind, origin: int) -> RouteTrace | None:
        calls[(kind, origin)] += 1
        if (kind, origin) in fail:
            return None
        return RouteTrace(edges=(EDGE[kind],), t_enter_s=(0.0,), t_exit_s=(EXIT_S[kind],))

    return plan, calls


# ---------------------------------------------------------------- EVAL-13
@pytest.mark.req("EVAL-13")
def test_planner_kinds_and_risk_sources():
    assert [k.value for k in PLANNER_ORDER] == ["a", "a_prime", "b", "c", "d"]
    assert set(PlannerKind) == set(PLANNER_ORDER)
    assert [RISK_SOURCE[k] for k in PLANNER_ORDER] == [
        "none",
        "current-only",
        "baseline",
        "model",
        "observed",
    ]


@pytest.mark.req("EVAL-13")
def test_sample_origins_population_weighted():
    nodes = np.array([10, 20, 30])
    w = np.array([0.0, 1.0, 3.0])
    o = sample_origins(nodes, w, n=4000, seed=0)
    assert o.shape == (4000,)
    assert set(o.tolist()) <= {20, 30}
    share = float(np.mean(o == 30))
    assert 0.72 < share < 0.78
    assert np.array_equal(o, sample_origins(nodes, w, n=4000, seed=0))
    assert len(sample_origins(nodes, w)) == 500


@pytest.mark.req("EVAL-13")
def test_sample_origins_errors():
    nodes = np.array([1, 2])
    for w in ([1.0], [-1.0, 2.0], [0.0, 0.0], [np.nan, 1.0]):
        with pytest.raises(ValueError):
            sample_origins(nodes, np.array(w), n=5)
    with pytest.raises(ValueError):
        sample_origins(nodes, np.array([1.0, 1.0]), n=0)


# ---------------------------------------------------------------- EVAL-14
@pytest.mark.req("EVAL-14")
def test_plans_once_per_planner_and_scores_on_observed():
    plan, calls = _make_plan(fail={(K.BASELINE, 3)})
    predicted = {K.MODEL: np.zeros_like(OBS_RISK)}  # the DL forecast saw nothing; observed says otherwise
    res = evaluate_planners([0, 1, 2, 3], plan, OBS_RISK, predicted_risk=predicted, n_resamples=50, seed=0)
    assert set(calls) == {(k, o) for k in PLANNER_ORDER for o in range(4)}
    assert all(v == 1 for v in calls.values())
    assert res.n_origins == 4 and res.n_paired == 3
    s = res.summaries
    assert s[K.BASELINE].n_routed == 3 and s[K.BASELINE].n_failed == 1
    assert s[K.MODEL].n_routed == 4 and s[K.MODEL].n_failed == 0
    # observed exposure = risk x 10 minutes
    expected = {K.NO_HAZARD: 8.0, K.CURRENT_ONLY: 7.0, K.BASELINE: 6.0, K.MODEL: 4.0, K.ORACLE: 0.0}
    for k, e in expected.items():
        assert s[k].mean_exposure_risk_min == pytest.approx(e), k
    assert s[K.MODEL].mean_predicted_exposure == pytest.approx(0.0)
    assert s[K.NO_HAZARD].mean_predicted_exposure is None


@pytest.mark.req("EVAL-14")
def test_summary_fields_intersect_ratio_blocked():
    plan, _ = _make_plan()
    blocked = np.zeros_like(OBS_RISK, dtype=bool)
    blocked[0] = True
    res = evaluate_planners(
        [5, 6, 7], plan, OBS_RISK, observed_blocked=blocked, crossing_edges={0}, n_resamples=50
    )
    s = res.summaries
    assert s[K.NO_HAZARD].pct_intersect == pytest.approx(100.0)
    assert s[K.BASELINE].pct_intersect == pytest.approx(100.0)
    assert s[K.MODEL].pct_intersect == pytest.approx(0.0)
    assert s[K.NO_HAZARD].mean_travel_time_ratio == pytest.approx(1.0)
    assert s[K.ORACLE].mean_travel_time_ratio == pytest.approx(1.5)
    assert s[K.NO_HAZARD].blocked_crossings == 3
    assert s[K.MODEL].blocked_crossings == 0


@pytest.mark.req("EVAL-14")
def test_empty_paired_set_and_no_origins():
    plan, _ = _make_plan(fail={(K.ORACLE, 0), (K.ORACLE, 1)})
    res = evaluate_planners([0, 1], plan, OBS_RISK, n_resamples=20)
    assert res.n_paired == 0
    assert math.isnan(res.summaries[K.MODEL].mean_exposure_risk_min)
    assert math.isnan(res.summaries[K.MODEL].pct_intersect)
    assert res.summaries[K.MODEL].blocked_crossings == 0
    assert res.summaries[K.ORACLE].n_failed == 2
    with pytest.raises(ValueError):
        evaluate_planners([], plan, OBS_RISK)


# ---------------------------------------------------------------- EVAL-15
@pytest.mark.req("EVAL-15")
def test_headline_comparisons():
    plan, _ = _make_plan()
    res = evaluate_planners(list(range(6)), plan, OBS_RISK, n_resamples=100, seed=1)
    assert res.gap_closed.estimate == pytest.approx(0.5)  # (8 - 4) / (8 - 0)
    assert res.gap_closed.low == pytest.approx(0.5) and res.gap_closed.high == pytest.approx(0.5)
    assert res.c_minus_b.estimate == pytest.approx(-2.0)
    assert res.c_minus_a_prime.estimate == pytest.approx(-3.0)
    assert res.gap_closed.n_blocks == 6


@pytest.mark.req("EVAL-15")
def test_headline_needs_two_paired_origins():
    plan, _ = _make_plan()
    res = evaluate_planners([0], plan, OBS_RISK, n_resamples=20)
    for ci in (res.gap_closed, res.c_minus_b, res.c_minus_a_prime):
        assert math.isnan(ci.estimate) and math.isnan(ci.low) and math.isnan(ci.high)
        assert ci.n_resamples == 0 and ci.n_blocks == 1


@pytest.mark.req("EVAL-14")
def test_waits_scored_at_observed_node_risk():
    def plan(kind: PlannerKind, origin: int) -> RouteTrace:
        # every planner waits 300 s at node 0, then drives its edge until EXIT_S
        return RouteTrace(edges=(EDGE[kind],), t_enter_s=(300.0,), t_exit_s=(EXIT_S[kind],), wait_nodes=(0,))

    node = np.full((1, 4), 0.1)
    res = evaluate_planners([0, 1], plan, OBS_RISK, n_resamples=20, seed=0, observed_node_risk=node)
    # NO_HAZARD: 300 s wait at 0.1 + 300 s on edge 0 at 0.8 -> 0.5 + 4.0 risk-minutes
    assert res.summaries[K.NO_HAZARD].mean_exposure_risk_min == pytest.approx(0.1 * 5 + 0.8 * 5)
    # ORACLE: 300 s wait at 0.1 + 600 s on edge 2 at 0.0
    assert res.summaries[K.ORACLE].mean_exposure_risk_min == pytest.approx(0.5)
    without = evaluate_planners([0, 1], plan, OBS_RISK, n_resamples=20, seed=0)
    assert without.summaries[K.NO_HAZARD].mean_exposure_risk_min == pytest.approx(8.0)  # edge fallback
