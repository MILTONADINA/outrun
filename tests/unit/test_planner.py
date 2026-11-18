"""Spec 04-astar: planners, replay scoring and the shelter-in-place rule (PLAN-01..PLAN-05).

Risk arrays use H = 6 in Hazard order (STORM, TORNADO, FLOOD, WILDFIRE, ICE, SMOKE).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from tests.fixtures.graphs.graphfix import make_road_graph

NAN = float("nan")
H = 6
STORM, TORNADO, WILDFIRE, SMOKE = 0, 1, 3, 5


def _edge(g, u, v):
    for e in g.out_edges(u):
        if int(g.dst[e]) == v:
            return e
    raise KeyError((u, v))


def _three_routes():
    """0 -> {1 top 100+100 s, 2 mid 60+60 s, 3 bottom 150+150 s} -> 4."""
    src = [0, 1, 0, 2, 0, 3]
    dst = [1, 4, 2, 4, 3, 4]
    length = [1000.0, 1000.0, 600.0, 600.0, 1500.0, 1500.0]
    lon = [-97.50, -97.49, -97.49, -97.49, -97.48]
    lat = [35.50, 35.51, 35.50, 35.49, 35.50]
    return make_road_graph(lon, lat, src, dst, length, [10.0] * 6)[0]


def _forecast_model(g, block_storm=NAN):
    """slot_s = 60 s. Mid road hazardous only in slot 0; top road's second edge hazardous from slot 1 on."""
    from outrun.routing.cost import CostModel

    risk = np.zeros((g.n_edges, H, 12))
    risk[_edge(g, 0, 2), STORM, 0] = 0.9
    risk[_edge(g, 2, 4), STORM, 0] = 0.9
    risk[_edge(g, 1, 4), STORM, 1:] = 0.9
    w = np.zeros(H)
    w[STORM] = 10.0
    b = np.full(H, NAN)
    b[STORM] = block_storm
    return CostModel(g.free_flow_s, risk, w, b, slot_s=60.0)


# ---------------------------------------------------------------- PLAN-01
@pytest.mark.req("PLAN-01")
def test_plan_route_returns_safe_route():
    from outrun.routing.astar import astar
    from outrun.routing.planner import plan_route, route_exposure

    g = _three_routes()
    m = _forecast_model(g)
    res = plan_route(g, m, 0, [4])
    assert res.kind == "route" and res.hazard_arrival_s is None and res.facility is None
    best = astar(g, m, 0, 4)
    assert res.route.cost == pytest.approx(best.cost)
    assert best.cost == pytest.approx(180.0)
    assert res.route.nodes == [0, 2, 4] and res.route.waits_s[0] == pytest.approx(60.0)
    assert res.exposure_risk_min == pytest.approx(route_exposure(res.route, m))


@pytest.mark.req("PLAN-01")
def test_plan_route_no_route():
    from outrun.routing.cost import CostModel
    from outrun.routing.planner import plan_route

    g = _three_routes()
    risk = np.zeros((g.n_edges, H, 12))
    risk[g.dst == 4, STORM, :] = 1.0
    b = np.full(H, NAN)
    b[STORM] = 0.5
    res = plan_route(g, CostModel(g.free_flow_s, risk, np.zeros(H), b), 0, 4)
    assert res.kind == "no_route" and res.route is None


# ---------------------------------------------------------------- PLAN-02
@pytest.mark.req("PLAN-02")
def test_transform_risk_sources():
    from outrun.routing.planner import RiskSource, transform_risk

    rng = np.random.default_rng(0)
    risk = rng.uniform(0, 1, (5, H, 12)).astype(np.float32)
    before = risk.copy()
    assert (
        np.all(transform_risk(risk, RiskSource.NONE) == 0)
        and transform_risk(risk, RiskSource.NONE).shape == risk.shape
    )
    cur = transform_risk(risk, RiskSource.CURRENT_ONLY)
    np.testing.assert_array_equal(cur, np.repeat(risk[:, :, :1], 12, axis=2))
    for src in (RiskSource.MODEL, RiskSource.BASELINE, RiskSource.OBSERVED):
        out = transform_risk(risk, src)
        np.testing.assert_array_equal(out, risk)
        assert out is not risk
    cur2 = transform_risk(risk, RiskSource.CURRENT_ONLY, t0_slot=2)
    np.testing.assert_array_equal(cur2, np.repeat(risk[:, :, 2:3], 12, axis=2))
    np.testing.assert_array_equal(risk, before)  # input never modified
    assert RiskSource("current-only") is RiskSource.CURRENT_ONLY
    assert {s.value for s in RiskSource} == {"model", "baseline", "observed", "none", "current-only"}


@pytest.mark.req("PLAN-02")
def test_baseline_planners_differ_from_forecast():
    from outrun.routing.astar import astar
    from outrun.routing.planner import plan_avoid_current_only, plan_shortest_time

    g = _three_routes()
    m = _forecast_model(g)
    a = plan_shortest_time(g, m, 0, 4)
    assert a.nodes == [0, 2, 4] and sum(a.waits_s) == 0.0 and a.cost == pytest.approx(120.0)
    a1 = plan_avoid_current_only(g, m, 0, 4)
    assert a1.nodes == [
        0,
        1,
        4,
    ]  # current hazard on the mid road; ignores the hazard arriving on the top road
    c = astar(g, m, 0, 4)
    assert c.nodes == [0, 2, 4] and c.cost == pytest.approx(180.0)
    # a request at t0_s = 60 freezes slot 1: the mid road is clear then and the top road's second edge is not
    late = plan_avoid_current_only(g, m, 0, 4, t0_s=60.0)
    assert late.nodes == [0, 2, 4] and late.start_s == pytest.approx(60.0) and sum(late.waits_s) == 0.0


# ---------------------------------------------------------------- PLAN-03
@pytest.mark.req("PLAN-03")
def test_replay_on_same_model_reproduces_route():
    from outrun.routing.astar import astar
    from outrun.routing.planner import replay

    g = _three_routes()
    m = _forecast_model(g)
    r = astar(g, m, 0, 4)
    s = replay(r, m)
    assert s.travel_s == pytest.approx(r.travel_s) and s.cost == pytest.approx(r.cost)
    assert s.blocked_edges == 0 and not s.intersects_hazard


@pytest.mark.req("PLAN-03")
def test_replay_scores_plan_on_observed():
    from outrun.routing.planner import plan_avoid_current_only, plan_shortest_time, replay

    g = _three_routes()
    observed = _forecast_model(g)
    a1 = plan_avoid_current_only(g, observed, 0, 4)  # top road
    s = replay(a1, observed)
    assert s.travel_s == pytest.approx(200.0)
    assert s.cost == pytest.approx(100.0 + 100.0 * (1 + 10 * 0.9))
    assert s.exposure_risk_min == pytest.approx(100.0 / 60.0 * 0.9)
    assert s.intersects_hazard and s.blocked_edges == 0
    sb = replay(a1, _forecast_model(g, block_storm=0.5))
    assert sb.blocked_edges == 1 and math.isfinite(sb.cost) and sb.cost == pytest.approx(s.cost)
    a = plan_shortest_time(g, observed, 0, 4)
    sa = replay(a, observed)
    assert sa.cost == pytest.approx(600.0 + 60.0) and sa.exposure_risk_min == pytest.approx(0.9)


@pytest.mark.req("PLAN-03")
@pytest.mark.req("ASTAR-20")
def test_replay_matches_eval_scoring_on_the_scoring_model():
    from outrun.eval.metrics import score_route
    from outrun.routing.planner import plan_avoid_current_only, replay, replay_trace

    g = _three_routes()
    observed = _forecast_model(g, block_storm=0.5)
    a1 = plan_avoid_current_only(g, _forecast_model(g), 0, 4)  # top road
    tr = replay_trace(a1, observed)
    assert list(tr.edges) == list(a1.edges)
    np.testing.assert_allclose(tr.t_exit_s, [100.0, 200.0])
    assert tr.t_start_s == pytest.approx(0.0) and tuple(tr.wait_nodes) == (0, 1)
    node = observed.combined_node_risk_table(g.n_nodes)
    assert node.shape == (g.n_nodes, 12) and not node.any()  # no node risk -> zeros
    ref = score_route(
        tr, observed.combined_risk_table(), observed.blocked_table(), slot_s=60.0, node_risk=node
    )
    s = replay(a1, observed)
    assert ref.exposure_risk_min == pytest.approx(100.0 / 60.0 * 0.9)
    assert s.exposure_risk_min == pytest.approx(ref.exposure_risk_min)
    assert bool(ref.intersects)
    assert bool(s.intersects_hazard)


@pytest.mark.req("PLAN-03")
@pytest.mark.req("PLAN-05")
def test_waiting_at_a_safe_node_is_credited():
    from outrun.routing.astar import astar
    from outrun.routing.planner import replay, route_exposure

    g = _three_routes()
    m = _forecast_model(g)
    r = astar(g, m, 0, 4)  # waits 60 s at the (safe) origin while the mid road is hazardous, then drives it
    assert r.nodes == [0, 2, 4] and r.waits_s[0] == pytest.approx(60.0)
    # the wait is charged at the origin's node risk (0, no node risk), not at edge 0->2's 0.9
    assert route_exposure(r, m) == pytest.approx(0.0)
    assert replay(r, m).exposure_risk_min == pytest.approx(0.0)


@pytest.mark.req("PLAN-03")
@pytest.mark.req("PLAN-05")
def test_late_start_is_scored_on_scenario_slots():
    from outrun.routing.astar import Route
    from outrun.routing.cost import CostModel
    from outrun.routing.planner import replay, replay_trace, route_exposure

    g = _three_routes()
    e02, e24 = _edge(g, 0, 2), _edge(g, 2, 4)
    risk = np.zeros((g.n_edges, H, 12))
    risk[e02, STORM, 1] = 0.3  # slot 1 = [60, 120)
    m = CostModel(g.free_flow_s, risk, np.full(H, 7.0), np.full(H, NAN), slot_s=60.0)
    r = Route(nodes=[0, 2, 4], edges=[e02, e24], depart_s=[60.0, 120.0], waits_s=[0.0, 0.0], start_s=60.0,
              arrive_s=180.0, cost=0.0, expansions=0)  # fmt: skip
    assert route_exposure(r, m) == pytest.approx(0.3)  # edge 0->2 driven in slot 1, not slot 0
    tr = replay_trace(r, m)
    assert tr.t_start_s == pytest.approx(60.0)
    np.testing.assert_allclose(tr.t_enter_s, [60.0, 120.0])
    np.testing.assert_allclose(tr.t_exit_s, [120.0, 180.0])
    assert replay(r, m).exposure_risk_min == pytest.approx(0.3)


# ---------------------------------------------------------------- PLAN-05
@pytest.mark.req("PLAN-05")
def test_route_exposure_is_the_eval_score_route_definition():
    from outrun.eval.metrics import route_trace, score_route
    from outrun.routing.astar import Route
    from outrun.routing.cost import CostModel
    from outrun.routing.planner import route_exposure

    g = _three_routes()
    e02, e24 = _edge(g, 0, 2), _edge(g, 2, 4)
    risk = np.zeros((g.n_edges, H, 12))
    risk[e02, STORM, 1] = 0.3
    risk[e02, SMOKE, 1] = 0.1
    node_risk = np.zeros((g.n_nodes, H, 12))
    node_risk[0, STORM, 0] = 0.5
    m = CostModel(g.free_flow_s, risk, np.full(H, 7.0), np.full(H, NAN), node_risk=node_risk, slot_s=60.0)
    r = Route(
        nodes=[0, 2, 4],
        edges=[e02, e24],
        depart_s=[60.0, 120.0],
        waits_s=[60.0, 0.0],
        start_s=0.0,
        arrive_s=180.0,
        cost=0.0,
        expansions=0,
    )
    # the 60 s wait at node 0 (slot 0) is charged at its node risk 0.5; the traversal of 0->2 (slot 1) at
    # max_h = 0.3 (hazards combined with max, not summed; ASTAR-20, EVAL-09)
    assert route_exposure(r, m) == pytest.approx(60 / 60 * 0.5 + 60 / 60 * 0.3)
    ref = score_route(route_trace(r), m.combined_risk_table(), m.blocked_table(), slot_s=60.0,
                      node_risk=m.combined_node_risk_table())  # fmt: skip
    assert route_exposure(r, m) == pytest.approx(ref.exposure_risk_min)
    empty = Route(
        nodes=[0], edges=[], depart_s=[], waits_s=[], start_s=0.0, arrive_s=0.0, cost=0.0, expansions=0
    )
    assert route_exposure(empty, m) == 0.0


# ---------------------------------------------------------------- PLAN-04
def _sip_graph():
    """0 -> 1 -> 2 (goal, 600 s each); hardened candidates 3 (120 s from 0) and 4 (400 s from 0)."""
    src = [0, 1, 0, 0]
    dst = [1, 2, 3, 4]
    length = [6000.0, 6000.0, 1200.0, 4000.0]
    lon = [-97.50, -97.45, -97.40, -97.51, -97.54]
    lat = [35.50, 35.50, 35.50, 35.50, 35.50]
    return make_road_graph(lon, lat, src, dst, length, [10.0] * 4)[0]


def _sip_model(g, from_slot=None, hazard=TORNADO, value=0.8, nodes=(0, 1)):
    from outrun.routing.cost import CostModel

    node_risk = np.zeros((g.n_nodes, H, 12))
    if from_slot is not None:
        node_risk[list(nodes), hazard, from_slot:] = value
    w = np.zeros(H)
    w[TORNADO] = 20.0
    b = np.full(H, NAN)
    b[TORNADO] = 0.5  # wildfire NaN -> default threshold 0.5
    return CostModel(g.free_flow_s, np.zeros((g.n_edges, H, 12)), w, b, node_risk=node_risk)


@pytest.mark.req("PLAN-04")
def test_sip_route_exits_before_hazard_arrives():
    from outrun.routing.planner import plan_route

    g = _sip_graph()
    res = plan_route(g, _sip_model(g, from_slot=2), 0, 2, hardened=[3, 4])
    assert res.kind == "route"
    assert res.hazard_arrival_s == pytest.approx(1800.0)
    assert res.footprint_exit_s == pytest.approx(1200.0)
    assert res.route.nodes == [0, 1, 2]


@pytest.mark.req("PLAN-04")
def test_sip_hardened_facility_when_hazard_arrives_first():
    from outrun.routing.planner import plan_route

    g = _sip_graph()
    res = plan_route(g, _sip_model(g, from_slot=1), 0, 2, hardened=[4, 3])
    assert res.kind == "hardened_facility" and res.facility == 3
    assert res.route.nodes == [0, 3]
    assert res.hazard_arrival_s == pytest.approx(900.0) and res.footprint_exit_s == pytest.approx(1200.0)


@pytest.mark.req("PLAN-04")
@pytest.mark.parametrize(
    "from_slot,hardened",
    [(1, []), (1, [4]), (0, [3, 4])],  # none listed; only one > 5 min away; hazard already at origin
)
def test_sip_shelter_in_place(from_slot, hardened):
    from outrun.routing.planner import plan_route

    g = _sip_graph()
    res = plan_route(g, _sip_model(g, from_slot=from_slot), 0, 2, hardened=hardened)
    assert res.kind == "shelter_in_place" and res.route is None and res.facility is None


@pytest.mark.req("PLAN-04")
def test_sip_wildfire_default_threshold_and_below_threshold():
    from outrun.routing.planner import plan_route

    g = _sip_graph()
    fire = plan_route(g, _sip_model(g, from_slot=1, hazard=WILDFIRE, value=0.6), 0, 2, hardened=[3])
    assert fire.kind == "hardened_facility" and fire.facility == 3
    weak = plan_route(g, _sip_model(g, from_slot=1, value=0.45), 0, 2, hardened=[3])
    assert weak.kind == "route" and weak.hazard_arrival_s is None
    smoke = plan_route(g, _sip_model(g, from_slot=0, hazard=SMOKE, value=1.0), 0, 2, hardened=[3])
    assert smoke.kind == "route"  # smoke is not a shelter-in-place hazard
    none = plan_route(g, _sip_model(g), 0, 2, hardened=[3])
    assert none.kind == "route" and none.hazard_arrival_s is None
