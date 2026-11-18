"""Spec 04-astar: edge costs and risk sampling (ASTAR-07..ASTAR-11, ASTAR-14). Synthetic arrays only."""

from __future__ import annotations

import math

import numpy as np
import pytest
from tests.fixtures.graphs.graphfix import haversine, make_road_graph, random_graph

NAN = float("nan")


def _model(**kw):
    from outrun.routing.cost import CostModel

    ff = np.array([100.0, 60.0])
    risk = np.zeros((2, 2, 3))
    risk[0, 0] = [0.0, 0.5, 1.0]
    risk[0, 1] = [0.0, 0.0, 0.2]
    args = dict(
        weights=np.array([2.0, 10.0]), block_at=np.array([NAN, 0.15]), speed_slope=np.array([0.5, 0.0])
    )
    args.update(kw)
    return CostModel(ff, risk, **args)


# ---------------------------------------------------------------- ASTAR-07 clock time
@pytest.mark.req("ASTAR-07")
def test_clock_time_speed_factor():
    m = _model()
    assert m.n_edges == 2 and m.n_slots == 3
    assert m.clock(0, 0.0) == pytest.approx(100.0)
    assert m.clock(0, 900.0) == pytest.approx(100.0 / 0.75)
    assert m.clock(0, 1800.0) == pytest.approx(200.0)  # clock still defined on a blocked edge
    assert m.clock(1, 1800.0) == pytest.approx(60.0)


@pytest.mark.req("ASTAR-07")
def test_clock_cap_and_bpr():
    from outrun.routing.cost import CostModel

    ff = np.array([100.0, 100.0])
    risk = np.ones((2, 2, 1))
    risk[1] = 0.0
    m = CostModel(
        ff,
        risk,
        np.zeros(2),
        np.array([NAN, NAN]),
        speed_slope=np.array([0.9, 0.9]),
        bpr=np.array([1.0, 2.0]),
    )
    assert m.clock(0, 0.0) == pytest.approx(300.0)  # (1/3)*(1/3) slow-down capped at 3x free flow
    assert m.clock(1, 0.0) == pytest.approx(200.0)  # BPR doubles the clock
    np.testing.assert_allclose(m.clock_table(), [[300.0], [200.0]])
    assert np.all(m.clock_table() >= ff[:, None] - 1e-9)


@pytest.mark.req("ASTAR-07")
def test_clock_table_matches_clock():
    m = _model()
    tab = m.clock_table()
    assert tab.shape == (2, 3)
    for e in range(2):
        for s in range(3):
            assert tab[e, s] == pytest.approx(m.clock(e, s * 900.0))


# ---------------------------------------------------------------- ASTAR-08 generalized cost and blocking
@pytest.mark.req("ASTAR-08")
def test_generalized_cost_and_blocking():
    m = _model()
    assert m.cost(0, 0.0) == pytest.approx(100.0)
    assert m.cost(0, 900.0) == pytest.approx((100.0 / 0.75) * (1 + 2 * 0.5))
    assert not m.blocked(0, 900.0)
    assert m.blocked(0, 1800.0) and math.isinf(m.cost(0, 1800.0))
    assert m.cost(1, 1800.0) == pytest.approx(60.0)
    tab = m.cost_table()
    assert tab.shape == (2, 3) and math.isinf(tab[0, 2])
    finite = np.isfinite(tab)
    assert np.all(tab[finite] >= m.clock_table()[finite] - 1e-9)


@pytest.mark.req("ASTAR-08")
def test_block_threshold_is_inclusive_and_lambda_scales_weights():
    m = _model(block_at=np.array([NAN, 0.2]))
    assert m.blocked(0, 1800.0)  # r == block_at blocks
    m2 = _model(lam=0.5)
    assert m2.cost(0, 900.0) == pytest.approx((100.0 / 0.75) * (1 + 0.5 * 2 * 0.5))
    m0 = _model(lam=0.0)
    assert m0.cost(0, 900.0) == pytest.approx(m0.clock(0, 900.0))
    assert m0.blocked(0, 1800.0)  # lambda never removes blocking


@pytest.mark.req("ASTAR-08")
def test_from_policies():
    from outrun.core.policy import HazardPolicy, Profile
    from outrun.core.risk import Hazard
    from outrun.routing.cost import CostModel

    tornado = HazardPolicy(
        block_at={Profile.CIVILIAN: 0.5, Profile.SENSITIVE: 0.5, Profile.EV: 0.5},
        speed_factor=lambda p: 1.0,
        exposure_weight={Profile.CIVILIAN: 20.0, Profile.SENSITIVE: 20.0, Profile.EV: 5.0},
        beyond_horizon="drop",
        decay_tau_min=None,
    )
    ice = HazardPolicy(
        block_at={Profile.CIVILIAN: None, Profile.SENSITIVE: None, Profile.EV: None},
        speed_factor=lambda p: max(1 / 3, 1 - 0.5 * p),
        exposure_weight={Profile.CIVILIAN: 3.0, Profile.SENSITIVE: 3.0, Profile.EV: 1.0},
        beyond_horizon="hold",
        decay_tau_min=None,
    )
    order = list(Hazard)
    risk = np.zeros((1, len(order), 2))
    risk[0, order.index(Hazard.TORNADO)] = [0.2, 0.6]
    risk[0, order.index(Hazard.ICE)] = [0.4, 0.4]
    risk[0, order.index(Hazard.STORM)] = [0.9, 0.9]  # no policy given -> ignored
    pol = {Hazard.TORNADO: tornado, Hazard.ICE: ice}
    civ = CostModel.from_policies(np.array([100.0]), risk, pol, Profile.CIVILIAN)
    assert civ.clock(0, 0.0) == pytest.approx(125.0)
    assert civ.cost(0, 0.0) == pytest.approx(125.0 * (1 + 20 * 0.2 + 3 * 0.4))
    assert civ.blocked(0, 900.0)
    ev = CostModel.from_policies(np.array([100.0]), risk, pol, Profile.EV)
    assert ev.cost(0, 0.0) == pytest.approx(125.0 * (1 + 5 * 0.2 + 1 * 0.4))


@pytest.mark.req("ASTAR-08")
@pytest.mark.parametrize(
    "kw",
    [
        {"weights": np.array([-1.0, 1.0])},
        {"speed_slope": np.array([-0.1, 0.0])},
        {"bpr": np.array([0.9, 1.0])},
        {"lam": -1.0},
        {"weights": np.array([1.0])},  # wrong H
        {"node_risk": np.zeros((2, 2, 4))},  # wrong S
        {"slot_s": 0.0},
        {"clock_cap": 0.5},
    ],
)
def test_cost_model_rejects_bad_parameters(kw):
    with pytest.raises(ValueError):
        _model(**kw)


@pytest.mark.req("ASTAR-08")
def test_cost_model_rejects_bad_arrays():
    from outrun.routing.cost import CostModel

    w, b = np.zeros(1), np.array([NAN])
    with pytest.raises(ValueError):
        CostModel(np.array([1.0]), np.full((1, 1, 2), 1.5), w, b)  # risk > 1
    with pytest.raises(ValueError):
        CostModel(np.array([1.0]), np.full((1, 1, 2), NAN), w, b)  # NaN risk
    with pytest.raises(ValueError):
        CostModel(np.array([1.0, 2.0]), np.zeros((1, 1, 2)), w, b)  # E mismatch
    with pytest.raises(ValueError):
        CostModel(np.array([0.0]), np.zeros((1, 1, 2)), w, b)  # free flow <= 0
    with pytest.raises(ValueError):
        CostModel(np.array([1.0]), np.zeros((1, 2)), w, b)  # not 3-D


# ---------------------------------------------------------------- ASTAR-09 slots
@pytest.mark.req("ASTAR-09")
def test_slot_indexing_piecewise_constant_and_held():
    m = _model()
    assert [m.slot(t) for t in (0.0, 899.9, 900.0, 1799.0, 1800.0, 1e6)] == [0, 0, 1, 1, 2, 2]
    assert m.cost(0, 950.0) == pytest.approx(m.cost(0, 1700.0))  # constant within a slot
    with pytest.raises(ValueError):
        m.slot(-1.0)
    m60 = _model(slot_s=60.0)
    assert m60.slot(61.0) == 1


@pytest.mark.req("ASTAR-09")
def test_with_risk_keeps_parameters():
    m = _model()
    z = m.with_risk(np.zeros((2, 2, 3)))
    assert z.cost(0, 900.0) == pytest.approx(100.0)
    assert not z.blocked(0, 1800.0)
    assert m.blocked(0, 1800.0)  # original untouched
    z2 = _model(lam=0.5, bpr=np.array([2.0, 1.0])).with_risk(np.zeros((2, 2, 3)))
    assert z2.clock(0, 0.0) == pytest.approx(200.0)  # bpr kept


# ---------------------------------------------------------------- ASTAR-10 wait cost
@pytest.mark.req("ASTAR-10")
def test_wait_cost_integrates_node_risk():
    node_risk = np.zeros((2, 2, 3))
    node_risk[0, 0] = [0.0, 0.5, 0.0]
    node_risk[0, 1] = [0.1, 0.0, 0.0]
    m = _model(node_risk=node_risk)
    # [600,900): 300 * (1 + 10*0.1) ; [900,1800): 900 * (1 + 2*0.5) ; [1800,2000): 200 * 1
    assert m.wait_cost(0, 600.0, 2000.0) == pytest.approx(300 * 2 + 900 * 2 + 200 * 1)
    assert m.wait_cost(1, 600.0, 2000.0) == pytest.approx(1400.0)
    assert m.wait_cost(0, 5.0, 5.0) == 0.0
    with pytest.raises(ValueError):
        m.wait_cost(0, 10.0, 5.0)
    assert _model(node_risk=node_risk, lam=0.0).wait_cost(0, 600.0, 2000.0) == pytest.approx(1400.0)


@pytest.mark.req("ASTAR-10")
def test_wait_cost_without_node_risk_is_duration():
    m = _model()
    assert m.wait_cost(0, 0.0, 5000.0) == pytest.approx(5000.0)


# ---------------------------------------------------------------- ASTAR-11 heuristic
@pytest.mark.req("ASTAR-11")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_heuristic_admissible_and_consistent(seed):
    from outrun.routing.cost import crowfly_speed_mps, make_heuristic

    g = random_graph(40, 3, seed)
    span = haversine(g.lon[g.src], g.lat[g.src], g.lon[g.dst], g.lat[g.dst])
    vstar = crowfly_speed_mps(g)
    assert vstar == pytest.approx(float(np.max(span / g.free_flow_s)))
    assert vstar <= float(np.max(g.speed_mps)) + 1e-9
    goal = 7
    h = make_heuristic(g, goal)
    assert h.shape == (g.n_nodes,) and h[goal] == 0.0 and np.all(h >= 0)
    np.testing.assert_allclose(h, haversine(g.lon, g.lat, g.lon[goal], g.lat[goal]) / vstar, rtol=1e-9)
    assert np.all(h[g.src] <= g.free_flow_s + h[g.dst] + 1e-6)  # consistent => admissible


@pytest.mark.req("ASTAR-11")
def test_multi_goal_heuristic_is_min():
    from outrun.routing.cost import make_heuristic

    g = random_graph(30, 3, 5)
    hs = np.stack([make_heuristic(g, k) for k in (2, 9, 17)])
    np.testing.assert_allclose(make_heuristic(g, [2, 9, 17]), hs.min(axis=0))


@pytest.mark.req("ASTAR-11")
def test_heuristic_zero_span_graph():
    from outrun.routing.cost import crowfly_speed_mps, make_heuristic

    g, _ = make_road_graph([-97.5] * 3, [35.5] * 3, [0, 1], [1, 2], [100.0, 100.0], [10.0, 10.0])
    assert crowfly_speed_mps(g) == 0.0
    assert np.all(make_heuristic(g, 2) == 0.0)


# ---------------------------------------------------------------- ASTAR-14 edge risk sampler
def _sampler_graph():
    # projected metres; grid: x_left=0, y_top=4000, 1 km cells, 4 x 4
    x = np.array([500.0, 3500.0, 500.0, -2500.0, 950.0, 1050.0])
    y = np.array([3500.0, 3500.0, 1500.0, 3500.0, 3500.0, 3500.0])
    src = [0, 0, 3, 4]
    dst = [1, 2, 0, 5]
    length = [3000.0, 2000.0, 3000.0, 100.0]
    g, order = make_road_graph(np.zeros(6), np.zeros(6), src, dst, length, [10.0] * 4, x_m=x, y_m=y)
    assert order.tolist() == [0, 1, 2, 3]  # src already sorted: edge ids are stable
    return g


@pytest.mark.req("ASTAR-14")
def test_edge_risk_sampler_max_over_crossed_cells():
    from outrun.routing.cost import EdgeRiskSampler

    g = _sampler_graph()
    s = EdgeRiskSampler(g, x_left=0.0, y_top=4000.0, cell_m=1000.0, shape=(4, 4), spacing_m=100.0)
    prob = np.zeros((1, 2, 4, 4), dtype=np.float32)
    prob[0, 0, 0, 2] = 0.7  # on edge 0 (row 0)
    prob[0, 0, 1, 1] = 0.9  # touches no edge
    prob[0, 0, 0, 0] = 0.2  # node 0's cell
    prob[0, 1, 0, 1] = 0.3  # on edge 0 and the 100 m edge 3 (its end point)
    prob[0, 1, 2, 0] = 0.6  # on edge 1 (vertical, end cell)
    r = s.sample(prob)
    assert r.shape == (4, 1, 2)
    np.testing.assert_allclose(r[0, 0], [0.7, 0.3], atol=1e-6)
    np.testing.assert_allclose(r[1, 0], [0.2, 0.6], atol=1e-6)
    np.testing.assert_allclose(r[2, 0], [0.2, 0.0], atol=1e-6)  # outside-grid part contributes 0
    np.testing.assert_allclose(r[3, 0], [0.2, 0.3], atol=1e-6)  # start cell + end point sampled
    nodes = s.sample_nodes(prob)
    assert nodes.shape == (6, 1, 2)
    np.testing.assert_allclose(nodes[:, 0, 0], [0.2, 0.0, 0.0, 0.0, 0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(nodes[:, 0, 1], [0.0, 0.0, 0.6, 0.0, 0.0, 0.3], atol=1e-6)


@pytest.mark.req("ASTAR-14")
def test_edge_risk_sampler_overrides_replace_and_shape_check():
    from outrun.routing.cost import EdgeRiskSampler

    g = _sampler_graph()
    s = EdgeRiskSampler(g, 0.0, 4000.0, 1000.0, (4, 4))
    prob = np.zeros((2, 2, 4, 4), dtype=np.float32)
    prob[0, 1, 2, 0] = 0.6
    r = s.sample(prob, edge_overrides={0: {1: np.array([0.05, 0.0])}, 1: {3: np.array([0.9, 0.8])}})
    np.testing.assert_allclose(r[1, 0], [0.05, 0.0], atol=1e-6)  # replaced, not max-ed
    np.testing.assert_allclose(r[3, 1], [0.9, 0.8], atol=1e-6)
    with pytest.raises(ValueError):
        s.sample(np.zeros((2, 2, 5, 4)))


# ---------------------------------------------------------------- ASTAR-08 where and when blocking applies
def _pol(block, slope, weight):
    from outrun.core.policy import HazardPolicy, Profile

    return HazardPolicy(
        block_at={Profile.CIVILIAN: block[0], Profile.SENSITIVE: block[0], Profile.EV: block[1]},
        speed_factor=lambda p: max(1 / 3, 1 - slope * p),
        exposure_weight={Profile.CIVILIAN: weight, Profile.SENSITIVE: weight, Profile.EV: weight / 2},
        beyond_horizon="hold",
        decay_tau_min=None,
    )


@pytest.mark.req("ASTAR-08")
def test_block_mask_and_block_until_in_constructor():
    from outrun.routing.cost import CostModel

    risk = np.full((2, 1, 1), 0.9)
    m = CostModel(
        np.array([100.0, 100.0]), risk, np.ones(1), np.array([0.5]), block_mask={0: np.array([False, True])}
    )
    assert not m.blocked(0, 0.0) and m.blocked(1, 0.0)
    assert m.cost(0, 0.0) == pytest.approx(100.0 * 1.9) and math.isinf(m.cost(1, 0.0))
    assert m.blocked_table().tolist() == [[False], [True]]
    with pytest.raises(ValueError):
        CostModel(
            np.array([100.0, 100.0]), risk, np.ones(1), np.array([0.5]), block_mask={1: np.ones(2, bool)}
        )
    with pytest.raises(ValueError):
        CostModel(
            np.array([100.0, 100.0]), risk, np.ones(1), np.array([0.5]), block_mask={0: np.ones(3, bool)}
        )
    t = CostModel(
        np.array([100.0]), np.full((1, 1, 2), 0.9), np.ones(1), np.array([0.5]), block_until_s={0: 900.0}
    )
    assert t.blocked(0, 0.0) and not t.blocked(0, 900.0)  # slot 1 starts at 900, not < 900
    assert t.blocked_table().tolist() == [[True, False]]
    assert t.cost(0, 900.0) == pytest.approx(190.0)  # exposure still applies after blocking ends


@pytest.mark.req("ASTAR-08")
def test_from_policies_flood_blocks_only_at_crossings():
    from outrun.core.policy import Profile
    from outrun.core.risk import Hazard
    from outrun.routing.cost import CostModel

    fl = list(Hazard).index(Hazard.FLOOD)
    risk = np.zeros((2, len(Hazard), 1))
    risk[0, fl] = 0.9  # ordinary road in a 1 km flood cell
    risk[1, fl] = 0.4  # low-water crossing at the civilian threshold
    pol = {Hazard.FLOOD: _pol((0.4, 0.7), 0.4, 10.0)}
    ff = np.array([100.0, 100.0])
    civ = CostModel.from_policies(ff, risk, pol, Profile.CIVILIAN, crossing=np.array([False, True]))
    assert not civ.blocked(0, 0.0) and civ.blocked(1, 0.0)
    assert civ.cost(0, 0.0) == pytest.approx(
        100.0 / (1 - 0.4 * 0.9) * (1 + 10 * 0.9)
    )  # slowed, penalised, open
    none = CostModel.from_policies(
        ff, risk, pol, Profile.CIVILIAN
    )  # no crossings known -> flood never blocks
    assert not none.blocked(0, 0.0) and not none.blocked(1, 0.0)
    ev = CostModel.from_policies(ff, risk, pol, Profile.EV, crossing=np.array([False, True]))
    assert not ev.blocked(1, 0.0)  # 0.4 < EV threshold 0.7


@pytest.mark.req("ASTAR-08")
def test_from_policies_tornado_blocks_only_within_20_minutes():
    from outrun.core.policy import Profile
    from outrun.core.risk import Hazard
    from outrun.routing.cost import CostModel

    tor = list(Hazard).index(Hazard.TORNADO)
    risk = np.zeros((1, len(Hazard), 3))
    risk[0, tor] = 0.6
    pol = {Hazard.TORNADO: _pol((0.5, 0.5), 0.0, 20.0)}
    m = CostModel.from_policies(
        np.array([100.0]), risk, pol, Profile.CIVILIAN
    )  # config: block_within_min = 20
    assert m.blocked(0, 0.0) and m.blocked(0, 900.0) and not m.blocked(0, 1800.0)
    assert m.cost(0, 1800.0) == pytest.approx(100.0 * (1 + 20 * 0.6))
    nolimit = CostModel.from_policies(np.array([100.0]), risk, pol, Profile.CIVILIAN, block_until_s={})
    assert nolimit.blocked(0, 1800.0)
    short = CostModel.from_policies(
        np.array([100.0]), risk, pol, Profile.CIVILIAN, block_until_s={Hazard.TORNADO: 900.0}
    )
    assert short.blocked(0, 0.0) and not short.blocked(0, 900.0)


# ---------------------------------------------------------------- ASTAR-14 / ASTAR-18 stack handoff
def _stack(t0, prob6, overrides):
    import xarray as xr

    from outrun.core.grid import GridWindow
    from outrun.core.risk import Hazard, RiskStack

    win = GridWindow(x0=0.0, y0=4000.0, nx=4, ny=4)
    data = xr.DataArray(
        prob6.astype(np.float16),
        dims=("hazard", "slot", "y", "x"),
        coords={
            "hazard": [h.value for h in Hazard],
            "slot": np.arange(prob6.shape[1]),
            "y": 3500.0 - 1000.0 * np.arange(4),
            "x": 500.0 + 1000.0 * np.arange(4),
        },
    )
    return RiskStack(data, t0, win, edge_overrides=overrides), win


@pytest.mark.req("ASTAR-14")
@pytest.mark.req("ASTAR-18")
def test_sample_stack_maps_hazard_keyed_overrides(t0):
    from outrun.core.risk import Hazard
    from outrun.routing.cost import EdgeRiskSampler

    g = _sampler_graph()
    prob6 = np.zeros((6, 2, 4, 4), np.float32)
    prob6[2, 1, 2, 0] = 0.6  # FLOOD, slot 1, on edge 1's end cell
    prob6[5, 0, 0, 2] = 0.7  # SMOKE, slot 0, on edge 0
    ov = {
        Hazard.FLOOD: {1: np.array([0.05, 0.0], np.float32)},
        Hazard.SMOKE: {3: np.array([0.9, 0.8], np.float32)},
    }
    stack, win = _stack(t0, prob6, ov)
    s = EdgeRiskSampler.from_window(g, win)
    ref = EdgeRiskSampler(g, 0.0, 4000.0, 1000.0, (4, 4))
    np.testing.assert_array_equal(s.sample(prob6), ref.sample(prob6))
    r = s.sample_stack(stack)
    assert r.shape == (4, 6, 2)
    np.testing.assert_allclose(r[1, 2], [0.05, 0.0], atol=1e-6)  # Hazard.FLOOD override replaced the 0.6
    np.testing.assert_allclose(r[3, 5], [0.9, 0.8], atol=1e-6)  # Hazard.SMOKE override
    np.testing.assert_allclose(r[0, 5], [0.7, 0.0], atol=1e-3)  # sampled where no override
    n = s.sample_stack_nodes(stack)
    assert n.shape == (6, 6, 2)
    np.testing.assert_allclose(n[2, 2], [0.0, 0.6], atol=1e-3)
    with pytest.raises(ValueError):  # Hazard keys passed straight to sample() are rejected, not dropped
        s.sample(prob6, edge_overrides={Hazard.FLOOD: {1: np.array([0.05, 0.0])}})
    bad, _ = _stack(t0, prob6, {Hazard.FLOOD: {4: np.array([0.1, 0.1], np.float32)}})  # graph has 4 edges
    with pytest.raises(ValueError):  # overrides keyed on another graph
        s.sample_stack(bad)


@pytest.mark.req("ASTAR-18")
def test_build_cost_model_from_stack(t0):
    import dataclasses

    from outrun.core.policy import Profile
    from outrun.routing.cost import build_cost_model

    g = dataclasses.replace(_sampler_graph(), crossing=np.array([False, True, False, False]))
    prob6 = np.zeros((6, 2, 4, 4), np.float32)
    prob6[2] = 0.9  # flood everywhere in the window
    stack, _ = _stack(t0, prob6, {})
    m = build_cost_model(
        g, stack, Profile.CIVILIAN
    )  # shipped policies: flood civilian block 0.4, w 10, slope 0.4
    assert m.n_edges == 4 and m.n_slots == 2
    assert not m.blocked(0, 0.0) and m.blocked(1, 0.0)  # only the crossing is walled off
    assert m.clock(0, 0.0) == pytest.approx(300.0 / (1 - 0.4 * 0.9), rel=1e-3)
    assert m.cost(0, 0.0) == pytest.approx(300.0 / (1 - 0.4 * 0.9) * (1 + 10 * 0.9), rel=1e-3)
    assert m.node_risk.shape == (6, 6, 2)


# ---------------------------------------------------------------- ASTAR-19 DiGraph converter
@pytest.mark.req("ASTAR-19")
def test_to_td_digraph_attributes_and_parallel_edges():
    from outrun.routing.cost import CostModel, node_path_to_edges, to_td_digraph

    g, order = make_road_graph(
        [-97.50, -97.49, -97.48], [35.5, 35.5, 35.5], [0, 0, 1], [1, 1, 2], [1000.0, 500.0, 600.0], [10.0] * 3
    )
    assert order.tolist() == [0, 1, 2]  # edge 0: 0->1 100 s, edge 1: 0->1 50 s (parallel), edge 2: 1->2 60 s
    risk = np.zeros((3, 1, 2))
    risk[1, 0, 0] = 0.25
    risk[2, 0, 1] = 0.6
    m = CostModel(g.free_flow_s, risk, np.array([2.0]), np.array([0.5]))
    G = to_td_digraph(g, m)
    assert G.number_of_nodes() == 3 and G.number_of_edges() == 2
    assert G.nodes[1]["x"] == pytest.approx(g.x_m[1]) and G.nodes[1]["y"] == pytest.approx(g.y_m[1])
    a = G[0][1]
    assert a["eid"] == 1  # faster parallel edge kept
    np.testing.assert_allclose(a["time"], [50.0, 50.0])
    np.testing.assert_allclose(a["cost"], [75.0, 50.0])
    np.testing.assert_allclose(a["risk"], [0.5, 0.0])
    b = G[1][2]
    assert b["eid"] == 2 and b["time"][0] == pytest.approx(60.0) and math.isinf(b["time"][1])
    assert math.isinf(b["cost"][1]) and b["risk"][1] == pytest.approx(1.2)
    assert node_path_to_edges(G, [0, 1, 2]).tolist() == [1, 2]
    assert node_path_to_edges(G, [0]).shape == (0,)
    with pytest.raises(KeyError):
        node_path_to_edges(G, [2, 0])


@pytest.mark.req("ASTAR-19")
def test_slot_times_from_waves():
    from outrun.routing.cost import slot_times_from_waves

    w = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])  # (W=2 waves, E=3 edges)
    out = slot_times_from_waves(w, wave_s=1800.0, slot_s=900.0, n_slots=6)
    assert out.shape == (3, 6)
    np.testing.assert_allclose(out[0], [10, 10, 11, 11, 11, 11])  # last wave held
    np.testing.assert_allclose(out[2], [30, 30, 31, 31, 31, 31])


# ---------------------------------------------------------------- ASTAR-20 scoring tables
@pytest.mark.req("ASTAR-20")
def test_combined_blocked_and_weighted_tables():
    m = _model()
    np.testing.assert_allclose(m.combined_risk_table(), [[0.0, 0.5, 1.0], [0.0, 0.0, 0.0]])
    assert m.blocked_table().tolist() == [[False, False, True], [False, False, False]]
    for e in range(2):
        for s in range(3):
            assert m.blocked_table()[e, s] == m.blocked(e, s * 900.0)
    np.testing.assert_allclose(m.weighted_risk_table(), [[0.0, 1.0, 4.0], [0.0, 0.0, 0.0]])
    assert _model(lam=3.0).weighted_risk_table()[0, 2] == pytest.approx(4.0)  # lam not applied
