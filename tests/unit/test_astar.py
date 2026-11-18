"""Spec 04-astar: time-dependent multi-label A* (ASTAR-01..ASTAR-06, ASTAR-12, ASTAR-13, ASTAR-15..ASTAR-17).

Synthetic graphs only (tests/fixtures/graphs/graphfix.py). ASTAR-04 compares against an independent brute-force
time-expanded Dijkstra written in the fixture module.
"""

from __future__ import annotations

import math
import time

import networkx as nx
import numpy as np
import pytest
from tests.fixtures.graphs.graphfix import (
    brute_force_td,
    dijkstra_costs,
    lattice,
    make_road_graph,
    random_graph,
)

NAN = float("nan")
BIG = 10_000  # label cap far above anything these tiny graphs need: exactness tests must not hit the cap


def _zero_model(g, n_slots=12):
    from outrun.routing.cost import CostModel

    return CostModel(g.free_flow_s, np.zeros((g.n_edges, 1, n_slots)), np.zeros(1), np.array([NAN]))


def _edge(g, u, v):
    for e in g.out_edges(u):
        if int(g.dst[e]) == v:
            return e
    raise KeyError((u, v))


def _check_route(g, m, r, origin, t0=0.0):
    """Structural consistency of a Route against the graph and cost model (ASTAR-16)."""
    assert r.nodes[0] == origin and len(r.nodes) == len(r.edges) + 1
    assert len(r.depart_s) == len(r.edges) and len(r.waits_s) == len(r.edges)
    assert r.start_s == pytest.approx(t0)
    t, cost = t0, 0.0
    for i, e in enumerate(r.edges):
        assert int(g.src[e]) == r.nodes[i] and int(g.dst[e]) == r.nodes[i + 1]
        w = r.waits_s[i]
        assert w >= 0
        cost += m.wait_cost(r.nodes[i], t, t + w)
        t += w
        assert r.depart_s[i] == pytest.approx(t, abs=1e-6)
        c = m.cost(e, t)
        assert math.isfinite(c)
        cost += c
        t += m.clock(e, t)
    assert r.arrive_s == pytest.approx(t, abs=1e-6)
    assert r.cost == pytest.approx(cost, rel=1e-9, abs=1e-6)


def _nx_cost(g, w, o, d):
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n_nodes))
    for e in range(g.n_edges):
        if math.isfinite(w[e]):
            G.add_edge(int(g.src[e]), int(g.dst[e]), weight=float(w[e]))
    try:
        return nx.dijkstra_path_length(G, o, d, weight="weight")
    except nx.NetworkXNoPath:
        return math.inf


# ---------------------------------------------------------------- ASTAR-01
@pytest.mark.req("ASTAR-01")
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_equals_networkx_dijkstra_zero_risk(seed):
    from outrun.routing.astar import astar

    g = random_graph(40, 3, seed)
    m = _zero_model(g)
    rng = np.random.default_rng(100 + seed)
    for _ in range(50):  # 4 x 50 = 200 OD pairs
        o, d = (int(x) for x in rng.choice(g.n_nodes, 2, replace=False))
        r = astar(g, m, o, d)
        assert r is not None and r.goal == d
        assert r.cost == pytest.approx(_nx_cost(g, g.free_flow_s, o, d), rel=1e-9)
        assert r.travel_s == pytest.approx(r.cost, rel=1e-9)
        assert not r.approximate


@pytest.mark.req("ASTAR-01")
@pytest.mark.parametrize("seed", [4, 5])
def test_equals_networkx_dijkstra_static_risk(seed):
    from outrun.routing.astar import astar
    from outrun.routing.cost import CostModel

    g = random_graph(40, 3, seed)
    rng = np.random.default_rng(seed)
    r_static = rng.uniform(0.0, 0.4, (g.n_edges, 2))
    risk = np.repeat(r_static[:, :, None], 12, axis=2)
    m = CostModel(
        g.free_flow_s, risk, np.array([2.0, 5.0]), np.array([NAN, NAN]), speed_slope=np.array([0.5, 0.0])
    )
    sf = np.clip(1 - 0.5 * r_static[:, 0], 1 / 3, 1)
    w = np.minimum(g.free_flow_s / sf, 3 * g.free_flow_s) * (1 + 2 * r_static[:, 0] + 5 * r_static[:, 1])
    for _ in range(25):
        o, d = (int(x) for x in rng.choice(g.n_nodes, 2, replace=False))
        r = astar(g, m, o, d)
        assert r is not None
        assert r.cost == pytest.approx(_nx_cost(g, w, o, d), rel=1e-9)
        _check_route(g, m, r, o)


# ---------------------------------------------------------------- ASTAR-02
@pytest.mark.req("ASTAR-02")
def test_fewer_expansions_than_dijkstra():
    from outrun.routing.astar import astar, dijkstra

    g = lattice(21, 21)
    m = _zero_model(g)
    o, d = 10 * 21 + 0, 10 * 21 + 20
    a, dj = astar(g, m, o, d), dijkstra(g, m, o, d)
    assert a.cost == pytest.approx(dj.cost, rel=1e-9)
    assert a.expansions <= 0.5 * dj.expansions
    nh = astar(g, m, o, d, use_heuristic=False)
    assert nh.cost == pytest.approx(dj.cost) and nh.expansions == dj.expansions


@pytest.mark.req("ASTAR-02")
def test_fewer_expansions_random_pairs():
    from outrun.routing.astar import astar, dijkstra

    g = random_graph(60, 3, 11)
    m = _zero_model(g)
    rng = np.random.default_rng(3)
    tot_a = tot_d = 0
    for _ in range(30):
        o, d = (int(x) for x in rng.choice(g.n_nodes, 2, replace=False))
        a, dj = astar(g, m, o, d), dijkstra(g, m, o, d)
        assert a.cost == pytest.approx(dj.cost, rel=1e-9)
        tot_a += a.expansions
        tot_d += dj.expansions
    assert tot_a < tot_d


# ---------------------------------------------------------------- ASTAR-03
@pytest.mark.req("ASTAR-03")
def test_never_uses_blocked_edges_static():
    from outrun.routing.astar import astar
    from outrun.routing.cost import CostModel

    g = lattice(5, 6)
    cols = 6
    wall = {r * cols + 3 for r in range(4)}  # column 3 closed except the bottom row (r = 4)
    blocked = np.array([int(g.src[e]) in wall or int(g.dst[e]) in wall for e in range(g.n_edges)])
    risk = np.zeros((g.n_edges, 1, 12))
    risk[blocked, 0, :] = 1.0
    m = CostModel(g.free_flow_s, risk, np.zeros(1), np.array([0.5]))
    o, d = 0 * cols + 0, 0 * cols + 5
    r = astar(g, m, o, d)
    assert r is not None
    assert not any(blocked[e] for e in r.edges)
    w = np.where(blocked, np.inf, g.free_flow_s)
    assert r.cost == pytest.approx(dijkstra_costs(g.n_nodes, g.src, g.dst, w, o)[d], rel=1e-9)


@pytest.mark.req("ASTAR-03")
@pytest.mark.parametrize("seed", [0, 1])
def test_never_enters_edge_while_blocked_time_varying(seed):
    from outrun.routing.astar import astar
    from outrun.routing.cost import CostModel

    g = random_graph(30, 3, seed, integer_minutes=True)
    rng = np.random.default_rng(seed)
    risk = rng.uniform(0, 1, (g.n_edges, 1, 12))
    m = CostModel(g.free_flow_s, risk, np.array([1.0]), np.array([0.7]), slot_s=120.0)
    for _ in range(10):
        o, d = (int(x) for x in rng.choice(g.n_nodes, 2, replace=False))
        r = astar(g, m, o, d)
        if r is None:
            continue
        for e, t in zip(r.edges, r.depart_s, strict=True):
            assert not m.blocked(e, t)
            assert risk[e, 0, m.slot(t)] < 0.7


# ---------------------------------------------------------------- ASTAR-04
def _td_model(g, seed, slot_s=120.0):
    """Time-varying risk; short slots keep the wait grid (60 s) small so the tests stay fast."""
    from outrun.routing.cost import CostModel

    rng = np.random.default_rng(seed)
    decay = np.clip(1 - np.arange(12) / 6.0, 0, 1)  # hazards fade over the window: waiting can pay
    risk = rng.uniform(0, 1, (g.n_edges, 2, 12)) * decay
    node_risk = rng.uniform(0, 0.1, (g.n_nodes, 2, 12))
    return CostModel(
        g.free_flow_s, risk, np.array([3.0, 10.0]), np.array([NAN, 0.8]), node_risk=node_risk, slot_s=slot_s
    )


@pytest.mark.req("ASTAR-04")
@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_equals_brute_force_time_expanded(seed):
    from outrun.routing.astar import astar

    g = random_graph(30, 3, seed, integer_minutes=True)
    m = _td_model(g, seed)
    rng = np.random.default_rng(50 + seed)
    for _ in range(8):
        o, d = (int(x) for x in rng.choice(g.n_nodes, 2, replace=False))
        ref = brute_force_td(g, m, o, d, step_s=60.0)
        r = astar(g, m, o, d, max_labels=BIG)
        if math.isinf(ref):
            assert r is None
            continue
        assert r is not None and not r.approximate
        assert r.cost == pytest.approx(ref, rel=1e-9, abs=1e-6)
        _check_route(g, m, r, o)


def _line_graph():
    """0 -> 1 -> 2 with 60 s edges."""
    return make_road_graph([-97.5, -97.49, -97.48], [35.5] * 3, [0, 1], [1, 2], [600.0, 600.0], [10.0, 10.0])[
        0
    ]


@pytest.mark.req("ASTAR-04")
def test_waits_for_temporary_block_to_clear():
    from outrun.routing.astar import astar
    from outrun.routing.cost import CostModel

    g = _line_graph()
    risk = np.zeros((2, 1, 12))
    risk[_edge(g, 1, 2), 0, :2] = 1.0  # closed for the first two slots (0 - 1800 s)
    m = CostModel(g.free_flow_s, risk, np.zeros(1), np.array([0.5]))
    r = astar(g, m, 0, 2)
    assert r is not None
    assert r.cost == pytest.approx(1860.0) and r.arrive_s == pytest.approx(1860.0)
    assert r.depart_s[-1] >= 1800.0
    assert sum(r.waits_s) == pytest.approx(1740.0)
    _check_route(g, m, r, 0)
    assert astar(g, m, 0, 2, allow_wait=False) is None
    late = astar(g, m, 0, 2, t0_s=1800.0)  # starting later: no wait needed
    assert late.cost == pytest.approx(120.0) and late.arrive_s == pytest.approx(1920.0)


def _pareto_graph():
    """Fast arrival at node 1 is useless (must wait in a risky cell); the slow safe path via 3 wins."""
    from outrun.routing.cost import CostModel

    lon = [-97.50, -97.49, -97.48, -97.50]
    lat = [35.50, 35.50, 35.50, 35.52]
    src, dst = [0, 0, 3, 1], [1, 3, 1, 2]
    length = [600.0, 4500.0, 4500.0, 600.0]  # 60, 450, 450, 60 s at 10 m/s
    g, _ = make_road_graph(lon, lat, src, dst, length, [10.0] * 4)
    risk = np.zeros((4, 2, 12))
    risk[_edge(g, 1, 2), 1, 0] = 1.0  # 1 -> 2 closed during slot 0 (0 - 900 s)
    node_risk = np.zeros((4, 2, 12))
    node_risk[[0, 1], 0, :] = 1.0  # origin and node 1 sit in a hazardous cell: waiting there costs 11x
    m = CostModel(g.free_flow_s, risk, np.array([10.0, 0.0]), np.array([NAN, 0.5]), node_risk=node_risk)
    return g, m


@pytest.mark.req("ASTAR-04")
@pytest.mark.req("ASTAR-13")
def test_multi_label_keeps_later_cheaper_label():
    from outrun.routing.astar import astar

    g, m = _pareto_graph()
    r = astar(g, m, 0, 2, max_labels=BIG, wait_step_s=30.0)
    assert r is not None
    assert r.nodes == [0, 3, 1, 2]
    assert r.cost == pytest.approx(960.0) and r.arrive_s == pytest.approx(960.0)
    assert not r.approximate
    _check_route(g, m, r, 0)


# ---------------------------------------------------------------- ASTAR-05
@pytest.mark.req("ASTAR-05")
def test_returns_none_when_goal_cut_off():
    from outrun.routing.astar import astar
    from outrun.routing.cost import CostModel

    g = lattice(4, 4)
    goal = 15
    risk = np.zeros((g.n_edges, 1, 12))
    risk[g.dst == goal, 0, :] = 1.0
    m = CostModel(g.free_flow_s, risk, np.zeros(1), np.array([0.5]))
    assert astar(g, m, 0, goal) is None
    assert astar(g, m, 0, goal, use_heuristic=False) is None


@pytest.mark.req("ASTAR-05")
def test_returns_none_when_disconnected():
    from outrun.routing.astar import astar

    g, _ = make_road_graph(
        [-97.5, -97.49, -97.4, -97.39], [35.5] * 4, [0, 1, 2, 3], [1, 0, 3, 2], [1000.0] * 4, [10.0] * 4
    )
    assert astar(g, _zero_model(g), 0, 3) is None


# ---------------------------------------------------------------- ASTAR-06
@pytest.mark.req("ASTAR-06")
@pytest.mark.slow
def test_metro_sized_query_under_one_second():
    from outrun.routing.astar import astar

    g = lattice(300, 300, spacing_m=100.0, speed_mps=15.0)  # 90,000 nodes, 358,800 edges, 30 x 30 km
    assert g.n_nodes == 90_000
    m = _zero_model(g)
    o, d = 20 * 300 + 20, 280 * 300 + 250
    astar(g, m, o, d)  # warm-up (lazy tables, caches)
    best = math.inf
    for _ in range(3):
        t = time.perf_counter()
        r = astar(g, m, o, d)
        best = min(best, time.perf_counter() - t)
    assert r is not None
    assert r.cost == pytest.approx((260 + 230) * 100.0 / 15.0, rel=1e-9)
    assert best < 1.0


# ---------------------------------------------------------------- ASTAR-12
@pytest.mark.req("ASTAR-12")
@pytest.mark.parametrize("seed", [0, 1])
def test_multi_goal_picks_cheapest_goal(seed):
    from outrun.routing.astar import astar

    g = random_graph(40, 3, seed, integer_minutes=True)
    m = _td_model(g, seed)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        o, *goals = (int(x) for x in rng.choice(g.n_nodes, 4, replace=False))
        singles = {k: astar(g, m, o, k, max_labels=BIG) for k in goals}
        costs = {k: (r.cost if r else math.inf) for k, r in singles.items()}
        r = astar(g, m, o, goals, max_labels=BIG)
        if all(math.isinf(c) for c in costs.values()):
            assert r is None
            continue
        assert r.goal in goals
        assert r.cost == pytest.approx(min(costs.values()), rel=1e-9)
        assert costs[r.goal] == pytest.approx(r.cost, rel=1e-9)


@pytest.mark.req("ASTAR-12")
def test_origin_in_goals_is_empty_route():
    from outrun.routing.astar import astar

    g = lattice(3, 3)
    r = astar(g, _zero_model(g), 4, [8, 4], t0_s=300.0)
    assert r.nodes == [4] and r.edges == [] and r.cost == 0.0
    assert r.start_s == 300.0 and r.arrive_s == 300.0


# ---------------------------------------------------------------- ASTAR-13
@pytest.mark.req("ASTAR-13")
def test_dominance_rule():
    from outrun.routing.astar import dominates

    def wait(a, b):
        return 2.0 * (b - a)

    assert dominates(0.0, 10.0, 5.0, 20.0, wait)  # 10 + 10 <= 20
    assert not dominates(0.0, 10.0, 5.0, 19.0, wait)  # 10 + 10 > 19
    assert not dominates(6.0, 1.0, 5.0, 100.0, wait)  # later label never dominates an earlier one
    assert dominates(5.0, 7.0, 5.0, 7.0, wait)  # equal labels


@pytest.mark.req("ASTAR-13")
def test_label_cap_sets_approximate():
    from outrun.routing.astar import astar

    g, m = _pareto_graph()
    r = astar(g, m, 0, 2, max_labels=1, wait_step_s=30.0)  # node 1 needs two non-dominated labels
    assert r is None or (r.approximate and r.cost >= 960.0 - 1e-6)


# ---------------------------------------------------------------- ASTAR-15
@pytest.mark.req("ASTAR-15")
def test_time_expanded_checker_hand_cases():
    from outrun.routing.astar import time_expanded_dijkstra
    from outrun.routing.cost import CostModel

    g = _line_graph()
    risk = np.zeros((2, 1, 12))
    risk[_edge(g, 1, 2), 0, :2] = 1.0
    m = CostModel(g.free_flow_s, risk, np.zeros(1), np.array([0.5]))
    cost, arrive = time_expanded_dijkstra(g, m, 0, 2)
    assert cost == pytest.approx(1860.0) and arrive == pytest.approx(1860.0)
    gp, mp = _pareto_graph()
    cost, arrive = time_expanded_dijkstra(gp, mp, 0, 2, step_s=30.0)  # 450 s edges: exact on a 30 s grid
    assert cost == pytest.approx(960.0) and arrive == pytest.approx(960.0)
    assert time_expanded_dijkstra(gp, mp, 2, 0, step_s=30.0) is None  # node 2 has no out-edges


@pytest.mark.req("ASTAR-15")
@pytest.mark.slow
def test_time_expanded_checker_matches_independent_reference():
    from outrun.routing.astar import time_expanded_dijkstra

    g = random_graph(20, 3, 7, integer_minutes=True)
    m = _td_model(g, 7)
    rng = np.random.default_rng(7)
    for _ in range(5):
        o, d = (int(x) for x in rng.choice(g.n_nodes, 2, replace=False))
        ref = brute_force_td(g, m, o, d)
        got = time_expanded_dijkstra(g, m, o, d)
        if math.isinf(ref):
            assert got is None
        else:
            assert got[0] == pytest.approx(ref, rel=1e-9)


# ---------------------------------------------------------------- ASTAR-16
@pytest.mark.req("ASTAR-16")
@pytest.mark.parametrize("seed", [3, 4])
def test_route_fields_are_consistent(seed):
    from outrun.routing.astar import astar

    g = random_graph(30, 3, seed, integer_minutes=True)
    m = _td_model(g, seed)
    rng = np.random.default_rng(seed)
    n_ok = 0
    for _ in range(10):
        o, d = (int(x) for x in rng.choice(g.n_nodes, 2, replace=False))
        r = astar(g, m, o, d, t0_s=120.0, max_labels=BIG)
        if r is None:
            continue
        n_ok += 1
        assert len(r.waits_s) == len(r.edges)
        assert r.expansions > 0
        assert r.travel_s == pytest.approx(r.arrive_s - 120.0)
        _check_route(g, m, r, o, t0=120.0)
    assert n_ok > 0


# ---------------------------------------------------------------- ASTAR-17
@pytest.mark.req("ASTAR-17")
def test_astar_input_errors():
    from outrun.routing.astar import astar
    from outrun.routing.cost import CostModel

    g = lattice(3, 3)
    m = _zero_model(g)
    for kwargs in (
        dict(origin=-1, goals=2),
        dict(origin=0, goals=9),
        dict(origin=0, goals=[]),
        dict(origin=0, goals=2, t0_s=-1.0),
        dict(origin=0, goals=2, max_labels=0),
        dict(origin=0, goals=2, wait_step_s=0.0),
    ):
        with pytest.raises(ValueError):
            astar(g, m, **kwargs)
    bad = CostModel(np.ones(3), np.zeros((3, 1, 12)), np.zeros(1), np.array([NAN]))
    with pytest.raises(ValueError):
        astar(g, bad, 0, 2)
