"""Spec 07-emergency.md: EV routing, dispatch, hospital leg, Pareto sweep, discover-and-reroute baseline.

Synthetic graphs only. Expected values come from in-test oracles (networkx Dijkstra, simple-path enumeration,
brute-force permutations) so ties never make a sound implementation fail.
"""

from __future__ import annotations

import itertools
import math

import networkx as nx
import numpy as np
import pytest

SLOT_S = 900.0


# ------------------------------------------------------------------ helpers / oracles
def random_graph(seed: int, n: int = 25, p: float = 0.15) -> nx.DiGraph:
    rng = np.random.default_rng(seed)
    g = nx.gnp_random_graph(n, p, seed=seed, directed=True)
    dg = nx.DiGraph()
    dg.add_nodes_from(g.nodes)
    for u, v in g.edges:
        dg.add_edge(u, v, time=float(rng.integers(30, 600)), risk=float(rng.choice([0.0, 0.0, 0.5, 2.0])))
    return dg


def tau(w, t: float) -> float:
    if np.ndim(w) == 0:
        return float(w)
    w = list(w)
    return float(w[min(int(t // SLOT_S), len(w) - 1)])


def simulate(g: nx.DiGraph, path, depart: float, weight: str = "time") -> float:
    t = depart
    for u, v in zip(path[:-1], path[1:], strict=True):
        t += tau(g.edges[u, v][weight], t)
    return t - depart


def brute_td(g: nx.DiGraph, src, dst, depart: float) -> float:
    if src == dst:
        return 0.0
    best = math.inf
    for path in nx.all_simple_paths(g, src, dst):
        best = min(best, simulate(g, path, depart))
    return best


def td_grid_graph(seed: int) -> nx.DiGraph:
    """3x3 grid (both directions) with FIFO per-slot weights (non-decreasing over 6 slots)."""
    rng = np.random.default_rng(seed)
    g = nx.DiGraph()
    base = nx.grid_2d_graph(3, 3)
    mapping = {node: i for i, node in enumerate(sorted(base.nodes))}
    for a, b in base.edges:
        for u, v in ((mapping[a], mapping[b]), (mapping[b], mapping[a])):
            start = float(rng.integers(200, 700))
            steps = np.cumsum(rng.integers(0, 400, size=6)).astype(float)
            g.add_edge(u, v, time=list(start + steps))
    return g


# ------------------------------------------------------------------ EMS-01
@pytest.mark.req("EMS-01")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_td_travel_times_static_equals_dijkstra(seed):
    from outrun.routing.emergency import td_travel_times_to

    g = random_graph(seed)
    target = 0
    sources = list(g.nodes)
    got = td_travel_times_to(g, target, sources, depart_s=0.0)
    assert got.shape == (len(sources),)
    for i, s in enumerate(sources):
        try:
            want = nx.dijkstra_path_length(g, s, target, weight="time")
        except nx.NetworkXNoPath:
            want = math.inf
        if math.isinf(want):
            assert math.isinf(got[i])
        else:
            assert got[i] == pytest.approx(want, abs=1e-9)
    assert got[sources.index(target)] == 0.0


@pytest.mark.req("EMS-01")
@pytest.mark.parametrize("depart", [0.0, 700.0, 2000.0])
def test_td_travel_times_time_dependent_equals_brute_force(depart):
    from outrun.routing.emergency import td_travel_times_to

    g = td_grid_graph(3)
    target = 8
    sources = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    got = td_travel_times_to(g, target, sources, depart_s=depart)
    for i, s in enumerate(sources):
        assert got[i] == pytest.approx(brute_td(g, s, target, depart), abs=1e-6)


@pytest.mark.req("EMS-01")
def test_td_travel_times_blocked_unreachable_and_errors():
    from outrun.routing.emergency import td_travel_times_to

    g = nx.DiGraph()
    g.add_edge("a", "b", time=10.0)
    g.add_edge("b", "c", time=math.inf)  # blocked
    g.add_edge("a", "c", time=50.0)
    g.add_node("d")
    got = td_travel_times_to(g, "c", ["a", "b", "d", "c"])
    assert got[0] == pytest.approx(50.0)
    assert math.isinf(got[1]) and math.isinf(got[2])
    assert got[3] == 0.0
    with pytest.raises(KeyError):
        td_travel_times_to(g, "c", ["zzz"])
    with pytest.raises(KeyError):
        td_travel_times_to(g, "zzz", ["a"])
    with pytest.raises(ValueError):
        td_travel_times_to(g, "c", ["a"], depart_s=-1.0)


@pytest.mark.req("EMS-01")
def test_inbound_eta_matches_td_travel_times():
    from outrun.routing.emergency import inbound_eta, td_travel_times_to

    g = td_grid_graph(5)
    stations = [0, 2, 6]
    np.testing.assert_allclose(
        inbound_eta(g, 4, stations, depart_s=300.0), td_travel_times_to(g, 4, stations, depart_s=300.0)
    )


# ------------------------------------------------------------------ EMS-02
@pytest.mark.req("EMS-02")
def test_ev_edge_times():
    from outrun.routing.emergency import ev_edge_times

    ff = np.array([10.0, 10.0, 10.0, 10.0])
    cong = np.array([10.0, 12.0, 30.0, np.inf])
    got = ev_edge_times(ff, cong, speedup=1.2)
    np.testing.assert_allclose(got[:3], [10.0, 10.0, 25.0])
    assert np.isinf(got[3])
    # broadcasting with a scalar free-flow time
    np.testing.assert_allclose(ev_edge_times(np.float64(5.0), np.array([6.0, 12.0]), speedup=2.0), [5.0, 6.0])
    with pytest.raises(ValueError):
        ev_edge_times(ff, cong, speedup=0.0)


# ------------------------------------------------------------------ EMS-03
def brute_dispatch(eta: np.ndarray) -> tuple[int, float]:
    """(max number of finite assignments, min total ETA among those)."""
    s, i = eta.shape
    best = (0, 0.0)
    for k in range(min(s, i), 0, -1):
        for incs in itertools.combinations(range(i), k):
            for sts in itertools.permutations(range(s), k):
                vals = [eta[a, b] for a, b in zip(sts, incs, strict=True)]
                if all(np.isfinite(vals)):
                    tot = float(sum(vals))
                    if best[0] < k or (best[0] == k and tot < best[1]):
                        best = (k, tot)
        if best[0] == k:
            break
    return best


def score(eta: np.ndarray, assign: list[int | None]) -> tuple[int, float]:
    used = [a for a in assign if a is not None]
    assert len(used) == len(set(used)), "a station was assigned twice"
    vals = [eta[a, j] for j, a in enumerate(assign) if a is not None]
    assert all(np.isfinite(vals)), "assigned an unreachable station"
    return len(vals), float(sum(vals))


@pytest.mark.req("EMS-03")
@pytest.mark.parametrize("seed", range(6))
def test_dispatch_matches_brute_force(seed):
    from outrun.routing.emergency import dispatch

    rng = np.random.default_rng(seed)
    s, i = int(rng.integers(1, 6)), int(rng.integers(1, 4))
    eta = rng.integers(60, 1200, size=(s, i)).astype(float)
    eta[rng.random((s, i)) < 0.25] = np.inf
    got = dispatch(eta, max_incidents=3)
    assert len(got) == i
    n, tot = score(eta, got)
    bn, btot = brute_dispatch(eta)
    assert n == bn
    assert tot == pytest.approx(btot)


@pytest.mark.req("EMS-03")
def test_dispatch_classic_case_and_edges():
    from outrun.routing.emergency import dispatch

    # Greedy (nearest per incident) would give station 0 to incident 0 (100) and station 1 to incident 1 (900):
    # total 1000. Optimum: station 1 -> incident 0 (150), station 0 -> incident 1 (200): total 350.
    eta = np.array([[100.0, 200.0], [150.0, 900.0]])
    assert dispatch(eta, max_incidents=3) == [1, 0]
    # more incidents than stations: one incident gets None
    got = dispatch(np.array([[10.0, 20.0, 30.0]]), max_incidents=3)
    assert sorted(x for x in got if x is not None) == [0] and got.count(None) == 2
    assert got[0] == 0
    # an incident no station can reach
    assert dispatch(np.array([[10.0, np.inf], [20.0, np.inf]]), max_incidents=3) == [0, None]
    assert dispatch(np.zeros((2, 0)), max_incidents=3) == []
    with pytest.raises(ValueError):
        dispatch(np.ones((5, 4)), max_incidents=3)


# ------------------------------------------------------------------ EMS-04
@pytest.mark.req("EMS-04")
def test_nearest_open_hospital():
    from outrun.routing.emergency import nearest_open_hospital

    g = nx.DiGraph()
    g.add_edge("inc", "h0", time=100.0)
    g.add_edge("inc", "h1", time=300.0)
    g.add_edge("inc", "h2", time=200.0)
    g.add_node("h3")  # unreachable
    hospitals = ["h0", "h1", "h2", "h3"]
    risk = np.array([0.9, 0.1, 0.5, 0.0])  # h0 is inside the hazard; h2 at exactly the threshold is open
    got = nearest_open_hospital(g, "inc", hospitals, risk, threshold=0.5)
    assert got is not None
    idx, eta = got
    assert idx == 2 and eta == pytest.approx(200.0)
    assert nearest_open_hospital(g, "inc", ["h0", "h3"], np.array([0.9, 0.0]), threshold=0.5) is None
    with pytest.raises(ValueError):
        nearest_open_hospital(g, "inc", hospitals, np.array([0.0]))


@pytest.mark.req("EMS-04")
def test_nearest_open_hospital_time_dependent():
    from outrun.routing.emergency import nearest_open_hospital

    g = nx.DiGraph()
    g.add_edge("inc", "h0", time=[100.0, 1000.0])  # fast only in slot 0
    g.add_edge("inc", "h1", time=[400.0, 400.0])
    hospitals, risk = ["h0", "h1"], np.zeros(2)
    assert nearest_open_hospital(g, "inc", hospitals, risk, depart_s=0.0)[0] == 0
    idx, eta = nearest_open_hospital(g, "inc", hospitals, risk, depart_s=1000.0)
    assert idx == 1 and eta == pytest.approx(400.0)


# ------------------------------------------------------------------ EMS-05
def reroute_graph() -> nx.DiGraph:
    """S -> A -> B -> T is the planned route (30). A -> C -> T (40) and S -> D -> T (60) are detours."""
    g = nx.DiGraph()
    for u, v, t in [
        ("S", "A", 10.0), ("A", "B", 10.0), ("B", "T", 10.0),
        ("A", "C", 15.0), ("C", "T", 25.0),
        ("S", "D", 30.0), ("D", "T", 30.0),
        ("A", "S", 10.0),
    ]:  # fmt: skip
        g.add_edge(u, v, time=t)
    return g


@pytest.mark.req("EMS-05")
def test_discover_and_reroute_no_block_is_plain_shortest():
    from outrun.routing.emergency import discover_and_reroute

    r = discover_and_reroute(reroute_graph(), "S", "T", blocked=set())
    assert r.path == ["S", "A", "B", "T"]
    assert r.eta_s == pytest.approx(30.0) and r.planned_eta_s == pytest.approx(30.0)
    assert r.detour_s == pytest.approx(0.0)
    assert not r.hit_block and r.n_replans == 0 and r.reached


@pytest.mark.req("EMS-05")
def test_discover_and_reroute_single_block():
    from outrun.routing.emergency import discover_and_reroute

    r = discover_and_reroute(reroute_graph(), "S", "T", blocked={("A", "B")})
    # drives S->A (10), discovers A->B, replans from A: A->C->T (40). True ETA 50, detour 20.
    assert r.path == ["S", "A", "C", "T"]
    assert r.eta_s == pytest.approx(50.0)
    assert r.planned_eta_s == pytest.approx(30.0)
    assert r.detour_s == pytest.approx(20.0)
    assert r.hit_block and r.n_replans == 1 and r.reached


@pytest.mark.req("EMS-05")
def test_discover_and_reroute_two_blocks_true_eta_includes_backtrack():
    from outrun.routing.emergency import discover_and_reroute

    r = discover_and_reroute(reroute_graph(), "S", "T", blocked={("A", "B"), ("C", "T")})
    # S->A (10); block A->B -> replan A->C->T; A->C (15); block C->T at C -> replan from C: no out-edge but C->T.
    # C has no other way -> stuck.
    assert not r.reached and math.isinf(r.eta_s)
    assert r.path == ["S", "A", "C"]
    assert r.n_replans == 2 and r.hit_block


@pytest.mark.req("EMS-05")
def test_discover_and_reroute_backtracks_when_possible():
    from outrun.routing.emergency import discover_and_reroute

    g = reroute_graph()
    r = discover_and_reroute(g, "S", "T", blocked={("A", "B"), ("A", "C")})
    # S->A (10); A->B blocked -> replan from A avoiding it: A->C->T (40); A->C is blocked too -> discovered at A
    # immediately -> replan: A->S->D->T (70). Driven: S,A,S,D,T = 10 + 10 + 30 + 30 = 80.
    assert r.path == ["S", "A", "S", "D", "T"]
    assert r.eta_s == pytest.approx(80.0)
    assert r.detour_s == pytest.approx(50.0)
    assert r.n_replans == 2 and r.reached


@pytest.mark.req("EMS-05")
def test_discover_and_reroute_unreachable_and_nearest_station():
    from outrun.routing.emergency import discover_and_reroute, nearest_station_straight_line

    g = reroute_graph()
    g.add_node("X")
    r = discover_and_reroute(g, "S", "X", blocked=set())
    assert not r.reached and r.path == ["S"] and math.isinf(r.eta_s) and math.isinf(r.planned_eta_s)

    st = np.array([[0.0, 0.0], [10.0, 0.0], [3.0, 4.0], [-3.0, 4.0]])
    assert nearest_station_straight_line(st, np.array([9.0, 1.0])) == 1
    assert nearest_station_straight_line(st, np.array([0.0, 5.0])) == 2  # tie 2/3 -> lowest index
    with pytest.raises(ValueError):
        nearest_station_straight_line(np.zeros((0, 2)), np.array([0.0, 0.0]))


# ------------------------------------------------------------------ EMS-06
def brute_weighted(g, s, t, lam):
    best = None
    for path in nx.all_simple_paths(g, s, t):
        es = list(zip(path[:-1], path[1:], strict=True))
        c = sum(g.edges[e]["time"] * (1 + lam * g.edges[e].get("risk", 0.0)) for e in es)
        if best is None or c < best:
            best = c
    return best


@pytest.mark.req("EMS-06")
def test_route_weighted_fast_risky_vs_slow_safe():
    from outrun.routing.emergency import pareto_sweep, route_weighted

    g = nx.DiGraph()
    g.add_edge("s", "m", time=100.0, risk=1.0)  # fast but through the hazard
    g.add_edge("m", "t", time=100.0, risk=1.0)
    g.add_edge("s", "n", time=150.0, risk=0.0)  # slow and safe
    g.add_edge("n", "t", time=150.0, risk=0.0)
    p0 = route_weighted(g, "s", "t", 0.0)
    assert (
        p0.path == ["s", "m", "t"]
        and p0.time_s == pytest.approx(200.0)
        and p0.exposure == pytest.approx(200.0)
    )
    p2 = route_weighted(g, "s", "t", 2.0)
    assert (
        p2.path == ["s", "n", "t"] and p2.time_s == pytest.approx(300.0) and p2.exposure == pytest.approx(0.0)
    )
    pts = pareto_sweep(g, "s", "t", lambdas=[0.0, 0.25, 0.5, 1.0, 2.0])
    assert [p.lam for p in pts] == [0.0, 0.25, 0.5, 1.0, 2.0]
    g.add_node("z")
    assert route_weighted(g, "s", "z", 1.0) is None
    assert pareto_sweep(g, "s", "z", lambdas=[0.0, 1.0]) == []
    with pytest.raises(ValueError):
        route_weighted(g, "s", "t", -0.5)


@pytest.mark.req("EMS-06")
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pareto_sweep_optimal_and_monotone(seed):
    from outrun.routing.emergency import pareto_sweep

    g = random_graph(seed, n=10, p=0.35)
    pairs = [(u, v) for u in g.nodes for v in g.nodes if u != v and nx.has_path(g, u, v)]
    s, t = pairs[0]
    lams = [0.0, 0.25, 0.5, 1.0, 2.0]
    pts = pareto_sweep(g, s, t, lambdas=lams)
    assert len(pts) == len(lams)
    for p in pts:
        es = list(zip(p.path[:-1], p.path[1:], strict=True))
        assert p.path[0] == s and p.path[-1] == t and all(g.has_edge(*e) for e in es)
        assert p.time_s == pytest.approx(sum(g.edges[e]["time"] for e in es))
        assert p.exposure == pytest.approx(sum(g.edges[e]["time"] * g.edges[e]["risk"] for e in es))
        assert p.time_s + p.lam * p.exposure == pytest.approx(brute_weighted(g, s, t, p.lam))
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        assert b.time_s >= a.time_s - 1e-9
        assert b.exposure <= a.exposure + 1e-9


# ------------------------------------------------------------------ EMS-07
@pytest.mark.req("EMS-07")
def test_ems_metrics():
    from outrun.routing.emergency import RerouteResult, ems_metrics

    rs = [
        RerouteResult(["a", "b"], 600.0, 600.0, 0.0, False, 0, True),
        RerouteResult(["a", "c"], 1200.0, 600.0, 600.0, True, 1, True),
        RerouteResult(["a"], math.inf, 600.0, math.inf, True, 1, False),
        RerouteResult(["a", "d"], 900.0, 900.0, 0.0, False, 0, True),
    ]
    m = ems_metrics(rs, exposures=[0.0, 10.0, 20.0, 30.0])
    assert m["mean_eta_min"] == pytest.approx(15.0)  # (10 + 20 + 15) / 3
    assert m["mean_detour_min"] == pytest.approx(10.0 / 3)
    assert m["pct_hit_block"] == pytest.approx(50.0)
    assert m["pct_unreached"] == pytest.approx(25.0)
    assert m["mean_exposure"] == pytest.approx(15.0)
    assert math.isnan(ems_metrics(rs)["mean_exposure"])
    with pytest.raises(ValueError):
        ems_metrics([])
