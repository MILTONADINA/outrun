"""Spec 05-shelters.md: demand, supply, unsafe-site filter, travel-time matrix, min-cost flow, baselines, metrics.

Tiny synthetic instances only. The min-cost-flow optimum is checked against an independent LP (scipy HiGHS); the
transportation LP is totally unimodular, so its optimum value equals the integer optimum.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
from scipy.optimize import linprog

INF = np.inf


# ------------------------------------------------------------------ helpers
def lp_optimum(demand, capacity, cost, u) -> float:
    z, s = cost.shape
    cells = [(i, j) for i in range(z) for j in range(s) if np.isfinite(cost[i, j])]
    nv = len(cells) + z
    c = np.array([cost[i, j] for i, j in cells] + [u] * z, dtype=float)
    a_eq = np.zeros((z, nv))
    for v, (i, _) in enumerate(cells):
        a_eq[i, v] = 1
    for i in range(z):
        a_eq[i, len(cells) + i] = 1
    a_ub = np.zeros((s, nv))
    for v, (_, j) in enumerate(cells):
        a_ub[j, v] = 1
    res = linprog(c, A_ub=a_ub if s else None, b_ub=capacity if s else None, A_eq=a_eq, b_eq=demand,
                  bounds=(0, None), method="highs")  # fmt: skip
    assert res.status == 0
    return float(res.fun)


def objective(a, cost, u) -> float:
    used = a.flow > 0
    return float((a.flow[used] * cost[used]).sum() + u * a.unserved.sum())


def check_feasible(a, demand, capacity, cost):
    assert a.flow.dtype == np.int64 and a.unserved.dtype == np.int64
    assert a.flow.shape == cost.shape and a.unserved.shape == demand.shape
    assert (a.flow >= 0).all() and (a.unserved >= 0).all()
    np.testing.assert_array_equal(a.flow.sum(axis=1) + a.unserved, demand)
    assert (a.flow.sum(axis=0) <= capacity).all()
    assert (a.flow[~np.isfinite(cost)] == 0).all()


def random_instance(seed):
    rng = np.random.default_rng(seed)
    z, s = int(rng.integers(2, 7)), int(rng.integers(1, 5))
    demand = rng.integers(0, 120, size=z).astype(np.int64)
    capacity = rng.integers(10, 150, size=s).astype(np.int64)
    cost = rng.integers(60, 3600, size=(z, s)).astype(float)
    cost[rng.random((z, s)) < 0.2] = INF
    return demand, capacity, cost


# ------------------------------------------------------------------ SHEL-01
@pytest.mark.req("SHEL-01")
def test_shelter_demand_rounding_and_errors():
    from outrun.optimize.shelters import shelter_demand

    pop = np.array([0, 4, 5, 15, 1234, 25])
    got = shelter_demand(pop, rate=0.1)
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, [0, 0, 1, 2, 123, 3])  # 0.5 -> 1, 1.5 -> 2, 2.5 -> 3 (half up)
    assert shelter_demand(np.array([], dtype=float), rate=0.1).dtype == np.int64
    assert shelter_demand(np.array([], dtype=float), rate=0.1).size == 0
    for bad in (np.array([-1.0, 3.0]), np.array([np.nan])):
        with pytest.raises(ValueError):
            shelter_demand(bad, rate=0.1)
    for rate in (-0.1, 1.5):
        with pytest.raises(ValueError):
            shelter_demand(np.array([10]), rate=rate)


# ------------------------------------------------------------------ SHEL-02
def fac(i, kind, cap):
    from outrun.optimize.shelters import Facility

    return Facility(id=f"{kind}{i}", kind=kind, node=i, x=float(i), y=0.0, capacity=cap)


@pytest.mark.req("SHEL-02")
def test_facility_kinds_match_data_pipeline_and_nan_capacity():
    import typing

    from outrun.data.facilities import FACILITY_KINDS
    from outrun.optimize.shelters import Facility, FacilityKind

    assert set(typing.get_args(FacilityKind)) == set(FACILITY_KINDS)
    f = Facility.from_record(
        {"facility_id": "fire_station-0", "kind": "fire_station", "capacity": float("nan")}, 3, 1.0, 2.0
    )
    assert f.kind == "fire_station" and f.capacity is None and f.node == 3 and f.id == "fire_station-0"
    s = Facility.from_record({"facility_id": "shelter-1", "kind": "shelter", "capacity": 300.0}, 4, 0.0, 0.0)
    assert s.capacity == 300 and isinstance(s.capacity, int)
    with pytest.raises(ValueError):
        Facility.from_record({"facility_id": "x", "kind": "fire_ems", "capacity": 1.0}, 0, 0.0, 0.0)


@pytest.mark.req("SHEL-02")
def test_build_supply_adds_assumed_fallbacks_when_few_shelters():
    from outrun.optimize.shelters import build_shelter_supply

    facs = [
        fac(0, "school", None),
        fac(1, "shelter", 300),
        fac(2, "hospital", 1000),
        fac(3, "fire_station", None),
        fac(4, "shelter", None),  # no capacity -> skipped
        fac(5, "school", 800),
        fac(6, "shelter", 0),  # zero capacity -> skipped
        fac(7, "fire_station", 20),
    ]
    sites = build_shelter_supply(facs, min_shelters=20, school_capacity=500, fire_station_capacity=50)
    assert [s.id for s in sites] == ["shelter1", "school0", "school5", "fire_station3", "fire_station7"]
    assert [s.capacity for s in sites] == [300, 500, 800, 50, 20]
    assert [s.assumed for s in sites] == [False, True, True, True, True]
    assert all(s.kind != "hospital" for s in sites)
    assert sites[0].node == 1 and sites[0].x == 1.0


@pytest.mark.req("SHEL-02")
def test_build_supply_no_fallback_when_enough_shelters():
    from outrun.optimize.shelters import build_shelter_supply

    facs = [fac(i, "shelter", 100 + i) for i in range(3)] + [
        fac(9, "school", None),
        fac(10, "fire_station", 5),
    ]
    sites = build_shelter_supply(facs, min_shelters=3, school_capacity=500, fire_station_capacity=50)
    assert [s.id for s in sites] == ["shelter0", "shelter1", "shelter2"]
    assert not any(s.assumed for s in sites)


# ------------------------------------------------------------------ SHEL-03
@pytest.mark.req("SHEL-03")
def test_filter_unsafe_sites_rules():
    from outrun.core.policy import Profile
    from outrun.core.risk import Hazard
    from outrun.optimize.shelters import ShelterSite, filter_unsafe_sites

    sites = [ShelterSite(f"s{i}", "shelter", i, 0.0, 0.0, 100, False) for i in range(7)]
    z = np.zeros(7)
    risk = {
        Hazard.WILDFIRE: np.array([0.6, 0, 0, 0, 0, 0, 0.5]),  # s6 exactly at threshold -> kept
        Hazard.TORNADO: np.array([0, 0.9, 0, 0, 0, 0, 0]),
        Hazard.FLOOD: np.array([0, 0, 0.51, 0, 0, 0, 0]),
        Hazard.SMOKE: np.array([0, 0, 0, 0.8, 0, 0, 0]),
        Hazard.STORM: np.array([0, 0, 0, 0, 0.99, 0, 0]),
        Hazard.ICE: np.array([0, 0, 0, 0, 0, 0.99, 0]),
    }
    civ = filter_unsafe_sites(sites, risk, Profile.CIVILIAN, threshold=0.5)
    assert [s.id for s in civ] == ["s3", "s4", "s5", "s6"]
    sens = filter_unsafe_sites(sites, risk, Profile.SENSITIVE, threshold=0.5)
    assert [s.id for s in sens] == ["s4", "s5", "s6"]
    ev = filter_unsafe_sites(sites, risk, Profile.EV, threshold=0.5)
    assert [s.id for s in ev] == ["s3", "s4", "s5", "s6"]
    assert len(sites) == 7  # input untouched
    assert [s.id for s in filter_unsafe_sites(sites, {}, Profile.CIVILIAN, threshold=0.5)] == [
        s.id for s in sites
    ]
    with pytest.raises(ValueError):
        filter_unsafe_sites(sites, {Hazard.FLOOD: z[:3]}, Profile.CIVILIAN, threshold=0.5)


# ------------------------------------------------------------------ SHEL-04
@pytest.mark.req("SHEL-04")
def test_travel_cost_matrix_static_equals_dijkstra():
    from outrun.optimize.shelters import travel_cost_matrix

    rng = np.random.default_rng(4)
    g0 = nx.gnp_random_graph(20, 0.2, seed=4, directed=True)
    g = nx.DiGraph()
    g.add_nodes_from(g0.nodes)
    for u, v in g0.edges:
        g.add_edge(u, v, time=float(rng.integers(30, 900)))
    zones, shelters = [0, 3, 5, 7, 11, 19], [2, 5, 13]
    got = travel_cost_matrix(g, zones, shelters)
    assert got.shape == (6, 3)
    for i, zn in enumerate(zones):
        for j, sn in enumerate(shelters):
            try:
                want = nx.dijkstra_path_length(g, zn, sn, weight="time")
            except nx.NetworkXNoPath:
                want = math.inf
            assert (math.isinf(got[i, j]) and math.isinf(want)) or got[i, j] == pytest.approx(want, abs=1e-9)
    assert got[zones.index(5), shelters.index(5)] == 0.0
    with pytest.raises(KeyError):
        travel_cost_matrix(g, [999], shelters)


@pytest.mark.req("SHEL-04")
def test_travel_cost_matrix_time_dependent():
    from outrun.optimize.shelters import travel_cost_matrix

    g = nx.DiGraph()
    g.add_edge("z", "a", time=[100.0, 100.0])
    g.add_edge("a", "s", time=[100.0, 2000.0])  # slows down in slot 1 (from 900 s)
    g.add_edge("z", "s", time=[800.0, 800.0])
    got0 = travel_cost_matrix(g, ["z"], ["s"], depart_s=0.0)
    assert got0[0, 0] == pytest.approx(200.0)
    got1 = travel_cost_matrix(
        g, ["z"], ["s"], depart_s=850.0
    )  # reaches a at 950 -> slot 1 -> 2100; direct 800
    assert got1[0, 0] == pytest.approx(800.0)


# ------------------------------------------------------------------ SHEL-05
@pytest.mark.req("SHEL-05")
def test_unserved_cost():
    from outrun.optimize.shelters import unserved_cost

    assert unserved_cost(np.array([[10.0, 300.0], [INF, 40.0]]), unserved_factor=10) == 3000
    assert unserved_cost(np.array([[INF, INF]]), unserved_factor=10) == 10
    assert unserved_cost(np.array([[0.0, 0.0]]), unserved_factor=10) == 10
    assert unserved_cost(np.array([[7.0]]), unserved_factor=2.5) == 18  # ceil(17.5)


@pytest.mark.req("SHEL-05")
@pytest.mark.parametrize("seed", range(8))
def test_min_cost_flow_is_exact_and_feasible(seed):
    from outrun.optimize.shelters import solve_min_cost_flow, unserved_cost

    demand, capacity, cost = random_instance(seed)
    a = solve_min_cost_flow(demand, capacity, cost, unserved_factor=10)
    assert a.method == "mcf"
    check_feasible(a, demand, capacity, cost)
    u = unserved_cost(cost, unserved_factor=10)
    assert objective(a, cost, u) == pytest.approx(lp_optimum(demand, capacity, cost, u), abs=1e-6)


@pytest.mark.req("SHEL-05")
def test_min_cost_flow_toy_unique_optimum():
    from outrun.optimize.shelters import solve_min_cost_flow

    # Zone 0 is close to both shelters; zone 1 can only use shelter 0 cheaply. Nearest-first would put zone 0 in
    # shelter 0 and force zone 1 far away; the optimum sends zone 0 to shelter 1.
    demand = np.array([100, 100])
    capacity = np.array([100, 100])
    cost = np.array([[60.0, 120.0], [60.0, 3000.0]])
    a = solve_min_cost_flow(demand, capacity, cost, unserved_factor=10)
    np.testing.assert_array_equal(a.flow, [[0, 100], [100, 0]])
    np.testing.assert_array_equal(a.unserved, [0, 0])


@pytest.mark.req("SHEL-05")
def test_min_cost_flow_overload_and_degenerate():
    from outrun.optimize.shelters import solve_min_cost_flow

    a = solve_min_cost_flow(
        np.array([80, 50]), np.array([60]), np.array([[100.0], [INF]]), unserved_factor=10
    )
    np.testing.assert_array_equal(a.flow, [[60], [0]])
    np.testing.assert_array_equal(a.unserved, [20, 50])
    b = solve_min_cost_flow(
        np.array([5, 0]), np.zeros(0, dtype=np.int64), np.zeros((2, 0)), unserved_factor=10
    )
    assert b.flow.shape == (2, 0)
    np.testing.assert_array_equal(b.unserved, [5, 0])
    c = solve_min_cost_flow(np.zeros(0, dtype=np.int64), np.array([10]), np.zeros((0, 1)), unserved_factor=10)
    assert c.flow.shape == (0, 1)
    with pytest.raises(ValueError):
        solve_min_cost_flow(np.array([1, 2]), np.array([5]), np.ones((3, 1)), unserved_factor=10)
    with pytest.raises(ValueError):
        solve_min_cost_flow(np.array([-1]), np.array([5]), np.ones((1, 1)), unserved_factor=10)
    with pytest.raises(ValueError):
        solve_min_cost_flow(np.array([1]), np.array([5]), np.array([[-3.0]]), unserved_factor=10)
    with pytest.raises(ValueError):
        solve_min_cost_flow(np.array([1]), np.array([5]), np.array([[np.nan]]), unserved_factor=10)


# ------------------------------------------------------------------ SHEL-06
@pytest.mark.req("SHEL-06")
def test_assign_nearest_ignores_capacity():
    from outrun.optimize.shelters import assign_nearest

    demand = np.array([100, 40, 30])
    capacity = np.array([50, 500])
    zone_xy = np.array([[0.0, 0.0], [1.0, 0.0], [9.0, 0.0]])
    site_xy = np.array([[0.0, 1.0], [10.0, 0.0]])
    cost = np.array([[60.0, 900.0], [60.0, INF], [900.0, 60.0]])
    a = assign_nearest(demand, capacity, zone_xy, site_xy, cost)
    assert a.method == "nearest"
    np.testing.assert_array_equal(a.flow, [[100, 0], [40, 0], [0, 30]])
    np.testing.assert_array_equal(a.unserved, [0, 0, 0])
    # equidistant -> lowest site index
    b = assign_nearest(np.array([7]), np.array([1, 1]), np.array([[5.0, 0.0]]), np.array([[0.0, 0.0], [10.0, 0.0]]),
                       np.array([[1.0, 1.0]]))  # fmt: skip
    np.testing.assert_array_equal(b.flow, [[7, 0]])
    with pytest.raises(ValueError):
        assign_nearest(demand, np.zeros(0), zone_xy, np.zeros((0, 2)), np.zeros((3, 0)))


@pytest.mark.req("SHEL-06")
def test_assign_greedy_order_and_capacity():
    from outrun.optimize.shelters import assign_greedy

    demand = np.array([30, 100, 30])
    capacity = np.array([80, 40, 10])
    cost = np.array(
        [
            [100.0, 200.0, 300.0],
            [100.0, 200.0, INF],
            [300.0, 100.0, 200.0],
        ]
    )
    a = assign_greedy(demand, capacity, cost)
    # zone 1 (100) first: 80 to site 0, 20 to site 1. Then zone 0 (tie with zone 2, lower index): site 0 full,
    # 20 to site 1 (remaining 20), 10 to site 2. Zone 2: site 1 full, site 2 full -> 30 unserved (site 0 full).
    assert a.method == "greedy"
    np.testing.assert_array_equal(a.flow, [[0, 20, 10], [80, 20, 0], [0, 0, 0]])
    np.testing.assert_array_equal(a.unserved, [0, 0, 30])
    check_feasible(a, demand, capacity, cost)
    e = assign_greedy(np.array([4]), np.zeros(0, dtype=np.int64), np.zeros((1, 0)))
    np.testing.assert_array_equal(e.unserved, [4])


@pytest.mark.req("SHEL-06")
@pytest.mark.parametrize("seed", range(4))
def test_greedy_always_feasible(seed):
    from outrun.optimize.shelters import assign_greedy

    demand, capacity, cost = random_instance(seed)
    check_feasible(assign_greedy(demand, capacity, cost), demand, capacity, cost)


# ------------------------------------------------------------------ SHEL-07
@pytest.mark.req("SHEL-07")
def test_assignment_metrics():
    from outrun.optimize.shelters import ShelterAssignment, assignment_metrics

    flow = np.array([[100, 0], [40, 0], [0, 30]], dtype=np.int64)
    a = ShelterAssignment("nearest", flow, np.array([0, 0, 10], dtype=np.int64))
    capacity = np.array([50, 500])
    cost = np.array([[60.0, 900.0], [120.0, INF], [900.0, 600.0]])
    expo = np.array([[0.0, 1.0], [0.5, 0.0], [0.0, 2.0]])
    m = assignment_metrics(a, capacity, cost, exposure=expo)
    assert m.served_persons == 170 and m.unserved_persons == 10
    assert m.total_person_minutes == pytest.approx((100 * 60 + 40 * 120 + 30 * 600) / 60)
    assert m.max_minutes == pytest.approx(10.0)
    assert m.overflow_persons == 90
    assert m.overflow_pct == pytest.approx(50.0)  # 90 / 180
    assert m.mean_exposure == pytest.approx((40 * 0.5 + 30 * 2.0) / 170)
    assert math.isnan(assignment_metrics(a, capacity, cost).mean_exposure)
    empty = ShelterAssignment("mcf", np.zeros((1, 1), dtype=np.int64), np.zeros(1, dtype=np.int64))
    me = assignment_metrics(empty, np.array([5]), np.array([[10.0]]))
    assert me.max_minutes == 0.0 and me.overflow_pct == 0.0 and me.total_person_minutes == 0.0


# ------------------------------------------------------------------ SHEL-08
@pytest.mark.req("SHEL-08")
@pytest.mark.parametrize("seed", range(8))
def test_mcf_never_worse_than_greedy_and_never_overflows(seed):
    from outrun.optimize.shelters import assign_greedy, assignment_metrics, solve_min_cost_flow, unserved_cost

    demand, capacity, cost = random_instance(seed)
    u = unserved_cost(cost, unserved_factor=10)
    m = solve_min_cost_flow(demand, capacity, cost, unserved_factor=10)
    g = assign_greedy(demand, capacity, cost)
    assert objective(m, cost, u) <= objective(g, cost, u) + 1e-9
    assert assignment_metrics(m, capacity, cost).overflow_persons == 0


@pytest.mark.req("SHEL-08")
def test_mcf_removes_nearest_overflow():
    from outrun.optimize.shelters import assign_nearest, assignment_metrics, solve_min_cost_flow

    demand = np.array([100, 100])
    capacity = np.array([120, 200])
    zone_xy = np.array([[0.0, 0.0], [1.0, 0.0]])
    site_xy = np.array([[0.0, 0.5], [5.0, 0.0]])
    cost = np.array([[60.0, 300.0], [60.0, 240.0]])
    near = assignment_metrics(assign_nearest(demand, capacity, zone_xy, site_xy, cost), capacity, cost)
    assert near.overflow_persons == 80
    mcf = solve_min_cost_flow(demand, capacity, cost, unserved_factor=10)
    mm = assignment_metrics(mcf, capacity, cost)
    assert mm.overflow_persons == 0 and mm.unserved_persons == 0
