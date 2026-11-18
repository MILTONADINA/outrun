"""Spec 06-evacuation.md: packets, diverse paths, wave-BPR evaluator, objective J, SA, MSA, baselines, lower bound.

Small synthetic instances. Hand-checkable numbers are written as the spec's formulas inside the test, so the test
documents the convention it checks. SA/MSA tests assert only spec guarantees (determinism, consistency, best-ever
<= init), never "finds the global optimum".
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

WAVE_S = 1800.0
SLOT_S = 900.0
PARAMS = dict(
    n_waves=4,
    wave_s=WAVE_S,
    slot_s=SLOT_S,
    bpr_alpha=0.15,
    bpr_beta=4.0,
    lambda_E=1.0,
    lambda_clear_per_vehicle=0.1,
    clearance_percentile=95.0,
    unserved_penalty=1000.0,
)


def inst(edge_t0, edge_cap, edge_risk, vehicles, candidates, **over):
    from outrun.optimize.evacuation import EvacInstance

    kw = {**PARAMS, **over}
    return EvacInstance(
        edge_t0=np.asarray(edge_t0, dtype=float),
        edge_cap=np.asarray(edge_cap, dtype=float),
        edge_risk=np.asarray(edge_risk, dtype=float),
        vehicles=np.asarray(vehicles, dtype=np.int64),
        candidates=[[np.asarray(p, dtype=np.int64) for p in c] for c in candidates],
        **kw,
    )


def plan(k, w):
    from outrun.optimize.evacuation import EvacPlan

    return EvacPlan(k=np.asarray(k, dtype=np.int64), w=np.asarray(w, dtype=np.int64))


def bpr(t0, x, cap):
    return t0 * (1 + 0.15 * (x / cap) ** 4)


def instance_a(**over):
    """3 edges; p0 (50 veh) has two candidates, p1 (30 veh) one, p2 (10 veh) none (unserved)."""
    risk = [[0.0, 0.5, 0.0, 0.0], [0.2, 0.2, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]]
    return inst([600, 1200, 300], [100, 200, 50], risk, [50, 30, 10], [[[0, 1], [2]], [[2]], []], **over)


def instance_slots():
    """Path A = edges [0, 1] (T0 1800, risky early on e0 and late on e1); path B = edge [2] (T0 1830, safe)."""
    risk = [[0.1, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.3], [0.0, 0.0, 0.0, 0.0]]
    return inst([600, 1200, 1830], [1e6, 1e6, 1e6], risk, [2], [[[0, 1], [2]]])


def random_instance(seed, n_edges=12, n_packets=8, n_waves=3):
    rng = np.random.default_rng(seed)
    t0 = rng.integers(200, 1500, size=n_edges).astype(float)
    cap = rng.integers(50, 400, size=n_edges).astype(float)
    risk = rng.choice([0.0, 0.0, 0.3, 1.0], size=(n_edges, 12))
    veh = rng.integers(5, 50, size=n_packets)
    cands = []
    for _ in range(n_packets):
        kk = int(rng.integers(1, 4))
        cands.append([rng.choice(n_edges, size=int(rng.integers(1, 5)), replace=False) for _ in range(kk)])
    return inst(t0, cap, risk, veh, cands, n_waves=n_waves)


def random_plan(instance, rng):
    k = [int(rng.integers(0, max(1, len(c)))) for c in instance.candidates]
    w = rng.integers(0, instance.n_waves, size=len(instance.candidates))
    return plan(k, w)


# ------------------------------------------------------------------ EVAC-01
@pytest.mark.req("EVAC-01")
def test_make_packets_split_and_order():
    from outrun.optimize.evacuation import Packet, make_packets

    pop = np.array([0.0, 1.39, 290.0, 125.0])  # vehicles: 0, 0.5004 -> 1, 104.4 -> 104, 45
    got = make_packets(pop, compliance=0.9, persons_per_vehicle=2.5, packet_max=50)
    assert got == [Packet(1, 1), Packet(2, 35), Packet(2, 35), Packet(2, 34), Packet(3, 45)]
    big = make_packets(np.array([1000.0]), compliance=1.0, persons_per_vehicle=1.0, packet_max=50)
    assert len(big) == 20 and all(p.vehicles == 50 for p in big)
    assert make_packets(np.array([1.0]), compliance=0.9, persons_per_vehicle=2.5, packet_max=50) == []


@pytest.mark.req("EVAC-01")
def test_make_packets_errors():
    from outrun.optimize.evacuation import make_packets

    ok = dict(compliance=0.9, persons_per_vehicle=2.5, packet_max=50)
    with pytest.raises(ValueError):
        make_packets(np.array([-5.0]), **ok)
    with pytest.raises(ValueError):
        make_packets(np.array([np.nan]), **ok)
    with pytest.raises(ValueError):
        make_packets(np.array([5.0]), compliance=1.2, persons_per_vehicle=2.5, packet_max=50)
    with pytest.raises(ValueError):
        make_packets(np.array([5.0]), compliance=0.9, persons_per_vehicle=0.0, packet_max=50)
    with pytest.raises(ValueError):
        make_packets(np.array([5.0]), compliance=0.9, persons_per_vehicle=2.5, packet_max=0)


# ------------------------------------------------------------------ EVAC-02
def parallel_routes(costs):
    g = nx.DiGraph()
    for i, c in enumerate(costs):
        g.add_edge("s", f"a{i}", cost=c / 2)
        g.add_edge(f"a{i}", "t", cost=c / 2)
    return g


@pytest.mark.req("EVAC-02")
def test_diverse_paths_four_parallel_routes():
    from outrun.optimize.evacuation import diverse_paths

    g = parallel_routes([20.0, 22.0, 24.0, 26.0])
    got = diverse_paths(g, "s", "t", k=4, penalty=1.4, weight="cost")
    assert got == [["s", "a0", "t"], ["s", "a1", "t"], ["s", "a2", "t"], ["s", "a3", "t"]]
    assert g.edges["s", "a0"]["cost"] == 10.0  # input graph untouched


@pytest.mark.req("EVAC-02")
def test_diverse_paths_penalty_compounds_and_duplicates_dropped():
    from outrun.optimize.evacuation import diverse_paths

    # A = 10, B = 15. Round 1: A (A -> 14). Round 2: A again (14 < 15) -> duplicate, A -> 19.6. Round 3: B.
    g = parallel_routes([10.0, 15.0])
    assert diverse_paths(g, "s", "t", k=3, penalty=1.4, weight="cost") == [["s", "a0", "t"], ["s", "a1", "t"]]
    assert diverse_paths(g, "s", "t", k=2, penalty=1.4, weight="cost") == [["s", "a0", "t"]]


@pytest.mark.req("EVAC-02")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_diverse_paths_valid_on_random_graph(seed):
    from outrun.optimize.evacuation import diverse_paths

    rng = np.random.default_rng(seed)
    g0 = nx.gnp_random_graph(30, 0.15, seed=seed, directed=True)
    g = nx.DiGraph()
    for u, v in g0.edges:
        g.add_edge(u, v, cost=float(rng.integers(10, 100)))
    s = 0
    targets = [t for t in nx.descendants(g, s) if t != s]
    assert targets
    t = targets[-1]
    paths = diverse_paths(g, s, t, k=4, penalty=1.4, weight="cost")
    assert 1 <= len(paths) <= 4
    assert len({tuple(p) for p in paths}) == len(paths)
    for p in paths:
        assert p[0] == s and p[-1] == t and len(set(p)) == len(p)
        assert all(g.has_edge(a, b) for a, b in zip(p[:-1], p[1:], strict=True))
    first = sum(g.edges[a, b]["cost"] for a, b in zip(paths[0][:-1], paths[0][1:], strict=True))
    assert first == pytest.approx(nx.dijkstra_path_length(g, s, t, weight="cost"))


@pytest.mark.req("EVAC-02")
def test_diverse_paths_edge_cases():
    from outrun.optimize.evacuation import diverse_paths

    g = parallel_routes([10.0])
    g.add_node("x")
    assert diverse_paths(g, "s", "s", k=4, penalty=1.4) == [["s"]]
    assert diverse_paths(g, "s", "x", k=4, penalty=1.4) == []
    with pytest.raises(KeyError):
        diverse_paths(g, "s", "nope", k=4, penalty=1.4)
    with pytest.raises(ValueError):
        diverse_paths(g, "s", "t", k=0, penalty=1.4)
    with pytest.raises(ValueError):
        diverse_paths(g, "s", "t", k=2, penalty=0.9)


# ------------------------------------------------------------------ EVAC-03
@pytest.mark.req("EVAC-03")
def test_bpr_time():
    from outrun.optimize.evacuation import bpr_time

    got = bpr_time(np.array([100.0, 100.0, 60.0]), np.array([0.0, 50.0, 200.0]), np.array([50.0, 50.0, 100.0]),
                   alpha=0.15, beta=4)  # fmt: skip
    np.testing.assert_allclose(got, [100.0, 115.0, 60.0 * (1 + 0.15 * 16)])
    with pytest.raises(ValueError):
        bpr_time(np.array([1.0]), np.array([1.0]), np.array([0.0]), alpha=0.15, beta=4)


@pytest.mark.req("EVAC-03")
def test_loads_waves_and_unserved():
    from outrun.optimize.evacuation import EvacEvaluator

    ev = EvacEvaluator(instance_a())
    loads = ev.loads(plan([0, 0, 0], [0, 1, 0]))
    want = np.zeros((4, 3))
    want[0, 0] = want[0, 1] = 50  # p0: T0 = 1800 -> window [0, 1800) -> wave 0 only (half-open)
    want[1, 2] = 30  # p1: window [1800, 2100) -> wave 1; p2 is unserved and loads nothing
    np.testing.assert_allclose(loads, want)


@pytest.mark.req("EVAC-03")
def test_loads_window_spanning_waves_extends_array():
    from outrun.optimize.evacuation import EvacEvaluator

    ev = EvacEvaluator(inst([2000.0], [3600.0], [[0.0]], [10], [[[0]]]))
    np.testing.assert_allclose(ev.loads(plan([0], [0]))[:, 0], [10, 10, 0, 0])  # [0, 2000) hits waves 0, 1
    l3 = ev.loads(plan([0], [3]))  # [5400, 7400) hits waves 3 and 4 -> 5 load waves
    assert l3.shape == (5, 1)
    np.testing.assert_allclose(l3[:, 0], [0, 0, 0, 10, 10])


# ------------------------------------------------------------------ EVAC-04
@pytest.mark.req("EVAC-04")
def test_evaluate_matches_hand_computation():
    from outrun.optimize.evacuation import EvacEvaluator

    ev = EvacEvaluator(instance_a())
    s = ev.evaluate(plan([0, 0, 0], [0, 1, 0]))
    cap_w = np.array([100, 200, 50]) * WAVE_S / 3600  # 50, 100, 25 vehicles per wave
    t00, t01 = bpr(600, 50, cap_w[0]), bpr(1200, 50, cap_w[1])
    t12 = bpr(300, 30, cap_w[2])
    assert s.congestion_vehh == pytest.approx((50 * t00 + 50 * t01 + 30 * t12) / 3600)
    # p0: e0 entered at 0 (slot 0, r = 0), e1 entered at 600 (slot 0, r = 0.2) -> 240 risk-s
    assert s.exposure_vehh == pytest.approx(50 * 0.2 * 1200 / 3600)
    a0, a1 = t00 + t01, WAVE_S + t12
    np.testing.assert_allclose(s.arrivals_s[:2], [a0, a1])
    assert math.isnan(s.arrivals_s[2])
    # served vehicles 80, 95 % = 76: cumulative 50 at a0, 80 at a1 -> a1
    assert s.clearance_h == pytest.approx(a1 / 3600)
    assert s.unserved_veh == pytest.approx(10)
    lam_c = 0.1 * 90
    want_j = s.congestion_vehh + 1.0 * s.exposure_vehh + lam_c * s.clearance_h + 1000.0 * 10
    assert pytest.approx(want_j) == s.J
    np.testing.assert_allclose(s.loads, ev.loads(plan([0, 0, 0], [0, 1, 0])))


@pytest.mark.req("EVAC-04")
def test_exposure_uses_entry_slot_with_clamping():
    from outrun.optimize.evacuation import EvacEvaluator

    ev = EvacEvaluator(instance_slots())
    # wave 0: e0 at 0 (slot 0, 0.1 * 600 = 60), e1 at 600 (slot 0, 0) -> 60
    assert ev.evaluate(plan([0], [0])).exposure_vehh == pytest.approx(2 * 60 / 3600)
    # wave 1: e0 at 1800 (slot 2), e1 at 2400 (slot 2) -> 0
    assert ev.evaluate(plan([0], [1])).exposure_vehh == pytest.approx(0.0)
    # wave 2: e0 at 3600 (slot 4 -> clamped 3, r 0), e1 at 4200 (slot 3, 0.3 * 1200 = 360) -> 360
    assert ev.evaluate(plan([0], [2])).exposure_vehh == pytest.approx(2 * 360 / 3600)


@pytest.mark.req("EVAC-04")
def test_clearance_percentile_and_invalid_plans():
    from outrun.optimize.evacuation import EvacEvaluator

    # 4 packets on 4 private edges with t0 = 100, 200, 300, 400 s; vehicles 10, 10, 10, 70
    ev = EvacEvaluator(inst([100, 200, 300, 400], [1e6] * 4, np.zeros((4, 1)), [10, 10, 10, 70],
                            [[[0]], [[1]], [[2]], [[3]]], clearance_percentile=25.0))  # fmt: skip
    s = ev.evaluate(plan([0, 0, 0, 0], [0, 0, 0, 0]))
    assert s.clearance_h == pytest.approx(300 / 3600, rel=1e-6)  # 25 veh reached at the third arrival
    ev95 = EvacEvaluator(inst([100], [1e6], np.zeros((1, 1)), [5], [[[0]]]))
    with pytest.raises(ValueError):
        ev95.evaluate(plan([1], [0]))  # k out of range
    with pytest.raises(ValueError):
        ev95.evaluate(plan([0], [4]))  # wave outside 0..n_waves-1


@pytest.mark.req("EVAC-04")
def test_evaluator_rejects_bad_instances():
    from outrun.optimize.evacuation import EvacEvaluator

    with pytest.raises(ValueError):
        EvacEvaluator(inst([100, 200], [10], np.zeros((2, 1)), [5], [[[0]]]))  # cap shape
    with pytest.raises(ValueError):
        EvacEvaluator(inst([100], [10], np.zeros((1, 1)), [5], [[[3]]]))  # edge index out of range


# ------------------------------------------------------------------ EVAC-05
@pytest.mark.req("EVAC-05")
@pytest.mark.parametrize("seed", range(4))
def test_delta_equals_full_reevaluation(seed):
    from outrun.optimize.evacuation import EvacEvaluator

    rng = np.random.default_rng(100 + seed)
    instance = random_instance(seed)
    ev = EvacEvaluator(instance)
    pl = random_plan(instance, rng)
    k0, w0 = pl.k.copy(), pl.w.copy()
    base = ev.evaluate(pl).J
    for _ in range(25):
        p = int(rng.integers(0, len(instance.candidates)))
        k = int(rng.integers(0, len(instance.candidates[p])))
        w = int(rng.integers(0, instance.n_waves))
        d = ev.delta(pl, p, k, w)
        k2, w2 = pl.k.copy(), pl.w.copy()
        k2[p], w2[p] = k, w
        want = ev.evaluate(plan(k2, w2)).J - base
        assert d == pytest.approx(want, abs=1e-9 * max(1.0, abs(base)))
        np.testing.assert_array_equal(pl.k, k0)
        np.testing.assert_array_equal(pl.w, w0)
    assert ev.delta(pl, 0, int(pl.k[0]), int(pl.w[0])) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ EVAC-06
def congested_choice_instance():
    """2 packets x 50 veh; edge 0 is narrow (cap 20 veh/h), edge 1 equal length but wide. All on edge 0 is bad."""
    return inst([600, 600], [20, 10000], np.zeros((2, 4)), [50, 50], [[[0], [1]], [[0], [1]]], n_waves=2)


@pytest.mark.req("EVAC-06")
def test_initial_temperature():
    from outrun.optimize.evacuation import initial_temperature

    assert initial_temperature([1.0, 2.0, 3.0], accept=0.8) == pytest.approx(-2.0 / math.log(0.8))
    assert initial_temperature([], accept=0.8) == 1.0
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            initial_temperature([1.0], accept=bad)
    with pytest.raises(ValueError):
        initial_temperature([-1.0], accept=0.8)


SA_KW = dict(
    iterations=2000,
    alpha=0.995,
    warmup_moves=100,
    warmup_accept=0.8,
    move_probs={"repick_path": 0.75, "repick_wave": 0.20, "swap_adjacent_waves": 0.05},
)


@pytest.mark.req("EVAC-06")
def test_sa_improves_bad_start_and_is_consistent():
    from outrun.optimize.evacuation import EvacEvaluator, simulated_annealing

    ev = EvacEvaluator(congested_choice_instance())
    calls = []
    res = simulated_annealing(ev, seed=0, callback=lambda i, c, b, t: calls.append((i, c, b, t)), callback_every=100,
                              **SA_KW)  # fmt: skip
    init_j = ev.evaluate(plan([0, 0], [0, 0])).J
    assert res.init_J == pytest.approx(init_j)
    assert res.best_J < init_j
    assert res.best_J == pytest.approx(ev.evaluate(res.best_plan).J, rel=1e-9)
    assert all(0 <= k < 2 for k in res.best_plan.k) and all(0 <= w < 2 for w in res.best_plan.w)
    assert [h[0] for h in res.history] == list(range(100, 2001, 100))
    assert calls == [tuple(h) for h in res.history]
    bests = [h[2] for h in res.history]
    assert all(b2 <= b1 + 1e-12 for b1, b2 in zip(bests[:-1], bests[1:], strict=True))
    assert res.T0 > 0 and all(len(h) == 4 for h in res.history)
    for i, _, _, temp in res.history:  # temperature after iteration i's cooling step
        assert temp == pytest.approx(res.T0 * 0.995**i, rel=1e-9)


@pytest.mark.req("EVAC-06")
def test_sa_deterministic_per_seed():
    from outrun.optimize.evacuation import EvacEvaluator, simulated_annealing

    instance = random_instance(7)
    r1 = simulated_annealing(EvacEvaluator(instance), seed=3, **SA_KW)
    r2 = simulated_annealing(EvacEvaluator(instance), seed=3, **SA_KW)
    np.testing.assert_array_equal(r1.best_plan.k, r2.best_plan.k)
    np.testing.assert_array_equal(r1.best_plan.w, r2.best_plan.w)
    assert r1.best_J == r2.best_J and r1.history == r2.history


@pytest.mark.req("EVAC-06")
@pytest.mark.parametrize("seed", [0, 1])
def test_sa_never_worse_than_its_start(seed):
    from outrun.optimize.evacuation import EvacEvaluator, simulated_annealing

    instance = random_instance(seed)
    ev = EvacEvaluator(instance)
    start = random_plan(instance, np.random.default_rng(seed))
    res = simulated_annealing(ev, init=start, seed=seed, zone_adjacency=[(0, 1)], **SA_KW)
    assert res.best_J <= ev.evaluate(start).J + 1e-9
    assert res.init_J == pytest.approx(ev.evaluate(start).J)


# ------------------------------------------------------------------ EVAC-07
@pytest.mark.req("EVAC-07")
def test_msa_picks_dominant_path():
    from outrun.optimize.evacuation import EvacEvaluator, msa_equilibrium

    instance = inst([100, 500], [1e6, 1e6], np.zeros((2, 4)), [20, 20, 20], [[[0], [1]]] * 3, n_waves=1)
    got = msa_equilibrium(EvacEvaluator(instance), iterations=20)
    np.testing.assert_array_equal(got.k, [0, 0, 0])
    np.testing.assert_array_equal(got.w, [0, 0, 0])


@pytest.mark.req("EVAC-07")
def test_msa_user_cost_includes_exposure():
    from outrun.optimize.evacuation import EvacEvaluator, msa_equilibrium

    risk = np.array([[10.0] * 4, [0.0] * 4])  # path 0: 100 s at r = 10 -> 1000 risk-s; user cost 1100 > 500
    instance = inst([100, 500], [1e6, 1e6], risk, [20, 20], [[[0], [1]]] * 2, n_waves=1)
    got = msa_equilibrium(EvacEvaluator(instance), iterations=20)
    np.testing.assert_array_equal(got.k, [1, 1])


@pytest.mark.req("EVAC-07")
def test_msa_deterministic_valid_and_sa_from_msa_not_worse():
    from outrun.optimize.evacuation import EvacEvaluator, msa_equilibrium, simulated_annealing

    instance = random_instance(11)
    ev = EvacEvaluator(instance)
    m1 = msa_equilibrium(ev, iterations=15)
    m2 = msa_equilibrium(ev, iterations=15)
    np.testing.assert_array_equal(m1.k, m2.k)
    np.testing.assert_array_equal(m1.w, m2.w)
    for p, c in enumerate(instance.candidates):
        assert 0 <= m1.k[p] < len(c) and 0 <= m1.w[p] < instance.n_waves
    j_msa = ev.evaluate(m1).J
    res = simulated_annealing(ev, init=m1, seed=0, **SA_KW)
    assert (
        res.best_J <= j_msa + 1e-9
    )  # system optimum search started at UE is never worse (DESIGN claim SA <= MSA)
    with pytest.raises(ValueError):
        msa_equilibrium(ev, iterations=0)


# ------------------------------------------------------------------ EVAC-08
@pytest.mark.req("EVAC-08")
def test_baseline_plans():
    from outrun.optimize.evacuation import EvacEvaluator, baseline_plan

    ev = EvacEvaluator(instance_slots())
    sh = baseline_plan(ev, "shortest")  # A: T0 1800 < B: 1830
    np.testing.assert_array_equal(sh.k, [0])
    np.testing.assert_array_equal(sh.w, [0])
    sf = baseline_plan(ev, "safest")  # A: 1800 + 60 = 1860 > B: 1830 + 0
    np.testing.assert_array_equal(sf.k, [1])
    np.testing.assert_array_equal(sf.w, [0])
    tie = EvacEvaluator(inst([100, 100], [1e6, 1e6], np.zeros((2, 1)), [5], [[[0], [1]]]))
    np.testing.assert_array_equal(baseline_plan(tie, "shortest").k, [0])
    with pytest.raises(ValueError):
        baseline_plan(ev, "fastest")


@pytest.mark.req("EVAC-08")
def test_lower_bound_value():
    from outrun.optimize.evacuation import EvacEvaluator

    # min over (k, w): A in wave 1 has exposure 0 -> 1800 s; 2 vehicles -> 1.0 veh-h
    assert EvacEvaluator(instance_slots()).lower_bound() == pytest.approx(1.0)


@pytest.mark.req("EVAC-08")
@pytest.mark.parametrize("seed", range(4))
def test_every_plan_scores_at_least_the_lower_bound(seed):
    from outrun.optimize.evacuation import EvacEvaluator, baseline_plan, msa_equilibrium

    instance = random_instance(seed)
    ev = EvacEvaluator(instance)
    lb = ev.lower_bound()
    rng = np.random.default_rng(seed)
    plans = [random_plan(instance, rng) for _ in range(10)]
    plans += [baseline_plan(ev, "shortest"), baseline_plan(ev, "safest"), msa_equilibrium(ev, iterations=5)]
    for pl in plans:
        assert lb - 1e-9 <= ev.evaluate(pl).J


# ------------------------------------------------------------------ EVAC-09
@pytest.mark.req("EVAC-09")
def test_lambda_e_zero_ablation():
    from outrun.optimize.evacuation import EvacEvaluator

    pl = plan([0, 0, 0], [0, 1, 0])
    s1 = EvacEvaluator(instance_a(lambda_E=1.0)).evaluate(pl)
    s0 = EvacEvaluator(instance_a(lambda_E=0.0)).evaluate(pl)
    assert s0.exposure_vehh == pytest.approx(s1.exposure_vehh) and s1.exposure_vehh > 0
    assert pytest.approx(s1.exposure_vehh) == s1.J - s0.J


# ------------------------------------------------------------------ EVAC-10
@pytest.mark.req("EVAC-10")
def test_evac_metrics():
    from outrun.optimize.evacuation import EvacEvaluator, evac_metrics

    ev = EvacEvaluator(instance_a())
    pl = plan([0, 0, 0], [0, 1, 0])
    s = ev.evaluate(pl)
    m = evac_metrics(ev, pl)
    assert m["vehicle_hours"] == pytest.approx(s.congestion_vehh)
    assert m["exposure_veh_h"] == pytest.approx(s.exposure_vehh)
    assert m["clearance_min"] == pytest.approx(s.clearance_h * 60)
    assert m["max_vc"] == pytest.approx(30 / 25)  # edge 2 in wave 1
    assert m["unserved_veh"] == pytest.approx(10)
    assert m["J"] == pytest.approx(s.J)
    lb = ev.lower_bound()
    assert m["gap_to_lb_pct"] == pytest.approx(100 * (s.J - lb) / lb)
