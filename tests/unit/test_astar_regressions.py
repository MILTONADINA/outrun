"""Spec 04-astar regressions: same-time label dominance under the cap (ASTAR-13) and float slot boundaries in the
wait jump and wait cost (ASTAR-07, ASTAR-10). Synthetic graphs only."""

from __future__ import annotations

import signal
from contextlib import contextmanager

import numpy as np
import pytest
from tests.fixtures.graphs.graphfix import make_road_graph

from outrun.routing.astar import astar
from outrun.routing.cost import CostModel

NAN = float("nan")


@contextmanager
def _deadline(seconds: int):
    """Fail instead of hanging: the float-boundary bugs were infinite loops."""

    def _raise(*_):
        raise TimeoutError(f"did not finish within {seconds} s")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _eid(g, u, v):
    return next(e for e in range(g.n_edges) if int(g.src[e]) == u and int(g.dst[e]) == v)


@pytest.mark.req("ASTAR-13")
def test_cheaper_equal_time_label_replaces_costlier_under_cap():
    # o=0 -> a=1 -> v=3 and o -> b=2 -> v, both arriving at v at the same clock. Without the heuristic, a (g lower)
    # is expanded first, so the costlier label reaches v first; the cheaper equal-time label must replace it.
    g, _ = make_road_graph(
        [-97.5, -97.49, -97.49, -97.48, -97.47],
        [35.5, 35.51, 35.49, 35.5, 35.5],
        [0, 0, 1, 2, 3],
        [1, 2, 3, 3, 4],
        [600] * 5,
        [10.0] * 5,
    )
    risk = np.zeros((g.n_edges, 1, 3))
    risk[_eid(g, 0, 2), 0, :] = 0.1
    risk[_eid(g, 1, 3), 0, :] = 0.5
    risk[_eid(g, 3, 4), 0, 2] = 0.3  # makes the model time-dependent
    cm = CostModel(g.free_flow_s, risk, np.array([10.0]), np.array([NAN]), slot_s=300.0)
    exact = astar(g, cm, 0, 4, 0.0, max_labels=10**6, use_heuristic=False)
    for use_h in (False, True):
        r = astar(g, cm, 0, 4, 0.0, max_labels=1, use_heuristic=use_h)
        assert r.cost == pytest.approx(exact.cost)
        assert r.nodes == [0, 2, 3, 4]
        assert r.approximate is False


@pytest.mark.req("ASTAR-07")
def test_wait_jump_terminates_when_clock_sits_on_float_slot_boundary():
    # slot_s = 0.1: t = 0.5 gives 0.5 // 0.1 == 4 but 5 * 0.1 == 0.5, so the next boundary equals t.
    g, _ = make_road_graph([-97.5, -97.49], [35.5, 35.5], [0], [1], [600], [10.0])
    risk = np.zeros((1, 1, 10))
    risk[0, 0, :6] = 1.0  # blocked in slots 0..5
    cm = CostModel(g.free_flow_s, risk, np.array([1.0]), np.array([0.5]), slot_s=0.1)
    with _deadline(5):
        r = astar(g, cm, 0, 1, 0.5, wait_step_s=0.1)
    assert r is not None
    assert cm.slot(r.depart_s[0]) >= 6
    assert r.cost == pytest.approx(60.0 + (r.depart_s[0] - 0.5))


@pytest.mark.req("ASTAR-10")
def test_wait_cost_terminates_on_float_slot_boundary():
    g, _ = make_road_graph([-97.5, -97.49], [35.5, 35.5], [0], [1], [600], [10.0])
    n_slots = 10
    node_risk = np.zeros((2, 1, n_slots))
    node_risk[0, 0, :] = np.arange(n_slots) / 10
    cm = CostModel(
        g.free_flow_s,
        np.zeros((1, 1, n_slots)),
        np.array([1.0]),
        np.array([NAN]),
        slot_s=0.1,
        node_risk=node_risk,
    )
    with _deadline(5):
        w = cm.wait_cost(0, 0.5, 0.7)
    assert w == pytest.approx(0.1 * 1.5 + 0.1 * 1.6)
