"""Own time-dependent multi-label A* with waiting, Dijkstra oracle, brute-force checker
(spec 04-astar, ASTAR-01..ASTAR-06, ASTAR-12, ASTAR-13, ASTAR-15..ASTAR-17).

A label is (node, clock t, cost g). Label (t1, g1) dominates (t2, g2) at the same node iff t1 <= t2 and
g1 + W(node, t1 -> t2) <= g2. Each node keeps a Pareto set (capped at ``max_labels``; a cap hit sets
``Route.approximate``). The queue is ordered by f = g + h; the search stops when a goal label is popped.
Edges are traversed with ``CostModel.clock``/``cost`` evaluated at the edge entry time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from outrun.roads.graph import RoadGraph
from outrun.routing.cost import CostModel

DEFAULT_MAX_LABELS = 8  # configs/planning.yaml astar.max_labels_per_node


@dataclass(frozen=True)
class Route:
    """Result of a search (ASTAR-16). Times are seconds since t0.

    ``depart_s[i]`` is the time the vehicle enters ``edges[i]`` (after any wait ``waits_s[i]`` at ``nodes[i]``).
    ``nodes`` has ``len(edges) + 1`` entries; ``nodes[0]`` is the origin and ``nodes[-1]`` the reached goal.
    """

    nodes: list[int]
    edges: list[int]
    depart_s: list[float]
    waits_s: list[float]
    start_s: float
    arrive_s: float
    cost: float
    expansions: int
    approximate: bool = False

    @property
    def goal(self) -> int:
        return self.nodes[-1]

    @property
    def travel_s(self) -> float:
        return self.arrive_s - self.start_s


def dominates(t1: float, g1: float, t2: float, g2: float, wait: Callable[[float, float], float]) -> bool:
    """Label dominance (ASTAR-13): ``t1 <= t2 and g1 + wait(t1, t2) <= g2``."""
    raise NotImplementedError


def astar(
    graph: RoadGraph,
    cost_model: CostModel,
    origin: int,
    goals: int | Sequence[int],
    t0_s: float = 0.0,
    *,
    use_heuristic: bool = True,
    allow_wait: bool = True,
    max_labels: int = DEFAULT_MAX_LABELS,
    wait_step_s: float = 60.0,
) -> Route | None:
    """Minimum generalized-cost route from ``origin`` to the cheapest of ``goals`` departing at ``t0_s``.

    Exact (ASTAR-01, ASTAR-04) unless ``Route.approximate``. ``use_heuristic=False`` is plain (multi-label)
    Dijkstra with identical results (ASTAR-02). Blocked edges are never used (ASTAR-03). Returns None if no goal
    is reachable with finite cost (ASTAR-05). Waiting (allow_wait): a label that reaches node n at time t may
    leave at t or at any later time ``t0_s + k * wait_step_s`` up to the end of the risk window
    (``n_slots * slot_s``); waits never extend past the window end. Exact w.r.t. continuous waiting when every clock
    time and t0_s are multiples of ``wait_step_s`` and the label cap is not hit.
    origin in goals -> zero-edge route with cost 0. Errors (ASTAR-17): node index out of range or empty goals
    -> ValueError; t0_s < 0 -> ValueError; ``cost_model.n_edges != graph.n_edges`` -> ValueError; max_labels < 1
    or wait_step_s <= 0 -> ValueError.
    """
    raise NotImplementedError


def dijkstra(
    graph: RoadGraph, cost_model: CostModel, origin: int, goals: int | Sequence[int], t0_s: float = 0.0, **kw
) -> Route | None:
    """Oracle: ``astar(..., use_heuristic=False)`` (ASTAR-02)."""
    raise NotImplementedError


def time_expanded_dijkstra(
    graph: RoadGraph,
    cost_model: CostModel,
    origin: int,
    goal: int,
    t0_s: float = 0.0,
    *,
    step_s: float = 60.0,
    horizon_s: float | None = None,
) -> tuple[float, float] | None:
    """Brute-force checker (ASTAR-15): plain Dijkstra over states (node, k) meaning clock ``t0_s + k*step_s``.

    Moves: wait one step (cost ``cost_model.wait_cost(n, t, t+step_s)``), or traverse edge e entering at t,
    arriving at ``t + clock(e, t)`` rounded UP to the next step multiple (exact when all clock times are
    multiples of ``step_s``), cost ``cost(e, t)``. States beyond ``horizon_s`` (default: window end + sum of
    capped clock times of all edges) are dropped. Returns ``(min cost, arrival clock)`` at ``goal`` (ties: the
    earliest arrival), or None if unreachable.
    """
    raise NotImplementedError
