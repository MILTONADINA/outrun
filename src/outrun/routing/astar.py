"""Own time-dependent multi-label A* with waiting, Dijkstra oracle, brute-force checker
(spec 04-astar, ASTAR-01..ASTAR-06, ASTAR-12, ASTAR-13, ASTAR-15..ASTAR-17).

A label is (node, clock t, cost g). Label (t1, g1) dominates (t2, g2) at the same node iff t1 <= t2 and
g1 + W(node, t1 -> t2) <= g2. Each node keeps a Pareto set (capped at ``max_labels``; a cap hit sets
``Route.approximate``). The queue is ordered by f = g + h; the search stops when a goal label is popped.
Edges are traversed with ``CostModel.clock``/``cost`` evaluated at the edge entry time.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from outrun.roads.graph import RoadGraph
from outrun.routing.cost import CostModel, make_heuristic

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
    return t1 <= t2 and g1 + wait(t1, t2) <= g2 + _EPS * max(1.0, abs(g2))


_EPS = 1e-12  # relative float tolerance for cost comparisons


def _check_inputs(graph, cost_model, origin, goals, t0_s, max_labels, wait_step_s) -> list[int]:
    """ASTAR-17 input errors; returns the goals as a list of ints."""
    n = graph.n_nodes
    goal_list = [int(goals)] if np.isscalar(goals) else [int(g) for g in goals]
    if not goal_list:
        raise ValueError("goals must not be empty")
    for node in [origin, *goal_list]:
        if not 0 <= int(node) < n:
            raise ValueError(f"node {node} outside [0, {n})")
    if not t0_s >= 0:
        raise ValueError(f"t0_s must be >= 0, got {t0_s}")
    if max_labels < 1:
        raise ValueError(f"max_labels must be >= 1, got {max_labels}")
    if not wait_step_s > 0:
        raise ValueError(f"wait_step_s must be > 0, got {wait_step_s}")
    if cost_model.n_edges != graph.n_edges:
        raise ValueError(f"cost model has {cost_model.n_edges} edges, graph has {graph.n_edges}")
    return goal_list


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

    Two search loops share this entry point:
    * static model (every slot identical): time does not change any cost, so one label per node suffices and
      the search is classic A* with a best-cost array (fast path, used for zero/static risk);
    * time-varying model: the multi-label search with waiting (:func:`_search_time_dependent`).
    """
    origin = int(origin)
    goal_list = _check_inputs(graph, cost_model, origin, goals, t0_s, max_labels, wait_step_s)
    t0_s = float(t0_s)
    if origin in goal_list:
        return Route([origin], [], [], [], t0_s, t0_s, 0.0, 0, False)
    h = make_heuristic(graph, goal_list).tolist() if use_heuristic else [0.0] * graph.n_nodes
    if cost_model.last_change_s() <= 0.0:
        return _search_static(graph, cost_model, origin, set(goal_list), t0_s, h)
    return _search_time_dependent(
        graph, cost_model, origin, set(goal_list), t0_s, h, allow_wait, max_labels, float(wait_step_s)
    )


def _search_static(graph, cost_model, origin, goals, t0_s, h) -> Route | None:
    """Classic A* when costs do not depend on time: one label per node, ``best_g`` array, lazy deletion."""
    clock_tab, cost_tab = cost_model.search_lists()
    indptr, dst = graph.indptr.tolist(), graph.dst.tolist()
    n = graph.n_nodes
    best_g = [math.inf] * n
    arrive = [0.0] * n
    parent_edge = [-1] * n
    done = [False] * n
    best_g[origin], arrive[origin] = 0.0, t0_s
    heap = [(h[origin], 0.0, origin)]  # (f, -g, node): ties go to the deeper label
    expansions = 0
    while heap:
        _, neg_g, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        expansions += 1
        if u in goals:
            return _static_route(graph, cost_model, origin, u, parent_edge, arrive, t0_s, expansions)
        g_u = -neg_g
        for e in range(indptr[u], indptr[u + 1]):
            v = dst[e]
            g_v = g_u + cost_tab[e][0]  # inf when blocked: never improves
            if g_v < best_g[v]:
                best_g[v] = g_v
                arrive[v] = arrive[u] + clock_tab[e][0]
                parent_edge[v] = e
                heapq.heappush(heap, (g_v + h[v], -g_v, v))
    return None


def _static_route(graph, cost_model, origin, goal, parent_edge, arrive, t0_s, expansions) -> Route:
    edges = []
    node = goal
    while node != origin:
        e = parent_edge[node]
        edges.append(e)
        node = int(graph.src[e])
    edges.reverse()
    return _route_from_plan(graph, cost_model, origin, edges, [0.0] * len(edges), t0_s, expansions, False)


def _route_from_plan(graph, cost_model, origin, edges, waits, t0_s, expansions, approximate) -> Route:
    """Re-drive a list of (wait, edge) moves to fill every Route field consistently (ASTAR-16)."""
    nodes, depart = [origin], []
    t, cost = t0_s, 0.0
    for e, w in zip(edges, waits, strict=True):
        cost += cost_model.wait_cost(nodes[-1], t, t + w)
        t += w
        depart.append(t)
        cost += cost_model.cost(e, t)
        t += cost_model.clock(e, t)
        nodes.append(int(graph.dst[e]))
    return Route(nodes, list(edges), depart, list(waits), t0_s, t, cost, expansions, approximate)


def _search_time_dependent(
    graph, cost_model, origin, goals, t0_s, h, allow_wait, max_labels, step
) -> Route | None:
    """Multi-label time-dependent A* with waiting on a grid (DESIGN 4.3, ASTAR-04/13).

    Labels live in parallel lists (``lab_*``). ``labels_at[n]`` is the Pareto set of live label ids at node n.
    The queue holds two kinds of items:
      * a *label* item: the label itself, departing now;
      * a *wait* item: "label L, but departing one wait step later". Wait items are not stored in the Pareto
        set (their own label always dominates them through the waiting rule), but they are dropped when some
        *other* label at the node dominates them, and they chain one step at a time up to the window end.
    """
    clock_tab, cost_tab = cost_model.search_lists()
    indptr, dst = graph.indptr.tolist(), graph.dst.tolist()
    slot_s, last_slot = cost_model.slot_s, cost_model.n_slots - 1
    window_end = cost_model.window_end_s
    last_change = cost_model.last_change_s()  # waiting after this time can never pay
    wait_cost = cost_model.wait_cost
    uniform_wait = not cost_model.has_node_risk()  # waiting costs 1 per second at every node

    lab_node: list[int] = []
    lab_t: list[float] = []
    lab_g: list[float] = []
    lab_parent: list[int] = []  # parent label id (-1 for the origin label)
    lab_edge: list[int] = []  # edge taken from the parent
    lab_wait: list[float] = []  # wait at the parent's node before that edge
    lab_alive: list[bool] = []
    labels_at: list[list[int]] = [[] for _ in range(graph.n_nodes)]
    approximate = False

    def label_dominates(i: int, t: float, g: float) -> bool:
        if allow_wait:
            if uniform_wait:  # dominates() with wait(t1, t2) = t2 - t1, inlined: this is the hot loop
                return lab_t[i] <= t and lab_g[i] + (t - lab_t[i]) <= g + _EPS * max(1.0, abs(g))
            return dominates(lab_t[i], lab_g[i], t, g, lambda a, b: wait_cost(lab_node[i], a, b))
        return lab_t[i] == t and lab_g[i] <= g  # without waiting, only an equal-time label can stand in

    def add_label(node, t, g, parent, edge, wait) -> int | None:
        nonlocal approximate
        pareto = labels_at[node]
        for i in pareto:
            if label_dominates(i, t, g):
                return None
        lab_node.append(node)
        lab_t.append(t)
        lab_g.append(g)
        lab_parent.append(parent)
        lab_edge.append(edge)
        lab_wait.append(wait)
        lab_alive.append(True)
        new = len(lab_node) - 1
        survivors = []
        for i in pareto:  # the new label may dominate older ones
            if label_dominates(new, lab_t[i], lab_g[i]):
                lab_alive[i] = False
            else:
                survivors.append(i)
        if len(survivors) >= max_labels:  # cap hit: keep the old set, drop the newcomer, flag the result
            approximate = True
            lab_alive[new] = False
            return None
        survivors.append(new)
        labels_at[node] = survivors
        return new

    counter = 0  # FIFO tie-break among equal (f, -g)
    first = add_label(origin, t0_s, 0.0, -1, -1, 0.0)
    # heap item: (f, -g, counter, label id, departure time, cost so far, accumulated wait)
    heap = [(h[origin], 0.0, counter, first, t0_s, 0.0, 0.0)]
    expansions = 0
    while heap:
        _, _, _, lab, t, g, waited = heapq.heappop(heap)
        if not lab_alive[lab]:
            continue
        u = lab_node[lab]
        if waited > 0.0 and any(label_dominates(i, t, g) for i in labels_at[u] if i != lab):
            continue  # another label at u can reach this (time, cost) more cheaply
        expansions += 1
        if waited == 0.0 and u in goals:
            return _td_route(
                graph, cost_model, lab, lab_parent, lab_edge, lab_wait, t0_s, expansions, approximate
            )

        s = min(int(t // slot_s), last_slot)
        for e in range(indptr[u], indptr[u + 1]):
            c = cost_tab[e][s]
            if c == math.inf:  # blocked at entry time: never used (ASTAR-03)
                continue
            v = dst[e]
            t_v, g_v = t + clock_tab[e][s], g + c
            new = add_label(v, t_v, g_v, lab, e, waited)
            if new is not None:
                counter += 1
                heapq.heappush(heap, (g_v + h[v], -g_v, counter, new, t_v, g_v, 0.0))

        if allow_wait and t < last_change:  # wait at u until the next allowed departure time
            if uniform_wait:
                # Every node waits at rate 1, so a wait that stays inside the current slot buys nothing here:
                # the same wait can be taken at the next node instead, at the same cost and clock. Only a wait
                # that crosses into the next slot can change an edge's cost, so jump straight to the first
                # grid time at or after that slot boundary.
                boundary = (s + 1) * slot_s
                t_next = t0_s + math.ceil((boundary - t0_s) / step - 1e-9) * step
            else:  # node risk: waiting is cheaper at some nodes, so try every grid step
                t_next = t0_s + (math.floor((t - t0_s) / step + 1e-9) + 1) * step
            if t_next <= window_end + 1e-9:
                g_next = g + wait_cost(u, t, t_next)
                counter += 1
                heapq.heappush(
                    heap, (g_next + h[u], -g_next, counter, lab, t_next, g_next, waited + (t_next - t))
                )
    return None


def _td_route(
    graph, cost_model, goal_lab, lab_parent, lab_edge, lab_wait, t0_s, expansions, approximate
) -> Route:
    edges, waits = [], []
    lab = goal_lab
    while lab_parent[lab] != -1:
        edges.append(lab_edge[lab])
        waits.append(lab_wait[lab])
        lab = lab_parent[lab]
    edges.reverse()
    waits.reverse()
    origin = int(graph.src[edges[0]])
    return _route_from_plan(graph, cost_model, origin, edges, waits, t0_s, expansions, approximate)


def dijkstra(
    graph: RoadGraph, cost_model: CostModel, origin: int, goals: int | Sequence[int], t0_s: float = 0.0, **kw
) -> Route | None:
    """Oracle: ``astar(..., use_heuristic=False)`` (ASTAR-02)."""
    return astar(graph, cost_model, origin, goals, t0_s, use_heuristic=False, **kw)


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
    if horizon_s is None:
        horizon_s = max(cost_model.window_end_s, t0_s) + float(cost_model.clock_table().max(axis=1).sum())
    k_max = int(math.floor((horizon_s - t0_s) / step_s + 1e-9))
    best = {(origin, 0): 0.0}
    heap = [(0.0, 0, origin)]  # (cost, step k, node): equal costs pop the earliest arrival first
    while heap:
        g, k, u = heapq.heappop(heap)
        if g > best.get((u, k), math.inf):
            continue
        t = t0_s + k * step_s
        if u == goal:
            return g, t
        moves = []
        if k + 1 <= k_max:
            moves.append((u, k + 1, cost_model.wait_cost(u, t, t + step_s)))
        for e in graph.out_edges(u):
            c = cost_model.cost(e, t)
            if math.isinf(c):
                continue
            kk = k + math.ceil(cost_model.clock(e, t) / step_s - 1e-9)
            if kk <= k_max:
                moves.append((int(graph.dst[e]), kk, c))
        for v, kk, c in moves:
            if g + c < best.get((v, kk), math.inf):
                best[(v, kk)] = g + c
                heapq.heappush(heap, (g + c, kk, v))
    return None
