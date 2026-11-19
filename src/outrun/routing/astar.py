"""Own time-dependent multi-label A* with waiting, Dijkstra oracle, brute-force checker
(spec 04-astar, ASTAR-01..ASTAR-06, ASTAR-12, ASTAR-13, ASTAR-15..ASTAR-17).

A label is (node, clock t, cost g). Label (t1, g1) dominates (t2, g2) at the same node iff t1 <= t2 and
g1 + W(node, t1 -> t2) <= g2. Each node keeps its non-dominated labels, grouped by risk slot so the test is a
binary search (see :func:`_search_time_dependent`). ``max_labels`` per node is an approximation cap: on large
graphs with moving storms it is often hit, and a hit sets ``Route.approximate``. It bounds labels per node, not
total runtime. The queue is ordered by f = g + h with a slot-aware heuristic that also counts risk
(:class:`SlotHeuristic`); the search stops when a goal label is popped.
Edges are traversed with ``CostModel.clock``/``cost`` evaluated at the edge entry time.
"""

from __future__ import annotations

import heapq
import math
from bisect import bisect_left, bisect_right
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
    """Label dominance (ASTAR-13): ``t1 <= t2 and g1 + wait(t1, t2) <= g2``.

    This is the reference definition. The search does not call it; it uses the equivalent per-slot key
    ``g - r*t`` (see :func:`_search_time_dependent`).
    """
    return t1 <= t2 and g1 + wait(t1, t2) <= g2 + _EPS * max(1.0, abs(g2))


_EPS = 1e-12  # relative float tolerance for cost comparisons


def _key_dominated(old_key: float, new_key: float) -> bool:
    """Same slot, old label not earlier: the new label (key ``new_key``) dominates the old one."""
    return old_key >= new_key - _EPS * max(1.0, abs(new_key))


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
    time and t0_s are multiples of ``wait_step_s`` and the label cap is not hit. ``max_labels`` trades exactness
    for a bounded label set per node: when a newcomer is dropped at the cap the result is flagged
    ``approximate=True`` (its cost may exceed the optimum). The cap bounds labels per node, not total runtime.
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
    if cost_model.last_change_s() <= 0.0:
        h = make_heuristic(graph, goal_list).tolist() if use_heuristic else [0.0] * graph.n_nodes
        return _search_static(graph, cost_model, origin, set(goal_list), t0_s, h)
    # time-varying model: the "cheapest slot" bound includes risk and is still admissible and consistent
    if use_heuristic:
        h = SlotHeuristic(graph, cost_model, goal_list)
    else:
        zeros = [0.0] * graph.n_nodes
        h = lambda s: zeros  # noqa: E731  (Dijkstra: no estimate)
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


class SlotHeuristic:
    """Time-aware admissible, consistent heuristic for the time-dependent search: ``h(s)[n]``.

    ``h(s)[n]`` = cheapest cost from n to the nearest goal on a static graph where edge e costs
    ``min over slots s' >= s of c_e(s')`` (edges blocked in all those slots are dropped; ``inf`` = no goal is
    reachable from n once the clock is in slot s).

    Why it is admissible: a label in slot s can only enter edges at slots >= s (the clock never runs backwards),
    every real entry pays ``c_e(t) >= min_{s' >= s} c_e(s')``, and waits cost >= 0.
    Why it is consistent: the minimum over a smaller set of slots is larger, so ``h(s)`` grows with s. For an edge
    u -> v entered in slot s and reached in slot s_v >= s, the Bellman equation gives
    ``h(s)[u] <= min_{s'>=s} c_e(s') + h(s)[v] <= c_e(t) + h(s_v)[v]``; a wait at u keeps the node, adds cost >= 0
    and can only raise h. So f = g + h never decreases along a path and A* still stops at the optimum.
    It includes the risk part of the cost, not just free-flow time: hazard that sits on an edge for the rest of
    the window is already charged (a storm that will have moved on is not). Each slot costs one reverse Dijkstra
    (scipy, in C, from all goals at once); slots are computed on first use and identical slots share one result.
    """

    def __init__(self, graph: RoadGraph, cost_model: CostModel, goals: Sequence[int]):
        cost = cost_model.cost_table()
        self._w = np.minimum.accumulate(cost[:, ::-1], axis=1)[:, ::-1]  # w[:, s] = min_{s' >= s} c_e(s')
        self._graph, self._goals = graph, list(goals)
        self._cache: dict[int, list[float]] = {}

    def __call__(self, s: int) -> list[float]:
        if s not in self._cache:
            nxt = s + 1
            if nxt < self._w.shape[1] and np.array_equal(self._w[:, s], self._w[:, nxt]):
                self._cache[s] = self(nxt)  # same edge bounds as the next slot: same h
            else:
                self._cache[s] = _reverse_dijkstra(self._graph, self._w[:, s], self._goals)
        return self._cache[s]


def _reverse_dijkstra(graph: RoadGraph, w: np.ndarray, goals: list[int]) -> list[float]:
    """Cheapest cost from every node to the nearest goal with per-edge weights ``w`` (inf = edge unusable)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra

    ok = np.isfinite(w)
    src, dst, w = graph.src[ok], graph.dst[ok], w[ok]
    # parallel edges: keep the cheapest (csr_matrix would add duplicates up)
    order = np.lexsort((w, src, dst))
    src, dst, w = src[order], dst[order], w[order]
    first = np.ones(len(w), bool)
    first[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
    n = graph.n_nodes
    reverse = csr_matrix((w[first], (dst[first], src[first])), shape=(n, n))  # entry [v, u] = edge u -> v
    return sp_dijkstra(reverse, directed=True, indices=goals, min_only=True).tolist()


def _search_time_dependent(
    graph, cost_model, origin, goals, t0_s, h, allow_wait, max_labels, step
) -> Route | None:
    """Multi-label time-dependent A* with waiting on a grid (DESIGN 4.3, ASTAR-04/13).

    Labels live in parallel lists (``lab_*``). The queue holds two kinds of items:
      * a *label* item: the label itself, departing now;
      * a *wait* item: "label L, but departing at a later grid time". Wait items are not stored as labels
        (their own label always dominates them through the waiting rule), but they are dropped when some
        *other* label at the node dominates them.

    Label sets ("frontiers"). Risk is piecewise constant per slot, so inside one slot s the wait rate at node n
    is a constant ``r = rate[n, s]`` and the dominance test ``g1 + W_n(t1, t2) <= g2`` (t1 <= t2, same slot)
    becomes ``g1 - r*t1 <= g2 - r*t2``: one number per label, the *key* ``g - r*t``. A frontier is the list of
    labels of one (node, slot) sorted by t; non-dominated labels have strictly decreasing keys, so
      * a new label is dominated iff the latest label with t_i <= t (it has the smallest key of those) has
        key <= its key;
      * the labels it dominates are the consecutive run starting at the first label with t_i >= t whose keys
        are >= its key (an equal-time label with a higher cost is part of that run).
    Both are a binary search plus a short scan, instead of a scan over every label at the node.
    Without node risk the rate is 1 in every slot, so one frontier per node is exact for the whole window.
    With node risk a label is also checked against the labels of earlier slots at the node with the exact
    ``wait_cost`` (dominance across a slot boundary); it never removes labels of other slots, which only
    keeps extra labels and so cannot lose the optimum. Without waiting only an equal-time label dominates.
    Every pruning step above is exact dominance, so the result is exact as long as the label cap (counted per
    node over all slots; often hit on large graphs, and it bounds labels, not runtime) is not hit.
    """
    clock_tab, cost_tab = cost_model.search_lists()
    indptr, dst = graph.indptr.tolist(), graph.dst.tolist()
    slot_s, last_slot = cost_model.slot_s, cost_model.n_slots - 1
    window_end = cost_model.window_end_s
    last_change = cost_model.last_change_s()  # waiting after this time can never pay
    wait_cost = cost_model.wait_cost
    rate_tab = cost_model.wait_rate_table()
    uniform_wait = rate_tab is None  # waiting costs 1 per second at every node and slot
    rates = None if uniform_wait else rate_tab.tolist()
    step_pays = None if uniform_wait else _wait_step_pays(graph, rate_tab)
    # nodes whose wait rate never changes: the key g - r*t is valid across all slots, so one frontier suffices
    rate_const = None if uniform_wait else (rate_tab == rate_tab[:, :1]).all(axis=1).tolist()

    lab_node: list[int] = []
    lab_t: list[float] = []
    lab_g: list[float] = []
    lab_parent: list[int] = []  # parent label id (-1 for the origin label)
    lab_edge: list[int] = []  # edge taken from the parent
    lab_wait: list[float] = []  # wait at the parent's node before that edge
    lab_alive: list[bool] = []
    n_at = [0] * graph.n_nodes  # live labels per node (for the cap)
    # frontier per (node, slot): [sorted t list, key list, label id list]; slot is always 0 when uniform_wait
    front: dict[tuple[int, int], list[list]] = {}
    same_t: dict[tuple[int, float], int] = {}  # allow_wait=False: best label per (node, exact time)
    approximate = False

    def slot_key(node: int, t: float) -> tuple[int, float]:
        """(frontier slot, wait rate) for a label at node, time t."""
        if uniform_wait:
            return 0, 1.0
        if rate_const[node]:
            return 0, rates[node][0]
        s = min(int(t // slot_s), last_slot)
        return s, rates[node][s]

    def dominated(node: int, t: float, g: float, skip: int) -> bool:
        """True if a live label at ``node`` other than ``skip`` dominates (t, g)."""
        tol = _EPS * max(1.0, abs(g))
        if not allow_wait:
            i = same_t.get((node, t))
            return i is not None and i != skip and lab_g[i] <= g + tol
        s, r = slot_key(node, t)
        f = front.get((node, s))
        if f is not None:
            ts, keys, ids = f
            j = bisect_right(ts, t) - 1  # latest label with t_i <= t: the smallest key among them
            if j >= 0 and ids[j] == skip:
                j -= 1  # ignore the label itself; keys fall with t, so ids[j-1] is the next-smallest key
            if j >= 0 and keys[j] <= g - r * t + tol:
                return True
        if not uniform_wait and not rate_const[node]:  # dominance from an earlier slot, exact piecewise wait
            for s2 in range(s):
                f2 = front.get((node, s2))
                if f2 is None:
                    continue
                for i in f2[2]:
                    if i != skip and lab_g[i] + wait_cost(node, lab_t[i], t) <= g + tol:
                        return True
        return False

    def add_label(node, t, g, parent, edge, wait) -> int | None:
        nonlocal approximate
        if dominated(node, t, g, -1):
            return None
        # Cap hit (approximation): keep the old labels, drop the newcomer and flag the result. A newcomer that
        # replaces labels it dominates does not grow the set, so it is always kept.
        if n_at[node] >= max_labels and not _would_remove_any(node, t, g):
            approximate = True
            return None
        new = len(lab_node)
        lab_node.append(node)
        lab_t.append(t)
        lab_g.append(g)
        lab_parent.append(parent)
        lab_edge.append(edge)
        lab_wait.append(wait)
        lab_alive.append(True)
        n_at[node] += 1
        if not allow_wait:
            old = same_t.get((node, t))
            if old is not None:
                lab_alive[old] = False
                n_at[node] -= 1
            same_t[(node, t)] = new
            return new
        s, r = slot_key(node, t)
        key = g - r * t
        f = front.setdefault((node, s), [[], [], []])
        ts, keys, ids = f
        j = bisect_left(ts, t)  # first label with t_i >= t, so an equal-time costlier label is removed too
        k = j
        while k < len(ts) and _key_dominated(keys[k], key):  # the run it dominates
            lab_alive[ids[k]] = False
            n_at[node] -= 1
            k += 1
        ts[j:k], keys[j:k], ids[j:k] = [t], [key], [new]
        return new

    def _would_remove_any(node, t, g) -> bool:
        if not allow_wait:
            return (node, t) in same_t
        s, r = slot_key(node, t)
        f = front.get((node, s))
        if f is None:
            return False
        j = bisect_left(f[0], t)  # same start as the removal run in add_label
        return j < len(f[0]) and _key_dominated(f[1][j], g - r * t)

    counter = 0  # FIFO tie-break among equal (f, -g)
    first = add_label(origin, t0_s, 0.0, -1, -1, 0.0)
    # heap item: (f, -g, counter, label id, departure time, cost so far, accumulated wait)
    h0 = h(min(int(t0_s // slot_s), last_slot))
    if h0[origin] == math.inf:
        return None
    heap = [(h0[origin], 0.0, counter, first, t0_s, 0.0, 0.0)]
    expansions = 0
    while heap:
        _, _, _, lab, t, g, waited = heapq.heappop(heap)
        if not lab_alive[lab]:
            continue
        u = lab_node[lab]
        if waited > 0.0 and dominated(u, t, g, lab):
            continue  # another label at u can reach this (time, cost) at least as cheaply
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
            h_v = h(min(int(t_v // slot_s), last_slot))[v]
            if h_v == math.inf:  # no goal reachable from v any more
                continue
            new = add_label(v, t_v, g_v, lab, e, waited)
            if new is not None:
                counter += 1
                heapq.heappush(heap, (g_v + h_v, -g_v, counter, new, t_v, g_v, 0.0))

        if allow_wait and t < last_change:  # wait at u until the next allowed departure time
            if uniform_wait or not step_pays[u][s]:
                # A wait that stays inside the current slot buys nothing here (see _wait_step_pays): the same
                # wait can be taken at the next node instead, at no more cost and the same clock. Only a wait
                # that crosses into the next slot can change an edge's cost, so jump straight to the first grid
                # time at or after that slot boundary.
                boundary = (s + 1) * slot_s
                t_next = t0_s + math.ceil((boundary - t0_s) / step - 1e-9) * step
                if t_next <= t:  # float rounding put t on the boundary itself: take one grid step instead
                    t_next = t0_s + (math.floor((t - t0_s) / step + 1e-9) + 1) * step
            else:  # waiting here may be cheaper than at the next node: try every grid step
                t_next = t0_s + (math.floor((t - t0_s) / step + 1e-9) + 1) * step
            h_u = h(min(int(t_next // slot_s), last_slot))[u] if t_next <= window_end + 1e-9 else math.inf
            if h_u < math.inf:  # (inf: the goal is out of reach from u once we have waited)
                g_next = g + wait_cost(u, t, t_next)
                counter += 1
                heapq.heappush(
                    heap, (g_next + h_u, -g_next, counter, lab, t_next, g_next, waited + (t_next - t))
                )
    return None


def _wait_step_pays(graph: RoadGraph, rate: np.ndarray) -> list[list[bool]]:
    """``pays[u][s]``: can a wait at node u that starts and ends inside slot s be part of an optimal plan?

    Exchange argument. Take a plan that waits w seconds at u inside slot s and then enters edge e = u -> v.
    Entering e right away instead costs the same and takes the same clock (both entry times are in slot s), and
    then waiting w seconds at v brings us to v at exactly the same time as before. That wait at v costs at most
    ``w * max_{s' >= s} rate[v, s']``, so the swap is never worse when that maximum is <= ``rate[u, s]``.
    If this holds for every out-edge of u, an in-slot wait at u can always be pushed forward, so the search
    only needs the wait that reaches the next slot boundary (False). Nodes next to a riskier node keep
    step-by-step waiting (True). (Same grid assumption as the exactness claim of ``astar``: clock times are
    multiples of the wait step, so the pushed wait at v also ends on the grid.)
    """
    suffix_max = np.maximum.accumulate(rate[:, ::-1], axis=1)[:, ::-1]  # max over slots s' >= s
    n, n_slots = rate.shape
    best_next = np.zeros((n, n_slots))  # max over out-neighbours v of suffix_max[v]
    np.maximum.at(best_next, graph.src, suffix_max[graph.dst])
    return (rate < best_next).tolist()


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
