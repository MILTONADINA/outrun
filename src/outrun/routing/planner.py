"""Safe-route API, baselines, replay scoring, shelter-in-place rule (spec 04-astar, PLAN-01..PLAN-05).

Planners (DESIGN 6.2): (a) shortest time (risk ignored), (a') avoid-current-only (risk frozen at t0),
(b)/(c)/(d) forecast-style planning on baseline/model/observed risk -- all the same A*, different risk arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from outrun.core.risk import RiskSource  # defined once in core (RISK-08); re-exported for callers
from outrun.eval.metrics import RouteTrace, route_trace, score_route
from outrun.roads.graph import RoadGraph
from outrun.routing.astar import Route, astar
from outrun.routing.cost import CostModel

#: Hazard indices (Hazard declaration order) that trigger the shelter-in-place rule: TORNADO, WILDFIRE.
SIP_HAZARDS: tuple[int, ...] = (1, 3)
SIP_DEFAULT_THRESHOLD = 0.5  # used when the profile never blocks on a SIP hazard
HARDENED_MAX_DRIVE_S = 300.0  # configs/planning.yaml astar.shelter_in_place_hardened_drive_min = 5


@dataclass(frozen=True)
class PlanResult:
    """Outcome of :func:`plan_route` (PLAN-01, PLAN-05)."""

    kind: Literal["route", "shelter_in_place", "hardened_facility", "no_route"]
    route: Route | None  # evacuation route, or route to the hardened facility; None otherwise
    facility: int | None = None  # hardened facility node for kind == "hardened_facility"
    hazard_arrival_s: float | None = None  # when a SIP hazard reaches the origin (None: never in window)
    footprint_exit_s: float | None = None  # when the planned route leaves the SIP footprint
    exposure_risk_min: float = 0.0  # planned exposure of ``route`` (see route_exposure)
    reason: str = ""


@dataclass(frozen=True)
class ReplayScore:
    """A route replayed on another risk field (PLAN-03): plan on forecast, score on observed."""

    travel_s: float
    cost: float
    exposure_risk_min: float
    blocked_edges: int  # edges entered while blocked in the scoring model
    intersects_hazard: bool  # score_route(...).intersects on the scoring model's tables (ASTAR-20)


def transform_risk(risk: np.ndarray, source: RiskSource, t0_slot: int = 0) -> np.ndarray:
    """Risk array (E or N, H, S) for a planner (PLAN-02). Never modifies the input.

    NONE -> zeros (plain shortest time, planner (a)); CURRENT_ONLY -> every slot equals slot ``t0_slot``
    (planner (a'));
    MODEL/BASELINE/OBSERVED -> an unchanged copy (the caller passes the matching RiskStack's samples).
    """
    if source == RiskSource.NONE:
        return np.zeros_like(risk)
    if source == RiskSource.CURRENT_ONLY:
        return np.repeat(risk[:, :, t0_slot : t0_slot + 1], risk.shape[2], axis=2)
    return risk.copy()


def _score(route: Route, trace: RouteTrace, cost_model: CostModel):
    """EVAL-09 score of ``trace`` on the model's combined tables (the one exposure definition)."""
    n_nodes = None if cost_model.has_node_risk() else max(route.nodes) + 1  # only needed for the zero table
    return score_route(
        trace,
        cost_model.combined_risk_table(),
        cost_model.blocked_table(),
        slot_s=cost_model.slot_s,
        node_risk=cost_model.combined_node_risk_table(n_nodes),
    )


def route_exposure(route: Route, cost_model: CostModel) -> float:
    """Integrated exposure in risk-minutes (PLAN-05, ASTAR-20): the single EVAL-09 definition,
    ``score_route(route_trace(route), cost_model.combined_risk_table(), cost_model.blocked_table(),
    slot_s=cost_model.slot_s, node_risk=cost_model.combined_node_risk_table(max(route.nodes) + 1))
    .exposure_risk_min`` (scenario clock; waits charged at the node waited at)."""
    return _score(route, route_trace(route), cost_model).exposure_risk_min


def _drive(route: Route, cost_model: CostModel):
    """Re-drive the route's waits and edges under ``cost_model``.

    Returns (enter_times, exit_times, total_cost, blocked_edge_count)."""
    # A blocked edge has cost inf in the model's table, so its cost is rebuilt as clock * (1 + lam * exposure).
    # (lam and the risk array are read from the model's private fields: CostModel has no public getter.)
    lam = cost_model.lam
    weighted = cost_model.weighted_risk_table()
    t, total, blocked = route.start_s, 0.0, 0
    enter, leave = [], []
    for i, e in enumerate(route.edges):
        wait = route.waits_s[i]
        total += cost_model.wait_cost(route.nodes[i], t, t + wait)
        t += wait
        enter.append(t)
        clock = cost_model.clock(e, t)
        if cost_model.blocked(e, t):
            blocked += 1
            total += clock * (1.0 + lam * weighted[e, cost_model.slot(t)])
        else:
            total += cost_model.cost(e, t)
        t += clock
        leave.append(t)
    return enter, leave, total, blocked


def replay_trace(route: Route, cost_model: CostModel) -> RouteTrace:
    """PLAN-03. The re-driven trace of ``route`` under ``cost_model``'s clock (scenario clock,
    ``t_start_s = route.start_s``, ``wait_nodes = route.nodes[:-1]``)."""
    enter, leave, _, _ = _drive(route, cost_model)
    return RouteTrace(
        edges=tuple(route.edges),
        t_enter_s=tuple(enter),
        t_exit_s=tuple(leave),
        t_start_s=route.start_s,
        wait_nodes=tuple(route.nodes[:-1]),
    )


def replay(route: Route, cost_model: CostModel) -> ReplayScore:
    """Re-drive ``route``'s edge sequence and waits under ``cost_model`` (PLAN-03).

    Starts at ``route.start_s``; at node i waits ``route.waits_s[i]``, then enters edges[i] and advances the clock
    by ``cost_model.clock``. Blocked edges are still traversed (counted, cost uses the clock time times the risk
    multiplier, never inf). Same model as planning -> travel_s/cost equal the route's. ``exposure_risk_min`` and
    ``intersects_hazard`` come from ``score_route`` on ``replay_trace`` and the model's combined/blocked tables.
    """
    enter, leave, cost, blocked = _drive(route, cost_model)
    trace = RouteTrace(tuple(route.edges), tuple(enter), tuple(leave), route.start_s, tuple(route.nodes[:-1]))
    score = _score(route, trace, cost_model)
    travel_s = (leave[-1] if leave else route.start_s) - route.start_s
    return ReplayScore(travel_s, cost, score.exposure_risk_min, blocked, bool(score.intersects))


def _sip_thresholds(cost_model: CostModel, sip_hazards: Sequence[int]) -> dict[int, float]:
    """Block threshold per shelter-in-place hazard (0.5 when the profile never blocks on it); H-range only."""
    block_at = cost_model.block_at
    thresholds = {}
    for h in sip_hazards:
        if h < cost_model.n_hazards:
            thresholds[h] = SIP_DEFAULT_THRESHOLD if np.isnan(block_at[h]) else float(block_at[h])
    return thresholds


def _node_arrival_times(route: Route) -> list[float]:
    """Time the vehicle is at each route node: the origin at its departure, others when it arrives."""
    if not route.edges:
        return [route.start_s]
    times = [route.depart_s[0]]
    times += [route.depart_s[i] - route.waits_s[i] for i in range(1, len(route.edges))]
    times.append(route.arrive_s)
    return times


def plan_route(
    graph: RoadGraph,
    cost_model: CostModel,
    origin: int,
    goals: int | Sequence[int],
    t0_s: float = 0.0,
    *,
    hardened: Sequence[int] = (),
    hardened_max_drive_s: float = HARDENED_MAX_DRIVE_S,
    sip_hazards: Sequence[int] = SIP_HAZARDS,
) -> PlanResult:
    """Safe route with the shelter-in-place rule (PLAN-01, PLAN-04).

    1. best = astar(graph, cost_model, origin, goals, t0_s).
    2. SIP threshold per hazard h in sip_hazards: ``block_at_h`` of the cost model, or 0.5 if NaN.
       Footprint = nodes whose node_risk for some SIP hazard reaches its threshold in any slot.
       hazard_arrival_s = start time of the earliest slot >= slot(t0_s) in which the origin's node_risk reaches
       a threshold (None if never). Hazard indices >= H are ignored.
    3. If hazard_arrival_s is None: kind "route" (or "no_route" if best is None).
    4. Else footprint_exit_s = the arrival time at the first node of best (origin included, at its departure
       time) that is outside the footprint (inf if none or best is None). If footprint_exit_s < hazard_arrival_s:
       kind "route". Otherwise: among ``hardened`` nodes, A*-route to each; choose the smallest travel_s with
       travel_s <= hardened_max_drive_s and arrive_s <= hazard_arrival_s (ties: lowest node id) -> kind
       "hardened_facility"; if none qualifies -> kind "shelter_in_place" with route None.
    """
    best = astar(graph, cost_model, origin, goals, t0_s)

    # which (node, slot) pairs reach a shelter-in-place threshold
    thresholds = _sip_thresholds(cost_model, sip_hazards)
    node_risk = cost_model.node_risk
    hit = None  # (N, S) bool
    if node_risk is not None and thresholds:
        hit = np.zeros(node_risk.shape[::2], dtype=bool)
        for h, threshold in thresholds.items():
            hit |= node_risk[:, h, :] >= threshold

    first_slot = cost_model.slot(t0_s)
    hazard_arrival_s = None
    if hit is not None:
        slots = np.flatnonzero(hit[origin, first_slot:])
        if len(slots):
            hazard_arrival_s = float((first_slot + slots[0]) * cost_model.slot_s)

    if hazard_arrival_s is None:
        if best is None:
            return PlanResult("no_route", None, reason="no goal is reachable")
        return PlanResult("route", best, exposure_risk_min=route_exposure(best, cost_model))

    footprint = hit.any(axis=1)
    footprint_exit_s = float("inf")
    if best is not None:
        for node, t in zip(best.nodes, _node_arrival_times(best), strict=True):
            if not footprint[node]:
                footprint_exit_s = float(t)
                break
    if best is not None and footprint_exit_s < hazard_arrival_s:
        return PlanResult(
            "route", best, hazard_arrival_s=hazard_arrival_s, footprint_exit_s=footprint_exit_s,
            exposure_risk_min=route_exposure(best, cost_model),
        )  # fmt: skip

    # the hazard reaches the origin before we can leave its footprint: nearest hardened facility, if close enough
    candidates = []
    for facility in sorted(set(hardened)):
        r = astar(graph, cost_model, origin, facility, t0_s)
        if r is not None and r.travel_s <= hardened_max_drive_s and r.arrive_s <= hazard_arrival_s:
            candidates.append((r.travel_s, facility, r))
    if candidates:
        _, facility, r = min(candidates, key=lambda c: (c[0], c[1]))
        return PlanResult(
            "hardened_facility", r, facility=facility, hazard_arrival_s=hazard_arrival_s,
            footprint_exit_s=footprint_exit_s, exposure_risk_min=route_exposure(r, cost_model),
            reason="hazard reaches the origin before the evacuation route leaves its footprint",
        )  # fmt: skip
    return PlanResult(
        "shelter_in_place", None, hazard_arrival_s=hazard_arrival_s, footprint_exit_s=footprint_exit_s,
        reason="no hardened facility is reachable in time",
    )  # fmt: skip


def plan_shortest_time(
    graph: RoadGraph, cost_model: CostModel, origin: int, goals: int | Sequence[int], t0_s: float = 0.0
) -> Route | None:
    """Planner (a) (PLAN-02): A* on ``cost_model.with_risk(zeros)`` -- risk, blocking and exposure ignored."""
    zeros = transform_risk(cost_model.risk, RiskSource.NONE)
    return astar(graph, cost_model.with_risk(zeros), origin, goals, t0_s)


def plan_avoid_current_only(
    graph: RoadGraph, cost_model: CostModel, origin: int, goals: int | Sequence[int], t0_s: float = 0.0
) -> Route | None:
    """Planner (a') (PLAN-02): the one current-only freeze, at the CostModel level: edge and node risk frozen at
    ``cost_model.slot(t0_s)`` for the whole window (``transform_risk(..., CURRENT_ONLY, t0_slot)``)."""
    t0_slot = cost_model.slot(t0_s)
    edge_risk = transform_risk(cost_model.risk, RiskSource.CURRENT_ONLY, t0_slot)
    node_risk = cost_model.node_risk
    if node_risk is not None:
        node_risk = transform_risk(node_risk, RiskSource.CURRENT_ONLY, t0_slot)
    return astar(graph, cost_model.with_risk(edge_risk, node_risk), origin, goals, t0_s)
