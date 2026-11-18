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
from outrun.eval.metrics import RouteTrace
from outrun.roads.graph import RoadGraph
from outrun.routing.astar import Route
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
    raise NotImplementedError


def route_exposure(route: Route, cost_model: CostModel) -> float:
    """Integrated exposure in risk-minutes (PLAN-05, ASTAR-20): the single EVAL-09 definition,
    ``score_route(route_trace(route), cost_model.combined_risk_table(), cost_model.blocked_table(),
    slot_s=cost_model.slot_s, node_risk=cost_model.combined_node_risk_table(max(route.nodes) + 1))
    .exposure_risk_min`` (scenario clock; waits charged at the node waited at)."""
    raise NotImplementedError


def replay_trace(route: Route, cost_model: CostModel) -> RouteTrace:
    """PLAN-03. The re-driven trace of ``route`` under ``cost_model``'s clock (scenario clock,
    ``t_start_s = route.start_s``, ``wait_nodes = route.nodes[:-1]``)."""
    raise NotImplementedError


def replay(route: Route, cost_model: CostModel) -> ReplayScore:
    """Re-drive ``route``'s edge sequence and waits under ``cost_model`` (PLAN-03).

    Starts at ``route.start_s``; at node i waits ``route.waits_s[i]``, then enters edges[i] and advances the clock
    by ``cost_model.clock``. Blocked edges are still traversed (counted, cost uses the clock time times the risk
    multiplier, never inf). Same model as planning -> travel_s/cost equal the route's. ``exposure_risk_min`` and
    ``intersects_hazard`` come from ``score_route`` on ``replay_trace`` and the model's combined/blocked tables.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def plan_shortest_time(
    graph: RoadGraph, cost_model: CostModel, origin: int, goals: int | Sequence[int], t0_s: float = 0.0
) -> Route | None:
    """Planner (a) (PLAN-02): A* on ``cost_model.with_risk(zeros)`` -- risk, blocking and exposure ignored."""
    raise NotImplementedError


def plan_avoid_current_only(
    graph: RoadGraph, cost_model: CostModel, origin: int, goals: int | Sequence[int], t0_s: float = 0.0
) -> Route | None:
    """Planner (a') (PLAN-02): the one current-only freeze, at the CostModel level: edge and node risk frozen at
    ``cost_model.slot(t0_s)`` for the whole window (``transform_risk(..., CURRENT_ONLY, t0_slot)``)."""
    raise NotImplementedError
