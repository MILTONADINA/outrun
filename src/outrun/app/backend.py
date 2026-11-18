"""Live computation behind the app (spec 14: APP-03, APP-05, APP-06, APP-07).

The server talks only to the ``AppBackend`` protocol, so tests inject a fake and the offline demo can run with no
backend at all (cached results, APP-11). ``DefaultBackend`` wires the other areas (routing/planner.py,
routing/emergency.py, optimize/evacuation.py, roads/graph.py) and imports them lazily inside its methods, so
importing this module never pulls in network clients (APP-10).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from outrun.app.models import DispatchRequest, RouteRequest

if TYPE_CHECKING:
    from outrun.roads.graph import RoadGraph
    from outrun.routing.astar import Route
    from outrun.routing.cost import CostModel
    from outrun.routing.planner import PlanResult

EvacEvent = tuple[str, dict[str, Any]]  # ("progress", {...}) ... ("done", result)


def route_fields(
    plan: PlanResult,
    shortest: Route | None,
    planning_model: CostModel,
    observed_model: CostModel | None,
    graph: RoadGraph,
) -> dict[str, Any]:
    """APP-03 keys except ``expansions``: ``outrun`` (LineString Feature or None when ``plan.route`` is None),
    ``shortest``, ``eta_min``, ``exposure`` {"predicted", "observed"} (ASTAR-20), ``blocked_avoided`` =
    ``replay(shortest, planning_model).blocked_edges`` (0 without shortest), ``shelter_in_place`` =
    ``plan.kind in {"shelter_in_place", "hardened_facility"}``. Routing modules are imported lazily (APP-10)."""
    raise NotImplementedError


@runtime_checkable
class AppBackend(Protocol):
    """What the server needs from the live planners. All results are JSON-serialisable dicts."""

    def route(self, req: RouteRequest) -> dict[str, Any]:
        """Safe route vs shortest (keys listed in APP-03)."""
        ...

    def dispatch(self, req: DispatchRequest) -> dict[str, Any]:
        """Hungarian dispatch + inbound routes + baseline (keys listed in APP-06)."""
        ...

    def run_evac(self, scenario_id: str) -> Iterator[EvacEvent]:
        """Live SA run: ``("progress", {"iter", "objective", "temperature"})`` pairs, last ``("done", result)``."""
        ...

    def city(self, name: str, allow_network: bool) -> dict[str, Any]:
        """Build or load a city graph; ``LookupError`` if not cached and network disallowed, or not found."""
        ...


class DefaultBackend:
    """Real backend over a scenarios folder (T6 skeleton, T23 full). Methods implement ``AppBackend``."""

    def __init__(self, scenarios_dir: Path) -> None:
        self.scenarios_dir = Path(scenarios_dir)

    def route(self, req: RouteRequest) -> dict[str, Any]:
        raise NotImplementedError

    def dispatch(self, req: DispatchRequest) -> dict[str, Any]:
        raise NotImplementedError

    def run_evac(self, scenario_id: str) -> Iterator[EvacEvent]:
        """Progress events straight from ``simulated_annealing``'s ``callback(iteration, current_J, best_J,
        temperature)`` (EVAC-06): ``{"iter": iteration, "objective": current_J, "temperature": temperature}``."""
        raise NotImplementedError

    def city(self, name: str, allow_network: bool) -> dict[str, Any]:
        raise NotImplementedError
