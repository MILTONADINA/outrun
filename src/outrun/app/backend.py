"""Live computation behind the app (spec 14: APP-03, APP-05, APP-06, APP-07).

The server talks only to the ``AppBackend`` protocol, so tests inject a fake and the offline demo can run with no
backend at all (cached results, APP-11). ``DefaultBackend`` wires the other areas (routing/planner.py,
routing/emergency.py, optimize/evacuation.py, roads/graph.py) and imports them lazily inside its methods, so
importing this module never pulls in network clients (APP-10).
"""

from __future__ import annotations

import json
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
    from outrun.routing.planner import replay, route_exposure

    def line(route: Route | None, properties: dict[str, Any]) -> dict[str, Any] | None:
        if route is None:
            return None
        coords = [[float(graph.lon[n]), float(graph.lat[n])] for n in route.nodes]
        return {"type": "Feature", "properties": properties, "geometry": {"type": "LineString", "coordinates": coords}}

    # the UI panel compares the two routes, so the shortest route carries its own minutes and exposure
    shortest_properties: dict[str, Any] = {"kind": "shortest"}
    if shortest is not None:
        shortest_properties["eta_min"] = shortest.travel_s / 60.0
        shortest_properties["exposure"] = float(route_exposure(shortest, planning_model))
    route = plan.route
    exposure: dict[str, Any] = {"predicted": None, "observed": None}
    if route is not None:
        exposure["predicted"] = float(route_exposure(route, planning_model))
        if observed_model is not None:
            exposure["observed"] = float(replay(route, observed_model).exposure_risk_min)
    return {
        "outrun": line(route, {"kind": plan.kind, "facility": plan.facility}),
        "shortest": line(shortest, shortest_properties),
        "eta_min": None if route is None else route.travel_s / 60.0,
        "exposure": exposure,
        "blocked_avoided": 0 if shortest is None else replay(shortest, planning_model).blocked_edges,
        "shelter_in_place": plan.kind in {"shelter_in_place", "hardened_facility"},
    }


def search_effort(graph: RoadGraph, model: CostModel, origin: int, goals: list[int], t0_s: float) -> dict[str, int]:
    """APP-03 ``expansions``: nodes plain Dijkstra and A* would settle before reaching the nearest goal.

    A full time-dependent Dijkstra over the whole city takes over a minute, so both numbers are counted on the
    edge costs in force at the departure slot, from one fast scipy Dijkstra run that stops at the goal's cost
    ``C``. Dijkstra settles every node with distance ``d <= C``. A* with a consistent heuristic ``h`` settles
    exactly the nodes with ``d + h < C``, so the second count is just a filter on the first.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    from outrun.routing.cost import make_heuristic

    cost = model.cost_table()[:, model.slot(t0_s)]  # inf where the edge is blocked
    open_edges = np.isfinite(cost)
    roads = csr_matrix((cost[open_edges], (graph.src[open_edges], graph.dst[open_edges])), shape=(graph.n_nodes,) * 2)
    dist = dijkstra(roads, indices=origin)
    best = dist[goals].min()
    if not np.isfinite(best):
        return {"astar": 0, "dijkstra": 0}
    h = make_heuristic(graph, goals)
    return {"astar": int(np.sum(dist + h < best)), "dijkstra": int(np.sum(dist <= best))}


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
    """Real backend over a scenarios folder (T6 skeleton, T23 full). Methods implement ``AppBackend``.

    ``route`` and ``city`` are live. ``dispatch`` and ``run_evac`` raise ``NotImplementedError`` until T21/T19
    are wired in T23; the server then falls back to cached results (APP-05) or answers 503 (APP-06).
    """

    def __init__(self, scenarios_dir: Path) -> None:
        self.scenarios_dir = Path(scenarios_dir)
        self._loaded: dict[tuple[str, str], tuple[Any, Any, Any, Any]] = {}  # (scenario, stack file) -> arrays
        self._hardened: dict[str, list[int]] = {}  # scenario -> hospital/school graph nodes (PLAN-04)
        self._models: dict[tuple, Any] = {}  # built CostModels, so a second click skips the ~1 s table build

    # ---- loading (cached per scenario so a click does not reload the graph)
    def _load(self, scenario_id: str, stack_name: str) -> tuple[Any, Any, Any, Any]:
        """(graph, stack, edge_risk, node_risk) for ``risk<stack_name>.zarr`` of a scenario."""
        key = (scenario_id, stack_name)
        if key not in self._loaded:
            from outrun.core.risk import RiskStack
            from outrun.roads.graph import load_npz
            from outrun.routing.cost import EdgeRiskSampler

            folder = self.scenarios_dir / scenario_id
            graph = load_npz(folder / "graph.npz")
            stack = RiskStack.from_zarr(folder / f"risk{stack_name}.zarr")
            sampler = EdgeRiskSampler.from_window(graph, stack.window)
            self._loaded[key] = (graph, stack, sampler.sample_stack(stack), sampler.sample_stack_nodes(stack))
        return self._loaded[key]

    def _cost_model(self, key: tuple, graph: Any, edge_risk: Any, node_risk: Any, profile: Any, lam: float) -> Any:
        """CostModel for ``key`` (scenario, stack, profile, lam, ...), built once and kept (at most 4)."""
        if key not in self._models:
            from outrun.core.policy import load_policies
            from outrun.routing.cost import CostModel

            if len(self._models) >= 4:
                self._models.clear()
            self._models[key] = CostModel.from_policies(
                graph.free_flow_s, edge_risk, load_policies(), profile,
                node_risk=node_risk, lam=lam, crossing=graph.crossing,
            )  # fmt: skip
        return self._models[key]

    @staticmethod
    def _nearest_node(graph: Any, lon: float, lat: float) -> int:
        """Closest graph node (flat-earth distance is fine inside one city)."""
        import numpy as np

        dx = (graph.lon - lon) * np.cos(np.radians(lat))
        return int(np.argmin(dx * dx + (graph.lat - lat) ** 2))

    def _shelter_nodes(self, graph: Any, scenario_id: str) -> list[int]:
        """Graph nodes of the shelters listed in ``results/shelters_mcf.json`` (Point features)."""
        path = self.scenarios_dir / scenario_id / "results" / "shelters_mcf.json"
        if not path.is_file():
            raise LookupError("no shelters file for this scenario")
        points = [f["geometry"]["coordinates"] for f in json.loads(path.read_text())["features"]
                  if f.get("geometry", {}).get("type") == "Point"]  # fmt: skip
        if not points:
            raise LookupError("scenario has no shelters")
        return sorted({self._nearest_node(graph, lon, lat) for lon, lat in points})

    def _hardened_nodes(self, graph: Any, scenario_id: str, origin: int, keep: int = 8) -> list[int]:
        """Hospital and school nodes for the shelter-in-place rule (PLAN-04): the ``keep`` nearest to ``origin`` that
        a 5 min drive can reach at crow-fly top speed (the rule runs one A* per candidate). Empty if the HIFLD data
        is absent."""
        import numpy as np

        from outrun.routing.cost import crowfly_speed_mps
        from outrun.routing.planner import HARDENED_MAX_DRIVE_S

        if scenario_id not in self._hardened:
            try:
                from outrun.config import get_paths
                from outrun.data.facilities import load_hifld_facilities
                from outrun.routing.okc_ems import snap_nodes

                bbox = tuple(json.loads((self.scenarios_dir / scenario_id / "scenario.json").read_text())["bbox"])
                df = load_hifld_facilities(get_paths().raw, bbox)
                df = df[df["kind"].isin(["hospital", "school"])]
                nodes = snap_nodes(graph, df[["lon", "lat"]].to_numpy(dtype=float)) if len(df) else np.array([], int)
                self._hardened[scenario_id] = sorted({int(n) for n in nodes})
            except (OSError, LookupError, KeyError):  # no facility files: the rule falls back to plain shelter-in-place
                self._hardened[scenario_id] = []
        nodes = np.asarray(self._hardened[scenario_id], dtype=np.int64)
        if not len(nodes):
            return []
        d = np.hypot(graph.x_m[nodes] - graph.x_m[origin], graph.y_m[nodes] - graph.y_m[origin])
        near = np.argsort(d)[:keep]
        return [int(nodes[k]) for k in near if d[k] <= HARDENED_MAX_DRIVE_S * crowfly_speed_mps(graph)]

    # ---- AppBackend
    def route(self, req: RouteRequest) -> dict[str, Any]:
        from outrun.config import load_config
        from outrun.core.policy import Profile
        from outrun.core.risk import RiskSource
        from outrun.routing.planner import plan_route, plan_shortest_time, transform_risk

        source = RiskSource(req.risk_source)
        stack_name = "" if source in (RiskSource.MODEL, RiskSource.NONE, RiskSource.CURRENT_ONLY) else f"_{source.value}"
        graph, stack, edge_risk, node_risk = self._load(req.scenario, stack_name)
        profile = Profile(req.profile)
        lam = req.lam if req.lam is not None else float(load_config("hazards")["lambda_default"])
        t0_s = 0.0 if req.t0 is None else max(0.0, (req.t0 - stack.t0).total_seconds())
        t0_slot = int(t0_s // 900.0)

        # planning model: NONE zeroes the risk, CURRENT_ONLY freezes it at the start slot (PLAN-02)
        planning_edge = transform_risk(edge_risk, source, t0_slot)
        planning_node = transform_risk(node_risk, source, t0_slot)
        planning = self._cost_model(
            (req.scenario, source, profile, lam, t0_slot), graph, planning_edge, planning_node, profile, lam
        )
        observed = None  # scored against what really happened, if the bundle has it
        if (self.scenarios_dir / req.scenario / "risk_observed.zarr").exists():
            _, _, obs_edge, obs_node = self._load(req.scenario, "_observed")
            observed = self._cost_model((req.scenario, "observed", profile, lam), graph, obs_edge, obs_node, profile, lam)

        origin = self._nearest_node(graph, *req.origin)
        if req.dest == "nearest_shelter":
            goals = self._shelter_nodes(graph, req.scenario)
        else:
            goals = [self._nearest_node(graph, *req.dest)]

        hardened = self._hardened_nodes(graph, req.scenario, origin)
        plan = plan_route(graph, planning, origin, goals, t0_s, hardened=hardened)
        shortest = plan_shortest_time(graph, planning, origin, goals, t0_s)
        out = route_fields(plan, shortest, planning, observed, graph)
        out["expansions"] = search_effort(graph, planning, origin, goals, t0_s)
        return out

    def dispatch(self, req: DispatchRequest) -> dict[str, Any]:
        """Hungarian dispatch from the HIFLD fire/EMS stations of the scenario bbox (EV profile, model risk), the
        hospital leg and the hazard-blind baseline (APP-06). ``LookupError`` -> the server answers 503."""
        import numpy as np

        from outrun.core.policy import Profile
        from outrun.routing.okc_ems import load_facility_points, plan_dispatch

        folder = self.scenarios_dir / req.scenario
        graph, stack, edge_risk, node_risk = self._load(req.scenario, "")
        bbox = tuple(json.loads((folder / "scenario.json").read_text())["bbox"])
        stations, hospitals = load_facility_points(bbox)
        t0_s = 0.0 if req.t0 is None else max(0.0, (req.t0 - stack.t0).total_seconds())
        model = self._cost_model((req.scenario, "", "ev", 1.0), graph, edge_risk, node_risk, Profile.EV, 1.0)
        return plan_dispatch(graph, model, stations, hospitals, np.asarray(req.incidents, dtype=float), t0_s)

    def run_evac(self, scenario_id: str) -> Iterator[EvacEvent]:
        """Progress events straight from ``simulated_annealing``'s ``callback(iteration, current_J, best_J,
        temperature)`` (EVAC-06): ``{"iter": iteration, "objective": current_J, "temperature": temperature}``.

        Needs the instance built by ``python -m outrun.optimize.okc_evacuation`` (``LookupError`` if absent, so the
        server falls back to the cached run). SA runs in a worker thread; the last event is ``("done", result)``."""
        import pickle
        import queue
        import threading

        from outrun.optimize.evacuation import EvacEvaluator, evac_metrics, simulated_annealing
        from outrun.optimize.okc_evacuation import cache_path, result_geojson

        path = cache_path(scenario_id)
        if not path.is_file():
            raise LookupError("no prepared evacuation instance for this scenario")
        oe = pickle.loads(path.read_bytes())
        ev = EvacEvaluator(oe.instance)
        events: queue.Queue = queue.Queue()

        def progress(iteration: int, current_j: float, _best_j: float, temperature: float) -> None:
            events.put(("progress", {"iter": iteration, "objective": current_j, "temperature": temperature}))

        def work() -> None:
            try:
                res = simulated_annealing(
                    ev, iterations=8000, zone_adjacency=oe.zone_adjacency, callback=progress, callback_every=100
                )
                events.put(("done", result_geojson(oe, res.best_plan, evac_metrics(ev, res.best_plan))))
            except Exception as exc:  # reported to the client as an SSE error event
                events.put(("error", {"detail": str(exc)}))

        threading.Thread(target=work, daemon=True).start()
        while True:
            kind, data = events.get()
            if kind == "error":
                raise RuntimeError(data["detail"])
            yield kind, data
            if kind == "done":
                return

    def city(self, name: str, allow_network: bool) -> dict[str, Any]:
        """Load the cached drive graph of ``name``; build it from OpenStreetMap only if the network is allowed."""
        from outrun.config import get_paths
        from outrun.roads.graph import build_graph, cache_key

        cache_dir = get_paths().proc / "graphs"
        cached = (cache_dir / f"{cache_key(place=name)}.npz").exists()
        if not cached and not allow_network:
            raise LookupError(f"{name!r} is not cached and the network is not allowed")
        try:
            graph = build_graph(place=name, cache_dir=cache_dir)
        except Exception as exc:  # osmnx raises many types for an unknown place or no connection
            raise LookupError(f"could not build a road graph for {name!r}: {exc}") from exc
        return {
            "name": name,
            "bbox": [float(graph.lon.min()), float(graph.lat.min()), float(graph.lon.max()), float(graph.lat.max())],
            "n_nodes": graph.n_nodes,
            "n_edges": graph.n_edges,
            "cached": cached,
        }
