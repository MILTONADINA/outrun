"""Capacitated shelter assignment: OR-Tools min-cost flow plus nearest / greedy baselines
(spec 05-shelters.md, DESIGN 4.4).

Costs are seconds (``inf`` = unreachable); people counts and flows are int64. Parameters defaulting to ``None`` are
read from ``configs/planning.yaml`` (``demand.*``, ``shelters.*``). OR-Tools is imported lazily inside
``solve_min_cost_flow``.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

    from outrun.core.policy import Profile
    from outrun.core.risk import Hazard

FacilityKind = Literal[
    "shelter", "school", "fire_station", "hospital"
]  # == data.facilities.FACILITY_KINDS (DATA-19)
_KINDS = ("shelter", "school", "fire_station", "hospital")
_DEFAULT_SLOT_S = 900.0


def _cfg(section: str, key: str) -> float:
    from outrun.config import load_config

    return load_config("planning")[section][key]


@dataclass(frozen=True)
class Facility:
    """A candidate facility snapped to the road graph (``x, y`` in metres on GRID)."""

    id: str
    kind: FacilityKind
    node: int
    x: float
    y: float
    capacity: int | None

    @classmethod
    def from_record(cls, record: Mapping[str, object], node: int, x: float, y: float) -> Facility:
        """SHEL-02. From a DATA-19 row (``facility_id``, ``kind``, ``capacity``): NaN/missing capacity -> None, finite
        -> int; a kind outside FACILITY_KINDS -> ValueError."""
        kind = record["kind"]
        if kind not in _KINDS:
            raise ValueError(f"unknown facility kind {kind!r}; expected one of {_KINDS}")
        cap = record.get("capacity")
        capacity = None if cap is None or not np.isfinite(float(cap)) else int(cap)  # type: ignore[arg-type]
        return cls(str(record["facility_id"]), kind, node, x, y, capacity)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ShelterSite:
    """A shelter destination with a capacity; ``assumed`` marks fallback sites with default capacity (UI flag)."""

    id: str
    kind: FacilityKind
    node: int
    x: float
    y: float
    capacity: int
    assumed: bool


@dataclass(frozen=True)
class ShelterAssignment:
    """Integer flow zone -> site and unserved people per zone. ``method`` in {"mcf", "nearest", "greedy"}."""

    method: str
    flow: np.ndarray  # (Z, S) int64
    unserved: np.ndarray  # (Z,) int64


@dataclass(frozen=True)
class ShelterMetrics:
    """SHEL-07 metrics of one assignment."""

    served_persons: int
    unserved_persons: int
    total_person_minutes: float
    max_minutes: float
    overflow_persons: int
    overflow_pct: float
    mean_exposure: float


def shelter_demand(population: np.ndarray, rate: float | None = None) -> np.ndarray:
    """SHEL-01: shelter-seekers per zone, ``floor(population * rate + 0.5)`` as int64."""
    rate = _cfg("demand", "public_shelter_rate") if rate is None else rate
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    pop = np.asarray(population, dtype=float)
    if np.isnan(pop).any() or (pop < 0).any():
        raise ValueError("population must be non-negative and not NaN")
    return np.floor(pop * rate + 0.5).astype(np.int64)


def build_shelter_supply(
    facilities: Sequence[Facility],
    min_shelters: int | None = None,
    school_capacity: int | None = None,
    fire_station_capacity: int | None = None,
) -> list[ShelterSite]:
    """SHEL-02: NSS shelters (capacity > 0), plus assumed schools and fire stations if fewer than ``min_shelters``.

    Hospitals are never sites. Order: shelters, schools, fire stations (input order within each kind).
    """
    if min_shelters is None:
        min_shelters = int(_cfg("shelters", "min_shelters_before_fallback"))
    if school_capacity is None:
        school_capacity = int(_cfg("shelters", "school_capacity_default"))
    if fire_station_capacity is None:
        fire_station_capacity = int(_cfg("shelters", "fire_station_capacity_default"))

    def site(f: Facility, capacity: int, assumed: bool) -> ShelterSite:
        return ShelterSite(f.id, f.kind, f.node, f.x, f.y, int(capacity), assumed)

    sites = [
        site(f, f.capacity, False)
        for f in facilities
        if f.kind == "shelter" and f.capacity and f.capacity > 0
    ]
    if len(sites) < min_shelters:
        for kind, default in (("school", school_capacity), ("fire_station", fire_station_capacity)):
            for f in facilities:
                if f.kind == kind:
                    own = f.capacity if f.capacity is not None and f.capacity > 0 else default
                    sites.append(site(f, own, True))
    return sites


def filter_unsafe_sites(
    sites: Sequence[ShelterSite],
    site_risk: Mapping[Hazard, np.ndarray],
    profile: Profile,
    threshold: float | None = None,
) -> list[ShelterSite]:
    """SHEL-03: drop sites with fire/tornado/flood ``r > threshold`` (or smoke for ``Profile.SENSITIVE``)."""
    from outrun.core.policy import Profile as P
    from outrun.core.risk import Hazard as H

    if threshold is None:
        threshold = _cfg("shelters", "remove_if_hazard_r_above")
    for arr in site_risk.values():
        if len(arr) != len(sites):
            raise ValueError("each site_risk array must have one entry per site")
    removing = [H.WILDFIRE, H.TORNADO, H.FLOOD] + ([H.SMOKE] if profile is P.SENSITIVE else [])
    unsafe = np.zeros(len(sites), dtype=bool)
    for h in removing:
        if h in site_risk:
            unsafe |= np.asarray(site_risk[h], dtype=float) > threshold
    return [s for s, bad in zip(sites, unsafe, strict=True) if not bad]


def _edge_weight(data: Mapping[str, object], weight: str):
    """Edge time weight as a float or a list of per-slot floats; None when missing / blocked (inf)."""
    w = data.get(weight)
    if w is None:
        return None
    if np.ndim(w) == 0:
        w = float(w)  # type: ignore[arg-type]
        return None if np.isinf(w) else w
    return [float(v) for v in w]  # type: ignore[union-attr]


def _durations_to(graph: nx.DiGraph, target, sources, depart_s, weight, slot_s) -> np.ndarray:
    """Earliest-arrival seconds from each source (leaving at ``depart_s``) to ``target``; inf if unreachable.

    Scalar weights: one Dijkstra from the target over the reversed graph. Per-slot weights: a forward label-setting
    search per source (exact under FIFO weights), stopped once the target is settled.
    """
    import heapq

    import networkx as nx

    if target not in graph:
        raise KeyError(target)
    for s in sources:
        if s not in graph:
            raise KeyError(s)
    out = np.full(len(sources), np.inf)
    time_dependent = any(
        np.ndim(d.get(weight)) > 0 for _, _, d in graph.edges(data=True) if d.get(weight) is not None
    )
    if not time_dependent:

        def w(_u, _v, d):
            x = _edge_weight(d, weight)
            return x  # None skips the edge

        dist = nx.single_source_dijkstra_path_length(graph.reverse(copy=False), target, weight=w)
        for i, s in enumerate(sources):
            out[i] = dist.get(s, np.inf)
        return out
    for i, s in enumerate(sources):
        best = {s: depart_s}
        heap = [(depart_s, 0, s)]
        tick = 0
        done = set()
        while heap:
            t, _, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            if u == target:
                out[i] = t - depart_s
                break
            for v, d in graph[u].items():
                w = _edge_weight(d, weight)
                if w is None:
                    continue
                tau = w if not isinstance(w, list) else w[min(int(t // slot_s), len(w) - 1)]
                if np.isinf(tau):
                    continue
                if t + tau < best.get(v, np.inf):
                    best[v] = t + tau
                    tick += 1
                    heapq.heappush(heap, (t + tau, tick, v))
    return out


def travel_cost_matrix(
    graph: nx.DiGraph,
    zone_nodes: Sequence[Hashable],
    shelter_nodes: Sequence[Hashable],
    depart_s: float = 0.0,
    weight: str = "time",
    slot_s: float | None = None,
) -> np.ndarray:
    """SHEL-04: (Z, S) time-dependent earliest-arrival seconds, one reverse search per shelter (EMS-01).

    The search is a private helper (same semantics as ``routing.emergency.td_travel_times_to``) so shelters do not
    depend on the EMS module being implemented.
    """
    slot = _DEFAULT_SLOT_S if slot_s is None else slot_s
    zones = list(zone_nodes)
    for z in zones:
        if z not in graph:
            raise KeyError(z)
    cost = np.full((len(zones), len(shelter_nodes)), np.inf)
    for j, sn in enumerate(shelter_nodes):
        cost[:, j] = _durations_to(graph, sn, zones, depart_s, weight, slot)
    return cost


def _check_instance(demand, capacity, cost_s) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    demand = np.asarray(demand)
    capacity = np.asarray(capacity)
    cost_s = np.asarray(cost_s, dtype=float)
    if cost_s.shape != (demand.shape[0], capacity.shape[0]):
        raise ValueError(
            f"cost_s shape {cost_s.shape} does not match (Z, S) = {(demand.size, capacity.size)}"
        )
    if (demand < 0).any() or (capacity < 0).any():
        raise ValueError("demand and capacity must be non-negative")
    if np.isnan(cost_s).any() or (cost_s < 0).any():
        raise ValueError("costs must be non-negative and not NaN")
    return demand.astype(np.int64), capacity.astype(np.int64), cost_s


def solve_min_cost_flow(
    demand: np.ndarray,
    capacity: np.ndarray,
    cost_s: np.ndarray,
    unserved_factor: float | None = None,
) -> ShelterAssignment:
    """SHEL-05: exact capacitated assignment with OR-Tools ``SimpleMinCostFlow`` and an "unserved" arc.

    Integer arc costs ``round(cost_s)``; unserved cost ``ceil(unserved_factor * max(1, max finite cost))``.
    """
    from ortools.graph.python import min_cost_flow

    demand, capacity, cost_s = _check_instance(demand, capacity, cost_s)
    n_z, n_s = cost_s.shape
    u_cost = unserved_cost(cost_s, unserved_factor)
    source, sink = n_z + n_s, n_z + n_s + 1  # zones 0..Z-1, shelters Z..Z+S-1
    mcf = min_cost_flow.SimpleMinCostFlow()
    cell_arcs: list[tuple[int, int, int]] = []  # (arc id, z, s)
    unserved_arcs: list[int] = []
    for z in range(n_z):
        mcf.add_arc_with_capacity_and_unit_cost(source, z, int(demand[z]), 0)
        for s in np.flatnonzero(np.isfinite(cost_s[z])):
            arc = mcf.add_arc_with_capacity_and_unit_cost(
                z, n_z + int(s), int(demand[z]), int(np.rint(cost_s[z, s]))
            )
            cell_arcs.append((arc, z, int(s)))
        unserved_arcs.append(mcf.add_arc_with_capacity_and_unit_cost(z, sink, int(demand[z]), u_cost))
    for s in range(n_s):
        mcf.add_arc_with_capacity_and_unit_cost(n_z + s, sink, int(capacity[s]), 0)
    total = int(demand.sum())
    mcf.set_node_supply(source, total)
    mcf.set_node_supply(sink, -total)
    status = mcf.solve()
    if status != mcf.OPTIMAL:
        raise RuntimeError(f"min-cost flow did not reach an optimum (status {status})")
    flow = np.zeros((n_z, n_s), dtype=np.int64)
    for arc, z, s in cell_arcs:
        flow[z, s] = mcf.flow(arc)
    unserved = np.array([mcf.flow(a) for a in unserved_arcs], dtype=np.int64)
    return ShelterAssignment("mcf", flow, unserved)


def unserved_cost(cost_s: np.ndarray, unserved_factor: float | None = None) -> int:
    """SHEL-05: the per-person cost U of the unserved arc."""
    factor = _cfg("shelters", "unserved_cost_factor") if unserved_factor is None else unserved_factor
    cost_s = np.asarray(cost_s, dtype=float)
    finite = cost_s[np.isfinite(cost_s)]
    biggest = max(1.0, float(finite.max())) if finite.size else 1.0
    return int(np.ceil(factor * biggest))


def assign_nearest(
    demand: np.ndarray,
    capacity: np.ndarray,
    zone_xy: np.ndarray,
    site_xy: np.ndarray,
    cost_s: np.ndarray,
) -> ShelterAssignment:
    """SHEL-06: every zone to its straight-line nearest site, capacity and reachability ignored."""
    demand, capacity, cost_s = _check_instance(demand, capacity, cost_s)
    if capacity.size == 0:
        raise ValueError("assign_nearest needs at least one site")
    d2 = ((np.asarray(zone_xy, float)[:, None, :] - np.asarray(site_xy, float)[None, :, :]) ** 2).sum(axis=2)
    flow = np.zeros(cost_s.shape, dtype=np.int64)
    flow[np.arange(demand.size), d2.argmin(axis=1)] = demand  # argmin: first minimum = lowest index
    return ShelterAssignment("nearest", flow, np.zeros(demand.size, dtype=np.int64))


def assign_greedy(demand: np.ndarray, capacity: np.ndarray, cost_s: np.ndarray) -> ShelterAssignment:
    """SHEL-06: zones by descending demand fill their cheapest sites with remaining capacity; rest unserved."""
    demand, capacity, cost_s = _check_instance(demand, capacity, cost_s)
    remaining = capacity.copy()
    flow = np.zeros(cost_s.shape, dtype=np.int64)
    unserved = demand.copy()
    for z in sorted(range(demand.size), key=lambda i: (-int(demand[i]), i)):
        for s in np.argsort(cost_s[z], kind="stable"):
            if unserved[z] == 0:
                break
            if not np.isfinite(cost_s[z, s]):
                break  # ascending order: everything after is also inf
            take = min(unserved[z], remaining[s])
            flow[z, s] += take
            remaining[s] -= take
            unserved[z] -= take
    return ShelterAssignment("greedy", flow, unserved)


def assignment_metrics(
    assignment: ShelterAssignment,
    capacity: np.ndarray,
    cost_s: np.ndarray,
    exposure: np.ndarray | None = None,
) -> ShelterMetrics:
    """SHEL-07: person-minutes, max minutes, overflow, unserved, mean exposure."""
    flow = assignment.flow
    cost_s = np.asarray(cost_s, dtype=float)
    used = flow > 0
    served = int(flow.sum())
    unserved = int(assignment.unserved.sum())
    total = float((flow[used] * cost_s[used]).sum() / 60.0) if used.any() else 0.0
    max_min = float(cost_s[used].max() / 60.0) if used.any() else 0.0
    overflow = int(np.maximum(0, flow.sum(axis=0) - np.asarray(capacity)).sum())
    demand_total = served + unserved
    pct = 100.0 * overflow / demand_total if demand_total else 0.0
    if exposure is None or served == 0:
        mean_exp = float("nan")
    else:
        mean_exp = float((flow * np.asarray(exposure, dtype=float)).sum() / served)
    return ShelterMetrics(served, unserved, total, max_min, overflow, pct, mean_exp)
