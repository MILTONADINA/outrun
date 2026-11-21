"""Emergency vehicle routing: reverse TD search, Hungarian dispatch, hospital leg, Pareto sweep and the
discover-and-reroute baseline (spec 07-emergency.md, DESIGN 4.6).

Graph convention (shared with ``outrun.optimize``): a ``networkx.DiGraph`` whose time weight (default attribute
``"time"``) is a scalar in seconds or a 1-D per-slot sequence (slot = ``slot_s`` seconds, default 900, last value held
beyond the end); ``inf`` or a missing attribute means blocked. Optional scalar ``"risk"`` attribute (>= 0).

Parameters defaulting to ``None`` are read from ``configs/planning.yaml`` (``emergency.*``).
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

Node = Hashable


@dataclass(frozen=True)
class RerouteResult:
    """Outcome of the discover-and-reroute baseline (EMS-05)."""

    path: list[Node]
    eta_s: float
    planned_eta_s: float
    detour_s: float
    hit_block: bool
    n_replans: int
    reached: bool


@dataclass(frozen=True)
class ParetoPoint:
    """One point of the fastest-safe trade-off (EMS-06). ``exposure`` is in risk-seconds."""

    lam: float
    time_s: float
    exposure: float
    path: list[Node]


_DEFAULT_SLOT_S = 900.0


def _cfg(key: str):
    from outrun.config import load_config

    return load_config("planning")["emergency"][key]


def _edge_weight(data, weight: str):
    """Edge time weight: float, list of per-slot floats, or None when missing / blocked (inf)."""
    w = data.get(weight)
    if w is None:
        return None
    if np.ndim(w) == 0:
        w = float(w)
        return None if math.isinf(w) else w
    return [float(v) for v in w]


def _tau(w, t: float, slot_s: float) -> float:
    return w if not isinstance(w, list) else w[min(int(t // slot_s), len(w) - 1)]


def td_travel_times_to(
    graph: nx.DiGraph,
    target: Node,
    sources: Sequence[Node],
    depart_s: float = 0.0,
    weight: str = "time",
    slot_s: float | None = None,
) -> np.ndarray:
    """EMS-01: earliest-arrival durations (s) from each source, leaving at clock ``depart_s``, to ``target``.

    Scalar weights: one Dijkstra from ``target`` over the reversed graph (all sources at once). Per-slot weights:
    a forward label-setting search per source, exact under FIFO. ``inf`` if unreachable; 0 for ``source == target``.
    Raises ``KeyError`` for unknown nodes, ``ValueError`` if ``depart_s < 0``.
    """
    import networkx as nx

    if depart_s < 0:
        raise ValueError("depart_s must be >= 0")
    slot = _DEFAULT_SLOT_S if slot_s is None else slot_s
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
        dist = nx.single_source_dijkstra_path_length(
            graph.reverse(copy=False), target, weight=lambda _u, _v, d: _edge_weight(d, weight)
        )
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
                tau = _tau(w, t, slot)
                if math.isinf(tau):
                    continue
                if t + tau < best.get(v, np.inf):
                    best[v] = t + tau
                    tick += 1
                    heapq.heappush(heap, (t + tau, tick, v))
    return out


def inbound_eta(
    graph: nx.DiGraph,
    incident: Node,
    stations: Sequence[Node],
    depart_s: float = 0.0,
    weight: str = "time",
) -> np.ndarray:
    """EMS-01: ETA (s) from every station to the incident at once (same as ``td_travel_times_to``)."""
    return td_travel_times_to(graph, incident, stations, depart_s, weight)


def ev_edge_times(
    free_flow_s: np.ndarray, congested_s: np.ndarray, speedup: float | None = None
) -> np.ndarray:
    """EMS-02: ``max(free_flow_s, congested_s / speedup)`` elementwise; ``speedup <= 0`` -> ``ValueError``."""
    speedup = float(_cfg("ev_speedup_vs_congested")) if speedup is None else speedup
    if speedup <= 0:
        raise ValueError("speedup must be positive")
    return np.maximum(np.asarray(free_flow_s, dtype=float), np.asarray(congested_s, dtype=float) / speedup)


def dispatch(eta: np.ndarray, max_incidents: int | None = None) -> list[int | None]:
    """EMS-03: station index per incident (Hungarian method).

    Maximise the number of incidents with a finite-ETA station, then minimise total ETA; distinct stations;
    ``None`` for incidents left without one. ``I > max_incidents`` -> ``ValueError``.
    """
    from scipy.optimize import linear_sum_assignment

    eta = np.asarray(eta, dtype=float)
    n_stations, n_inc = eta.shape
    limit = int(_cfg("max_incidents")) if max_incidents is None else max_incidents
    if n_inc > limit:
        raise ValueError(f"{n_inc} incidents exceeds the limit of {limit}")
    if n_inc == 0:
        return []
    finite = np.isfinite(eta)
    # Lexicographic objective in one pass: a big-M penalty per infeasible pair beats any total finite ETA.
    big = (eta[finite].sum() if finite.any() else 0.0) + 1.0
    cost = np.where(finite, eta, big)
    rows, cols = linear_sum_assignment(cost)
    out: list[int | None] = [None] * n_inc
    for s, i in zip(rows, cols, strict=True):
        if finite[s, i]:
            out[i] = int(s)
    return out


def nearest_open_hospital(
    graph: nx.DiGraph,
    incident: Node,
    hospitals: Sequence[Node],
    hospital_risk: np.ndarray,
    depart_s: float = 0.0,
    threshold: float = 0.5,
    weight: str = "time",
) -> tuple[int, float] | None:
    """EMS-04: ``(index, eta_s)`` of the fastest-reached hospital with ``risk <= threshold``; ``None`` if none."""
    risk = np.asarray(hospital_risk, dtype=float)
    if len(risk) != len(hospitals):
        raise ValueError("hospital_risk must have one entry per hospital")
    best: tuple[int, float] | None = None
    for i, h in enumerate(hospitals):
        if risk[i] > threshold:
            continue
        t = float(td_travel_times_to(graph, h, [incident], depart_s, weight)[0])
        if math.isfinite(t) and (best is None or t < best[1]):
            best = (i, t)
    return best


def nearest_station_straight_line(station_xy: np.ndarray, incident_xy: np.ndarray) -> int:
    """EMS-05: Euclidean-nearest station index (ties lowest); no stations -> ``ValueError``."""
    xy = np.asarray(station_xy, dtype=float).reshape(-1, 2)
    if len(xy) == 0:
        raise ValueError("no stations")
    return int(np.argmin(np.hypot(*(xy - np.asarray(incident_xy, dtype=float)).T)))


def _shortest_path(graph, source, target, weight, banned=frozenset()):
    """Dijkstra node path avoiding ``banned`` edges (and missing / inf weights); None if unreachable."""
    import networkx as nx

    def w(u, v, d):
        if (u, v) in banned:
            return None
        return _edge_weight(d, weight)

    try:
        return nx.dijkstra_path(graph, source, target, weight=w)
    except nx.NetworkXNoPath:
        return None


def discover_and_reroute(
    graph: nx.DiGraph,
    source: Node,
    target: Node,
    blocked: set[tuple[Node, Node]],
    weight: str = "time",
) -> RerouteResult:
    """EMS-05: hazard-blind shortest path that discovers a blocked edge only on trying to enter it, then replans.

    ``eta_s`` is the true driven time including detours; see spec for the unreachable cases.
    """
    plan = _shortest_path(graph, source, target, weight)
    if plan is None:
        return RerouteResult([source], math.inf, math.inf, math.inf, False, 0, False)
    planned = sum(graph.edges[a, b][weight] for a, b in zip(plan[:-1], plan[1:], strict=True))
    known: set[tuple[Node, Node]] = set()
    path, eta, replans = [source], 0.0, 0
    idx = 0
    while path[-1] != target:
        u, v = plan[idx], plan[idx + 1]
        if (u, v) in blocked:
            known.add((u, v))
            replans += 1
            plan = _shortest_path(graph, u, target, weight, frozenset(known))
            idx = 0
            if plan is None:
                return RerouteResult(path, math.inf, planned, math.inf, True, replans, False)
            continue
        eta += float(graph.edges[u, v][weight])
        path.append(v)
        idx += 1
    return RerouteResult(path, eta, planned, eta - planned, replans > 0, replans, True)


def route_weighted(
    graph: nx.DiGraph,
    source: Node,
    target: Node,
    lam: float,
    time: str = "time",
    risk: str = "risk",
) -> ParetoPoint | None:
    """EMS-06: path minimising ``sum time_e * (1 + lam * risk_e)``; ``None`` if unreachable; ``lam < 0`` -> error."""
    import networkx as nx

    if lam < 0:
        raise ValueError("lam must be >= 0")

    def w(_u, _v, d):
        t = d.get(time)
        if t is None or math.isinf(t):
            return None
        return float(t) * (1.0 + lam * float(d.get(risk, 0.0)))

    try:
        path = nx.dijkstra_path(graph, source, target, weight=w)
    except nx.NetworkXNoPath:
        return None
    edges = [graph.edges[a, b] for a, b in zip(path[:-1], path[1:], strict=True)]
    time_s = sum(float(e[time]) for e in edges)
    exposure = sum(float(e[time]) * float(e.get(risk, 0.0)) for e in edges)
    return ParetoPoint(float(lam), time_s, exposure, path)


def pareto_sweep(
    graph: nx.DiGraph,
    source: Node,
    target: Node,
    lambdas: Sequence[float] | None = None,
    time: str = "time",
    risk: str = "risk",
) -> list[ParetoPoint]:
    """EMS-06: one ``route_weighted`` point per lambda, in input order; ``[]`` if unreachable."""
    lams = list(_cfg("pareto_lambdas")) if lambdas is None else list(lambdas)
    points = [route_weighted(graph, source, target, lam, time, risk) for lam in lams]
    return [p for p in points if p is not None] if all(p is not None for p in points) else []


def ems_metrics(
    results: Sequence[RerouteResult], exposures: Sequence[float] | None = None
) -> dict[str, float]:
    """EMS-07: ``mean_eta_min``, ``mean_detour_min``, ``pct_hit_block``, ``pct_unreached``, ``mean_exposure``."""
    if not results:
        raise ValueError("results must not be empty")
    n = len(results)
    reached = [r for r in results if r.reached]
    return {
        "mean_eta_min": float(np.mean([r.eta_s for r in reached])) / 60.0 if reached else math.nan,
        "mean_detour_min": float(np.mean([r.detour_s for r in reached])) / 60.0 if reached else math.nan,
        "pct_hit_block": 100.0 * sum(r.hit_block for r in results) / n,
        "pct_unreached": 100.0 * (n - len(reached)) / n,
        "mean_exposure": math.nan if exposures is None else float(np.mean(exposures)),
    }
