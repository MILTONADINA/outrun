"""Emergency vehicle routing: reverse TD search, Hungarian dispatch, hospital leg, Pareto sweep and the
discover-and-reroute baseline (spec 07-emergency.md, DESIGN 4.6).

Graph convention (shared with ``outrun.optimize``): a ``networkx.DiGraph`` whose time weight (default attribute
``"time"``) is a scalar in seconds or a 1-D per-slot sequence (slot = ``slot_s`` seconds, default 900, last value held
beyond the end); ``inf`` or a missing attribute means blocked. Optional scalar ``"risk"`` attribute (>= 0).

Parameters defaulting to ``None`` are read from ``configs/planning.yaml`` (``emergency.*``).
"""

from __future__ import annotations

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


def td_travel_times_to(
    graph: nx.DiGraph,
    target: Node,
    sources: Sequence[Node],
    depart_s: float = 0.0,
    weight: str = "time",
    slot_s: float | None = None,
) -> np.ndarray:
    """EMS-01: earliest-arrival durations (s) from each source, leaving at clock ``depart_s``, to ``target``.

    Exact under FIFO weights; ``inf`` if unreachable; 0 for ``source == target``. Intended method: one search from
    ``target`` over the reversed graph. Raises ``KeyError`` for unknown nodes, ``ValueError`` if ``depart_s < 0``.
    """
    raise NotImplementedError


def inbound_eta(
    graph: nx.DiGraph,
    incident: Node,
    stations: Sequence[Node],
    depart_s: float = 0.0,
    weight: str = "time",
) -> np.ndarray:
    """EMS-01: ETA (s) from every station to the incident at once (same as ``td_travel_times_to``)."""
    raise NotImplementedError


def ev_edge_times(
    free_flow_s: np.ndarray, congested_s: np.ndarray, speedup: float | None = None
) -> np.ndarray:
    """EMS-02: ``max(free_flow_s, congested_s / speedup)`` elementwise; ``speedup <= 0`` -> ``ValueError``."""
    raise NotImplementedError


def dispatch(eta: np.ndarray, max_incidents: int | None = None) -> list[int | None]:
    """EMS-03: station index per incident (Hungarian method).

    Maximise the number of incidents with a finite-ETA station, then minimise total ETA; distinct stations;
    ``None`` for incidents left without one. ``I > max_incidents`` -> ``ValueError``.
    """
    raise NotImplementedError


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
    raise NotImplementedError


def nearest_station_straight_line(station_xy: np.ndarray, incident_xy: np.ndarray) -> int:
    """EMS-05: Euclidean-nearest station index (ties lowest); no stations -> ``ValueError``."""
    raise NotImplementedError


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
    raise NotImplementedError


def route_weighted(
    graph: nx.DiGraph,
    source: Node,
    target: Node,
    lam: float,
    time: str = "time",
    risk: str = "risk",
) -> ParetoPoint | None:
    """EMS-06: path minimising ``sum time_e * (1 + lam * risk_e)``; ``None`` if unreachable; ``lam < 0`` -> error."""
    raise NotImplementedError


def pareto_sweep(
    graph: nx.DiGraph,
    source: Node,
    target: Node,
    lambdas: Sequence[float] | None = None,
    time: str = "time",
    risk: str = "risk",
) -> list[ParetoPoint]:
    """EMS-06: one ``route_weighted`` point per lambda, in input order; ``[]`` if unreachable."""
    raise NotImplementedError


def ems_metrics(
    results: Sequence[RerouteResult], exposures: Sequence[float] | None = None
) -> dict[str, float]:
    """EMS-07: ``mean_eta_min``, ``mean_detour_min``, ``pct_hit_block``, ``pct_unreached``, ``mean_exposure``."""
    raise NotImplementedError
