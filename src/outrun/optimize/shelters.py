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
        raise NotImplementedError


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
    raise NotImplementedError


def build_shelter_supply(
    facilities: Sequence[Facility],
    min_shelters: int | None = None,
    school_capacity: int | None = None,
    fire_station_capacity: int | None = None,
) -> list[ShelterSite]:
    """SHEL-02: NSS shelters (capacity > 0), plus assumed schools and fire stations if fewer than ``min_shelters``.

    Hospitals are never sites. Order: shelters, schools, fire stations (input order within each kind).
    """
    raise NotImplementedError


def filter_unsafe_sites(
    sites: Sequence[ShelterSite],
    site_risk: Mapping[Hazard, np.ndarray],
    profile: Profile,
    threshold: float | None = None,
) -> list[ShelterSite]:
    """SHEL-03: drop sites with fire/tornado/flood ``r > threshold`` (or smoke for ``Profile.SENSITIVE``)."""
    raise NotImplementedError


def travel_cost_matrix(
    graph: nx.DiGraph,
    zone_nodes: Sequence[Hashable],
    shelter_nodes: Sequence[Hashable],
    depart_s: float = 0.0,
    weight: str = "time",
    slot_s: float | None = None,
) -> np.ndarray:
    """SHEL-04: (Z, S) time-dependent earliest-arrival seconds, one reverse search per shelter (EMS-01)."""
    raise NotImplementedError


def solve_min_cost_flow(
    demand: np.ndarray,
    capacity: np.ndarray,
    cost_s: np.ndarray,
    unserved_factor: float | None = None,
) -> ShelterAssignment:
    """SHEL-05: exact capacitated assignment with OR-Tools ``SimpleMinCostFlow`` and an "unserved" arc.

    Integer arc costs ``round(cost_s)``; unserved cost ``ceil(unserved_factor * max(1, max finite cost))``.
    """
    raise NotImplementedError


def unserved_cost(cost_s: np.ndarray, unserved_factor: float | None = None) -> int:
    """SHEL-05: the per-person cost U of the unserved arc."""
    raise NotImplementedError


def assign_nearest(
    demand: np.ndarray,
    capacity: np.ndarray,
    zone_xy: np.ndarray,
    site_xy: np.ndarray,
    cost_s: np.ndarray,
) -> ShelterAssignment:
    """SHEL-06: every zone to its straight-line nearest site, capacity and reachability ignored."""
    raise NotImplementedError


def assign_greedy(demand: np.ndarray, capacity: np.ndarray, cost_s: np.ndarray) -> ShelterAssignment:
    """SHEL-06: zones by descending demand fill their cheapest sites with remaining capacity; rest unserved."""
    raise NotImplementedError


def assignment_metrics(
    assignment: ShelterAssignment,
    capacity: np.ndarray,
    cost_s: np.ndarray,
    exposure: np.ndarray | None = None,
) -> ShelterMetrics:
    """SHEL-07: person-minutes, max minutes, overflow, unserved, mean exposure."""
    raise NotImplementedError
