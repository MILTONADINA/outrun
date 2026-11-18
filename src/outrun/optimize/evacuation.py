"""Whole-city evacuation: packets, k diverse paths, wave-BPR evaluator, simulated annealing, MSA user equilibrium,
shortest / safest baselines and the lower bound (spec 06-evacuation.md, DESIGN 4.5).

Units: seconds and vehicles internally; the objective J and its terms in vehicle-hours. Parameters defaulting to
``None`` are read from ``configs/planning.yaml`` (``demand.*``, ``evacuation.*``).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

DEFAULT_UNSERVED_PENALTY = 1e6  # M, veh-h per unserved vehicle (not in planning.yaml: "M large")


@dataclass(frozen=True)
class Packet:
    """Up to ``packet_max`` vehicles from one zone that move together (EVAC-01)."""

    zone: int
    vehicles: int


@dataclass(frozen=True)
class EvacInstance:
    """Pure data for the evaluator (spec 06, "Instance"). Paths are edge-index arrays in travel order."""

    edge_t0: np.ndarray  # (E,) free-flow seconds
    edge_cap: np.ndarray  # (E,) veh/h
    edge_risk: np.ndarray  # (E, n_slots) profile-weighted risk >= 0
    vehicles: np.ndarray  # (P,) int
    candidates: list[list[np.ndarray]]  # per packet; [] = unserved
    n_waves: int | None = None
    wave_s: float | None = None
    slot_s: float = 900.0
    bpr_alpha: float | None = None
    bpr_beta: float | None = None
    lambda_E: float | None = None
    lambda_clear_per_vehicle: float | None = None
    clearance_percentile: float | None = None
    unserved_penalty: float = DEFAULT_UNSERVED_PENALTY
    zones: np.ndarray | None = None  # (P,) zone of each packet; None -> packet index (for the swap move)


@dataclass(frozen=True)
class EvacPlan:
    """Decision per packet: candidate path index ``k`` and departure wave ``w`` (int arrays of length P)."""

    k: np.ndarray
    w: np.ndarray


@dataclass(frozen=True)
class EvacScore:
    """EVAC-04 objective and its parts (veh-h, hours); ``arrivals_s`` is nan for unserved packets."""

    J: float
    congestion_vehh: float
    exposure_vehh: float
    clearance_h: float
    unserved_veh: float
    arrivals_s: np.ndarray
    loads: np.ndarray  # (n_load_waves, E)


@dataclass(frozen=True)
class SAResult:
    """EVAC-06 result; ``history`` rows are ``(iteration, current_J, best_J, temperature)`` with
    ``temperature = T0 * alpha**iteration``; ``T0`` is the initial temperature used."""

    best_plan: EvacPlan
    best_J: float
    init_J: float
    history: list[tuple[int, float, float, float]] = field(default_factory=list)
    T0: float = float("nan")


def make_packets(
    population: np.ndarray,
    compliance: float | None = None,
    persons_per_vehicle: float | None = None,
    packet_max: int | None = None,
) -> list[Packet]:
    """EVAC-01: vehicles per zone ``floor(pop * compliance / ppv + 0.5)`` split into near-equal packets."""
    raise NotImplementedError


def diverse_paths(
    graph: nx.DiGraph,
    source: Hashable,
    target: Hashable,
    k: int | None = None,
    penalty: float | None = None,
    weight: str = "cost",
) -> list[list[Hashable]]:
    """EVAC-02: up to ``k`` distinct node paths; each round penalises used edges by ``penalty`` (compounding)."""
    raise NotImplementedError


def bpr_time(
    t0_s: np.ndarray,
    flow: np.ndarray,
    capacity: np.ndarray,
    alpha: float | None = None,
    beta: float | None = None,
) -> np.ndarray:
    """EVAC-03: ``t0 * (1 + alpha * (flow / capacity) ** beta)``; ``capacity <= 0`` -> ``ValueError``."""
    raise NotImplementedError


class EvacEvaluator:
    """Shared congestion/exposure/clearance evaluator for every method (EVAC-03..05, EVAC-08)."""

    def __init__(self, instance: EvacInstance) -> None:
        """Resolve config defaults and validate shapes (``ValueError``)."""
        raise NotImplementedError

    def loads(self, plan: EvacPlan) -> np.ndarray:
        """EVAC-03: vehicles per (wave, edge), each packet loading every wave its free-flow window overlaps."""
        raise NotImplementedError

    def evaluate(self, plan: EvacPlan) -> EvacScore:
        """EVAC-04: J = congestion + lambda_E * exposure + lambda_C * clearance + M * unserved (veh-h)."""
        raise NotImplementedError

    def delta(self, plan: EvacPlan, p: int, k: int, w: int) -> float:
        """EVAC-05: J(plan with packet p -> (k, w)) - J(plan), without modifying ``plan``."""
        raise NotImplementedError

    def lower_bound(self) -> float:
        """EVAC-08: sum_p v_p * min_{k,w} (T0 + lambda_E * Exp) / 3600 over served packets (veh-h)."""
        raise NotImplementedError


def initial_temperature(uphill_deltas: Sequence[float], accept: float | None = None) -> float:
    """EVAC-06: ``-mean(uphill_deltas) / ln(accept)``; 1.0 for no deltas."""
    raise NotImplementedError


def simulated_annealing(
    evaluator: EvacEvaluator,
    init: EvacPlan | None = None,
    *,
    iterations: int | None = None,
    alpha: float | None = None,
    warmup_moves: int | None = None,
    warmup_accept: float | None = None,
    move_probs: dict[str, float] | None = None,
    seed: int = 0,
    zone_adjacency: Sequence[tuple[int, int]] | None = None,
    callback: Callable[[int, float, float, float], None] | None = None,
    callback_every: int = 100,
) -> SAResult:
    """EVAC-06: Metropolis SA over (path, wave) per packet; returns the best plan ever seen (incl. ``init``).

    ``callback(iteration, current_J, best_J, temperature)`` every ``callback_every`` iterations (APP-05 progress).
    """
    raise NotImplementedError


def msa_equilibrium(evaluator: EvacEvaluator, iterations: int | None = None) -> EvacPlan:
    """EVAC-07: method of successive averages on the risk-aware user cost; largest-share choice per packet."""
    raise NotImplementedError


def baseline_plan(evaluator: EvacEvaluator, method: Literal["shortest", "safest"]) -> EvacPlan:
    """EVAC-08: all-take-shortest (min T0) or all-take-safest (min T0 + lambda_E * Exp at wave 0), wave 0."""
    raise NotImplementedError


def evac_metrics(evaluator: EvacEvaluator, plan: EvacPlan) -> dict[str, float]:
    """EVAC-10: vehicle_hours, exposure_veh_h, clearance_min, max_vc, unserved_veh, J, gap_to_lb_pct."""
    raise NotImplementedError
