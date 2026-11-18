"""Routing evaluation: plan on a forecast, score on observed data (specs/15-evaluation.md, EVAL-13..EVAL-15).

DESIGN 6.2. Five planners differ only in the risk source they plan on (the CLI switch ``--risk-source``):

====  ===========  ================================
kind  risk source  meaning
====  ===========  ================================
a     none         no hazard (plain fastest)
a'    current-only risk frozen at t0, no forecast
b     baseline     baseline-model risk layers
c     model        deep-learning risk layers
d     observed     oracle, planned on ``observed()``
====  ===========  ================================

Every planner's route is scored on the same OBSERVED (edge x slot) arrays. This module never imports the router:
the caller injects ``plan(kind, origin) -> RouteTrace | None`` (built on ``outrun.routing.planner`` by the CLI),
so the evaluation logic is testable on synthetic routes.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from outrun.eval.bootstrap import CI
from outrun.eval.metrics import RouteTrace

__all__ = [
    "PLANNER_ORDER",
    "RISK_SOURCE",
    "PlannerKind",
    "PlannerSummary",
    "RoutingEvalResult",
    "evaluate_planners",
    "sample_origins",
]


class PlannerKind(StrEnum):
    """The five routing-evaluation planners (DESIGN 6.2)."""

    NO_HAZARD = "a"
    CURRENT_ONLY = "a_prime"
    BASELINE = "b"
    MODEL = "c"
    ORACLE = "d"


PLANNER_ORDER: tuple[PlannerKind, ...] = (
    PlannerKind.NO_HAZARD,
    PlannerKind.CURRENT_ONLY,
    PlannerKind.BASELINE,
    PlannerKind.MODEL,
    PlannerKind.ORACLE,
)

#: EVAL-13: the ``--risk-source`` value each planner plans on.
RISK_SOURCE: dict[PlannerKind, str] = {
    PlannerKind.NO_HAZARD: "none",
    PlannerKind.CURRENT_ONLY: "current-only",
    PlannerKind.BASELINE: "baseline",
    PlannerKind.MODEL: "model",
    PlannerKind.ORACLE: "observed",
}


@dataclass(frozen=True)
class PlannerSummary:
    """Per-planner aggregates over the paired origin set (EVAL-14)."""

    n_routed: int  # origins (of all) for which this planner returned a route
    n_failed: int  # origins (of all) for which it returned None
    pct_intersect: float  # 0..100, share of paired routes that intersect observed hazard
    mean_exposure_risk_min: float  # mean observed exposure, risk-minutes
    mean_travel_time_ratio: float  # mean of travel_time / travel_time of planner (a), per origin
    blocked_crossings: int  # total blocked crossings traversed over paired routes
    mean_predicted_exposure: float | None  # mean exposure on this planner's own forecast arrays, if given


@dataclass(frozen=True)
class RoutingEvalResult:
    """Output of ``evaluate_planners`` (EVAL-14, EVAL-15)."""

    n_origins: int
    n_paired: int
    summaries: dict[PlannerKind, PlannerSummary]
    gap_closed: CI  # (E_a - E_c) / (E_a - E_d), paired bootstrap over origins
    c_minus_b: CI  # mean(E_c - E_b), paired bootstrap (negative = DL better)
    c_minus_a_prime: CI  # mean(E_c - E_a'), paired bootstrap


def sample_origins(
    node_ids: Sequence[int] | np.ndarray, weights: Sequence[float] | np.ndarray, n: int = 500, seed: int = 0
) -> np.ndarray:
    """EVAL-13. ``n`` population-weighted origins drawn with replacement, P(node) proportional to its weight."""
    raise NotImplementedError


def evaluate_planners(
    origins: Sequence[int],
    plan: Callable[[PlannerKind, int], RouteTrace | None],
    observed_risk: np.ndarray,
    observed_blocked: np.ndarray | None = None,
    predicted_risk: Mapping[PlannerKind, np.ndarray] | None = None,
    crossing_edges: Collection[int] = (),
    slot_s: float = 900.0,
    threshold: float = 0.5,
    n_resamples: int = 1000,
    seed: int = 0,
    observed_node_risk: np.ndarray | None = None,
) -> RoutingEvalResult:
    """EVAL-14/EVAL-15. Route every origin with all five planners and score every route on observed arrays."""
    raise NotImplementedError
