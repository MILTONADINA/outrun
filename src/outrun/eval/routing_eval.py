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

from outrun.eval.bootstrap import CI, gap_closed_ci, paired_bootstrap_ci
from outrun.eval.metrics import RouteTrace, score_route

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
    ids = np.asarray(node_ids)
    w = np.asarray(weights, dtype=np.float64)
    if len(ids) != len(w):
        raise ValueError("node_ids and weights must have the same length")
    if n < 1:
        raise ValueError("n must be >= 1")
    if np.isnan(w).any() or (w < 0).any():
        raise ValueError("weights must be non-negative and not NaN")
    total = w.sum()
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    rng = np.random.default_rng(seed)
    return ids[rng.choice(len(ids), size=n, replace=True, p=w / total)]


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
    origins = list(origins)
    if not origins:
        raise ValueError("origins must not be empty")
    crossing = frozenset(int(e) for e in crossing_edges)
    routes: dict[PlannerKind, list[RouteTrace | None]] = {k: [] for k in PLANNER_ORDER}
    for origin in origins:
        for kind in PLANNER_ORDER:
            routes[kind].append(plan(kind, origin))

    def score(trace: RouteTrace, risk: np.ndarray, blocked: np.ndarray | None, node_risk: np.ndarray | None):
        return score_route(
            trace,
            risk,
            blocked,
            slot_s=slot_s,
            threshold=threshold,
            crossing_edges=crossing,
            node_risk=node_risk,
        )

    paired = [i for i in range(len(origins)) if all(routes[k][i] is not None for k in PLANNER_ORDER)]
    n_paired = len(paired)
    scores = {
        k: [score(routes[k][i], observed_risk, observed_blocked, observed_node_risk) for i in paired]
        for k in PLANNER_ORDER
    }
    exposure = {
        k: np.array([s.exposure_risk_min for s in scores[k]], dtype=np.float64) for k in PLANNER_ORDER
    }
    base_time = np.array([s.travel_time_s for s in scores[PlannerKind.NO_HAZARD]], dtype=np.float64)

    summaries: dict[PlannerKind, PlannerSummary] = {}
    for k in PLANNER_ORDER:
        n_routed = sum(r is not None for r in routes[k])
        sc = scores[k]
        if n_paired:
            times = np.array([s.travel_time_s for s in sc], dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(base_time > 0, times / np.where(base_time > 0, base_time, 1.0), np.nan)
            ratio_mean = float(np.nanmean(ratio)) if np.isfinite(ratio).any() else float("nan")
            pct = 100.0 * float(np.mean([s.intersects for s in sc]))
            mean_exp = float(np.mean(exposure[k]))
            crossings = int(sum(s.blocked_crossings for s in sc))
        else:
            ratio_mean = pct = mean_exp = float("nan")
            crossings = 0
        predicted = None
        if predicted_risk is not None and k in predicted_risk and n_paired:
            predicted = float(
                np.mean(
                    [
                        score_route(
                            routes[k][i], predicted_risk[k], slot_s=slot_s, threshold=threshold
                        ).exposure_risk_min
                        for i in paired
                    ]
                )
            )
        summaries[k] = PlannerSummary(
            n_routed, len(origins) - n_routed, pct, mean_exp, ratio_mean, crossings, predicted
        )

    if n_paired < 2:
        nan_ci = CI(float("nan"), float("nan"), float("nan"), 0, n_paired)
        gap, c_b, c_ap = nan_ci, nan_ci, nan_ci
    else:
        ea, eap, eb, ec, ed = (exposure[k] for k in PLANNER_ORDER)
        gap = gap_closed_ci(ea, ec, ed, n_resamples=n_resamples, seed=seed)
        c_b = paired_bootstrap_ci(
            lambda c, b: float(np.mean(c - b)), (ec, eb), n_resamples=n_resamples, seed=seed
        )
        c_ap = paired_bootstrap_ci(
            lambda c, a: float(np.mean(c - a)), (ec, eap), n_resamples=n_resamples, seed=seed
        )
    return RoutingEvalResult(len(origins), n_paired, summaries, gap, c_b, c_ap)
