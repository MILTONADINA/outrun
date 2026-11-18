"""Verification metrics for the six hazard models and for routes (specs/15-evaluation.md, EVAL-01..EVAL-09).

Route exposure scoring (EVAL-09) also lives here: ``RouteTrace``, ``RouteScore``, ``score_route``.

Pure numpy/scipy functions with no dependency on other ``outrun`` areas. Conventions shared by every function:

* Binary labels are ``{0, 1}``; the value ``-1`` means "uncertain" and is ignored wherever labels are accepted.
* An optional boolean ``mask`` (same shape as the data) selects the cells that count (``True`` = counted).
* A score whose denominator is zero is ``float("nan")``, never an exception.
* Event thresholds are inclusive: a value ``v`` is an event when ``v >= threshold``.
* Mismatched shapes raise ``ValueError``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Contingency",
    "ReliabilityCurve",
    "RouteScore",
    "RouteTrace",
    "auprc",
    "brier",
    "brier_skill_score",
    "contingency",
    "csi",
    "csi_over_leads",
    "ece",
    "fss",
    "mae",
    "neighbourhood_csi",
    "quantile_coverage",
    "reliability_curve",
    "rmse",
    "roc_auc",
    "route_trace",
    "score_route",
]


@dataclass(frozen=True)
class Contingency:
    """2x2 contingency table counts (EVAL-01). Ratios are ``nan`` when their denominator is 0."""

    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int

    @property
    def csi(self) -> float:
        """hits / (hits + misses + false_alarms)."""
        d = self.hits + self.misses + self.false_alarms
        return self.hits / d if d else float("nan")

    @property
    def pod(self) -> float:
        """hits / (hits + misses)."""
        d = self.hits + self.misses
        return self.hits / d if d else float("nan")

    @property
    def far(self) -> float:
        """false_alarms / (hits + false_alarms)."""
        d = self.hits + self.false_alarms
        return self.false_alarms / d if d else float("nan")


@dataclass(frozen=True)
class ReliabilityCurve:
    """Equal-width reliability bins on [0, 1] (EVAL-07).

    ``bin_edges`` has ``n_bins + 1`` entries; ``mean_pred``, ``obs_freq`` and ``counts`` have ``n_bins``.
    Empty bins have ``counts == 0`` and ``nan`` in ``mean_pred`` / ``obs_freq``.
    """

    bin_edges: np.ndarray
    mean_pred: np.ndarray
    obs_freq: np.ndarray
    counts: np.ndarray


def contingency(forecast: np.ndarray, observed: np.ndarray, mask: np.ndarray | None = None) -> Contingency:
    """EVAL-01. Contingency counts of two boolean arrays of the same shape, restricted to ``mask``."""
    raise NotImplementedError


def csi(
    forecast: np.ndarray, observed: np.ndarray, threshold: float, mask: np.ndarray | None = None
) -> float:
    """EVAL-01. Critical success index after binarising both fields with ``>= threshold``."""
    raise NotImplementedError


def csi_over_leads(
    forecast: np.ndarray, observed: np.ndarray, thresholds: Sequence[float], lead_axis: int = 0
) -> dict:
    """EVAL-02. Per-lead CSI curves and their lead average for each threshold (storm CSI-74/133 headline).

    Returns ``{"per_lead": {thr: ndarray (L,)}, "mean": {thr: float}, "mean_all": float}``.
    """
    raise NotImplementedError


def neighbourhood_csi(forecast: np.ndarray, observed: np.ndarray, threshold: float, radius: int) -> float:
    """EVAL-03. Neighbourhood CSI on a 2-D field with a square (2*radius+1) neighbourhood (flood)."""
    raise NotImplementedError


def fss(forecast: np.ndarray, observed: np.ndarray, threshold: float, window: int) -> float:
    """EVAL-04. Fractions skill score with an odd square window and zero padding outside the field."""
    raise NotImplementedError


def roc_auc(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray | None = None) -> float:
    """EVAL-05. Area under the ROC curve; ``-1`` labels ignored; ``nan`` if only one class remains."""
    raise NotImplementedError


def auprc(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray | None = None) -> float:
    """EVAL-05. Average precision (step-wise PR-AUC, sklearn definition); ``nan`` if no positives remain."""
    raise NotImplementedError


def brier(labels: np.ndarray, probs: np.ndarray, mask: np.ndarray | None = None) -> float:
    """EVAL-06. Mean squared error between probabilities and {0,1} outcomes."""
    raise NotImplementedError


def brier_skill_score(
    labels: np.ndarray,
    probs: np.ndarray,
    ref_probs: float | np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> float:
    """EVAL-06. 1 - BS / BS_ref; the default reference is climatology (the base rate of the evaluated labels)."""
    raise NotImplementedError


def reliability_curve(
    labels: np.ndarray, probs: np.ndarray, n_bins: int = 10, mask: np.ndarray | None = None
) -> ReliabilityCurve:
    """EVAL-07. Equal-width reliability bins; the last bin includes 1.0."""
    raise NotImplementedError


def ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10, mask: np.ndarray | None = None) -> float:
    """EVAL-07. Expected calibration error: sum over non-empty bins of (n_b / N) * |mean_pred_b - obs_freq_b|."""
    raise NotImplementedError


def rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """EVAL-08. Root mean squared error over pairs where neither value is NaN."""
    raise NotImplementedError


def mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """EVAL-08. Mean absolute error over pairs where neither value is NaN."""
    raise NotImplementedError


def quantile_coverage(y_true: np.ndarray, q_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    """EVAL-08. Fraction of valid pairs with ``y_true <= q_pred`` (smoke q90 coverage)."""
    raise NotImplementedError


@dataclass(frozen=True)
class RouteTrace:
    """A planned route as the router executed it (EVAL-09). Times are seconds after the scenario t0.

    Edge ``i`` is entered at ``t_enter_s[i]`` and left at ``t_exit_s[i]``. Departure from the origin is at
    ``t_start_s``; any gap ``t_enter_s[i] - t_exit_s[i-1]`` (or ``t_enter_s[0] - t_start_s``) is waiting at the
    edge's tail node ``wait_nodes[i]`` (``None`` = tail nodes unknown; waits are then charged at the edge's risk).
    """

    edges: tuple[int, ...]
    t_enter_s: tuple[float, ...]
    t_exit_s: tuple[float, ...]
    t_start_s: float = 0.0
    wait_nodes: tuple[int, ...] | None = None

    @property
    def travel_time_s(self) -> float:
        """Travel time including waits (``t_exit_s[-1] - t_start_s``; 0.0 for an empty route)."""
        return float(self.t_exit_s[-1]) - float(self.t_start_s) if self.t_exit_s else 0.0


@dataclass(frozen=True)
class RouteScore:
    """Scores of one route against one (edge x slot) risk field (EVAL-09)."""

    intersects: bool  # traversed an (edge, slot) with risk >= threshold or blocked
    exposure_risk_min: float  # integral of risk over time, in risk-minutes (waits included)
    travel_time_s: float
    blocked_crossings: int  # traversed crossing edges that were blocked while traversed


def score_route(
    trace: RouteTrace,
    edge_risk: np.ndarray,
    edge_blocked: np.ndarray | None = None,
    slot_s: float = 900.0,
    threshold: float = 0.5,
    crossing_edges: frozenset[int] | set[int] = frozenset(),
    node_risk: np.ndarray | None = None,
) -> RouteScore:
    """EVAL-09. Score a route on an (n_edges, n_slots) risk array (and optional boolean blocked array).

    Waits are charged at ``node_risk[trace.wait_nodes[i]]`` when both are given, else at the edge's risk.

    Plan on forecast, score on observed: callers pass the observed arrays here; passing a forecast array gives
    the "predicted exposure". The last slot extends to infinity (hold).
    """
    raise NotImplementedError


def route_trace(route: Any, t0_s: float = 0.0) -> RouteTrace:
    """EVAL-09. Convert a router ``Route`` (spec 04; duck-typed: ``nodes``, ``edges``, ``depart_s``, ``waits_s``,
    ``start_s``, ``arrive_s``) to a ``RouteTrace`` with times measured from ``t0_s`` (default: scenario clock),
    ``t_start_s = start_s - t0_s`` and ``wait_nodes = nodes[:-1]``."""
    raise NotImplementedError
