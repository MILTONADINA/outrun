"""Edge risk sampling, time-dependent edge costs and the graph/stack handoffs (spec 04-astar, ASTAR-07..ASTAR-11,
ASTAR-14, ASTAR-18..ASTAR-20).

Clock time ``t`` is seconds since the scenario issue time t0 (t >= 0). The hazard axis H of every risk array
is in ``outrun.core.risk.Hazard`` declaration order (STORM, TORNADO, FLOOD, WILDFIRE, ICE, SMOKE) when built
from policies; the array constructor accepts any H >= 1.

    slot(t)    = min(floor(t / slot_s), S - 1)                       (piecewise constant, last slot held)
    tau_e(t)   = min( free_flow_e / prod_h sf_h(r[e,h,slot(t)]) * bpr_e , clock_cap * free_flow_e )
    c_e(t)     = tau_e(t) * (1 + lam * sum_h w_h * r[e,h,slot(t)])   or +inf if r[e,h,slot(t)] >= block_at_h
    sf_h(p)    = clip(1 - speed_slope_h * p, 1/3, 1)

Every factor is >= 1 relative to free flow, so ``c_e(t) >= tau_e(t) >= free_flow_e`` (heuristic admissibility).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # core interfaces are owned by spec 02; imported, never redefined
    import networkx as nx

    from outrun.core.grid import GridWindow
    from outrun.core.policy import HazardPolicy, Profile
    from outrun.core.risk import Hazard, RiskStack
    from outrun.roads.graph import RoadGraph

SLOT_S = 900.0  # 15 min
CLOCK_CAP = 3.0  # clock capped at 3x free flow (configs/hazards.yaml clock_cap_factor)


class CostModel:
    """Time-dependent clock time and generalized cost per edge for one profile (ASTAR-07..ASTAR-10).

    Args:
        free_flow_s: (E,) free-flow seconds, > 0.
        risk: (E, H, S) edge risk in [0, 1] (from :class:`EdgeRiskSampler`).
        weights: (H,) exposure weights w_h >= 0.
        block_at: (H,) block thresholds; NaN = never blocks.
        speed_slope: (H,) slopes >= 0 for ``sf_h``; None = all 0 (no slow-down).
        node_risk: (N, H, S) risk at each node's cell, used for wait cost and shelter-in-place; None = zeros.
        bpr: (E,) congestion multiplier >= 1; None = ones.
        lam: "safer <-> faster" multiplier on all weights, >= 0.
        slot_s, clock_cap: slot length (s) and clock cap factor (>= 1).
        block_mask: {hazard index: (E,) bool}; that hazard blocks only where True (flood: crossings) (ASTAR-08).
        block_until_s: {hazard index: seconds}; that hazard blocks only in slots s with s*slot_s < value
            (tornado: 1200 s) (ASTAR-08). Both affect blocking only, never exposure or speed.

    Raises ValueError on wrong shapes, risk outside [0, 1] or NaN, negative weights/slopes/lam, bpr < 1,
    free_flow_s <= 0, slot_s <= 0 or clock_cap < 1.
    """

    def __init__(
        self,
        free_flow_s: np.ndarray,
        risk: np.ndarray,
        weights: np.ndarray,
        block_at: np.ndarray,
        *,
        speed_slope: np.ndarray | None = None,
        node_risk: np.ndarray | None = None,
        bpr: np.ndarray | None = None,
        lam: float = 1.0,
        slot_s: float = SLOT_S,
        clock_cap: float = CLOCK_CAP,
        block_mask: Mapping[int, np.ndarray] | None = None,
        block_until_s: Mapping[int, float] | None = None,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def from_policies(
        cls,
        free_flow_s: np.ndarray,
        risk: np.ndarray,
        policies: Mapping[Hazard, HazardPolicy],
        profile: Profile,
        *,
        node_risk: np.ndarray | None = None,
        lam: float = 1.0,
        crossing: np.ndarray | None = None,
        block_until_s: Mapping[Hazard, float] | None = None,
        **kwargs: Any,
    ) -> CostModel:
        """Build from ``HazardPolicy`` objects (ASTAR-08). ``risk`` has H = len(Hazard) in Hazard order.

        weight_h = ``policy.exposure_weight[profile]``; block_at_h = ``policy.block_at[profile]`` (None -> never);
        the speed factor is ``policy.speed_factor`` applied elementwise (instead of the slope form).
        A hazard missing from ``policies`` has weight 0, never blocks and speed factor 1.
        FLOOD blocks only on ``crossing`` edges (None -> FLOOD never blocks). ``block_until_s`` None -> read
        ``hazards.<h>.block_within_min`` from configs/hazards.yaml (tornado 1200 s); ``{}`` -> no time limit.
        """
        raise NotImplementedError

    @property
    def n_edges(self) -> int:
        raise NotImplementedError

    @property
    def n_slots(self) -> int:
        raise NotImplementedError

    @property
    def slot_s(self) -> float:
        raise NotImplementedError

    @property
    def block_at(self) -> np.ndarray:
        """(H,) block thresholds (NaN = never), used by the shelter-in-place rule (PLAN-04)."""
        raise NotImplementedError

    @property
    def node_risk(self) -> np.ndarray | None:
        """(N, H, S) node risk or None."""
        raise NotImplementedError

    def slot(self, t_s: float) -> int:
        """Slot index of clock time ``t_s`` (ASTAR-09): ``min(floor(t_s/slot_s), S-1)``; t_s < 0 -> ValueError."""
        raise NotImplementedError

    def clock(self, e: int, t_s: float) -> float:
        """tau_e(t): seconds to traverse edge ``e`` when entering it at ``t_s`` (ASTAR-07)."""
        raise NotImplementedError

    def cost(self, e: int, t_s: float) -> float:
        """c_e(t): generalized cost of entering edge ``e`` at ``t_s``; ``math.inf`` if blocked (ASTAR-08)."""
        raise NotImplementedError

    def blocked(self, e: int, t_s: float) -> bool:
        """True iff some hazard has ``r[e,h,slot(t)] >= block_at_h`` (ASTAR-08)."""
        raise NotImplementedError

    def clock_table(self) -> np.ndarray:
        """(E, S) array of tau per edge and slot (ASTAR-07)."""
        raise NotImplementedError

    def cost_table(self) -> np.ndarray:
        """(E, S) array of c per edge and slot, inf where blocked (ASTAR-08)."""
        raise NotImplementedError

    def combined_risk_table(self) -> np.ndarray:
        """(E, S) ``max_h r[e, h, s]``: the risk the single exposure definition scores (ASTAR-20, EVAL-09)."""
        raise NotImplementedError

    def blocked_table(self) -> np.ndarray:
        """(E, S) bool, ``blocked_table()[e, s] == blocked(e, s * slot_s)`` (ASTAR-20)."""
        raise NotImplementedError

    def combined_node_risk_table(self, n_nodes: int | None = None) -> np.ndarray:
        """ASTAR-20. ``(N, S)`` = ``max_h node_risk[n, h, s]``; zeros ``(n_nodes, S)`` when node_risk is None
        (ValueError if then ``n_nodes`` is None, or if it differs from node_risk's N)."""
        raise NotImplementedError

    def weighted_risk_table(self) -> np.ndarray:
        """(E, S) ``sum_h w_h r[e, h, s]`` (profile weights, lam not applied) (ASTAR-20)."""
        raise NotImplementedError

    def wait_cost(self, node: int, t_from: float, t_to: float) -> float:
        """Cost of waiting at ``node`` from ``t_from`` to ``t_to`` (ASTAR-10).

        Integral over [t_from, t_to] of ``1 + lam * sum_h w_h * node_risk[node, h, slot(t)]`` dt (piecewise
        constant per slot, so >= t_to - t_from). ``t_to == t_from`` -> 0; ``t_to < t_from`` -> ValueError.
        """
        raise NotImplementedError

    def with_risk(self, risk: np.ndarray, node_risk: np.ndarray | None = None) -> CostModel:
        """Same parameters, different risk arrays (used by planners and replay scoring, ASTAR-09)."""
        raise NotImplementedError


class EdgeRiskSampler:
    """Grid risk -> edge risk r[e, h, s] (ASTAR-14). "What you see is what routes": values come from the same
    arrays the overlays are rendered from.

    The grid is north-up: ``col = floor((x - x_left) / cell_m)``, ``row = floor((y_top - y) / cell_m)``; a point
    is inside iff ``0 <= row < ny`` and ``0 <= col < nx``. Each edge polyline (graph geometry if present, else the
    straight segment between its nodes, in EPSG:5070 metres) is sampled at arc lengths 0, spacing_m,
    2*spacing_m, ... and at its end point. r = max over the cells of those sample points; points outside the
    grid contribute 0. Node risk is the value of the cell containing the node (0 outside).
    """

    def __init__(
        self,
        graph: RoadGraph,
        x_left: float,
        y_top: float,
        cell_m: float,
        shape: tuple[int, int],
        spacing_m: float = 100.0,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def from_window(cls, graph: RoadGraph, window: GridWindow, spacing_m: float = 100.0) -> EdgeRiskSampler:
        """``EdgeRiskSampler(graph, window.x0, window.y0, window.res, window.shape, spacing_m)`` (ASTAR-18)."""
        raise NotImplementedError

    def sample_stack(self, stack: RiskStack) -> np.ndarray:
        """(E, 6, S) edge risk of ``stack.data`` with ``stack.edge_overrides`` (Hazard keys mapped to
        ``list(Hazard).index(h)``) replacing the sampled values (ASTAR-18)."""
        raise NotImplementedError

    def sample_stack_nodes(self, stack: RiskStack) -> np.ndarray:
        """(N, 6, S) node risk of ``stack.data`` (ASTAR-18)."""
        raise NotImplementedError

    def sample(
        self, prob: np.ndarray, edge_overrides: Mapping[int, Mapping[int, np.ndarray]] | None = None
    ) -> np.ndarray:
        """``prob`` (H, S, ny, nx) -> (E, H, S) float32 edge risk.

        ``edge_overrides[h][e]`` is a length-S array that REPLACES the sampled value of edge e for hazard index h
        (point features: low-water crossings, bridges). Shape mismatch -> ValueError. A key that is not an int in
        [0, H) (e.g. a ``Hazard``) -> ValueError, so overrides are never silently dropped.
        """
        raise NotImplementedError

    def sample_nodes(self, prob: np.ndarray) -> np.ndarray:
        """``prob`` (H, S, ny, nx) -> (N, H, S) node risk (cell containing each node)."""
        raise NotImplementedError


def crowfly_speed_mps(graph: RoadGraph) -> float:
    """v* = max over edges of haversine(src, dst) / free_flow_s (ASTAR-11); 0.0 if every edge has zero span.

    v* <= the maximum free-flow speed whenever length_m >= the great-circle span, so it is at least as tight
    as DESIGN's v_max, and it guarantees ``haversine(u, v) / v* <= free_flow_e`` on every edge by construction.
    """
    raise NotImplementedError


def make_heuristic(graph: RoadGraph, goals: int | list[int]) -> np.ndarray:
    """h(n) = min over goals of haversine(n, goal) / v* for every node, as an (N,) array (ASTAR-11).

    h(goal) = 0; h is admissible and consistent: ``h[u] <= free_flow_e + h[v]`` for every edge (u, v), hence
    ``<= c_e(t)`` for every t. If v* == 0, h is all zeros.
    """
    raise NotImplementedError


def build_cost_model(
    graph: RoadGraph,
    stack: RiskStack,
    profile: Profile,
    policies: Mapping[Hazard, HazardPolicy] | None = None,
    *,
    lam: float = 1.0,
    spacing_m: float = 100.0,
) -> CostModel:
    """The one RiskStack -> CostModel path (ASTAR-18): sampler from ``stack.window``, ``sample_stack`` and
    ``sample_stack_nodes``, then ``from_policies(..., crossing=graph.crossing)`` (policies default to
    ``load_policies()``)."""
    raise NotImplementedError


def to_td_digraph(graph: RoadGraph, cost_model: CostModel) -> nx.DiGraph:
    """RoadGraph + CostModel -> the EMS-convention DiGraph used by shelters, evacuation and EMS (ASTAR-19).

    Nodes 0..N-1 with x/y = x_m/y_m. One edge per (u, v); parallel edges keep the smallest free_flow_s (ties:
    lowest id). Per-slot lists: "time" = clock_table with inf where blocked, "cost" = cost_table, "risk" =
    weighted_risk_table; "eid" = CSR edge id.
    """
    raise NotImplementedError


def node_path_to_edges(G: nx.DiGraph, nodes: list[int]) -> np.ndarray:
    """int64 CSR edge ids ``G[u][v]["eid"]`` along a node path (ASTAR-19). < 2 nodes -> empty; missing pair -> KeyError."""
    raise NotImplementedError


def slot_times_from_waves(
    wave_times: np.ndarray, wave_s: float = 1800.0, slot_s: float = 900.0, n_slots: int = 12
) -> np.ndarray:
    """(W, E) congested wave times -> (E, n_slots); slot s uses wave min(floor(s*slot_s/wave_s), W-1) (ASTAR-19)."""
    raise NotImplementedError
