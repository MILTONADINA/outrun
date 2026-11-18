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

Implementation note: because risk is piecewise constant per slot, everything the router needs is precomputed
once as (E, S) tables (clock, cost, blocked) and an (N, S) table of wait-cost rates. ``clock``/``cost``/
``blocked`` are then just table look-ups at ``slot(t)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from outrun.roads.features import haversine_m

if TYPE_CHECKING:  # core interfaces are owned by spec 02; imported, never redefined
    import networkx as nx

    from outrun.core.grid import GridWindow
    from outrun.core.policy import HazardPolicy, Profile
    from outrun.core.risk import Hazard, RiskStack
    from outrun.roads.graph import RoadGraph

SLOT_S = 900.0  # 15 min
CLOCK_CAP = 3.0  # clock capped at 3x free flow (configs/hazards.yaml clock_cap_factor)
MIN_SPEED_FACTOR = 1.0 / 3.0

# A speed-factor function maps an array of probabilities to an array of factors in [1/3, 1] (same shape).
SpeedFn = Callable[[np.ndarray], np.ndarray]


def _slope_speed_fn(slope: float) -> SpeedFn:
    """sf(p) = clip(1 - slope * p, 1/3, 1), vectorized."""
    return lambda p: np.clip(1.0 - slope * p, MIN_SPEED_FACTOR, 1.0)


def _policy_speed_fn(scalar_fn: Callable[[float], float]) -> SpeedFn:
    """Apply a policy's (possibly scalar-only) speed factor elementwise.

    The policy function is evaluated once per *distinct* probability value and the results are scattered back,
    which is exact for any function and fast because risk arrays have few distinct values.
    """

    def fn(p: np.ndarray) -> np.ndarray:
        values, inverse = np.unique(p, return_inverse=True)
        factors = np.array([float(scalar_fn(float(v))) for v in values], dtype=np.float64)
        return factors[inverse].reshape(p.shape)

    return fn


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
        ff = np.asarray(free_flow_s, dtype=np.float64)
        risk = np.asarray(risk)
        if risk.ndim != 3:
            raise ValueError(f"risk must be 3-D (E, H, S), got shape {risk.shape}")
        n_e, n_h, n_s = risk.shape
        if ff.ndim != 1 or ff.shape[0] != n_e:
            raise ValueError(f"free_flow_s has shape {ff.shape}, risk has {n_e} edges")
        if n_h < 1 or n_s < 1:
            raise ValueError(f"risk needs H >= 1 and S >= 1, got shape {risk.shape}")
        if not np.all(ff > 0):
            raise ValueError("free_flow_s must be > 0 on every edge")
        _check_prob(risk, "risk")

        weights = _vector(weights, n_h, "weights")
        block_at = _vector(block_at, n_h, "block_at")
        slopes = np.zeros(n_h) if speed_slope is None else _vector(speed_slope, n_h, "speed_slope")
        if np.any(weights < 0):
            raise ValueError("weights must be >= 0")
        if np.any(slopes < 0):
            raise ValueError("speed_slope must be >= 0")
        if not lam >= 0:
            raise ValueError(f"lam must be >= 0, got {lam}")
        if not slot_s > 0:
            raise ValueError(f"slot_s must be > 0, got {slot_s}")
        if not clock_cap >= 1:
            raise ValueError(f"clock_cap must be >= 1, got {clock_cap}")

        bpr_arr = np.ones(n_e) if bpr is None else np.asarray(bpr, dtype=np.float64)
        if bpr_arr.shape != (n_e,):
            raise ValueError(f"bpr must have shape ({n_e},), got {bpr_arr.shape}")
        if np.any(bpr_arr < 1):
            raise ValueError("bpr must be >= 1")

        if node_risk is not None:
            node_risk = np.asarray(node_risk)
            if node_risk.ndim != 3 or node_risk.shape[1:] != (n_h, n_s):
                raise ValueError(f"node_risk must be (N, {n_h}, {n_s}), got {node_risk.shape}")
            _check_prob(node_risk, "node_risk")

        masks: dict[int, np.ndarray] = {}
        for h, mask in (block_mask or {}).items():
            _check_hazard_key(h, n_h, "block_mask")
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != (n_e,):
                raise ValueError(f"block_mask[{h}] must have shape ({n_e},), got {mask.shape}")
            masks[int(h)] = mask
        until: dict[int, float] = {}
        for h, secs in (block_until_s or {}).items():
            _check_hazard_key(h, n_h, "block_until_s")
            until[int(h)] = float(secs)

        self._ff = ff
        self._risk = risk
        self._weights = weights
        self._block_at = block_at
        self._slopes = slopes
        self._speed_fns: list[SpeedFn] = [_slope_speed_fn(float(s)) for s in slopes]
        self._node_risk = node_risk
        self._bpr = bpr_arr
        self._lam = float(lam)
        self._slot_s = float(slot_s)
        self._clock_cap = float(clock_cap)
        self._block_mask = masks
        self._block_until = until
        self._build_tables()

    # ------------------------------------------------------------------ construction helpers
    def _build_tables(self) -> None:
        """Precompute the (E, S) clock, cost and blocked tables and the (N, S) wait-rate table."""
        self.__dict__.pop("_lists", None)  # drop search caches of an earlier build
        self.__dict__.pop("_last_change", None)
        risk = self._risk
        n_e, n_h, n_s = risk.shape

        slowdown = np.ones((n_e, n_s))  # prod_h sf_h(r)
        exposure = np.zeros((n_e, n_s))  # sum_h w_h r
        blocked = np.zeros((n_e, n_s), dtype=bool)
        slot_start = np.arange(n_s) * self._slot_s
        for h in range(n_h):
            r_h = np.asarray(risk[:, h, :], dtype=np.float64)
            slowdown *= self._speed_fns[h](r_h)
            exposure += self._weights[h] * r_h
            b = self._block_at[h]
            if math.isnan(b):
                continue
            hits = r_h >= b
            if h in self._block_mask:
                hits &= self._block_mask[h][:, None]
            if h in self._block_until:
                hits &= (slot_start < self._block_until[h])[None, :]
            blocked |= hits

        ff = self._ff[:, None]
        clock = np.minimum(ff / slowdown * self._bpr[:, None], self._clock_cap * ff)
        cost = clock * (1.0 + self._lam * exposure)
        cost[blocked] = np.inf

        self._clock_tab = _read_only(clock)
        self._cost_tab = _read_only(cost)
        self._blocked_tab = _read_only(blocked)
        self._exposure_tab = _read_only(exposure)
        if self._node_risk is None:
            self._wait_rate = None
        else:
            node_exposure = np.einsum(
                "nhs,h->ns", np.asarray(self._node_risk, dtype=np.float64), self._weights
            )
            self._wait_rate = _read_only(1.0 + self._lam * node_exposure)

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
        from outrun.core.risk import Hazard

        order = list(Hazard)
        n_h = len(order)
        weights = np.zeros(n_h)
        block_at = np.full(n_h, np.nan)
        for h, hazard in enumerate(order):
            policy = policies.get(hazard)
            if policy is None:
                continue
            weights[h] = float(policy.exposure_weight[profile])
            b = policy.block_at[profile]
            block_at[h] = np.nan if b is None else float(b)

        # Flood is a 1 km field: only low-water crossings are walled off (DESIGN 4.2, FLOOD-10).
        n_edges = np.asarray(free_flow_s).shape[0]
        flood_mask = np.zeros(n_edges, dtype=bool) if crossing is None else np.asarray(crossing, dtype=bool)
        block_mask = {order.index(Hazard.FLOOD): flood_mask}

        if block_until_s is None:
            block_until_s = _config_block_until()
        until = {order.index(Hazard(h)): float(s) for h, s in block_until_s.items()}

        model = cls(
            free_flow_s,
            risk,
            weights,
            block_at,
            node_risk=node_risk,
            lam=lam,
            block_mask=block_mask,
            block_until_s=until,
            **kwargs,
        )
        # Replace the slope speed factors by the policies' own functions (identity for missing hazards).
        model._speed_fns = [
            _policy_speed_fn(policies[hz].speed_factor) if hz in policies else (lambda p: np.ones_like(p))
            for hz in order
        ]
        model._build_tables()
        return model

    # ------------------------------------------------------------------ properties
    @property
    def n_edges(self) -> int:
        return int(self._risk.shape[0])

    @property
    def n_hazards(self) -> int:
        return int(self._risk.shape[1])

    @property
    def n_slots(self) -> int:
        return int(self._risk.shape[2])

    @property
    def slot_s(self) -> float:
        return self._slot_s

    @property
    def window_end_s(self) -> float:
        """End of the risk window, ``n_slots * slot_s``; risk is held constant after it."""
        return self.n_slots * self._slot_s

    @property
    def block_at(self) -> np.ndarray:
        """(H,) block thresholds (NaN = never), used by the shelter-in-place rule (PLAN-04)."""
        return self._block_at.copy()

    @property
    def node_risk(self) -> np.ndarray | None:
        """(N, H, S) node risk or None."""
        return self._node_risk

    # ------------------------------------------------------------------ per-edge queries
    def slot(self, t_s: float) -> int:
        """Slot index of clock time ``t_s`` (ASTAR-09): ``min(floor(t_s/slot_s), S-1)``; t_s < 0 -> ValueError."""
        if not t_s >= 0:
            raise ValueError(f"clock time must be >= 0, got {t_s}")
        return min(int(t_s // self._slot_s), self.n_slots - 1)

    def clock(self, e: int, t_s: float) -> float:
        """tau_e(t): seconds to traverse edge ``e`` when entering it at ``t_s`` (ASTAR-07)."""
        return float(self._clock_tab[e, self.slot(t_s)])

    def cost(self, e: int, t_s: float) -> float:
        """c_e(t): generalized cost of entering edge ``e`` at ``t_s``; ``math.inf`` if blocked (ASTAR-08)."""
        return float(self._cost_tab[e, self.slot(t_s)])

    def blocked(self, e: int, t_s: float) -> bool:
        """True iff some hazard has ``r[e,h,slot(t)] >= block_at_h`` (ASTAR-08)."""
        return bool(self._blocked_tab[e, self.slot(t_s)])

    # ------------------------------------------------------------------ tables (read-only arrays)
    def clock_table(self) -> np.ndarray:
        """(E, S) array of tau per edge and slot (ASTAR-07)."""
        return self._clock_tab

    def cost_table(self) -> np.ndarray:
        """(E, S) array of c per edge and slot, inf where blocked (ASTAR-08)."""
        return self._cost_tab

    def combined_risk_table(self) -> np.ndarray:
        """(E, S) ``max_h r[e, h, s]``: the risk the single exposure definition scores (ASTAR-20, EVAL-09)."""
        return np.asarray(self._risk, dtype=np.float64).max(axis=1)

    def blocked_table(self) -> np.ndarray:
        """(E, S) bool, ``blocked_table()[e, s] == blocked(e, s * slot_s)`` (ASTAR-20)."""
        return self._blocked_tab

    def combined_node_risk_table(self, n_nodes: int | None = None) -> np.ndarray:
        """ASTAR-20. ``(N, S)`` = ``max_h node_risk[n, h, s]``; zeros ``(n_nodes, S)`` when node_risk is None
        (ValueError if then ``n_nodes`` is None, or if it differs from node_risk's N)."""
        if self._node_risk is None:
            if n_nodes is None:
                raise ValueError("the model has no node risk: pass n_nodes to get a zero table")
            return np.zeros((int(n_nodes), self.n_slots))
        if n_nodes is not None and int(n_nodes) != self._node_risk.shape[0]:
            raise ValueError(f"n_nodes={n_nodes} differs from node_risk's N={self._node_risk.shape[0]}")
        return np.asarray(self._node_risk, dtype=np.float64).max(axis=1)

    def weighted_risk_table(self) -> np.ndarray:
        """(E, S) ``sum_h w_h r[e, h, s]`` (profile weights, lam not applied) (ASTAR-20)."""
        return self._exposure_tab

    def wait_rate_table(self) -> np.ndarray | None:
        """(N, S) cost per second of waiting at a node, ``1 + lam * sum_h w_h q[n,h,s]``; None = rate 1."""
        return self._wait_rate

    def wait_cost(self, node: int, t_from: float, t_to: float) -> float:
        """Cost of waiting at ``node`` from ``t_from`` to ``t_to`` (ASTAR-10).

        Integral over [t_from, t_to] of ``1 + lam * sum_h w_h * node_risk[node, h, slot(t)]`` dt (piecewise
        constant per slot, so >= t_to - t_from). ``t_to == t_from`` -> 0; ``t_to < t_from`` -> ValueError.
        """
        if t_to < t_from:
            raise ValueError(f"wait must not go back in time ({t_from} -> {t_to})")
        if self._wait_rate is None:
            return float(t_to - t_from)
        self.slot(t_from)  # validates t_from >= 0
        rate = self._wait_rate[node]
        total, t = 0.0, float(t_from)
        while t < t_to:  # integrate slot by slot
            s = self.slot(t)
            end = t_to if s == self.n_slots - 1 else min(t_to, (s + 1) * self._slot_s)
            total += (end - t) * float(rate[s])
            t = end
        return total

    def has_node_risk(self) -> bool:
        """True when waiting costs more than 1 per second at some node and slot (node risk was given)."""
        return self._wait_rate is not None

    def last_change_s(self) -> float:
        """Start of the last slot in which any clock, cost, blocking or wait rate differs from the slot before.

        From this time on the model is static, so waiting can no longer pay off. 0.0 = static model.
        """
        if not hasattr(self, "_last_change"):
            last = 0
            for s in range(1, self.n_slots):
                same = np.array_equal(self._cost_tab[:, s], self._cost_tab[:, s - 1]) and np.array_equal(
                    self._clock_tab[:, s], self._clock_tab[:, s - 1]
                )
                if self._wait_rate is not None:
                    same = same and np.array_equal(self._wait_rate[:, s], self._wait_rate[:, s - 1])
                if not same:
                    last = s
            self._last_change = last * self._slot_s
        return self._last_change

    def search_lists(self) -> tuple[list[list[float]], list[list[float]]]:
        """``(clock, cost)`` tables as nested Python lists, cached: the A* inner loop indexes them per edge,
        and list indexing is much faster than numpy scalar indexing."""
        if not hasattr(self, "_lists"):
            self._lists = (self._clock_tab.tolist(), self._cost_tab.tolist())
        return self._lists

    def with_risk(self, risk: np.ndarray, node_risk: np.ndarray | None = None) -> CostModel:
        """Same parameters, different risk arrays (used by planners and replay scoring, ASTAR-09)."""
        model = CostModel(
            self._ff,
            risk,
            self._weights,
            self._block_at,
            speed_slope=self._slopes,
            node_risk=node_risk,
            bpr=self._bpr,
            lam=self._lam,
            slot_s=self._slot_s,
            clock_cap=self._clock_cap,
            block_mask=self._block_mask,
            block_until_s=self._block_until,
        )
        model._speed_fns = list(self._speed_fns)  # keep policy speed functions, if any
        model._build_tables()
        return model


# ---------------------------------------------------------------------- validation helpers
def _check_prob(a: np.ndarray, name: str) -> None:
    if np.isnan(a).any():
        raise ValueError(f"{name} contains NaN")
    if a.size and (a.min() < 0 or a.max() > 1):
        raise ValueError(f"{name} must be in [0, 1]")


def _vector(a: Any, n: int, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
    return a


def _check_hazard_key(h: Any, n_h: int, name: str) -> None:
    if isinstance(h, bool) or not isinstance(h, (int, np.integer)) or not 0 <= int(h) < n_h:
        raise ValueError(f"{name} keys must be hazard indices in [0, {n_h}), got {h!r}")


def _read_only(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


def _config_block_until() -> dict[Hazard, float]:
    """``hazards.<h>.block_within_min * 60`` for every hazard in configs/hazards.yaml that has the key."""
    from outrun.core.policy import load_hazard_config
    from outrun.core.risk import Hazard

    cfg = load_hazard_config().get("hazards") or {}
    out: dict[Hazard, float] = {}
    for hazard in Hazard:
        minutes = (cfg.get(hazard.value) or {}).get("block_within_min")
        if minutes is not None:
            out[hazard] = float(minutes) * 60.0
    return out


# ---------------------------------------------------------------------- grid -> edges
class EdgeRiskSampler:
    """Grid risk -> edge risk r[e, h, s] (ASTAR-14). "What you see is what routes": values come from the same
    arrays the overlays are rendered from.

    The grid is north-up: ``col = floor((x - x_left) / cell_m)``, ``row = floor((y_top - y) / cell_m)``; a point
    is inside iff ``0 <= row < ny`` and ``0 <= col < nx``. Each edge polyline (graph geometry if present, else the
    straight segment between its nodes, in EPSG:5070 metres) is sampled at arc lengths 0, spacing_m,
    2*spacing_m, ... and at its end point. r = max over the cells of those sample points; points outside the
    grid contribute 0. Node risk is the value of the cell containing the node (0 outside).

    The geometry work is done once in ``__init__``: every edge gets the list of (flat) cells it touches, so
    ``sample`` is just a gather and a per-edge max.
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
        if not cell_m > 0 or not spacing_m > 0:
            raise ValueError("cell_m and spacing_m must be > 0")
        self.n_edges = int(graph.n_edges)
        self.n_nodes = int(graph.n_nodes)
        self.shape = (int(shape[0]), int(shape[1]))
        self._x_left, self._y_top, self._cell = float(x_left), float(y_top), float(cell_m)

        edge_of_point, px, py = _sample_points(graph, float(spacing_m))
        cell = self._flat_cell(px, py)
        inside = cell >= 0
        # unique (edge, cell) pairs, sorted by edge: an edge crossing a cell with 10 samples needs it only once
        n_cells = self.shape[0] * self.shape[1]
        key = np.unique(edge_of_point[inside] * n_cells + cell[inside])
        self._pair_edge = key // n_cells
        self._pair_cell = key % n_cells
        self._node_cell = self._flat_cell(np.asarray(graph.x_m, float), np.asarray(graph.y_m, float))

    def _flat_cell(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Flat cell index ``row * nx + col`` of each point, -1 outside the grid."""
        ny, nx_ = self.shape
        col = np.floor((x - self._x_left) / self._cell).astype(np.int64)
        row = np.floor((self._y_top - y) / self._cell).astype(np.int64)
        inside = (row >= 0) & (row < ny) & (col >= 0) & (col < nx_)
        return np.where(inside, row * nx_ + col, -1)

    @classmethod
    def from_window(cls, graph: RoadGraph, window: GridWindow, spacing_m: float = 100.0) -> EdgeRiskSampler:
        """``EdgeRiskSampler(graph, window.x0, window.y0, window.res, window.shape, spacing_m)`` (ASTAR-18)."""
        return cls(graph, window.x0, window.y0, window.res, window.shape, spacing_m)

    def sample_stack(self, stack: RiskStack) -> np.ndarray:
        """(E, 6, S) edge risk of ``stack.data`` with ``stack.edge_overrides`` (Hazard keys mapped to
        ``list(Hazard).index(h)``) replacing the sampled values (ASTAR-18)."""
        from outrun.core.risk import Hazard

        order = list(Hazard)
        overrides = {order.index(h): ov for h, ov in (stack.edge_overrides or {}).items()}
        return self.sample(np.asarray(stack.data, dtype=np.float32), overrides)

    def sample_stack_nodes(self, stack: RiskStack) -> np.ndarray:
        """(N, 6, S) node risk of ``stack.data`` (ASTAR-18)."""
        return self.sample_nodes(np.asarray(stack.data, dtype=np.float32))

    def _check_prob_shape(self, prob: np.ndarray) -> None:
        if prob.ndim != 4 or prob.shape[2:] != self.shape:
            raise ValueError(f"prob must be (H, S, {self.shape[0]}, {self.shape[1]}), got {prob.shape}")

    def sample(
        self, prob: np.ndarray, edge_overrides: Mapping[int, Mapping[int, np.ndarray]] | None = None
    ) -> np.ndarray:
        """``prob`` (H, S, ny, nx) -> (E, H, S) float32 edge risk.

        ``edge_overrides[h][e]`` is a length-S array that REPLACES the sampled value of edge e for hazard index h
        (point features: low-water crossings, bridges). Shape mismatch -> ValueError. A key that is not an int in
        [0, H) (e.g. a ``Hazard``) -> ValueError, so overrides are never silently dropped.
        """
        prob = np.asarray(prob)
        self._check_prob_shape(prob)
        n_h, n_s = prob.shape[:2]
        overrides = edge_overrides or {}
        for h, per_edge in overrides.items():  # validate before doing any work
            _check_hazard_key(h, n_h, "edge_overrides")
            for e, values in per_edge.items():
                if not 0 <= int(e) < self.n_edges:
                    raise ValueError(f"edge_overrides[{h}] has edge id {e} outside [0, {self.n_edges})")
                if np.shape(values) != (n_s,):
                    raise ValueError(f"edge_overrides[{h}][{e}] must have length {n_s}")

        out = np.zeros((self.n_edges, n_h, n_s), dtype=np.float32)
        if self._pair_edge.size:
            # segment starts of each edge's run of (edge, cell) pairs, for a per-edge max with reduceat
            edges, starts = np.unique(self._pair_edge, return_index=True)
            flat = prob.reshape(n_h, n_s, -1)
            for h in range(n_h):  # one hazard at a time keeps memory at (S, pairs)
                vals = flat[h][:, self._pair_cell].astype(np.float32)
                out[edges, h, :] = np.maximum.reduceat(vals, starts, axis=1).T
        for h, per_edge in overrides.items():
            for e, values in per_edge.items():
                out[int(e), int(h), :] = np.asarray(values, dtype=np.float32)
        return out

    def sample_nodes(self, prob: np.ndarray) -> np.ndarray:
        """``prob`` (H, S, ny, nx) -> (N, H, S) node risk (cell containing each node)."""
        prob = np.asarray(prob)
        self._check_prob_shape(prob)
        n_h, n_s = prob.shape[:2]
        flat = prob.reshape(n_h, n_s, -1).astype(np.float32)
        out = np.zeros((self.n_nodes, n_h, n_s), dtype=np.float32)
        inside = self._node_cell >= 0
        out[inside] = flat[:, :, self._node_cell[inside]].transpose(2, 0, 1)
        return out


def _sample_points(graph: RoadGraph, spacing_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample points of every edge polyline at arc lengths 0, spacing, 2*spacing, ... and the end point.

    Returns ``(edge_id, x, y)`` arrays, one entry per sample point (vectorized over all edges).
    """
    n_e = int(graph.n_edges)
    # 1. polylines as a flat vertex list with offsets (straight segments when there is no geometry)
    if graph.geom_offsets is not None:
        offsets = np.asarray(graph.geom_offsets, dtype=np.int64)
        vx, vy = np.asarray(graph.geom_x, float), np.asarray(graph.geom_y, float)
    else:
        offsets = np.arange(n_e + 1, dtype=np.int64) * 2
        vx = np.column_stack([graph.x_m[graph.src], graph.x_m[graph.dst]]).ravel().astype(float)
        vy = np.column_stack([graph.y_m[graph.src], graph.y_m[graph.dst]]).ravel().astype(float)
    n_vert = np.diff(offsets)

    # 2. segments between consecutive vertices of the same edge, and their cumulative arc length per edge
    seg_edge = np.repeat(np.arange(n_e), np.maximum(n_vert - 1, 0))
    is_seg_start = np.ones(len(vx), dtype=bool)
    is_seg_start[offsets[1:] - 1] = False  # the last vertex of an edge starts no segment
    seg_i = np.flatnonzero(is_seg_start[: len(vx)])[: len(seg_edge)]
    seg_len = np.hypot(vx[seg_i + 1] - vx[seg_i], vy[seg_i + 1] - vy[seg_i])
    edge_len = np.bincount(seg_edge, weights=seg_len, minlength=n_e)
    seg_cum_end = np.cumsum(seg_len)  # global arc position (edges laid end to end)
    edge_arc0 = np.concatenate([[0.0], np.cumsum(edge_len)[:-1]])

    # 3. sample arcs: k * spacing for k = 0..floor(L/spacing), then the end point L
    n_pts = np.floor(edge_len / spacing_m).astype(np.int64) + 1
    pt_edge = np.repeat(np.arange(n_e), n_pts)
    first = np.concatenate([[0], np.cumsum(n_pts)[:-1]])
    k = np.arange(len(pt_edge)) - np.repeat(first, n_pts)
    arc = np.minimum(k * spacing_m, edge_len[pt_edge])
    pt_edge = np.concatenate([pt_edge, np.arange(n_e)])
    arc = np.concatenate([arc, edge_len])

    # 4. locate the segment holding each arc position and interpolate along it
    pts_x = vx[offsets[pt_edge]].copy()  # default: first vertex (zero-length edges)
    pts_y = vy[offsets[pt_edge]].copy()
    if len(seg_len):
        g_arc = edge_arc0[pt_edge] + arc
        seg = np.searchsorted(seg_cum_end, g_arc, side="left")
        seg = np.clip(seg, 0, len(seg_len) - 1)
        has_seg = n_vert[pt_edge] >= 2
        # keep the segment inside the point's own edge (floating point at edge boundaries)
        lo = np.searchsorted(seg_edge, pt_edge, side="left")
        hi = np.searchsorted(seg_edge, pt_edge, side="right") - 1
        seg = np.where(has_seg, np.clip(seg, lo, np.maximum(hi, lo)), 0)
        seg_start_arc = seg_cum_end[seg] - seg_len[seg]
        frac = np.where(
            seg_len[seg] > 0, (g_arc - seg_start_arc) / np.where(seg_len[seg] > 0, seg_len[seg], 1), 0
        )
        frac = np.clip(frac, 0.0, 1.0)
        i0 = seg_i[seg]
        ix = vx[i0] + frac * (vx[i0 + 1] - vx[i0])
        iy = vy[i0] + frac * (vy[i0 + 1] - vy[i0])
        pts_x = np.where(has_seg, ix, pts_x)
        pts_y = np.where(has_seg, iy, pts_y)
    return pt_edge, pts_x, pts_y


# ---------------------------------------------------------------------- heuristic
def crowfly_speed_mps(graph: RoadGraph) -> float:
    """v* = max over edges of haversine(src, dst) / free_flow_s (ASTAR-11); 0.0 if every edge has zero span.

    v* <= the maximum free-flow speed whenever length_m >= the great-circle span, so it is at least as tight
    as DESIGN's v_max, and it guarantees ``haversine(u, v) / v* <= free_flow_e`` on every edge by construction.
    """
    if graph.n_edges == 0:
        return 0.0
    span = haversine_m(graph.lon[graph.src], graph.lat[graph.src], graph.lon[graph.dst], graph.lat[graph.dst])
    return float(np.max(span / np.asarray(graph.free_flow_s, dtype=np.float64)))


def make_heuristic(graph: RoadGraph, goals: int | list[int]) -> np.ndarray:
    """h(n) = min over goals of haversine(n, goal) / v* for every node, as an (N,) array (ASTAR-11).

    h(goal) = 0; h is admissible and consistent: ``h[u] <= free_flow_e + h[v]`` for every edge (u, v), hence
    ``<= c_e(t)`` for every t. If v* == 0, h is all zeros.
    """
    goal_list = [int(goals)] if np.isscalar(goals) else [int(g) for g in goals]
    vstar = crowfly_speed_mps(graph)
    h = np.zeros(graph.n_nodes)
    if vstar <= 0 or not goal_list:
        return h
    dist = np.full(graph.n_nodes, np.inf)
    for goal in goal_list:
        dist = np.minimum(dist, haversine_m(graph.lon, graph.lat, graph.lon[goal], graph.lat[goal]))
    h = dist / vstar
    h[goal_list] = 0.0
    return h


# ---------------------------------------------------------------------- handoffs
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
    from outrun.core.policy import load_policies

    sampler = EdgeRiskSampler.from_window(graph, stack.window, spacing_m)
    return CostModel.from_policies(
        graph.free_flow_s,
        sampler.sample_stack(stack),
        policies or load_policies(),
        profile,
        node_risk=sampler.sample_stack_nodes(stack),
        lam=lam,
        crossing=graph.crossing,
    )


def to_td_digraph(graph: RoadGraph, cost_model: CostModel) -> nx.DiGraph:
    """RoadGraph + CostModel -> the EMS-convention DiGraph used by shelters, evacuation and EMS (ASTAR-19).

    Nodes 0..N-1 with x/y = x_m/y_m. One edge per (u, v); parallel edges keep the smallest free_flow_s (ties:
    lowest id). Per-slot lists: "time" = clock_table with inf where blocked, "cost" = cost_table, "risk" =
    weighted_risk_table; "eid" = CSR edge id.
    """
    import networkx as nx

    if cost_model.n_edges != graph.n_edges:
        raise ValueError("cost model and graph have different edge counts")
    time_tab = np.where(cost_model.blocked_table(), np.inf, cost_model.clock_table())
    cost_tab = cost_model.cost_table()
    risk_tab = cost_model.weighted_risk_table()

    # keep, per ordered pair, the fastest edge (lexsort: by pair, then free flow, then id)
    ff = np.asarray(graph.free_flow_s)
    order = np.lexsort((np.arange(graph.n_edges), ff, graph.dst, graph.src))
    kept: dict[tuple[int, int], int] = {}
    for e in order.tolist():
        kept.setdefault((int(graph.src[e]), int(graph.dst[e])), e)

    G = nx.DiGraph()
    for n in range(graph.n_nodes):
        G.add_node(n, x=float(graph.x_m[n]), y=float(graph.y_m[n]))
    for (u, v), e in kept.items():
        G.add_edge(
            u, v, time=time_tab[e].tolist(), cost=cost_tab[e].tolist(), risk=risk_tab[e].tolist(), eid=e
        )
    return G


def node_path_to_edges(G: nx.DiGraph, nodes: list[int]) -> np.ndarray:
    """int64 CSR edge ids ``G[u][v]["eid"]`` along a node path (ASTAR-19). < 2 nodes -> empty; missing pair -> KeyError."""
    eids = []
    for u, v in zip(nodes[:-1], nodes[1:], strict=True):
        if not G.has_edge(u, v):
            raise KeyError((u, v))
        eids.append(G[u][v]["eid"])
    return np.asarray(eids, dtype=np.int64)


def slot_times_from_waves(
    wave_times: np.ndarray, wave_s: float = 1800.0, slot_s: float = 900.0, n_slots: int = 12
) -> np.ndarray:
    """(W, E) congested wave times -> (E, n_slots); slot s uses wave min(floor(s*slot_s/wave_s), W-1) (ASTAR-19)."""
    wave_times = np.asarray(wave_times, dtype=np.float64)
    if wave_times.ndim != 2 or wave_times.shape[0] < 1:
        raise ValueError(f"wave_times must be (W >= 1, E), got {wave_times.shape}")
    wave_of_slot = np.minimum(
        np.floor(np.arange(n_slots) * slot_s / wave_s).astype(int), wave_times.shape[0] - 1
    )
    return wave_times[wave_of_slot].T.copy()
