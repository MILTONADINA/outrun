"""Whole-city evacuation: packets, k diverse paths, wave-BPR evaluator, simulated annealing, MSA user equilibrium,
shortest / safest baselines and the lower bound (spec 06-evacuation.md, DESIGN 4.5).

Units: seconds and vehicles internally; the objective J and its terms in vehicle-hours. Parameters defaulting to
``None`` are read from ``configs/planning.yaml`` (``demand.*``, ``evacuation.*``).
"""

from __future__ import annotations

import math
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


def _cfg() -> tuple[dict, dict]:
    from outrun.config import load_config

    cfg = load_config("planning")
    return cfg["demand"], cfg["evacuation"]


def make_packets(
    population: np.ndarray,
    compliance: float | None = None,
    persons_per_vehicle: float | None = None,
    packet_max: int | None = None,
) -> list[Packet]:
    """EVAC-01: vehicles per zone ``floor(pop * compliance / ppv + 0.5)`` split into near-equal packets."""
    if compliance is None or persons_per_vehicle is None or packet_max is None:
        demand, evac = _cfg()
        compliance = demand["evacuation_compliance"] if compliance is None else compliance
        persons_per_vehicle = (
            demand["persons_per_vehicle"] if persons_per_vehicle is None else persons_per_vehicle
        )
        packet_max = evac["packet_max_vehicles"] if packet_max is None else packet_max
    pop = np.asarray(population, dtype=float)
    if not np.all(np.isfinite(pop)) or np.any(pop < 0):
        raise ValueError("population must be finite and >= 0")
    if not 0.0 <= compliance <= 1.0:
        raise ValueError("compliance must be in [0, 1]")
    if persons_per_vehicle <= 0:
        raise ValueError("persons_per_vehicle must be > 0")
    if packet_max < 1:
        raise ValueError("packet_max must be >= 1")
    packets: list[Packet] = []
    for z, v in enumerate(np.floor(pop * compliance / persons_per_vehicle + 0.5).astype(np.int64).tolist()):
        if v <= 0:
            continue
        n = math.ceil(v / packet_max)
        base, extra = divmod(v, n)
        packets.extend(Packet(z, base + 1 if i < extra else base) for i in range(n))
    return packets


def diverse_paths(
    graph: nx.DiGraph,
    source: Hashable,
    target: Hashable,
    k: int | None = None,
    penalty: float | None = None,
    weight: str = "cost",
) -> list[list[Hashable]]:
    """EVAC-02: up to ``k`` distinct node paths; each round penalises used edges by ``penalty`` (compounding)."""
    import networkx as nx

    if k is None or penalty is None:
        _, evac = _cfg()
        k = evac["k_paths"] if k is None else k
        penalty = evac["path_diversity_penalty"] if penalty is None else penalty
    if k < 1 or penalty < 1:
        raise ValueError("k must be >= 1 and penalty >= 1")
    if source not in graph or target not in graph:
        raise KeyError(source if source not in graph else target)
    if source == target:
        return [[source]]
    work: dict[tuple, float] = {}

    def w(u, v, d):
        return work.get((u, v), d.get(weight, 1.0))

    paths: list[list[Hashable]] = []
    for _ in range(k):
        try:
            nodes = nx.shortest_path(graph, source, target, weight=w)
        except nx.NetworkXNoPath:
            return []
        for u, v in zip(nodes[:-1], nodes[1:], strict=True):
            work[(u, v)] = w(u, v, graph[u][v]) * penalty
        if nodes not in paths:
            paths.append(nodes)
    return paths


def bpr_time(
    t0_s: np.ndarray,
    flow: np.ndarray,
    capacity: np.ndarray,
    alpha: float | None = None,
    beta: float | None = None,
) -> np.ndarray:
    """EVAC-03: ``t0 * (1 + alpha * (flow / capacity) ** beta)``; ``capacity <= 0`` -> ``ValueError``."""
    if alpha is None or beta is None:
        _, evac = _cfg()
        alpha = evac["bpr_alpha"] if alpha is None else alpha
        beta = evac["bpr_beta"] if beta is None else beta
    cap = np.asarray(capacity, dtype=float)
    if np.any(cap <= 0):
        raise ValueError("capacity must be > 0")
    return np.asarray(t0_s, dtype=float) * (1.0 + alpha * (np.asarray(flow, dtype=float) / cap) ** beta)


class EvacEvaluator:
    """Shared congestion/exposure/clearance evaluator for every method (EVAC-03..05, EVAC-08)."""

    def __init__(self, instance: EvacInstance) -> None:
        """Resolve config defaults and validate shapes (``ValueError``)."""
        self.instance = instance
        demand_evac = None

        def pick(value, key):
            nonlocal demand_evac
            if value is not None:
                return value
            if demand_evac is None:
                demand_evac = _cfg()[1]
            return demand_evac[key]

        self.n_waves = int(pick(instance.n_waves, "waves"))
        self.wave_s = (
            float(instance.wave_s) if instance.wave_s is not None else float(pick(None, "wave_minutes")) * 60
        )
        self.slot_s = float(instance.slot_s)
        self.alpha = float(pick(instance.bpr_alpha, "bpr_alpha"))
        self.beta = float(pick(instance.bpr_beta, "bpr_beta"))
        self.lambda_E = float(pick(instance.lambda_E, "lambda_exposure"))
        lam_c = float(pick(instance.lambda_clear_per_vehicle, "lambda_clear_per_vehicle"))
        self.q = float(pick(instance.clearance_percentile, "clearance_percentile"))
        self.M = float(instance.unserved_penalty)
        if self.n_waves < 1 or self.wave_s <= 0 or self.slot_s <= 0:
            raise ValueError("n_waves, wave_s and slot_s must be positive")

        self.t0 = np.asarray(instance.edge_t0, dtype=float)
        self.cap = np.asarray(instance.edge_cap, dtype=float)
        self.risk = np.asarray(instance.edge_risk, dtype=float)
        self.veh = np.asarray(instance.vehicles, dtype=np.int64)
        E = self.t0.shape[0]
        P = self.veh.shape[0]
        if self.t0.ndim != 1 or self.cap.shape != (E,):
            raise ValueError("edge_t0 and edge_cap must be 1-D arrays of the same length")
        if self.risk.ndim != 2 or self.risk.shape[0] != E or self.risk.shape[1] < 1:
            raise ValueError("edge_risk must have shape (E, n_slots)")
        if np.any(self.t0 <= 0) or np.any(self.cap <= 0) or np.any(self.risk < 0) or np.any(self.veh <= 0):
            raise ValueError("edge_t0, edge_cap and vehicles must be > 0 and edge_risk >= 0")
        if len(instance.candidates) != P:
            raise ValueError("candidates must have one entry per packet")
        if instance.zones is not None and np.asarray(instance.zones).shape != (P,):
            raise ValueError("zones must have shape (P,)")
        self.E, self.P = E, P
        self.zones = np.arange(P) if instance.zones is None else np.asarray(instance.zones)
        self.n_slots = self.risk.shape[1]
        self.lambda_C = lam_c * float(self.veh.sum())
        self.cap_wave = self.cap * self.wave_s / 3600.0

        # candidate rows: row_base[p] + k; unserved packets have n_cand 0
        self.n_cand = np.array([len(c) for c in instance.candidates], dtype=np.int64)
        self.row_base = np.concatenate([[0], np.cumsum(self.n_cand)])[:-1]
        paths = [np.asarray(path, dtype=np.int64).ravel() for c in instance.candidates for path in c]
        for path in paths:
            if path.size and (path.min() < 0 or path.max() >= E):
                raise ValueError("edge index out of range in a candidate path")
        self.row_len = np.array([len(x) for x in paths], dtype=np.int64)
        self.row_off = np.concatenate([[0], np.cumsum(self.row_len)])[:-1]
        self.edges_all = np.concatenate(paths) if paths else np.zeros(0, dtype=np.int64)
        R = len(paths)
        self.row_T0 = np.array([self.t0[x].sum() for x in paths], dtype=float)
        self.row_exp = np.zeros((R, self.n_waves))  # risk-seconds per departure wave
        for r, path in enumerate(paths):
            tt = self.t0[path]
            before = np.cumsum(tt) - tt
            for w in range(self.n_waves):
                slot = np.minimum(
                    np.floor((w * self.wave_s + before) / self.slot_s).astype(np.int64), self.n_slots - 1
                )
                self.row_exp[r, w] = float((tt * self.risk[path, slot]).sum())
        w_idx = np.arange(self.n_waves)[None, :]
        last = np.ceil((w_idx * self.wave_s + self.row_T0[:, None]) / self.wave_s).astype(np.int64) - 1
        self.row_last = np.maximum(w_idx, last)  # (R, n_waves) last wave loaded
        self.W_alloc = int(self.row_last.max()) + 1 if R else self.n_waves
        self.W_alloc = max(self.W_alloc, self.n_waves)

    # ---- helpers
    def _check_plan(self, plan: EvacPlan) -> tuple[np.ndarray, np.ndarray]:
        k = np.asarray(plan.k, dtype=np.int64)
        w = np.asarray(plan.w, dtype=np.int64)
        if k.shape != (self.P,) or w.shape != (self.P,):
            raise ValueError("plan.k and plan.w must have shape (P,)")
        if np.any(w < 0) or np.any(w >= self.n_waves):
            raise ValueError("wave out of range")
        served = self.n_cand > 0
        if np.any(k[served] < 0) or np.any(k[served] >= self.n_cand[served]):
            raise ValueError("path index out of range for a served packet")
        return k, w

    def _cells(self, row: int, w: int) -> np.ndarray:
        """Flat (wave, edge) cell ids loaded by choice ``(row, w)``."""
        edges = self.edges_all[self.row_off[row] : self.row_off[row] + self.row_len[row]]
        waves = np.arange(w, self.row_last[row, w] + 1)
        return (waves[:, None] * self.E + edges[None, :]).ravel()

    def _cell_cost(self, cells: np.ndarray, flat_loads: np.ndarray) -> float:
        """Sum of ``x * t(x) / 3600`` over the given distinct flat cells (veh-h)."""
        e = cells % self.E
        x = flat_loads[cells]
        return float((x * bpr_time(self.t0[e], x, self.cap_wave[e], self.alpha, self.beta)).sum() / 3600.0)

    def _clearance_h(self, arrivals: np.ndarray, served: np.ndarray) -> float:
        if not served.any():
            return 0.0
        a = arrivals[served]
        v = self.veh[served]
        order = np.argsort(a, kind="stable")
        cum = np.cumsum(v[order])
        thr = self.q / 100.0 * float(v.sum())
        idx = min(int(np.searchsorted(cum, thr, side="left")), len(a) - 1)
        return float(a[order][idx]) / 3600.0

    def _state(self, plan: EvacPlan) -> _State:
        k, w = self._check_plan(plan)
        return _State(self, k.copy(), w.copy())

    # ---- spec API
    def loads(self, plan: EvacPlan) -> np.ndarray:
        """EVAC-03: vehicles per (wave, edge), each packet loading every wave its free-flow window overlaps."""
        return self._state(plan).trimmed_loads()

    def evaluate(self, plan: EvacPlan) -> EvacScore:
        """EVAC-04: J = congestion + lambda_E * exposure + lambda_C * clearance + M * unserved (veh-h)."""
        st = self._state(plan)
        arr = st.arrivals()
        served = self.n_cand > 0
        clear = self._clearance_h(arr, served)
        unserved = float(self.veh[~served].sum())
        J = st.cong + self.lambda_E * st.expo + self.lambda_C * clear + self.M * unserved
        return EvacScore(J, st.cong, st.expo, clear, unserved, arr, st.trimmed_loads())

    def delta(self, plan: EvacPlan, p: int, k: int, w: int) -> float:
        """EVAC-05: J(plan with packet p -> (k, w)) - J(plan), without modifying ``plan``."""
        st = self._state(plan)
        if self.n_cand[p] == 0 or (int(k) == st.k[p] and int(w) == st.w[p]):
            return 0.0
        if not 0 <= int(k) < self.n_cand[p] or not 0 <= int(w) < self.n_waves:
            raise ValueError("move out of range")
        before = st.J
        st.move(int(p), int(k), int(w))
        return st.J - before

    def lower_bound(self) -> float:
        """EVAC-08: sum_p v_p * min_{k,w} (T0 + lambda_E * Exp) / 3600 over served packets (veh-h)."""
        total = 0.0
        for p in np.flatnonzero(self.n_cand > 0):
            rows = self.row_base[p] + np.arange(self.n_cand[p])
            total += self.veh[p] * float((self.row_T0[rows, None] + self.lambda_E * self.row_exp[rows]).min())
        return total / 3600.0


class _State:
    """Mutable plan with incrementally maintained loads, congestion and exposure (used by delta and SA)."""

    def __init__(self, ev: EvacEvaluator, k: np.ndarray, w: np.ndarray) -> None:
        self.ev, self.k, self.w = ev, k, w
        self.flat = np.zeros(ev.W_alloc * ev.E)
        self.served = ev.n_cand > 0
        self.n_unserved_veh = float(ev.veh[~self.served].sum())
        cells = [ev._cells(ev.row_base[p] + k[p], w[p]) for p in np.flatnonzero(self.served)]
        weights = [
            np.full(len(c), ev.veh[p], dtype=float)
            for p, c in zip(np.flatnonzero(self.served), cells, strict=True)
        ]
        if cells:
            self.flat += np.bincount(np.concatenate(cells), np.concatenate(weights), minlength=self.flat.size)
        self.cong = ev._cell_cost(np.flatnonzero(self.flat), self.flat)
        self.expo = 0.0
        for p in np.flatnonzero(self.served):
            self.expo += ev.veh[p] * ev.row_exp[ev.row_base[p] + k[p], w[p]] / 3600.0
        self.clear = ev._clearance_h(self.arrivals(), self.served)
        self._undo: list[tuple] = []

    @property
    def J(self) -> float:
        ev = self.ev
        return self.cong + ev.lambda_E * self.expo + ev.lambda_C * self.clear + ev.M * self.n_unserved_veh

    def trimmed_loads(self) -> np.ndarray:
        ev = self.ev
        rows = ev.row_base[self.served] + self.k[self.served]
        n = ev.n_waves
        if rows.size:
            n = max(n, 1 + int(ev.row_last[rows, self.w[self.served]].max()))
        return self.flat.reshape(ev.W_alloc, ev.E)[:n].copy()

    def arrivals(self) -> np.ndarray:
        """Congested arrival (s) of every packet (nan if unserved), from its own departure wave's travel times."""
        ev = self.ev
        out = np.full(ev.P, np.nan)
        idx = np.flatnonzero(self.served)
        if idx.size == 0:
            return out
        rows = ev.row_base[idx] + self.k[idx]
        lens = ev.row_len[rows]
        starts = np.cumsum(lens) - lens
        pos = np.arange(int(lens.sum())) - np.repeat(starts, lens) + np.repeat(ev.row_off[rows], lens)
        edges = ev.edges_all[pos]
        waves = np.repeat(self.w[idx], lens)
        x = self.flat[waves * ev.E + edges]
        t = bpr_time(ev.t0[edges], x, ev.cap_wave[edges], ev.alpha, ev.beta)
        tot = np.bincount(np.repeat(np.arange(idx.size), lens), t, minlength=idx.size)
        out[idx] = self.w[idx] * ev.wave_s + tot
        return out

    def move(self, p: int, k: int, w: int) -> None:
        ev = self.ev
        old_k, old_w = int(self.k[p]), int(self.w[p])
        self._undo.append((p, old_k, old_w, self.cong, self.expo, self.clear))
        if not self.served[p]:
            return
        r_old, r_new = ev.row_base[p] + old_k, ev.row_base[p] + k
        c_old, c_new = ev._cells(r_old, old_w), ev._cells(r_new, w)
        touched = np.unique(np.concatenate([c_old, c_new]))
        before = ev._cell_cost(touched, self.flat)
        np.subtract.at(self.flat, c_old, float(ev.veh[p]))
        np.add.at(self.flat, c_new, float(ev.veh[p]))
        self.cong += ev._cell_cost(touched, self.flat) - before
        self.expo += ev.veh[p] * (ev.row_exp[r_new, w] - ev.row_exp[r_old, old_w]) / 3600.0
        self.k[p], self.w[p] = k, w
        self.clear = ev._clearance_h(self.arrivals(), self.served)

    def undo(self) -> None:
        p, k, w, cong, expo, clear = self._undo.pop()
        if self.served[p]:
            ev = self.ev
            r_cur, r_old = ev.row_base[p] + self.k[p], ev.row_base[p] + k
            np.subtract.at(self.flat, ev._cells(r_cur, int(self.w[p])), float(ev.veh[p]))
            np.add.at(self.flat, ev._cells(r_old, w), float(ev.veh[p]))
            self.k[p], self.w[p] = k, w
        self.cong, self.expo, self.clear = cong, expo, clear

    def commit(self) -> None:
        self._undo.clear()


def initial_temperature(uphill_deltas: Sequence[float], accept: float | None = None) -> float:
    """EVAC-06: ``-mean(uphill_deltas) / ln(accept)``; 1.0 for no deltas."""
    if accept is None:
        accept = _cfg()[1]["sa"]["warmup_accept"]
    if not 0.0 < accept < 1.0:
        raise ValueError("accept must be in (0, 1)")
    d = np.asarray(list(uphill_deltas), dtype=float)
    if np.any(d < 0):
        raise ValueError("uphill deltas must be >= 0")
    if d.size == 0:
        return 1.0
    return float(-d.mean() / math.log(accept))


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
    ev = evaluator
    sa_cfg = _cfg()[1]["sa"] if None in (iterations, alpha, warmup_moves, warmup_accept, move_probs) else {}
    iterations = int(sa_cfg["iterations"] if iterations is None else iterations)
    alpha = float(sa_cfg["alpha"] if alpha is None else alpha)
    warmup_moves = int(sa_cfg["warmup_moves"] if warmup_moves is None else warmup_moves)
    warmup_accept = float(sa_cfg["warmup_accept"] if warmup_accept is None else warmup_accept)
    probs = dict(sa_cfg["move_probs"] if move_probs is None else move_probs)
    if init is None:
        init = EvacPlan(np.zeros(ev.P, dtype=np.int64), np.zeros(ev.P, dtype=np.int64))
    st = ev._state(init)
    init_J = ev.evaluate(init).J
    rng = np.random.default_rng(seed)

    served = np.flatnonzero(ev.n_cand > 0)
    multi = np.flatnonzero(ev.n_cand > 1)
    pairs = [(int(a), int(b)) for a, b in (zone_adjacency or [])]
    by_zone: dict[int, list[int]] = {}
    for p in served.tolist():
        by_zone.setdefault(int(ev.zones[p]), []).append(p)
    pairs = [(a, b) for a, b in pairs if a in by_zone and b in by_zone]
    p_path = probs.get("repick_path", 0.75)
    p_wave = probs.get("repick_wave", 0.20)
    p_swap = probs.get("swap_adjacent_waves", 0.05)
    if not pairs:
        p_wave, p_swap = p_wave + p_swap, 0.0
    tot = p_path + p_wave + p_swap
    cut_path, cut_wave = p_path / tot, (p_path + p_wave) / tot

    def propose_and_apply() -> None:
        """Apply one random move to ``st`` (undo entries pushed); a no-op move pushes nothing."""
        if served.size == 0:
            return
        u = rng.random()
        if u < cut_path and multi.size:
            p = int(multi[rng.integers(multi.size)])
            k = int(rng.integers(ev.n_cand[p] - 1))
            st.move(p, k + 1 if k >= st.k[p] else k, int(st.w[p]))
        elif u < cut_wave or not pairs:
            if ev.n_waves < 2:
                return
            p = int(served[rng.integers(served.size)])
            w = int(rng.integers(ev.n_waves - 1))
            st.move(p, int(st.k[p]), w + 1 if w >= st.w[p] else w)
        else:
            a, b = pairs[int(rng.integers(len(pairs)))]
            pa = by_zone[a][int(rng.integers(len(by_zone[a])))]
            pb = by_zone[b][int(rng.integers(len(by_zone[b])))]
            wa, wb = int(st.w[pa]), int(st.w[pb])
            if pa == pb or wa == wb:
                return
            st.move(pa, int(st.k[pa]), wb)
            st.move(pb, int(st.k[pb]), wa)

    # warm-up: uphill deltas of random moves, not applied
    uphill = []
    for _ in range(warmup_moves):
        j0 = st.J
        propose_and_apply()
        d = st.J - j0
        while st._undo:
            st.undo()
        if d > 0:
            uphill.append(d)
    T0 = initial_temperature(uphill, warmup_accept)

    cur = best = init_J
    best_k, best_w = st.k.copy(), st.w.copy()
    history: list[tuple[int, float, float, float]] = []
    T = T0
    for i in range(1, iterations + 1):
        propose_and_apply()
        d = st.J - cur
        if st._undo and (d <= 0 or rng.random() < math.exp(-d / T)):
            cur = st.J
            if cur < best:
                best = cur
                best_k, best_w = st.k.copy(), st.w.copy()
        else:
            while st._undo:
                st.undo()
        st.commit()
        T *= alpha
        if i % callback_every == 0:
            temp = T0 * alpha**i
            history.append((i, cur, best, temp))
            if callback is not None:
                callback(i, cur, best, temp)
    best_plan = EvacPlan(best_k, best_w)
    best_J = ev.evaluate(best_plan).J
    if best_J > init_J:
        best_plan, best_J = EvacPlan(np.asarray(init.k).copy(), np.asarray(init.w).copy()), init_J
    return SAResult(best_plan, best_J, init_J, history, T0)


def msa_equilibrium(evaluator: EvacEvaluator, iterations: int | None = None) -> EvacPlan:
    """EVAC-07: method of successive averages on the risk-aware user cost; largest-share choice per packet."""
    ev = evaluator
    if iterations is None:
        iterations = int(_cfg()[1]["msa_iterations"])
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    nW = ev.n_waves
    served = np.flatnonzero(ev.n_cand > 0)
    plan_k, plan_w = np.zeros(ev.P, dtype=np.int64), np.zeros(ev.P, dtype=np.int64)
    if served.size == 0:
        return EvacPlan(plan_k, plan_w)
    kmax = int(ev.n_cand[served].max())
    n_ch = int(ev.n_cand[served].sum()) * nW
    cmat = np.full((served.size, kmax * nW), -1, dtype=np.int64)
    ch_veh = np.zeros(n_ch)
    ch_exp = np.zeros(n_ch)
    load_cells, load_ch, dep_cells, dep_ch = [], [], [], []
    c = 0
    for i, p in enumerate(served):
        for k in range(int(ev.n_cand[p])):
            row = ev.row_base[p] + k
            edges = ev.edges_all[ev.row_off[row] : ev.row_off[row] + ev.row_len[row]]
            for w in range(nW):
                cmat[i, k * nW + w] = c
                ch_veh[c] = ev.veh[p]
                ch_exp[c] = ev.row_exp[row, w]
                cells = ev._cells(row, w)
                load_cells.append(cells)
                load_ch.append(np.full(cells.size, c))
                dep_cells.append(w * ev.E + edges)
                dep_ch.append(np.full(edges.size, c))
                c += 1
    load_cells, load_ch = np.concatenate(load_cells), np.concatenate(load_ch)
    dep_cells, dep_ch = np.concatenate(dep_cells), np.concatenate(dep_ch)
    valid = cmat >= 0
    safe = np.where(valid, cmat, 0)
    edge_of_dep = dep_cells % ev.E
    edge_of_load = load_cells % ev.E

    def user_cost(share: np.ndarray | None) -> np.ndarray:
        if share is None:  # free flow
            t = ev.t0[edge_of_dep]
        else:
            x = np.bincount(load_cells, (share * ch_veh)[load_ch], minlength=ev.W_alloc * ev.E)
            t = bpr_time(ev.t0[edge_of_dep], x[dep_cells], ev.cap_wave[edge_of_dep], ev.alpha, ev.beta)
        return np.bincount(dep_ch, t, minlength=n_ch) + ev.lambda_E * ch_exp

    def target_of(cost: np.ndarray) -> np.ndarray:
        best = np.where(valid, cost[safe], np.inf).argmin(axis=1)  # first minimum: lowest k, then lowest w
        tgt = np.zeros(n_ch)
        tgt[cmat[np.arange(served.size), best]] = 1.0
        return tgt

    del edge_of_load
    share = target_of(user_cost(None))
    for n in range(1, iterations + 1):
        share = share + (target_of(user_cost(share)) - share) / (n + 1)
    pick = np.where(valid, share[safe], -1.0).argmax(axis=1)
    plan_k[served], plan_w[served] = pick // nW, pick % nW
    return EvacPlan(plan_k, plan_w)


def baseline_plan(evaluator: EvacEvaluator, method: Literal["shortest", "safest"]) -> EvacPlan:
    """EVAC-08: all-take-shortest (min T0) or all-take-safest (min T0 + lambda_E * Exp at wave 0), wave 0."""
    if method not in ("shortest", "safest"):
        raise ValueError(f"unknown baseline {method!r}")
    ev = evaluator
    k = np.zeros(ev.P, dtype=np.int64)
    for p in np.flatnonzero(ev.n_cand > 0):
        rows = ev.row_base[p] + np.arange(ev.n_cand[p])
        cost = ev.row_T0[rows] + (ev.lambda_E * ev.row_exp[rows, 0] if method == "safest" else 0.0)
        k[p] = int(np.argmin(cost))
    return EvacPlan(k, np.zeros(ev.P, dtype=np.int64))


def evac_metrics(evaluator: EvacEvaluator, plan: EvacPlan) -> dict[str, float]:
    """EVAC-10: vehicle_hours, exposure_veh_h, clearance_min, max_vc, unserved_veh, J, gap_to_lb_pct."""
    s = evaluator.evaluate(plan)
    lb = evaluator.lower_bound()
    vc = s.loads / evaluator.cap_wave[None, :]
    return {
        "vehicle_hours": s.congestion_vehh,
        "exposure_veh_h": s.exposure_vehh,
        "clearance_min": s.clearance_h * 60.0,
        "max_vc": float(vc.max()) if vc.size else 0.0,
        "unserved_veh": s.unserved_veh,
        "J": s.J,
        "gap_to_lb_pct": 100.0 * (s.J - lb) / lb if lb != 0 else float("nan"),
    }
