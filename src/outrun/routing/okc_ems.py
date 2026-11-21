"""Real-road EMS dispatch demo and app backend core (T21; spec 07, DESIGN 4.6).

Glue between the road graph + cost model (routing.cost) and the pure EMS functions (routing.emergency):

1. one scalar EV graph for a departure slot: ``time`` = EV edge time (``ev_edge_times``) with ``inf`` on edges the
   EV profile blocks, ``blind`` = the same time ignoring blocks (what a hazard-blind navigator believes), ``risk`` =
   profile-weighted hazard sum of the slot;
2. ETA from every fire/EMS station by one reverse search per incident, Hungarian assignment, fastest-safe route
   (``route_weighted``) and the hospital leg;
3. the baseline: straight-line nearest station, hazard-blind shortest path, discover-and-reroute.

Run ``python -m outrun.routing.okc_ems`` for the El Reno demo. Simplification, stated on purpose: the search uses the
edge times of the departure slot (15 min), not per-slot lists. Emergency trips are under 30 min, and a scalar graph
lets one reverse Dijkstra answer all stations on a 143k-node city graph in about a second.
"""

from __future__ import annotations

import json
import math
import time as _time
from pathlib import Path
from typing import Any

import numpy as np

from outrun.routing.emergency import (
    discover_and_reroute,
    dispatch,
    ems_metrics,
    ev_edge_times,
    inbound_eta,
    nearest_open_hospital,
    nearest_station_straight_line,
    pareto_sweep,
    route_weighted,
)

HOSPITAL_CANDIDATES = 6  # nearest open hospitals (crow-fly) examined by the exact search
HOSPITAL_RISK_THRESHOLD = 0.5


def snap_nodes(graph: Any, lonlat: np.ndarray) -> np.ndarray:
    """Nearest node of the largest strongly connected component for each (lon, lat) row (no dead-end islands)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    adj = csr_matrix((np.ones(graph.n_edges), (graph.src, graph.dst)), shape=(graph.n_nodes,) * 2)
    _, comp = connected_components(adj, directed=True, connection="strong")
    ok = np.flatnonzero(comp == np.bincount(comp).argmax())
    out = []
    for lon, lat in np.asarray(lonlat, dtype=float).reshape(-1, 2):
        dx = (graph.lon[ok] - lon) * math.cos(math.radians(lat))
        out.append(int(ok[np.argmin(dx * dx + (graph.lat[ok] - lat) ** 2)]))
    return np.asarray(out, dtype=np.int64)


def build_ev_graph(graph: Any, model: Any, slot: int) -> tuple[Any, set[tuple[int, int]]]:
    """Scalar EV DiGraph for ``slot`` plus the set of blocked (u, v) edges. See module docstring for attributes."""
    import networkx as nx

    clock = model.clock_table()[:, slot]
    blind = ev_edge_times(graph.free_flow_s, clock)
    blocked = model.blocked_table()[:, slot]
    risk = model.weighted_risk_table()[:, slot]
    best: dict[tuple[int, int], int] = {}
    for e in np.argsort(blind, kind="stable").tolist():  # fastest parallel edge wins
        best.setdefault((int(graph.src[e]), int(graph.dst[e])), e)
    G = nx.DiGraph()
    G.add_nodes_from(range(graph.n_nodes))
    blocked_set: set[tuple[int, int]] = set()
    for (u, v), e in best.items():
        is_blocked = bool(blocked[e])
        G.add_edge(
            u, v, time=math.inf if is_blocked else float(blind[e]), blind=float(blind[e]), risk=float(risk[e])
        )  # fmt: skip
        if is_blocked:
            blocked_set.add((u, v))
    return G, blocked_set


def _line(graph: Any, nodes: list[int], props: dict[str, Any]) -> dict[str, Any]:
    coords = [[float(graph.lon[n]), float(graph.lat[n])] for n in nodes]
    return {"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": coords}}


def _path_stats(G: Any, path: list[int], weight: str) -> tuple[float, float]:
    """(time_s, exposure risk-seconds) of a node path."""
    t = x = 0.0
    for u, v in zip(path[:-1], path[1:], strict=True):
        d = G.edges[u, v]
        t += d[weight]
        x += d[weight] * d["risk"]
    return t, x


def _json_safe(x: Any) -> Any:
    """NaN / inf -> None and numpy scalars -> Python, so the dict survives ``JSONResponse`` (allow_nan=False)."""
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def plan_dispatch(
    graph: Any,
    model: Any,
    stations: np.ndarray,
    hospitals: np.ndarray,
    incidents: np.ndarray,
    t0_s: float,
    lam: float = 1.0,
    with_pareto: bool = False,
    ev_graph: tuple[Any, set] | None = None,
) -> dict[str, Any]:
    """Dispatch 1-3 incidents (lon/lat rows) from ``stations`` (lon/lat rows) and route each patient to a hospital.

    Returns ``assignment`` ([{"incident", "unit", "eta_min", ...}]), ``routes`` and ``baseline`` GeoJSON collections,
    ``metrics`` (risk-aware vs hazard-blind baseline) and the per-incident ``comparison`` rows.
    """
    slot = model.slot(t0_s)
    G, blocked = ev_graph or build_ev_graph(graph, model, slot)
    st_nodes = snap_nodes(graph, stations).tolist()
    hosp_nodes = snap_nodes(graph, hospitals).tolist() if len(hospitals) else []
    inc_nodes = snap_nodes(graph, incidents).tolist()
    node_risk = model.node_risk
    hosp_risk = (
        node_risk[hosp_nodes, :, slot].max(axis=1) if node_risk is not None and hosp_nodes else np.zeros(len(hosp_nodes))
    )  # fmt: skip

    eta = np.column_stack([inbound_eta(G, n, st_nodes) for n in inc_nodes])  # (S, I) seconds
    units = dispatch(eta)

    routes, baseline, rows, base_results, base_exposure = [], [], [], [], []
    for i, unit in enumerate(units):
        inc = inc_nodes[i]
        row: dict[str, Any] = {"incident": i, "unit": unit}
        if unit is not None:
            aware = route_weighted(G, st_nodes[unit], inc, lam)
            if aware is not None:
                row.update(eta_min=aware.time_s / 60.0, exposure=aware.exposure)
                routes.append(
                    _line(graph, aware.path, {"kind": "to_incident", "incident": i, "unit": unit, "lam": lam})
                )
                if with_pareto:
                    row["pareto"] = [
                        {"lam": p.lam, "time_s": p.time_s, "exposure": p.exposure}
                        for p in pareto_sweep(G, st_nodes[unit], inc)
                    ]
        # hospital leg from the incident (open hospitals only, nearest few by straight line)
        if hosp_nodes:
            open_idx = [k for k in range(len(hosp_nodes)) if hosp_risk[k] <= HOSPITAL_RISK_THRESHOLD]
            d = np.hypot(*(hospitals[open_idx] - incidents[i]).T) if open_idx else np.array([])
            cand = [open_idx[k] for k in np.argsort(d)[:HOSPITAL_CANDIDATES]]
            best = nearest_open_hospital(
                G, inc, [hosp_nodes[k] for k in cand], np.zeros(len(cand)), threshold=HOSPITAL_RISK_THRESHOLD
            )
            if best is not None:
                k = cand[best[0]]
                leg = route_weighted(G, inc, hosp_nodes[k], lam)
                if leg is not None:
                    row.update(hospital=k, hospital_eta_min=leg.time_s / 60.0, hospital_exposure=leg.exposure)
                    routes.append(
                        _line(graph, leg.path, {"kind": "to_hospital", "incident": i, "hospital": k})
                    )
        # baseline: straight-line nearest station, hazard-blind shortest path, discover-and-reroute
        s_xy = np.c_[graph.x_m[st_nodes], graph.y_m[st_nodes]]
        b_station = nearest_station_straight_line(s_xy, np.array([graph.x_m[inc], graph.y_m[inc]]))
        res = discover_and_reroute(G, st_nodes[b_station], inc, blocked, weight="blind")
        base_results.append(res)
        exposure = _path_stats(G, res.path, "blind")[1] if res.reached else math.nan
        base_exposure.append(exposure)
        row.update(
            baseline_unit=b_station, baseline_eta_min=res.eta_s / 60.0, baseline_planned_eta_min=res.planned_eta_s / 60.0,
            baseline_detour_min=res.detour_s / 60.0, baseline_hit_block=res.hit_block, baseline_exposure=exposure,
        )  # fmt: skip
        if res.reached:
            baseline.append(_line(graph, res.path, {"kind": "baseline", "incident": i, "unit": b_station}))
        rows.append(row)

    finite_exp = [x for x in base_exposure if not math.isnan(x)]
    metrics = ems_metrics(base_results, finite_exp if len(finite_exp) == len(base_results) else None)
    aware_rows = [r for r in rows if "eta_min" in r]
    metrics["aware_mean_eta_min"] = (
        float(np.mean([r["eta_min"] for r in aware_rows])) if aware_rows else math.nan
    )
    metrics["aware_mean_exposure"] = (
        float(np.mean([r["exposure"] for r in aware_rows])) if aware_rows else math.nan
    )
    return _json_safe({
        "assignment": [{"incident": r["incident"], "unit": r["unit"]} for r in rows],
        "routes": {"type": "FeatureCollection", "features": routes},
        "baseline": {"type": "FeatureCollection", "features": baseline},
        "comparison": rows,
        "metrics": metrics,
        "lambda": lam,
        "n_blocked_edges": len(blocked),
    })  # fmt: skip


def load_facility_points(bbox: tuple[float, float, float, float], raw_dir: str | Path | None = None):
    """(stations (S, 2), hospitals (H, 2)) lon/lat arrays from the HIFLD layers (T11 loaders)."""
    from outrun.config import get_paths
    from outrun.data.facilities import load_hifld_facilities

    raw = Path(raw_dir) if raw_dir else get_paths().raw
    df = load_hifld_facilities(raw, bbox)
    pts = {k: df[df["kind"] == k][["lon", "lat"]].to_numpy(dtype=float) for k in ("fire_station", "hospital")}
    if not len(pts["fire_station"]):
        raise LookupError("no fire/EMS stations in the scenario bbox")
    return pts["fire_station"], pts["hospital"]


def compare_routing(
    graph: Any, G: Any, blocked: set, station_nodes: list[int], incident_nodes: list[int], lam: float = 1.0
) -> dict:
    """Same-station comparison per incident: hazard-blind baseline (discover-and-reroute, true ETA) against the
    risk-aware route (lam) and the lam = 0 route that only avoids blocked edges. The station is the straight-line
    nearest (EMS-05 baseline). Means over incidents reachable by all three."""
    rows, results = [], []
    s_xy = np.c_[graph.x_m[station_nodes], graph.y_m[station_nodes]]
    for inc in incident_nodes:
        st = station_nodes[nearest_station_straight_line(s_xy, np.array([graph.x_m[inc], graph.y_m[inc]]))]
        res = discover_and_reroute(G, st, inc, blocked, weight="blind")
        aware, fast = route_weighted(G, st, inc, lam), route_weighted(G, st, inc, 0.0)
        results.append(res)
        if res.reached and aware and fast:
            rows.append(
                (res.eta_s, _path_stats(G, res.path, "blind")[1], res.hit_block, aware.time_s, aware.exposure,
                 fast.time_s, fast.exposure)
            )  # fmt: skip
    m = ems_metrics(results, None)
    if not rows:
        return {**m, "n": len(results), "n_compared": 0}
    r = np.array(rows, dtype=float)
    m.update(
        n=len(results), n_compared=len(rows),
        blind_true_eta_min=float(r[:, 0].mean() / 60), blind_exposure=float(r[:, 1].mean()),
        aware_eta_min=float(r[:, 3].mean() / 60), aware_exposure=float(r[:, 4].mean()),
        fast_eta_min=float(r[:, 5].mean() / 60), fast_exposure=float(r[:, 6].mean()),
    )  # fmt: skip
    return m


def _demo(
    scenario: str = "elreno-2019",
    scenarios_dir: str = "scenarios",
    stack: str = "",
    slot: int = 3,
    lam: float = 1.0,
) -> dict[str, Any]:
    from outrun.app.backend import DefaultBackend
    from outrun.core.policy import Profile

    be = DefaultBackend(Path(scenarios_dir))
    sc = json.loads((Path(scenarios_dir) / scenario / "scenario.json").read_text())
    graph, _stack, edge_risk, node_risk = be._load(scenario, stack)
    model = be._cost_model((scenario, stack, "ev", lam), graph, edge_risk, node_risk, Profile.EV, lam)
    stations, hospitals = load_facility_points(tuple(sc["bbox"]))
    t0 = _time.time()
    risk = node_risk[:, :, slot].max(axis=1)  # incidents: highest-risk road nodes in the slot, 15 km apart
    picks: list[int] = []
    for n in np.argsort(-risk):
        if all(math.hypot(graph.x_m[n] - graph.x_m[p], graph.y_m[n] - graph.y_m[p]) > 15_000 for p in picks):
            picks.append(int(n))
        if len(picks) == 3:
            break
    inc = np.c_[graph.lon[picks], graph.lat[picks]]
    G, blocked = build_ev_graph(graph, model, slot)
    out = plan_dispatch(
        graph, model, stations, hospitals, inc, slot * 900.0, lam=lam, with_pareto=True, ev_graph=(G, blocked)
    )
    out["incident_node_risk"] = risk[picks].tolist()
    # aggregate over 40 random incidents on the edge of the tornado footprint (risk 0.2-0.5, just below the block)
    rng = np.random.default_rng(0)
    tornado = node_risk[:, 1, slot]
    pool = np.flatnonzero((tornado >= 0.2) & (tornado < 0.5))
    if len(pool) < 40:
        pool = np.argsort(-risk)[:400]  # no tornado footprint in this stack: the riskiest nodes instead
    sn = snap_nodes(graph, stations).tolist()
    out["aggregate"] = compare_routing(
        graph, G, blocked, sn, rng.choice(pool, 40, replace=False).tolist(), lam
    )
    out["n_stations"], out["n_hospitals"], out["seconds"] = len(stations), len(hospitals), _time.time() - t0
    return out


if __name__ == "__main__":
    import sys

    stack, slot = (sys.argv[1] if len(sys.argv) > 1 else ""), (int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    res = _demo(stack=stack, slot=slot)
    print(
        json.dumps({k: v for k, v in res.items() if k not in ("routes", "baseline")}, indent=1, default=float)
    )
