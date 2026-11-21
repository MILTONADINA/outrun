"""Real OKC evacuation instance (T19): shelter destinations from T11, risk-aware candidate paths on the OKC road graph,
hazard exposure from a scenario's RiskStack. ``python -m outrun.optimize.okc_evacuation`` builds it and compares
all-shortest / all-safest / MSA / SA under the one wave-BPR evaluator (spec 06, DESIGN 4.5).
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from outrun.config import get_paths
from outrun.optimize.evacuation import (
    EvacEvaluator,
    EvacInstance,
    EvacPlan,
    baseline_plan,
    diverse_paths,
    evac_metrics,
    make_packets,
    msa_equilibrium,
    simulated_annealing,
)

METHODS = ("shortest", "safest", "msa", "sa")


@dataclass
class OkcEvac:
    """An evacuation instance plus what the app and the report need to draw it."""

    instance: EvacInstance
    zone_adjacency: list[tuple[int, int]]
    packet_zone_xy: np.ndarray  # (P, 2) lon, lat of each packet's origin
    used_edges: np.ndarray  # instance edge i is graph edge used_edges[i]
    build_s: float


def cache_path(scenario: str) -> Path:
    return Path(get_paths().proc) / "evac" / f"{scenario}.pkl"


def build_evac(
    scenario_dir: str | Path,
    radius_km: float = 8.0,
    packet_max: int = 100,
    k_paths: int | None = None,
    min_shelters: int = 10_000,
    profile: str = "civilian",
    log=print,
) -> OkcEvac:
    """Zones within ``radius_km`` of the centre -> packets -> mcf shelter -> k diverse risk-aware paths."""
    from scipy.spatial import cKDTree

    from outrun.core.policy import Profile
    from outrun.core.risk import RiskStack
    from outrun.optimize.okc_shelters import _largest_scc_nodes, build_okc_instance
    from outrun.optimize.shelters import solve_min_cost_flow
    from outrun.roads.graph import load_npz
    from outrun.routing.cost import build_cost_model, node_path_to_edges, to_td_digraph

    t_start = time.perf_counter()
    scenario_dir = Path(scenario_dir)
    graph = load_npz(scenario_dir / "graph.npz")
    stack = RiskStack.from_zarr(scenario_dir / "risk.zarr")
    cm = build_cost_model(graph, stack, Profile(profile))
    log(f"cost model: {cm.n_edges} edges, {cm.n_slots} slots ({time.perf_counter() - t_start:.0f}s)")

    shel = build_okc_instance(radius_km, min_shelters=min_shelters)
    flow = solve_min_cost_flow(shel.demand, shel.capacity, shel.cost_s).flow
    log(f"shelters: {len(shel.sites)} sites, {len(shel.demand)} zones ({time.perf_counter() - t_start:.0f}s)")

    scc = _largest_scc_nodes(graph)
    tree = cKDTree(np.c_[graph.x_m[scc], graph.y_m[scc]])
    zone_node = scc[tree.query(shel.zone_xy)[1]]

    G = to_td_digraph(graph, cm)
    cost_tab = cm.cost_table()
    scalar = np.where(np.isfinite(cost_tab), cost_tab, np.nan)
    scalar = np.nan_to_num(np.nanmean(scalar, axis=1), nan=1e6)  # blocked-all-slots edges stay very expensive
    for _, _, d in G.edges(data=True):
        d["evac_cost"] = float(scalar[d["eid"]])
    log(f"digraph ready ({time.perf_counter() - t_start:.0f}s)")

    packets = make_packets(shel.population, packet_max=packet_max)
    veh_per_zone = np.zeros(len(shel.population))
    for p in packets:
        veh_per_zone[p.zone] += p.vehicles
    remaining = flow / np.maximum(flow.sum(axis=1, keepdims=True), 1) * veh_per_zone[:, None]
    nearest = np.where(np.isfinite(shel.cost_s), shel.cost_s, np.inf).argmin(axis=1)
    paths_cache: dict[tuple[int, int], list[np.ndarray]] = {}
    candidates, vehicles, zones = [], [], []
    for p in packets:
        z = p.zone
        s = int(remaining[z].argmax()) if remaining[z].max() > 0 else int(nearest[z])
        remaining[z, s] -= p.vehicles
        key = (int(zone_node[z]), int(shel.sites[s].node))
        if key not in paths_cache:
            nodes = diverse_paths(G, key[0], key[1], k=k_paths, weight="evac_cost")
            paths_cache[key] = [node_path_to_edges(G, n) for n in nodes]
        candidates.append(paths_cache[key])
        vehicles.append(p.vehicles)
        zones.append(z)
    log(
        f"{len(packets)} packets, {len(paths_cache)} (zone, shelter) pairs ({time.perf_counter() - t_start:.0f}s)"
    )

    used = np.unique(np.concatenate([c for cs in candidates for c in cs] or [np.zeros(0, dtype=np.int64)]))
    remap = np.full(graph.n_edges, -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    inst = EvacInstance(
        edge_t0=np.asarray(graph.free_flow_s)[used],
        edge_cap=np.asarray(graph.capacity_vph)[used],
        edge_risk=cm.weighted_risk_table()[used],
        vehicles=np.asarray(vehicles, dtype=np.int64),
        candidates=[[remap[c] for c in cs] for cs in candidates],
        slot_s=float(cm.slot_s),
        zones=np.asarray(zones),
    )
    pairs = sorted(tuple(sorted(map(int, pr))) for pr in cKDTree(shel.zone_xy).query_pairs(r=2000.0))
    lon, lat = graph.lon[zone_node], graph.lat[zone_node]
    pkt_xy = np.c_[lon[zones], lat[zones]]
    return OkcEvac(inst, pairs, pkt_xy, used, time.perf_counter() - t_start)


def run_methods(ev: EvacEvaluator, adjacency, iterations: int, restarts: int = 1, callback=None) -> dict:
    """Score every method with the one evaluator: ``{method: {"plan", "metrics", "runtime_s"}}``."""
    out: dict[str, dict] = {}

    def record(name: str, plan: EvacPlan, runtime: float, extra: dict | None = None) -> None:
        out[name] = {"plan": plan, "metrics": evac_metrics(ev, plan), "runtime_s": runtime, **(extra or {})}

    for name in ("shortest", "safest"):
        t = time.perf_counter()
        record(name, baseline_plan(ev, name), time.perf_counter() - t)
    t = time.perf_counter()
    record("msa", msa_equilibrium(ev), time.perf_counter() - t)
    t = time.perf_counter()
    runs = [
        simulated_annealing(
            ev,
            iterations=iterations,
            seed=s,
            zone_adjacency=adjacency,
            callback=callback if s == 0 else None,
            callback_every=max(1, iterations // 200),
        )  # fmt: skip
        for s in range(restarts)
    ]
    best = min(runs, key=lambda r: r.best_J)
    record(
        "sa",
        best.best_plan,
        time.perf_counter() - t,
        {"restart_J": [r.best_J for r in runs], "history": runs[0].history},
    )
    return out


def result_geojson(oe: OkcEvac, plan: EvacPlan, metrics: dict) -> dict:
    feats = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
            "properties": {"packet": i, "vehicles": int(oe.instance.vehicles[i]), "k": int(plan.k[i]), "wave": int(plan.w[i])},
        }
        for i, (x, y) in enumerate(oe.packet_zone_xy)
    ]  # fmt: skip
    return {"type": "FeatureCollection", "features": feats, "metrics": metrics}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="elreno-2019")
    ap.add_argument("--radius-km", type=float, default=8.0)
    ap.add_argument("--packet-max", type=int, default=100)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--write-results", action="store_true", help="write scenarios/<id>/results/evac_*.json")
    a = ap.parse_args()
    scen = Path("scenarios") / a.scenario
    cp = cache_path(a.scenario)
    if cp.is_file() and not a.rebuild:
        oe = pickle.loads(cp.read_bytes())
    else:
        oe = build_evac(scen, a.radius_km, a.packet_max)
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(pickle.dumps(oe))
    inst = oe.instance
    ev = EvacEvaluator(inst)
    print(
        f"instance: {len(inst.vehicles)} packets, {int(inst.vehicles.sum())} vehicles, {inst.edge_t0.size} edges, "
        f"build {oe.build_s:.0f}s, LB {ev.lower_bound():.1f} veh-h"
    )
    iters = a.iterations if a.iterations is not None else 40000
    res = run_methods(ev, oe.zone_adjacency, iters, a.restarts)
    abl = EvacEvaluator(EvacInstance(**{**inst.__dict__, "lambda_E": 0.0}))
    t = time.perf_counter()
    sa0 = simulated_annealing(abl, iterations=iters, seed=0, zone_adjacency=oe.zone_adjacency)
    m0 = evac_metrics(ev, sa0.best_plan)  # scored with the real lambda_E evaluator
    res["sa_lambda0"] = {"plan": sa0.best_plan, "metrics": m0, "runtime_s": time.perf_counter() - t}
    print(
        f"{'method':<11}{'clear_min':>10}{'veh-h':>10}{'exposure':>10}{'max v/c':>9}{'unserved':>9}{'J':>12}{'gap%':>8}{'runtime s':>10}"
    )
    for name, r in res.items():
        m = r["metrics"]
        print(
            f"{name:<11}{m['clearance_min']:>10.1f}{m['vehicle_hours']:>10.1f}{m['exposure_veh_h']:>10.2f}"
            f"{m['max_vc']:>9.2f}{m['unserved_veh']:>9.0f}{m['J']:>12.1f}{m['gap_to_lb_pct']:>8.1f}{r['runtime_s']:>10.1f}"
        )
    if a.write_results:
        out = scen / "results"
        out.mkdir(exist_ok=True)
        for name in METHODS:
            (out / f"evac_{name}.json").write_text(
                json.dumps(result_geojson(oe, res[name]["plan"], res[name]["metrics"]))
            )
        trace = [
            {"iter": i, "objective": cur, "temperature": temp} for i, cur, _, temp in res["sa"]["history"]
        ]
        (out / "evac_sa_trace.json").write_text(json.dumps(trace))


if __name__ == "__main__":
    main()
