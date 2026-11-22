"""T24 drivers that need real data: routing evaluation (EVAL-14), group metrics (EVAL-22), latency (EVAL-21).

``outrun eval --routing`` and ``outrun eval --split test --final`` call these; the per-hazard test evaluators live
in ``outrun.hazards.<hazard>.final_eval``. Nothing here trains or tunes: it reads the frozen scenario bundles.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from outrun.eval.report import (
    time_calls,
    write_group_metrics,
    write_latency,
    write_routing_metrics,
)
from outrun.eval.routing_eval import (
    PLANNER_ORDER,
    PlannerKind,
    RoutingEvalResult,
    evaluate_planners,
    sample_origins,
)


def git_sha(repo: Path | None = None) -> str:
    """Current commit sha plus ``-dirty`` if the working tree has uncommitted changes."""
    cwd = repo or Path(__file__).resolve().parents[3]
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd, capture_output=True, text=True
    ).stdout.strip()
    return f"{sha}-dirty" if sha and dirty else sha


def _population_origins(graph: Any, n: int, seed: int) -> np.ndarray:
    """``n`` population-weighted residential origins: census block groups snapped to the road graph's largest
    strongly connected component, drawn with probability proportional to population."""
    from scipy.spatial import cKDTree

    from outrun.config import get_paths
    from outrun.data.census import read_pl94171_block_groups
    from outrun.optimize.okc_shelters import OKC_BBOX, _largest_scc_nodes
    from outrun.roads.graph import _project

    bg = read_pl94171_block_groups(get_paths().raw / "census_2020_pl94171_ok" / "ok2020.pl.zip")
    lon0, lat0, lon1, lat1 = OKC_BBOX
    bg = bg[bg["lon"].between(lon0, lon1) & bg["lat"].between(lat0, lat1) & (bg["population"] > 0)]
    scc = _largest_scc_nodes(graph)
    tree = cKDTree(np.c_[graph.x_m[scc], graph.y_m[scc]])
    x, y = _project(bg["lon"].to_numpy(float), bg["lat"].to_numpy(float))
    nodes = scc[tree.query(np.c_[x, y])[1]]
    return sample_origins(nodes, bg["population"].to_numpy(float), n=n, seed=seed)


def run_routing(
    scenarios_dir: Path,
    scenario_id: str,
    out_dir: Path,
    sha: str,
    n_origins: int = 500,
    primary: bool = False,
    n_resamples: int = 1000,
    seed: int = 0,
    cap_check_n: int = 50,
) -> tuple[RoutingEvalResult, dict[str, Any]]:
    """Five planners (a, a', b, c, d) on ``scenario_id``: every route planned on its own risk source, re-driven and
    scored on the OBSERVED stack. Writes ``routing_<id>.json``. Extras (A* label-cap check) are returned."""
    from outrun.app.backend import DefaultBackend
    from outrun.config import load_config
    from outrun.core.policy import Profile
    from outrun.core.risk import RiskSource
    from outrun.routing.astar import astar
    from outrun.routing.planner import replay_trace, transform_risk

    be = DefaultBackend(scenarios_dir)
    graph, _stack, edge, node = be._load(scenario_id, "")
    _, _, base_edge, base_node = be._load(scenario_id, "_baseline")
    _, _, obs_edge, obs_node = be._load(scenario_id, "_observed")
    profile, lam = Profile("civilian"), float(load_config("hazards")["lambda_default"])
    sources = {
        PlannerKind.NO_HAZARD: (edge, node, RiskSource.NONE),
        PlannerKind.CURRENT_ONLY: (edge, node, RiskSource.CURRENT_ONLY),
        PlannerKind.BASELINE: (base_edge, base_node, RiskSource.BASELINE),
        PlannerKind.MODEL: (edge, node, RiskSource.MODEL),
        PlannerKind.ORACLE: (obs_edge, obs_node, RiskSource.OBSERVED),
    }
    models = {}
    for kind, (e, n, src) in sources.items():
        models[kind] = be._cost_model(
            (scenario_id, "eval", kind.value),
            graph,
            transform_risk(e, src, 0),
            transform_risk(n, src, 0),
            profile,
            lam,
        )
    observed = models[PlannerKind.ORACLE]
    goals = be._shelter_nodes(graph, scenario_id)
    origins = _population_origins(graph, n_origins, seed)
    approx = {k: 0 for k in PLANNER_ORDER}

    def plan(kind: PlannerKind, origin: int):
        route = astar(graph, models[kind], int(origin), goals, 0.0)
        if route is None:
            return None
        approx[kind] += bool(route.approximate)
        return replay_trace(route, observed)

    result = evaluate_planners(
        [int(o) for o in origins],
        plan,
        observed.combined_risk_table(),
        observed.blocked_table(),
        predicted_risk={k: models[k].combined_risk_table() for k in PLANNER_ORDER},
        crossing_edges=np.flatnonzero(np.asarray(graph.crossing)).tolist(),
        n_resamples=n_resamples,
        seed=seed,
        observed_node_risk=observed.combined_node_risk_table(graph.n_nodes),
    )
    write_routing_metrics(out_dir, scenario_id, result, sha, primary=primary)
    extra = {"approximate_routes": {k.value: v for k, v in approx.items()}, "n_goals": len(goals)}
    extra["label_cap_check"] = _label_cap_check(
        graph, models[PlannerKind.MODEL], observed, goals, origins[:cap_check_n]
    )
    return result, extra


def _label_cap_check(graph, model, observed, goals, origins, big_cap: int = 64) -> dict[str, float]:
    """DECISION (T5): report how much the A* label cap (default 8, routes flagged ``approximate``) matters. Re-plan
    planner (c) for ``origins`` with ``max_labels=big_cap`` and compare observed exposure and travel time."""
    from outrun.eval.metrics import score_route
    from outrun.routing.astar import astar
    from outrun.routing.planner import replay_trace

    rows = []
    for o in origins:
        small = astar(graph, model, int(o), goals, 0.0)
        big = astar(graph, model, int(o), goals, 0.0, max_labels=big_cap)
        if small is None or big is None:
            continue
        sc = [
            score_route(replay_trace(r, observed), observed.combined_risk_table(), observed.blocked_table(),
                        node_risk=observed.combined_node_risk_table(graph.n_nodes))
            for r in (small, big)
        ]  # fmt: skip
        rows.append(
            (small.approximate, sc[0].exposure_risk_min, sc[1].exposure_risk_min, small.cost, big.cost)
        )
    if not rows:
        return {"n": 0}
    r = np.array(rows, dtype=float)
    return {
        "n": len(rows), "big_cap": big_cap, "frac_capped_routes_approximate": float(r[:, 0].mean()),
        "mean_observed_exposure_cap8": float(r[:, 1].mean()), f"mean_observed_exposure_cap{big_cap}": float(r[:, 2].mean()),
        "mean_cost_cap8": float(r[:, 3].mean()), f"mean_cost_cap{big_cap}": float(r[:, 4].mean()),
    }  # fmt: skip


SEEDS = (0, 1, 2, 3, 4)
SA_ITERATIONS = (
    20000  # same as the scenario bundles (app.build.build_scenario), so seed 0 reproduces the app's plan
)


def _shelter_metrics() -> dict[str, dict[str, Any]]:
    """mcf / nearest / greedy metrics of the T22 shelter comparison (static free-flow times, the same file the app
    shows): ``PROC/evac/okc_shelters.json``, built by ``app.build._shelter_collections``."""
    from outrun.config import get_paths

    path = Path(get_paths().proc) / "evac" / "okc_shelters.json"
    cache = json.loads(path.read_text())
    return {
        m: {k: (float("nan") if v is None else v) for k, v in fc["metrics"].items()}
        for m, fc in cache.items()
    }


def run_group(
    scenarios_dir: Path, scenario_id: str, out_dir: Path, sha: str, primary: bool = False, seeds=SEEDS
) -> tuple[Path, Any]:
    """Shelters, whole-city evacuation (SA vs MSA/shortest/safest/lambda_E=0, one SA per seed) and emergency-vehicle
    dispatch on the real OKC graph, written as ``group_<id>.json``. Returns the path and the SA run-time stats."""
    import pickle

    from outrun.config import get_paths
    from outrun.eval.report import LatencyStats
    from outrun.optimize.evacuation import (
        EvacEvaluator,
        EvacInstance,
        baseline_plan,
        evac_metrics,
        msa_equilibrium,
        simulated_annealing,
    )

    oe = pickle.loads((Path(get_paths().proc) / "evac" / f"{scenario_id}.pkl").read_bytes())
    inst = oe.instance
    ev = EvacEvaluator(inst)
    ev0 = EvacEvaluator(EvacInstance(**{**inst.__dict__, "lambda_E": 0.0}))
    lower_bound = float(ev.lower_bound())
    evac: dict[str, list[dict[str, float]]] = {}
    for name in ("shortest", "safest"):
        evac[name] = [evac_metrics(ev, baseline_plan(ev, name))] * len(seeds)
    evac["msa"] = [evac_metrics(ev, msa_equilibrium(ev))] * len(
        seeds
    )  # deterministic: same plan for every seed
    evac["sa"], evac["sa_lambda0"], sa_times = [], [], []
    for seed in seeds:
        t = time.perf_counter()
        run = simulated_annealing(ev, iterations=SA_ITERATIONS, seed=seed, zone_adjacency=oe.zone_adjacency)
        sa_times.append(time.perf_counter() - t)
        evac["sa"].append(evac_metrics(ev, run.best_plan))
        run0 = simulated_annealing(ev0, iterations=SA_ITERATIONS, seed=seed, zone_adjacency=oe.zone_adjacency)
        evac["sa_lambda0"].append(evac_metrics(ev, run0.best_plan))  # scored with the real lambda_E evaluator
    sa_stats = LatencyStats(
        len(sa_times), float(np.median(sa_times)), float(np.percentile(sa_times, 95)), float(np.max(sa_times))
    )
    path = write_group_metrics(
        out_dir, scenario_id, shelters=_shelter_metrics(), evacuation=evac, lower_bound=lower_bound,
        ems=_ems_metrics(scenarios_dir, scenario_id), git_sha=sha, seeds=list(seeds), primary=primary,
    )  # fmt: skip
    write_latency(out_dir, "sa_run", sa_stats, sha)
    return path, sa_stats


def _ems_metrics(
    scenarios_dir: Path, scenario_id: str, n_incidents: int = 36, seed: int = 0
) -> dict[str, Any]:
    """Hungarian risk-aware dispatch vs the straight-line-nearest + discover-and-reroute baseline (EMS-03/05/07),
    over ``n_incidents`` incidents on the riskiest road nodes of the model stack, dispatched 3 at a time. Both are
    scored on the model stack's blocked set (the one the app uses); the aware route plans around blocks, the
    baseline discovers them. Pareto: EMS-06 sweep for the first dispatched incident."""
    from outrun.app.backend import DefaultBackend
    from outrun.core.policy import Profile
    from outrun.routing.emergency import RerouteResult, ems_metrics
    from outrun.routing.okc_ems import build_ev_graph, load_facility_points, plan_dispatch

    be = DefaultBackend(scenarios_dir)
    graph, _stack, edge, node = be._load(scenario_id, "")
    bbox = tuple(json.loads((scenarios_dir / scenario_id / "scenario.json").read_text())["bbox"])
    bundled = be._facilities(scenario_id, bbox)
    if bundled is not None:
        stations, hospitals = bundled["fire_station"], bundled.get("hospital", np.zeros((0, 2)))
    else:
        stations, hospitals = load_facility_points(bbox)
    model = be._cost_model((scenario_id, "", "ev", 1.0), graph, edge, node, Profile.EV, 1.0)
    risk = node.max(axis=1)  # (N, slots)
    slot = int(np.argmax(risk.max(axis=0)))
    pool = np.argsort(-risk[:, slot])[: max(400, 4 * n_incidents)]
    picks = np.random.default_rng(seed).choice(pool, n_incidents, replace=False)
    G, blocked = build_ev_graph(graph, model, slot)
    rows: list[dict[str, Any]] = []
    for b in range(0, n_incidents, 3):
        inc = np.c_[graph.lon[picks[b : b + 3]], graph.lat[picks[b : b + 3]]]
        res = plan_dispatch(
            graph, model, stations, hospitals, inc, slot * 900.0, with_pareto=(b == 0), ev_graph=(G, blocked)
        )
        rows += res["comparison"]
    aware = [
        RerouteResult([], r["eta_min"] * 60.0, r["eta_min"] * 60.0, 0.0, False, 0, True)
        if r.get("eta_min") is not None
        else RerouteResult([], float("inf"), float("inf"), float("inf"), False, 0, False)
        for r in rows
    ]
    aware_exp = [r.get("exposure") for r in rows]
    base = []
    for r in rows:
        eta = r.get("baseline_eta_min")
        if eta is None:
            base.append(RerouteResult([], float("inf"), float("inf"), float("inf"), False, 0, False))
        else:
            base.append(
                RerouteResult([], eta * 60.0, r["baseline_planned_eta_min"] * 60.0, r["baseline_detour_min"] * 60.0,
                              bool(r["baseline_hit_block"]), int(bool(r["baseline_hit_block"])), True)
            )  # fmt: skip
    base_exp = [r.get("baseline_exposure") for r in rows]
    pareto = [
        {"lam": p["lam"], "eta_min": p["time_s"] / 60.0, "exposure": p["exposure"]}
        for r in rows
        if r.get("pareto")
        for p in r["pareto"]
    ]
    ok_a = all(x is not None for x in aware_exp)
    ok_b = all(x is not None for x in base_exp)
    return {
        "hungarian": ems_metrics(aware, aware_exp if ok_a else None),
        "reroute_baseline": ems_metrics(base, base_exp if ok_b else None),
        "pareto": pareto[: len(pareto)],
        "n_incidents": n_incidents,
        "slot": slot,
    }


def run_latency(scenarios_dir: Path, out_dir: Path, sha: str, scenario_id: str) -> Any:
    """AC-12: 20 A* queries on the real OKC metro graph, timed after 3 warm-up queries (``time_calls``)."""
    from outrun.app.backend import DefaultBackend
    from outrun.core.policy import Profile
    from outrun.routing.astar import astar

    be = DefaultBackend(scenarios_dir)
    graph, _s, edge, node = be._load(scenario_id, "")
    model = be._cost_model((scenario_id, "latency"), graph, edge, node, Profile("civilian"), 1.0)
    goals = be._shelter_nodes(graph, scenario_id)
    origins = [int(o) for o in _population_origins(graph, 20, seed=12345)]
    stats = time_calls(lambda o: astar(graph, model, o, goals, 0.0), origins, warmup=3)
    write_latency(out_dir, "astar_metro_query", stats, sha)
    return stats
