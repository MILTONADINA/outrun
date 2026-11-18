"""Command-line interface: ``outrun fetch|prep|train|eval|scenario|route|serve|export-static|disk``
(spec 14: CLI-01..CLI-08; DESIGN 2.2).

``python -m outrun`` and the ``outrun`` console script both reach ``main`` through ``outrun.__main__``.
Hazard and profile choices are derived from ``outrun.core.risk.Hazard`` / ``outrun.core.policy.Profile``
(``name.lower()``), never duplicated as literals. Heavy modules are imported inside the handlers only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from outrun import __version__
from outrun.core.policy import Profile
from outrun.core.risk import RISK_SOURCES, Hazard

COMMANDS: tuple[str, ...] = (
    "fetch",
    "prep",
    "train",
    "eval",
    "scenario",
    "route",
    "serve",
    "export-static",
    "disk",
)

GB = 1e9


@dataclass(frozen=True)
class DiskStatus:
    """Result of ``check_disk`` (CLI-08). Sizes in GB (1e9 bytes)."""

    path: Path
    total_gb: float
    used_gb: float
    free_gb: float
    need_gb: float
    max_used_gb: float
    ok: bool


HAZARD_NAMES: tuple[str, ...] = tuple(h.name.lower() for h in Hazard)
PROFILE_NAMES: tuple[str, ...] = tuple(p.name.lower() for p in Profile)


def build_parser() -> argparse.ArgumentParser:
    """Parser with prog ``outrun``, ``--version`` and the nine sub-commands; options and defaults per CLI-02,
    route argument groups per CLI-05. ``args.command`` is the sub-command name. (CLI-01, CLI-02, CLI-05)
    """
    parser = argparse.ArgumentParser(prog="outrun", description="Hazard-aware evacuation routing.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = sub.add_parser("fetch", help="download manifest sources to RAW")
    p.add_argument(
        "--source",
        dest="sources",
        action="append",
        default=[],
        metavar="NAME",
        help="source id (repeatable); default: all sources",
    )
    p.add_argument("--manifest", default="data/manifest.yaml")
    p.add_argument("--dry-run", action="store_true", help="list what would be fetched, download nothing")

    p = sub.add_parser("prep", help="preprocess one dataset to PROC")
    p.add_argument("dataset")
    p.add_argument("--limit", type=int, default=None)

    p = sub.add_parser("train", help="train one hazard model")
    p.add_argument("hazard", choices=HAZARD_NAMES)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("eval", help="evaluate on val (or, once, on test with --final)")
    p.add_argument("--hazard", choices=(*HAZARD_NAMES, "all"), default="all")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--routing", action="store_true")
    p.add_argument("--final", action="store_true")

    p = sub.add_parser("scenario", help="precompute scenario bundles")
    p.add_argument("scenario_id", nargs="?", default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--out", default="scenarios")

    p = sub.add_parser("route", help="plan a risk-aware route on any US place or bbox")
    area = p.add_mutually_exclusive_group(required=True)
    area.add_argument(
        "--place",
        default=None,
        help='e.g. "Tulsa, Oklahoma, USA" (city limits from OSM; the first run downloads, later runs use the cache)',
    )
    area.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("W", "S", "E", "N"),
        help="lon/lat box; the cached OKC metro graph is --bbox -98.1 35.1 -97.1 35.8",
    )
    p.add_argument("--origin", type=float, nargs=2, required=True, metavar=("LON", "LAT"))
    goal = p.add_mutually_exclusive_group(required=True)
    goal.add_argument("--dest", type=float, nargs=2, default=None, metavar=("LON", "LAT"))
    goal.add_argument("--nearest-shelter", action="store_true")
    p.add_argument("--profile", choices=PROFILE_NAMES, default="civilian")
    p.add_argument("--risk-source", dest="risk_source", choices=RISK_SOURCES, default="model")
    p.add_argument("--lambda", dest="lam", type=float, default=None, help="safer <-> faster multiplier")
    p.add_argument("--scenario", default=None)
    p.add_argument("--out", default=None, help="write both routes as GeoJSON here")
    p.add_argument(
        "--demo-storm",
        action="store_true",
        help="ignore --risk-source and put a synthetic storm between origin and destination",
    )
    p.add_argument(
        "--shelters",
        default=None,
        help="table (csv/parquet) with lon/lat columns for --nearest-shelter; "
        "default PROC/facilities/shelter.parquet",
    )
    p.add_argument("--refresh", action="store_true", help="rebuild the cached road graph")

    p = sub.add_parser("serve", help="run the web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--demo", action="store_true")
    p.add_argument("--scenarios", default="scenarios")
    p.add_argument("--reports", default="reports")

    p = sub.add_parser("export-static", help="export the static demo site")
    p.add_argument("--out", default="docs")
    p.add_argument("--scenarios", default="scenarios")
    p.add_argument("--reports", default="reports")

    p = sub.add_parser("disk", help="check RAW and PROC disk budgets")
    p.add_argument("--need-gb", type=float, default=0.0)
    return parser


def check_disk(path: Path, need_gb: float, max_used_gb: float) -> DiskStatus:
    """Disk check from ``shutil.disk_usage(path)``: ``ok = used + need <= max_used and need <= free``.
    ``ValueError`` if ``need_gb < 0``; ``FileNotFoundError`` if ``path`` is missing. (CLI-08)
    """
    if need_gb < 0:
        raise ValueError(f"need_gb must be >= 0, got {need_gb}")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    usage = shutil.disk_usage(path)
    total, used, free = usage.total / GB, usage.used / GB, usage.free / GB
    ok = (used + need_gb <= max_used_gb) and (need_gb <= free)
    return DiskStatus(path, total, used, free, float(need_gb), float(max_used_gb), ok)


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download manifest sources (delegates to ``outrun.data.fetch``)."""
    from outrun.config import get_paths
    from outrun.data import fetch
    from outrun.data.manifest import load_manifest

    manifest = load_manifest(args.manifest)
    sources = [manifest.get(s) for s in args.sources] if args.sources else list(manifest.sources)
    sources = [s for s in sources if s.target != "stream"]  # streamed sources are never stored raw
    raw = get_paths().raw
    if args.dry_run:
        for s in sources:
            print(f"{s.id}: {s.url} (~{s.approx_bytes / GB:.1f} GB)")
        return 0
    # Batched state writes: some sources have tens of thousands of files. flush() in finally keeps every record.
    state = fetch.FetchState(raw / "fetch_state.json", save_every_s=30)
    try:
        for s in sources:
            print(f"fetching {s.id} ...", flush=True)
            results = fetch.fetch_source(s, raw, fetch.default_transport(s), state, fetch.default_lister(s))
            print(f"{s.id}: {len(results)} file(s) downloaded", flush=True)
    finally:
        state.flush()
    return 0


def cmd_prep(args: argparse.Namespace) -> int:
    """Preprocess one dataset to PROC (delegates to ``outrun.data.<dataset>``)."""
    raise NotImplementedError("prep handlers arrive with the dataset tasks (T8+)")


def cmd_train(args: argparse.Namespace) -> int:
    """Train one hazard model (delegates to ``outrun.hazards.base`` training loop)."""
    raise NotImplementedError("training entry points arrive with the hazard tasks (T9+)")


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate on val (default) or, with ``--final``, the one-time test run (delegates to ``outrun.eval``)."""
    raise NotImplementedError("evaluation arrives with the eval tasks")


def cmd_scenario(args: argparse.Namespace) -> int:
    """Precompute scenario bundles (layout in specs/14-app.md) into ``args.out``."""
    raise NotImplementedError("scenario bundles arrive with the app tasks")


def cmd_route(args: argparse.Namespace) -> int:
    """Plan one safe route on a place or bbox graph (delegates to ``outrun.routing.planner``; ROAD-06)."""
    from outrun.config import get_paths
    from outrun.roads.features import snap_to_nodes
    from outrun.roads.graph import build_graph
    from outrun.routing.astar import astar
    from outrun.routing.cost import build_cost_model

    if args.scenario is not None:
        raise NotImplementedError("--scenario needs scenario bundles (later task)")

    # 1. Road graph: built from OSM once, then loaded from the PROC cache (ROAD-06).
    bbox = tuple(args.bbox) if args.bbox is not None else None
    graph = build_graph(args.place, bbox, cache_dir=get_paths().proc / "roads", refresh=args.refresh)
    print(f"graph {graph.meta.get('source', '')}: {graph.n_nodes} nodes, {graph.n_edges} edges")

    # 2. Snap origin and goal(s) to graph nodes.
    origin = int(snap_to_nodes(graph, args.origin[0], args.origin[1])[0][0])
    if args.dest is not None:
        goals = [int(snap_to_nodes(graph, args.dest[0], args.dest[1])[0][0])]
    else:
        goals = _shelter_nodes(graph, args.shelters)

    # 3. Risk on the graph's window, then the cost model for the chosen profile.
    stack = _route_risk_stack(graph, args, goals[0], origin)
    profile = Profile[args.profile.upper()]
    lam = 1.0 if args.lam is None else args.lam
    cost_model = build_cost_model(graph, stack, profile, lam=lam)

    # 4. Two routes: risk-aware A* and plain shortest time (same model with all risk set to zero).
    safe = astar(graph, cost_model, origin, goals)
    zero_model = cost_model.with_risk(np.zeros((cost_model.n_edges, len(Hazard), cost_model.n_slots)))
    fast = astar(graph, zero_model, origin, goals)
    if safe is None and fast is None:
        print("no route: the goal is not reachable", file=sys.stderr)
        return 1

    _print_route("risk-aware", safe, graph, cost_model)
    _print_route("shortest  ", fast, graph, cost_model)
    if args.out:
        _write_routes_geojson(Path(args.out), graph, {"risk_aware": safe, "shortest": fast})
        print(f"wrote {args.out}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """``outrun.app.server.create_app(scenarios_dir=..., reports_dir=..., demo=args.demo or None)`` then
    ``uvicorn.run(app, host=args.host, port=args.port)``; returns 0. (CLI-06)
    """
    import uvicorn

    from outrun.app import server

    app = server.create_app(
        scenarios_dir=Path(args.scenarios), reports_dir=Path(args.reports), demo=args.demo or None
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_export_static(args: argparse.Namespace) -> int:
    """``outrun.app.export.export_static(scenarios, reports, out)``; prints the file count, returns 0;
    ``FileExistsError`` -> message on stderr, returns 1. (CLI-07)
    """
    from outrun.app import export

    try:
        files = export.export_static(Path(args.scenarios), Path(args.reports), Path(args.out))
    except FileExistsError as e:
        print(f"export-static: {e}", file=sys.stderr)
        return 1
    print(f"wrote {len(files)} files to {args.out}")
    return 0


def cmd_disk(args: argparse.Namespace) -> int:
    """Check RAW (HDD limit) and PROC (SSD limit) with ``check_disk``; 0 if both ok else 1. (CLI-08)"""
    from outrun.config import get_paths
    from outrun.data.disk import load_disk_budget

    paths, budget = get_paths(), load_disk_budget()
    all_ok = True
    for name, path, cap_bytes in (
        ("RAW", paths.raw, budget.hdd_cap_bytes),
        ("PROC", paths.proc, budget.ssd_cap_bytes),
    ):
        st = check_disk(path, args.need_gb, cap_bytes / GB)
        print(
            f"{name} {st.path}: used {st.used_gb:.1f} GB + need {st.need_gb:.1f} GB "
            f"(cap {st.max_used_gb:.0f} GB, free {st.free_gb:.1f} GB) -> {'ok' if st.ok else 'NOT OK'}"
        )
        all_ok = all_ok and st.ok
    return 0 if all_ok else 1


# ---------------------------------------------------------------- route helpers
def _shelter_nodes(graph, shelters: str | None) -> list[int]:
    """Graph nodes of every shelter inside the graph (for ``--nearest-shelter``)."""
    import pandas as pd

    from outrun.config import get_paths
    from outrun.roads.features import snap_to_nodes

    path = Path(shelters) if shelters else get_paths().proc / "facilities" / "shelter.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no shelter table at {path}; pass --shelters or use --dest")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    inside = df["lon"].between(graph.lon.min(), graph.lon.max()) & df["lat"].between(
        graph.lat.min(), graph.lat.max()
    )
    if not inside.any():
        raise ValueError(f"no shelters from {path} inside the graph")
    nodes, _ = snap_to_nodes(graph, df.loc[inside, "lon"].to_numpy(), df.loc[inside, "lat"].to_numpy())
    return sorted(set(int(n) for n in nodes))


def _route_risk_stack(graph, args: argparse.Namespace, goal: int, origin: int):
    """RiskStack on a window around the graph.

    ``none`` -> all zeros. ``--demo-storm`` -> one synthetic storm cell sitting halfway between origin and goal
    (useful before any model is trained). Otherwise the hazard adapters build the stack (RISK-08).
    """
    from outrun.core.grid import bbox_to_window, lonlat_to_xy
    from outrun.core.risk import RiskSource, RiskStack, build_stack
    from outrun.core.synthetic import moving_gaussian

    graph_bbox = (
        float(graph.lon.min()),
        float(graph.lat.min()),
        float(graph.lon.max()),
        float(graph.lat.max()),
    )
    window = bbox_to_window(graph_bbox, buffer_km=2.0)
    t0 = datetime.now(UTC).replace(second=0, microsecond=0)

    if args.demo_storm:
        mid_lon = (graph.lon[origin] + graph.lon[goal]) / 2
        mid_lat = (graph.lat[origin] + graph.lat[goal]) / 2
        mx, my = lonlat_to_xy(mid_lon, mid_lat)
        row = float(
            (np.asarray(my) - window.y_centers()[0]) / (window.y_centers()[1] - window.y_centers()[0])
        )
        col = float(
            (np.asarray(mx) - window.x_centers()[0]) / (window.x_centers()[1] - window.x_centers()[0])
        )
        storm = moving_gaussian(12, window.shape, start=(row, col), velocity=(0.0, 0.0), sigma=2.0, peak=0.9)
        data = RiskStack.zeros(t0, window).data.copy()
        data[list(Hazard).index(Hazard.STORM)] = storm
        return RiskStack(data, t0, window)

    source = RiskSource(args.risk_source)
    if source is RiskSource.NONE:
        return RiskStack.zeros(t0, window)
    from outrun.core.scenario import Scenario
    from outrun.hazards.base import load_default_adapters

    scn = Scenario(
        id="cli-route",
        t0=t0,
        bbox=graph_bbox,
        city=str(graph.meta.get("source", "")),
        hazards=tuple(Hazard),
        window=window,
    )
    registry = load_default_adapters()
    adapters = [registry.get(h) for h in registry.hazards()]
    # current-only freezes at the CostModel level (PLAN-02); the stack itself is the model forecast.
    stack_source = RiskSource.MODEL if source is RiskSource.CURRENT_ONLY else source
    return build_stack(adapters, scn, stack_source)


def _print_route(label: str, route, graph, cost_model) -> None:
    """One summary line: travel time, length and the route's cost under the risk-aware model."""
    if route is None:
        print(f"{label}: no route")
        return
    km = float(graph.length_m[route.edges].sum()) / 1000 if route.edges else 0.0
    # Re-drive the route under the risk-aware model, so both routes are timed and priced the same way
    # (the shortest route was planned on a zero-risk model, whose clock ignores storm slow-downs).
    t, risk_cost = route.start_s, 0.0
    for e, wait in zip(route.edges, route.waits_s, strict=True):
        t += wait
        risk_cost += cost_model.cost(e, t)
        t += cost_model.clock(e, t)
    travel_s = t - route.start_s
    print(
        f"{label}: {travel_s / 60:6.1f} min, {km:6.2f} km, {len(route.edges)} edges, "
        f"risk-aware cost {risk_cost / 60:.1f} min-equivalent"
    )


def _write_routes_geojson(path: Path, graph, routes: dict) -> None:
    """GeoJSON FeatureCollection with one LineString (node lon/lat) per route."""
    features = [
        {
            "type": "Feature",
            "properties": {"name": name, "travel_s": r.travel_s, "cost": r.cost},
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(graph.lon[n]), float(graph.lat[n])] for n in r.nodes],
            },
        }
        for name, r in routes.items()
        if r is not None
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "fetch": cmd_fetch,
    "prep": cmd_prep,
    "train": cmd_train,
    "eval": cmd_eval,
    "scenario": cmd_scenario,
    "route": cmd_route,
    "serve": cmd_serve,
    "export-static": cmd_export_static,
    "disk": cmd_disk,
}


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` (default ``sys.argv[1:]``), apply the cross-argument checks (CLI-04 test-split guard,
    CLI-05 bbox order; violations exit with status 2 before any handler runs), then call
    ``HANDLERS[args.command](args)`` (looked up at call time) and return its int. (CLI-03, CLI-04, CLI-05)
    """
    parser = build_parser()
    args = parser.parse_args(argv)  # argv None -> sys.argv[1:]
    if args.command == "eval" and args.split == "test" and not args.final:
        parser.error("--split test is the one-time final run; add --final (DESIGN 6.3)")  # exits 2
    if args.command == "route" and args.bbox is not None:
        w, s, e, n = args.bbox
        if w >= e or s >= n:
            parser.error(f"--bbox needs W < E and S < N, got {args.bbox}")
    return int(HANDLERS[args.command](args))
