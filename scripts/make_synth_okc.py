"""Write the demo scenario bundle ``scenarios/synth-okc`` (spec 14 bundle layout) for the app and the static export.

A synthetic storm cell sweeps across the OKC metro from the north to the south while a trip runs west -> east, so on
the real OKC road graph the risk-aware route differs from the shortest one. Offline; reads only the cached OKC graph.

    .venv/bin/python scripts/make_synth_okc.py [--out scenarios] [--graph PATH]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from outrun.config import get_paths
from outrun.core.grid import bbox_to_window, lonlat_to_xy
from outrun.core.policy import Profile
from outrun.core.risk import Hazard, RiskStack
from outrun.core.synthetic import crossing_storm
from outrun.roads.features import snap_to_nodes
from outrun.roads.graph import load_npz
from outrun.routing.astar import astar
from outrun.routing.cost import build_cost_model

BBOX = (-98.1, 35.1, -97.1, 35.8)  # the cached OKC metro graph
ORIGIN = (-97.75, 35.47)  # sample trip: west OKC -> east OKC, across the storm path
DEST = (-97.30, 35.47)
T0 = datetime(2019, 5, 26, 3, 0, tzinfo=UTC)
SID = "synth-okc"


def synth_stack(window, origin_lonlat, dest_lonlat, t0=T0, hazard=Hazard.STORM) -> RiskStack:
    """RiskStack with one moving storm cell crossing the origin -> destination line (one hazard, rest zero)."""
    (ox, oy), (dx, dy) = (lonlat_to_xy(*p) for p in (origin_lonlat, dest_lonlat))
    dy_m = window.y_centers()[1] - window.y_centers()[0]
    dx_m = window.x_centers()[1] - window.x_centers()[0]

    def to_rc(x, y):
        return float((y - window.y_centers()[0]) / dy_m), float((x - window.x_centers()[0]) / dx_m)

    (r0, c0), (r1, c1) = to_rc(ox, oy), to_rc(dx, dy)
    cube = crossing_storm(
        12,
        window.shape,
        ((r0 + r1) / 2, (c0 + c1) / 2),
        (r1 - r0, c1 - c0),
        hit_slot=1,
        cells_per_slot=3.0,
        sigma=5.0,
    )
    data = RiskStack.zeros(t0, window).data.copy()
    data[list(Hazard).index(hazard)] = cube
    return RiskStack(data, t0, window)


def _summarise(graph, route, model) -> dict:
    t, exposure = route.start_s, 0.0
    risk = model.weighted_risk_table()
    for e, wait in zip(route.edges, route.waits_s, strict=True):
        t += wait
        exposure += float(risk[e][model.slot(t)])
        t += model.clock(e, t)
    return {
        "travel_min": (t - route.start_s) / 60,
        "km": float(graph.length_m[route.edges].sum()) / 1000,
        "exposure": exposure,
    }


def _feature(graph, route, props) -> dict:
    coords = [[float(graph.lon[n]), float(graph.lat[n])] for n in route.nodes]
    return {"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": coords}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="scenarios")
    ap.add_argument("--graph", default=str(get_paths().proc / "roads" / "okc.npz"))
    args = ap.parse_args()

    graph = load_npz(args.graph)
    window = bbox_to_window(BBOX, buffer_km=2.0)
    stack = synth_stack(window, ORIGIN, DEST)
    baseline = RiskStack.zeros(T0, window)

    folder = Path(args.out) / SID
    for source, st in (("model", stack), ("baseline", baseline)):
        for slot in range(stack.n_slots):
            p = folder / "overlays" / source / "storm" / f"{slot:02d}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            st.overlay_png(Hazard.STORM, slot, p)

    nodes, _ = snap_to_nodes(graph, [ORIGIN[0], DEST[0]], [ORIGIN[1], DEST[1]])
    a, b = int(nodes[0]), int(nodes[1])
    model = build_cost_model(graph, stack, Profile.CIVILIAN)
    shortest = astar(graph, build_cost_model(graph, baseline, Profile.CIVILIAN), a, b)
    safe = astar(graph, model, a, b)
    s_short, s_safe = _summarise(graph, shortest, model), _summarise(graph, safe, model)
    result = {
        "outrun": _feature(graph, safe, {"kind": "route", "facility": None}),
        "shortest": _feature(graph, shortest, {"kind": "shortest"}),
        "eta_min": s_safe["travel_min"],
        "exposure": {"predicted": s_safe["exposure"], "observed": None},
        "shelter_in_place": False,
        "blocked_avoided": 0,
        "summary": {"shortest": s_short, "outrun": s_safe},
    }
    res = folder / "results"
    res.mkdir(parents=True, exist_ok=True)
    preset = {"origin": list(ORIGIN), "dest": list(DEST), "profile": "civilian", "result": result}
    (res / "routes_preset.json").write_text(json.dumps([preset]))

    meta = {
        "id": SID,
        "title": "Synthetic moving storm, Oklahoma City",
        "t0": T0.isoformat(),
        "city": "Oklahoma City, OK",
        "bbox": list(BBOX),
        "hazards": ["storm"],
        "held_out": False,
        "synthetic": True,
        "sources": ["synthetic moving Gaussian storm", "OSM roads (cached OKC graph)"],
        "n_slots": stack.n_slots,
        "slot_minutes": 15,
    }
    (folder / "scenario.json").write_text(json.dumps(meta, indent=2))

    # Live /api/route inputs (DefaultBackend._load): graph.npz + risk.zarr. Both are local-only (.gitignore).
    graph_link = folder / "graph.npz"
    if graph_link.is_symlink() or graph_link.exists():
        graph_link.unlink()
    try:
        graph_link.symlink_to(Path(args.graph).resolve())
    except OSError:
        shutil.copy2(args.graph, graph_link)
    zarr_path = folder / "risk.zarr"
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    stack.to_zarr(zarr_path)
    print(f"wrote {folder}")
    for name, s in (("shortest", s_short), ("risk-aware", s_safe)):
        print(f"{name:10s} {s['travel_min']:.1f} min  {s['km']:.1f} km  exposure {s['exposure']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
