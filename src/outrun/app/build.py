"""Scenario bundle builder behind ``outrun scenario`` (T22; layout in specs/14-app.md).

A bundle is self-contained: a clean clone (no RAW, no PROC, no ``models/``) serves it from the committed files alone.

    scenarios/_graphs/<city>.npz              shared road graph (OSM, ODbL), one per city
    scenarios/_graphs/<city>.facilities.json  HIFLD fire/EMS stations, hospitals, schools (columnar lon/lat/kind)
    scenarios/<id>/scenario.json              metadata plus ``"graph": "<city>"``
    scenarios/<id>/overlays/<source>/<hazard>/<slot>.png
    scenarios/<id>/risk[_baseline|_observed].zarr   the RiskStacks the live backend samples
    scenarios/<id>/results/                   routes_preset, shelters_*, evac_*, dispatch_preset JSON

Order matters: graph, facilities and risk stacks are written first; every preset, shelter, evacuation and dispatch
result is then computed through the finished bundle (``DefaultBackend`` / the OKC optimizers), so the cached numbers
are the live numbers. Heavy imports stay inside functions (``outrun.cli`` imports this module lazily).
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

OKC_BBOX = (-98.1, 35.1, -97.1, 35.8)
DIXIE_BBOX = (-121.45, 39.90, -120.70, 40.55)
GRAPH_FILES = {"okc": "okc.npz", "greenville": "greenville.npz"}
GRAPH_BBOX = {"okc": OKC_BBOX, "greenville": (-121.5, 39.88, -120.65, 40.58)}
N_SLOTS = 12


@dataclass(frozen=True)
class ScenarioSpec:
    """One scenario. ``layers()`` returns ``({source: [RiskLayer]}, [source notes])``; sources are model/baseline/observed."""

    id: str
    title: str
    t0: datetime
    bbox: tuple[float, float, float, float]
    city: str
    hazards: tuple[str, ...]
    graph: str
    held_out: bool
    layers: Callable[[], tuple[dict, list[str]]]
    synthetic: bool = False
    extras: tuple[str, ...] = ("routes",)  # subset of routes, shelters, evac, ems
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- paths and shared files
def _paths():
    from outrun.config import get_paths

    return get_paths()


def write_graph_files(out: Path, graph: str, force: bool = False) -> None:
    """``_graphs/<graph>.npz`` (copy of PROC/roads) and ``<graph>.facilities.json`` (HIFLD, from RAW)."""
    folder = out / "_graphs"
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / f"{graph}.npz"
    if force or not dst.exists():
        shutil.copy2(_paths().proc / "roads" / GRAPH_FILES[graph], dst)
    fac = folder / f"{graph}.facilities.json"
    if force or not fac.exists():
        from outrun.data.facilities import load_hifld_facilities

        df = load_hifld_facilities(_paths().raw, GRAPH_BBOX[graph])
        df = df[df["kind"].isin(["fire_station", "hospital", "school"])]
        fac.write_text(
            json.dumps(
                {
                    "source": "HIFLD fire/EMS stations, hospitals, public and private schools",
                    "kind": df["kind"].tolist(),
                    "lon": [round(float(v), 5) for v in df["lon"]],
                    "lat": [round(float(v), 5) for v in df["lat"]],
                },
                separators=(",", ":"),
            )
        )


# ---------------------------------------------------------------- layer makers
def _window(bbox):
    from outrun.core.grid import bbox_to_window

    return bbox_to_window(bbox, buffer_km=2.0)


def _okc_graph():
    from outrun.roads.graph import load_npz

    return load_npz(_paths().proc / "roads" / "okc.npz")


def layers_synth() -> tuple[dict, list[str]]:
    from outrun.core.grid import lonlat_to_xy
    from outrun.core.risk import Hazard, RiskStack
    from outrun.core.synthetic import crossing_storm

    t0 = SPECS["synth-okc"].t0
    window = _window(OKC_BBOX)
    (ox, oy), (dx, dy) = (lonlat_to_xy(*p) for p in ((-97.75, 35.47), (-97.30, 35.47)))
    yc, xc = window.y_centers(), window.x_centers()

    def rc(x, y):
        return float((y - yc[0]) / (yc[1] - yc[0])), float((x - xc[0]) / (xc[1] - xc[0]))

    (r0, c0), (r1, c1) = rc(ox, oy), rc(dx, dy)
    cube = crossing_storm(
        N_SLOTS, window.shape, ((r0 + r1) / 2, (c0 + c1) / 2), (r1 - r0, c1 - c0),
        hit_slot=1, cells_per_slot=3.0, sigma=5.0,
    )  # fmt: skip
    data = RiskStack.zeros(t0, window).data.copy()
    data[list(Hazard).index(Hazard.STORM)] = cube
    stack = RiskStack(data, t0, window)
    return {"model": stack, "baseline": RiskStack.zeros(t0, window)}, [
        "synthetic moving Gaussian storm (model), no hazard (baseline)",
        "OSM roads (cached OKC graph)",
    ]


def _layers_to_stacks(spec_t0, bbox, groups: dict) -> dict:
    from outrun.core.risk import RiskStack

    return {name: RiskStack.from_layers(group, spec_t0, _window(bbox)) for name, group in groups.items()}


def layers_storm(sevir_id: str, t0: datetime, mode: str, tornado: bool, sid: str, tornado_tag: str):
    def make():
        from outrun.core.risk import Hazard
        from outrun.core.scenario import Scenario
        from outrun.hazards.storm.provider import ExtrapolationStormAdapter, real_provider, storm_adapter

        hz = (Hazard.STORM, Hazard.TORNADO) if tornado else (Hazard.STORM,)
        scn = Scenario(
            id=sid, t0=t0, bbox=OKC_BBOX, city="Oklahoma City, OK", hazards=hz,
            held_out={h: True for h in hz}, case={"sevir_id": sevir_id}, window=_window(OKC_BBOX),
        )  # fmt: skip
        provider = real_provider(mode)
        model_ad = storm_adapter(scn, mode, provider)
        base_ad = ExtrapolationStormAdapter(provider)
        groups = {"model": [model_ad.predict(scn)], "baseline": [base_ad.baseline(scn)]}
        notes = ["SEVIR VIL (NEXRAD) 2 km, regridded to EPSG:5070 1 km"]
        if tornado:
            from outrun.hazards.tornado.adapter import TornadoAdapter
            from outrun.hazards.tornado.provider import real_provider as tornado_provider

            tor = TornadoAdapter(
                provider=tornado_provider(),
                checkpoint=_paths().models / "tornado" / "best.pt",
                source_tag=tornado_tag,
            )
            groups["model"].append(tor.predict(scn))
            groups["baseline"].append(tor.baseline(scn))
            notes.append(
                "tornado: TorNet chips from NEXRAD sites within the box (ending <= t0+35 min, demo timing slack), swath along storm-provider motion"
            )
        obs = model_ad.observed(scn)
        if obs is not None:
            groups["observed"] = [obs]
        return groups, notes + ["OSM roads (cached OKC graph)"]

    return make


def layers_dixie():
    from outrun.core.risk import Hazard
    from outrun.core.scenario import Scenario
    from outrun.hazards.wildfire.adapter import WildfireAdapter
    from outrun.hazards.wildfire.provider import DEMO, wsts_provider

    scn = Scenario(
        id="dixie-2021", t0=SPECS["dixie-2021"].t0, bbox=DIXIE_BBOX, city="Greenville / Plumas County, CA",
        hazards=(Hazard.WILDFIRE,), held_out={Hazard.WILDFIRE: True},
        case={"wsts_fire": DEMO.fire, "day0": DEMO.day0.isoformat()}, window=_window(DIXIE_BBOX),
    )  # fmt: skip
    ad = WildfireAdapter(
        provider=wsts_provider(),
        checkpoint=_paths().models / "wildfire" / "best.pt",
        source_tag="wildfire.unet@dixie",
    )
    return {"model": [ad.predict(scn)], "baseline": [ad.baseline(scn)], "observed": [ad.observed(scn)]}, [
        "WildfireSpreadTS 2021 (VIIRS 375 m + GRIDMET), mapped to the 12 NDWS channels at 1 km; held-out test year, demo render only (not scored)",
        "model: calibrated fire U-Net day-1 probability grown hourly from the day-0 VIIRS mask (t0 = day 1 12:00 UTC)",
        "OSM roads (Geofabrik norcal extract, Greenville CA graph)",
    ]


def layers_okfire():
    from outrun.core.risk import Hazard
    from outrun.core.scenario import Scenario
    from outrun.hazards.wildfire.provider import hms_replay_layers, read_hms_fire_points

    scn = Scenario(
        id="ok-fire-2025", t0=SPECS["ok-fire-2025"].t0, bbox=OKC_BBOX, city="Oklahoma City, OK",
        hazards=(Hazard.WILDFIRE,), held_out={Hazard.WILDFIRE: True},
        case={"replay": True, "hms_day": "2025-03-14"}, window=_window(OKC_BBOX),
    )  # fmt: skip
    rep = hms_replay_layers(scn, read_hms_fire_points([date(2025, 3, 14), date(2025, 3, 15)]))
    return {"model": [rep["model"]], "observed": [rep["observed"]]}, [
        "REPLAY of observed NOAA HMS fire-detection points (GOES-East FDC, VIIRS NOAA-20/21, S-NPP), 2025-03-14; not a model forecast and not model-validated",
        "model layer = persistence of the detections in the 3 h before t0 (20:00 UTC) with a 1-cell uncertainty ring; observed = detections after t0",
        "OSM roads (cached OKC graph)",
    ]


def _okc_scn(sid: str, hazard, case: dict):
    from outrun.core.scenario import Scenario

    s = SPECS[sid]
    return Scenario(
        id=sid, t0=s.t0, bbox=OKC_BBOX, city="Oklahoma City, OK", hazards=(hazard,), held_out={hazard: True},
        case=case, window=_window(OKC_BBOX),
    )  # fmt: skip


def layers_flood():
    from outrun.core.risk import Hazard
    from outrun.hazards.flood.adapter import FloodAdapter
    from outrun.hazards.flood.provider import real_provider

    scn = _okc_scn("flood-okc-2025", Hazard.FLOOD, {"storm_events_episode": 203955})
    ad = FloodAdapter(
        provider=real_provider(), checkpoint=_paths().models / "flood" / "best.pt", graph=_okc_graph(),
        source_tag="flood.unet@2025-07-12",
    )  # fmt: skip
    groups = {"model": [ad.predict(scn)], "baseline": [ad.baseline(scn)]}
    obs = ad.observed(scn)
    if obs is not None:
        groups["observed"] = [obs]
    return groups, [
        "MRMS QPE (Pass2) 1 h/3 h/6 h/24 h accumulations + FLASH FFG/ARI ratios on the 0.01 degree grid, ETOPO terrain; U-Net features as in training",
        "model: calibrated flash-flood U-Net; baseline: FFG-ratio rule; observed: NWS Storm Events flash-flood reports (5 km discs)",
        "official TEST year (2025): demo render only, no metric computed; issue hour = hour of peak window-mean 1 h QPE",
        "low-water crossings and bridges from the OKC OSM graph carry per-edge flood overrides",
        "OSM roads (cached OKC graph)",
    ]


def layers_ice():
    from outrun.core.risk import Hazard
    from outrun.hazards.ice.adapter import IceAdapter
    from outrun.hazards.ice.provider import real_provider

    scn = _okc_scn("ice-okc-2025", Hazard.ICE, {"storm_events_episode": 200487})
    ad = IceAdapter(
        provider=real_provider(), checkpoint=_paths().models / "ice" / "best.pt", graph=_okc_graph(),
        source_tag="ice.milcnn@2025-02-18",
    )  # fmt: skip
    groups = {"model": [ad.predict(scn)], "baseline": [ad.baseline(scn)]}
    obs = ad.observed(scn)
    if obs is not None:
        groups["observed"] = [obs]
    return groups, [
        "HRRR f01 surface fields (t and t-3 h) + ETOPO elevation, 3 km; NWS forecast zones touching the OKC box",
        "model: ice MIL-CNN calibrated on val; baseline: physics rule (freezing precipitation at T<=0 C); observed: labelled Storm Events zones",
        "official TEST winter (2024-25): demo render only, no metric computed; cycle = the 3-hourly HRRR cycle of 2025-02-17..19 with the highest forecast probability (chosen from the model output, not from labels)",
        "bridge prior (x1.5) applied to bridge edges",
        "OSM roads (cached OKC graph)",
    ]


def layers_smoke():
    from outrun.core.risk import Hazard
    from outrun.hazards.smoke.adapter import SmokeAdapter

    scn = _okc_scn("smoke-okc-2024", Hazard.SMOKE, {"aqs_day": "2024-04-05"})
    ad = SmokeAdapter(checkpoint=_paths().models / "smoke" / "best.pt", source_tag="smoke.cnnmlp@2024-04-05")
    groups = {"model": [ad.predict(scn)], "baseline": [ad.baseline(scn)]}
    obs = ad.observed(scn)
    if obs is not None:
        groups["observed"] = [obs]
    return groups, [
        "HRRR-Smoke near-surface mass density (9 km store) + EPA AQS PM2.5 monitors + NOAA HMS smoke polygons",
        "model: CNN+MLP PM2.5 median; baseline: calibrated raw HRRR-Smoke; observed: AQS monitors, inverse-distance weighted, at t0 + 6 h",
        "official TEST year (2024): demo render only, no metric computed; top Oklahoma AQS day 2024-04-05; cycle = the 6-hourly cycle of 04-04..06 with the highest forecast PM2.5 (20.7 ug/m3), which stays below the 35.4 ug/m3 risk threshold, so the model and baseline risk layers are empty over OKC (honest result, thresholds not changed)",
        "OSM roads (cached OKC graph)",
    ]


SPECS: dict[str, ScenarioSpec] = {}


def _register(*specs: ScenarioSpec) -> None:
    SPECS.update({s.id: s for s in specs})


_register(
    ScenarioSpec(
        "synth-okc",
        "Synthetic moving storm, Oklahoma City",
        datetime(2019, 5, 26, 3, 0, tzinfo=UTC),
        OKC_BBOX,
        "Oklahoma City, OK",
        ("storm",),
        "okc",
        False,
        layers_synth,
        synthetic=True,
        extras=("routes", "shelters", "ems"),
    ),
    ScenarioSpec(
        "elreno-2019",
        "El Reno EF3, 2019-05-26 (SEVIR S830307)",
        datetime(2019, 5, 26, 3, 0, tzinfo=UTC),
        OKC_BBOX,
        "El Reno / Oklahoma City, OK",
        ("storm", "tornado"),
        "okc",
        True,
        layers_storm(
            "S830307",
            datetime(2019, 5, 26, 3, 0, tzinfo=UTC),
            "auto",
            True,
            "elreno-2019",
            "tornado.cnn@elreno",
        ),
        extras=("routes", "shelters", "evac", "ems"),
    ),
    ScenarioSpec(
        "may20-2019",
        "Oklahoma tornado outbreak, 2019-05-20 (SEVIR S830835, primary case)",
        datetime(2019, 5, 20, 20, 30, tzinfo=UTC),
        OKC_BBOX,
        "Oklahoma City, OK",
        ("storm", "tornado"),
        "okc",
        True,
        layers_storm(
            "S830835",
            datetime(2019, 5, 20, 20, 30, tzinfo=UTC),
            "auto",
            True,
            "may20-2019",
            "tornado.cnn@may20",
        ),
        extras=("routes", "shelters", "evac", "ems"),
    ),
    ScenarioSpec(
        "flood-okc-2025",
        "Flash flood, Oklahoma City metro, 2025-07-12 (test year, demo render)",
        datetime(2025, 7, 12, 21, 0, tzinfo=UTC),
        OKC_BBOX,
        "Oklahoma City, OK",
        ("flood",),
        "okc",
        True,
        layers_flood,
        extras=("routes", "shelters", "evac", "ems"),
    ),
    ScenarioSpec(
        "ice-okc-2025",
        "Ice storm, Oklahoma City metro, 2025-02-18 (test winter, demo render)",
        datetime(2025, 2, 18, 15, 0, tzinfo=UTC),
        OKC_BBOX,
        "Oklahoma City, OK",
        ("ice",),
        "okc",
        True,
        layers_ice,
        extras=("routes", "shelters", "ems"),
    ),
    ScenarioSpec(
        "smoke-okc-2024",
        "Smoke, Oklahoma City metro, 2024-04-05 (test year, demo render)",
        datetime(2024, 4, 5, 6, 0, tzinfo=UTC),
        OKC_BBOX,
        "Oklahoma City, OK",
        ("smoke",),
        "okc",
        True,
        layers_smoke,
        extras=("routes", "shelters", "ems"),
    ),
    ScenarioSpec(
        "dixie-2021",
        "Dixie Fire, 2021-08-06 (WildfireSpreadTS 2021 demo)",
        datetime(2021, 8, 6, 12, 0, tzinfo=UTC),
        DIXIE_BBOX,
        "Greenville / Plumas County, CA",
        ("wildfire",),
        "greenville",
        True,
        layers_dixie,
        extras=("routes", "ems"),
    ),
    ScenarioSpec(
        "ok-fire-2025",
        "REPLAY: Oklahoma wildfires, 2025-03-14 (HMS detections, no model)",
        datetime(2025, 3, 14, 20, 0, tzinfo=UTC),
        OKC_BBOX,
        "Oklahoma City, OK",
        ("wildfire",),
        "okc",
        True,
        layers_okfire,
        extras=("routes", "shelters", "ems"),
    ),
)


# ---------------------------------------------------------------- bundle writing
def _layer_stack(group, spec: ScenarioSpec):
    from outrun.core.risk import RiskStack

    if isinstance(group, RiskStack):
        return group, [], []
    stack = RiskStack.from_layers(group, spec.t0, _window(spec.bbox))
    return stack, [ly.hazard for ly in group], [f"{ly.source} ({ly.kind})" for ly in group]


def write_layers(out: Path, spec: ScenarioSpec, log=print) -> Path:
    """Overlays, zarr stacks and ``scenario.json`` of one scenario."""
    folder = out / spec.id
    for sub in ("overlays", "results"):
        if (folder / sub).exists():
            shutil.rmtree(folder / sub)
    for z in folder.glob("risk*.zarr"):
        shutil.rmtree(z)
    groups, notes = spec.layers()
    sources: list[str] = []
    for name, group in groups.items():
        stack, hazards, labels = _layer_stack(group, spec)
        if not hazards:
            from outrun.core.risk import Hazard

            hazards = [Hazard(h) for h in spec.hazards]
            labels = [f"synthetic.{name}"]
        for h in hazards:
            for slot in range(stack.n_slots):
                p = folder / "overlays" / name / h.value / f"{slot:02d}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                stack.overlay_png(h, slot, p)
            mx = max(float(stack.get(h, k).max()) for k in range(stack.n_slots))
            log(f"  {name:9s} {h.value:9s} max p {mx:.3f}")
        sources += [f"{name}: {s}" for s in labels]
        stack.to_zarr(folder / ("risk.zarr" if name == "model" else f"risk_{name}.zarr"))
    meta = {
        "id": spec.id, "title": spec.title, "t0": spec.t0.isoformat(), "city": spec.city, "bbox": list(spec.bbox),
        "hazards": list(spec.hazards), "held_out": spec.held_out, "synthetic": spec.synthetic,
        "sources": sources + notes, "n_slots": N_SLOTS, "slot_minutes": 15, "graph": spec.graph,
    }  # fmt: skip
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results").mkdir(exist_ok=True)
    (folder / "scenario.json").write_text(json.dumps(meta, indent=2))
    for stale in (folder / "graph.npz",):  # shared graph replaces per-bundle copies and symlinks
        if stale.is_symlink() or stale.exists():
            stale.unlink()
    return folder


# ---------------------------------------------------------------- routes
def _line_to_origin_dest(graph, risk_node, rng, n_candidates: int, lo_km: float, hi_km: float):
    """Candidate (origin, dest) lon/lat pairs: half start inside the hazard footprint, half anywhere; 'hot' first."""
    from outrun.routing.okc_ems import snap_nodes

    n = graph.n_nodes
    hot = np.flatnonzero(risk_node.max(axis=(1, 2)) > 0.2)
    pairs = []
    for i in range(n_candidates * 4):
        a = int(rng.choice(hot)) if (len(hot) and i % 2 == 0) else int(rng.integers(n))
        b = int(rng.integers(n))
        d = np.hypot(graph.x_m[a] - graph.x_m[b], graph.y_m[a] - graph.y_m[b]) / 1000
        if lo_km <= d <= hi_km:
            pairs.append((a, b))
        if len(pairs) == n_candidates:
            break
    nodes = snap_nodes(graph, np.array([[graph.lon[v], graph.lat[v]] for p in pairs for v in p]))
    return [
        ((round(float(graph.lon[nodes[2 * k]]), 4), round(float(graph.lat[nodes[2 * k]]), 4)),
         (round(float(graph.lon[nodes[2 * k + 1]]), 4), round(float(graph.lat[nodes[2 * k + 1]]), 4)))
        for k in range(len(pairs))
    ]  # fmt: skip


def write_routes(out: Path, spec: ScenarioSpec, n_pairs: int = 10, n_shelter: int = 6, log=print) -> int:
    """results/routes_preset.json through the live backend: ``n_pairs`` coordinate pairs x (civilian, ev) plus
    ``n_shelter`` nearest-shelter origins. Pairs are the candidates where the risk-aware route differs most."""
    from outrun.app.backend import DefaultBackend
    from outrun.app.models import RouteRequest

    be = DefaultBackend(out)
    graph, _stack, _edge, node_risk = be._load(spec.id, "")
    rng = np.random.default_rng(0)
    big = spec.graph == "okc"
    cands = _line_to_origin_dest(graph, node_risk, rng, 48, 12 if big else 6, 60 if big else 25)
    scored = []
    for origin, dest in cands:
        try:
            r = be.route(RouteRequest(scenario=spec.id, origin=origin, dest=dest, profile="civilian"))
        except Exception:  # noqa: BLE001 - unreachable pair: skip
            continue
        if r["outrun"] is None and not r["shelter_in_place"]:
            continue
        s_ex = (r["shortest"] or {}).get("properties", {}).get("exposure") or 0.0
        o_ex = (r["exposure"] or {}).get("predicted") or 0.0
        scored.append(
            (
                float(s_ex)
                - float(o_ex)
                + 5.0 * r["blocked_avoided"]
                + (50.0 if r["shelter_in_place"] else 0.0),
                origin,
                dest,
            )
        )
    scored.sort(key=lambda t: -t[0])
    chosen = [(o, d) for _, o, d in scored[:n_pairs]]
    presets = []
    for origin, dest in chosen:
        for profile in ("civilian", "ev"):
            res = be.route(RouteRequest(scenario=spec.id, origin=origin, dest=dest, profile=profile))
            presets.append({"origin": list(origin), "dest": list(dest), "profile": profile, "result": res})
    if "shelters" in spec.extras:
        for origin, _ in chosen[:n_shelter]:
            try:
                res = be.route(
                    RouteRequest(scenario=spec.id, origin=origin, dest="nearest_shelter", profile="civilian")
                )
            except Exception as exc:  # noqa: BLE001
                log(f"  nearest_shelter from {origin}: {exc}")
                continue
            presets.append(
                {"origin": list(origin), "dest": "nearest_shelter", "profile": "civilian", "result": res}
            )
    (out / spec.id / "results" / "routes_preset.json").write_text(json.dumps(presets, separators=(",", ":")))
    log(f"  routes_preset: {len(presets)} presets from {len(scored)} routable candidates")
    return len(presets)


# ---------------------------------------------------------------- shelters
def _shelter_collections() -> dict[str, dict]:
    """GeoJSON for mcf / nearest / greedy on the OKC block groups (static free-flow times, like T11)."""
    from pyproj import Transformer

    from outrun.optimize.okc_shelters import build_okc_instance, compare
    from outrun.optimize.shelters import (
        assign_greedy,
        assign_nearest,
        assignment_metrics,
        solve_min_cost_flow,
    )

    inst = build_okc_instance()
    rows = {r["method"]: r for r in compare(inst)}
    to_ll = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    zlon, zlat = to_ll.transform(inst.zone_xy[:, 0], inst.zone_xy[:, 1])
    slon, slat = to_ll.transform(inst.site_xy[:, 0], inst.site_xy[:, 1])
    solvers = {
        "mcf": lambda: solve_min_cost_flow(inst.demand, inst.capacity, inst.cost_s),
        "nearest": lambda: assign_nearest(
            inst.demand, inst.capacity, inst.zone_xy, inst.site_xy, inst.cost_s
        ),
        "greedy": lambda: assign_greedy(inst.demand, inst.capacity, inst.cost_s),
    }
    out = {}
    for method, solve in solvers.items():
        a = solve()
        m = assignment_metrics(a, inst.capacity, inst.cost_s)
        load = a.flow.sum(axis=0)
        feats = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(float(slon[j]), 5), round(float(slat[j]), 5)],
                },
                "properties": {
                    "id": s.id,
                    "kind": str(s.kind),
                    "capacity": int(s.capacity),
                    "load": int(load[j]),
                    "overflow": int(max(0, load[j] - s.capacity)),
                    "assumed": bool(s.assumed),
                },
            }
            for j, s in enumerate(inst.sites)
            if load[j] > 0
        ]
        zi, sj = np.nonzero(a.flow)
        feats += [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [round(float(zlon[z]), 5), round(float(zlat[z]), 5)],
                        [round(float(slon[s]), 5), round(float(slat[s]), 5)],
                    ],
                },
                "properties": {
                    "people": int(a.flow[z, s]),
                    "minutes": round(float(inst.cost_s[z, s]) / 60, 1),
                },
            }
            for z, s in zip(zi.tolist(), sj.tolist(), strict=True)
        ]
        metrics = {k: (None if isinstance(v, float) and v != v else v) for k, v in m.__dict__.items()}
        metrics["objective"] = rows[method]["objective"]
        out[method] = {"type": "FeatureCollection", "features": feats, "metrics": metrics}
    return out


_SHELTER_CACHE: dict[str, dict] = {}


def write_shelters(out: Path, spec: ScenarioSpec, log=print) -> None:
    if not _SHELTER_CACHE:
        cache = _paths().proc / "evac" / "okc_shelters.json"
        if cache.exists():
            _SHELTER_CACHE.update(json.loads(cache.read_text()))
        else:
            _SHELTER_CACHE.update(_shelter_collections())
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(_SHELTER_CACHE))
    for method, fc in _SHELTER_CACHE.items():
        (out / spec.id / "results" / f"shelters_{method}.json").write_text(
            json.dumps(fc, separators=(",", ":"))
        )
    served = ", ".join(f"{m} {d['metrics']['served_persons']} served" for m, d in _SHELTER_CACHE.items())
    log(f"  shelters: {served}")


# ---------------------------------------------------------------- evacuation
def write_evac(out: Path, spec: ScenarioSpec, iterations: int, log=print) -> None:
    import pickle

    from outrun.optimize.evacuation import EvacEvaluator
    from outrun.optimize.okc_evacuation import build_evac, cache_path, result_geojson, run_methods

    oe = build_evac(out / spec.id, log=lambda *a: None)
    cp = cache_path(spec.id)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_bytes(pickle.dumps(oe))
    res = run_methods(EvacEvaluator(oe.instance), oe.zone_adjacency, iterations)
    folder = out / spec.id / "results"
    for name in ("shortest", "safest", "msa", "sa"):
        (folder / f"evac_{name}.json").write_text(
            json.dumps(result_geojson(oe, res[name]["plan"], res[name]["metrics"]), separators=(",", ":"))
        )
    trace = [{"iter": i, "objective": cur, "temperature": temp} for i, cur, _, temp in res["sa"]["history"]]
    (folder / "evac_sa_trace.json").write_text(json.dumps(trace, separators=(",", ":")))
    log(
        "  evac: "
        + ", ".join(
            f"{k} J={v['metrics']['J']:.0f} clear={v['metrics']['clearance_min']:.0f}min"
            for k, v in res.items()
        )
    )


# ---------------------------------------------------------------- EMS dispatch presets
def write_dispatch(out: Path, spec: ScenarioSpec, log=print) -> int:
    """results/dispatch_preset.json: dispatch for 2 and 3 incidents placed on the riskiest road nodes of the best slot,
    computed through ``DefaultBackend.dispatch`` (the same code the live endpoint runs)."""
    from datetime import timedelta

    from outrun.app.backend import DefaultBackend
    from outrun.app.models import DispatchRequest

    be = DefaultBackend(out)
    graph, stack, _edge, node_risk = be._load(spec.id, "")
    risk = node_risk.max(axis=1)  # (N, slots)
    slot = int(np.argmax(risk.max(axis=0)))
    r = risk[:, slot]
    picks: list[int] = []
    for n in np.argsort(-r):
        if all(np.hypot(graph.x_m[n] - graph.x_m[p], graph.y_m[n] - graph.y_m[p]) > 6000 for p in picks):
            picks.append(int(n))
        if len(picks) == 3:
            break
    presets = []
    for k in (2, 3):
        inc = [[round(float(graph.lon[n]), 5), round(float(graph.lat[n]), 5)] for n in picks[:k]]
        t0 = stack.t0 + timedelta(minutes=15 * slot)
        try:
            res = be.dispatch(DispatchRequest(scenario=spec.id, incidents=inc, t0=t0))
        except LookupError as exc:
            log(f"  dispatch skipped: {exc}")
            return 0
        presets.append({"incidents": inc, "t0": t0.isoformat(), "result": {**res, "source": "cached"}})
    (out / spec.id / "results" / "dispatch_preset.json").write_text(
        json.dumps(presets, separators=(",", ":"))
    )
    log(f"  dispatch_preset: {len(presets)} (slot {slot})")
    return len(presets)


# ---------------------------------------------------------------- entry
def build_scenario(out: Path, scenario_id: str, evac_iterations: int = 20000, log=print) -> Path:
    spec = SPECS[scenario_id]
    t = time.perf_counter()
    log(f"[{scenario_id}] layers")
    write_graph_files(out, spec.graph)
    folder = write_layers(out, spec, log)
    if "shelters" in spec.extras:  # first: the nearest-shelter route presets read results/shelters_mcf.json
        write_shelters(out, spec, log)
    if "routes" in spec.extras:
        write_routes(out, spec, log=log)
    if "evac" in spec.extras:
        write_evac(out, spec, evac_iterations, log)
    if "ems" in spec.extras:
        write_dispatch(out, spec, log)
    log(f"[{scenario_id}] done in {time.perf_counter() - t:.0f}s")
    return folder
