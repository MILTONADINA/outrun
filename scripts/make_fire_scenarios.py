"""Write the two wildfire scenario bundles (T20) into ``scenarios/`` (spec 14 bundle layout).

``dixie-2021``  WSTS 2021 demo fire (Dixie Fire, day 0 = 2021-08-05) on the Greenville CA road graph. ``risk.zarr`` =
                calibrated fire U-Net grown hourly from the day-0 VIIRS mask, ``risk_baseline.zarr`` = downwind
                persistence, ``risk_observed.zarr`` = the next-day VIIRS mask. WSTS 2021 is the held-out test year:
                this is a demo render only, no metric is computed on it here.
``ok-fire-2025`` OK March 2025 replay on the OKC graph. NOAA HMS fire points (satellite detections of 2025-03-14) only,
                no weather model: ``risk.zarr`` = persistence of the detections of the 3 h before t0 (kind
                "baseline"), ``risk_observed.zarr`` = detections that followed. A labelled replay of observed
                detections, not a model validation.

    .venv/bin/python scripts/make_fire_scenarios.py {dixie,okfire} [--out scenarios] [--routes]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

from outrun.config import get_paths
from outrun.core.grid import bbox_to_window
from outrun.core.risk import Hazard, RiskStack
from outrun.core.scenario import Scenario
from outrun.hazards.wildfire.adapter import WildfireAdapter
from outrun.hazards.wildfire.provider import DEMO, hms_replay_layers, read_hms_fire_points, wsts_provider

DIXIE_BBOX = (-121.45, 39.90, -120.70, 40.55)
# t0: noon UTC (5 am PDT) on day 1 = 12 h after the end of day 0 (the day-0 mask is at its day-1 forecast weight
# 12/24 = 0.5, growing to 0.625 across the 3 h window). Chosen for a visible, mid-growth layer; no label was looked at.
DIXIE_T0 = datetime(2021, 8, 6, 12, 0, tzinfo=UTC)
OK_BBOX = (-98.10, 35.10, -97.10, 35.80)
# t0: 20:00 UTC (3 pm CDT) 2025-03-14, the height of the detections in the OKC box (HMS Time field, UTC).
OK_T0 = datetime(2025, 3, 14, 20, 0, tzinfo=UTC)

# Preset (origin, dest) lon/lat pairs for results/routes_preset.json, picked by scanning low-risk nodes around the fire
# layer for pairs the live router can connect; the last Dixie pair starts inside the fire (shelter in place).
PRESET_PAIRS: dict[str, list] = {
    "dixie-2021": [
        ((-120.9324, 39.9324), (-121.0431, 40.0568)),
        ((-120.6586, 40.4263), (-121.1511, 40.2614)),
        ((-120.7456, 39.9061), (-121.0508, 39.9510)),
        ((-120.9345, 39.9360), (-120.7925, 40.1640)),
        ((-120.9451, 40.1336), (-121.2076, 40.2377)),
    ],
    "ok-fire-2025": [
        ((-97.1547, 35.2735), (-97.2041, 35.1415)),
        ((-97.4997, 35.6360), (-97.7632, 35.1425)),
        ((-97.4803, 35.6644), (-97.7315, 35.2256)),
        ((-97.3387, 35.6175), (-97.4816, 35.6877)),
    ],
}


def dixie() -> tuple[Scenario, dict, list[str], Path]:
    scn = Scenario(
        id="dixie-2021", t0=DIXIE_T0, bbox=DIXIE_BBOX, city="Greenville / Plumas County, CA",
        hazards=(Hazard.WILDFIRE,), held_out={Hazard.WILDFIRE: True},
        title="Dixie Fire, 2021-08-06 (WildfireSpreadTS 2021 demo)",
        case={"wsts_fire": DEMO.fire, "day0": DEMO.day0.isoformat()},
        window=bbox_to_window(DIXIE_BBOX, buffer_km=2.0),
    )  # fmt: skip
    ad = WildfireAdapter(
        provider=wsts_provider(), checkpoint=get_paths().models / "wildfire" / "best.pt",
        source_tag="wildfire.unet@dixie",
    )  # fmt: skip
    layers = {"model": [ad.predict(scn)], "baseline": [ad.baseline(scn)], "observed": [ad.observed(scn)]}
    notes = [
        "WildfireSpreadTS 2021 (VIIRS 375 m + GRIDMET), mapped to the 12 NDWS channels at 1 km; held-out test year, demo render only (not scored)",
        "model: calibrated fire U-Net day-1 probability grown hourly from the day-0 VIIRS mask (t0 = day 1 12:00 UTC)",
        "OSM roads (Geofabrik norcal extract, Greenville CA graph)",
    ]
    return scn, layers, notes, Path(get_paths().proc) / "roads" / "greenville.npz"


def okfire() -> tuple[Scenario, dict, list[str], Path]:
    scn = Scenario(
        id="ok-fire-2025", t0=OK_T0, bbox=OK_BBOX, city="Oklahoma City, OK",
        hazards=(Hazard.WILDFIRE,), held_out={Hazard.WILDFIRE: True},
        title="REPLAY: Oklahoma wildfires, 2025-03-14 (HMS detections, no model)",
        case={"replay": True, "hms_day": "2025-03-14"}, window=bbox_to_window(OK_BBOX, buffer_km=2.0),
    )  # fmt: skip
    pts = read_hms_fire_points([date(2025, 3, 14), date(2025, 3, 15)])
    rep = hms_replay_layers(scn, pts)
    layers = {"model": [rep["model"]], "observed": [rep["observed"]]}
    notes = [
        "REPLAY of observed NOAA HMS fire-detection points (GOES-East FDC, VIIRS NOAA-20/21, S-NPP), 2025-03-14; not a model forecast and not model-validated",
        "model layer = persistence of the detections in the 3 h before t0 (20:00 UTC) with a 1-cell uncertainty ring; observed = detections after t0",
        "OSM roads (cached OKC graph)",
    ]
    return scn, layers, notes, Path(get_paths().proc) / "roads" / "okc.npz"


def write_bundle(out: Path, scn: Scenario, layers: dict, notes: list[str], graph: Path) -> Path:
    folder = out / scn.id
    sources: list[str] = []
    for name, group in layers.items():
        stack = RiskStack.from_layers(group, scn.t0, scn.grid_window())
        for layer in group:
            for slot in range(stack.n_slots):
                p = folder / "overlays" / name / layer.hazard.value / f"{slot:02d}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                stack.overlay_png(layer.hazard, slot, p)
            sources.append(f"{name}: {layer.source} ({layer.kind})")
            mx = max(float(stack.get(layer.hazard, k).max()) for k in range(stack.n_slots))
            print(f"{name:9s} {layer.source:48s} max p {mx:.3f}")
        z = folder / ("risk.zarr" if name == "model" else f"risk_{name}.zarr")
        if z.exists():
            shutil.rmtree(z)
        stack.to_zarr(z)
    meta = {
        "id": scn.id, "title": scn.title, "t0": scn.t0.isoformat(), "city": scn.city, "bbox": list(scn.bbox),
        "hazards": [h.value for h in scn.hazards], "held_out": True, "synthetic": False,
        "sources": sources + notes, "n_slots": 12, "slot_minutes": 15,
    }  # fmt: skip
    (folder / "scenario.json").write_text(json.dumps(meta, indent=2))
    g = folder / "graph.npz"
    if g.is_symlink() or g.exists():
        g.unlink()
    shutil.copy2(graph, g)
    print(f"wrote {folder}")
    return folder


def write_routes(
    folder: Path, scn_id: str, pairs: list[tuple[tuple[float, float], tuple[float, float]]]
) -> None:
    """Cached presets (APP-11) computed through the live backend for each (origin, dest) and profile."""
    from outrun.app.backend import DefaultBackend
    from outrun.app.models import RouteRequest

    be = DefaultBackend(folder.parent)
    presets = []
    for origin, dest in pairs:
        for profile in ("civilian", "ev"):
            res = be.route(RouteRequest(scenario=scn_id, origin=origin, dest=dest, profile=profile))
            presets.append({"origin": list(origin), "dest": list(dest), "profile": profile, "result": res})
            eta, ex = res["eta_min"], res["exposure"]["predicted"]
            msg = "no route" if eta is None else f"{eta:.0f} min, exposure {ex:.2f}"
            print(
                f"  {profile:9s} {origin} -> {dest}: {msg}, blocked avoided {res['blocked_avoided']}, in place {res['shelter_in_place']}"
            )
    (folder / "results").mkdir(exist_ok=True)
    (folder / "results" / "routes_preset.json").write_text(json.dumps(presets))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("which", choices=["dixie", "okfire"])
    ap.add_argument("--out", default="scenarios")
    ap.add_argument(
        "--routes", action="store_true", help="also write results/routes_preset.json (live backend)"
    )
    args = ap.parse_args()
    scn, layers, notes, graph = (dixie if args.which == "dixie" else okfire)()
    folder = write_bundle(Path(args.out), scn, layers, notes, graph)
    if args.routes:
        write_routes(folder, scn.id, PRESET_PAIRS[scn.id])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
