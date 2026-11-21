"""Write the El Reno 2019-05-26 storm layers into ``scenarios/elreno-2019`` (spec 14 bundle layout).

The storm RiskLayer comes from real SEVIR frames (event S830307, VIL only, no IR107) reprojected to the EPSG:5070 1 km
window of the cached OKC graph: ``risk.zarr`` / ``overlays/model`` = the storm model chosen by ``--mode`` (default
auto: VIL-only U-Net if its checkpoint exists, else the extrapolation baseline), ``risk_baseline.zarr`` =
extrapolation baseline, ``risk_observed.zarr`` = the SEVIR frames that followed. The source label of each layer is
written to ``scenario.json`` ``sources``. The tornado layer (TorNet chips + CNN + swath along storm motion) is added next to it; others come in T20/T22.

    .venv/bin/python scripts/make_elreno_storm.py [--out scenarios] [--mode auto|dl|dl_vil_only|baseline]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from outrun.config import get_paths
from outrun.core.grid import bbox_to_window
from outrun.core.risk import Hazard, RiskStack
from outrun.core.scenario import Scenario
from outrun.hazards.storm.provider import ExtrapolationStormAdapter, real_provider, storm_adapter
from outrun.hazards.tornado.adapter import TornadoAdapter
from outrun.hazards.tornado.provider import real_provider as tornado_provider

BBOX = (-98.1, 35.1, -97.1, 35.8)
SID = "elreno-2019"
T0 = datetime(2019, 5, 26, 3, 0, tzinfo=UTC)


def elreno_scenario() -> Scenario:
    return Scenario(
        id=SID, t0=T0, bbox=BBOX, city="El Reno / Oklahoma City, OK", hazards=(Hazard.STORM, Hazard.TORNADO),
        held_out={Hazard.STORM: True, Hazard.TORNADO: True}, title="El Reno EF3, 2019-05-26 (SEVIR S830307)",
        case={"sevir_id": "S830307"}, window=bbox_to_window(BBOX, buffer_km=2.0),
    )  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="scenarios")
    ap.add_argument("--mode", default="auto", choices=["auto", "dl", "dl_vil_only", "baseline"])
    ap.add_argument("--graph", default=str(get_paths().proc / "roads" / "okc.npz"))
    args = ap.parse_args()

    scn = elreno_scenario()
    provider = real_provider(args.mode)
    model_ad = storm_adapter(scn, args.mode, provider)
    base_ad = ExtrapolationStormAdapter(provider)
    tor_ad = TornadoAdapter(
        provider=tornado_provider(), checkpoint=get_paths().models / "tornado" / "best.pt",
        source_tag="tornado.cnn@elreno",
    )  # fmt: skip
    layers = {
        "model": [model_ad.predict(scn), tor_ad.predict(scn)],
        "baseline": [base_ad.baseline(scn), tor_ad.baseline(scn)],
    }
    obs = model_ad.observed(scn)
    if obs is not None:
        layers["observed"] = [obs]
    folder = Path(args.out) / SID
    sources = []
    for name, group in layers.items():
        stack = RiskStack.from_layers(group, scn.t0, scn.grid_window())
        for layer in group:
            for slot in range(stack.n_slots):
                p = folder / "overlays" / name / layer.hazard.value / f"{slot:02d}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                stack.overlay_png(layer.hazard, slot, p)
            sources.append(f"{name}: {layer.source} ({layer.kind})")
            print(
                f"{name:9s} {layer.source:42s} max p {max(float(stack.get(layer.hazard, k).max()) for k in range(stack.n_slots)):.3f}"
            )
        z = folder / ("risk.zarr" if name == "model" else f"risk_{name}.zarr")
        if z.exists():
            shutil.rmtree(z)
        stack.to_zarr(z)
    meta = {
        "id": SID, "title": scn.title, "t0": scn.t0.isoformat(), "city": scn.city, "bbox": list(BBOX),
        "hazards": ["storm", "tornado"], "held_out": True, "synthetic": False,
        "sources": sources + ["SEVIR VIL (NEXRAD) 2 km, regridded to EPSG:5070 1 km",
        "tornado: TorNet KTLX chips (ending <= t0+35 min, demo timing slack), swath along storm-provider motion (fallback HRRR/default)",
        "OSM roads (cached OKC graph)"],
        "n_slots": 12, "slot_minutes": 15,
    }  # fmt: skip
    (folder / "scenario.json").write_text(json.dumps(meta, indent=2))
    g = folder / "graph.npz"
    if g.is_symlink() or g.exists():
        g.unlink()
    shutil.copy2(args.graph, g)
    print(f"wrote {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
