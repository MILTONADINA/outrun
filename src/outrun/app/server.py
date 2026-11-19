"""FastAPI app: ``/api/*`` endpoints, SSE stream and the static MapLibre page (spec 14: APP-01..APP-11,
APP-14, APP-15). Run with ``outrun serve`` (CLI-06).

Module-level imports are limited to FastAPI/stdlib and ``outrun.app.*`` so the server starts offline (APP-10).
"""

from __future__ import annotations

import itertools
import json
import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from outrun.app import bundle
from outrun.app.backend import AppBackend, EvacEvent
from outrun.app.models import OVERLAY_SOURCES, DispatchRequest, RouteRequest
from outrun.app.sse import format_sse
from outrun.core.policy import Profile
from outrun.core.risk import Hazard

STATIC_DIR = Path(__file__).resolve().parent / "static"
PRESET_MATCH_RADIUS_M = 250.0
EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres (Earth radius ``EARTH_RADIUS_M``), used by the APP-11 preset match."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _inside(bbox: list[float], lon: float, lat: float) -> bool:
    return bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3]


def _read_json(path: Path) -> Any | None:
    """Parsed JSON file, or None if it is missing or broken."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _same_dest(a: Any, b: Any) -> bool:
    """APP-11: both "nearest_shelter", or both coordinates within 250 m."""
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return haversine_m(a[0], a[1], b[0], b[1]) <= PRESET_MATCH_RADIUS_M


def _cached_route(folder: Path, req: RouteRequest) -> dict[str, Any] | None:
    """Closest matching preset in ``results/routes_preset.json`` (APP-11), or None."""
    presets = _read_json(folder / "results" / "routes_preset.json")
    if not isinstance(presets, list):
        return None
    best, best_d = None, PRESET_MATCH_RADIUS_M
    for p in presets:
        d = haversine_m(req.origin[0], req.origin[1], p["origin"][0], p["origin"][1])
        if p["profile"] == req.profile and d <= best_d and _same_dest(p["dest"], req.dest):
            best, best_d = p, d
    if best is None:
        return None
    return {**best["result"], "source": "cached", "scenario": req.scenario, "cached_origin": best["origin"]}


def create_app(
    scenarios_dir: Path | str = "scenarios",
    reports_dir: Path | str = "reports",
    backend: AppBackend | None = None,
    demo: bool | None = None,
) -> FastAPI:
    """Build the app.

    - ``scenarios_dir``: bundles (layout in specs/14-app.md); ``reports_dir``: ``metrics/*.json``.
    - ``backend``: live planners (``None`` = cached-only mode: routes from presets, SA replayed, dispatch/city 503).
    - ``demo``: ``None`` reads ``OUTRUN_DEMO == "1"``; demo mode forbids network for the city box.

    Routes: ``GET /`` and ``/static/*`` (APP-09), ``GET /api/health`` (APP-15), ``GET /api/scenarios`` (APP-01),
    ``GET /api/scenarios/{id}/overlay/{hazard}/{slot}.png`` (APP-02), ``POST /api/route`` (APP-03, APP-11),
    ``GET /api/shelters/{id}`` (APP-04), ``GET /api/evac/{id}`` and ``/api/evac/{id}/run`` (APP-05),
    ``POST /api/dispatch`` (APP-06), ``GET /api/city`` (APP-07), ``GET /api/metrics`` (APP-08).
    Invalid scenario ids answer 404 without file access (APP-14).
    """
    scenarios_root = Path(scenarios_dir)
    reports_root = Path(reports_dir)
    demo_mode = os.environ.get("OUTRUN_DEMO") == "1" if demo is None else demo
    app = FastAPI(title="Outrun")

    def scenario_folder(scenario_id: str) -> tuple[Path, dict[str, Any]]:
        """Folder and metadata of a valid bundle, else 404. Bad ids are rejected before any file access."""
        if not bundle.is_valid_scenario_id(scenario_id):
            raise HTTPException(404, "unknown scenario")
        folder = bundle.scenario_dir(scenarios_root, scenario_id)
        meta = bundle.load_scenario_meta(folder)
        if meta is None:
            raise HTTPException(404, "unknown scenario")
        return folder, meta

    def result_file(scenario_id: str, name: str) -> Path:
        folder, _ = scenario_folder(scenario_id)
        path = folder / "results" / name
        if not path.is_file():
            raise HTTPException(404, f"{name} not found")
        return path

    def check_inside(meta: dict[str, Any], points: list[Any]) -> None:
        for lon, lat in points:
            if not _inside(meta["bbox"], lon, lat):
                raise HTTPException(422, f"point [{lon}, {lat}] is outside the scenario bbox {meta['bbox']}")

    # ---- health, scenarios, metrics
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        n = len(bundle.list_scenarios(scenarios_root))
        return {"status": "ok", "demo": demo_mode, "live": backend is not None, "n_scenarios": n}

    @app.get("/api/scenarios")
    def scenarios() -> list[dict[str, Any]]:
        return bundle.list_scenarios(scenarios_root)

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        return bundle.read_metrics(reports_root)

    # ---- overlay PNG (APP-02)
    @app.get("/api/scenarios/{scenario_id}/overlay/{hazard}/{slot}.png")
    def overlay(scenario_id: str, hazard: str, slot: str, source: str = "model") -> FileResponse:
        folder, meta = scenario_folder(scenario_id)
        if hazard not in [h.name.lower() for h in Hazard]:
            raise HTTPException(422, f"unknown hazard {hazard!r}")
        if source not in OVERLAY_SOURCES:
            raise HTTPException(422, f"unknown source {source!r}")
        try:
            slot_no = int(slot)
        except ValueError:
            raise HTTPException(422, "slot must be an integer") from None
        if not 0 <= slot_no < meta["n_slots"]:
            raise HTTPException(422, f"slot must be in [0, {meta['n_slots'] - 1}]")
        path = folder / "overlays" / source / hazard / f"{slot_no:02d}.png"
        if not path.is_file():
            raise HTTPException(404, "overlay not found")
        return FileResponse(path, media_type="image/png")

    # ---- live route with cached fallback (APP-03, APP-11)
    @app.post("/api/route")
    def route(req: RouteRequest) -> JSONResponse:
        folder, meta = scenario_folder(req.scenario)
        if req.profile not in [p.name.lower() for p in Profile]:
            raise HTTPException(422, f"unknown profile {req.profile!r}")
        check_inside(meta, [req.origin] + ([] if req.dest == "nearest_shelter" else [req.dest]))
        if backend is not None:
            try:
                return JSONResponse({**backend.route(req), "source": "live", "scenario": req.scenario})
            except Exception:  # any live failure falls back to the cached presets
                pass
        cached = _cached_route(folder, req)
        if cached is None:
            raise HTTPException(503, "no live router and no cached route are available for this request")
        return JSONResponse(cached)

    # ---- shelters and evacuation results (APP-04, APP-05)
    @app.get("/api/shelters/{scenario_id}")
    def shelters(scenario_id: str, method: Literal["mcf", "nearest", "greedy"] = "mcf") -> Response:
        path = result_file(scenario_id, f"shelters_{method}.json")
        return Response(path.read_bytes(), media_type="application/json")

    @app.get("/api/evac/{scenario_id}")
    def evac(scenario_id: str, method: Literal["sa", "msa", "shortest", "safest"] = "sa") -> Response:
        path = result_file(scenario_id, f"evac_{method}.json")
        return Response(path.read_bytes(), media_type="application/json")

    def live_events(first: EvacEvent | None, rest: Iterator[EvacEvent]) -> Iterator[str]:
        """SSE text for a live SA run; an error mid-run becomes one ``error`` event."""
        event_id = 0
        try:
            head = [] if first is None else [first]
            for kind, data in itertools.chain(head, rest):
                if kind == "done":
                    data = {**data, "source": "live"}
                yield format_sse(kind, data, id=event_id)
                event_id += 1
        except Exception as exc:
            yield format_sse("error", {"detail": str(exc)}, id=event_id)

    def cached_events(trace: list[Any], final: dict[str, Any]) -> Iterator[str]:
        for i, point in enumerate(trace):
            yield format_sse("progress", point, id=i)
        yield format_sse("done", {**final, "source": "cached"}, id=len(trace))

    @app.get("/api/evac/{scenario_id}/run")
    def evac_run(scenario_id: str) -> StreamingResponse:
        folder, _ = scenario_folder(scenario_id)
        stream = None
        if backend is not None:
            try:
                events = iter(backend.run_evac(scenario_id))
                first = next(events, None)  # a failure before the first event means "use the cache"
                stream = live_events(first, events)
            except Exception:
                stream = None
        if stream is None:
            trace = _read_json(folder / "results" / "evac_sa_trace.json")
            final = _read_json(folder / "results" / "evac_sa.json")
            if not isinstance(trace, list) or not isinstance(final, dict):
                raise HTTPException(503, "no live evacuation run and no cached run are available")
            stream = cached_events(trace, final)
        return StreamingResponse(stream, media_type="text/event-stream")

    # ---- dispatch and city box (APP-06, APP-07)
    @app.post("/api/dispatch")
    def dispatch(req: DispatchRequest) -> JSONResponse:
        _, meta = scenario_folder(req.scenario)
        check_inside(meta, list(req.incidents))
        if backend is None:
            raise HTTPException(503, "no live dispatch backend")
        try:
            return JSONResponse({**backend.dispatch(req), "source": "live"})
        except Exception as exc:
            raise HTTPException(503, f"dispatch failed: {exc}") from exc

    @app.get("/api/city")
    def city(name: str = Query(...)) -> dict[str, Any]:
        name = name.strip()
        if not 1 <= len(name) <= 100:
            raise HTTPException(422, "name must be 1 to 100 characters")
        if backend is None:
            raise HTTPException(503, "no live backend for city graphs")
        try:
            return backend.city(name, allow_network=not demo_mode)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, f"city lookup failed: {exc}") from exc

    # ---- static page (APP-09)
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
