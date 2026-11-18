"""FastAPI app: ``/api/*`` endpoints, SSE stream and the static MapLibre page (spec 14: APP-01..APP-11,
APP-14, APP-15). Run with ``outrun serve`` (CLI-06).

Module-level imports are limited to FastAPI/stdlib and ``outrun.app.*`` so the server starts offline (APP-10).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from outrun.app.backend import AppBackend

STATIC_DIR = Path(__file__).resolve().parent / "static"
PRESET_MATCH_RADIUS_M = 250.0
EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres (Earth radius ``EARTH_RADIUS_M``), used by the APP-11 preset match."""
    raise NotImplementedError


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
    raise NotImplementedError
