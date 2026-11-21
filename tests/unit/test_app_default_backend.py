"""`outrun serve` wiring: ``create_app`` without a ``backend`` argument uses DefaultBackend (live), ``backend=None`` stays
cached-only, and a live failure (no graph in a clean bundle) still answers from the cached presets (APP-11, APP-15)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from outrun.app.backend import DefaultBackend
from outrun.app.server import create_app

SID = "tiny"
PRESET = {
    "origin": [-97.5, 35.5],
    "dest": "nearest_shelter",
    "profile": "civilian",
    "result": {
        "outrun": None,
        "shortest": None,
        "eta_min": 3.0,
        "exposure": {"predicted": 0, "observed": 0},
        "blocked_avoided": 0,
        "expansions": {"astar": 1, "dijkstra": 2},
        "shelter_in_place": False,
    },
}


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "scenarios" / SID
    (d / "results").mkdir(parents=True)
    (d / "scenario.json").write_text(
        json.dumps(
            {
                "id": SID,
                "title": "t",
                "t0": "2019-05-26T03:00:00+00:00",
                "city": "x",
                "bbox": [-97.6, 35.4, -97.4, 35.6],
                "hazards": ["storm"],
                "held_out": False,
                "synthetic": True,
                "sources": [],
            }
        )
    )
    (d / "results" / "routes_preset.json").write_text(json.dumps([PRESET]))
    return tmp_path


@pytest.mark.req("APP-15")
def test_default_is_live_and_none_is_cached_only(root):
    live = TestClient(create_app(scenarios_dir=root / "scenarios", reports_dir=root / "r", demo=True))
    cached = TestClient(
        create_app(scenarios_dir=root / "scenarios", reports_dir=root / "r", backend=None, demo=True)
    )
    assert live.get("/api/health").json()["live"] is True
    assert cached.get("/api/health").json()["live"] is False
    assert isinstance(DefaultBackend(root / "scenarios"), DefaultBackend)


@pytest.mark.req("APP-11")
def test_live_failure_falls_back_to_cached_route(root):
    c = TestClient(create_app(scenarios_dir=root / "scenarios", reports_dir=root / "r", demo=True))
    r = c.post("/api/route", json={"scenario": SID, "origin": [-97.5, 35.5], "dest": "nearest_shelter"})
    assert r.status_code == 200 and r.json()["source"] == "cached"
    assert c.post("/api/dispatch", json={"scenario": SID, "incidents": [[-97.5, 35.5]]}).status_code == 503
