"""APP-16 (specs/14-app.md): every tab renders a MapLibre map offline, with zero non-localhost requests.

Opt-in: ``make e2e`` (``pytest tests/e2e -m e2e``). Needs ``playwright`` and its Chromium
(``.venv/bin/python -m playwright install chromium``); skipped otherwise, so collection never fails.
"""

from __future__ import annotations

import io
import json
import socket
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

SID = "e2e-okc"
BBOX = [-97.6, 35.4, -97.4, 35.6]
TABS = ("forecast", "route", "shelters", "evac", "dispatch", "results")
ROUTE = {
    "outrun": {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "LineString", "coordinates": [[-97.5, 35.5], [-97.45, 35.55]]},
    },
    "shortest": {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "LineString", "coordinates": [[-97.5, 35.5], [-97.48, 35.56]]},
    },
    "eta_min": 12.5,
    "exposure": {"predicted": 0.1, "observed": 0.05},
    "blocked_avoided": 2,
    "expansions": {"astar": 40, "dijkstra": 95},
    "shelter_in_place": False,
}


def _png(v: int) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (32, 32), (v % 256, 0, 255 - v % 256, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _bundle(root: Path) -> Path:
    d = root / "scenarios" / SID
    meta = {
        "id": SID,
        "title": "E2E synthetic",
        "t0": "2019-05-26T03:00:00+00:00",
        "city": "Oklahoma City, OK",
        "bbox": BBOX,
        "hazards": ["storm"],
        "held_out": False,
        "synthetic": True,
        "sources": ["synthetic"],
    }
    d.mkdir(parents=True)
    (d / "scenario.json").write_text(json.dumps(meta))
    for s in range(12):
        p = d / "overlays" / "model" / "storm" / f"{s:02d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_png(s))
    res = d / "results"
    res.mkdir()
    fc = {"type": "FeatureCollection", "features": [], "metrics": {}}
    for name in ("shelters_mcf", "shelters_nearest", "shelters_greedy", "evac_sa", "evac_msa"):
        (res / f"{name}.json").write_text(json.dumps(fc))
    (res / "evac_sa_trace.json").write_text(json.dumps([{"iter": 0, "objective": 1.0, "temperature": 1.0}]))
    preset = [{"origin": [-97.5, 35.5], "dest": "nearest_shelter", "profile": "civilian", "result": ROUTE}]
    (res / "routes_preset.json").write_text(json.dumps(preset))
    (root / "reports" / "metrics").mkdir(parents=True)
    return root


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path):
    uvicorn = pytest.importorskip("uvicorn")
    from outrun.app.server import create_app

    root = _bundle(tmp_path)
    app = create_app(scenarios_dir=root / "scenarios", reports_dir=root / "reports", backend=None, demo=True)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    deadline = time.time() + 15
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    assert srv.started, "server did not start"
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    th.join(timeout=10)


@pytest.mark.req("APP-16")
def test_every_tab_renders_map_offline(server):
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # Chromium not installed
            pytest.skip(f"playwright chromium unavailable: {exc}")
        page = browser.new_page(viewport={"width": 1000, "height": 700})
        requests: list[str] = []
        errors: list[str] = []
        overlay_ok: list[int] = []
        page.on("request", lambda r: requests.append(r.url))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("response", lambda r: overlay_ok.append(r.status) if "/overlay/" in r.url else None)

        page.goto(server + "/")
        page.wait_for_selector("canvas.maplibregl-canvas", state="visible", timeout=15_000)
        for tab in TABS:
            page.click(f"[data-tab='{tab}']")
            page.wait_for_timeout(300)
            if tab == "forecast":
                page.wait_for_timeout(1000)
                assert 200 in overlay_ok, "no overlay PNG was loaded on the forecast tab"
            if tab == "route":
                box = page.locator("#map").bounding_box()
                with page.expect_response(lambda r: "/api/route" in r.url, timeout=10_000) as resp:
                    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                assert resp.value.status == 200
                page.wait_for_function("window.outrunState && window.outrunState.route", timeout=10_000)
                route = page.evaluate("window.outrunState.route")
                assert route["type"] == "Feature" and route["geometry"]["type"] == "LineString"
        assert errors == [], errors
        foreign = [
            u for u in requests if not (u.startswith(server + "/") or u.startswith(("data:", "blob:")))
        ]
        assert foreign == [], f"non-localhost requests: {foreign}"
        browser.close()
