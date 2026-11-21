"""APP-16 (specs/14-app.md), second part: the interactive features of every tab work offline with a live-style backend.

Drives swipe, play, lambda slider, model card, shelter bars, SA curve (SSE), wave animation, dispatch table and the
results tables against a fake backend. Opt-in like test_offline.py (``make e2e``)."""

from __future__ import annotations

import json
import threading
import time

import pytest
from tests.e2e.test_offline import BBOX, ROUTE, SID, TABS, _bundle, _free_port, _png

pytestmark = pytest.mark.e2e

POINTS = [[-97.55, 35.45], [-97.5, 35.5], [-97.45, 35.55]]


def _extend_bundle(root):
    d = root / "scenarios" / SID
    meta = json.loads((d / "scenario.json").read_text())
    meta["hazards"] = ["storm"]
    (d / "scenario.json").write_text(json.dumps(meta))
    for s in range(12):  # observed overlays for the swipe
        p = d / "overlays" / "observed" / "storm" / f"{s:02d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_png(s + 40))
    res = d / "results"
    shel = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "id": f"shelter-{i}",
                    "kind": "shelter",
                    "capacity": 100 * (i + 1),
                    "load": 100 * (i + 1) + (5 if i == 0 else 0),
                    "overflow": 5 if i == 0 else 0,
                    "assumed": False,
                },
                "geometry": {"type": "Point", "coordinates": c},
            }
            for i, c in enumerate(POINTS)
        ],
        "metrics": {"served_persons": 600, "unserved_persons": 0, "overflow_pct": 1.5},
    }
    for m in ("mcf", "nearest", "greedy"):
        (res / f"shelters_{m}.json").write_text(json.dumps(shel))
    evac = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"packet": i, "vehicles": 20 + i, "k": 0, "wave": i % 3},
                "geometry": {"type": "Point", "coordinates": c},
            }
            for i, c in enumerate(POINTS)
        ],
        "metrics": {"clearance_min": 30.0, "J": 100.0, "max_vc": 1.2},
    }
    for m in ("sa", "msa", "shortest", "safest"):
        (res / f"evac_{m}.json").write_text(json.dumps(evac))
    trace = [{"iter": 100 * i, "objective": 500.0 - 40 * i, "temperature": 10.0 - i} for i in range(8)]
    (res / "evac_sa_trace.json").write_text(json.dumps(trace))
    (root / "reports" / "metrics" / "storm_val.json").write_text(
        json.dumps(
            {
                "csi": 0.7,
                "baseline": {"csi": 0.6},
                "reliability": {"predicted": [0.1, 0.5, 0.9], "observed": [0.12, 0.45, 0.88]},
            }
        )
    )


class FakeBackend:
    def __init__(self):
        self.route_bodies = []

    def route(self, req):
        self.route_bodies.append(req.model_dump(by_alias=True))
        return dict(ROUTE)

    def dispatch(self, req):
        line = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "LineString", "coordinates": POINTS[:2]},
        }
        fc = {"type": "FeatureCollection", "features": [line]}
        return {
            "assignment": [{"incident": 0, "unit": 7}],
            "routes": fc,
            "baseline": fc,
            "comparison": [
                {
                    "incident": 0,
                    "unit": 7,
                    "eta_min": 4.5,
                    "baseline_eta_min": 6.0,
                    "baseline_detour_min": 1.5,
                    "baseline_hit_block": True,
                }
            ],
            "metrics": {"mean_eta_min": 4.5, "pct_hit_block": 100.0, "pct_unreached": 0.0},
        }

    def run_evac(self, scenario_id):
        for i in range(6):
            yield "progress", {"iter": 100 * i, "objective": 500.0 - 50 * i, "temperature": 9.0 - i}
        yield (
            "done",
            {"type": "FeatureCollection", "features": [], "metrics": {"clearance_min": 21.0, "J": 250.0}},
        )

    def city(self, name, allow_network):
        raise LookupError("not cached")


@pytest.fixture
def live_server(tmp_path):
    uvicorn = pytest.importorskip("uvicorn")
    from outrun.app.server import create_app

    root = _bundle(tmp_path)
    _extend_bundle(root)
    backend = FakeBackend()
    app = create_app(
        scenarios_dir=root / "scenarios", reports_dir=root / "reports", backend=backend, demo=True
    )
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    deadline = time.time() + 15
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    assert srv.started
    yield f"http://127.0.0.1:{port}", backend
    srv.should_exit = True
    th.join(timeout=10)


@pytest.mark.req("APP-16")
def test_interactive_features_offline(live_server):
    url, backend = live_server
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"playwright chromium unavailable: {exc}")
        page = browser.new_page(viewport={"width": 1100, "height": 750})
        requests, errors = [], []
        page.on("request", lambda r: requests.append(r.url))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url + "/")
        page.wait_for_selector("canvas.maplibregl-canvas", state="visible", timeout=15_000)
        page.wait_for_selector("#scenario-card h3", timeout=10_000)
        assert page.locator("#scenario-card .badge.synth").count() == 1

        # forecast: swipe, play, model card
        page.check("#swipe-toggle")
        page.wait_for_selector("#map2 canvas", state="attached", timeout=10_000)
        page.wait_for_selector("#swipe-bar", state="visible", timeout=10_000)
        page.fill("#swipe-slider", "30")
        with page.expect_response(
            lambda r: "source=observed" in r.url and "/overlay/" in r.url, timeout=10_000
        ):
            page.fill("#slot-slider", "3")
        page.uncheck("#swipe-toggle")
        assert not page.is_visible("#swipe-bar")
        page.click("#play-btn")
        page.wait_for_function("document.querySelector('#slot-slider').value !== '0'", timeout=5000)
        page.click("#play-btn")
        page.click("#card-btn")
        page.wait_for_function(
            "document.querySelector('#drawer-body').innerText.includes('storm_val')", timeout=5000
        )
        assert page.locator("#drawer-body canvas").count() == 1  # reliability plot
        page.click("#drawer-close")

        # route: lambda slider is sent to the backend
        page.click("[data-tab='route']")
        page.uncheck("#lambda-default")
        page.fill("#lambda-slider", "2.5")
        box = page.locator("#map").bounding_box()
        with page.expect_response(lambda r: "/api/route" in r.url):
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        assert backend.route_bodies[-1]["lambda"] == 2.5
        assert page.locator("#route-table").is_visible()

        # shelters: capacity bars with an overflow bar
        page.click("[data-tab='shelters']")
        page.wait_for_selector("#shelter-bars .fill.over", timeout=5000)
        assert page.locator("#shelter-bars .bar").count() == 3

        # evacuation: live SA curve, then wave animation
        page.click("[data-tab='evac']")
        page.click("#evac-run")
        page.wait_for_function(
            "document.querySelector('#evac-progress').innerText.startsWith('done')", timeout=10_000
        )
        assert page.evaluate(
            "(() => { const c = document.getElementById('sa-curve'); const d = c.getContext('2d')"
            ".getImageData(0, 0, c.width, c.height).data; return d.some(v => v !== 0); })()"
        )
        page.click("#evac-play")
        page.wait_for_function(
            "document.querySelector('#evac-progress').innerText.includes('waves')", timeout=5000
        )
        page.click("#evac-play")

        # dispatch: table with the baseline comparison
        page.click("[data-tab='dispatch']")
        page.mouse.click(box["x"] + box["width"] / 3, box["y"] + box["height"] / 2)
        page.click("#dispatch-go")
        page.wait_for_selector("#dispatch-table td", timeout=5000)
        assert "yes" in page.locator("#dispatch-table").inner_text()

        # results: metrics and per-method tables
        page.click("[data-tab='results']")
        page.wait_for_function(
            "document.querySelector('#metrics-tables').innerText.includes('storm_val')", timeout=5000
        )
        page.wait_for_function(
            "document.querySelector('#metrics-tables').innerText.includes('Shelter methods')", timeout=5000
        )
        for tab in TABS:
            page.click(f"[data-tab='{tab}']")
        assert errors == [], errors
        foreign = [u for u in requests if not (u.startswith(url + "/") or u.startswith(("data:", "blob:")))]
        assert foreign == [], foreign
        browser.close()
        assert BBOX
