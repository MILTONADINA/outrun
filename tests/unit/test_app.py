"""specs/14-app.md (APP-*): FastAPI app over a synthetic scenario bundle in tmp_path, with an injected fake backend.

Offline and fast: no real data, no network, no browser. Bundles are written by the ``bundle_root`` fixture below.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from outrun.app import bundle as bundle_mod
from outrun.app.export import export_static
from outrun.app.models import DispatchRequest, RouteRequest
from outrun.app.server import STATIC_DIR, create_app
from outrun.app.sse import format_sse

SID = "synth-okc"
BBOX = [-97.6, 35.4, -97.4, 35.6]
HAZARDS = ["storm", "tornado", "flood", "wildfire", "ice", "smoke"]
REQUIRED = {"id", "title", "t0", "city", "bbox", "hazards", "held_out", "synthetic", "sources"}
ENTRY_KEYS = REQUIRED | {"n_slots", "slot_minutes", "overlay_sources"}

ROUTE_RESULT = {
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
DISPATCH_RESULT = {
    "assignment": [{"incident": 0, "unit": "E1"}],
    "routes": {"type": "FeatureCollection", "features": []},
    "baseline": {"type": "FeatureCollection", "features": []},
}
SHELTERS_MCF = {"type": "FeatureCollection", "features": [], "metrics": {"overflow": 0, "method": "mcf"}}
SHELTERS_NEAREST = {
    "type": "FeatureCollection",
    "features": [],
    "metrics": {"overflow": 120, "method": "nearest"},
}
EVAC_SA = {"type": "FeatureCollection", "features": [], "metrics": {"J": 101.5, "clearance_min": 140}}
EVAC_TRACE = [
    {"iter": 0, "objective": 150.0, "temperature": 10.0},
    {"iter": 100, "objective": 120.0, "temperature": 5.0},
    {"iter": 200, "objective": 101.5, "temperature": 2.5},
]
# Preset A at (-97.50, 35.50); preset B ~167 m north of A; a "nearest_shelter" preset for profile ev.
PRESETS = [
    {"origin": [-97.50, 35.50], "dest": "nearest_shelter", "profile": "civilian", "result": {"label": "A"}},
    {"origin": [-97.50, 35.5015], "dest": "nearest_shelter", "profile": "civilian", "result": {"label": "B"}},
    {"origin": [-97.45, 35.45], "dest": [-97.42, 35.47], "profile": "civilian", "result": {"label": "C"}},
    {"origin": [-97.55, 35.55], "dest": "nearest_shelter", "profile": "ev", "result": {"label": "D"}},
]


# ----------------------------------------------------------------- fixtures
def _png(value: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", (32, 32), (value % 256, 0, 255 - value % 256, 128)).save(buf, format="PNG")
    return buf.getvalue()


def _meta(sid: str, **over) -> dict:
    meta = {
        "id": sid,
        "title": f"Synthetic {sid}",
        "t0": "2019-05-26T03:00:00+00:00",
        "city": "Oklahoma City, OK",
        "bbox": list(BBOX),
        "hazards": ["storm", "tornado"],
        "held_out": False,
        "synthetic": True,
        "sources": ["synthetic moving Gaussians"],
    }
    meta.update(over)
    return meta


def _write_bundle(root: Path, sid: str, meta: dict | str | None) -> Path:
    d = root / sid
    d.mkdir(parents=True)
    if isinstance(meta, dict):
        (d / "scenario.json").write_text(json.dumps(meta))
    elif isinstance(meta, str):
        (d / "scenario.json").write_text(meta)
    return d


@pytest.fixture
def bundle_root(tmp_path) -> Path:
    root = tmp_path / "scenarios"
    root.mkdir()
    # main bundle: 12 slots (default), all three overlay sources, results
    d = _write_bundle(root, SID, _meta(SID))
    for s in range(12):
        p = d / "overlays" / "model" / "storm" / f"{s:02d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_png(s))
    for src, hz, s in (("baseline", "storm", 0), ("observed", "tornado", 3)):
        p = d / "overlays" / src / hz / f"{s:02d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_png(100 + s))
    res = d / "results"
    res.mkdir()
    (res / "shelters_mcf.json").write_text(json.dumps(SHELTERS_MCF))
    (res / "shelters_nearest.json").write_text(json.dumps(SHELTERS_NEAREST))
    (res / "evac_sa.json").write_text(json.dumps(EVAC_SA))
    (res / "evac_sa_trace.json").write_text(json.dumps(EVAC_TRACE))
    (res / "routes_preset.json").write_text(json.dumps(PRESETS))
    # second valid bundle, 4 slots, only model overlays, no results
    a = _write_bundle(root, "alpha", _meta("alpha", n_slots=4, slot_minutes=15, held_out=True))
    p = a / "overlays" / "model" / "flood" / "00.png"
    p.parent.mkdir(parents=True)
    p.write_bytes(_png(7))
    # invalid bundles: all must be skipped silently
    _write_bundle(root, "broken", "{not json")
    _write_bundle(root, "noscenario", None)
    _write_bundle(root, "mismatch", _meta("other-id"))
    _write_bundle(root, "badbbox", _meta("badbbox", bbox=[-97.4, 35.4, -97.6, 35.6]))
    m = _meta("missingkey")
    del m["city"]
    _write_bundle(root, "missingkey", m)
    bad = _write_bundle(root, "Bad.Id", _meta("Bad.Id"))
    (bad / "results").mkdir()
    (bad / "results" / "shelters_mcf.json").write_text(json.dumps(SHELTERS_MCF))
    (root / "README.txt").write_text("not a bundle")
    return root


@pytest.fixture
def reports_root(tmp_path) -> Path:
    r = tmp_path / "reports"
    (r / "metrics").mkdir(parents=True)
    (r / "metrics" / "storm.json").write_text(json.dumps({"csi": 0.41, "baseline_csi": 0.33}))
    (r / "metrics" / "a_routing.json").write_text(json.dumps({"gap_closed": 0.6}))
    (r / "metrics" / "bad.json").write_text("{oops")
    return r


class FakeBackend:
    def __init__(self):
        self.route_calls: list[RouteRequest] = []
        self.dispatch_calls: list[DispatchRequest] = []
        self.city_calls: list[tuple[str, bool]] = []
        self.evac_calls: list[str] = []
        self.city_raises: Exception | None = None

    def route(self, req):
        self.route_calls.append(req)
        return json.loads(json.dumps(ROUTE_RESULT))

    def dispatch(self, req):
        self.dispatch_calls.append(req)
        return json.loads(json.dumps(DISPATCH_RESULT))

    def run_evac(self, scenario_id):
        self.evac_calls.append(scenario_id)
        for i in range(3):
            yield ("progress", {"iter": i, "objective": 10.0 - i, "temperature": 1.0 / (i + 1)})
        yield ("done", {"metrics": {"J": 8.0}})

    def city(self, name, allow_network):
        self.city_calls.append((name, allow_network))
        if self.city_raises is not None:
            raise self.city_raises
        return {
            "name": name,
            "bbox": [-96.1, 36.0, -95.8, 36.3],
            "n_nodes": 100,
            "n_edges": 250,
            "cached": True,
        }


class FailingBackend:
    def route(self, req):
        raise RuntimeError("router crashed")

    def dispatch(self, req):
        raise RuntimeError("dispatch crashed")

    def run_evac(self, scenario_id):
        raise RuntimeError("sa crashed")

    def city(self, name, allow_network):
        raise RuntimeError("city crashed")


class MidStreamFailBackend(FakeBackend):
    def run_evac(self, scenario_id):
        yield ("progress", {"iter": 0, "objective": 5.0, "temperature": 1.0})
        raise RuntimeError("diverged")


@pytest.fixture
def fake() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def client(bundle_root, reports_root, fake) -> TestClient:
    return TestClient(
        create_app(scenarios_dir=bundle_root, reports_dir=reports_root, backend=fake, demo=False)
    )


@pytest.fixture
def offline_client(bundle_root, reports_root) -> TestClient:
    return TestClient(
        create_app(scenarios_dir=bundle_root, reports_dir=reports_root, backend=None, demo=True)
    )


def _route_body(**over) -> dict:
    body = {"scenario": SID, "origin": [-97.50, 35.5004], "dest": "nearest_shelter"}
    body.update(over)
    return body


def _parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        ev = {}
        for line in block.split("\n"):
            key, _, value = line.partition(": ")
            ev[key] = value
        ev["data"] = json.loads(ev["data"])
        events.append(ev)
    return events


# ----------------------------------------------------------------- APP-01
@pytest.mark.req("APP-01")
def test_scenarios_lists_only_valid_bundles_sorted(client, bundle_root):
    r = client.get("/api/scenarios")
    assert r.status_code == 200
    body = r.json()
    assert [e["id"] for e in body] == ["alpha", SID]
    for e in body:
        assert set(e) == ENTRY_KEYS
    synth = body[1]
    assert synth["n_slots"] == 12 and synth["slot_minutes"] == 15
    assert synth["overlay_sources"] == ["baseline", "model", "observed"]
    assert synth["bbox"] == BBOX and synth["city"] == "Oklahoma City, OK"
    assert synth["title"] == f"Synthetic {SID}" and synth["synthetic"] is True
    alpha = body[0]
    assert alpha["n_slots"] == 4 and alpha["held_out"] is True and alpha["overlay_sources"] == ["model"]


@pytest.mark.req("APP-01")
def test_list_scenarios_function_matches_endpoint(client, bundle_root):
    assert bundle_mod.list_scenarios(bundle_root) == client.get("/api/scenarios").json()


@pytest.mark.req("APP-01")
def test_missing_or_empty_scenarios_dir(tmp_path, reports_root):
    assert bundle_mod.list_scenarios(tmp_path / "nope") == []
    (tmp_path / "empty").mkdir()
    assert bundle_mod.list_scenarios(tmp_path / "empty") == []
    c = TestClient(create_app(scenarios_dir=tmp_path / "nope", reports_dir=reports_root, demo=False))
    r = c.get("/api/scenarios")
    assert r.status_code == 200 and r.json() == []


@pytest.mark.req("APP-01")
def test_load_scenario_meta_rejects_invalid(bundle_root):
    for name in ("broken", "noscenario", "mismatch", "badbbox", "missingkey", "Bad.Id"):
        assert bundle_mod.load_scenario_meta(bundle_root / name) is None, name
    meta = bundle_mod.load_scenario_meta(bundle_root / SID)
    assert meta is not None and meta["id"] == SID and set(meta) == ENTRY_KEYS


# ----------------------------------------------------------------- APP-02
@pytest.mark.req("APP-02")
def test_overlay_png_bytes_identical(client, bundle_root):
    for slot in (0, 5, 11):
        r = client.get(f"/api/scenarios/{SID}/overlay/storm/{slot}.png")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("image/png")
        assert (
            r.content == (bundle_root / SID / "overlays" / "model" / "storm" / f"{slot:02d}.png").read_bytes()
        )


@pytest.mark.req("APP-02")
def test_overlay_source_parameter(client, bundle_root):
    r = client.get(f"/api/scenarios/{SID}/overlay/storm/0.png", params={"source": "baseline"})
    assert r.status_code == 200
    assert r.content == (bundle_root / SID / "overlays" / "baseline" / "storm" / "00.png").read_bytes()
    r = client.get(f"/api/scenarios/{SID}/overlay/tornado/3.png", params={"source": "observed"})
    assert r.status_code == 200
    assert r.content == (bundle_root / SID / "overlays" / "observed" / "tornado" / "03.png").read_bytes()
    r_default = client.get(f"/api/scenarios/{SID}/overlay/storm/0.png")
    r_model = client.get(f"/api/scenarios/{SID}/overlay/storm/0.png", params={"source": "model"})
    assert r_default.content == r_model.content


@pytest.mark.req("APP-02")
@pytest.mark.parametrize(
    "url",
    [
        f"/api/scenarios/{SID}/overlay/hail/0.png",
        f"/api/scenarios/{SID}/overlay/storm/12.png",
        f"/api/scenarios/{SID}/overlay/storm/-1.png",
        f"/api/scenarios/{SID}/overlay/storm/abc.png",
        f"/api/scenarios/{SID}/overlay/storm/0.png?source=forecast",
        "/api/scenarios/alpha/overlay/flood/4.png",
    ],
)
def test_overlay_validation_422(client, url):
    assert client.get(url).status_code == 422


@pytest.mark.req("APP-02")
def test_overlay_not_found_404(client):
    assert client.get(f"/api/scenarios/{SID}/overlay/flood/0.png").status_code == 404
    assert client.get(f"/api/scenarios/{SID}/overlay/storm/1.png?source=baseline").status_code == 404
    assert client.get("/api/scenarios/nope/overlay/storm/0.png").status_code == 404
    assert client.get("/api/scenarios/broken/overlay/storm/0.png").status_code == 404


@pytest.mark.req("APP-02")
def test_every_hazard_name_accepted(client):
    for hz in HAZARDS:
        # files absent except storm, so 404 (not 422) proves the hazard name was accepted
        code = client.get(f"/api/scenarios/{SID}/overlay/{hz}/0.png").status_code
        assert code == (200 if hz == "storm" else 404), hz


# ----------------------------------------------------------------- APP-03
@pytest.mark.req("APP-03")
def test_route_live(client, fake):
    r = client.post("/api/route", json=_route_body(**{"lambda": 0.5, "profile": "sensitive"}))
    assert r.status_code == 200
    body = r.json()
    for k, v in ROUTE_RESULT.items():
        assert body[k] == v
    assert body["source"] == "live" and body["scenario"] == SID
    assert len(fake.route_calls) == 1
    req = fake.route_calls[0]
    assert isinstance(req, RouteRequest)
    assert req.scenario == SID and req.profile == "sensitive" and req.lam == pytest.approx(0.5)
    assert tuple(req.origin) == pytest.approx((-97.50, 35.5004))
    assert req.dest == "nearest_shelter" and req.risk_source == "model"


@pytest.mark.req("APP-03")
def test_route_defaults_and_coordinate_dest(client, fake):
    r = client.post("/api/route", json=_route_body(dest=[-97.42, 35.47]))
    assert r.status_code == 200
    req = fake.route_calls[0]
    assert req.profile == "civilian" and req.lam is None and req.t0 is None and req.risk_source == "model"
    assert tuple(req.dest) == pytest.approx((-97.42, 35.47))


@pytest.mark.req("APP-03")
@pytest.mark.parametrize(
    "over",
    [
        {"origin": [200.0, 35.5]},
        {"origin": [-97.5, -95.0]},
        {"profile": "truck"},
        {"lambda": -1.0},
        {"origin": None},
        {"dest": "home"},
        {"risk_source": "oracle"},
        {"dest": [-97.5]},
    ],
)
def test_route_validation_422(client, fake, over):
    body = _route_body(**over)
    if over.get("origin", 0) is None:
        del body["origin"]
    r = client.post("/api/route", json=body)
    assert r.status_code == 422
    assert fake.route_calls == []


@pytest.mark.req("APP-03")
def test_route_outside_bbox_422(client, fake):
    r = client.post("/api/route", json=_route_body(origin=[-96.0, 36.0]))
    assert r.status_code == 422 and "bbox" in json.dumps(r.json()["detail"])
    r = client.post("/api/route", json=_route_body(dest=[-97.0, 35.5]))
    assert r.status_code == 422 and "bbox" in json.dumps(r.json()["detail"])
    assert fake.route_calls == []


@pytest.mark.req("APP-03")
def test_route_unknown_scenario_404(client, fake):
    assert client.post("/api/route", json=_route_body(scenario="nope")).status_code == 404
    assert fake.route_calls == []


# ----------------------------------------------------------------- APP-04
@pytest.mark.req("APP-04")
def test_shelters_methods(client):
    r = client.get(f"/api/shelters/{SID}")
    assert r.status_code == 200 and r.json() == SHELTERS_MCF
    assert client.get(f"/api/shelters/{SID}", params={"method": "mcf"}).json() == SHELTERS_MCF
    assert client.get(f"/api/shelters/{SID}", params={"method": "nearest"}).json() == SHELTERS_NEAREST


@pytest.mark.req("APP-04")
def test_shelters_errors(client):
    assert client.get(f"/api/shelters/{SID}", params={"method": "greedy"}).status_code == 404
    assert client.get(f"/api/shelters/{SID}", params={"method": "lp"}).status_code == 422
    assert client.get("/api/shelters/nope").status_code == 404


# ----------------------------------------------------------------- APP-05
@pytest.mark.req("APP-05")
def test_evac_results(client):
    r = client.get(f"/api/evac/{SID}")
    assert r.status_code == 200 and r.json() == EVAC_SA
    assert client.get(f"/api/evac/{SID}", params={"method": "msa"}).status_code == 404
    assert client.get(f"/api/evac/{SID}", params={"method": "ga"}).status_code == 422
    assert client.get("/api/evac/nope").status_code == 404


@pytest.mark.req("APP-05")
def test_evac_run_streams_live_events(client, fake):
    r = client.get(f"/api/evac/{SID}/run")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(r.text)
    assert [e["event"] for e in events] == ["progress", "progress", "progress", "done"]
    assert [int(e["id"]) for e in events] == [0, 1, 2, 3]
    assert events[0]["data"] == {"iter": 0, "objective": 10.0, "temperature": 1.0}
    assert events[-1]["data"] == {"metrics": {"J": 8.0}, "source": "live"}
    assert fake.evac_calls == [SID]


@pytest.mark.req("APP-05")
def test_evac_run_error_mid_stream(bundle_root, reports_root):
    c = TestClient(
        create_app(
            scenarios_dir=bundle_root, reports_dir=reports_root, backend=MidStreamFailBackend(), demo=False
        )
    )
    r = c.get(f"/api/evac/{SID}/run")
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert [e["event"] for e in events] == ["progress", "error"]
    assert "diverged" in events[-1]["data"]["detail"]


@pytest.mark.req("APP-05")
@pytest.mark.parametrize("backend_cls", [None, FailingBackend])
def test_evac_run_cached_replay(bundle_root, reports_root, backend_cls):
    backend = backend_cls() if backend_cls else None
    c = TestClient(
        create_app(scenarios_dir=bundle_root, reports_dir=reports_root, backend=backend, demo=True)
    )
    r = c.get(f"/api/evac/{SID}/run")
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert [e["event"] for e in events] == ["progress"] * 3 + ["done"]
    assert [int(e["id"]) for e in events] == [0, 1, 2, 3]
    assert [e["data"] for e in events[:3]] == EVAC_TRACE
    assert events[-1]["data"] == {**EVAC_SA, "source": "cached"}


@pytest.mark.req("APP-05")
def test_evac_run_no_cache_503_and_unknown_404(offline_client):
    assert offline_client.get("/api/evac/alpha/run").status_code == 503
    assert offline_client.get("/api/evac/nope/run").status_code == 404


# ----------------------------------------------------------------- APP-06
@pytest.mark.req("APP-06")
def test_dispatch_live(client, fake):
    r = client.post("/api/dispatch", json={"scenario": SID, "incidents": [[-97.5, 35.5], [-97.45, 35.45]]})
    assert r.status_code == 200
    body = r.json()
    for k, v in DISPATCH_RESULT.items():
        assert body[k] == v
    assert body["source"] == "live"
    assert len(fake.dispatch_calls) == 1 and isinstance(fake.dispatch_calls[0], DispatchRequest)
    assert len(fake.dispatch_calls[0].incidents) == 2


@pytest.mark.req("APP-06")
@pytest.mark.parametrize(
    "incidents",
    [[], [[-97.5, 35.5]] * 4, [[-96.0, 36.0]], [[-97.5, 95.0]]],
)
def test_dispatch_validation_422(client, fake, incidents):
    r = client.post("/api/dispatch", json={"scenario": SID, "incidents": incidents})
    assert r.status_code == 422
    assert fake.dispatch_calls == []


@pytest.mark.req("APP-06")
def test_dispatch_unknown_and_no_backend(client, offline_client, bundle_root, reports_root):
    assert (
        client.post("/api/dispatch", json={"scenario": "nope", "incidents": [[-97.5, 35.5]]}).status_code
        == 404
    )
    body = {"scenario": SID, "incidents": [[-97.5, 35.5]]}
    assert offline_client.post("/api/dispatch", json=body).status_code == 503
    failing = TestClient(
        create_app(scenarios_dir=bundle_root, reports_dir=reports_root, backend=FailingBackend(), demo=False)
    )
    assert failing.post("/api/dispatch", json=body).status_code == 503


# ----------------------------------------------------------------- APP-07
@pytest.mark.req("APP-07")
def test_city_live_allows_network_outside_demo(client, fake):
    r = client.get("/api/city", params={"name": "  Tulsa, OK  "})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Tulsa, OK" and body["n_nodes"] == 100 and body["cached"] is True
    assert fake.city_calls == [("Tulsa, OK", True)]


@pytest.mark.req("APP-07")
def test_city_demo_mode_forbids_network(bundle_root, reports_root, fake):
    c = TestClient(create_app(scenarios_dir=bundle_root, reports_dir=reports_root, backend=fake, demo=True))
    assert c.get("/api/city", params={"name": "Norman, OK"}).status_code == 200
    assert fake.city_calls == [("Norman, OK", False)]


@pytest.mark.req("APP-07")
@pytest.mark.parametrize("params", [{}, {"name": ""}, {"name": "   "}, {"name": "x" * 101}])
def test_city_validation_422(client, fake, params):
    assert client.get("/api/city", params=params).status_code == 422
    assert fake.city_calls == []


@pytest.mark.req("APP-07")
def test_city_lookup_error_404_and_no_backend_503(client, fake, offline_client):
    fake.city_raises = LookupError("Atlantis not cached")
    r = client.get("/api/city", params={"name": "Atlantis"})
    assert r.status_code == 404 and "Atlantis not cached" in r.json()["detail"]
    assert offline_client.get("/api/city", params={"name": "Tulsa, OK"}).status_code == 503


# ----------------------------------------------------------------- APP-08
@pytest.mark.req("APP-08")
def test_metrics_merged(client, reports_root):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "metrics": {"a_routing": {"gap_closed": 0.6}, "storm": {"csi": 0.41, "baseline_csi": 0.33}},
        "invalid": ["bad.json"],
    }
    assert list(body["metrics"]) == ["a_routing", "storm"]
    assert bundle_mod.read_metrics(reports_root) == body


@pytest.mark.req("APP-08")
def test_metrics_missing_folder(tmp_path, bundle_root):
    assert bundle_mod.read_metrics(tmp_path / "noreports") == {"metrics": {}, "invalid": []}
    c = TestClient(create_app(scenarios_dir=bundle_root, reports_dir=tmp_path / "noreports", demo=False))
    assert c.get("/api/metrics").json() == {"metrics": {}, "invalid": []}


# ----------------------------------------------------------------- APP-09
@pytest.mark.req("APP-09")
def test_index_and_assets_served(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert r.text == (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


@pytest.mark.req("APP-09")
def test_index_structure_and_relative_assets():
    import re

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'id="map"' in html and 'id="city-box"' in html
    tabs = set(re.findall(r'data-tab="([a-z_-]+)"', html))
    assert {"forecast", "route", "shelters", "evac", "dispatch", "results"} <= tabs
    refs = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert refs, "index.html must reference its assets"
    for ref in refs:
        assert not ref.startswith("/"), f"asset path must be relative: {ref}"
    assert "static/app.js" in refs


@pytest.mark.req("APP-09")
@pytest.mark.parametrize("name", ["index.html", "app.js", "style.css"])
def test_no_external_urls(name):
    import re

    text = (STATIC_DIR / name).read_text(encoding="utf-8")
    assert "http://" not in text and "https://" not in text
    assert not re.search(r"""(?:src|href|url\()\s*=?\s*["']?//""", text)


# ----------------------------------------------------------------- APP-10
@pytest.mark.req("APP-10")
def test_server_works_with_network_blocked(no_network, bundle_root, reports_root):
    c = TestClient(create_app(scenarios_dir=bundle_root, reports_dir=reports_root, backend=None, demo=True))
    for url in [
        "/",
        "/static/app.js",
        "/api/health",
        "/api/scenarios",
        f"/api/scenarios/{SID}/overlay/storm/0.png",
        f"/api/shelters/{SID}",
        f"/api/evac/{SID}",
        f"/api/evac/{SID}/run",
        "/api/metrics",
    ]:
        assert c.get(url).status_code == 200, url
    assert c.post("/api/route", json=_route_body()).status_code == 200


@pytest.mark.req("APP-10")
def test_server_import_pulls_no_network_clients():
    code = (
        "import sys; before=set(sys.modules); import outrun.app.server; "
        "bad=[m for m in ('osmnx','herbie','boto3','requests') if m in sys.modules and m not in before]; "
        "print(','.join(bad))"
    )
    src = str(Path(__file__).resolve().parents[2] / "src")
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PYTHONPATH": src, "PATH": ""},
        check=True,
    )
    assert out.stdout.strip() == ""


# ----------------------------------------------------------------- APP-11
@pytest.mark.req("APP-11")
def test_cached_route_closest_preset_wins(offline_client):
    r = offline_client.post("/api/route", json=_route_body())  # 44 m from A, 122 m from B
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "A" and body["source"] == "cached" and body["scenario"] == SID
    assert body["cached_origin"] == pytest.approx([-97.50, 35.50])
    r = offline_client.post("/api/route", json=_route_body(origin=[-97.50, 35.5012]))  # nearer B
    assert r.json()["label"] == "B"


@pytest.mark.req("APP-11")
def test_cached_route_coordinate_dest_and_profile(offline_client):
    r = offline_client.post(
        "/api/route", json=_route_body(origin=[-97.4505, 35.4502], dest=[-97.4202, 35.4701])
    )
    assert r.status_code == 200 and r.json()["label"] == "C"
    r = offline_client.post("/api/route", json=_route_body(origin=[-97.55, 35.55], profile="ev"))
    assert r.status_code == 200 and r.json()["label"] == "D"


@pytest.mark.req("APP-11")
@pytest.mark.parametrize(
    "over",
    [
        {"origin": [-97.50, 35.51]},  # ~1.1 km from every preset
        {"profile": "ev"},  # A/B are civilian
        {"dest": [-97.42, 35.47]},  # A/B are nearest_shelter
        {"origin": [-97.4505, 35.4502], "dest": [-97.43, 35.47]},  # C's dest ~900 m off
    ],
)
def test_cached_route_no_match_503(offline_client, over):
    assert offline_client.post("/api/route", json=_route_body(**over)).status_code == 503


@pytest.mark.req("APP-11")
def test_cached_route_when_backend_fails_or_no_file(bundle_root, reports_root, offline_client):
    c = TestClient(
        create_app(scenarios_dir=bundle_root, reports_dir=reports_root, backend=FailingBackend(), demo=False)
    )
    r = c.post("/api/route", json=_route_body())
    assert r.status_code == 200 and r.json()["source"] == "cached" and r.json()["label"] == "A"
    alpha = offline_client.post("/api/route", json=_route_body(scenario="alpha"))
    assert alpha.status_code == 503


# ----------------------------------------------------------------- APP-12
@pytest.mark.req("APP-12")
def test_format_sse_exact():
    assert format_sse("progress", {"objective": 1.5, "iter": 2}, id=7) == (
        'event: progress\nid: 7\ndata: {"iter":2,"objective":1.5}\n\n'
    )
    assert format_sse("done", [1, 2]) == "event: done\ndata: [1,2]\n\n"
    assert (
        format_sse("sa_error", {"detail": "x\ny"}, id=0)
        == 'event: sa_error\nid: 0\ndata: {"detail":"x\\ny"}\n\n'
    )


@pytest.mark.req("APP-12")
@pytest.mark.parametrize(
    "event,id_",
    [("Progress", None), ("1x", None), ("", None), ("a b", None), ("ok", -1), ("ok", "3"), ("ok", 1.5)],
)
def test_format_sse_rejects_bad_input(event, id_):
    with pytest.raises(ValueError):
        format_sse(event, {}, id=id_)


# ----------------------------------------------------------------- APP-13
def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.req("APP-13")
def test_export_static_layout(tmp_path, bundle_root, reports_root):
    out = tmp_path / "docs"
    written = export_static(bundle_root, reports_root, out)
    assert written == sorted(written)
    expected = {
        "index.html",
        "config.json",
        ".nojekyll",
        "static/app.js",
        "static/style.css",
        "data/scenarios.json",
        "data/metrics.json",
        f"data/scenarios/{SID}/overlay/model/storm/00.png",
        f"data/scenarios/{SID}/overlay/model/storm/11.png",
        f"data/scenarios/{SID}/overlay/baseline/storm/00.png",
        f"data/scenarios/{SID}/overlay/observed/tornado/03.png",
        "data/scenarios/alpha/overlay/model/flood/00.png",
        f"data/scenarios/{SID}/shelters_mcf.json",
        f"data/scenarios/{SID}/shelters_nearest.json",
        f"data/scenarios/{SID}/evac_sa.json",
        f"data/scenarios/{SID}/evac_sa_trace.json",
        f"data/scenarios/{SID}/routes_preset.json",
    }
    assert expected <= set(written)
    assert "static/index.html" not in written and ".outrun-static.json" not in written
    assert not any("Bad.Id" in w or "broken" in w for w in written)
    assert (out / "index.html").read_bytes() == (STATIC_DIR / "index.html").read_bytes()
    assert json.loads((out / "config.json").read_text()) == {"mode": "static"}
    assert json.loads((out / "data" / "scenarios.json").read_text()) == bundle_mod.list_scenarios(bundle_root)
    assert json.loads((out / "data" / "metrics.json").read_text()) == bundle_mod.read_metrics(reports_root)
    src_png = bundle_root / SID / "overlays" / "observed" / "tornado" / "03.png"
    assert (
        out / f"data/scenarios/{SID}/overlay/observed/tornado/03.png"
    ).read_bytes() == src_png.read_bytes()
    assert json.loads((out / f"data/scenarios/{SID}/shelters_mcf.json").read_text()) == SHELTERS_MCF
    manifest = json.loads((out / ".outrun-static.json").read_text())
    assert manifest == {"files": written}
    for w in written:
        assert (out / w).is_file(), w


@pytest.mark.req("APP-13")
def test_export_static_idempotent_and_preserves_foreign_files(tmp_path, bundle_root, reports_root):
    out = tmp_path / "docs"
    out.mkdir()
    (out / "DESIGN.md").write_text("design doc")
    first = export_static(bundle_root, reports_root, out)
    snap = _tree(out)
    second = export_static(bundle_root, reports_root, out)
    assert second == first
    assert _tree(out) == snap
    assert (out / "DESIGN.md").read_text() == "design doc"


@pytest.mark.req("APP-13")
def test_export_static_refuses_to_overwrite_foreign_file(tmp_path, bundle_root, reports_root):
    out = tmp_path / "docs"
    out.mkdir()
    (out / "index.html").write_text("someone else's page")
    with pytest.raises(FileExistsError):
        export_static(bundle_root, reports_root, out)
    assert (out / "index.html").read_text() == "someone else's page"
    assert sorted(p.name for p in out.iterdir()) == ["index.html"]


@pytest.mark.req("APP-13")
def test_export_static_removes_only_stale_owned_files(tmp_path, bundle_root, reports_root):
    out = tmp_path / "docs"
    export_static(bundle_root, reports_root, out)
    (out / "notes.txt").write_text("keep me")
    (bundle_root / SID / "results" / "shelters_nearest.json").unlink()
    written = export_static(bundle_root, reports_root, out)
    assert f"data/scenarios/{SID}/shelters_nearest.json" not in written
    assert not (out / f"data/scenarios/{SID}/shelters_nearest.json").exists()
    assert (out / "notes.txt").read_text() == "keep me"


# ----------------------------------------------------------------- APP-14
@pytest.mark.req("APP-14")
@pytest.mark.parametrize("bad", ["..", ".hidden", "Bad.Id", "..%2F..%2Fsecret", "a" * 65])
def test_invalid_ids_404_everywhere(client, tmp_path, bad):
    secret = tmp_path / "secret"
    (secret / "results").mkdir(parents=True)
    (secret / "results" / "shelters_mcf.json").write_text(json.dumps({"secret": True}))
    for url in [
        f"/api/scenarios/{bad}/overlay/storm/0.png",
        f"/api/shelters/{bad}",
        f"/api/evac/{bad}",
        f"/api/evac/{bad}/run",
    ]:
        r = client.get(url)
        assert r.status_code == 404, url
        assert b"secret" not in r.content
    r = client.post("/api/route", json=_route_body(scenario=bad))
    assert r.status_code in (404, 422) and b"secret" not in r.content


@pytest.mark.req("APP-14")
def test_scenario_dir_validation(tmp_path):
    assert bundle_mod.scenario_dir(tmp_path, "el-reno_2019") == tmp_path / "el-reno_2019"
    for bad in ("..", "../x", ".hidden", "Bad.Id", "", "a/b", "a" * 65, "-lead"):
        with pytest.raises(ValueError):
            bundle_mod.scenario_dir(tmp_path, bad)
    assert bundle_mod.is_valid_scenario_id("a" * 64)
    assert not bundle_mod.is_valid_scenario_id("a" * 65)


# ----------------------------------------------------------------- APP-15
@pytest.mark.req("APP-15")
def test_health(client, offline_client):
    assert client.get("/api/health").json() == {"status": "ok", "demo": False, "live": True, "n_scenarios": 2}
    assert offline_client.get("/api/health").json() == {
        "status": "ok",
        "demo": True,
        "live": False,
        "n_scenarios": 2,
    }


@pytest.mark.req("APP-15")
@pytest.mark.parametrize(
    "env,explicit,expected",
    [("1", None, True), ("0", None, False), (None, None, False), ("1", False, False), (None, True, True)],
)
def test_demo_flag_from_env(monkeypatch, bundle_root, reports_root, env, explicit, expected):
    if env is None:
        monkeypatch.delenv("OUTRUN_DEMO", raising=False)
    else:
        monkeypatch.setenv("OUTRUN_DEMO", env)
    c = TestClient(create_app(scenarios_dir=bundle_root, reports_dir=reports_root, demo=explicit))
    assert c.get("/api/health").json()["demo"] is expected


# ---------------------------------------------------------------- APP-03 PlanResult -> response fields
def _mid_road_case(block):
    import numpy as np
    from tests.fixtures.graphs.graphfix import make_road_graph

    from outrun.routing.cost import CostModel

    src, dst = [0, 1, 0, 2], [1, 3, 2, 3]  # top 0-1-3 (100+100 s), mid 0-2-3 (60+60 s)
    g = make_road_graph(
        [-97.50, -97.49, -97.49, -97.48],
        [35.50, 35.51, 35.49, 35.50],
        src,
        dst,
        [1000.0, 1000.0, 600.0, 600.0],
        [10.0] * 4,
    )[0]
    risk = np.zeros((g.n_edges, 1, 12))
    mid = [e for e in range(g.n_edges) if (int(g.src[e]), int(g.dst[e])) in {(0, 2), (2, 3)}]
    risk[mid, 0, :] = 0.9
    m = CostModel(g.free_flow_s, risk, np.array([1.0]), np.array([0.5 if block else float("nan")]))
    return g, m, mid


def _route(g, nodes, t=0.0):
    from outrun.routing.astar import Route

    edges, depart = [], []
    for u, v in zip(nodes[:-1], nodes[1:], strict=True):
        e = next(e for e in g.out_edges(u) if int(g.dst[e]) == v)
        edges.append(e)
        depart.append(t)
        t += float(g.free_flow_s[e])
    return Route(nodes=nodes, edges=edges, depart_s=depart, waits_s=[0.0] * len(edges), start_s=0.0, arrive_s=t,
                 cost=t, expansions=1)  # fmt: skip


@pytest.mark.req("APP-03")
def test_route_fields_mapping_from_plan_result():
    from outrun.app.backend import route_fields
    from outrun.routing.planner import PlanResult

    g, m, mid = _mid_road_case(block=True)
    safe = _route(g, [0, 1, 3])
    short = _route(g, [0, 2, 3])
    out = route_fields(PlanResult(kind="route", route=safe), short, m, None, g)
    assert out["shelter_in_place"] is False
    assert out["outrun"]["type"] == "Feature" and out["outrun"]["geometry"]["type"] == "LineString"
    assert out["outrun"]["geometry"]["coordinates"] == [[float(g.lon[n]), float(g.lat[n])] for n in [0, 1, 3]]
    assert out["shortest"]["geometry"]["coordinates"][1] == [float(g.lon[2]), float(g.lat[2])]
    assert out["eta_min"] == pytest.approx(200.0 / 60.0)
    assert out["blocked_avoided"] == 2  # both mid-road edges are blocked under the planning model
    assert out["exposure"]["predicted"] == pytest.approx(0.0) and out["exposure"]["observed"] is None
    json.dumps(out)  # JSON-serialisable

    sip = route_fields(PlanResult(kind="shelter_in_place", route=None), short, m, m, g)
    assert sip["shelter_in_place"] is True and sip["outrun"] is None and sip["eta_min"] is None
    assert sip["exposure"] == {"predicted": None, "observed": None}
    hard = route_fields(PlanResult(kind="hardened_facility", route=safe, facility=3), short, m, m, g)
    assert hard["shelter_in_place"] is True and hard["outrun"]["properties"]["facility"] == 3
    nr = route_fields(PlanResult(kind="no_route", route=None), None, m, None, g)
    assert nr["outrun"] is None and nr["shortest"] is None and nr["blocked_avoided"] == 0
    g2, m2, _ = _mid_road_case(block=False)
    assert (
        route_fields(
            PlanResult(kind="route", route=_route(g2, [0, 2, 3])), _route(g2, [0, 2, 3]), m2, m2, g2
        )["blocked_avoided"]
        == 0
    )


# ---------------------------------------------------------------- APP-17 vendored map libraries are used
@pytest.mark.req("APP-17")
def test_vendored_maplibre_and_pmtiles_are_loaded_and_used(offline_client):
    import re

    vendor = STATIC_DIR / "vendor"
    for name in ("maplibre-gl.js", "maplibre-gl.css", "pmtiles.js"):
        f = vendor / name
        assert f.is_file() and f.stat().st_size > 10_000, f"{name} must be the real vendored library"
    assert offline_client.get("/static/vendor/maplibre-gl.js").status_code == 200
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert re.search(r"<script[^>]*\bsrc=\"static/vendor/maplibre-gl\.js\"", html)
    assert re.search(r"<link[^>]*\bhref=\"static/vendor/maplibre-gl\.css\"", html)
    assert re.search(r"<script[^>]*\bsrc=\"static/vendor/pmtiles\.js\"", html)
    assert re.search(r"<script[^>]*\bsrc=\"static/app\.js\"", html)
    js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    assert "new maplibregl.Map(" in js and "window.outrunMap" in js and "window.outrunState" in js
    for api in ("/api/scenarios", "/overlay/", "/api/route", "/api/shelters/", "/api/evac/", "/api/dispatch",
                "/api/city", "/api/metrics"):  # fmt: skip
        assert api in js, f"app.js never calls {api}"
