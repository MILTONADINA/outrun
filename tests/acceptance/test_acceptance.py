"""specs/16-acceptance.md: real data + trained models, read from the files `outrun eval` writes in T24. Opt-in:

    OUTRUN_ACCEPTANCE=1 .venv/bin/python -m pytest tests/acceptance -m acceptance

Missing inputs -> skip with a message (never a false failure). Target rows record misses as xfail with numbers.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.acceptance

ROOT = Path(__file__).resolve().parents[2]
REPORTS = Path(os.environ.get("OUTRUN_REPORTS", ROOT / "reports"))
METRICS = REPORTS / "metrics"
SPLITS = Path(os.environ.get("OUTRUN_SPLITS", ROOT / "splits"))
HAZARDS = ("storm", "tornado", "flood", "wildfire", "ice", "smoke")
PRIMARY = {
    "storm": "csi_mean_74_133",
    "tornado": "auc_tor_vs_warned_null",
    "flood": "auprc",
    "wildfire": "prauc",
    "ice": "zone_auprc",
    "smoke": "rmse_smoke_episodes",
}


def _enabled() -> None:
    if os.environ.get("OUTRUN_ACCEPTANCE") != "1":
        pytest.skip("set OUTRUN_ACCEPTANCE=1 to run acceptance tests")


def _need(path: Path, how: str = "run `outrun eval` (T24)") -> Path:
    _enabled()
    if not path.exists():
        pytest.skip(f"missing {path}: {how}")
    return path


def _load(path: Path) -> dict:
    return json.loads(_need(path).read_text(encoding="utf-8"))


def _hazard(h: str) -> dict:
    return _load(METRICS / f"{h}.json")


def _check_split(d: dict, split: str, need_ci: bool = True) -> None:
    """dl and baseline rows need a finite primary estimate (and, with need_ci, finite ordered CI bounds).
    `reference` rows (published numbers) carry no CI and are not checked."""
    assert split in d["splits"], f"{d['hazard']}: no '{split}' split"
    models = d["splits"][split]
    kinds = [m["kind"] for m in models.values()]
    assert "dl" in kinds and "baseline" in kinds, (d["hazard"], split, kinds)
    for name, m in models.items():
        if m["kind"] not in ("dl", "baseline"):
            continue
        ci = m["metrics"][d["primary_metric"]]
        assert math.isfinite(ci["estimate"]), (name, ci)
        if need_ci:
            assert math.isfinite(ci["low"]) and math.isfinite(ci["high"]), (name, ci)
            assert ci["low"] <= ci["high"], (name, ci)
            assert ci["n_resamples"] >= 1, (name, ci)


def _check_hazard_file(h: str) -> dict:
    d = _hazard(h)
    assert d["hazard"] == h
    assert d["git_sha"], "metrics must be logged with the git sha"
    assert d["primary_metric"] == PRIMARY[h], d["primary_metric"]
    assert d["n_resamples"] >= 1000, "DESIGN 6.1: 1,000 day-block resamples"
    _check_split(d, "test")
    return d


@pytest.mark.req("AC-01")
def test_ac01_storm_metrics():
    d = _check_hazard_file("storm")
    assert d["higher_is_better"] is True


@pytest.mark.req("AC-02")
def test_ac02_tornado_metrics():
    _check_hazard_file("tornado")


@pytest.mark.req("AC-03")
def test_ac03_flood_metrics():
    _check_hazard_file("flood")


@pytest.mark.req("AC-04")
def test_ac04_wildfire_metrics_and_wsts2021():
    d = _check_hazard_file("wildfire")
    _check_split(d, "test_wsts2021")


@pytest.mark.req("AC-05")
def test_ac05_ice_metrics():
    _check_hazard_file("ice")


@pytest.mark.req("AC-06")
def test_ac06_smoke_metrics():
    d = _check_hazard_file("smoke")
    assert d["higher_is_better"] is False


@pytest.mark.req("AC-07")
def test_ac07_case_holdout_reported():
    _enabled()
    files = [METRICS / f"{h}.json" for h in HAZARDS if (METRICS / f"{h}.json").exists()]
    if not files:
        pytest.skip(f"no hazard metrics in {METRICS}: run `outrun eval` (T24)")
    with_case = [json.loads(f.read_text()) for f in files]
    with_case = [d for d in with_case if "case_holdout" in d["splits"]]
    assert with_case, "no hazard reports a separate case_holdout split"
    for d in with_case:
        _check_split(d, "case_holdout", need_ci=False)  # a short case window may give NaN CI bounds


@pytest.mark.req("AC-08")
@pytest.mark.parametrize(("h", "ablation"), [("storm", "vil_only"), ("smoke", "no_hms")])
def test_ac08_satellite_ablations(h, ablation):
    d = _hazard(h)
    assert ablation in d.get("ablations", {}), f"{h}: ablation '{ablation}' not reported"
    assert any(m["kind"] == "dl" for m in d["ablations"][ablation].values())


@pytest.mark.req("AC-09")
@pytest.mark.parametrize("h", HAZARDS)
def test_ac09_dl_vs_best_baseline(h):
    d = _hazard(h)
    pm, hib = d["primary_metric"], d["higher_is_better"]
    models = d["splits"]["test"]
    sign = 1.0 if hib else -1.0
    dl = max(
        (m["metrics"][pm]["estimate"] for m in models.values() if m["kind"] == "dl"), key=lambda v: sign * v
    )
    bl_name, bl = max(
        ((n, m["metrics"][pm]["estimate"]) for n, m in models.items() if m["kind"] == "baseline"),
        key=lambda t: sign * t[1],
    )
    if not sign * dl > sign * bl:
        pytest.xfail(f"{h}: DL {pm}={dl:.4f} does not beat best baseline {bl_name}={bl:.4f} (report it)")


def _primary_routing() -> dict:
    _enabled()
    files = sorted(METRICS.glob("routing_*.json"))
    prim = [json.loads(f.read_text()) for f in files]
    prim = [r for r in prim if r.get("primary")]
    if not prim:
        pytest.skip(f"no routing_*.json with primary=true in {METRICS}: run `outrun eval` (T24)")
    assert len(prim) == 1, "exactly one primary routing case"
    return prim[0]


@pytest.mark.req("AC-10")
def test_ac10_routing_eval_exists():
    r = _primary_routing()
    assert set(r["planners"]) == {"a", "a_prime", "b", "c", "d"}
    assert r["n_paired"] >= 100, r["n_paired"]
    g = r["gap_closed"]
    assert all(math.isfinite(g[k]) for k in ("estimate", "low", "high")), g
    assert r["git_sha"]


@pytest.mark.req("AC-11")
def test_ac11_routing_exposure_reduction():
    r = _primary_routing()
    g = r["gap_closed"]
    e_a = r["planners"]["a"]["mean_exposure_risk_min"]
    e_d = r["planners"]["d"]["mean_exposure_risk_min"]
    if not (g["estimate"] > 0 and e_d <= e_a):
        pytest.xfail(
            f"gap closed {g['estimate']:.3f} [{g['low']:.3f}, {g['high']:.3f}], E_a={e_a:.2f}, E_d={e_d:.2f} (report it)"
        )


def _latency(name: str) -> float:
    d = _load(METRICS / "latency.json")
    if name not in d:
        pytest.skip(f"latency.json has no '{name}' entry: run `outrun eval` (T24)")
    return float(d[name]["median_s"])


@pytest.mark.req("AC-12")
def test_ac12_astar_latency():
    m = _latency("astar_metro_query")
    if not m < 1.0:
        pytest.xfail(f"A* median query {m:.3f} s >= 1 s target (report it)")


@pytest.mark.req("AC-13")
def test_ac13_sa_runtime():
    m = _latency("sa_run")
    if not m < 60.0:
        pytest.xfail(f"SA median run {m:.1f} s >= 60 s target (report it)")


@pytest.mark.req("AC-14")
def test_ac14_splits_frozen_and_test_run_once():
    _need(SPLITS / "hashes.json", "freeze splits (T7)")
    from outrun.eval.splits import verify_hashes

    assert verify_hashes(SPLITS) == [], "frozen splits changed after T7"
    ledger = _load(REPORTS / "test_ledger.json")
    for h in HAZARDS:
        path = METRICS / f"{h}.json"
        if not path.exists():
            pytest.skip(f"missing {path}: run `outrun eval` (T24)")
        d = json.loads(path.read_text())
        for name, m in d["splits"]["test"].items():
            if m["kind"] != "dl":
                continue
            shas = [e["git_sha"] for e in ledger if e["model"] == name and e["split"] == "test"]
            assert shas, f"{name}: no test run in the ledger"
            assert d["git_sha"] in shas, f"{name}: metrics sha {d['git_sha']} not in ledger {shas}"


@pytest.mark.req("AC-15")
@pytest.mark.parametrize(
    ("place", "origin", "dest"),
    [
        ("Tulsa, OK", ("-95.993", "36.154"), ("-95.937", "36.104")),
        ("Lawrence, KS", ("-95.236", "38.972"), ("-95.262", "38.940")),
    ],
)
def test_ac15_any_us_city(place, origin, dest):
    _enabled()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "outrun", "route", "--place", place, "--origin", *origin, "--dest", *dest,
             "--risk-source", "none"],
            capture_output=True, text=True, timeout=900, cwd=ROOT,
        )  # fmt: skip
    except subprocess.TimeoutExpired:
        pytest.fail(f"outrun route --place {place!r} timed out")
    if proc.returncode != 0 and any(s in proc.stderr.lower() for s in ("connection", "network", "resolve")):
        pytest.skip(f"no OSM access and no cached graph for {place!r}")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip(), "route command printed nothing"


# ---------------------------------------------------------------- AC-16..18 group metrics (EVAL-22)
def _primary_group() -> dict:
    _enabled()
    files = sorted(METRICS.glob("group_*.json")) if METRICS.exists() else []
    prim = [json.loads(f.read_text()) for f in files]
    prim = [g for g in prim if g.get("primary")]
    if not prim:
        pytest.skip(f"no group_*.json with primary=true in {METRICS}: run `outrun eval` (T24)")
    assert len(prim) == 1, "exactly one primary group-metrics case"
    assert prim[0]["git_sha"]
    return prim[0]


@pytest.mark.req("AC-16")
def test_ac16_shelter_assignment_real_case():
    s = _primary_group()["shelters"]
    assert {"mcf", "nearest", "greedy"} <= set(s)
    assert s["mcf"]["overflow_persons"] <= s["greedy"]["overflow_persons"]
    assert s["mcf"]["unserved_persons"] <= s["greedy"]["unserved_persons"]


@pytest.mark.req("AC-17")
def test_ac17_city_evacuation_real_case():
    ev = _primary_group()["evacuation"]
    methods = ev["methods"]
    assert {"sa", "msa", "shortest", "safest", "sa_lambda0"} <= set(methods)
    assert len(ev["seeds"]) == 5 and all(len(methods[m]) == 5 for m in methods)
    assert all(math.isfinite(r["clearance_min"]) for r in methods["sa"])
    mean_j = sum(r["J"] for r in methods["sa"]) / len(methods["sa"])
    assert ev["lower_bound"] <= mean_j * (1 + 1e-6), (ev["lower_bound"], mean_j)


@pytest.mark.req("AC-18")
def test_ac18_emergency_vehicles_real_case():
    ems = _primary_group()["ems"]
    for k in ("hungarian", "reroute_baseline"):
        assert math.isfinite(ems[k]["mean_eta_min"]), k
    assert len(ems["pareto"]) >= 1


# ---------------------------------------------------------------- AC-19 interactive map app on the real bundle
@pytest.mark.req("AC-19")
def test_ac19_app_serves_primary_bundle():
    r = _primary_routing()
    sid = r["scenario"]
    scen_root = Path(os.environ.get("OUTRUN_SCENARIOS", ROOT / "scenarios"))
    d = _need(scen_root / sid / "scenario.json", "run `outrun scenario` (T25)").parent
    pngs = sorted((d / "overlays").rglob("*.png"))
    assert pngs, f"no overlay PNG in {d}"
    presets = json.loads(_need(d / "results" / "routes_preset.json").read_text())
    assert presets, "routes_preset.json is empty"
    from fastapi.testclient import TestClient

    from outrun.app.server import STATIC_DIR, create_app

    for name in ("maplibre-gl.js", "maplibre-gl.css", "pmtiles.js"):
        assert (STATIC_DIR / "vendor" / name).stat().st_size > 10_000, name
    c = TestClient(create_app(scenarios_dir=scen_root, reports_dir=REPORTS, backend=None, demo=True))
    lst = c.get("/api/scenarios")
    assert lst.status_code == 200 and sid in json.dumps(lst.json())
    src, hz, slot = pngs[0].relative_to(d / "overlays").parts
    png = c.get(f"/api/scenarios/{sid}/overlay/{hz}/{int(slot[:2])}.png", params={"source": src})
    assert png.status_code == 200
    p0 = presets[0]
    body = {"scenario": sid, "origin": p0["origin"], "dest": p0["dest"], "profile": p0["profile"]}
    assert c.post("/api/route", json=body).status_code == 200
