"""specs/15-evaluation.md: report writer, test ledger, latency (EVAL-16..EVAL-21). Files go to tmp_path only."""

from __future__ import annotations

import json
import math
import time

import pytest

from outrun.eval.bootstrap import CI
from outrun.eval.report import (
    HAZARDS,
    HeldOutReuseError,
    LatencyStats,
    ModelMetrics,
    RunLedger,
    format_ci,
    load_model_metrics,
    time_calls,
    write_latency,
    write_metrics_md,
    write_model_metrics,
    write_routing_metrics,
)
from outrun.eval.routing_eval import PLANNER_ORDER, PlannerSummary, RoutingEvalResult

NAN = float("nan")


def ci(est: float, lo: float | None = None, hi: float | None = None) -> CI:
    return CI(est, est - 0.02 if lo is None else lo, est + 0.02 if hi is None else hi, 1000, 120)


def _storm_results(dl: float = 0.40, base: float = 0.35) -> dict[str, list[ModelMetrics]]:
    return {
        "test": [
            ModelMetrics("storm.unet_vil_ir", "dl", {"csi_mean_74_133": ci(dl), "fss": ci(0.6)}),
            ModelMetrics("persistence", "baseline", {"csi_mean_74_133": ci(0.30)}),
            ModelMetrics("pysteps", "baseline", {"csi_mean_74_133": ci(base), "fss": ci(0.55)}),
        ],
        "case_holdout": [
            ModelMetrics("storm.unet_vil_ir", "dl", {"csi_mean_74_133": ci(0.20)}),
            ModelMetrics("pysteps", "baseline", {"csi_mean_74_133": ci(0.25)}),
        ],
    }


def _routing_result(gap: CI) -> RoutingEvalResult:
    summ = PlannerSummary(500, 0, 10.0, 3.0, 1.1, 0, None)
    return RoutingEvalResult(
        n_origins=500,
        n_paired=500,
        summaries={k: summ for k in PLANNER_ORDER},
        gap_closed=gap,
        c_minus_b=ci(-1.0),
        c_minus_a_prime=ci(-2.0),
    )


# ---------------------------------------------------------------- EVAL-16
@pytest.mark.req("EVAL-16")
def test_format_ci():
    assert format_ci(CI(0.512, 0.48, 0.54, 1000, 10)) == "0.512 [0.480, 0.540]"
    assert format_ci(CI(0.5, 0.4, 0.6, 1, 2), digits=1) == "0.5 [0.4, 0.6]"
    assert format_ci({"estimate": 0.512, "low": 0.48, "high": 0.54, "n_resamples": 9, "n_blocks": 3}) == (
        "0.512 [0.480, 0.540]"
    )
    assert format_ci(CI(NAN, NAN, NAN, 0, 0)) == "n/a"
    assert format_ci(CI(0.512, NAN, NAN, 0, 1)) == "0.512 [n/a]"


# ---------------------------------------------------------------- EVAL-17
@pytest.mark.req("EVAL-17")
def test_write_model_metrics_schema(tmp_path):
    out = tmp_path / "reports" / "metrics"
    p = write_model_metrics(
        out, "storm", _storm_results(), "csi_mean_74_133", True, "abc123",
        ablations={"vil_only": [ModelMetrics("storm.unet_vil", "dl", {"csi_mean_74_133": ci(0.38)})]},
    )  # fmt: skip
    assert p == out / "storm.json" and p.exists()
    d = load_model_metrics(p)
    assert d == json.loads(p.read_text())
    assert d["schema"] == 1 and d["hazard"] == "storm" and d["git_sha"] == "abc123"
    assert d["primary_metric"] == "csi_mean_74_133" and d["higher_is_better"] is True
    assert d["n_resamples"] == 1000 and d["created_utc"].endswith("Z")
    assert set(d["splits"]) == {"test", "case_holdout"}
    m = d["splits"]["test"]["storm.unet_vil_ir"]
    assert m["kind"] == "dl"
    assert m["metrics"]["csi_mean_74_133"] == pytest.approx(
        {"estimate": 0.40, "low": 0.38, "high": 0.42, "n_resamples": 1000, "n_blocks": 120}
    )
    assert d["ablations"]["vil_only"]["storm.unet_vil"]["kind"] == "dl"
    assert "storm" in HAZARDS and len(HAZARDS) == 6


@pytest.mark.req("EVAL-17")
def test_write_model_metrics_accepts_test_suffix_split(tmp_path):
    res = {
        "test": [ModelMetrics("fire.unet", "dl", {"prauc": ci(0.3)}),
                 ModelMetrics("persistence", "baseline", {"prauc": ci(0.2)}),
                 ModelMetrics("ndws_published", "reference", {"prauc": ci(0.284)})],
        "test_wsts2021": [ModelMetrics("fire.unet", "dl", {"prauc": ci(0.25)}),
                          ModelMetrics("persistence", "baseline", {"prauc": ci(0.2)})],
    }  # fmt: skip
    d = load_model_metrics(write_model_metrics(tmp_path, "wildfire", res, "prauc", True, "sha1"))
    assert d["ablations"] == {}
    assert set(d["splits"]) == {"test", "test_wsts2021"}


@pytest.mark.req("EVAL-17")
@pytest.mark.parametrize(
    "case",
    [
        "bad_hazard",
        "empty_sha",
        "no_test",
        "bad_split",
        "no_baseline",
        "no_dl",
        "missing_primary",
        "dup_model",
    ],
)
def test_write_model_metrics_rejects_invalid(tmp_path, case):
    res = _storm_results()
    kwargs = dict(hazard="storm", primary_metric="csi_mean_74_133", higher_is_better=True, git_sha="abc")
    if case == "bad_hazard":
        kwargs["hazard"] = "hail"
    elif case == "empty_sha":
        kwargs["git_sha"] = ""
    elif case == "no_test":
        res = {"case_holdout": res["case_holdout"]}
    elif case == "bad_split":
        res["train"] = res["test"]
    elif case == "no_baseline":
        res["test"] = [m for m in res["test"] if m.kind != "baseline"]
    elif case == "no_dl":
        res["test"] = [m for m in res["test"] if m.kind != "dl"]
    elif case == "missing_primary":
        res["test"] = res["test"] + [ModelMetrics("lr", "baseline", {"fss": ci(0.1)})]
    elif case == "dup_model":
        res["test"] = res["test"] + [res["test"][1]]
    with pytest.raises(ValueError):
        write_model_metrics(tmp_path / "m", results=res, **kwargs)
    assert not (tmp_path / "m" / f"{kwargs['hazard']}.json").exists()


# ---------------------------------------------------------------- EVAL-18
@pytest.mark.req("EVAL-18")
def test_write_routing_metrics(tmp_path):
    p = write_routing_metrics(tmp_path, "elreno", _routing_result(ci(0.42)), "abc123", primary=True)
    assert p == tmp_path / "routing_elreno.json"
    d = json.loads(p.read_text())
    assert (
        d["schema"] == 1 and d["scenario"] == "elreno" and d["primary"] is True and d["git_sha"] == "abc123"
    )
    assert d["n_origins"] == 500 and d["n_paired"] == 500
    assert set(d["planners"]) == {"a", "a_prime", "b", "c", "d"}
    assert d["planners"]["c"]["mean_exposure_risk_min"] == pytest.approx(3.0)
    assert d["planners"]["c"]["mean_predicted_exposure"] is None
    assert d["gap_closed"]["estimate"] == pytest.approx(0.42)
    assert d["c_minus_b"]["estimate"] == pytest.approx(-1.0)
    assert d["c_minus_a_prime"]["estimate"] == pytest.approx(-2.0)
    p2 = write_routing_metrics(tmp_path, "nan_case", _routing_result(CI(NAN, NAN, NAN, 0, 1)), "abc123")
    assert math.isnan(json.loads(p2.read_text())["gap_closed"]["estimate"])
    with pytest.raises(ValueError):
        write_routing_metrics(tmp_path, "x", _routing_result(ci(0.1)), "")


# ---------------------------------------------------------------- EVAL-19
@pytest.mark.req("EVAL-19")
def test_metrics_md_honest_verdicts(tmp_path):
    mdir = tmp_path / "metrics"
    write_model_metrics(mdir, "storm", _storm_results(dl=0.40, base=0.35), "csi_mean_74_133", True, "sha_s")
    smoke = {"test": [ModelMetrics("smoke.cnn_mlp", "dl", {"rmse_smoke_episodes": ci(9.0)}),
                      ModelMetrics("persistence", "baseline", {"rmse_smoke_episodes": ci(8.0)}),
                      ModelMetrics("hrrr_smoke", "baseline", {"rmse_smoke_episodes": ci(12.0)})]}  # fmt: skip
    write_model_metrics(mdir, "smoke", smoke, "rmse_smoke_episodes", False, "sha_k")
    write_routing_metrics(
        mdir, "elreno", _routing_result(CI(0.42, 0.30, 0.55, 1000, 500)), "sha_r", primary=True
    )
    out = write_metrics_md(mdir, tmp_path / "metrics.md")
    text = out.read_text()
    assert text.startswith("# Outrun metrics")
    assert "## storm" in text and "## smoke" in text and "git sha: sha_s" in text
    assert text.index("## storm") < text.index("## smoke")  # HAZARDS order
    assert "| model | kind | csi_mean_74_133 | fss |" in text
    assert "0.400 [0.380, 0.420]" in text
    assert "**test:** DL model `storm.unet_vil_ir` beats the best baseline `pysteps` on csi_mean_74_133: " \
           "0.400 vs 0.350." in text  # fmt: skip
    assert "**case_holdout:** DL model `storm.unet_vil_ir` does not beat the best baseline `pysteps` on " \
           "csi_mean_74_133: 0.200 vs 0.250." in text  # fmt: skip
    # lower is better for smoke RMSE: best baseline is persistence (8.0), DL 9.0 loses
    assert "**test:** DL model `smoke.cnn_mlp` does not beat the best baseline `persistence` on " \
           "rmse_smoke_episodes: 9.000 vs 8.000." in text  # fmt: skip
    assert "## Routing: elreno" in text
    assert (
        "DL planning (c) closes 42.0% [30.0%, 55.0%] of the observed-exposure gap between (a) and (d)."
        in text
    )


@pytest.mark.req("EVAL-19")
def test_metrics_md_reference_never_best_baseline_and_ties(tmp_path):
    res = {"test": [ModelMetrics("tor.cnn", "dl", {"auc_tor_vs_warned_null": ci(0.80)}),
                    ModelMetrics("gbm", "baseline", {"auc_tor_vs_warned_null": ci(0.80)}),
                    ModelMetrics("published_cnn", "reference", {"auc_tor_vs_warned_null": ci(0.874)})]}  # fmt: skip
    write_model_metrics(tmp_path / "m", "tornado", res, "auc_tor_vs_warned_null", True, "sha")
    text = write_metrics_md(tmp_path / "m", tmp_path / "out.md").read_text()
    assert "does not beat the best baseline `gbm`" in text  # tie counts as not beating; reference ignored
    assert "published_cnn" in text  # still listed in the table


@pytest.mark.req("EVAL-19")
def test_metrics_md_empty_dir_and_missing_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    assert write_metrics_md(tmp_path / "empty", tmp_path / "a.md").read_text().startswith("# Outrun metrics")
    with pytest.raises(FileNotFoundError):
        write_metrics_md(tmp_path / "nope", tmp_path / "b.md")
    write_routing_metrics(tmp_path / "r", "x", _routing_result(CI(NAN, NAN, NAN, 0, 1)), "sha")
    text = write_metrics_md(tmp_path / "r", tmp_path / "c.md").read_text()
    assert "DL planning (c) closes n/a of the observed-exposure gap between (a) and (d)." in text


# ---------------------------------------------------------------- EVAL-20
@pytest.mark.req("EVAL-20")
def test_ledger_allows_one_test_run_per_model(tmp_path):
    path = tmp_path / "reports" / "test_ledger.json"
    led = RunLedger(path)
    e = led.record("storm.unet_vil_ir", "abc123")
    assert e["model"] == "storm.unet_vil_ir" and e["split"] == "test" and e["git_sha"] == "abc123"
    assert e["rerun_reason"] is None and e["utc"]
    assert path.exists()
    led.record("storm.unet_vil_ir", "abc123", split="case_holdout")  # a different split is fine
    with pytest.raises(HeldOutReuseError):
        led.record("storm.unet_vil_ir", "def456")
    with pytest.raises(HeldOutReuseError):
        RunLedger(path).record("storm.unet_vil_ir", "def456", rerun_reason="")
    led2 = RunLedger(path)
    led2.record("storm.unet_vil_ir", "def456", rerun_reason="metric bug fixed in eval/metrics.py")
    runs = RunLedger(path).runs("storm.unet_vil_ir", "test")
    assert [r["git_sha"] for r in runs] == ["abc123", "def456"]
    assert len(RunLedger(path).runs()) == 3
    with pytest.raises(ValueError):
        led2.record("flood.unet", "")


# ---------------------------------------------------------------- EVAL-21
@pytest.mark.req("EVAL-21")
def test_time_calls_warmup_and_stats():
    seen = []

    def fn(x):
        seen.append(x)
        time.sleep(0.005)

    stats = time_calls(fn, [1, 2, 3, 4], warmup=6)
    assert seen == [1, 2, 3, 4, 1, 2] + [1, 2, 3, 4]
    assert isinstance(stats, LatencyStats) and stats.n == 4
    assert 0.004 <= stats.median_s <= stats.p95_s <= stats.max_s
    with pytest.raises(ValueError):
        time_calls(fn, [])
    with pytest.raises(ValueError):
        time_calls(fn, [1], warmup=-1)


@pytest.mark.req("EVAL-21")
def test_write_latency_merges(tmp_path):
    write_latency(tmp_path, "astar_metro_query", LatencyStats(20, 0.4, 0.7, 0.9), "sha1")
    p = write_latency(tmp_path, "sa_run", LatencyStats(5, 40.0, 55.0, 58.0), "sha2")
    d = json.loads(p.read_text())
    assert p == tmp_path / "latency.json"
    assert (
        d["astar_metro_query"]["median_s"] == pytest.approx(0.4)
        and d["astar_metro_query"]["git_sha"] == "sha1"
    )
    assert d["sa_run"]["n"] == 5 and d["sa_run"]["max_s"] == pytest.approx(58.0)
    assert set(d["sa_run"]) == {"n", "median_s", "p95_s", "max_s", "git_sha", "created_utc"}


# ---------------------------------------------------------------- EVAL-22 group metrics
def _group_inputs(n_seeds=5):
    shel = {
        m: {"served_persons": 900, "unserved_persons": u, "total_person_minutes": 1e4, "max_minutes": mx,
            "overflow_persons": o, "overflow_pct": o / 10, "mean_exposure": float("nan")}
        for m, u, mx, o in (("mcf", 0, 22.0, 0), ("nearest", 0, 18.0, 140), ("greedy", 30, 25.0, 0))
    }  # fmt: skip
    evac_row = {"vehicle_hours": 120.0, "exposure_veh_h": 3.5, "clearance_min": 95.0, "max_vc": 1.2,
                "unserved_veh": 0.0, "J": 130.0, "gap_to_lb_pct": 12.5}  # fmt: skip
    evac = {
        m: [dict(evac_row) for _ in range(n_seeds)] for m in ("sa", "msa", "shortest", "safest", "sa_lambda0")
    }
    ems_row = {"mean_eta_min": 9.0, "mean_detour_min": 1.0, "pct_hit_block": 0.0, "pct_unreached": 0.0,
               "mean_exposure": 0.2}  # fmt: skip
    ems = {"hungarian": ems_row, "reroute_baseline": dict(ems_row, pct_hit_block=40.0),
           "pareto": [{"lam": 0.0, "eta_min": 8.0, "exposure": 0.5}, {"lam": 1.0, "eta_min": 9.0, "exposure": 0.2}]}  # fmt: skip
    return shel, evac, ems


@pytest.mark.req("EVAL-22")
def test_write_group_metrics_schema_and_round_trip(tmp_path):
    from outrun.eval.report import load_group_metrics, write_group_metrics

    shel, evac, ems = _group_inputs()
    p = write_group_metrics(tmp_path, "elreno", shelters=shel, evacuation=evac, lower_bound=115.0, ems=ems,
                            git_sha="abc", seeds=[0, 1, 2, 3, 4], primary=True)  # fmt: skip
    assert p == tmp_path / "group_elreno.json"
    d = json.loads(p.read_text())
    assert d["schema"] == 1 and d["scenario"] == "elreno" and d["primary"] is True and d["git_sha"] == "abc"
    assert d["created_utc"]
    assert set(d["shelters"]) == {"mcf", "nearest", "greedy"}
    assert d["shelters"]["nearest"]["overflow_persons"] == 140
    assert d["evacuation"]["seeds"] == [0, 1, 2, 3, 4] and d["evacuation"]["lower_bound"] == 115.0
    assert set(d["evacuation"]["methods"]) == {"sa", "msa", "shortest", "safest", "sa_lambda0"}
    assert all(len(v) == 5 for v in d["evacuation"]["methods"].values())
    assert set(d["ems"]) == {"hungarian", "reroute_baseline", "pareto"} and len(d["ems"]["pareto"]) == 2
    back = load_group_metrics(p)
    assert math.isnan(back["shelters"]["mcf"]["mean_exposure"])
    assert back["evacuation"] == d["evacuation"]


@pytest.mark.req("EVAL-22")
def test_write_group_metrics_accepts_dataclass_and_rejects_bad_input(tmp_path):
    from outrun.eval.report import write_group_metrics
    from outrun.optimize.shelters import ShelterMetrics

    shel, evac, ems = _group_inputs()
    shel_dc = {m: ShelterMetrics(**v) for m, v in shel.items()}
    p = write_group_metrics(tmp_path, "c1", shelters=shel_dc, evacuation=evac, lower_bound=1.0, ems=ems,
                            git_sha="s", seeds=[0, 1, 2, 3, 4])  # fmt: skip
    assert json.loads(p.read_text())["shelters"]["greedy"]["unserved_persons"] == 30
    kw = dict(shelters=shel, evacuation=evac, lower_bound=1.0, ems=ems, git_sha="s", seeds=[0, 1, 2, 3, 4])
    bad = [
        dict(kw, shelters={k: v for k, v in shel.items() if k != "greedy"}),
        dict(kw, evacuation={k: v for k, v in evac.items() if k != "sa_lambda0"}),
        dict(kw, evacuation=dict(evac, msa=evac["msa"][:3])),  # seed-count mismatch
        dict(kw, ems={k: v for k, v in ems.items() if k != "reroute_baseline"}),
        dict(kw, git_sha=""),
        dict(kw, seeds=[]),
        dict(kw, lower_bound=float("nan")),
    ]
    for b in bad:
        with pytest.raises(ValueError):
            write_group_metrics(tmp_path, "c2", **b)


@pytest.mark.req("EVAL-19")
@pytest.mark.req("EVAL-22")
def test_metrics_md_renders_group_metrics(tmp_path):
    from outrun.eval.report import write_group_metrics

    mdir = tmp_path / "metrics"
    shel, evac, ems = _group_inputs()
    write_group_metrics(mdir, "elreno", shelters=shel, evacuation=evac, lower_bound=115.0, ems=ems,
                        git_sha="g", seeds=[0, 1, 2, 3, 4])  # fmt: skip
    text = write_metrics_md(mdir, tmp_path / "m.md").read_text()
    assert "## Group: elreno" in text
    assert "### Shelters" in text and "| method | overflow_persons | unserved_persons | max_minutes |" in text
    assert text.index("| mcf |") < text.index("| nearest |") < text.index("| greedy |")
    assert "### Evacuation" in text
    assert (
        "| method | seeds | vehicle_hours | exposure_veh_h | clearance_min | max_vc | gap_to_lb_pct |" in text
    )
    assert "| sa | 5 | 120.000 | 3.500 | 95.000 | 1.200 | 12.500 |" in text
    assert "lower bound: 115" in text
    assert "### Emergency vehicles" in text and "| hungarian |" in text and "| reroute_baseline |" in text
