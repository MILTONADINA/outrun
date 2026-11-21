"""Smoke val evaluation (T18): baselines, model, q90 coverage and a val-only q90 shift. Never reads test.

``OUTRUN_PROC=... .venv/bin/python -m outrun.hazards.smoke.evaluate`` -> ``logs/smoke_val.json`` + a printed table.
All fitting (calibrated HRRR-Smoke, linear regression) uses TRAIN rows; the model and the q90 shift are the only
things that look at val, and the shift is fitted on Jul-Sep and checked on Oct-Dec before the final all-val fit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from outrun.config import get_paths
from outrun.hazards.smoke.baselines import LinearSmokeBaseline, MassdenCalibrated, persistence, smoke_metrics
from outrun.hazards.smoke.dataset import SmokeFeatureConfig
from outrun.hazards.smoke.model import to_pm
from outrun.hazards.smoke.train import SmokeData, SmokePart, load_best, predict_log

LINEAR_FIT_ROWS = 400_000


def _massden_center(data: SmokeData, part: SmokePart) -> np.ndarray:
    """Raw HRRR-Smoke MASSDEN f06 (ug/m3) at the monitor's cell: patch channel 1 holds log1p(MASSDEN)."""
    return np.expm1(data.patches[part.cycle, part.site, 1, 4, 4].astype(np.float64))


def _flat(data: SmokeData, part: SmokePart, idx: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None):
    x = np.concatenate(
        [
            data.patches[part.cycle[idx], part.site[idx]].reshape(len(idx), -1).astype(np.float32),
            part.tab[idx],
        ],
        axis=1,
    )
    if mean is None:
        return x, x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    return (x - mean) / std, mean, std


def baseline_predictions(data: SmokeData) -> dict[str, np.ndarray]:
    """PM2.5 (ug/m3) on val for persistence, 30-day median, calibrated HRRR-Smoke and linear regression."""
    tr, va = data.parts["train"], data.parts["val"]
    fill = float(np.nanmedian(tr.background))  # only the first weeks of 2022 have no 30-day background
    out = {"persistence": persistence(va.hist)}
    out["background_median"] = np.where(np.isnan(va.background), fill, va.background)
    ok = ~np.isnan(tr.background)
    cal = MassdenCalibrated().fit(_massden_center(data, tr)[ok], tr.background[ok], tr.pm_target[ok])
    out["hrrr_calibrated"] = cal.predict(_massden_center(data, va), out["background_median"])
    out["_cal"] = np.array([cal.a, cal.b])
    idx = np.sort(np.random.default_rng(0).choice(len(tr), size=min(LINEAR_FIT_ROWS, len(tr)), replace=False))
    x, mean, std = _flat(data, tr, idx, None, None)
    lin = LinearSmokeBaseline().fit(x, tr.target[idx])
    pred = np.empty(len(va))
    for lo in range(0, len(va), 100_000):
        j = np.arange(lo, min(lo + 100_000, len(va)))
        pred[j] = lin.predict(_flat(data, va, j, mean, std)[0])
    out["linear_regression"] = np.maximum(np.expm1(pred), 0.0)
    return out


def coverage(pm_q: np.ndarray, obs: np.ndarray) -> float:
    return float(np.mean(obs <= pm_q))


def evaluate(out_json: Path | None = None) -> dict:
    data = SmokeData(("train", "val"))
    va = data.parts["val"]
    obs, ep = va.pm_target.astype(np.float64), va.episode
    ok_site = np.array([s.startswith("40-") for s in va.site_id])  # Oklahoma monitors
    res: dict = {"n_val": int(len(va)), "n_episode": int(ep.sum()), "n_oklahoma": int(ok_site.sum()),
                 "n_oklahoma_episode": int((ep & ok_site).sum())}  # fmt: skip
    table: dict[str, dict] = {}
    for name, pred in baseline_predictions(data).items():
        if name.startswith("_"):
            res["massden_calibration_a_b"] = pred.tolist()
            cal_dir = get_paths().models / "smoke"
            cal_dir.mkdir(parents=True, exist_ok=True)
            (cal_dir / "massden_calibration.json").write_text(
                json.dumps({"a": float(pred[0]), "b": float(pred[1])})
            )
            continue
        m = smoke_metrics(pred, pred, obs, ep)
        m["q90_coverage"] = float("nan")  # baselines are point forecasts
        table[name] = m
        table[name]["oklahoma_rmse"] = smoke_metrics(pred[ok_site], pred[ok_site], obs[ok_site], ep[ok_site])[
            "rmse"
        ]
    models = {"cnn_mlp": ("smoke", True), "cnn_mlp_no_hms": ("smoke_nohms", False)}
    for name, (folder, use_hms) in models.items():
        d = get_paths().models / folder
        if not (d / "best.pt").exists():
            continue
        net = load_best(d, use_hms)
        d2 = data if use_hms else SmokeData(("train", "val"), SmokeFeatureConfig(use_hms=False))
        pm = to_pm(predict_log(net, d2, d2.parts["val"]))
        m = smoke_metrics(pm[:, 0], pm[:, 1], obs, ep)
        m["q50_coverage"] = coverage(pm[:, 0], obs)
        m["q90_coverage_episode"] = coverage(pm[ep, 1], obs[ep]) if ep.any() else float("nan")
        m["q90_coverage_oklahoma"] = coverage(pm[ok_site, 1], obs[ok_site])
        m["oklahoma_rmse"] = smoke_metrics(pm[ok_site, 0], pm[ok_site, 1], obs[ok_site], ep[ok_site])["rmse"]
        table[name] = m
        if name == "cnn_mlp":
            res["calibration"] = q90_shift(net, data, d)
    res["models"] = table
    out_json = out_json or get_paths().logs / "smoke_val.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=2))
    print_table(res)
    return res


def q90_shift(net, data: SmokeData, model_dir: Path) -> dict:
    """Additive log-space shift of the q90 so that val coverage is 0.90.

    Honest check: fit on Jul-Sep val rows, report coverage on Oct-Dec; the saved shift is fitted on all val (its
    all-val coverage is 0.90 by construction, the Oct-Dec number is the real signal). Saved next to best.pt."""
    va = data.parts["val"]
    z = predict_log(net, data, va)
    resid = va.target - z[:, 1]  # q90 is high enough when target <= z + shift
    first = va.month <= 9
    s_first = float(np.quantile(resid[first], 0.9))
    cov_before_second = float(np.mean(resid[~first] <= 0.0))
    cov_after_second = float(np.mean(resid[~first] <= s_first))
    s_all = float(np.quantile(resid, 0.9))
    cal = {
        "q90_log_shift": s_all,
        "val_coverage_before": float(np.mean(resid <= 0.0)),
        "val_coverage_after_by_construction": float(np.mean(resid <= s_all)),
        "octdec_coverage_before": cov_before_second,
        "octdec_coverage_after_shift_fit_on_julsep": cov_after_second,
        "julsep_shift": s_first,
    }
    (model_dir / "calibration.json").write_text(json.dumps(cal, indent=2))
    return cal


def print_table(res: dict) -> None:
    keys = [
        "rmse",
        "mae",
        "rmse_episode",
        "mae_episode",
        "csi",
        "pod",
        "far",
        "q90_coverage",
        "oklahoma_rmse",
    ]
    print(
        f"val n={res['n_val']} episode n={res['n_episode']} OK n={res['n_oklahoma']} (episode {res['n_oklahoma_episode']})"
    )
    print(f"{'model':20s}" + "".join(f"{k:>14s}" for k in keys))
    for name, m in res["models"].items():
        print(f"{name:20s}" + "".join(f"{m.get(k, float('nan')):14.3f}" for k in keys))
    if "calibration" in res:
        print(json.dumps(res["calibration"], indent=1))


if __name__ == "__main__":
    evaluate(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
