"""Smoke one-shot evaluation (T24): ``evaluate(split)`` scores frozen models and baselines on val or test.

``split="val"`` is the dry run (val ids only). ``split="test"`` is the same code path with the test ids
(``split.test``) and, as an extra key, ``test_spatial`` (the held-out monitors inside the test year, taken from
``split.extra["test_spatial"]``) and ``case_holdout`` when ``split.extra["case_holdout"]`` is non-empty.

Frozen artefacts only: ``models/smoke/best.pt``, ``models/smoke_nohms/best.pt``, ``models/smoke/calibration.json``
(val-fitted q90 shift, used for a separate coverage row only) and ``models/smoke/massden_calibration.json``. The
linear baseline is fitted on the TRAIN split (same 400k-row subsample, seed 0 as ``evaluate.py``).

Bootstrap: day-block over UTC issue days. Every metric is a ratio of sums, so each day is reduced once to a small
vector of sufficient statistics (counts and sums); ``day_block_ci`` then resamples the per-day rows (one block per
row, so a draw of n days with replacement is exactly the day-block bootstrap).
"""

from __future__ import annotations

import gc
import json
import math

import numpy as np
import pandas as pd

from outrun.config import get_paths
from outrun.eval.bootstrap import CI, day_block_ci
from outrun.eval.hazard_eval import HazardEval
from outrun.eval.report import ModelMetrics
from outrun.eval.splits import read_split
from outrun.hazards.smoke.baselines import LinearSmokeBaseline, persistence
from outrun.hazards.smoke.dataset import SmokeFeatureConfig
from outrun.hazards.smoke.evaluate import LINEAR_FIT_ROWS, _flat, _massden_center
from outrun.hazards.smoke.model import to_pm
from outrun.hazards.smoke.train import SmokeData, SmokePart, load_best, predict_log

PRIMARY = "rmse_smoke_episodes"
THRESHOLD = 35.4  # ug/m3, obs/pred > threshold counts as an exceedance (same as smoke_metrics)
NAN_CI = CI(float("nan"), float("nan"), float("nan"), 0, 0)

# columns of the per-day statistics matrix
_COLS = [
    "n",
    "sse",
    "sae",
    "n_ep",
    "sse_ep",
    "sae_ep",
    "hit",
    "miss",
    "fa",
    "n_q",
    "cov",
    "n_q_ep",
    "cov_ep",
    "cov_sh",
]
_C = {c: i for i, c in enumerate(_COLS)}


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _metric_fns() -> dict[str, callable]:
    """Metric name -> function of the summed statistics vector."""
    c = _C
    return {
        PRIMARY: lambda s: math.sqrt(_ratio(s[c["sse_ep"]], s[c["n_ep"]])),
        "mae_smoke_episodes": lambda s: _ratio(s[c["sae_ep"]], s[c["n_ep"]]),
        "rmse": lambda s: math.sqrt(_ratio(s[c["sse"]], s[c["n"]])),
        "mae": lambda s: _ratio(s[c["sae"]], s[c["n"]]),
        "csi_35_4": lambda s: _ratio(s[c["hit"]], s[c["hit"]] + s[c["miss"]] + s[c["fa"]]),
        "pod_35_4": lambda s: _ratio(s[c["hit"]], s[c["hit"]] + s[c["miss"]]),
        "far_35_4": lambda s: _ratio(s[c["fa"]], s[c["hit"]] + s[c["fa"]]),
        "q90_coverage": lambda s: _ratio(s[c["cov"]], s[c["n_q"]]),
        "q90_coverage_episodes": lambda s: _ratio(s[c["cov_ep"]], s[c["n_q_ep"]]),
        "q90_coverage_shifted": lambda s: _ratio(s[c["cov_sh"]], s[c["n_q"]]),
    }


def _day_stats(
    q50: np.ndarray, obs: np.ndarray, ep: np.ndarray, day_idx: np.ndarray, n_days: int,
    q90: np.ndarray | None, q90_shifted: np.ndarray | None,
) -> np.ndarray:  # fmt: skip
    """(n_days, len(_COLS)) per-day sums. Rows with a NaN q50 are dropped by the caller's mask."""
    err = q50 - obs
    o_hi, p_hi = obs > THRESHOLD, q50 > THRESHOLD
    cols = {
        "n": np.ones_like(err), "sse": err**2, "sae": np.abs(err),
        "n_ep": ep.astype(float), "sse_ep": np.where(ep, err**2, 0.0), "sae_ep": np.where(ep, np.abs(err), 0.0),
        "hit": (o_hi & p_hi).astype(float), "miss": (o_hi & ~p_hi).astype(float), "fa": (~o_hi & p_hi).astype(float),
    }  # fmt: skip
    if q90 is not None:
        covered = obs <= q90
        cols["n_q"] = np.ones_like(err)
        cols["cov"] = covered.astype(float)
        cols["n_q_ep"] = ep.astype(float)
        cols["cov_ep"] = (covered & ep).astype(float)
        cols["cov_sh"] = (obs <= q90_shifted).astype(float)
    out = np.zeros((n_days, len(_COLS)))
    for name, v in cols.items():
        out[:, _C[name]] = np.bincount(day_idx, weights=v, minlength=n_days)
    return out


def _model_metrics(
    name: str, kind: str, q50: np.ndarray, part: SmokePart, rows: np.ndarray, n_resamples: int,
    q90: np.ndarray | None = None, q90_shifted: np.ndarray | None = None,
) -> ModelMetrics:  # fmt: skip
    """Metrics of one prediction vector on ``rows`` (bool mask into ``part``) with day-block CIs."""
    ok = rows & ~np.isnan(q50)
    days = part.issue_time[ok].astype("datetime64[D]")
    uniq, day_idx = np.unique(days, return_inverse=True)
    stats = _day_stats(
        q50[ok].astype(np.float64), part.pm_target[ok].astype(np.float64), part.episode[ok], day_idx.reshape(-1),
        len(uniq), None if q90 is None else q90[ok].astype(np.float64),
        None if q90_shifted is None else q90_shifted[ok].astype(np.float64),
    )  # fmt: skip
    metrics: dict[str, CI] = {}
    for key, fn in _metric_fns().items():
        if key.startswith("q90") and q90 is None:
            continue  # point forecasts have no q90
        ci = day_block_ci(
            lambda s, fn=fn: fn(s.sum(axis=0)), [stats], np.arange(len(uniq)), n_resamples, seed=0
        )
        metrics[key] = ci
    return ModelMetrics(name, kind, metrics)  # type: ignore[arg-type]


def _baseline_predictions(data: SmokeData, eval_parts: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """name -> part -> PM2.5 (ug/m3). Calibration from the frozen file, linear regression fitted on TRAIN."""
    tr = data.parts["train"]
    fill = float(np.nanmedian(tr.background))  # first weeks of 2022 have no 30-day background
    cal = json.loads((get_paths().models / "smoke" / "massden_calibration.json").read_text())
    idx = np.sort(np.random.default_rng(0).choice(len(tr), size=min(LINEAR_FIT_ROWS, len(tr)), replace=False))
    x, mean, std = _flat(data, tr, idx, None, None)
    lin = LinearSmokeBaseline().fit(x, tr.target[idx])
    out: dict[str, dict[str, np.ndarray]] = {
        "persistence": {},
        "hrrr_calibrated": {},
        "linear_regression": {},
    }
    for part in eval_parts:
        p = data.parts[part]
        bg = np.where(np.isnan(p.background), fill, p.background)
        out["persistence"][part] = persistence(p.hist)
        out["hrrr_calibrated"][part] = np.maximum(cal["a"] * _massden_center(data, p) + cal["b"] + bg, 0.0)
        pred = np.empty(len(p))
        for lo in range(0, len(p), 100_000):
            j = np.arange(lo, min(lo + 100_000, len(p)))
            pred[j] = lin.predict(_flat(data, p, j, mean, std)[0])
        out["linear_regression"][part] = np.maximum(np.expm1(pred), 0.0)
    return out


def _rows_by_name(split, part_name: str, part: SmokePart, parts: list[str]) -> dict[str, np.ndarray]:
    """Row masks for the reported split keys: the main part plus ``test_spatial`` for the test year."""
    rows = {part_name: np.ones(len(part), dtype=bool)}
    if part_name == "test":
        ids = [
            f"{s}|{pd.Timestamp(t).tz_localize('UTC').isoformat()}"
            for s, t in zip(part.site_id, part.issue_time, strict=True)
        ]
        held = {i.id for i in split.extra["test_spatial"]}
        rows["test_spatial"] = np.fromiter((i in held for i in ids), dtype=bool, count=len(ids))
    return rows


def _reliability(q: np.ndarray, obs: np.ndarray, level: float, n_bins: int = 10) -> dict:
    """Quantile reliability: observed exceedance rate per predicted-quantile decile."""
    ok = ~np.isnan(q)
    edges = np.quantile(q[ok], np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, q[ok], side="right") - 1, 0, n_bins - 1)
    o = obs[ok]
    below = o <= q[ok]
    return {
        "level": level,
        "bin_mean_pred": [float(q[ok][b == k].mean()) for k in range(n_bins)],
        "bin_coverage": [float(below[b == k].mean()) for k in range(n_bins)],
    }


def evaluate(split: str, n_resamples: int = 1000) -> HazardEval:
    """Score the smoke models and baselines on ``split`` ("val" dry run or the one-shot "test")."""
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    sp = read_split(get_paths().splits / "smoke.json")
    eval_parts = [split] + (["case_holdout"] if split == "test" and sp.extra.get("case_holdout") else [])
    allow = split == "test"
    models_dir = get_paths().models
    shift = json.loads((models_dir / "smoke" / "calibration.json").read_text())["q90_log_shift"]

    data = SmokeData(("train", *eval_parts), allow_test=allow)
    base = _baseline_predictions(data, eval_parts)
    results: dict[str, list[ModelMetrics]] = {}
    figures: dict = {}
    net = load_best(models_dir / "smoke", True)
    for part_name in eval_parts:
        part = data.parts[part_name]
        z = predict_log(net, data, part)
        pm = to_pm(z)
        pm_sh = to_pm(z + np.array([0.0, shift]))
        for key, rows in _rows_by_name(sp, part_name, part, eval_parts).items():
            res = [
                _model_metrics(
                    "smoke.cnn_mlp", "dl", pm[:, 0], part, rows, n_resamples, pm[:, 1], pm_sh[:, 1]
                )
            ]
            for name, preds in base.items():
                res.append(_model_metrics(name, "baseline", preds[part_name], part, rows, n_resamples))
            results[key] = res
            if key == split:
                figures["q50_reliability"] = _reliability(pm[:, 0], part.pm_target.astype(float), 0.5)
                figures["q90_reliability"] = _reliability(pm[:, 1], part.pm_target.astype(float), 0.9)
    del data, net
    gc.collect()

    abl_data = SmokeData(tuple(eval_parts), SmokeFeatureConfig(use_hms=False), allow_test=allow)
    abl_net = load_best(models_dir / "smoke_nohms", False)
    ablation: list[ModelMetrics] = []
    for part_name in eval_parts:
        part = abl_data.parts[part_name]
        z = predict_log(abl_net, abl_data, part)
        pm, pm_sh = to_pm(z), to_pm(z + np.array([0.0, shift]))
        for key, rows in _rows_by_name(sp, part_name, part, eval_parts).items():
            label = "smoke.cnn_mlp_no_hms" if key == split else f"smoke.cnn_mlp_no_hms[{key}]"
            ablation.append(
                _model_metrics(label, "dl", pm[:, 0], part, rows, n_resamples, pm[:, 1], pm_sh[:, 1])
            )

    notes = [
        "Primary metric rmse_smoke_episodes is RMSE of the q50 forecast (ug/m3, +6 h) over rows where HMS shows "
        "medium/heavy smoke at the monitor at the target time (HMS is used only to select rows, never as input to "
        "scoring). LOWER is better.",
        "CIs: day-block bootstrap over UTC issue days (per-day sufficient statistics, seed 0). Rows within a day "
        "across monitors are resampled together, so the CI reflects day-to-day and not monitor-level variability.",
        "CSI/POD/FAR use obs and q50 > 35.4 ug/m3. Persistence beats the model on CSI/POD on val; the CNN wins "
        "RMSE/MAE. HMS adds nothing measurable on val (the no-HMS ablation is equal or better).",
        "q90_coverage uses the raw q90 head; q90_coverage_shifted adds the val-fitted frozen log shift from "
        "models/smoke/calibration.json (the adapter does not apply it). Nothing is refit or tuned on test.",
        "AQS hour value labelled t covers t..t+1 h, so the newest history hour is available at t+1 h: up to 1 h "
        "latency leak in the history features (spec-mandated). HRRR-Smoke store is 9 km (81 km patch).",
        "Calibrated HRRR baseline uses the frozen train-fit models/smoke/massden_calibration.json; linear "
        "regression is fitted on a 400k-row TRAIN subsample (seed 0). Baseline rows with an all-NaN history "
        "(persistence) are dropped for that baseline only.",
    ]
    if split == "test":
        notes.append(
            "test_spatial = rows at the 198 spatially held-out monitors (inside the test year, never seen in "
            "train/val); the main model and baselines are scored on it as an extra split key. "
            + (
                ""
                if "case_holdout" in results
                else "case_holdout is empty for smoke (no quarantined smoke days), so there is no case_holdout key."
            )
        )
    return HazardEval(
        hazard="smoke",
        primary_metric=PRIMARY,
        higher_is_better=False,
        results=results,
        ablations={"no_hms": ablation},
        dl_models=["smoke.cnn_mlp"],
        notes=notes,
        figures=figures,
    )
