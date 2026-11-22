"""One-shot held-out evaluation of the tornado detector (T24): ``evaluate(split) -> HazardEval``.

The same code path runs on ``"val"`` (dry run) and ``"test"`` (official test chips under ``results["test"]`` plus
the quarantined case hold-out under ``results["case_holdout"]``). Only frozen artefacts are used: the CNN
checkpoint, ``calibrator.json`` and ``operating_point.json`` in ``models/tornado``. The GBM and the range-only
logistic model are fit on the TRAIN split only (seed 0, as in ``val_eval``); the delta-V rule has no parameters.
Primary metric ``auc_tor_vs_warned_null`` (TOR vs warned-null AUC, higher is better) with a day-block bootstrap.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from outrun.config import get_paths
from outrun.eval.bootstrap import CI, day_block_ci
from outrun.eval.hazard_eval import HazardEval
from outrun.eval.metrics import ece, reliability_curve
from outrun.eval.report import ModelMetrics
from outrun.eval.splits import read_split

from .baselines import gbm_predict_proba, range_only_predict_proba, train_gbm, train_range_only
from .calibration import IsotonicCalibrator
from .dataset import TornetChips, keep_indices, load_meta, tornet_root
from .features import HAND_FEATURES
from .metrics import tornado_scores
from .val_eval import DV_THRESHOLD, hand_feature_table

PRIMARY = "auc_tor_vs_warned_null"
PUBLISHED_TORNET_AUC = 0.874
CNN = "tornado.cnn"
BOOTSTRAPPED = ("auc", "auprc", "csi")  # the other metrics carry the estimate only (nan bounds)


def _split_parts(split: str) -> dict[str, tuple[str, str]]:
    """Result key -> (processed array split, id set in ``splits/tornet.json``)."""
    if split == "val":
        return {"val": ("val", "val")}
    if split == "test":
        return {"test": ("test", "test"), "case_holdout": ("case_holdout", "case_holdout")}
    raise ValueError(f"split must be 'val' or 'test', got {split!r}")


def _frozen_split_items(part: str):
    """Items of one id set: test = split.test, case hold-out = split.extra['case_holdout'], val = split.val."""
    sp = read_split(Path(__file__).resolve().parents[4] / "splits" / "tornet.json")
    items = sp.extra[part] if part == "case_holdout" else getattr(sp, part)
    return {it.id: it.day for it in items}


def _cnn_probs(ckpt: Path, split: str, device: str = "cpu", batch_size: int = 64) -> np.ndarray:
    """Raw CNN probabilities for every chip of a processed split (generalises ``val_eval.cnn_val_probs``)."""
    import torch
    from torch.utils.data import DataLoader

    from .model import TornadoCNN, predict_proba

    state = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    in_ch = int(state["features.0.weight"].shape[1])
    model = TornadoCNN(in_channels=in_ch)
    model.load_state_dict(state)
    model.to(device)
    with np.load(ckpt.with_name("norm_stats.npz")) as z:
        mean, std = z["mean"], z["std"]
    ds = TornetChips(split, mean, std, include_range=in_ch == 27)
    out = [
        predict_proba(model, x.to(device)).cpu().numpy() for x, _ in DataLoader(ds, batch_size, num_workers=4)
    ]
    return np.concatenate(out).astype(np.float64)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def _auprc(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def _csi(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    hit = int(((p >= threshold) & (y == 1)).sum())
    den = hit + int(((p < threshold) & (y == 1)).sum()) + int(((p >= threshold) & (y == 0)).sum())
    return hit / den if den else float("nan")


def _ci(fn: Callable[..., float], arrays: list[np.ndarray], days: np.ndarray, n_resamples: int) -> CI:
    """Day-block bootstrap CI; point estimate with nan bounds if there are fewer than 2 days."""
    if len(np.unique(days)) < 2:
        return CI(float(fn(*arrays)), float("nan"), float("nan"), 0, len(np.unique(days)))
    return day_block_ci(fn, arrays, days, n_resamples=n_resamples, seed=0)


def _point(x: float) -> CI:
    return CI(float(x), float("nan"), float("nan"), 0, 0)


def _model_metrics(
    name: str,
    kind: str,
    p: np.ndarray,
    thr: float,
    data: dict,
    n_resamples: int,
    calibrated: bool,
) -> ModelMetrics:
    y, days, rng, warned = data["y"], data["days"], data["range"], data["warned"]
    sc = tornado_scores(y, p, data["category"], rng, threshold=thr)
    m = {
        PRIMARY: _ci(_auc, [y[warned], p[warned]], days[warned], n_resamples),
        "auc": _ci(_auc, [y, p], days, n_resamples),
        "auprc": _ci(_auprc, [y, p], days, n_resamples),
        "csi": _ci(lambda a, b: _csi(a, b, thr), [y, p], days, n_resamples),
        "pod": _point(sc["pod"]),
        "far": _point(sc["far"]),
        "brier": _point(sc["brier"]),
    }
    for bin_name, auc in sc["auc_by_range"].items():
        m[f"auc_range_{bin_name}"] = _point(auc)
    if calibrated:
        m["ece"] = _point(ece(y, p, n_bins=10))
    return ModelMetrics(name=name, kind=kind, metrics=m)


def evaluate(split: str, n_resamples: int = 1000) -> HazardEval:
    parts = _split_parts(split)
    models_dir = get_paths().models / "tornado"
    ckpt = models_dir / "best.pt"
    cal = IsotonicCalibrator.from_dict(json.loads((models_dir / "calibrator.json").read_text()))
    thr_cnn = float(json.loads((models_dir / "operating_point.json").read_text())["threshold_calibrated"])

    # baselines with trainable parts: TRAIN only
    root = tornet_root()
    tr_meta = load_meta("train", root)
    tr_keep = keep_indices("train", tr_meta)
    tr_f = hand_feature_table("train", root)
    y_tr = tr_meta["label"].to_numpy()
    gbm = train_gbm(tr_f[tr_keep], y_tr[tr_keep], seed=0)
    rng_model = train_range_only(tr_f[tr_keep, -1], y_tr[tr_keep])

    results: dict[str, list[ModelMetrics]] = {}
    figures: dict = {}
    for key, (proc_split, id_part) in parts.items():
        meta = load_meta(proc_split, root)
        day_of = _frozen_split_items(id_part)
        ids = meta["filename"].tolist()
        if set(ids) != set(day_of) or len(ids) != len(day_of):
            raise ValueError(f"processed {proc_split} chips do not match the frozen {id_part} id set")
        days = np.array([date.fromisoformat(str(day_of[i])) for i in ids], dtype=object)
        feats = hand_feature_table(proc_split, root)
        centers = feats[:, HAND_FEATURES.index("range_km")]
        y = meta["label"].to_numpy().astype(int)
        category = meta["category"].to_numpy()
        data = {
            "y": y,
            "days": days,
            "range": centers,
            "category": category,
            "warned": (y == 1) | (category == "WRN"),
        }
        p_raw = _cnn_probs(ckpt, proc_split)
        p_cal = cal.transform(p_raw)
        dv = feats[:, HAND_FEATURES.index("max_dv")]
        rows = [
            _model_metrics(CNN, "dl", p_cal, thr_cnn, data, n_resamples, True),
            _model_metrics("tornado.cnn_raw", "baseline", p_raw, 0.5, data, n_resamples, False),
            _model_metrics(
                "tornado.gbm", "baseline", gbm_predict_proba(gbm, feats), 0.5, data, n_resamples, False
            ),
            _model_metrics(
                "tornado.range_only",
                "baseline",
                range_only_predict_proba(rng_model, centers),
                0.5,
                data,
                n_resamples,
                False,
            ),
            _model_metrics(
                "tornado.delta_v_rule",
                "baseline",
                (dv >= DV_THRESHOLD).astype(float),
                0.5,
                data,
                n_resamples,
                False,
            ),
        ]
        rows.append(ModelMetrics("tornet_published", "reference", {PRIMARY: _point(PUBLISHED_TORNET_AUC)}))
        results[key] = rows
        curve = reliability_curve(y, p_cal, n_bins=10)
        figures[f"reliability_{key}"] = {
            "n_bins": 10,
            "counts": np.asarray(curve.counts).tolist(),
            "mean_pred": np.nan_to_num(np.asarray(curve.mean_pred), nan=-1.0).tolist(),
            "obs_freq": np.nan_to_num(np.asarray(curve.obs_freq), nan=-1.0).tolist(),
        }
        figures[f"n_chips_{key}"] = len(y)
        figures[f"n_tor_{key}"] = int(y.sum())
        figures[f"n_blocks_{key}"] = len(np.unique(days))

    notes = [
        f"CNN row = frozen isotonic calibrator; csi/pod/far at the val-frozen calibrated cut {thr_cnn:.4f} "
        "(operating_point.json, picked on val). Baselines use the spec cut 0.5 for csi/pod/far; "
        "'tornado.cnn_raw' is the uncalibrated CNN (reported as a baseline-kind row for transparency only; "
        "it is not a separate model).",
        f"Primary {PRIMARY} = ROC-AUC on TOR chips vs warned-null (WRN) chips only (TORN-07); day-block bootstrap over "
        "UTC calendar days of the chips (id days from splits/tornet.json); other metrics bootstrapped: auc, auprc, "
        "csi; pod/far/brier/ece/auc_range_* are point estimates (nan bounds).",
        f"Reference row: TorNet published AUC {PUBLISHED_TORNET_AUC} is a different task/protocol (all chips, "
        "own test split), not directly comparable to the warned-null AUC; compare against 'auc' instead.",
        "No tornado no-range checkpoint exists in models/, so the required-if-present ablation is omitted; "
        "tornado.range_only quantifies the range shortcut.",
        "GBM and range-only logistic are fit on train only (87 all-missing train chips dropped, as in val_eval).",
        "ECE of the calibrated CNN on val is ~0 because the isotonic map was fit on val (optimistic by design); "
        "on test it is a genuine out-of-sample calibration error.",
        "Test chips from 2019-05-17..19 quarantine days are test-only by construction (SPLIT-08); "
        "case_holdout is limited to near-KTLX chips and is a small set (single-class AUC gives nan).",
    ]
    return HazardEval(
        hazard="tornado",
        primary_metric=PRIMARY,
        higher_is_better=True,
        results=results,
        ablations={},
        dl_models=[CNN],
        notes=notes,
        figures=figures,
    )
