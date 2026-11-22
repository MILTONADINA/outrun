"""T24: one-shot ice evaluation (zone AUPRC with day-block CI) for the val dry run and the held-out test.

``evaluate(split)`` is the same code path for ``"val"`` and ``"test"``; only the split ids differ. Only frozen
artefacts are used: ``models/ice/best.pt``, ``models/ice/calibrator.pkl`` and the val-chosen raw threshold in
``models/ice/val_metrics.json``. The logistic and climatology baselines are fit on the train bags only (same
settings as ``eval_baselines.py``); their CSI thresholds come from val, never from the evaluated split.

RAM: only the HRRR time rows the evaluated bags need are loaded (a season, not the whole 15 GB crop).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from outrun.config import get_paths
from outrun.eval import metrics as M
from outrun.eval.bootstrap import CI, day_block_ci
from outrun.eval.hazard_eval import HazardEval
from outrun.eval.report import ModelMetrics
from outrun.eval.splits import read_split
from outrun.hazards.base import load_checkpoint
from outrun.hazards.ice.baselines import LogisticZoneBaseline, ZoneMonthClimatology
from outrun.hazards.ice.dataset import bags_for_split
from outrun.hazards.ice.eval_baselines import best_threshold, features
from outrun.hazards.ice.model import ZoneCalibrator
from outrun.hazards.ice.train import BagStore, ZoneLogitNet, _Loader, zone_probabilities

SPLIT_FILE = Path("splits/ice.json")
DL_RAW, DL_CAL = "ice.milcnn", "ice.milcnn_calibrated"


def _csi(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    return M.csi((p >= threshold).astype(float), y.astype(float), 0.5)


def _ci(fn: Callable[..., float], arrays: list[np.ndarray], days: np.ndarray, n: int) -> CI:
    return day_block_ci(fn, arrays, days, n_resamples=n, seed=0)


def _row(name, kind, y, p, days, n, base_rate, threshold, paired=()) -> ModelMetrics:
    """Primary + secondary metrics, each a day-block CI. ``paired`` = ((label, other_scores), ...)."""
    arrays = [y, p]
    ms = {
        "zone_auprc": _ci(M.auprc, arrays, days, n),
        "roc_auc": _ci(M.roc_auc, arrays, days, n),
        "brier": _ci(M.brier, arrays, days, n),
        "bss_vs_climatology": _ci(partial(M.brier_skill_score, ref_probs=base_rate), arrays, days, n),
        "ece": _ci(M.ece, arrays, days, n),
        "csi": _ci(partial(_csi, threshold=threshold), arrays, days, n),
    }
    for label, other in paired:
        ms[f"auprc_minus_{label}"] = _ci(
            lambda a, b, c: M.auprc(a, b) - M.auprc(a, c), [y, p, other], days, n
        )
    return ModelMetrics(name, kind, ms)


def _score_bags(bags: pd.DataFrame, d: Path, model: ZoneLogitNet, device: str) -> np.ndarray:
    """Raw MIL-CNN zone probabilities; the HRRR store holds only the time rows these bags use."""
    times = np.unique(np.concatenate([bags["i_t"].to_numpy(), bags["i_tm3"].to_numpy()]))
    store = BagStore(d, times=times)
    remapped = bags.assign(
        i_t=np.searchsorted(times, bags["i_t"]), i_tm3=np.searchsorted(times, bags["i_tm3"])
    )
    return zone_probabilities(model, _Loader(store, remapped, 256, False), device)


def evaluate(split: str, n_resamples: int = 1000) -> HazardEval:
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    paths = get_paths()
    d = paths.proc / "hrrr" / "ice" / "bags"
    all_bags = pd.read_parquet(d / "bags.parquet")
    zones = np.load(d / "zones.npz")
    all_bags["zone_index"] = all_bags["zone_id"].map({z: i for i, z in enumerate(zones["zone_ids"])})
    scan = np.load(d / "scan.npz")
    sp = read_split(SPLIT_FILE)
    # official test = split.test; case hold-out = split.extra["case_holdout"] (empty for ice, see notes)
    parts = {"val": ["val"], "test": ["test"] + (["case_holdout"] if sp.extra.get("case_holdout") else [])}[
        split
    ]
    train, val = (bags_for_split(all_bags, sp, p) for p in ("train", "val"))

    # frozen / train-only fits
    month = lambda df: df["window_start"].dt.month.to_numpy()  # noqa: E731
    clim = ZoneMonthClimatology().fit(train["zone_id"], month(train), train["label"])
    lr = LogisticZoneBaseline(0).fit(features(train, scan), train["label"].to_numpy())
    base_rate = float(train["label"].mean())
    y_val = val["label"].to_numpy()
    thr_clim = best_threshold(clim.predict(val["zone_id"], month(val)), y_val)
    thr_lr = best_threshold(lr.predict_proba(features(val, scan)), y_val)
    frozen = json.loads((paths.models / "ice" / "val_metrics.json").read_text())
    thr_raw = float(frozen["milcnn_raw"]["threshold"])
    cal = ZoneCalibrator.load(paths.models / "ice" / "calibrator.pkl")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ZoneLogitNet()
    load_checkpoint(paths.models / "ice" / "best.pt", model)
    model.to(device)

    results: dict[str, list[ModelMetrics]] = {}
    figures: dict = {}
    for part in parts:
        bags = bags_for_split(all_bags, sp, part)
        y = bags["label"].to_numpy()
        days = bags["window_start"].dt.date.to_numpy()  # UTC calendar day of the 6-h window start
        p_raw = _score_bags(bags, d, model, device)
        p_cal = cal.transform(p_raw)
        p_rule = bags["rule"].to_numpy(float)
        p_lr = lr.predict_proba(features(bags, scan))
        p_clim = clim.predict(bags["zone_id"], month(bags))
        n = n_resamples
        results[part] = [
            _row(DL_RAW, "dl", y, p_raw, days, n, base_rate, thr_raw,
                 paired=(("physics_rule", p_rule), ("logistic", p_lr))),
            _row(DL_CAL, "dl", y, p_cal, days, n, base_rate, 0.5),
            _row("physics_rule", "baseline", y, p_rule, days, n, base_rate, 0.5),
            _row("logistic", "baseline", y, p_lr, days, n, base_rate, thr_lr),
            _row("climatology", "baseline", y, p_clim, days, n, base_rate, thr_clim),
        ]  # fmt: skip
        figures[part] = {
            name: {
                "mean_pred": np.nan_to_num(c.mean_pred, nan=-1).tolist(),
                "obs_freq": np.nan_to_num(c.obs_freq, nan=-1).tolist(),
                "counts": c.counts.tolist(),
            }
            for name, p in (("reliability_raw", p_raw), ("reliability_calibrated", p_cal))
            for c in [M.reliability_curve(y, p, 10)]
        }
        figures[part]["n_bags"] = len(bags)
        figures[part]["n_positive"] = int(y.sum())
    notes = [
        "Zone AUPRC on (zone, 6-h window) bags, day-block bootstrap over the UTC day of the window start.",
        "Negatives are a deterministic 1-in-20 sample (positives all kept): prevalence is inflated about 10x vs the "
        "true 0.7%, so AUPRC, Brier and BSS refer to that universe and are not operational rates.",
        "BSS reference = constant train positive rate (14.5%), not the zone-month climatology row.",
        "ice.milcnn_calibrated uses the isotonic calibrator fitted on ALL of val: on val its Brier/ECE are "
        "in-sample (optimistic); on test they are out-of-sample. The raw row uses the val-chosen threshold "
        f"{thr_raw:.3f} for CSI. Logistic/climatology CSI thresholds are the val-best thresholds of train-fit models.",
        "No case_holdout set exists for ice (splits/ice.json extra.case_holdout is empty); no urban/rural split "
        "(no urban label in the bag table); no bridge-prior ablation (the prior is a routing-layer multiplier, "
        "not a separate checkpoint); no ice ablation checkpoints in models/.",
        "Rule-vs-model is evaluated on identical windows (auprc_minus_physics_rule is a paired day-block CI).",
        "Labels are NWS Storm Events weak labels (weather in zone), not road-surface ice.",
    ]
    return HazardEval(
        hazard="ice",
        primary_metric="zone_auprc",
        higher_is_better=True,
        results=results,
        ablations={},
        dl_models=[DL_RAW, DL_CAL],
        notes=notes,
        figures=figures,
    )
