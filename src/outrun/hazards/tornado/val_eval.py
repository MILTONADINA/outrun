"""Validation scores for the tornado baselines and the CNN (never touches test or case-holdout chips).

``python -m outrun.hazards.tornado.val_eval [--cnn models/tornado/best.pt]`` writes ``val_scores.json``.
GBM is fit on train hand features; the delta-V rule and range-only model need no val data; the CNN's isotonic
calibrator is fit on val (TORN-05) and saved as ``calibrator.json`` next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

from .baselines import gbm_predict_proba, range_only_predict_proba, train_gbm, train_range_only
from .calibration import IsotonicCalibrator
from .dataset import keep_indices, load_meta, tornet_root
from .features import HAND_FEATURES, chip_from_stored, hand_features
from .metrics import tornado_scores

DV_THRESHOLD = 40.0


def _hand_chunk(args: tuple[str, int, int, np.ndarray]) -> np.ndarray:
    path, lo, hi, centers = args
    x = np.load(path, mmap_mode="r")
    return np.stack(
        [hand_features(chip_from_stored(np.asarray(x[i])), centers[i - lo]) for i in range(lo, hi)]
    )


def hand_feature_table(split: str, root: Path | None = None, workers: int = 8) -> np.ndarray:
    """(N, 7) hand features for every chip of a split, cached as ``{split}_hand.npy`` next to the arrays."""
    root = Path(root) if root is not None else tornet_root()
    cache = root / f"{split}_hand.npy"
    if cache.exists():
        return np.load(cache)
    rng = np.load(root / f"{split}_range_km.npy")
    centers = rng.mean(axis=1)
    n, step = len(centers), 256
    jobs = [
        (str(root / f"{split}_x.npy"), lo, min(lo + step, n), centers[lo : min(lo + step, n)])
        for lo in range(0, n, step)
    ]
    with mp.Pool(workers) as pool:
        parts = pool.map(_hand_chunk, jobs)
    table = np.concatenate(parts)
    np.save(cache, table)
    return table


def _scores(meta, p: np.ndarray, centers: np.ndarray, threshold: float) -> dict:
    return tornado_scores(
        meta["label"].to_numpy(), p, meta["category"].to_numpy(), centers, threshold=threshold
    )


def pick_operating_threshold(meta, p_cal: np.ndarray, centers: np.ndarray) -> dict:
    """Val-only operating point on calibrated probabilities: the cut with the best CSI (ties -> higher cut).

    Calibrated P rarely reaches the spec's 0.5, so 0.5 gives POD ~ 0.03 on val. This cut is chosen on val only and
    written to ``operating_point.json`` (frozen: never re-picked on test or case-holdout chips).
    """
    grid = np.unique(np.quantile(p_cal, np.linspace(0.5, 0.999, 300)))
    csi = np.array([_scores(meta, p_cal, centers, float(t))["csi"] for t in grid])
    best = int(len(grid) - 1 - np.nanargmax(csi[::-1]))
    return {
        "selected_on": "val",
        "threshold_calibrated": float(grid[best]),
        "criterion": "max CSI on val, calibrated probabilities",
        "scores_at_threshold": _scores(meta, p_cal, centers, float(grid[best])),
        "frozen": True,
    }


def cnn_val_probs(ckpt: Path, device: str = "cpu", batch_size: int = 64) -> np.ndarray:
    """Raw CNN probabilities on every val chip."""
    import torch
    from torch.utils.data import DataLoader

    from .dataset import TornetChips
    from .model import TornadoCNN, predict_proba

    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = payload["model"]
    in_ch = int(next(v for k, v in state.items() if k == "features.0.weight").shape[1])
    model = TornadoCNN(in_channels=in_ch)
    model.load_state_dict(state)
    model.to(device)
    with np.load(ckpt.with_name("norm_stats.npz")) as z:
        mean, std = z["mean"], z["std"]
    ds = TornetChips("val", mean, std, include_range=in_ch == 27)
    out = []
    for x, _ in DataLoader(ds, batch_size=batch_size, num_workers=4):
        out.append(predict_proba(model, x.to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cnn", default=None, help="checkpoint to score on val (best.pt)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="json output (default: next to the checkpoint or logs/)")
    args = ap.parse_args(argv)

    root = tornet_root()
    tr_meta, va_meta = load_meta("train", root), load_meta("val", root)
    tr_keep = keep_indices("train", tr_meta)
    tr_f, va_f = hand_feature_table("train", root), hand_feature_table("val", root)
    va_centers = va_f[:, HAND_FEATURES.index("range_km")]
    y_tr = tr_meta["label"].to_numpy()
    results: dict = {"n_val": len(va_meta), "val_prevalence": float(va_meta["label"].mean())}

    gbm = train_gbm(tr_f[tr_keep], y_tr[tr_keep], seed=0)
    results["gbm"] = _scores(va_meta, gbm_predict_proba(gbm, va_f), va_centers, 0.5)
    rm = train_range_only(tr_f[tr_keep, -1], y_tr[tr_keep])
    results["range_only"] = _scores(va_meta, range_only_predict_proba(rm, va_centers), va_centers, 0.5)
    dv = va_f[:, HAND_FEATURES.index("max_dv")]
    results["delta_v_rule_40"] = _scores(va_meta, (dv >= DV_THRESHOLD).astype(float), va_centers, 0.5)
    # continuous shear as a score: ranking quality of the rule's feature without the 40 m/s cut
    results["delta_v_continuous"] = _scores(va_meta, dv / dv.max(), va_centers, DV_THRESHOLD / dv.max())

    if args.cnn:
        ckpt = Path(args.cnn)
        p = cnn_val_probs(ckpt, args.device)
        results["cnn_raw"] = _scores(va_meta, p, va_centers, 0.5)
        cal = IsotonicCalibrator().fit(p, va_meta["label"].to_numpy())
        (ckpt.with_name("calibrator.json")).write_text(json.dumps(cal.to_dict()))
        results["cnn_calibrated_on_val"] = _scores(va_meta, cal.transform(p), va_centers, 0.5)
        op = pick_operating_threshold(va_meta, cal.transform(p), va_centers)
        results["operating_point"] = op
        ckpt.with_name("operating_point.json").write_text(json.dumps(op, indent=1))
        # best-CSI cut on val (a val-selected operating point, reported alongside the spec's 0.5)
        grid = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 200)))
        csi = [_scores(va_meta, p, va_centers, float(t))["csi"] for t in grid]
        best = int(np.nanargmax(csi))
        results["cnn_raw_best_csi_threshold"] = float(grid[best])
        results["cnn_raw_at_best_csi"] = _scores(va_meta, p, va_centers, float(grid[best]))
    out = (
        Path(args.out)
        if args.out
        else (
            Path(args.cnn).with_name("val_scores.json")
            if args.cnn
            else Path("logs/tornado_baselines_val.json")
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
