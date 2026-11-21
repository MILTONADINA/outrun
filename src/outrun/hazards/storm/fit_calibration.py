"""Fit the per-lead isotonic calibrator of a storm checkpoint on VAL only (STORM-07).

``python -m outrun.hazards.storm.fit_calibration [checkpoint]`` writes ``calibrator.json`` next to the checkpoint and
prints Brier score and reliability per lead before and after. Data come from ``SevirWindowDataset.from_split`` (which
applies the frozen no-data-frame filter) on the ``val`` part only; test and quarantined case events are never read.
The in-sample numbers are optimistic, so a 2-fold check (fit on half the val events, score on the other half) is
printed as well.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from outrun.core.calibrate import brier_score, reliability_bins
from outrun.eval.splits import read_split

from .adapter import StormAdapter
from .calibration import PerLeadCalibrator
from .dataset import N_OUT, SevirWindowDataset
from .model import RISK_THRESHOLD, RISK_THRESHOLD_INDEX


def val_predictions(checkpoint: Path, batch_size: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw (uncalibrated) P(VIL >= 133) ``(N, 12, h, w)`` float32, truth bool, and the event row of each window."""
    from outrun.config import get_paths

    paths = get_paths()
    meta = pd.read_parquet(paths.proc / "sevir" / "sevir.meta.parquet")
    event_ids = meta["id"].astype(str).tolist()
    events = np.load(paths.proc / "sevir" / "sevir.npy", mmap_mode="r")
    split = read_split(paths.splits / "sevir.json")
    model = StormAdapter(checkpoint=checkpoint)._load_model()
    assert model is not None
    ds = SevirWindowDataset.from_split(events, event_ids, split, "val", use_ir=model.in_channels == 26)
    probs, truth = [], []
    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            xs, ys = zip(*(ds[i] for i in range(start, min(start + batch_size, len(ds)))), strict=True)
            _, logits = model(torch.stack(xs))
            probs.append(torch.sigmoid(logits)[:, :, RISK_THRESHOLD_INDEX].numpy())
            truth.append(torch.stack(ys).numpy() >= RISK_THRESHOLD)
    return np.concatenate(probs), np.concatenate(truth), ds.windows[:, 0]


def lead_scores(p: np.ndarray, y: np.ndarray) -> list[dict]:
    out = []
    for lead in range(p.shape[1]):
        rel = reliability_bins(p[:, lead], y[:, lead], n_bins=10)
        n = rel.counts.sum()
        gap = np.nansum(rel.counts * np.abs(rel.mean_pred - rel.frac_pos)) / n
        out.append({"lead_min": 5 * (lead + 1), "brier": brier_score(p[:, lead], y[:, lead]), "ece": float(gap)})
    return out


def main(checkpoint: Path) -> dict:
    p, y, win_event = val_predictions(checkpoint)
    cal = PerLeadCalibrator().fit(p, y)
    after = cal.transform(p)
    # honest check: fit on the even events, score on the odd ones, and the other way round
    folds = (np.unique(win_event) % 2 == 0)[np.searchsorted(np.unique(win_event), win_event)]
    oos = np.empty_like(p)
    for part in (folds, ~folds):
        oos[~part] = PerLeadCalibrator().fit(p[part], y[part]).transform(p[~part])
    report = {
        "n_windows": int(len(p)),
        "base_rate_by_lead": y.mean(axis=(0, 2, 3)).round(5).tolist(),
        "before": lead_scores(p, y),
        "after_in_sample": lead_scores(after, y),
        "after_2fold": lead_scores(oos, y),
    }
    cal_path = Path(checkpoint).with_name("calibrator.json")
    cal_path.write_text(json.dumps(cal.to_dict()))
    (cal_path.with_name("calibration_val.json")).write_text(json.dumps(report, indent=1))
    print(f"wrote {cal_path} ({N_OUT} leads, {len(p)} val windows)")
    for name in ("before", "after_in_sample", "after_2fold"):
        s = report[name]
        print(f"{name:16s} brier mean {np.mean([r['brier'] for r in s]):.5f} ece mean {np.mean([r['ece'] for r in s]):.5f}"
              f"  | lead5 {s[0]['brier']:.5f}/{s[0]['ece']:.4f} lead30 {s[5]['brier']:.5f}/{s[5]['ece']:.4f}"
              f" lead60 {s[11]['brier']:.5f}/{s[11]['ece']:.4f}")
    return report


if __name__ == "__main__":
    from outrun.config import get_paths

    main(Path(sys.argv[1]) if len(sys.argv) > 1 else get_paths().models / "storm" / "best.pt")
