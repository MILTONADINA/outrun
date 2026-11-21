"""Fit the flood isotonic calibrator on VAL only and report val metrics before/after (FLOOD-10, CAL-*).

``python -m outrun.hazards.flood.calibrate_val [checkpoint] [--no-flash]`` writes ``calibrator.json`` and
``calibration_val.json`` next to the checkpoint. Pixels = the shared val mask of ``evaluate.score`` (ring and invalid
pixels ignored). The val set is the FLOOD-13 balanced sample (prevalence about 5 %, not the true base rate), so the
calibrated P is calibrated to that sample; the 2-fold numbers (fit on even patches, score on odd, and back) are the
honest ones. Test samples are never opened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from outrun.core.calibrate import IsotonicCalibrator, brier_score, expected_calibration_error

from .dataset import select_channels
from .evaluate import load_arrays, score
from .model import FloodUNet


def raw_val_probs(checkpoint: Path, use_flash: bool, device: str = "cpu"):
    from outrun.hazards.base import load_checkpoint

    X, Y, kinds = load_arrays("val")
    net = FloodUNet(in_channels=11 if use_flash else 8)
    load_checkpoint(checkpoint, net)
    net.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(np.ascontiguousarray(select_channels(X[i : i + 256], use_flash))).to(device)
            out.append(torch.sigmoid(net(xb))[:, 0].cpu().numpy())
    return np.concatenate(out), X, Y, kinds


def main(checkpoint: Path, use_flash: bool = True) -> dict:
    p, X, Y, kinds = raw_val_probs(checkpoint, use_flash)
    mask = (Y != -1) & (X[:, -1] > 0)
    t = Y == 1
    rng = np.random.default_rng(0)
    pm, tm = p[mask], t[mask]
    sub = rng.choice(
        len(pm), size=min(len(pm), 4_000_000), replace=False
    )  # isotonic fit on a pixel subsample
    cal = IsotonicCalibrator().fit(pm[sub], tm[sub])
    after = cal.transform(p).astype(np.float32)
    even = (np.arange(len(p)) % 2 == 0)[:, None, None] & np.ones_like(mask)
    oos = np.empty_like(p)
    for part in (even, ~even):
        m = mask & part
        s = rng.choice(int(m.sum()), size=min(int(m.sum()), 2_000_000), replace=False)
        c = IsotonicCalibrator().fit(p[m][s], t[m][s])
        oos[~part] = c.transform(p[~part])
    rep = {}
    for name, q in (("raw", p), ("isotonic_in_sample", after), ("isotonic_2fold", oos)):
        r = score(q, X, Y, kinds)
        r["ece"] = expected_calibration_error(q[mask], t[mask])
        r["brier_check"] = brier_score(q[mask], t[mask])
        rep[name] = r
    out = Path(checkpoint).with_name("calibrator.json")
    out.write_text(json.dumps(cal.to_dict()))
    out.with_name("calibration_val.json").write_text(json.dumps(rep, indent=1))
    for k, v in rep.items():
        print(
            k,
            {
                a: round(b, 4)
                for a, b in v.items()
                if a
                in (
                    "auprc",
                    "roc_auc",
                    "brier",
                    "ece",
                    "pod_pixel",
                    "far_pixel",
                    "csi_pixel",
                    "alarm_rate_hard_neg_patches",
                )
            },
        )
    return rep


if __name__ == "__main__":
    assert "test" not in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(Path(args[0]) if args else Path("models/flood/best.pt"), use_flash="--no-flash" not in sys.argv)
