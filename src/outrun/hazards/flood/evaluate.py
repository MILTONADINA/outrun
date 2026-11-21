"""Flood val scoring: rule baselines, logistic regression (fit on train) and, later, the U-Net, on one shared val set.

Val set = the FLOOD-13 balanced val patches (same sampler as training). One shared pixel mask for every row:
``y != -1`` (ring ignored) and ``valid == 1``. Pixel AUPRC is the headline; the rule baselines are {0, 1} so their
AUPRC is a single operating point (also given: POD, FAR, CSI at p >= 0.5). Patch-level: POD over positive patches
(FLOOD-12) and the alarm rate on hard-negative patches. Test samples are never opened here.

``python -m outrun.hazards.flood.evaluate baselines`` writes logs/flood_baselines_val.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from outrun.config import get_paths

from . import baselines as B
from .dataset import FLOOD_CHANNELS, SampleKind
from .metrics import event_pod
from .train import balanced_part


def score(prob: np.ndarray, X: np.ndarray, y: np.ndarray, kinds: list[SampleKind], thr: float = 0.5) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    mask = (y != -1) & (X[:, -1] > 0)
    p, t = prob[mask], (y[mask] == 1).astype(int)
    pred = p >= thr
    tp, fp, fn = int((pred & (t == 1)).sum()), int((pred & (t == 0)).sum()), int((~pred & (t == 1)).sum())
    pos_patches = [i for i, k in enumerate(kinds) if k == SampleKind.POSITIVE]
    hn_patches = [i for i, k in enumerate(kinds) if k == SampleKind.HARD_NEGATIVE]
    pod_patch = (
        float(np.mean([event_pod(prob[i], [(y[i] == 1) & (X[i, -1] > 0)], thr) for i in pos_patches]))
        if pos_patches
        else float("nan")
    )
    hn_alarm = (
        float(np.mean([bool(((prob[i] >= thr) & (X[i, -1] > 0)).any()) for i in hn_patches]))
        if hn_patches
        else float("nan")
    )
    return {
        "n_pixels": int(mask.sum()),
        "prevalence": float(t.mean()),
        "auprc": float(average_precision_score(t, p)),
        "roc_auc": float(roc_auc_score(t, p)),
        "brier": float(brier_score_loss(t, np.clip(p, 0, 1))),
        "pod_pixel": tp / max(tp + fn, 1),
        "far_pixel": fp / max(tp + fp, 1),
        "csi_pixel": tp / max(tp + fp + fn, 1),
        "pod_positive_patches": pod_patch,
        "alarm_rate_hard_neg_patches": hn_alarm,
    }


def load_arrays(part: str, balanced: bool = True):
    from outrun.eval.splits import read_split

    split = read_split(get_paths().splits / "flood.json")
    full, sub, kinds = balanced_part(split, part)
    idx = [int(r) for r in sub.indices]
    base = full.base
    X = np.stack([np.asarray(base.x[int(base.rows[i])], dtype=np.float32) for i in idx])
    Y = np.stack([np.asarray(base.y[int(base.rows[i])]) for i in idx])
    return X, Y, kinds


def baselines_val() -> dict:
    assert "test" not in sys.argv
    Xtr, Ytr, _ = load_arrays("train")
    Xva, Yva, kva = load_arrays("val")
    out: dict[str, dict] = {}
    out["ffg_ratio>=1"] = score(B.rule_baseline_from_features(Xva), Xva, Yva, kva)
    ari = B.ari_baseline(Xva[:, FLOOD_CHANNELS.index("ari_max")])
    out["qpe_ari>=10yr"] = score(ari, Xva, Yva, kva)
    out["ari_rule_3h50_6h75"] = score(B.ari_rule_fallback(Xva[:, 1], Xva[:, 2]), Xva, Yva, kva)
    lr = B.LogisticBaseline(seed=0).fit(Xtr, Ytr)
    out["logistic_regression"] = score(lr.predict_proba(Xva), Xva, Yva, kva)
    out["_meta"] = {
        "n_val_patches": len(kva),
        "n_train_patches": len(Xtr),
        "val_kinds": {k.value: sum(1 for x in kva if x == k) for k in SampleKind},
    }
    return out


def unet_val(checkpoint: str | Path, use_flash: bool = True, device: str = "cpu") -> dict:
    """Score a FloodUNet checkpoint on the same val set and mask as the baselines (selection reference, not test)."""
    import torch

    from outrun.hazards.base import load_checkpoint

    from .dataset import select_channels
    from .model import FloodUNet

    X, Y, kinds = load_arrays("val")
    net = FloodUNet(in_channels=11 if use_flash else 8)
    load_checkpoint(checkpoint, net)
    net.to(device).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(np.ascontiguousarray(select_channels(X[i : i + 256], use_flash))).to(device)
            probs.append(torch.sigmoid(net(xb))[:, 0].cpu().numpy())
    return score(np.concatenate(probs), X, Y, kinds)


if __name__ == "__main__":
    res = baselines_val()
    best = Path(get_paths().models / "flood" / "best.pt")
    if best.exists():
        res["flood_unet_best_epoch (val-selected)"] = unet_val(best, device="mps")
    path = Path("logs/flood_baselines_val.json")
    path.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
