"""NDWS validation scores for the wildfire baselines and the U-Net (selection data only; test is never read).

``python -m outrun.hazards.wildfire.val_eval [--model-dir DIR]`` prints and writes ``val_metrics.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from outrun.config import get_paths

from .baselines import PixelLogistic, downwind_persistence, persistence
from .dataset import NdwsDataset, normalize
from .metrics import fire_pr_auc
from .train import fit_train_stats, load_stats, ndws_files


def brier(prob: np.ndarray, label: np.ndarray) -> float:
    keep = label != -1
    return float(np.mean((prob[keep].astype(np.float64) - (label[keep] == 1)) ** 2))


def best_f1(prob: np.ndarray, label: np.ndarray) -> dict[str, float]:
    """Max F1 over thresholds (chosen on val, so not a test number)."""
    from sklearn.metrics import precision_recall_curve

    keep = label != -1
    p, r, t = precision_recall_curve(label[keep] == 1, prob[keep])
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    i = int(np.argmax(f1))
    return {"f1": float(f1[i]), "threshold": float(t[i])}


def unet_val_probs(model_dir: Path, x_va: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """(raw logits, normalised x) of the best checkpoint on val rows (N, 64, 64)."""
    import torch

    stats = load_stats(model_dir / "norm_stats.npz")
    payload = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
    net = None
    from .model import FireUNet

    net = FireUNet.from_state_dict(payload["model"]).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x_va), 256):
            z = torch.from_numpy(normalize(x_va[i : i + 256], stats)).to(device)
            out.append(net(z)[:, 0].cpu().numpy())
    return np.concatenate(out), x_va


def evaluate(model_dir: Path | None, with_lr: bool = True) -> dict:
    from outrun.eval.splits import read_split

    split = read_split(get_paths().splits / "ndws.json")
    x_tr, y_tr, ids_tr = ndws_files("train")
    x_va, y_va, ids_va = ndws_files("eval")
    stats_path = (model_dir / "norm_stats.npz") if model_dir else None
    stats = load_stats(stats_path) if stats_path and stats_path.exists() else fit_train_stats(x_tr, ids_tr)
    va = NdwsDataset.from_split(x_va, y_va, ids_va, split, "val", stats)
    x, y = np.asarray(va.x[va.rows]), np.asarray(va.y[va.rows])
    prev = x[:, 11]
    out: dict[str, dict] = {}

    def add(name: str, prob: np.ndarray) -> None:
        d = fire_pr_auc(prob, y, prev)
        d.update(brier=brier(prob, y), **best_f1(prob, y))
        out[name] = d
        print(
            f"{name:22s} PR-AUC all {d['all']:.4f}  new-burn {d['new_burn']:.4f}  Brier {d['brier']:.5f}  "
            f"F1 {d['f1']:.3f}",
            flush=True,
        )

    out["val_prevalence"] = {
        "all": float((y[y != -1] == 1).mean()),
        "new_burn": float((y[(prev == 0) & (y != -1)] == 1).mean()),
    }
    print("val prevalence (all / new-burn):", out["val_prevalence"], flush=True)
    add("persistence", persistence(x))
    add("downwind persistence", downwind_persistence(x))
    if with_lr:
        ds_tr = NdwsDataset.from_split(x_tr, y_tr, ids_tr, split, "train", stats)
        xs = normalize(np.asarray(ds_tr.x[ds_tr.rows]), stats)
        lr = PixelLogistic(seed=0).fit(xs, np.asarray(ds_tr.y[ds_tr.rows]))
        add("pixel logistic", lr.predict_proba(normalize(x, stats)))
    if model_dir and (model_dir / "best.pt").exists():
        from .model import TemperatureScaler

        logits, _ = unet_val_probs(model_dir, x)
        p_raw = 1.0 / (1.0 + np.exp(-logits))
        add("U-Net (uncalibrated)", p_raw)
        cal = TemperatureScaler().fit(logits, y)  # fitted on val: selection data
        cal.save(model_dir / "calibrator.json")
        print(f"calibration: T={cal.temperature:.3f} bias={cal.bias:.3f}", flush=True)
        add("U-Net (calibrated)", cal.transform(p_raw))
        out["calibration"] = {"temperature": cal.temperature, "bias": cal.bias}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--no-lr", action="store_true")
    a = ap.parse_args()
    md = Path(a.model_dir) if a.model_dir else get_paths().models / "wildfire"
    res = evaluate(md if md.exists() else None, with_lr=not a.no_lr)
    md.mkdir(parents=True, exist_ok=True)
    (md / "val_metrics.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
