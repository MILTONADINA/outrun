"""T15: gate checks and baseline scores on VAL only (physics rule, zone-month climatology, logistic regression).

``.venv/bin/python -m outrun.hazards.ice.eval_baselines``. Fits on train, thresholds on val. Test is never scored;
only its size and label count are printed (gate check).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from outrun.config import get_paths
from outrun.eval.splits import read_split
from outrun.hazards.ice.baselines import LogisticZoneBaseline, ZoneMonthClimatology, zone_metrics
from outrun.hazards.ice.dataset import bags_for_split


def features(bags: pd.DataFrame, scan: np.lib.npyio.NpzFile) -> np.ndarray:
    """Same 36-value layout as zone_features: mean(t), mean(t-3h), max(t), max(t-3h) over the whole zone."""
    z, it, im = bags["zone_index"].to_numpy(), bags["i_t"].to_numpy(), bags["i_tm3"].to_numpy()
    return np.concatenate(
        [scan["mean"][it, z], scan["mean"][im, z], scan["max"][it, z], scan["max"][im, z]], axis=1
    ).astype(np.float64)


def best_threshold(p: np.ndarray, y: np.ndarray) -> float:
    cands = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 200)))

    def csi(t: float) -> float:
        hit = p >= t
        tp, fp, fn = (hit & (y == 1)).sum(), (hit & (y == 0)).sum(), (~hit & (y == 1)).sum()
        return tp / max(tp + fp + fn, 1)

    return float(max(cands, key=csi))


def main() -> None:
    d = get_paths().proc / "hrrr" / "ice" / "bags"
    bags = pd.read_parquet(d / "bags.parquet")
    zones = np.load(d / "zones.npz")
    zindex = {z: i for i, z in enumerate(zones["zone_ids"])}
    bags["zone_index"] = bags["zone_id"].map(zindex)
    scan = np.load(d / "scan.npz")
    split = read_split(Path("splits/ice.json"))
    parts = {p: bags_for_split(bags, split, p) for p in ("train", "val", "test")}
    report: dict = {"counts": {}}
    for p, df in parts.items():
        report["counts"][p] = {
            "bags": len(df),
            "positives": int(df.label.sum()),
            "prevalence": float(df.label.mean()),
            "hard_neg_frac_of_negatives": float(df[df.label == 0].hard_negative.mean()),
        }
    tr, va = parts["train"], parts["val"]
    y = va["label"].to_numpy()
    month = lambda df: df["window_start"].dt.month.to_numpy()  # noqa: E731
    clim = ZoneMonthClimatology().fit(tr["zone_id"], month(tr), tr["label"])
    p_clim = clim.predict(va["zone_id"], month(va))
    lr = LogisticZoneBaseline(0).fit(features(tr, scan), tr["label"].to_numpy())
    p_lr = lr.predict_proba(features(va, scan))
    p_rule = va["rule"].to_numpy(float)
    report["val"] = {
        "physics_rule": zone_metrics(p_rule, y, p_clim, 0.5),
        "climatology": zone_metrics(p_clim, y, p_clim, best_threshold(p_clim, y)),
        "logistic": zone_metrics(p_lr, y, p_clim, best_threshold(p_lr, y)),
    }
    report["val_note"] = "rule is a hard 0/1 score; thresholds for climatology/LR chosen on val for max CSI"
    out = d / "baselines_val.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
