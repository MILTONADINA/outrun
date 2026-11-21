"""T16: score the trained MIL-CNN on VAL only, fit the zone calibrator on val, report the bridge prior.

``OUTRUN_PROC=... .venv/bin/python -m outrun.hazards.ice.evaluate [--device mps]``. Test bags are never loaded.
Calibration is fitted on val, so its honest Brier is reported by 2-fold cross-fitting over val zones; the saved
calibrator (``models/ice/calibrator.pkl``) is fitted on all of val.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np

from outrun.config import get_paths
from outrun.hazards.base import load_checkpoint
from outrun.hazards.ice.baselines import LogisticZoneBaseline, ZoneMonthClimatology, zone_metrics
from outrun.hazards.ice.eval_baselines import best_threshold, features
from outrun.hazards.ice.model import ZoneCalibrator
from outrun.hazards.ice.train import BagStore, ZoneLogitNet, _Loader, load_bags, zone_probabilities


def bridge_report(graph_path: Path) -> dict:
    """Bridge edges (OSM ``bridge`` flag, ROAD-08) of the OKC road graph that the ICE bridge prior applies to."""
    from outrun.roads.graph import load_npz

    g = load_npz(graph_path)
    return {"graph": graph_path.name, "edges": int(g.n_edges), "bridge_edges": int(np.sum(g.bridge))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--model-dir", type=Path, default=get_paths().models / "ice")
    a = ap.parse_args(argv)
    d, parts = load_bags(("train", "val"))
    tr, va = parts["train"], parts["val"]
    store = BagStore(d)
    model = ZoneLogitNet()
    load_checkpoint(a.model_dir / "best.pt", model)
    model.to(a.device)
    p = zone_probabilities(model, _Loader(store, va, 256, False), a.device)
    y = va["label"].to_numpy()
    month = lambda df: df["window_start"].dt.month.to_numpy()  # noqa: E731
    clim = ZoneMonthClimatology().fit(tr["zone_id"], month(tr), tr["label"])
    p_clim = clim.predict(va["zone_id"], month(va))
    scan = np.load(d / "scan.npz")
    lr = LogisticZoneBaseline(0).fit(features(tr, scan), tr["label"].to_numpy())
    p_lr = lr.predict_proba(features(va, scan))
    # honest calibrated scores: cross-fit the isotonic map over two zone-disjoint halves of val
    fold = np.array([zlib.crc32(z.encode()) % 2 for z in va["zone_id"]])
    p_cv = np.zeros_like(p)
    for k in (0, 1):
        p_cv[fold == k] = ZoneCalibrator().fit(p[fold != k], y[fold != k]).transform(p[fold == k])
    cal = ZoneCalibrator().fit(p, y)
    cal.save(a.model_dir / "calibrator.pkl")
    thr_raw, thr_cv = best_threshold(p, y), best_threshold(p_cv, y)
    rep = {
        "val_bags": len(va),
        "val_positives": int(y.sum()),
        "milcnn_raw": zone_metrics(p, y, p_clim, thr_raw) | {"threshold": thr_raw},
        "milcnn_calibrated_crossfit": zone_metrics(p_cv, y, p_clim, thr_cv) | {"threshold": thr_cv},
        "logistic": zone_metrics(p_lr, y, p_clim, best_threshold(p_lr, y)),
        "physics_rule": zone_metrics(va["rule"].to_numpy(float), y, p_clim, 0.5),
        "climatology": zone_metrics(p_clim, y, p_clim, best_threshold(p_clim, y)),
        "calibration_note": "positive rate inflated by 1-in-20 negative sampling; calibrated probs refer to that universe",
        "bridges": bridge_report(get_paths().proc / "roads" / "okc.npz"),
    }
    (a.model_dir / "val_metrics.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
