"""Detection scores incl. the warned-null AUC headline and range bins (TORN-07).

May delegate to ``outrun.eval.metrics``; the definitions in the spec are the contract.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

RANGE_BINS: dict[str, tuple[float, float]] = {
    "0-50": (0.0, 50.0),
    "50-100": (50.0, 100.0),
    "100+": (100.0, np.inf),
}


def tornado_scores(
    y_true: np.ndarray,
    p: np.ndarray,
    category: np.ndarray,
    range_km: np.ndarray,
    threshold: float = 0.5,
    positive: str = "TOR",
    warned_null: str = "WRN",
) -> dict:
    """auc, auprc, auc_warned, auc_by_range, brier, pod, far, csi; nan for undefined scores (TORN-07)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=np.float64)
    category = np.asarray(category)
    rng = np.asarray(range_km, dtype=np.float64)
    if not (len(y) == len(p) == len(category) == len(rng)):
        raise ValueError("y_true, p, category and range_km must have the same length")

    def auc(mask: np.ndarray) -> float:
        if len(np.unique(y[mask])) < 2:
            return float("nan")
        return float(roc_auc_score(y[mask], p[mask]))

    everything = np.ones(len(y), dtype=bool)
    warned = (y == 1) | (category == warned_null)
    hit = int(((p >= threshold) & (y == 1)).sum())
    miss = int(((p < threshold) & (y == 1)).sum())
    false_alarm = int(((p >= threshold) & (y == 0)).sum())

    def ratio(num: int, den: int) -> float:
        return num / den if den > 0 else float("nan")

    return {
        "auc": auc(everything),
        "auprc": float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
        "auc_warned": auc(warned),
        "auc_by_range": {name: auc((rng >= lo) & (rng < hi)) for name, (lo, hi) in RANGE_BINS.items()},
        "brier": float(np.mean((p - y) ** 2)) if len(y) else float("nan"),
        "pod": ratio(hit, hit + miss),
        "far": ratio(false_alarm, hit + false_alarm),
        "csi": ratio(hit, hit + miss + false_alarm),
    }
