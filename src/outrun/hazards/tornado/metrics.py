"""Detection scores incl. the warned-null AUC headline and range bins (TORN-07).

May delegate to ``outrun.eval.metrics``; the definitions in the spec are the contract.
"""

from __future__ import annotations

import numpy as np

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
    raise NotImplementedError
