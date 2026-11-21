"""Fire PR-AUC on all and new-burn pixels, per-fire macro average (specs/11-wildfire.md: FIRE-08)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pr_auc(prob: np.ndarray, label: np.ndarray) -> float:
    """Average precision over label != -1; nan if no positives; shape mismatch -> ValueError (FIRE-08)."""
    from sklearn.metrics import average_precision_score

    prob, label = np.asarray(prob), np.asarray(label)
    if prob.shape != label.shape:
        raise ValueError(f"prob {prob.shape} and label {label.shape} must have the same shape")
    keep = label != -1
    pos = label[keep] == 1
    if not pos.any():
        return float("nan")
    return float(average_precision_score(pos, prob[keep]))


def fire_pr_auc(prob: np.ndarray, label: np.ndarray, prev_mask: np.ndarray) -> dict[str, float]:
    """{"all": pr_auc, "new_burn": pr_auc on prev_mask == 0 and label != -1} (FIRE-08)."""
    prob, label, prev = np.asarray(prob), np.asarray(label), np.asarray(prev_mask)
    if not prob.shape == label.shape == prev.shape:
        raise ValueError("prob, label and prev_mask must have the same shape")
    new = prev == 0
    return {"all": pr_auc(prob, label), "new_burn": pr_auc(prob[new], label[new])}


def per_fire_macro(values: Sequence[float]) -> float:
    """Mean of the finite values; nan if none (FIRE-08)."""
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")
