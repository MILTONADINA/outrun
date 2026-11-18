"""Fire PR-AUC on all and new-burn pixels, per-fire macro average (specs/11-wildfire.md: FIRE-08)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def pr_auc(prob: np.ndarray, label: np.ndarray) -> float:
    """Average precision over label != -1; nan if no positives; shape mismatch -> ValueError (FIRE-08)."""
    raise NotImplementedError


def fire_pr_auc(prob: np.ndarray, label: np.ndarray, prev_mask: np.ndarray) -> dict[str, float]:
    """{"all": pr_auc, "new_burn": pr_auc on prev_mask == 0 and label != -1} (FIRE-08)."""
    raise NotImplementedError


def per_fire_macro(values: Sequence[float]) -> float:
    """Mean of the finite values; nan if none (FIRE-08)."""
    raise NotImplementedError
