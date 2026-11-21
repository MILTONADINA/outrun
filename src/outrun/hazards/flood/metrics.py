"""Flood-specific metrics (specs/10-flood.md: FLOOD-12). Generic metrics live in outrun.eval.metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def event_pod(prob: np.ndarray, report_masks: Sequence[np.ndarray], threshold: float = 0.5) -> float:
    """Fraction of reports whose footprint has a pixel with prob >= threshold; nan if no reports;
    mask shape mismatch -> ValueError (FLOOD-12)."""
    prob = np.asarray(prob)
    hit = 0
    for m in report_masks:
        m = np.asarray(m, dtype=bool)
        if m.shape != prob.shape:
            raise ValueError(f"mask shape {m.shape} != prob shape {prob.shape}")
        hit += bool((prob[m] >= threshold).any())
    return hit / len(report_masks) if len(report_masks) else float("nan")
