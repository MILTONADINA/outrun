"""Flood-specific metrics (specs/10-flood.md: FLOOD-12). Generic metrics live in outrun.eval.metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def event_pod(prob: np.ndarray, report_masks: Sequence[np.ndarray], threshold: float = 0.5) -> float:
    """Fraction of reports whose footprint has a pixel with prob >= threshold; nan if no reports;
    mask shape mismatch -> ValueError (FLOOD-12)."""
    raise NotImplementedError
