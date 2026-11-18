"""Flood baselines: FLASH FFG ratio, QPE ARI, ARI-style fallback, logistic regression (FLOOD-06, FLOOD-07)."""

from __future__ import annotations

import numpy as np


def ffg_ratio_baseline(
    ffg_ratio_1h: np.ndarray, ffg_ratio_3h: np.ndarray, threshold: float = 1.0
) -> np.ndarray:
    """float32 1 where max(ratio_1h, ratio_3h) >= threshold, else 0; NaN -> 0 (FLOOD-06)."""
    raise NotImplementedError


def ari_baseline(ari_max_years: np.ndarray, threshold_years: float = 10.0) -> np.ndarray:
    """float32 1 where max QPE ARI >= threshold_years, else 0; NaN -> 0 (FLOOD-06)."""
    raise NotImplementedError


def ari_rule_fallback(
    qpe_3h: np.ndarray, qpe_6h: np.ndarray, mm_3h: float = 50.0, mm_6h: float = 75.0
) -> np.ndarray:
    """float32 1 where qpe_3h >= 50 mm or qpe_6h >= 75 mm, else 0; NaN -> 0 (FLOOD-06)."""
    raise NotImplementedError


def rule_baseline_from_features(features: np.ndarray) -> np.ndarray:
    """ffg_ratio_baseline on the FFG channels of an (11,H,W) or (N,11,H,W) FLOOD-01 stack (FLOOD-06)."""
    raise NotImplementedError


class LogisticBaseline:
    """Per-pixel logistic regression on the FLOOD-01 channels; ignores y == -1 and valid == 0 (FLOOD-07)."""

    def __init__(self, C: float = 1.0, seed: int = 0) -> None:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticBaseline:
        """X (N, 11, H, W), y (N, H, W); fewer than two classes kept -> ValueError."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """float32 (N, H, W) in [0, 1], 0 where valid == 0; before fit -> RuntimeError."""
        raise NotImplementedError
