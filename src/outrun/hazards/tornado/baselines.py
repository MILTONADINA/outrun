"""Delta-V rule, gradient boosting on hand features, range-only shortcut model (TORN-06)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def delta_v_rule(chip: Mapping[str, np.ndarray], threshold: float = 40.0) -> float:
    """1.0 if the tilt-0 gate-to-gate shear ``max_dv`` >= threshold m/s, else 0.0 (TORN-06)."""
    raise NotImplementedError


def train_gbm(features: np.ndarray, labels: np.ndarray, seed: int = 0) -> HistGradientBoostingClassifier:
    """Fit HistGradientBoostingClassifier(random_state=seed) on (N, 7) hand features, NaN allowed (TORN-06)."""
    raise NotImplementedError


def gbm_predict_proba(model: HistGradientBoostingClassifier, features: np.ndarray) -> np.ndarray:
    """(N,) P(tornado) in [0, 1] (TORN-06)."""
    raise NotImplementedError


def train_range_only(range_km: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Logistic regression on range alone: the range-shortcut check (TORN-06)."""
    raise NotImplementedError


def range_only_predict_proba(model: LogisticRegression, range_km: np.ndarray) -> np.ndarray:
    """(N,) probabilities in [0, 1] (TORN-06)."""
    raise NotImplementedError
