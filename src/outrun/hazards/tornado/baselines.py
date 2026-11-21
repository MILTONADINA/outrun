"""Delta-V rule, gradient boosting on hand features, range-only shortcut model (TORN-06)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .features import max_dv


def delta_v_rule(chip: Mapping[str, np.ndarray], threshold: float = 40.0) -> float:
    """1.0 if the tilt-0 gate-to-gate shear ``max_dv`` >= threshold m/s, else 0.0 (TORN-06)."""
    return 1.0 if max_dv(chip, tilt=0) >= threshold else 0.0


def train_gbm(features: np.ndarray, labels: np.ndarray, seed: int = 0) -> HistGradientBoostingClassifier:
    """Fit HistGradientBoostingClassifier(random_state=seed) on (N, 7) hand features, NaN allowed (TORN-06)."""
    return HistGradientBoostingClassifier(random_state=seed).fit(features, labels)


def gbm_predict_proba(model: HistGradientBoostingClassifier, features: np.ndarray) -> np.ndarray:
    """(N,) P(tornado) in [0, 1] (TORN-06)."""
    return model.predict_proba(features)[:, list(model.classes_).index(1)]


def train_range_only(range_km: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Logistic regression on range alone: the range-shortcut check (TORN-06)."""
    return LogisticRegression().fit(np.asarray(range_km, dtype=np.float64).reshape(-1, 1), labels)


def range_only_predict_proba(model: LogisticRegression, range_km: np.ndarray) -> np.ndarray:
    """(N,) probabilities in [0, 1] (TORN-06)."""
    x = np.asarray(range_km, dtype=np.float64).reshape(-1, 1)
    return model.predict_proba(x)[:, list(model.classes_).index(1)]
