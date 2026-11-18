"""Ice baselines (physics rule, zone-month climatology, logistic regression) and zone metrics (ICE-05, ICE-09)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def physics_rule(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    """ICE-05. ((CFRZR==1) | (CICEP==1) | ((PRATE>0) & (TMP:surface<=273.15))) & (TMP:surface<=273.15).

    Elementwise bool array; missing key -> KeyError. Temperatures in kelvin.
    """
    raise NotImplementedError


def physics_rule_zone(fields: Mapping[str, np.ndarray], mask: np.ndarray) -> float:
    """ICE-05. 1.0 if physics_rule is true in any in-zone cell, else 0.0."""
    raise NotImplementedError


class ZoneMonthClimatology:
    """ICE-05. Training positive rate per (zone, month); unseen pairs fall back to the global training rate."""

    def fit(
        self, zone_ids: Sequence[str], months: Sequence[int], labels: Sequence[int]
    ) -> ZoneMonthClimatology:
        """Fit rates. Returns self."""
        raise NotImplementedError

    def predict(self, zone_ids: Sequence[str], months: Sequence[int]) -> np.ndarray:
        """Rates in [0, 1]. Before fit -> RuntimeError."""
        raise NotImplementedError


def zone_features(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ICE-05. 36 values: in-zone mean of channels 0-17, then in-zone max of channels 0-17."""
    raise NotImplementedError


class LogisticZoneBaseline:
    """ICE-05. Standardised sklearn logistic regression on zone_features."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticZoneBaseline:
        """Fit. Returns self."""
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(label = 1) per row, in [0, 1]."""
        raise NotImplementedError


def zone_metrics(p: np.ndarray, y: np.ndarray, climatology: np.ndarray, threshold: float) -> dict[str, float]:
    """ICE-09. auprc, roc_auc, brier, brier_skill (vs climatology), csi, pod, far at p >= threshold.

    Empty input or single-class y -> ValueError; zero denominators -> 0.0.
    """
    raise NotImplementedError
