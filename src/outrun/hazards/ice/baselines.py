"""Ice baselines (physics rule, zone-month climatology, logistic regression) and zone metrics (ICE-05, ICE-09)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KELVIN_0C = 273.15


def physics_rule(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    """ICE-05. ((CFRZR==1) | (CICEP==1) | ((PRATE>0) & (TMP:surface<=273.15))) & (TMP:surface<=273.15).

    Elementwise bool array; missing key -> KeyError. Temperatures in kelvin.
    """
    t = np.asarray(fields["TMP:surface"])
    cold = t <= KELVIN_0C
    freezing = (np.asarray(fields["CFRZR"]) == 1) | (np.asarray(fields["CICEP"]) == 1)
    wet = np.asarray(fields["PRATE"]) > 0
    return (freezing | (wet & cold)) & cold


def physics_rule_zone(fields: Mapping[str, np.ndarray], mask: np.ndarray) -> float:
    """ICE-05. 1.0 if physics_rule is true in any in-zone cell, else 0.0."""
    return float(np.any(physics_rule(fields) & np.asarray(mask, bool)))


class ZoneMonthClimatology:
    """ICE-05. Training positive rate per (zone, month); unseen pairs fall back to the global training rate."""

    _rate: dict[tuple[str, int], float] | None = None
    _global: float = 0.0

    def fit(
        self, zone_ids: Sequence[str], months: Sequence[int], labels: Sequence[int]
    ) -> ZoneMonthClimatology:
        """Fit rates. Returns self."""
        self._global = float(np.mean(labels))
        total: dict[tuple[str, int], list[float]] = {}
        for z, m, y in zip(zone_ids, months, labels, strict=True):
            cell = total.setdefault((str(z), int(m)), [0.0, 0.0])
            cell[0] += float(y)
            cell[1] += 1.0
        self._rate = {k: v[0] / v[1] for k, v in total.items()}
        return self

    def predict(self, zone_ids: Sequence[str], months: Sequence[int]) -> np.ndarray:
        """Rates in [0, 1]. Before fit -> RuntimeError."""
        if self._rate is None:
            raise RuntimeError("ZoneMonthClimatology.predict called before fit")
        return np.array(
            [self._rate.get((str(z), int(m)), self._global) for z, m in zip(zone_ids, months, strict=True)]
        )


def zone_features(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """ICE-05. 36 values: in-zone mean of channels 0-17, then in-zone max of channels 0-17."""
    mask = np.asarray(mask, bool)
    vals = np.asarray(x[:18], dtype=np.float64)[:, mask]
    return np.concatenate([vals.mean(axis=1), vals.max(axis=1)])


class LogisticZoneBaseline:
    """ICE-05. Standardised sklearn logistic regression on zone_features."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticZoneBaseline:
        """Fit. Returns self."""
        self._model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=self.seed)
        ).fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(label = 1) per row, in [0, 1]."""
        if self._model is None:
            raise RuntimeError("LogisticZoneBaseline.predict_proba called before fit")
        return self._model.predict_proba(X)[:, 1]


def zone_metrics(p: np.ndarray, y: np.ndarray, climatology: np.ndarray, threshold: float) -> dict[str, float]:
    """ICE-09. auprc, roc_auc, brier, brier_skill (vs climatology), csi, pod, far at p >= threshold.

    Empty input or single-class y -> ValueError; zero denominators -> 0.0.
    """
    p, y, climatology = (np.asarray(a, dtype=float) for a in (p, y, climatology))
    if p.size == 0 or y.size == 0:
        raise ValueError("empty input")
    if len(np.unique(y)) < 2:
        raise ValueError("y must contain both classes")
    brier = float(np.mean((p - y) ** 2))
    brier_clim = float(np.mean((climatology - y) ** 2))
    hit = p >= threshold
    tp = float(np.sum(hit & (y == 1)))
    fp = float(np.sum(hit & (y == 0)))
    fn = float(np.sum(~hit & (y == 1)))

    def ratio(a: float, b: float) -> float:
        return a / b if b > 0 else 0.0

    return {
        "auprc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": brier,
        "brier_skill": 1.0 - brier / brier_clim if brier_clim > 0 else 0.0,
        "csi": ratio(tp, tp + fp + fn),
        "pod": ratio(tp, tp + fn),
        "far": ratio(fp, tp + fp),
    }
