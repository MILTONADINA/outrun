"""Flood baselines: FLASH FFG ratio, QPE ARI, ARI-style fallback, logistic regression (FLOOD-06, FLOOD-07)."""

from __future__ import annotations

import numpy as np

from .dataset import FLOOD_CHANNELS


def _ge(a: np.ndarray, thr: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return np.where(np.isfinite(a) & (a >= thr), 1.0, 0.0).astype(np.float32)


def ffg_ratio_baseline(
    ffg_ratio_1h: np.ndarray, ffg_ratio_3h: np.ndarray, threshold: float = 1.0
) -> np.ndarray:
    """float32 1 where max(ratio_1h, ratio_3h) >= threshold, else 0; NaN -> 0 (FLOOD-06)."""
    return np.maximum(_ge(ffg_ratio_1h, threshold), _ge(ffg_ratio_3h, threshold))


def ari_baseline(ari_max_years: np.ndarray, threshold_years: float = 10.0) -> np.ndarray:
    """float32 1 where max QPE ARI >= threshold_years, else 0; NaN -> 0 (FLOOD-06)."""
    return _ge(ari_max_years, threshold_years)


def ari_rule_fallback(
    qpe_3h: np.ndarray, qpe_6h: np.ndarray, mm_3h: float = 50.0, mm_6h: float = 75.0
) -> np.ndarray:
    """float32 1 where qpe_3h >= 50 mm or qpe_6h >= 75 mm, else 0; NaN -> 0 (FLOOD-06)."""
    return np.maximum(_ge(qpe_3h, mm_3h), _ge(qpe_6h, mm_6h))


def rule_baseline_from_features(features: np.ndarray) -> np.ndarray:
    """ffg_ratio_baseline on the FFG channels of an (11,H,W) or (N,11,H,W) FLOOD-01 stack (FLOOD-06)."""
    f = np.asarray(features)
    a, b = FLOOD_CHANNELS.index("ffg_ratio_1h"), FLOOD_CHANNELS.index("ffg_ratio_3h")
    return ffg_ratio_baseline(f[..., a, :, :], f[..., b, :, :])


class LogisticBaseline:
    """Per-pixel logistic regression on the FLOOD-01 channels; ignores y == -1 and valid == 0 (FLOOD-07)."""

    _LOG = tuple(
        FLOOD_CHANNELS.index(c)
        for c in ("qpe_1h", "qpe_3h", "qpe_6h", "qpe_24h", "ffg_ratio_1h", "ffg_ratio_3h", "ari_max")
    )

    def __init__(self, C: float = 1.0, seed: int = 0, max_pixels: int = 2_000_000) -> None:
        self.C, self.seed, self.max_pixels = C, seed, max_pixels
        self._model = None
        self._mu = self._sd = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        """(N, 11, H, W) -> (N*H*W, 10): signed log1p on heavy-tailed channels (valid mask is not a feature)."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 4 or X.shape[1] != len(FLOOD_CHANNELS):
            raise ValueError(f"expected (N, 11, H, W), got {X.shape}")
        Z = X.copy()
        for c in self._LOG:
            Z[:, c] = np.sign(Z[:, c]) * np.log1p(np.abs(Z[:, c]))
        return Z[:, :-1].transpose(0, 2, 3, 1).reshape(-1, len(FLOOD_CHANNELS) - 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticBaseline:
        """X (N, 11, H, W), y (N, H, W); fewer than two classes kept -> ValueError."""
        from sklearn.linear_model import LogisticRegression

        Z = self._design(X)
        yy = np.asarray(y).reshape(-1)
        keep = (yy != -1) & (np.asarray(X)[:, -1].reshape(-1) > 0)
        Z, yy = Z[keep], yy[keep].astype(int)
        if len(np.unique(yy)) < 2:
            raise ValueError("logistic baseline needs both classes among the kept pixels")
        if len(yy) > self.max_pixels:  # uniform random subsample: the class prior stays that of the data
            rng = np.random.default_rng(self.seed)
            sel = np.sort(rng.choice(len(yy), size=self.max_pixels, replace=False))
            Z, yy = Z[sel], yy[sel]
            if len(np.unique(yy)) < 2:
                raise ValueError("logistic baseline needs both classes among the kept pixels")
        self._mu, self._sd = Z.mean(0), Z.std(0) + 1e-6
        self._model = LogisticRegression(C=self.C, max_iter=500, random_state=self.seed)
        self._model.fit((Z - self._mu) / self._sd, yy)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """float32 (N, H, W) in [0, 1], 0 where valid == 0; before fit -> RuntimeError."""
        if self._model is None:
            raise RuntimeError("LogisticBaseline.predict_proba called before fit")
        X = np.asarray(X)
        p = self._model.predict_proba((self._design(X) - self._mu) / self._sd)[:, 1]
        p = p.reshape(X.shape[0], X.shape[2], X.shape[3]).astype(np.float32)
        return np.where(X[:, -1] > 0, p, 0.0).astype(np.float32)
