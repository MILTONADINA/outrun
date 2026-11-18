"""Synthetic moving-Gaussian hazards for tests and the skeleton app (specs/02-grid-risk.md, RISK-11, RISK-12)."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from outrun.core.grid import GridWindow
from outrun.core.policy import HazardPolicy
from outrun.core.risk import Hazard, LayerKind, RiskLayer, RiskStack
from outrun.core.scenario import Scenario

SYNTHETIC_STARTS: dict[Hazard, tuple[float, float]] = {
    Hazard.STORM: (4.0, 4.0),
    Hazard.TORNADO: (16.0, 4.0),
    Hazard.FLOOD: (4.0, 16.0),
    Hazard.WILDFIRE: (24.0, 24.0),
    Hazard.ICE: (16.0, 16.0),
    Hazard.SMOKE: (28.0, 8.0),
}


def moving_gaussian(
    n_slots: int = 12,
    shape: tuple[int, int] = (32, 32),
    start: tuple[float, float] = (8.0, 8.0),
    velocity: tuple[float, float] = (1.5, 1.0),
    sigma: float = 3.0,
    peak: float = 0.9,
) -> np.ndarray:
    """float32 ``(n_slots, ny, nx)`` Gaussian blob, centre ``start + s * velocity`` in (row, col) cells (RISK-11).

    ValueError for ``sigma <= 0``, ``peak`` outside (0, 1] or ``n_slots < 1``.
    """
    raise NotImplementedError


def synthetic_window(n: int = 32) -> GridWindow:
    """``GridWindow(x0=-150_000, y0=1_300_000, nx=n, ny=n, res=1000)`` (RISK-12)."""
    raise NotImplementedError


def synthetic_scenario(t0: datetime | None = None, n: int = 32) -> Scenario:
    """Scenario ``"synthetic"`` on ``synthetic_window(n)``, all hazards, none held out (RISK-12)."""
    raise NotImplementedError


def synthetic_layer(
    hazard: Hazard,
    window: GridWindow,
    t0: datetime,
    *,
    kind: LayerKind = "forecast",
    start: tuple[float, float] | None = None,
    velocity: tuple[float, float] = (1.5, 1.0),
    sigma: float = 3.0,
    peak: float = 0.9,
    n_slots: int = 12,
    horizon_min: float = 180,
    source: str | None = None,
) -> RiskLayer:
    """Moving-Gaussian ``RiskLayer`` on ``window`` (float16, issued t0, valid t0 + horizon) (RISK-12)."""
    raise NotImplementedError


class SyntheticAdapter:
    """A ``HazardAdapter`` built from moving Gaussians (RISK-12).

    predict: forecast Gaussian; baseline: predict slot 0 held (persistence); observed: start shifted (+1, +1).
    """

    def __init__(self, hazard: Hazard, policy: HazardPolicy | None = None) -> None:
        raise NotImplementedError

    def baseline(self, scn: Scenario) -> RiskLayer:
        raise NotImplementedError

    def predict(self, scn: Scenario) -> RiskLayer:
        raise NotImplementedError

    def observed(self, scn: Scenario) -> RiskLayer | None:
        raise NotImplementedError


def synthetic_stack(t0: datetime | None = None, n: int = 32) -> RiskStack:
    """``build_stack`` of the six SyntheticAdapters with source ``"model"`` on ``synthetic_scenario`` (RISK-12)."""
    raise NotImplementedError
