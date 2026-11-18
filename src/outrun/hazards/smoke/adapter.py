"""Smoke gridded application, risk mapping and HazardAdapter (SMOKE-07, SMOKE-08)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # core types live in outrun.core (DESIGN 1.2); never redefined here
    from torch import nn

    from outrun.core.policy import HazardPolicy
    from outrun.core.risk import Hazard, RiskLayer
    from outrun.core.scenario import Scenario


@dataclass(frozen=True)
class SmokeInputs:
    """Provider output (SMOKE-08): per-cell ``patches`` (h*w, 8, 9, 9) and ``tabs`` (h*w, n_tabular) in row-major
    order, ``grid_shape`` (h, w), calibrated ``hrrr_pm`` (h, w), ``obs_pm`` (Y, X) on GRID or None, GRID centres
    ``x``/``y``, ``regrid`` (k, h, w) -> (k, Y, X)."""

    patches: np.ndarray
    tabs: np.ndarray
    grid_shape: tuple[int, int]
    hrrr_pm: np.ndarray
    obs_pm: np.ndarray | None
    x: np.ndarray
    y: np.ndarray
    regrid: Callable[[np.ndarray], np.ndarray]
    held_out: bool = False


def idw(
    station_xy: np.ndarray, values: np.ndarray, x: np.ndarray, y: np.ndarray, power: float = 2.0
) -> np.ndarray:
    """SMOKE-07. Inverse-distance weighting to (len(y), len(x)) cell centres; exact within 1 m of a station.

    NaN station values ignored; no valid station -> ValueError.
    """
    raise NotImplementedError


def pm_to_risk(pm: float | np.ndarray) -> float | np.ndarray:
    """SMOKE-07. clip((pm - 35.4) / (150.4 - 35.4), 0, 1); thresholds from configs/hazards.yaml."""
    raise NotImplementedError


def hold_to_slots(field: np.ndarray, n_slots: int = 12) -> np.ndarray:
    """SMOKE-07. Every routing slot equals the +6 h field (HORIZON-03 hold)."""
    raise NotImplementedError


def shelter_deprioritized(r: float | np.ndarray) -> bool | np.ndarray:
    """SMOKE-07. r > 0.5 (shelter_deprioritize_r)."""
    raise NotImplementedError


def smoke_policy(cfg: Mapping[str, Any] | None = None) -> HazardPolicy:
    """SMOKE-08. Exposure-only HazardPolicy from hazards.smoke in configs/hazards.yaml (or ``cfg``).

    block_at None for every Profile; exposure civilian 0.5, sensitive 5, ev 0; speed_factor == 1; hold.
    """
    raise NotImplementedError


def make_smoke_layer(
    pm_slots: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    kind: str,
    issued: datetime,
    valid_until: datetime,
    source: str,
    held_out: bool,
) -> RiskLayer:
    """SMOKE-08. SMOKE RiskLayer with prob = pm_to_risk(pm_slots) as float16 (slot, y, x); no edge overrides.

    Bad kind, shape mismatch or valid_until <= issued -> ValueError.
    """
    raise NotImplementedError


class SmokeAdapter:
    """SMOKE-08. HazardAdapter for smoke (CNN+MLP forecast, calibrated HRRR-Smoke baseline, AQS IDW truth)."""

    def __init__(
        self,
        provider: Callable[[Scenario], SmokeInputs] | None = None,
        model: nn.Module | None = None,
        source_tag: str = "smoke.cnnmlp@dev",
        *,
        checkpoint: str | Path | None = None,
    ) -> None:
        """All optional; loads nothing (zero-argument factory, RISK-13). model None + checkpoint -> lazy load;
        neither -> predict raises RuntimeError."""
        raise NotImplementedError

    @property
    def hazard(self) -> Hazard:
        from outrun.core.risk import Hazard

        return Hazard.SMOKE

    @property
    def policy(self) -> HazardPolicy:
        return smoke_policy()

    def baseline(self, scn: Scenario) -> RiskLayer:
        """Calibrated raw HRRR-Smoke layer, kind "baseline", source "smoke.hrrr..."."""
        raise NotImplementedError

    def predict(self, scn: Scenario) -> RiskLayer:
        """SmokeNet median on the 3 km grid (obs channel = IDW of monitors), kind "forecast", "smoke.cnnmlp..."."""
        raise NotImplementedError

    def observed(self, scn: Scenario) -> RiskLayer | None:
        """AQS IDW at t0 + 6 h, kind "observed", or None with no monitors in the bbox."""
        raise NotImplementedError


def default_adapter() -> SmokeAdapter:
    """Registered zero-argument factory (SMOKE-08, RISK-13): checkpoint ``models/smoke/best.pt``."""
    raise NotImplementedError
