"""Wildfire HazardAdapter and RiskLayer construction (specs/11-wildfire.md: FIRE-10, FIRE-11).

Implements the outrun.core.risk.HazardAdapter protocol (DESIGN 1.2). Policy values: configs/hazards.yaml.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from outrun.core.risk import Hazard, RiskLayer

if TYPE_CHECKING:
    from torch import nn

    from outrun.core.policy import HazardPolicy
    from outrun.core.scenario import Scenario


@dataclass(frozen=True)
class WildfireInputs:
    """Provider output (FIRE-10): normalised ``x_model`` and raw ``x_raw`` (12, h, w), ``day0_mask`` (h, w),
    ``day0_time`` (aware UTC), ``day1_mask`` truth or None, GRID centres ``x``/``y``, ``regrid`` (k,h,w)->(k,Y,X)."""

    x_model: np.ndarray
    x_raw: np.ndarray
    day0_mask: np.ndarray
    day0_time: datetime
    day1_mask: np.ndarray | None
    x: np.ndarray
    y: np.ndarray
    regrid: Callable[[np.ndarray], np.ndarray]
    held_out: bool = False


def to_risk_layer(
    prob: np.ndarray,
    *,
    kind: str,
    issued: datetime,
    source: str,
    held_out: bool,
    valid_until: datetime | None = None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> RiskLayer:
    """(12, ny, nx) -> RiskLayer(hazard=WILDFIRE, edge_overrides=None), prob float16 DataArray ("slot","y","x")
    clipped to [0,1], NaN -> 0; valid_until defaults to issued + 24 h. Bad shape/kind/naive issued -> ValueError."""
    raise NotImplementedError


class WildfireAdapter:
    """Wildfire adapter: U-Net day-1 forecast grown hourly from day 0, downwind persistence baseline,
    next-day VIIRS mask observed (FIRE-09, FIRE-10, FIRE-11)."""

    hazard: Hazard = Hazard.WILDFIRE

    def __init__(
        self,
        provider: Callable[[Scenario], WildfireInputs] | None = None,
        model: nn.Module | None = None,
        calibrator: Any | None = None,
        policy: HazardPolicy | None = None,
        source_tag: str = "wildfire.unet@dev",
        *,
        checkpoint: str | Path | None = None,
    ) -> None:
        """All optional; loads nothing (zero-argument factory, RISK-13). model None + checkpoint -> lazy load;
        neither -> predict raises RuntimeError (FIRE-10)."""
        raise NotImplementedError

    @property
    def policy(self) -> HazardPolicy:
        """block_at civ/sens 0.5, EV 0.7; speed_factor 1; w civ/sens 20, EV 5; hold (FIRE-10)."""
        raise NotImplementedError

    def baseline(self, scn: Scenario) -> RiskLayer:
        """Downwind-dilated persistence (FIRE-06) grown hourly (FIRE-09)."""
        raise NotImplementedError

    def predict(self, scn: Scenario) -> RiskLayer:
        """Calibrated U-Net day-1 P, hourly linear growth from the day-0 mask (FIRE-04, FIRE-09)."""
        raise NotImplementedError

    def observed(self, scn: Scenario) -> RiskLayer | None:
        """Next-day WSTS VIIRS mask (FIRE-11)."""
        raise NotImplementedError


def default_adapter() -> WildfireAdapter:
    """Registered zero-argument factory (FIRE-10, RISK-13): checkpoint ``models/wildfire/best.pt``."""
    raise NotImplementedError
