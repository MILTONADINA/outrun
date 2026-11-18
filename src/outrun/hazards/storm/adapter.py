"""``StormAdapter``: the storm ``HazardAdapter`` (STORM-08). The only storm module that imports ``outrun.core``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from outrun.core.policy import HazardPolicy
from outrun.core.risk import Hazard, RiskLayer
from outrun.core.scenario import Scenario

from .calibration import PerLeadCalibrator
from .model import StormUNet


@dataclass(frozen=True)
class StormFrames:
    """Inputs for one scenario. ``vil``/``ir`` ``(13, h, w)`` uint8 (t-60..t), ``future_vil`` ``(12, h, w)``
    uint8 truth or None, GRID cell-centre coordinates ``x`` ``(X,)``/``y`` ``(Y,)`` in metres, ``regrid`` maps
    ``(k, h, w)`` -> ``(k, Y, X)``."""

    vil: np.ndarray
    ir: np.ndarray | None
    future_vil: np.ndarray | None
    x: np.ndarray
    y: np.ndarray
    regrid: Callable[[np.ndarray], np.ndarray]
    held_out: bool = False


class StormAdapter:
    """Storm risk layers: P(VIL >= 133), 4 in-horizon slots (60 min), decay after (STORM-08); slots after the horizon via ``outrun.core.policy.apply_beyond_horizon`` (HORIZON-01).

    Reads only ``scn.t0`` from the scenario; all data comes from ``provider(scn)``.
    """

    hazard: Hazard = Hazard.STORM

    def __init__(
        self,
        provider: Callable[[Scenario], StormFrames] | None = None,
        model: StormUNet | None = None,
        calibrator: PerLeadCalibrator | None = None,
        policy: HazardPolicy | None = None,
        source_tag: str = "storm.unet_vil_ir@dev",
        *,
        checkpoint: str | Path | None = None,
    ) -> None:
        """All optional; loads nothing (zero-argument factory, RISK-13). provider None -> real provider, lazily.
        model None + checkpoint -> lazy load on first predict; neither -> predict raises RuntimeError."""
        raise NotImplementedError

    @property
    def policy(self) -> HazardPolicy:
        """``load_policies()[Hazard.STORM]`` unless one was passed."""
        raise NotImplementedError

    def baseline(self, scn: Scenario) -> RiskLayer:
        """Persistence of the last-frame VIL >= 133 mask; kind "baseline"."""
        raise NotImplementedError

    def predict(self, scn: Scenario) -> RiskLayer:
        """U-Net -> calibrated P(VIL >= 133) -> slots -> GRID; kind "forecast". RuntimeError without a model."""
        raise NotImplementedError

    def observed(self, scn: Scenario) -> RiskLayer | None:
        """True future VIL >= 133 per slot, or None without truth; kind "observed"."""
        raise NotImplementedError


def default_adapter() -> StormAdapter:
    """Registered zero-argument factory (STORM-08, RISK-13): ``checkpoint = get_paths().models/"storm"/"best.pt"``."""
    raise NotImplementedError
