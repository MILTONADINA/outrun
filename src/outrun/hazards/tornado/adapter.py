"""``TornadoAdapter``: the tornado ``HazardAdapter`` (TORN-10). The only tornado module importing ``outrun.core``."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from outrun.core.policy import HazardPolicy
from outrun.core.risk import Hazard, RiskLayer
from outrun.core.scenario import Scenario

from .calibration import IsotonicCalibrator
from .model import TornadoCNN


@dataclass(frozen=True)
class TornadoDetection:
    """One TorNet chip near the case: raw chip mapping, its range/azimuth axes, and its centre on GRID."""

    chip: Mapping[str, np.ndarray]
    range_km: np.ndarray
    azimuth_deg: np.ndarray
    center_range_km: float
    x_m: float
    y_m: float


@dataclass(frozen=True)
class TornadoInputs:
    """Inputs for one scenario. ``observed`` is ``(12, Y, X)`` rasterized SPC tracks (DATA-07) or None."""

    detections: Sequence[TornadoDetection]
    x: np.ndarray
    y: np.ndarray
    storm_motion: Callable[[float, float], tuple[float, float] | None] | None = None
    hrrr_wind: Callable[[float, float], tuple[float, float] | None] | None = None
    observed: np.ndarray | None = None
    held_out: bool = False


class TornadoAdapter:
    """Tornado risk layers: detection P projected 30 min along the storm motion; drop after (TORN-10); slots after the horizon via ``outrun.core.policy.apply_beyond_horizon`` (HORIZON-02).

    Reads only ``scn.t0`` from the scenario; all data comes from ``provider(scn)``.
    """

    hazard: Hazard = Hazard.TORNADO

    def __init__(
        self,
        provider: Callable[[Scenario], TornadoInputs] | None = None,
        model: TornadoCNN | None = None,
        calibrator: IsotonicCalibrator | None = None,
        norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
        policy: HazardPolicy | None = None,
        source_tag: str = "tornado.cnn@dev",
        *,
        checkpoint: str | Path | None = None,
    ) -> None:
        """All optional; loads nothing (zero-argument factory, RISK-13). provider None -> real provider, lazily.
        model None + checkpoint -> lazy load on first predict; neither -> predict raises RuntimeError."""
        raise NotImplementedError

    @property
    def policy(self) -> HazardPolicy:
        """``load_policies()[Hazard.TORNADO]`` unless one was passed."""
        raise NotImplementedError

    def baseline(self, scn: Scenario) -> RiskLayer:
        """Delta-V rule per detection -> swath, max-combined; kind "baseline"."""
        raise NotImplementedError

    def predict(self, scn: Scenario) -> RiskLayer:
        """CNN -> calibrated P -> swath, max-combined; kind "forecast". RuntimeError without a model."""
        raise NotImplementedError

    def observed(self, scn: Scenario) -> RiskLayer | None:
        """Rasterized SPC tracks or None; kind "observed"."""
        raise NotImplementedError


def default_adapter() -> TornadoAdapter:
    """Registered zero-argument factory (TORN-10, RISK-13): ``checkpoint = get_paths().models/"tornado"/"best.pt"``."""
    raise NotImplementedError
