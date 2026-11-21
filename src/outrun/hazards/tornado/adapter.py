"""``TornadoAdapter``: the tornado ``HazardAdapter`` (TORN-10). The only tornado module importing ``outrun.core``."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from outrun.config import get_paths
from outrun.core.policy import HazardPolicy, apply_beyond_horizon, load_hazard_config, load_policies
from outrun.core.risk import Hazard, LayerKind, RiskLayer
from outrun.core.scenario import Scenario
from outrun.hazards.base import REGISTRY

from .baselines import delta_v_rule
from .calibration import IsotonicCalibrator
from .features import N_CHANNELS, NORM_CLIP, build_chip_tensor, normalize
from .model import TornadoCNN, predict_proba
from .swath import combine_swaths, project_swath, resolve_motion

N_SLOTS = 12
SLOT_MINUTES = 15
N_VALID_SLOTS = 2  # 30 min native horizon / 15 min slots


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
        self._provider = provider
        self.model = model
        self.calibrator = calibrator
        self.norm_stats = norm_stats
        self._policy = policy
        self.source_tag = source_tag
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None

    def _inputs(self, scn: Scenario) -> TornadoInputs:
        if self._provider is None:
            try:
                from .provider import real_provider  # real NEXRAD/TorNet provider (T12)
            except ImportError as exc:
                raise RuntimeError(
                    "no tornado provider given and the real provider is not available"
                ) from exc
            self._provider = real_provider()
        return self._provider(scn)

    def _load_model(self) -> TornadoCNN | None:
        """Lazy checkpoint load; widths and channel count come from the state dict. Picks up norm_stats.npz and
        calibrator.json next to the checkpoint when not given."""
        if self.model is None and self.checkpoint is not None:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            state = payload["model"]
            convs = [k for k in state if k.startswith("features.") and k.endswith(".weight")]
            widths = tuple(int(state[k].shape[0]) for k in convs[1::2])
            model = TornadoCNN(
                in_channels=int(state[convs[0]].shape[1]),
                widths=widths,
                hidden=int(state["head.0.weight"].shape[0]),
            )
            model.load_state_dict(state)
            self.model = model.eval()
            stats = self.checkpoint.with_name("norm_stats.npz")
            if self.norm_stats is None and stats.exists():
                with np.load(stats) as z:
                    self.norm_stats = (z["mean"], z["std"])
            cal = self.checkpoint.with_name("calibrator.json")
            if self.calibrator is None and cal.exists():
                self.calibrator = IsotonicCalibrator.from_dict(json.loads(cal.read_text()))
        return self.model

    @property
    def policy(self) -> HazardPolicy:
        """``load_policies()[Hazard.TORNADO]`` unless one was passed."""
        if self._policy is None:
            self._policy = load_policies()[Hazard.TORNADO]
        return self._policy

    def _layer(
        self, scn: Scenario, kind: LayerKind, swaths: list[np.ndarray], inputs: TornadoInputs, source: str
    ) -> RiskLayer:
        """Max-combine the per-detection 2-slot swaths, extend to 12 slots with the drop rule, wrap as a layer."""
        cfg = load_hazard_config()["hazards"]["tornado"]
        issued = scn.t0
        valid_until = issued + timedelta(minutes=float(cfg["native_horizon_min"]))
        n_valid = int(math.ceil(float(cfg["native_horizon_min"]) / SLOT_MINUTES))
        ny, nx = len(inputs.y), len(inputs.x)
        valid = combine_swaths(swaths) if swaths else np.zeros((n_valid, ny, nx), np.float32)
        prob = apply_beyond_horizon(
            np.clip(valid, 0.0, 1.0),
            self.policy,
            t0=scn.t0,
            issued=issued,
            valid_until=valid_until,
            n_slots=N_SLOTS,
            slot_minutes=SLOT_MINUTES,
            linger_min=float(cfg.get("blocked_linger_min", 15)),
        )
        return self._wrap(prob, kind, issued, valid_until, source, inputs)

    def _wrap(
        self,
        prob: np.ndarray,
        kind: LayerKind,
        issued: datetime,
        valid_until: datetime,
        source: str,
        inputs: TornadoInputs,
    ) -> RiskLayer:
        da = xr.DataArray(
            np.asarray(prob).astype(np.float16),
            dims=("slot", "y", "x"),
            coords={"slot": np.arange(prob.shape[0]), "y": np.asarray(inputs.y), "x": np.asarray(inputs.x)},
        )
        return RiskLayer(
            hazard=Hazard.TORNADO,
            prob=da,
            edge_overrides=None,
            kind=kind,
            issued=issued,
            valid_until=valid_until,
            source=source,
            held_out=inputs.held_out,
        )

    def _swath(self, p: float, det: TornadoDetection, inputs: TornadoInputs) -> np.ndarray:
        storm = inputs.storm_motion(det.x_m, det.y_m) if inputs.storm_motion is not None else None
        wind = inputs.hrrr_wind(det.x_m, det.y_m) if inputs.hrrr_wind is not None else None
        u, v, _ = resolve_motion(storm, wind)
        return project_swath(p, det.x_m, det.y_m, (u, v), inputs.x, inputs.y, n_slots=N_VALID_SLOTS)

    def baseline(self, scn: Scenario) -> RiskLayer:
        """Delta-V rule per detection -> swath, max-combined; kind "baseline"."""
        inputs = self._inputs(scn)
        swaths = [self._swath(delta_v_rule(d.chip), d, inputs) for d in inputs.detections]
        return self._layer(scn, "baseline", swaths, inputs, "tornado.delta_v_rule")

    def predict(self, scn: Scenario) -> RiskLayer:
        """CNN -> calibrated P -> swath, max-combined; kind "forecast". RuntimeError without a model."""
        model = self._load_model()
        if model is None:
            raise RuntimeError("TornadoAdapter has no model: pass model=... or checkpoint=...")
        inputs = self._inputs(scn)
        swaths: list[np.ndarray] = []
        if inputs.detections:
            include_range = model.in_channels == N_CHANNELS
            X = np.stack(
                [
                    build_chip_tensor(d.chip, d.range_km, d.azimuth_deg, include_range)
                    for d in inputs.detections
                ]
            )
            if self.norm_stats is not None:
                X = normalize(X, self.norm_stats[0], self.norm_stats[1], clip=NORM_CLIP)
            p = predict_proba(model, torch.from_numpy(X)).numpy().astype(np.float64)
            if self.calibrator is not None:
                p = self.calibrator.transform(p)
            swaths = [self._swath(float(pi), d, inputs) for pi, d in zip(p, inputs.detections, strict=True)]
        return self._layer(scn, "forecast", swaths, inputs, self.source_tag)

    def observed(self, scn: Scenario) -> RiskLayer | None:
        """Rasterized SPC tracks or None; kind "observed"."""
        inputs = self._inputs(scn)
        if inputs.observed is None:
            return None
        cfg = load_hazard_config()["hazards"]["tornado"]
        valid_until = scn.t0 + timedelta(minutes=float(cfg["native_horizon_min"]))
        return self._wrap(inputs.observed, "observed", scn.t0, valid_until, "tornado.observed_spc", inputs)


def default_adapter() -> TornadoAdapter:
    """Registered zero-argument factory (TORN-10, RISK-13): ``checkpoint = get_paths().models/"tornado"/"best.pt"``."""
    return TornadoAdapter(checkpoint=get_paths().models / "tornado" / "best.pt")


REGISTRY.register(Hazard.TORNADO, default_adapter)
