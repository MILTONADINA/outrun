"""``StormAdapter``: the storm ``HazardAdapter`` (STORM-08). The only storm module that imports ``outrun.core``."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from outrun.config import get_paths
from outrun.core.policy import HazardPolicy, apply_beyond_horizon, load_hazard_config, load_policies
from outrun.core.risk import Hazard, LayerKind, RiskLayer
from outrun.core.scenario import Scenario
from outrun.hazards.base import REGISTRY

from .calibration import PerLeadCalibrator, frames_to_slots
from .model import RISK_THRESHOLD, RISK_THRESHOLD_INDEX, StormUNet

N_SLOTS = 12  # routing slots of 15 min


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
        self._provider = provider
        self.model = model
        self.calibrator = calibrator
        self._policy = policy
        self.source_tag = source_tag
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None

    @property
    def policy(self) -> HazardPolicy:
        """``load_policies()[Hazard.STORM]`` unless one was passed."""
        if self._policy is None:
            self._policy = load_policies()[Hazard.STORM]
        return self._policy

    # ------------------------------------------------------------------ helpers
    def _frames(self, scn: Scenario) -> StormFrames:
        if self._provider is None:
            # The real SEVIR/MRMS provider is built lazily so that constructing the adapter stays offline.
            try:
                from .provider import real_provider
            except ImportError as exc:
                raise RuntimeError(
                    "no storm data provider: pass provider=... (the SEVIR provider is not built)"
                ) from exc
            self._provider = real_provider()
        return self._provider(scn)

    def _load_model(self) -> StormUNet | None:
        """Lazy checkpoint load; shapes come from the state dict, so any ``base``/``levels``/channel count works."""
        if self.model is None and self.checkpoint is not None:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            state = payload["model"]
            base, in_channels = state["inc.0.weight"].shape[:2]
            levels = sum(1 for k in state if k.startswith("down.") and k.endswith(".0.weight"))
            model = StormUNet(in_channels=in_channels, base=base, levels=levels)
            model.load_state_dict(state)
            self.model = model.eval()
            cal_path = self.checkpoint.with_name("calibrator.json")
            if self.calibrator is None and cal_path.exists():
                self.calibrator = PerLeadCalibrator.from_dict(json.loads(cal_path.read_text()))
        return self.model

    def _layer(
        self, scn: Scenario, kind: LayerKind, slots: np.ndarray, frames: StormFrames, source: str
    ) -> RiskLayer:
        """Regrid the 4 in-horizon slots, extend to 12 slots with the policy's decay rule, wrap as a RiskLayer."""
        cfg = load_hazard_config().get("hazards", {}).get("storm", {})
        issued = scn.t0
        valid_until = issued + timedelta(minutes=float(cfg.get("native_horizon_min", 60)))
        zero_after = cfg.get("decay_zero_after_min", 120.0)
        prob = apply_beyond_horizon(
            np.clip(frames.regrid(slots), 0.0, 1.0),
            self.policy,
            t0=scn.t0,
            issued=issued,
            valid_until=valid_until,
            n_slots=N_SLOTS,
            zero_after_min=None if zero_after is None else float(zero_after),
        )
        da = xr.DataArray(
            prob.astype(np.float16),
            dims=("slot", "y", "x"),
            coords={"slot": np.arange(N_SLOTS), "y": frames.y, "x": frames.x},
        )
        return RiskLayer(
            hazard=Hazard.STORM,
            prob=da,
            edge_overrides=None,
            kind=kind,
            issued=issued,
            valid_until=valid_until,
            source=source,
            held_out=frames.held_out,
        )

    # ------------------------------------------------------------------ HazardAdapter
    def baseline(self, scn: Scenario) -> RiskLayer:
        """Persistence of the last-frame VIL >= 133 mask; kind "baseline"."""
        frames = self._frames(scn)
        mask = (frames.vil[-1] >= RISK_THRESHOLD).astype(np.float32)
        slots = np.repeat(mask[None], 4, axis=0)
        return self._layer(scn, "baseline", slots, frames, "storm.persistence")

    def predict(self, scn: Scenario) -> RiskLayer:
        """U-Net -> calibrated P(VIL >= 133) -> slots -> GRID; kind "forecast". RuntimeError without a model."""
        model = self._load_model()
        if model is None:
            raise RuntimeError("StormAdapter has no model: pass model=... or checkpoint=...")
        frames = self._frames(scn)
        channels = [frames.vil.astype(np.float32) / 255.0]
        if model.in_channels == 26:
            if frames.ir is None:
                raise ValueError("the 26-channel storm model needs IR107 frames but frames.ir is None")
            channels.append(frames.ir.astype(np.float32) / 255.0)
        x = torch.from_numpy(np.concatenate(channels, axis=0))[None]
        model.eval()
        with torch.no_grad():
            _, logits = model(x)
        p = torch.sigmoid(logits)[0, :, RISK_THRESHOLD_INDEX].numpy()  # (12, h, w)
        if self.calibrator is not None:
            p = self.calibrator.transform(p[None])[0]
        return self._layer(scn, "forecast", frames_to_slots(p), frames, self.source_tag)

    def observed(self, scn: Scenario) -> RiskLayer | None:
        """True future VIL >= 133 per slot, or None without truth; kind "observed"."""
        frames = self._frames(scn)
        if frames.future_vil is None:
            return None
        exceed = (frames.future_vil >= RISK_THRESHOLD).astype(np.float32)
        return self._layer(scn, "observed", frames_to_slots(exceed), frames, "storm.observed_sevir")


def default_adapter() -> StormAdapter:
    """Registered zero-argument factory (STORM-08, RISK-13): ``checkpoint = get_paths().models/"storm"/"best.pt"``."""
    return StormAdapter(checkpoint=get_paths().models / "storm" / "best.pt")


REGISTRY.register(Hazard.STORM, default_adapter)
