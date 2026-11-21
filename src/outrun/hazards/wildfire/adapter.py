"""Wildfire HazardAdapter and RiskLayer construction (specs/11-wildfire.md: FIRE-10, FIRE-11).

Implements the outrun.core.risk.HazardAdapter protocol (DESIGN 1.2). Policy values: configs/hazards.yaml.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import xarray as xr

from outrun.config import get_paths
from outrun.core.policy import load_policies
from outrun.core.risk import Hazard, RiskLayer
from outrun.hazards.base import REGISTRY

from .baselines import downwind_persistence
from .growth import hourly_growth, observed_from_mask
from .model import FireUNet, TemperatureScaler

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


N_SLOTS = 12
KINDS = ("forecast", "baseline", "observed")


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
    prob = np.asarray(prob)
    if prob.ndim != 3 or prob.shape[0] != N_SLOTS:
        raise ValueError(f"prob must be ({N_SLOTS}, ny, nx), got {prob.shape}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if issued.tzinfo is None:
        raise ValueError("issued must be timezone-aware (UTC)")
    values = np.clip(np.nan_to_num(prob.astype(np.float32), nan=0.0), 0.0, 1.0).astype(np.float16)
    coords: dict[str, np.ndarray] = {"slot": np.arange(N_SLOTS)}
    if x is not None:
        coords["x"] = np.asarray(x)
    if y is not None:
        coords["y"] = np.asarray(y)
    da = xr.DataArray(values, dims=("slot", "y", "x"), coords=coords)
    return RiskLayer(
        hazard=Hazard.WILDFIRE,
        prob=da,
        edge_overrides=None,
        kind=kind,  # type: ignore[arg-type]
        issued=issued,
        valid_until=valid_until if valid_until is not None else issued + timedelta(hours=24),
        source=source,
        held_out=held_out,
    )


def _predict_day1(model: torch.nn.Module, x_model: np.ndarray) -> np.ndarray:
    """sigmoid(model(x))[0, 0] on one (12, h, w) sample; edge-padded to a multiple of 2**levels and cropped back."""
    levels = int(getattr(model, "levels", 3))
    m = 2**levels
    h, w = x_model.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    xt = torch.from_numpy(np.asarray(x_model, dtype=np.float32))[None]
    if ph or pw:
        xt = torch.nn.functional.pad(xt, (0, pw, 0, ph), mode="replicate")
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(xt))[0, 0, :h, :w]
    return p.numpy().astype(np.float32)


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
        self._provider = provider
        self.model = model
        self.calibrator = calibrator
        self._policy = policy
        self.source_tag = source_tag
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None

    @property
    def policy(self) -> HazardPolicy:
        """block_at civ/sens 0.5, EV 0.7; speed_factor 1; w civ/sens 20, EV 5; hold (FIRE-10)."""
        if self._policy is None:
            self._policy = load_policies()[Hazard.WILDFIRE]
        return self._policy

    def _inputs(self, scn: Scenario) -> WildfireInputs:
        if self._provider is None:
            try:
                from .provider import real_provider  # WSTS/NDWS provider (T20)
            except ImportError as exc:
                raise RuntimeError(
                    "no wildfire provider given and the real provider is not available"
                ) from exc
            self._provider = real_provider()
        inputs = self._provider(scn)
        win = scn.grid_window()
        for given, want, name in ((inputs.x, win.x_centers(), "x"), (inputs.y, win.y_centers(), "y")):
            if np.shape(given) != np.shape(want) or not np.allclose(given, want, rtol=0, atol=1e-6):
                raise ValueError(f"provider {name} centres differ from the scenario grid window")
        return inputs

    def _load_model(self) -> nn.Module | None:
        """Lazy checkpoint load (architecture from weight shapes); picks up calibrator.json next to it."""
        if self.model is None and self.checkpoint is not None:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            self.model = FireUNet.from_state_dict(payload["model"]).eval()
            cal = self.checkpoint.with_name("calibrator.json")
            if self.calibrator is None and cal.exists():
                self.calibrator = TemperatureScaler.load(cal)
        return self.model

    def _layer(
        self, scn: Scenario, inputs: WildfireInputs, day1: np.ndarray, kind: str, source: str
    ) -> RiskLayer:
        grown = hourly_growth(inputs.day0_mask, day1, inputs.day0_time, scn.t0)
        return self._wrap(scn, inputs, inputs.regrid(grown), kind, source)

    def _wrap(
        self, scn: Scenario, inputs: WildfireInputs, prob: np.ndarray, kind: str, source: str
    ) -> RiskLayer:
        win = scn.grid_window()
        return to_risk_layer(
            prob, kind=kind, issued=scn.t0, source=source, held_out=inputs.held_out,
            x=win.x_centers(), y=win.y_centers(),
        )  # fmt: skip

    def baseline(self, scn: Scenario) -> RiskLayer:
        """Downwind-dilated persistence (FIRE-06) grown hourly (FIRE-09)."""
        inputs = self._inputs(scn)
        return self._layer(
            scn, inputs, downwind_persistence(inputs.x_raw), "baseline", "wildfire.persistence"
        )

    def predict(self, scn: Scenario) -> RiskLayer:
        """Calibrated U-Net day-1 P, hourly linear growth from the day-0 mask (FIRE-04, FIRE-09)."""
        model = self._load_model()
        if model is None:
            raise RuntimeError("WildfireAdapter has no model: pass model=... or checkpoint=...")
        inputs = self._inputs(scn)
        p1 = _predict_day1(model, inputs.x_model)
        if self.calibrator is not None:
            p1 = np.asarray(self.calibrator.transform(p1), dtype=np.float32)
        return self._layer(scn, inputs, p1, "forecast", self.source_tag)

    def observed(self, scn: Scenario) -> RiskLayer | None:
        """Next-day WSTS VIIRS mask (FIRE-11)."""
        inputs = self._inputs(scn)
        if inputs.day1_mask is None:
            return None
        return self._wrap(
            scn, inputs, inputs.regrid(observed_from_mask(inputs.day1_mask)), "observed", "wildfire.viirs"
        )


def default_adapter() -> WildfireAdapter:
    """Registered zero-argument factory (FIRE-10, RISK-13): checkpoint ``models/wildfire/best.pt``."""
    return WildfireAdapter(checkpoint=get_paths().models / "wildfire" / "best.pt")


REGISTRY.register(Hazard.WILDFIRE, default_adapter)
