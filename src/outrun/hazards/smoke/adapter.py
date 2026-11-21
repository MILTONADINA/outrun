"""Smoke gridded application, risk mapping and HazardAdapter (SMOKE-07, SMOKE-08)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from outrun.config import get_paths
from outrun.core.policy import HazardPolicy, Profile, load_hazard_config, load_policies, make_speed_factor
from outrun.core.risk import Hazard, RiskLayer
from outrun.hazards.base import REGISTRY

from .model import to_pm

if TYPE_CHECKING:  # core types live in outrun.core (DESIGN 1.2); never redefined here
    from torch import nn

    from outrun.core.scenario import Scenario

N_SLOTS = 12
HORIZON = timedelta(hours=6)
KINDS = ("forecast", "baseline", "observed")


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
    xy = np.asarray(station_xy, dtype=np.float64).reshape(-1, 2)
    v = np.asarray(values, dtype=np.float64)
    ok = ~np.isnan(v)
    if not ok.any():
        raise ValueError("idw needs at least one station with a value")
    xy, v = xy[ok], v[ok]
    xx, yy = np.meshgrid(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    cells = np.stack([xx.ravel(), yy.ravel()], axis=1)
    out = np.empty(len(cells), dtype=np.float64)
    for lo in range(0, len(cells), 20000):  # chunked: (cells x stations) distance matrix
        c = cells[lo : lo + 20000]
        d = np.hypot(c[:, None, 0] - xy[None, :, 0], c[:, None, 1] - xy[None, :, 1])
        near = d.min(axis=1) <= 1.0
        w = 1.0 / np.maximum(d, 1e-9) ** power
        res = (w * v[None, :]).sum(axis=1) / w.sum(axis=1)
        res[near] = v[d[near].argmin(axis=1)]
        out[lo : lo + 20000] = res
    return out.reshape(len(y), len(x)).astype(np.float32)


def _smoke_cfg() -> Mapping[str, Any]:
    return load_hazard_config()["hazards"]["smoke"]


def pm_to_risk(pm: float | np.ndarray) -> float | np.ndarray:
    """SMOKE-07. clip((pm - 35.4) / (150.4 - 35.4), 0, 1); thresholds from configs/hazards.yaml."""
    cfg = _smoke_cfg()
    lo, hi = float(cfg["pm25_low"]), float(cfg["pm25_high"])
    r = np.clip((np.asarray(pm, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return float(r) if r.ndim == 0 else r


def hold_to_slots(field: np.ndarray, n_slots: int = N_SLOTS) -> np.ndarray:
    """SMOKE-07. Every routing slot equals the +6 h field (HORIZON-03 hold)."""
    return np.repeat(np.asarray(field)[None], n_slots, axis=0)


def shelter_deprioritized(r: float | np.ndarray) -> bool | np.ndarray:
    """SMOKE-07. r > 0.5 (shelter_deprioritize_r)."""
    t = float(_smoke_cfg()["shelter_deprioritize_r"])
    out = np.asarray(r) > t
    return bool(out) if out.ndim == 0 else out


def smoke_policy(cfg: Mapping[str, Any] | None = None) -> HazardPolicy:
    """SMOKE-08. Exposure-only HazardPolicy from hazards.smoke in configs/hazards.yaml (or ``cfg``).

    block_at None for every Profile; exposure civilian 0.5, sensitive 5, ev 0; speed_factor == 1; hold.
    ``cfg`` may be the whole hazards file, its ``hazards`` mapping, or the ``smoke`` section itself.
    """
    if cfg is None:
        return load_policies()[Hazard.SMOKE]
    h = cfg
    if "hazards" in h:
        h = h["hazards"]
    if "smoke" in h:
        h = h["smoke"]
    tau = h.get("decay_tau_min")
    return HazardPolicy(
        block_at={
            p: (None if h["block_at"][p.value] is None else float(h["block_at"][p.value])) for p in Profile
        },
        speed_factor=make_speed_factor(float(h["speed_factor_slope"])),
        exposure_weight={p: float(h["exposure_weight"][p.value]) for p in Profile},
        beyond_horizon=h["beyond_horizon"],
        decay_tau_min=None if tau is None else float(tau),
    )


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
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    pm = np.asarray(pm_slots, dtype=np.float64)
    x, y = np.asarray(x), np.asarray(y)
    if pm.ndim != 3 or pm.shape[1:] != (len(y), len(x)):
        raise ValueError(f"pm_slots must be (slots, {len(y)}, {len(x)}), got {pm.shape}")
    risk = pm_to_risk(np.nan_to_num(pm, nan=0.0)).astype(np.float16)
    da = xr.DataArray(risk, dims=("slot", "y", "x"), coords={"slot": np.arange(pm.shape[0]), "y": y, "x": x})
    return RiskLayer(
        hazard=Hazard.SMOKE,
        prob=da,
        edge_overrides=None,
        kind=kind,  # type: ignore[arg-type]
        issued=issued,
        valid_until=valid_until,
        source=source,
        held_out=held_out,
    )


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
        self._provider = provider
        self.model = model
        self.source_tag = source_tag
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None

    @property
    def hazard(self) -> Hazard:
        return Hazard.SMOKE

    @property
    def policy(self) -> HazardPolicy:
        return smoke_policy()

    # ------------------------------------------------------------------ helpers
    def _inputs(self, scn: Scenario) -> SmokeInputs:
        provider = self._provider
        if provider is None:
            from .provider import real_provider  # lazy: needs the HRRR store, AQS and HMS on disk

            provider = self._provider = real_provider()
        inputs = provider(scn)
        win = scn.grid_window()
        for got, want, name in ((inputs.x, win.x_centers(), "x"), (inputs.y, win.y_centers(), "y")):
            got = np.asarray(got)
            if got.shape != np.shape(want) or not np.allclose(got, want, atol=1e-6):
                raise ValueError(f"provider {name} differ from the scenario window cell centres")
        return inputs

    def _load_model(self) -> nn.Module | None:
        if self.model is None and self.checkpoint is not None:
            from outrun.hazards.base import load_checkpoint

            from .model import SmokeNet

            state = _read_state(self.checkpoint)
            net = SmokeNet(n_tabular=int(state["model"]["tab_mean"].shape[0]))
            load_checkpoint(self.checkpoint, net)
            self.model = net.eval()
        return self.model

    def _layer(
        self, scn: Scenario, inputs: SmokeInputs, pm2d: np.ndarray, kind: str, source: str
    ) -> RiskLayer:
        slots = np.asarray(inputs.regrid(hold_to_slots(np.asarray(pm2d, dtype=np.float32))), dtype=np.float32)
        win = scn.grid_window()
        return make_smoke_layer(
            slots,
            win.x_centers(),
            win.y_centers(),
            kind=kind,
            issued=scn.t0,
            valid_until=scn.t0 + HORIZON,
            source=source,
            held_out=inputs.held_out,
        )

    # ------------------------------------------------------------------ protocol
    def baseline(self, scn: Scenario) -> RiskLayer:
        """Calibrated raw HRRR-Smoke layer, kind "baseline", source "smoke.hrrr..."."""
        inputs = self._inputs(scn)
        return self._layer(scn, inputs, inputs.hrrr_pm, "baseline", "smoke.hrrr.calibrated")

    def predict(self, scn: Scenario) -> RiskLayer:
        """SmokeNet median on the model grid (obs channel = IDW of monitors), kind "forecast", "smoke.cnnmlp..."."""
        model = self._load_model()
        if model is None:
            raise RuntimeError("SmokeAdapter has no model and no checkpoint")
        import torch

        inputs = self._inputs(scn)
        model.eval()
        out = []
        with torch.no_grad():
            for lo in range(0, len(inputs.patches), 8192):
                p = torch.as_tensor(inputs.patches[lo : lo + 8192], dtype=torch.float32)
                t = torch.as_tensor(inputs.tabs[lo : lo + 8192], dtype=torch.float32)
                out.append(model(p, t).cpu().numpy())
        pm = to_pm(np.concatenate(out))[:, 0].reshape(inputs.grid_shape)
        return self._layer(scn, inputs, pm, "forecast", self.source_tag)

    def observed(self, scn: Scenario) -> RiskLayer | None:
        """AQS IDW at t0 + 6 h, kind "observed", or None with no monitors in the bbox."""
        inputs = self._inputs(scn)
        if inputs.obs_pm is None:
            return None
        win = scn.grid_window()
        slots = hold_to_slots(np.asarray(inputs.obs_pm, dtype=np.float32))
        return make_smoke_layer(
            slots,
            win.x_centers(),
            win.y_centers(),
            kind="observed",
            issued=scn.t0,
            valid_until=scn.t0 + HORIZON,
            source="smoke.aqs.idw",
            held_out=inputs.held_out,
        )


def _read_state(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def default_adapter() -> SmokeAdapter:
    """Registered zero-argument factory (SMOKE-08, RISK-13): checkpoint ``models/smoke/best.pt``."""
    return SmokeAdapter(checkpoint=get_paths().models / "smoke" / "best.pt")


REGISTRY.register(Hazard.SMOKE, default_adapter)
