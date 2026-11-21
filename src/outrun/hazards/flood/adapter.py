"""Flood HazardAdapter and RiskLayer construction (specs/10-flood.md: FLOOD-10, FLOOD-11).

Implements the outrun.core.risk.HazardAdapter protocol (DESIGN 1.2). Policy values: configs/hazards.yaml.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from outrun.config import get_paths
from outrun.core.policy import load_policies
from outrun.core.risk import Hazard, RiskLayer
from outrun.hazards.base import REGISTRY

from .baselines import rule_baseline_from_features
from .crossings import CrossingKind, crossing_overrides
from .dataset import FloodReport, observed_from_reports, select_channels

N_SLOTS = 12
HORIZON = timedelta(hours=6)
KINDS = ("forecast", "baseline", "observed")

if TYPE_CHECKING:
    from torch import nn

    from outrun.core.policy import HazardPolicy
    from outrun.core.scenario import Scenario
    from outrun.roads.graph import RoadGraph


@dataclass(frozen=True)
class FloodInputs:
    """Provider output for one scenario (FLOOD-10). ``features`` (11, h, w) FLOOD-01 stack on the model grid;
    ``reports`` truth or None; ``x``/``y`` GRID centres of ``scn.grid_window()``; ``regrid`` (k, h, w) -> (k, Y, X)."""

    features: np.ndarray
    reports: Sequence[FloodReport] | None
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
    edge_overrides: dict[int, np.ndarray] | None = None,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
) -> RiskLayer:
    """(12, ny, nx) -> RiskLayer(hazard=FLOOD), prob float16 DataArray ("slot","y","x") clipped to [0,1], NaN -> 0.

    valid_until defaults to issued + 6 h. Bad shape / kind / naive issued -> ValueError (FLOOD-10).
    """
    prob = np.asarray(prob)
    if prob.ndim != 3 or prob.shape[0] != N_SLOTS:
        raise ValueError(f"prob must be ({N_SLOTS}, ny, nx), got {prob.shape}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if issued.tzinfo is None:
        raise ValueError("issued must be timezone-aware")
    clean = np.clip(np.nan_to_num(prob.astype(np.float32), nan=0.0), 0.0, 1.0).astype(np.float16)
    coords: dict[str, Any] = {"slot": np.arange(N_SLOTS)}
    if y is not None:
        coords["y"] = np.asarray(y)
    if x is not None:
        coords["x"] = np.asarray(x)
    da = xr.DataArray(clean, dims=("slot", "y", "x"), coords=coords)
    return RiskLayer(
        hazard=Hazard.FLOOD,
        prob=da,
        edge_overrides=edge_overrides,
        kind=kind,
        issued=issued,
        valid_until=issued + HORIZON if valid_until is None else valid_until,
        source=source,
        held_out=held_out,
    )


class FloodAdapter:
    """Flash-flood adapter: U-Net forecast, FFG-ratio rule baseline, Storm Events observed (FLOOD-10, FLOOD-11)."""

    hazard: Hazard = Hazard.FLOOD

    def __init__(
        self,
        provider: Callable[[Scenario], FloodInputs] | None = None,
        model: nn.Module | None = None,
        calibrator: Any | None = None,
        policy: HazardPolicy | None = None,
        source_tag: str = "flood.unet@dev",
        *,
        graph: RoadGraph | Callable[[Scenario], RoadGraph] | None = None,
        checkpoint: str | Path | None = None,
        use_flash: bool = True,
    ) -> None:
        """All optional; loads nothing (zero-argument factory, RISK-13). ``graph`` keys crossing ``edge_overrides`` by
        CSR edge id (None -> ``edge_overrides=None``). model None + checkpoint -> lazy load; neither -> predict raises
        RuntimeError (FLOOD-10)."""
        self._provider = provider
        self.model = model
        self.calibrator = calibrator
        self._policy = policy
        self.source_tag = source_tag
        self.graph = graph
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.use_flash = use_flash

    @property
    def policy(self) -> HazardPolicy:
        """block_at civ/sens 0.4, EV 0.7; speed_factor 1 - 0.4p; w civ/sens 10, EV 3; hold (FLOOD-10)."""
        if self._policy is None:
            self._policy = load_policies()[Hazard.FLOOD]
        return self._policy

    # ------------------------------------------------------------------ helpers
    def _inputs(self, scn: Scenario) -> FloodInputs:
        if self._provider is None:
            raise RuntimeError(
                "no flood provider given and no real MRMS/FLASH scenario provider is available yet"
            )
        inputs = self._provider(scn)
        win = scn.grid_window()
        for got, want, name in ((inputs.x, win.x_centers(), "x"), (inputs.y, win.y_centers(), "y")):
            got = np.asarray(got)
            if got.shape != np.shape(want) or not np.allclose(got, want, atol=1e-6):
                raise ValueError(f"provider {name} differ from the scenario window cell centres")
        return inputs

    def _load_model(self) -> nn.Module | None:
        if self.model is None and self.checkpoint is not None:
            from outrun.hazards.base import load_checkpoint

            from .model import FloodUNet

            net = FloodUNet(in_channels=11 if self.use_flash else 8)
            load_checkpoint(self.checkpoint, net)
            self.model = net.eval()
            cal = self.checkpoint.with_name("calibrator.json")
            if self.calibrator is None and cal.exists():
                from outrun.core.calibrate import IsotonicCalibrator

                self.calibrator = IsotonicCalibrator.from_dict(json.loads(cal.read_text()))
        return self.model

    def _graph(self, scn: Scenario) -> RoadGraph | None:
        g = self.graph
        return g(scn) if callable(g) else g

    def _overrides(self, scn: Scenario, p_slots: np.ndarray) -> dict[int, np.ndarray] | None:
        """Crossing overrides keyed by CSR edge id of the routing graph (FLOOD-08, FLOOD-10)."""
        graph = self._graph(scn)
        if graph is None:
            return None
        from outrun.routing.cost import EdgeRiskSampler

        kinds: dict[int, CrossingKind] = {
            int(e): CrossingKind.LOW_WATER for e in np.flatnonzero(graph.crossing)
        }
        kinds.update({int(e): CrossingKind.BRIDGE for e in np.flatnonzero(graph.bridge)})  # bridge wins
        if not kinds:
            return {}
        sampler = EdgeRiskSampler.from_window(graph, scn.grid_window())
        sampled = sampler.sample(p_slots[None].astype(np.float32))[:, 0, :]  # (E, 12)
        return crossing_overrides({e: sampled[e] for e in kinds}, kinds)

    def _layer(
        self, scn: Scenario, inputs: FloodInputs, p2d: np.ndarray, kind: str, source: str
    ) -> RiskLayer:
        held = np.repeat(np.asarray(p2d, dtype=np.float32)[None], N_SLOTS, axis=0)  # (12, h, w)
        p_slots = np.asarray(inputs.regrid(held), dtype=np.float32)  # (12, Y, X)
        return to_risk_layer(
            p_slots,
            kind=kind,
            issued=scn.t0,
            source=source,
            held_out=inputs.held_out,
            valid_until=scn.t0 + HORIZON,
            edge_overrides=self._overrides(scn, p_slots),
            x=scn.grid_window().x_centers(),
            y=scn.grid_window().y_centers(),
        )

    # ------------------------------------------------------------------ protocol
    def baseline(self, scn: Scenario) -> RiskLayer:
        """rule_baseline_from_features held over 12 slots, regridded, window coords; crossing overrides with a graph
        (FLOOD-06, FLOOD-08, FLOOD-10); source "flood.ffg..."."""
        inputs = self._inputs(scn)
        return self._layer(
            scn, inputs, rule_baseline_from_features(inputs.features), "baseline", "flood.ffg_ratio_rule"
        )

    def predict(self, scn: Scenario) -> RiskLayer:
        """sigmoid(model(features)) -> calibrator -> 12 slots -> regrid; crossing overrides from
        ``EdgeRiskSampler.from_window(graph, window).sample`` on this layer (FLOOD-04, FLOOD-08, FLOOD-10).
        RuntimeError without a model."""
        import torch

        model = self._load_model()
        if model is None:
            raise RuntimeError("FloodAdapter has no model: pass model=... or checkpoint=...")
        inputs = self._inputs(scn)
        feats = np.asarray(inputs.features, dtype=np.float32)
        if not self.use_flash:
            feats = select_channels(feats, use_flash=False)
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(feats))[None]))[0, 0].numpy()
        if self.calibrator is not None:
            p = np.asarray(self.calibrator.transform(p), dtype=np.float32)
        return self._layer(scn, inputs, p, "forecast", self.source_tag)

    def observed(self, scn: Scenario) -> RiskLayer | None:
        """observed_from_reports(reports, x, y, scn.t0) (FLOOD-11), or None when reports is None."""
        inputs = self._inputs(scn)
        if inputs.reports is None:
            return None
        win = scn.grid_window()
        obs = observed_from_reports(inputs.reports, win.x_centers(), win.y_centers(), scn.t0)
        return to_risk_layer(
            obs,
            kind="observed",
            issued=scn.t0,
            source="flood.stormevents",
            held_out=inputs.held_out,
            valid_until=scn.t0 + HORIZON,
            x=win.x_centers(),
            y=win.y_centers(),
        )


def default_adapter() -> FloodAdapter:
    """Registered zero-argument factory (FLOOD-10, RISK-13): checkpoint ``models/flood/best.pt``; graph built lazily
    with ``build_graph(bbox=scn.bbox, cache_dir=get_paths().proc / "graphs")``."""

    def graph(scn: Scenario) -> RoadGraph:
        from outrun.roads.graph import build_graph

        return build_graph(bbox=scn.bbox, cache_dir=get_paths().proc / "graphs")

    return FloodAdapter(checkpoint=get_paths().models / "flood" / "best.pt", graph=graph)


REGISTRY.register(Hazard.FLOOD, default_adapter)
