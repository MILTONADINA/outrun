"""Ice risk layer and HazardAdapter: slots (hold), bridge prior, speed factor, policy (ICE-07, ICE-08)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from outrun.config import get_paths
from outrun.core.policy import HazardPolicy, load_policies, make_speed_factor
from outrun.core.risk import LAYER_KINDS, Hazard, RiskLayer
from outrun.hazards.base import REGISTRY

if TYPE_CHECKING:  # core types live in outrun.core (DESIGN 1.2); never redefined here
    from torch import nn

    from outrun.core.scenario import Scenario
    from outrun.roads.graph import RoadGraph


@dataclass(frozen=True)
class IceInputs:
    """Provider output (ICE-08): ``x_steps`` (n_steps, 20, h, w) ICE-03 inputs per 3-h step, ``fields_steps`` one
    ICE-05 field mapping per step, ``truth`` (12, Y, X) on GRID or None, GRID centres ``x``/``y``, ``regrid``."""

    x_steps: np.ndarray
    fields_steps: Sequence[Mapping[str, np.ndarray]]
    truth: np.ndarray | None
    x: np.ndarray
    y: np.ndarray
    regrid: Callable[[np.ndarray], np.ndarray]
    held_out: bool = False


def steps_to_slots(
    step_probs: np.ndarray, step_hours: float = 3.0, n_slots: int = 12, slot_minutes: int = 15
) -> np.ndarray:
    """ICE-07. Map 3-h model steps (n_steps, Y, X) to routing slots (n_slots, Y, X), holding the last step.

    Slot s uses step min(floor(s * slot_minutes / (60 * step_hours)), n_steps - 1). n_steps == 0 -> ValueError.
    """
    step_probs = np.asarray(step_probs)
    n_steps = step_probs.shape[0]
    if n_steps == 0:
        raise ValueError("steps_to_slots needs at least one step")
    idx = np.minimum(
        np.floor(np.arange(n_slots) * slot_minutes / (60.0 * step_hours)).astype(int), n_steps - 1
    )
    return step_probs[idx]


def bridge_overrides(
    edge_probs: Mapping[int, np.ndarray], is_bridge: Mapping[int, bool], multiplier: float = 1.5
) -> dict[int, np.ndarray]:
    """ICE-07. Bridge prior: for bridge edges only, p' = min(1, multiplier * p) per slot. Inputs not mutated."""
    return {
        int(e): np.minimum(1.0, multiplier * np.asarray(p, dtype=np.float64))
        for e, p in edge_probs.items()
        if is_bridge.get(e, False)
    }


def _ice_cfg(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if cfg is None:
        from outrun.core.policy import load_hazard_config

        cfg = load_hazard_config()
    for key in ("hazards", "ice"):
        if key in cfg:
            cfg = cfg[key]
    return cfg


def ice_speed_factor(p: float | np.ndarray) -> float | np.ndarray:
    """ICE-07. clip(1 - 0.5 p, 1/3, 1), same as core.policy.make_speed_factor(0.5). Slope from configs/hazards.yaml."""
    return make_speed_factor(float(_ice_cfg(None)["speed_factor_slope"]))(p)


def ice_policy(cfg: Mapping[str, Any] | None = None) -> HazardPolicy:
    """ICE-08. HazardPolicy from hazards.ice in configs/hazards.yaml (or ``cfg``, the same mapping shape).

    block_at None for every Profile; exposure civilian 3, sensitive 3, ev 1; speed_factor = ice_speed_factor;
    beyond_horizon "hold"; decay_tau_min None.
    """
    if cfg is None:
        return load_policies()[Hazard.ICE]
    from outrun.core.policy import Profile

    h = _ice_cfg(cfg)
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


def make_ice_layer(
    prob_slots: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    kind: str,
    issued: datetime,
    valid_until: datetime,
    source: str,
    held_out: bool,
    edge_overrides: dict[int, np.ndarray] | None = None,
) -> RiskLayer:
    """ICE-08. Wrap (slot, y, x) probabilities as an ICE RiskLayer (float16 DataArray, clipped to [0, 1]).

    Bad kind, shape mismatch or valid_until <= issued -> ValueError.
    """
    if kind not in LAYER_KINDS:
        raise ValueError(f"kind must be one of {LAYER_KINDS}, got {kind!r}")
    prob_slots = np.asarray(prob_slots)
    if prob_slots.ndim != 3 or prob_slots.shape[1:] != (len(y), len(x)):
        raise ValueError(f"prob_slots {prob_slots.shape} does not match (slot, {len(y)}, {len(x)})")
    if valid_until <= issued:
        raise ValueError("valid_until must be after issued")
    clean = np.clip(np.nan_to_num(prob_slots.astype(np.float32), nan=0.0), 0.0, 1.0).astype(np.float16)
    da = xr.DataArray(clean, dims=("slot", "y", "x"), coords={"x": np.asarray(x), "y": np.asarray(y)})
    return RiskLayer(
        hazard=Hazard.ICE,
        prob=da,
        edge_overrides=edge_overrides,
        kind=kind,
        issued=issued,
        valid_until=valid_until,
        source=source,
        held_out=held_out,
    )


class IceAdapter:
    """ICE-08. HazardAdapter for icy roads (MIL-CNN forecast, physics-rule baseline, labelled-zone truth)."""

    def __init__(
        self,
        provider: Callable[[Scenario], IceInputs] | None = None,
        model: nn.Module | None = None,
        calibrator: Any | None = None,
        source_tag: str = "ice.milcnn@dev",
        *,
        graph: RoadGraph | Callable[[Scenario], RoadGraph] | None = None,
        checkpoint: str | Path | None = None,
    ) -> None:
        """All optional; loads nothing (zero-argument factory, RISK-13). ``graph`` keys bridge ``edge_overrides`` by
        CSR edge id (None -> ``edge_overrides=None``). model None + checkpoint -> lazy load; neither -> predict
        raises RuntimeError."""
        self._provider = provider
        self.model = model
        self.calibrator = calibrator
        self.source_tag = source_tag
        self.graph = graph
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None

    @property
    def hazard(self) -> Hazard:
        return Hazard.ICE

    @property
    def policy(self) -> HazardPolicy:
        return ice_policy()

    def _inputs(self, scn: Scenario) -> IceInputs:
        if self._provider is None:
            try:
                from .provider import real_provider  # HRRR f01 provider
            except ImportError as exc:
                raise RuntimeError(
                    "no ice provider given and the real HRRR provider is not available"
                ) from exc
            self._provider = real_provider()
        inputs = self._provider(scn)
        win = scn.grid_window()
        for given, want, name in ((inputs.x, win.x_centers(), "x"), (inputs.y, win.y_centers(), "y")):
            if np.shape(given) != np.shape(want) or not np.allclose(given, want, rtol=0, atol=1e-6):
                raise ValueError(f"provider {name} centres differ from the scenario grid window")
        return inputs

    def _load_model(self) -> nn.Module | None:
        """Lazy checkpoint load (``ZoneLogitNet`` or bare ``IceMILNet`` weights) and the calibrator next to it."""
        if self.model is None and self.checkpoint is not None:
            import torch

            from outrun.hazards.ice.model import IceMILNet, ZoneCalibrator

            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            state = {k.removeprefix("net."): v for k, v in payload["model"].items()}
            model = IceMILNet(
                in_channels=state["enc.0.0.weight"].shape[1], base=state["head.weight"].shape[1]
            )
            model.load_state_dict(state)
            self.model = model.eval()
            cal = self.checkpoint.with_name("calibrator.pkl")
            if self.calibrator is None and cal.exists():
                self.calibrator = ZoneCalibrator.load(cal)
        return self.model

    def _layer(
        self, scn: Scenario, inputs: IceInputs, steps: np.ndarray, kind: str, source: str, overrides=None
    ) -> RiskLayer:
        win = scn.grid_window()
        slots = np.asarray(inputs.regrid(steps_to_slots(steps)), dtype=np.float32)
        return make_ice_layer(
            slots, win.x_centers(), win.y_centers(), kind=kind, issued=scn.t0,
            valid_until=scn.t0 + timedelta(minutes=180), source=source, held_out=inputs.held_out,
            edge_overrides=overrides,
        )  # fmt: skip

    def baseline(self, scn: Scenario) -> RiskLayer:
        """steps_to_slots(physics_rule per step) regridded, window coords; kind "baseline", source "ice.rule...",
        no edge overrides."""
        from outrun.hazards.ice.baselines import physics_rule

        inputs = self._inputs(scn)
        steps = np.stack([physics_rule(f).astype(np.float32) for f in inputs.fields_steps])
        return self._layer(scn, inputs, steps, "baseline", "ice.rule")

    def predict(self, scn: Scenario) -> RiskLayer:
        """steps_to_slots(calibrator(cell_probabilities(model, x_steps))) regridded; kind "forecast"; with a graph,
        bridge_overrides of ``EdgeRiskSampler.from_window(graph, window).sample`` on this layer. RuntimeError
        without a model."""
        import torch

        from outrun.hazards.ice.dataset import normalize_bag_input
        from outrun.hazards.ice.model import cell_probabilities

        model = self._load_model()
        if model is None:
            raise RuntimeError("IceAdapter has no model: pass model=... or checkpoint=...")
        inputs = self._inputs(scn)
        x = torch.from_numpy(normalize_bag_input(inputs.x_steps))
        p = cell_probabilities(model, x.to(next(model.parameters()).device))
        if self.calibrator is not None:
            p = np.asarray(self.calibrator.transform(p), dtype=np.float32)
        layer = self._layer(scn, inputs, p, "forecast", self.source_tag)
        graph = self.graph(scn) if callable(self.graph) else self.graph
        if graph is None:
            return layer
        from outrun.routing.cost import EdgeRiskSampler

        edge_probs = EdgeRiskSampler.from_window(graph, scn.grid_window()).sample(
            np.asarray(layer.prob.values, dtype=np.float32)[None]
        )
        bridges = np.flatnonzero(graph.bridge).tolist()
        overrides = bridge_overrides({e: edge_probs[e, 0] for e in bridges}, {e: True for e in bridges})
        return self._layer(scn, inputs, p, "forecast", self.source_tag, overrides)

    def observed(self, scn: Scenario) -> RiskLayer | None:
        """Labelled Storm Events zones on GRID (kind "observed"), or None without coverage."""
        inputs = self._inputs(scn)
        if inputs.truth is None:
            return None
        win = scn.grid_window()
        return make_ice_layer(
            np.asarray(inputs.truth, dtype=np.float32), win.x_centers(), win.y_centers(), kind="observed",
            issued=scn.t0, valid_until=scn.t0 + timedelta(minutes=180), source="ice.stormevents",
            held_out=inputs.held_out,
        )  # fmt: skip


def default_adapter() -> IceAdapter:
    """Registered zero-argument factory (ICE-08, RISK-13): checkpoint ``models/ice/best.pt``; graph built lazily with
    ``build_graph(bbox=scn.bbox, cache_dir=get_paths().proc / "graphs")``."""

    def graph(scn: Scenario) -> RoadGraph:
        from outrun.roads.graph import build_graph

        return build_graph(bbox=scn.bbox, cache_dir=get_paths().proc / "graphs")

    return IceAdapter(checkpoint=get_paths().models / "ice" / "best.pt", graph=graph)


REGISTRY.register(Hazard.ICE, default_adapter)
