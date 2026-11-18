"""Ice risk layer and HazardAdapter: slots (hold), bridge prior, speed factor, policy (ICE-07, ICE-08)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    raise NotImplementedError


def bridge_overrides(
    edge_probs: Mapping[int, np.ndarray], is_bridge: Mapping[int, bool], multiplier: float = 1.5
) -> dict[int, np.ndarray]:
    """ICE-07. Bridge prior: for bridge edges only, p' = min(1, multiplier * p) per slot. Inputs not mutated."""
    raise NotImplementedError


def ice_speed_factor(p: float | np.ndarray) -> float | np.ndarray:
    """ICE-07. clip(1 - 0.5 p, 1/3, 1), same as core.policy.make_speed_factor(0.5). Slope from configs/hazards.yaml."""
    raise NotImplementedError


def ice_policy(cfg: Mapping[str, Any] | None = None) -> HazardPolicy:
    """ICE-08. HazardPolicy from hazards.ice in configs/hazards.yaml (or ``cfg``, the same mapping shape).

    block_at None for every Profile; exposure civilian 3, sensitive 3, ev 1; speed_factor = ice_speed_factor;
    beyond_horizon "hold"; decay_tau_min None.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
        raise NotImplementedError

    @property
    def hazard(self) -> Hazard:
        from outrun.core.risk import Hazard

        return Hazard.ICE

    @property
    def policy(self) -> HazardPolicy:
        return ice_policy()

    def baseline(self, scn: Scenario) -> RiskLayer:
        """steps_to_slots(physics_rule per step) regridded, window coords; kind "baseline", source "ice.rule...",
        no edge overrides."""
        raise NotImplementedError

    def predict(self, scn: Scenario) -> RiskLayer:
        """steps_to_slots(calibrator(cell_probabilities(model, x_steps))) regridded; kind "forecast"; with a graph,
        bridge_overrides of ``EdgeRiskSampler.from_window(graph, window).sample`` on this layer. RuntimeError
        without a model."""
        raise NotImplementedError

    def observed(self, scn: Scenario) -> RiskLayer | None:
        """Labelled Storm Events zones on GRID (kind "observed"), or None without coverage."""
        raise NotImplementedError


def default_adapter() -> IceAdapter:
    """Registered zero-argument factory (ICE-08, RISK-13): checkpoint ``models/ice/best.pt``; graph built lazily with
    ``build_graph(bbox=scn.bbox, cache_dir=get_paths().proc / "graphs")``."""
    raise NotImplementedError
