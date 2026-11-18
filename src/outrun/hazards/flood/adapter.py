"""Flood HazardAdapter and RiskLayer construction (specs/10-flood.md: FLOOD-10, FLOOD-11).

Implements the outrun.core.risk.HazardAdapter protocol (DESIGN 1.2). Policy values: configs/hazards.yaml.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from outrun.core.risk import Hazard, RiskLayer

from .dataset import FloodReport

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
    raise NotImplementedError


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
        raise NotImplementedError

    @property
    def policy(self) -> HazardPolicy:
        """block_at civ/sens 0.4, EV 0.7; speed_factor 1 - 0.4p; w civ/sens 10, EV 3; hold (FLOOD-10)."""
        raise NotImplementedError

    def baseline(self, scn: Scenario) -> RiskLayer:
        """rule_baseline_from_features held over 12 slots, regridded, window coords; crossing overrides with a graph
        (FLOOD-06, FLOOD-08, FLOOD-10); source "flood.ffg..."."""
        raise NotImplementedError

    def predict(self, scn: Scenario) -> RiskLayer:
        """sigmoid(model(features)) -> calibrator -> 12 slots -> regrid; crossing overrides from
        ``EdgeRiskSampler.from_window(graph, window).sample`` on this layer (FLOOD-04, FLOOD-08, FLOOD-10).
        RuntimeError without a model."""
        raise NotImplementedError

    def observed(self, scn: Scenario) -> RiskLayer | None:
        """observed_from_reports(reports, x, y, scn.t0) (FLOOD-11), or None when reports is None."""
        raise NotImplementedError


def default_adapter() -> FloodAdapter:
    """Registered zero-argument factory (FLOOD-10, RISK-13): checkpoint ``models/flood/best.pt``; graph built lazily
    with ``build_graph(bbox=scn.bbox, cache_dir=get_paths().proc / "graphs")``."""
    raise NotImplementedError
