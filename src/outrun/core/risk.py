"""Hazards, risk layers, the RiskStack and the adapter protocol (DESIGN 1.2; specs/02-grid-risk.md RISK-01..08).

The router and the app see only a ``RiskStack`` (+ ``HazardPolicy`` objects). Overlay PNGs are rendered from the
same ``stack.data`` array the router samples ("what you see is what routes", RISK-07).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
import xarray as xr

from outrun.core.grid import GridWindow

if TYPE_CHECKING:
    from outrun.core.policy import HazardPolicy
    from outrun.core.scenario import Scenario

SLOT_MINUTES = 15
N_SLOTS = 12
LayerKind = Literal["forecast", "baseline", "observed"]
LAYER_KINDS: tuple[str, ...] = ("forecast", "baseline", "observed")


class RiskSource(StrEnum):
    """``--risk-source`` switch (DESIGN 1.2, RISK-08). Defined once here; planner, app and CLI import it."""

    MODEL = "model"
    BASELINE = "baseline"
    OBSERVED = "observed"
    NONE = "none"
    CURRENT_ONLY = "current-only"


RISK_SOURCES: tuple[str, ...] = tuple(m.value for m in RiskSource)


class Hazard(Enum):
    """The six hazards, in stack order (RISK-01)."""

    STORM = "storm"
    TORNADO = "tornado"
    FLOOD = "flood"
    WILDFIRE = "wildfire"
    ICE = "ice"
    SMOKE = "smoke"


# Overlay colour per hazard (RISK-06): alpha carries the probability, RGB identifies the hazard.
HAZARD_COLORS: dict[Hazard, tuple[int, int, int]] = {
    Hazard.STORM: (30, 144, 255),
    Hazard.TORNADO: (220, 20, 60),
    Hazard.FLOOD: (0, 90, 200),
    Hazard.WILDFIRE: (255, 120, 0),
    Hazard.ICE: (120, 220, 255),
    Hazard.SMOKE: (130, 130, 130),
}


def _is_aware(t: Any) -> bool:
    return isinstance(t, datetime) and t.tzinfo is not None and t.utcoffset() is not None


@dataclass(frozen=True)
class RiskLayer:
    """One hazard's calibrated probabilities on GRID for the routing slots (DESIGN 1.2, RISK-01).

    ``prob`` dims are exactly ``("slot", "y", "x")`` (float16 recommended). ``edge_overrides`` maps edge id to a
    per-slot probability array for point features (low-water crossings, bridges). Construction validates the
    fields and raises ValueError on bad input.
    """

    hazard: Hazard
    prob: xr.DataArray
    edge_overrides: dict[int, np.ndarray] | None
    kind: LayerKind
    issued: datetime
    valid_until: datetime
    source: str
    held_out: bool

    def __post_init__(self) -> None:
        if not isinstance(self.hazard, Hazard):
            raise ValueError(f"hazard must be a Hazard, got {self.hazard!r}")
        if not isinstance(self.prob, xr.DataArray) or tuple(self.prob.dims) != ("slot", "y", "x"):
            raise ValueError("prob must be an xr.DataArray with dims ('slot', 'y', 'x')")
        if self.prob.sizes["slot"] < 1:
            raise ValueError("prob needs at least one slot")
        vals = np.asarray(self.prob.values, dtype=np.float64)
        if not np.all(np.isfinite(vals)) or vals.min(initial=0.0) < 0.0 or vals.max(initial=0.0) > 1.0:
            raise ValueError("prob values must be finite and in [0, 1]")
        if self.kind not in LAYER_KINDS:
            raise ValueError(f"kind must be one of {LAYER_KINDS}, got {self.kind!r}")
        if not _is_aware(self.issued) or not _is_aware(self.valid_until):
            raise ValueError("issued and valid_until must be timezone-aware datetimes")
        if self.valid_until <= self.issued:
            raise ValueError("valid_until must be after issued")
        if self.edge_overrides is not None:
            n = self.prob.sizes["slot"]
            for eid, arr in self.edge_overrides.items():
                a = np.asarray(arr, dtype=np.float64)
                if a.ndim != 1 or a.shape[0] != n:
                    raise ValueError(f"edge_overrides[{eid}] must be 1-D of length {n}")
                if not np.all(np.isfinite(a)) or a.min(initial=0.0) < 0.0 or a.max(initial=0.0) > 1.0:
                    raise ValueError(f"edge_overrides[{eid}] values must be in [0, 1]")


@dataclass(frozen=True)
class LayerMeta:
    """Provenance of one hazard in a stack (RISK-02, RISK-04)."""

    kind: str
    source: str
    issued: datetime
    valid_until: datetime
    held_out: bool


@runtime_checkable
class HazardAdapter(Protocol):
    """One hazard plug-in (DESIGN 1.2). Implemented in ``hazards/<h>/adapter.py``; registry in ``hazards.base``."""

    hazard: Hazard
    policy: HazardPolicy

    def baseline(self, scn: Scenario) -> RiskLayer: ...

    def predict(self, scn: Scenario) -> RiskLayer: ...

    def observed(self, scn: Scenario) -> RiskLayer | None: ...


class RiskStack:
    """All six hazards on one window: ``data`` dims ``(hazard, slot, y, x)``, float16 (RISK-02..07).

    Attributes: ``data`` (xr.DataArray), ``t0`` (aware datetime), ``window`` (GridWindow),
    ``meta`` (dict[Hazard, LayerMeta]), ``edge_overrides`` (dict[Hazard, dict[int, np.ndarray]]).
    """

    def __init__(
        self,
        data: xr.DataArray,
        t0: datetime,
        window: GridWindow,
        meta: Mapping[Hazard, LayerMeta] | None = None,
        edge_overrides: Mapping[Hazard, Mapping[int, np.ndarray]] | None = None,
    ) -> None:
        """Wrap a ``(hazard, slot, y, x)`` DataArray; validates dims, coords, values in [0, 1] and aware t0."""
        raise NotImplementedError

    @property
    def n_slots(self) -> int:
        """Number of slots."""
        raise NotImplementedError

    @classmethod
    def zeros(cls, t0: datetime, window: GridWindow, n_slots: int = N_SLOTS) -> RiskStack:
        """All-zero stack with empty meta (RISK-02)."""
        raise NotImplementedError

    @classmethod
    def from_layers(
        cls,
        layers: Sequence[RiskLayer],
        t0: datetime,
        window: GridWindow,
        policies: Mapping[Hazard, HazardPolicy] | None = None,
        n_slots: int = N_SLOTS,
    ) -> RiskStack:
        """Stack layers in Hazard order, applying each policy's beyond-horizon rule (RISK-02, HORIZON-01..04).

        Missing hazards are zero. Coord mismatch, duplicate hazards or a naive t0 raise ValueError.
        """
        raise NotImplementedError

    def slot_start(self, s: int) -> datetime:
        """``t0 + 15 s min``; IndexError outside ``0..n_slots-1`` (RISK-03)."""
        raise NotImplementedError

    def slot_of(self, t: datetime) -> int:
        """``floor((t - t0) / 15 min)`` clamped to ``n_slots - 1``; ValueError for ``t < t0`` or naive t (RISK-03)."""
        raise NotImplementedError

    def get(self, hazard: Hazard | str, slot: int) -> np.ndarray:
        """Stored values ``(ny, nx)`` as float32 (RISK-05)."""
        raise NotImplementedError

    def sample(self, hazard: Hazard | str, slot: int, x: Any, y: Any) -> np.ndarray:
        """Value of the cell containing each point (EPSG:5070 m), float32 of the input shape; 0 outside (RISK-05)."""
        raise NotImplementedError

    def sample_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """``(6, n_slots, n_points)`` float32; ``[i, s] == sample(list(Hazard)[i], s, x, y)`` (RISK-05)."""
        raise NotImplementedError

    def overlay_png(self, hazard: Hazard | str, slot: int, path: str | Path | None = None) -> bytes:
        """RGBA PNG (nx x ny, row 0 north), alpha = floor(255 p + 0.5), RGB = HAZARD_COLORS (RISK-06, RISK-07)."""
        raise NotImplementedError

    def overlay_corners(self) -> list[list[float]]:
        """Window corners ``[lon, lat]``: top-left, top-right, bottom-right, bottom-left (RISK-06)."""
        raise NotImplementedError

    def to_zarr(self, path: str | Path) -> None:
        """Write data, coords, t0, window, meta and edge overrides to a zarr store (RISK-04)."""
        raise NotImplementedError

    @classmethod
    def from_zarr(cls, path: str | Path) -> RiskStack:
        """Inverse of ``to_zarr``, exact; FileNotFoundError for a missing path (RISK-04)."""
        raise NotImplementedError


def decode_overlay(png: bytes) -> np.ndarray:
    """Overlay PNG bytes -> ``alpha / 255`` as float32 ``(ny, nx)`` (RISK-06)."""
    raise NotImplementedError


def build_stack(
    adapters: Sequence[HazardAdapter],
    scn: Scenario,
    source: RiskSource,
    policies: Mapping[Hazard, HazardPolicy] | None = None,
    n_slots: int = N_SLOTS,
) -> RiskStack:
    """The ``--risk-source`` switch: model | baseline | observed | none | current-only (RISK-08).

    ``current-only`` holds each adapter's ``predict`` slot 0 for all slots, for overlays only (the router freezes at
    the CostModel level, PLAN-02). Unknown source -> ValueError.
    """
    raise NotImplementedError
