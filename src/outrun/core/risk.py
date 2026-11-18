"""Hazards, risk layers, the RiskStack and the adapter protocol (DESIGN 1.2; specs/02-grid-risk.md RISK-01..08).

The router and the app see only a ``RiskStack`` (+ ``HazardPolicy`` objects). Overlay PNGs are rendered from the
same ``stack.data`` array the router samples ("what you see is what routes", RISK-07).
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import numpy as np
import xarray as xr

from outrun.core.grid import GridWindow, xy_to_lonlat
from outrun.core.policy import apply_beyond_horizon, load_hazard_config, load_policies

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


def _as_hazard(h: Hazard | str) -> Hazard:
    """Accept a Hazard or its string value ("storm")."""
    return h if isinstance(h, Hazard) else Hazard(h)


def _coords_match(da: xr.DataArray, window: GridWindow) -> bool:
    """True if ``da`` has explicit ``y``/``x`` coords equal to the window cell centres (within 1e-6 m).

    A dim without a coordinate is rejected: xarray would silently index it 0..n-1.
    """
    for name, centres in (("y", window.y_centers()), ("x", window.x_centers())):
        if name not in da.coords:
            return False
        c = np.asarray(da.coords[name].values, dtype=np.float64)
        if c.shape != centres.shape or not np.allclose(c, centres, rtol=0.0, atol=1e-6):
            return False
    return True


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
        if not _is_aware(t0):
            raise ValueError("t0 must be a timezone-aware datetime")
        if not isinstance(data, xr.DataArray) or tuple(data.dims) != ("hazard", "slot", "y", "x"):
            raise ValueError("data must be an xr.DataArray with dims ('hazard', 'slot', 'y', 'x')")
        if data.shape[0] != len(Hazard) or data.shape[1] < 1 or data.shape[2:] != window.shape:
            raise ValueError(
                f"data shape {data.shape} does not fit 6 hazards x slots x window {window.shape}"
            )
        hazard_names = [h.value for h in Hazard]
        if "hazard" in data.coords and [str(v) for v in data.coords["hazard"].values] != hazard_names:
            raise ValueError("data hazard coords must be the Hazard values in enum order")
        if not _coords_match(data, window):
            raise ValueError("data y/x coords must equal the window cell centres")
        values = np.asarray(data.values, dtype=np.float16)
        if not np.all(np.isfinite(values)) or values.min() < 0 or values.max() > 1:
            raise ValueError("stack values must be finite and in [0, 1]")

        n_slots = data.shape[1]
        self.data = xr.DataArray(
            values,
            dims=("hazard", "slot", "y", "x"),
            coords={
                "hazard": [h.value for h in Hazard],
                "slot": np.arange(n_slots),
                "y": window.y_centers(),
                "x": window.x_centers(),
            },
        )
        self.t0 = t0
        self.window = window
        self.meta: dict[Hazard, LayerMeta] = {_as_hazard(h): m for h, m in (meta or {}).items()}
        self.edge_overrides: dict[Hazard, dict[int, np.ndarray]] = {}
        for h, overrides in (edge_overrides or {}).items():
            clean = {}
            for eid, arr in overrides.items():
                a = np.asarray(arr, dtype=np.float32)
                if a.shape != (n_slots,):
                    raise ValueError(f"edge override {h}/{eid} must have shape ({n_slots},), got {a.shape}")
                clean[int(eid)] = a
            if clean:  # hazards without overrides get no key
                self.edge_overrides[_as_hazard(h)] = clean

    @property
    def n_slots(self) -> int:
        """Number of slots."""
        return int(self.data.shape[1])

    @classmethod
    def zeros(cls, t0: datetime, window: GridWindow, n_slots: int = N_SLOTS) -> RiskStack:
        """All-zero stack with empty meta (RISK-02)."""
        return cls(
            _stack_array(np.zeros((len(Hazard), n_slots, *window.shape), np.float16), window), t0, window
        )

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
        if not _is_aware(t0):
            raise ValueError("t0 must be a timezone-aware datetime")
        seen = [lay.hazard for lay in layers]
        if len(set(seen)) != len(seen):
            raise ValueError(f"more than one layer for the same hazard: {[h.value for h in seen]}")
        for lay in layers:
            if not _coords_match(lay.prob, window):
                raise ValueError(
                    f"{lay.hazard.value} layer: y/x coords missing or not the window cell centres"
                )
        if not layers:
            return cls.zeros(t0, window, n_slots)

        if policies is None:
            policies = load_policies()
        extras = load_hazard_config().get("hazards") or {}

        values = np.zeros((len(Hazard), n_slots, *window.shape), dtype=np.float16)
        meta: dict[Hazard, LayerMeta] = {}
        overrides: dict[Hazard, dict[int, np.ndarray]] = {}
        for lay in layers:
            h = lay.hazard
            cfg = extras.get(h.value) or {}
            zero_after = cfg.get("decay_zero_after_min", 120.0)
            rule_kwargs = dict(
                policy=policies[h],
                t0=t0,
                issued=lay.issued,
                valid_until=lay.valid_until,
                n_slots=n_slots,
                slot_minutes=SLOT_MINUTES,
                zero_after_min=None if zero_after is None else float(zero_after),
                linger_min=float(cfg.get("blocked_linger_min", 15.0)),
            )
            i = list(Hazard).index(h)
            values[i] = apply_beyond_horizon(np.asarray(lay.prob.values, np.float32), **rule_kwargs)
            if lay.edge_overrides:
                overrides[h] = {
                    eid: apply_beyond_horizon(np.asarray(arr, np.float32), **rule_kwargs)
                    for eid, arr in lay.edge_overrides.items()
                }
            meta[h] = _meta_of(lay)
        return cls(_stack_array(values, window), t0, window, meta=meta, edge_overrides=overrides)

    def slot_start(self, s: int) -> datetime:
        """``t0 + 15 s min``; IndexError outside ``0..n_slots-1`` (RISK-03)."""
        if not 0 <= s < self.n_slots:
            raise IndexError(f"slot {s} outside 0..{self.n_slots - 1}")
        return self.t0 + timedelta(minutes=SLOT_MINUTES * s)

    def slot_of(self, t: datetime) -> int:
        """``floor((t - t0) / 15 min)`` clamped to ``n_slots - 1``; ValueError for ``t < t0`` or naive t (RISK-03)."""
        if not _is_aware(t):
            raise ValueError("t must be a timezone-aware datetime")
        if t < self.t0:
            raise ValueError(f"{t} is before the stack t0 {self.t0}")
        s = (t - self.t0) // timedelta(minutes=SLOT_MINUTES)  # timedelta // timedelta is an exact int
        return min(int(s), self.n_slots - 1)

    def get(self, hazard: Hazard | str, slot: int) -> np.ndarray:
        """Stored values ``(ny, nx)`` as float32 (RISK-05)."""
        i = list(Hazard).index(_as_hazard(hazard))
        return self.data.values[i, slot].astype(np.float32)

    def sample(self, hazard: Hazard | str, slot: int, x: Any, y: Any) -> np.ndarray:
        """Value of the cell containing each point (EPSG:5070 m), float32 of the input shape; 0 outside (RISK-05)."""
        grid = self.get(hazard, slot)
        row, col = self.window.cell_of(x, y)
        inside = row >= 0
        out = np.zeros(row.shape, dtype=np.float32)
        out[inside] = grid[row[inside], col[inside]]
        return out

    def sample_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """``(6, n_slots, n_points)`` float32; ``[i, s] == sample(list(Hazard)[i], s, x, y)`` (RISK-05)."""
        row, col = self.window.cell_of(np.ravel(x), np.ravel(y))
        inside = row >= 0
        out = np.zeros((len(Hazard), self.n_slots, row.size), dtype=np.float32)
        out[:, :, inside] = self.data.values[:, :, row[inside], col[inside]].astype(np.float32)
        return out

    def overlay_png(self, hazard: Hazard | str, slot: int, path: str | Path | None = None) -> bytes:
        """RGBA PNG (nx x ny, row 0 north), alpha = floor(255 p + 0.5), RGB = HAZARD_COLORS (RISK-06, RISK-07)."""
        from PIL import Image

        h = _as_hazard(hazard)
        p = self.get(h, slot)  # the same float32 values the router samples (RISK-07)
        alpha = np.clip(np.floor(255.0 * p + 0.5), 0, 255).astype(np.uint8)
        rgba = np.zeros((*p.shape, 4), dtype=np.uint8)
        rgba[alpha > 0, :3] = HAZARD_COLORS[h]
        rgba[..., 3] = alpha
        buf = io.BytesIO()
        Image.fromarray(rgba).save(buf, format="PNG")
        png = buf.getvalue()
        if path is not None:
            Path(path).write_bytes(png)
        return png

    def overlay_corners(self) -> list[list[float]]:
        """Window corners ``[lon, lat]``: top-left, top-right, bottom-right, bottom-left (RISK-06)."""
        xmin, ymin, xmax, ymax = self.window.bounds
        lon, lat = xy_to_lonlat(np.array([xmin, xmax, xmax, xmin]), np.array([ymax, ymax, ymin, ymin]))
        return [[float(a), float(b)] for a, b in zip(lon, lat, strict=True)]

    def to_zarr(self, path: str | Path) -> None:
        """Write data, coords, t0, window, meta and edge overrides to a zarr store (RISK-04)."""
        ds = xr.Dataset({"risk": self.data})
        # Edge overrides: per hazard, an id vector and an (edge, slot) matrix on their own edge dimension.
        for h, overrides in self.edge_overrides.items():
            ids = np.array(sorted(overrides), dtype=np.int64)
            ds[f"edge_ids_{h.value}"] = xr.DataArray(ids, dims=(f"edge_{h.value}",))
            ds[f"edge_prob_{h.value}"] = xr.DataArray(
                np.stack([overrides[e] for e in ids]).astype(np.float32), dims=(f"edge_{h.value}", "slot")
            )
        # Scalars go in one JSON attribute so datetimes keep their UTC offset exactly.
        ds.attrs["outrun"] = json.dumps(
            {
                "t0": self.t0.isoformat(),
                "window": asdict(self.window),
                "meta": {
                    h.value: {
                        "kind": m.kind,
                        "source": m.source,
                        "issued": m.issued.isoformat(),
                        "valid_until": m.valid_until.isoformat(),
                        "held_out": bool(m.held_out),
                    }
                    for h, m in self.meta.items()
                },
                "edge_hazards": [h.value for h in self.edge_overrides],
            }
        )
        ds.to_zarr(str(path), mode="w", consolidated=False)

    @classmethod
    def from_zarr(cls, path: str | Path) -> RiskStack:
        """Inverse of ``to_zarr``, exact; FileNotFoundError for a missing path (RISK-04)."""
        if not Path(path).exists():
            raise FileNotFoundError(f"no risk stack at {path}")
        ds = xr.open_zarr(str(path), consolidated=False).load()
        info = json.loads(ds.attrs["outrun"])
        window = GridWindow(**info["window"])
        meta = {
            Hazard(h): LayerMeta(
                kind=m["kind"],
                source=m["source"],
                issued=datetime.fromisoformat(m["issued"]),
                valid_until=datetime.fromisoformat(m["valid_until"]),
                held_out=m["held_out"],
            )
            for h, m in info["meta"].items()
        }
        overrides = {
            Hazard(h): {
                int(e): row.copy()
                for e, row in zip(ds[f"edge_ids_{h}"].values, ds[f"edge_prob_{h}"].values, strict=True)
            }
            for h in info["edge_hazards"]
        }
        return cls(
            ds["risk"], datetime.fromisoformat(info["t0"]), window, meta=meta, edge_overrides=overrides
        )


def _stack_array(values: np.ndarray, window: GridWindow) -> xr.DataArray:
    """``(hazard, slot, y, x)`` DataArray with the standard coords."""
    return xr.DataArray(
        values,
        dims=("hazard", "slot", "y", "x"),
        coords={
            "hazard": [h.value for h in Hazard],
            "slot": np.arange(values.shape[1]),
            "y": window.y_centers(),
            "x": window.x_centers(),
        },
    )


def _meta_of(layer: RiskLayer, source: str | None = None) -> LayerMeta:
    return LayerMeta(
        kind=layer.kind,
        source=layer.source if source is None else source,
        issued=layer.issued,
        valid_until=layer.valid_until,
        held_out=layer.held_out,
    )


def decode_overlay(png: bytes) -> np.ndarray:
    """Overlay PNG bytes -> ``alpha / 255`` as float32 ``(ny, nx)`` (RISK-06)."""
    from PIL import Image

    alpha = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))[..., 3]
    return alpha.astype(np.float32) / 255.0


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
    if str(source) not in RISK_SOURCES:  # check before calling any adapter
        raise ValueError(f"unknown risk source {source!r}; expected one of {RISK_SOURCES}")
    source = RiskSource(str(source))
    t0, window = scn.t0, scn.grid_window()

    if source is RiskSource.NONE:
        return RiskStack.zeros(t0, window, n_slots)

    if source is RiskSource.CURRENT_ONLY:
        # Hold each forecast's slot 0 for the whole window. No beyond-horizon rule; overlays only.
        values = np.zeros((len(Hazard), n_slots, *window.shape), dtype=np.float16)
        meta: dict[Hazard, LayerMeta] = {}
        overrides: dict[Hazard, dict[int, np.ndarray]] = {}
        for ad in adapters:
            lay = ad.predict(scn)
            if lay.hazard in meta:
                raise ValueError(f"more than one adapter for {lay.hazard.value}")
            if not _coords_match(lay.prob, window):
                raise ValueError(
                    f"{lay.hazard.value} layer: y/x coords missing or not the window cell centres"
                )
            values[list(Hazard).index(lay.hazard)] = np.asarray(lay.prob.values[0], dtype=np.float16)
            if lay.edge_overrides:
                overrides[lay.hazard] = {
                    eid: np.full(n_slots, np.asarray(arr, np.float32)[0], dtype=np.float32)
                    for eid, arr in lay.edge_overrides.items()
                }
            meta[lay.hazard] = _meta_of(lay, source=lay.source + "+current-only")
        return RiskStack(_stack_array(values, window), t0, window, meta=meta, edge_overrides=overrides)

    if source is RiskSource.MODEL:
        layers = [ad.predict(scn) for ad in adapters]
    elif source is RiskSource.BASELINE:
        layers = [ad.baseline(scn) for ad in adapters]
    else:  # OBSERVED: an adapter without ground truth returns None and its hazard stays zero
        layers = [lay for lay in (ad.observed(scn) for ad in adapters) if lay is not None]
    return RiskStack.from_layers(layers, t0, window, policies=policies, n_slots=n_slots)
