"""Synthetic moving-Gaussian hazards for tests and the skeleton app (specs/02-grid-risk.md, RISK-11, RISK-12)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import xarray as xr

from outrun.core.grid import GridWindow, xy_to_lonlat
from outrun.core.policy import HazardPolicy, load_policies
from outrun.core.risk import Hazard, LayerKind, RiskLayer, RiskSource, RiskStack, build_stack
from outrun.core.scenario import Scenario

SYNTHETIC_STARTS: dict[Hazard, tuple[float, float]] = {
    Hazard.STORM: (4.0, 4.0),
    Hazard.TORNADO: (16.0, 4.0),
    Hazard.FLOOD: (4.0, 16.0),
    Hazard.WILDFIRE: (24.0, 24.0),
    Hazard.ICE: (16.0, 16.0),
    Hazard.SMOKE: (28.0, 8.0),
}

DEFAULT_T0 = datetime(2019, 5, 26, 3, 0, tzinfo=UTC)


def moving_gaussian(
    n_slots: int = 12,
    shape: tuple[int, int] = (32, 32),
    start: tuple[float, float] = (8.0, 8.0),
    velocity: tuple[float, float] = (1.5, 1.0),
    sigma: float = 3.0,
    peak: float = 0.9,
) -> np.ndarray:
    """float32 ``(n_slots, ny, nx)`` Gaussian blob, centre ``start + s * velocity`` in (row, col) cells (RISK-11).

    ValueError for ``sigma <= 0``, ``peak`` outside (0, 1] or ``n_slots < 1``.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if not 0 < peak <= 1:
        raise ValueError(f"peak must be in (0, 1], got {peak}")
    if n_slots < 1:
        raise ValueError(f"n_slots must be >= 1, got {n_slots}")
    ny, nx = shape
    rows, cols = np.mgrid[0:ny, 0:nx].astype(np.float64)
    cube = np.empty((n_slots, ny, nx), dtype=np.float32)
    for s in range(n_slots):
        cy = start[0] + s * velocity[0]
        cx = start[1] + s * velocity[1]
        cube[s] = peak * np.exp(-((rows - cy) ** 2 + (cols - cx) ** 2) / (2 * sigma**2))
    return cube


def synthetic_window(n: int = 32) -> GridWindow:
    """``GridWindow(x0=-150_000, y0=1_300_000, nx=n, ny=n, res=1000)`` (RISK-12)."""
    return GridWindow(x0=-150_000.0, y0=1_300_000.0, nx=n, ny=n, res=1000.0)


def synthetic_scenario(t0: datetime | None = None, n: int = 32) -> Scenario:
    """Scenario ``"synthetic"`` on ``synthetic_window(n)``, all hazards, none held out (RISK-12)."""
    window = synthetic_window(n)
    # bbox = lon/lat extent of the window outline (densified: Albers edges are curved in lon/lat).
    xmin, ymin, xmax, ymax = window.bounds
    t = np.linspace(0.0, 1.0, 50)
    x = np.concatenate(
        [xmin + (xmax - xmin) * t, np.full(50, xmax), xmin + (xmax - xmin) * t, np.full(50, xmin)]
    )
    y = np.concatenate(
        [np.full(50, ymin), ymin + (ymax - ymin) * t, np.full(50, ymax), ymin + (ymax - ymin) * t]
    )
    lon, lat = xy_to_lonlat(x, y)
    return Scenario(
        id="synthetic",
        t0=DEFAULT_T0 if t0 is None else t0,
        bbox=(float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())),
        city="Synthetic",
        hazards=tuple(Hazard),
        title="Synthetic moving Gaussians",
        synthetic=True,
        window=window,
    )


def synthetic_layer(
    hazard: Hazard,
    window: GridWindow,
    t0: datetime,
    *,
    kind: LayerKind = "forecast",
    start: tuple[float, float] | None = None,
    velocity: tuple[float, float] = (1.5, 1.0),
    sigma: float = 3.0,
    peak: float = 0.9,
    n_slots: int = 12,
    horizon_min: float = 180,
    source: str | None = None,
) -> RiskLayer:
    """Moving-Gaussian ``RiskLayer`` on ``window`` (float16, issued t0, valid t0 + horizon) (RISK-12)."""
    cube = moving_gaussian(n_slots, window.shape, start or SYNTHETIC_STARTS[hazard], velocity, sigma, peak)
    return _layer(hazard, window, t0, cube, kind, horizon_min, source)


def _layer(
    hazard: Hazard,
    window: GridWindow,
    t0: datetime,
    cube: np.ndarray,
    kind: LayerKind,
    horizon_min: float,
    source: str | None,
) -> RiskLayer:
    prob = xr.DataArray(
        cube.astype(np.float16),
        dims=("slot", "y", "x"),
        coords={"slot": np.arange(cube.shape[0]), "y": window.y_centers(), "x": window.x_centers()},
    )
    return RiskLayer(
        hazard=hazard,
        prob=prob,
        edge_overrides=None,
        kind=kind,
        issued=t0,
        valid_until=t0 + timedelta(minutes=horizon_min),
        source=source or f"synthetic.{hazard.value}.{kind}",
        held_out=False,
    )


class SyntheticAdapter:
    """A ``HazardAdapter`` built from moving Gaussians (RISK-12).

    predict: forecast Gaussian; baseline: predict slot 0 held (persistence); observed: start shifted (+1, +1).
    """

    def __init__(self, hazard: Hazard, policy: HazardPolicy | None = None) -> None:
        self.hazard = hazard
        self.policy = policy if policy is not None else load_policies()[hazard]

    def baseline(self, scn: Scenario) -> RiskLayer:
        forecast = self.predict(scn)
        # Persistence: slot 0 of the forecast repeated in every slot.
        held = np.repeat(forecast.prob.values[:1], forecast.prob.sizes["slot"], axis=0)
        return _layer(self.hazard, scn.grid_window(), scn.t0, held, "baseline", 180, None)

    def predict(self, scn: Scenario) -> RiskLayer:
        return synthetic_layer(self.hazard, scn.grid_window(), scn.t0, kind="forecast")

    def observed(self, scn: Scenario) -> RiskLayer | None:
        r, c = SYNTHETIC_STARTS[self.hazard]
        return synthetic_layer(
            self.hazard, scn.grid_window(), scn.t0, kind="observed", start=(r + 1.0, c + 1.0)
        )


def synthetic_stack(t0: datetime | None = None, n: int = 32) -> RiskStack:
    """``build_stack`` of the six SyntheticAdapters with source ``"model"`` on ``synthetic_scenario`` (RISK-12)."""
    adapters = [SyntheticAdapter(h) for h in Hazard]
    policies = {ad.hazard: ad.policy for ad in adapters}
    return build_stack(adapters, synthetic_scenario(t0, n), RiskSource.MODEL, policies=policies)
