"""Synthetic helpers for adapter tests (FLOOD-10, FIRE-10, ICE-08, SMOKE-08). Offline, CPU only.

The scenario window is an 8 x 8 block of 1 km GRID cells with west edge 0 m and north edge 8000 m, so the cell
centres are x = 500, 1500, ..., 7500 and y = 7500, 6500, ..., 500. The "model grid" in these tests is the same
8 x 8 grid, so ``identity_regrid`` stands in for the production regridder.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

import numpy as np

N = 8
X_CENTERS = 500.0 + 1000.0 * np.arange(N)
Y_CENTERS = 7500.0 - 1000.0 * np.arange(N)


def window():
    from outrun.core.grid import GridWindow

    return GridWindow(x0=0.0, y0=8000.0, nx=N, ny=N)


def scenario(t0: datetime, hazard):
    from outrun.core.scenario import Scenario

    return Scenario(
        id="adapter-test",
        t0=t0,
        bbox=(-97.6, 35.4, -97.5, 35.5),
        city="Synthetic",
        hazards=(hazard,),
        window=window(),
    )


def identity_regrid(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def tiny_graph():
    """Four nodes on the row y = 7500 m, edges 0->1, 1->2, 2->3 (and back). Edge flags set after CSR sorting:
    the edge 1->2 is a low-water crossing, the edge 2->3 is a bridge."""
    from tests.fixtures.graphs.graphfix import make_road_graph

    x = np.array([500.0, 2500.0, 4500.0, 6500.0])
    y = np.full(4, 7500.0)
    lon = -97.55 + x / 1e6
    lat = 35.45 + y / 1e6
    src = [0, 1, 1, 2, 2, 3]
    dst = [1, 0, 2, 1, 3, 2]
    g, _ = make_road_graph(lon, lat, src, dst, [2500.0] * 6, [15.0] * 6, x_m=x, y_m=y)
    crossing = (g.src == 1) & (g.dst == 2)
    bridge = (g.src == 2) & (g.dst == 3)
    return dataclasses.replace(g, crossing=crossing, bridge=bridge)


def check_layer(layer, hazard, kind: str, t0: datetime, prefix: str, held_out: bool) -> np.ndarray:
    """Shared RiskLayer assertions; returns the float32 values (12, 8, 8)."""
    from outrun.core.risk import RiskLayer

    assert isinstance(layer, RiskLayer)
    assert layer.hazard is hazard and layer.kind == kind
    assert layer.issued == t0 and layer.valid_until > t0
    assert layer.source.startswith(prefix)
    assert layer.held_out is held_out
    assert layer.prob.dims == ("slot", "y", "x") and layer.prob.shape == (12, N, N)
    assert layer.prob.dtype == np.float16
    np.testing.assert_allclose(layer.prob.coords["x"].values, X_CENTERS)
    np.testing.assert_allclose(layer.prob.coords["y"].values, Y_CENTERS)
    v = layer.prob.values.astype(np.float32)
    assert np.all(np.isfinite(v)) and v.min() >= 0.0 and v.max() <= 1.0
    return v
