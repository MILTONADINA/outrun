"""Low-water crossing edge overrides for the flood RiskLayer (specs/10-flood.md: FLOOD-08).

Crossing detection itself (OSM ford / road-by-waterway without bridge) belongs to roads/features.py; this module
takes edge ids with their crossing kind.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

import numpy as np


class CrossingKind(StrEnum):
    LOW_WATER = "low_water"
    DEPRESSION = "depression"  # low-water crossing in a local depression
    BRIDGE = "bridge"


def in_depression(
    dem: np.ndarray, cell_m: float, row: int, col: int, radius_m: float = 300.0, drop_m: float = 2.0
) -> bool:
    """True iff dem[row, col] < median(cells within radius_m, centre included, NaN ignored) - drop_m.

    Neighbourhood clipped at the border; (row, col) outside the DEM -> ValueError (FLOOD-08).
    """
    dem = np.asarray(dem, dtype=np.float64)
    if not (0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]):
        raise ValueError(f"(row, col)=({row}, {col}) is outside the DEM {dem.shape}")
    r = int(np.floor(radius_m / cell_m))
    r0, r1 = max(row - r, 0), min(row + r + 1, dem.shape[0])
    c0, c1 = max(col - r, 0), min(col + r + 1, dem.shape[1])
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    inside = np.hypot(rr - row, cc - col) * cell_m <= radius_m
    window = dem[r0:r1, c0:c1][inside]
    window = window[np.isfinite(window)]
    centre = dem[row, col]
    if window.size == 0 or not np.isfinite(centre):
        return False
    return bool(centre < np.median(window) - drop_m)


def crossing_overrides(
    edge_prob: Mapping[int, np.ndarray],
    kinds: Mapping[int, CrossingKind],
    crossing_mult: float = 1.5,
    depression_mult: float = 2.0,
) -> dict[int, np.ndarray]:
    """Per-edge float32 (n_slots,) overrides for edges in ``kinds``: LOW_WATER min(1, 1.5p), DEPRESSION min(1, 2p),
    BRIDGE 0. Edge in kinds but not in edge_prob -> KeyError. Inputs not modified (FLOOD-08)."""
    out: dict[int, np.ndarray] = {}
    for edge, kind in kinds.items():
        p = np.asarray(edge_prob[edge], dtype=np.float32)  # KeyError if missing
        if kind == CrossingKind.BRIDGE:
            out[edge] = np.zeros_like(p)
        elif kind == CrossingKind.DEPRESSION:
            out[edge] = np.minimum(1.0, depression_mult * p).astype(np.float32)
        else:
            out[edge] = np.minimum(1.0, crossing_mult * p).astype(np.float32)
    return out
