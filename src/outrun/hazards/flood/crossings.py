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
    raise NotImplementedError


def crossing_overrides(
    edge_prob: Mapping[int, np.ndarray],
    kinds: Mapping[int, CrossingKind],
    crossing_mult: float = 1.5,
    depression_mult: float = 2.0,
) -> dict[int, np.ndarray]:
    """Per-edge float32 (n_slots,) overrides for edges in ``kinds``: LOW_WATER min(1, 1.5p), DEPRESSION min(1, 2p),
    BRIDGE 0. Edge in kinds but not in edge_prob -> KeyError. Inputs not modified (FLOOD-08)."""
    raise NotImplementedError
