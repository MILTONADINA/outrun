"""Road features: facility snapping, low-water-crossing proxy, bridge tags (spec 03-roads, ROAD-05/07/08)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from outrun.roads.graph import RoadGraph

#: Mean Earth radius used for every haversine distance in Outrun (metres).
EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> np.ndarray:
    """Great-circle distance in metres (vectorized, broadcasting), radius ``EARTH_RADIUS_M`` (ROAD-05)."""
    raise NotImplementedError


def snap_to_nodes(graph: RoadGraph, lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    """Snap facility points to their nearest graph node (ROAD-05).

    Returns ``(node_idx int64, dist_m float64)``, one per point; nearest by haversine distance, ties -> lowest
    node index. Empty input -> two empty arrays. Mismatched lon/lat lengths -> ValueError.
    """
    raise NotImplementedError


def low_water_crossing_flags(
    edge_lines: Sequence[np.ndarray],
    waterways: Sequence[np.ndarray],
    bridge: np.ndarray,
    ford: np.ndarray,
) -> np.ndarray:
    """Low-water-crossing proxy (ROAD-07): OSM ford, or road crossing a waterway without a bridge.

    ``edge_lines[e]`` and ``waterways[k]`` are (n, 2) polylines in projected metres. Edge ``e`` is flagged iff
    ``ford[e]`` or (not ``bridge[e]`` and its polyline intersects or touches any waterway polyline).
    Returns a bool array of length ``len(edge_lines)``. Length mismatch between edge_lines/bridge/ford -> ValueError.
    """
    raise NotImplementedError


def parse_bridge(value: Any) -> bool:
    """Parse an OSM ``bridge`` tag (ROAD-08).

    False for None, NaN, False, "no", "0", "" (case-insensitive); a list is True iff any element parses True;
    every other value ("yes", "viaduct", "movable", True, "1", ...) is True.
    """
    raise NotImplementedError
