"""Road features: facility snapping, low-water-crossing proxy, bridge tags (spec 03-roads, ROAD-05/07/08)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from outrun.roads.graph import RoadGraph

#: Mean Earth radius used for every haversine distance in Outrun (metres).
EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> np.ndarray:
    """Great-circle distance in metres (vectorized, broadcasting), radius ``EARTH_RADIUS_M`` (ROAD-05)."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(a, dtype=np.float64)) for a in (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def snap_to_nodes(graph: RoadGraph, lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    """Snap facility points to their nearest graph node (ROAD-05).

    Returns ``(node_idx int64, dist_m float64)``, one per point; nearest by haversine distance, ties -> lowest
    node index. Empty input -> two empty arrays. Mismatched lon/lat lengths -> ValueError.
    """
    lon = np.atleast_1d(np.asarray(lon, dtype=np.float64))
    lat = np.atleast_1d(np.asarray(lat, dtype=np.float64))
    if lon.shape != lat.shape:
        raise ValueError("lon and lat lengths differ")
    idx = np.zeros(lon.shape[0], dtype=np.int64)
    dist = np.zeros(lon.shape[0], dtype=np.float64)
    for i in range(lon.shape[0]):  # one vectorized pass over all nodes per point
        d = haversine_m(lon[i], lat[i], graph.lon, graph.lat)
        best = int(np.flatnonzero(d <= d.min() + 1e-6)[0])  # ties -> lowest node index
        idx[i], dist[i] = best, d[best]
    return idx, dist


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
    n = len(edge_lines)
    bridge = np.asarray(bridge, dtype=bool)
    ford = np.asarray(ford, dtype=bool)
    if bridge.shape != (n,) or ford.shape != (n,):
        raise ValueError("edge_lines, bridge and ford must have the same length")
    out = ford.copy()
    candidates = np.flatnonzero(~out & ~bridge)
    if len(waterways) == 0 or candidates.size == 0:
        return out
    from shapely import LineString, STRtree

    water = STRtree([LineString(np.asarray(w, dtype=np.float64)) for w in waterways])
    roads = [LineString(np.asarray(edge_lines[e], dtype=np.float64)) for e in candidates]
    hit_roads, _ = water.query(roads, predicate="intersects")  # intersects includes touching
    out[candidates[np.unique(hit_roads)]] = True
    return out


def parse_bridge(value: Any) -> bool:
    """Parse an OSM ``bridge`` tag (ROAD-08).

    False for None, NaN, False, "no", "0", "" (case-insensitive); a list is True iff any element parses True;
    every other value ("yes", "viaduct", "movable", True, "1", ...) is True.
    """
    if isinstance(value, (list, tuple)):
        return any(parse_bridge(v) for v in value)
    if value is None or value is False:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return not (isinstance(value, str) and value.strip().lower() in ("no", "0", ""))
