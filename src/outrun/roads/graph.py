"""Road graph in CSR form (spec 03-roads, ROAD-01..ROAD-06, ROAD-09).

The router sees only :class:`RoadGraph`: plain numpy arrays, edges stored in CSR order (sorted by source node),
node coordinates in lon/lat (EPSG:4326) and in GRID metres (EPSG:5070). OSMnx/networkx are used while building.
The optimizers (shelters, evacuation, EMS; specs 05-07) take a ``networkx.DiGraph`` view that is built in exactly
one place, ``outrun.routing.cost.to_td_digraph(graph, cost_model)`` (ASTAR-19), which carries blocking, profile
weights and lam from the CostModel and the CSR edge id of every edge.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Bumped whenever the ``.npz`` layout changes (ROAD-04).
NPZ_FORMAT_VERSION = 1

#: Fallback free-flow speeds (km/h) per normalized highway class when a class has no tagged speed (ROAD-03).
DEFAULT_SPEED_KPH: dict[str, float] = {
    "motorway": 105.0,
    "trunk": 90.0,
    "primary": 70.0,
    "secondary": 60.0,
    "tertiary": 50.0,
    "unclassified": 40.0,
    "residential": 40.0,
    "living_street": 15.0,
    "service": 25.0,
    "other": 40.0,
}

#: Loader used by :func:`build_graph`: ``loader(place, bbox) -> networkx.MultiDiGraph`` (OSMnx-style attributes).
GraphLoader = Callable[[str | None, tuple[float, float, float, float] | None], Any]


@dataclass(frozen=True, eq=False)
class RoadGraph:
    """Directed drive network. Pure data (ROAD-01).

    Node arrays have length N, edge arrays length E. Edges are in CSR order: ``src`` is non-decreasing and the
    out-edges of node ``u`` are the edge ids ``indptr[u] .. indptr[u+1]-1``. ``dst`` is the CSR ``indices`` array.
    """

    lon: np.ndarray  # (N,) float64 degrees
    lat: np.ndarray  # (N,) float64 degrees
    x_m: np.ndarray  # (N,) float64, EPSG:5070 metres
    y_m: np.ndarray  # (N,) float64, EPSG:5070 metres
    indptr: np.ndarray  # (N+1,) int64
    src: np.ndarray  # (E,) int64, non-decreasing
    dst: np.ndarray  # (E,) int64
    length_m: np.ndarray  # (E,) float64 > 0
    speed_mps: np.ndarray  # (E,) float64 > 0
    free_flow_s: np.ndarray  # (E,) float64 = length_m / speed_mps
    lanes: np.ndarray  # (E,) int64 >= 1
    capacity_vph: np.ndarray  # (E,) float64 vehicles / hour (all lanes)
    highway: np.ndarray  # (E,) str, normalized highway class (see normalize_highway)
    bridge: np.ndarray  # (E,) bool
    crossing: np.ndarray  # (E,) bool, low-water-crossing proxy (ROAD-07)
    osm_node_id: np.ndarray  # (N,) int64 (synthetic graphs: 0..N-1)
    geom_offsets: np.ndarray | None = None  # (E+1,) int64 into geom_x/geom_y; None = straight edges
    geom_x: np.ndarray | None = None  # polyline vertices (EPSG:5070 m), endpoints included
    geom_y: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)  # e.g. {"source": "Tulsa, Oklahoma, USA"}

    @property
    def n_nodes(self) -> int:
        return int(self.lon.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.src.shape[0])

    def out_edges(self, u: int) -> range:
        """Edge ids leaving node ``u`` (CSR slice)."""
        return range(int(self.indptr[u]), int(self.indptr[u + 1]))

    def reverse_csr(self) -> tuple[np.ndarray, np.ndarray]:
        """Incoming adjacency (ROAD-09): ``(in_indptr (N+1,), in_edges (E,))``.

        The edge ids entering node ``v`` are ``in_edges[in_indptr[v]:in_indptr[v+1]]``, in ascending edge-id order.
        """
        raise NotImplementedError


def normalize_highway(value: Any) -> str:
    """Normalize an OSM ``highway`` tag (ROAD-03).

    A list takes its first element; a ``_link`` suffix is stripped; classes not in ``DEFAULT_SPEED_KPH`` map to
    ``"other"``; None/NaN map to ``"other"``.
    """
    raise NotImplementedError


def impute_speeds(highway: np.ndarray, speed_kph: np.ndarray) -> np.ndarray:
    """Fill missing (NaN or <= 0) speeds and return m/s (ROAD-03).

    A missing speed takes the median of the known speeds of the same normalized class in this graph; if that class
    has none, ``DEFAULT_SPEED_KPH[class]``. Known speeds are kept. Output = kph / 3.6.
    """
    raise NotImplementedError


def edge_capacity_vph(
    highway: np.ndarray, lanes: np.ndarray, table: Mapping[str, float] | None = None
) -> np.ndarray:
    """Capacity in vehicles/hour = per-lane capacity x lanes (ROAD-03).

    ``table`` maps motorway/trunk/primary/secondary/other to veh/h/lane; default is
    ``configs/planning.yaml`` ``roads.capacity_veh_per_h_per_lane``. Classes not in the table use ``"other"``.
    Lanes < 1 or NaN count as 1.
    """
    raise NotImplementedError


def from_edge_arrays(
    lon: np.ndarray,
    lat: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    length_m: np.ndarray,
    speed_mps: np.ndarray,
    *,
    highway: np.ndarray | None = None,
    lanes: np.ndarray | None = None,
    bridge: np.ndarray | None = None,
    crossing: np.ndarray | None = None,
    x_m: np.ndarray | None = None,
    y_m: np.ndarray | None = None,
    capacity_table: Mapping[str, float] | None = None,
) -> tuple[RoadGraph, np.ndarray]:
    """Build a CSR :class:`RoadGraph` from plain edge lists (ROAD-01).

    Returns ``(graph, order)`` where ``order[k]`` is the input index of CSR edge ``k`` (stable sort by ``src``).
    Defaults: highway "other", lanes 1, bridge/crossing False, x_m/y_m projected from lon/lat (EPSG:5070).
    Raises ValueError on: no nodes, mismatched array lengths, node index outside [0, N), length or speed <= 0
    or non-finite.
    """
    raise NotImplementedError


def from_networkx(G: Any, capacity_table: Mapping[str, float] | None = None) -> RoadGraph:
    """Convert an OSMnx-style ``MultiDiGraph`` (ROAD-02).

    Nodes: attributes ``x`` (lon), ``y`` (lat); node order = ascending OSM id, kept in ``osm_node_id``.
    Edges: ``length`` (m, required: missing -> ValueError), optional ``speed_kph``, ``highway``, ``lanes``,
    ``bridge``, ``ford``, ``geometry`` (object with ``.coords`` of lon/lat). Self-loops are dropped. Parallel
    edges (same u, v) keep only the one with the smallest free-flow time (tie: smallest key). Speeds are imputed
    (ROAD-03), bridges parsed (ROAD-08), ``crossing`` = ford tag (ROAD-07 waterway test is applied separately).
    """
    raise NotImplementedError


def save_npz(graph: RoadGraph, path: str | Path) -> Path:
    """Write the graph to a compressed ``.npz`` (ROAD-04). Creates parent folders; stores ``format_version``."""
    raise NotImplementedError


def load_npz(path: str | Path) -> RoadGraph:
    """Read a graph written by :func:`save_npz` (ROAD-04). Arrays round-trip exactly.

    Missing file -> FileNotFoundError; ``format_version`` != ``NPZ_FORMAT_VERSION`` -> ValueError.
    """
    raise NotImplementedError


def cache_key(place: str | None = None, bbox: tuple[float, float, float, float] | None = None) -> str:
    """Cache file stem (ROAD-06).

    place: lowercase, every run of non-alphanumeric characters -> "-", leading/trailing "-" stripped
    ("Tulsa, Oklahoma, USA" -> "tulsa-oklahoma-usa"). bbox (west, south, east, north):
    ``f"bbox_{w:.4f}_{s:.4f}_{e:.4f}_{n:.4f}"``. Exactly one of place/bbox, else ValueError.
    """
    raise NotImplementedError


def build_graph(
    place: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    *,
    cache_dir: str | Path,
    loader: GraphLoader | None = None,
    refresh: bool = False,
) -> RoadGraph:
    """Build (or load from cache) the drive graph of any US place or bbox (ROAD-06; ``outrun route --place``).

    Cache file: ``cache_dir / f"{cache_key(...)}.npz"``. If it exists and not ``refresh``, it is loaded and
    ``loader`` is NOT called. Otherwise ``loader(place, bbox)`` (default: ``osmnx.graph_from_place`` /
    ``graph_from_bbox`` with ``network_type="drive"``, simplified) is called, converted with
    :func:`from_networkx`, saved, and returned. ``graph.meta["source"]`` is the place string or the cache key.
    Errors (ValueError): both or neither of place/bbox; bbox with west >= east, south >= north or latitude
    outside [-90, 90]; loader returns a graph with no edges.
    """
    raise NotImplementedError
