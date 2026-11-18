"""Road graph in CSR form (spec 03-roads, ROAD-01..ROAD-06, ROAD-09).

The router sees only :class:`RoadGraph`: plain numpy arrays, edges stored in CSR order (sorted by source node),
node coordinates in lon/lat (EPSG:4326) and in GRID metres (EPSG:5070). OSMnx/networkx are used while building.
The optimizers (shelters, evacuation, EMS; specs 05-07) take a ``networkx.DiGraph`` view that is built in exactly
one place, ``outrun.routing.cost.to_td_digraph(graph, cost_model)`` (ASTAR-19), which carries blocking, profile
weights and lam from the CostModel and the CSR edge id of every edge.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
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
        n = self.n_nodes
        order = np.argsort(self.dst, kind="stable").astype(np.int64)
        counts = np.bincount(self.dst, minlength=n)
        in_indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(counts, out=in_indptr[1:])
        return in_indptr, order


def normalize_highway(value: Any) -> str:
    """Normalize an OSM ``highway`` tag (ROAD-03).

    A list takes its first element; a ``_link`` suffix is stripped; classes not in ``DEFAULT_SPEED_KPH`` map to
    ``"other"``; None/NaN map to ``"other"``.
    """
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) else None
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "other"
    name = str(value).strip().lower()
    if name.endswith("_link"):
        name = name[: -len("_link")]
    return name if name in DEFAULT_SPEED_KPH else "other"


def impute_speeds(highway: np.ndarray, speed_kph: np.ndarray) -> np.ndarray:
    """Fill missing (NaN or <= 0) speeds and return m/s (ROAD-03).

    A missing speed takes the median of the known speeds of the same normalized class in this graph; if that class
    has none, ``DEFAULT_SPEED_KPH[class]``. Known speeds are kept. Output = kph / 3.6.
    """
    hw = np.array([normalize_highway(h) for h in np.asarray(highway, dtype=object)], dtype=object)
    kph = np.asarray(speed_kph, dtype=np.float64).copy()
    known = np.isfinite(kph) & (kph > 0)
    for cls in np.unique(hw[~known]) if (~known).any() else []:
        same = hw == cls
        good = kph[same & known]
        fill = float(np.median(good)) if good.size else DEFAULT_SPEED_KPH[cls]
        kph[same & ~known] = fill
    return kph / 3.6


def edge_capacity_vph(
    highway: np.ndarray, lanes: np.ndarray, table: Mapping[str, float] | None = None
) -> np.ndarray:
    """Capacity in vehicles/hour = per-lane capacity x lanes (ROAD-03).

    ``table`` maps motorway/trunk/primary/secondary/other to veh/h/lane; default is
    ``configs/planning.yaml`` ``roads.capacity_veh_per_h_per_lane``. Classes not in the table use ``"other"``.
    Lanes < 1 or NaN count as 1.
    """
    if table is None:
        table = _default_capacity_table()
    lanes_f = np.asarray(lanes, dtype=np.float64)
    lanes_f = np.where(np.isfinite(lanes_f) & (lanes_f >= 1), lanes_f, 1.0)
    per_lane = np.array([float(table.get(str(h), table["other"])) for h in np.asarray(highway)], dtype=np.float64)
    return per_lane * lanes_f


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
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    n = lon.shape[0]
    if n == 0:
        raise ValueError("graph has no nodes")
    if lat.shape[0] != n:
        raise ValueError("lon/lat lengths differ")
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    length_m = np.asarray(length_m, dtype=np.float64)
    speed_mps = np.asarray(speed_mps, dtype=np.float64)
    e = src.shape[0]
    highway = np.full(e, "other") if highway is None else np.asarray(highway).astype(str)
    lanes = np.ones(e, dtype=np.int64) if lanes is None else np.asarray(lanes, dtype=np.int64)
    bridge = np.zeros(e, dtype=bool) if bridge is None else np.asarray(bridge, dtype=bool)
    crossing = np.zeros(e, dtype=bool) if crossing is None else np.asarray(crossing, dtype=bool)
    for name, arr in (("dst", dst), ("length_m", length_m), ("speed_mps", speed_mps), ("highway", highway),
                      ("lanes", lanes), ("bridge", bridge), ("crossing", crossing)):
        if arr.shape != (e,):
            raise ValueError(f"{name} has shape {arr.shape}, expected ({e},)")
    if e and (src.min() < 0 or dst.min() < 0 or src.max() >= n or dst.max() >= n):
        raise ValueError("edge endpoint outside [0, N)")
    if not (np.all(np.isfinite(length_m)) and np.all(length_m > 0)):
        raise ValueError("edge lengths must be finite and > 0")
    if not (np.all(np.isfinite(speed_mps)) and np.all(speed_mps > 0)):
        raise ValueError("edge speeds must be finite and > 0")
    if x_m is None or y_m is None:
        x_m, y_m = _project(lon, lat)
    order = np.argsort(src, kind="stable").astype(np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    lanes = np.maximum(lanes[order], 1)
    hw = highway[order]
    graph = RoadGraph(
        lon=lon, lat=lat, x_m=np.asarray(x_m, dtype=np.float64), y_m=np.asarray(y_m, dtype=np.float64),
        indptr=indptr, src=src[order], dst=dst[order], length_m=length_m[order], speed_mps=speed_mps[order],
        free_flow_s=length_m[order] / speed_mps[order], lanes=lanes,
        capacity_vph=edge_capacity_vph(hw, lanes, capacity_table), highway=hw,
        bridge=bridge[order], crossing=crossing[order], osm_node_id=np.arange(n, dtype=np.int64),
    )
    return graph, order


def from_networkx(G: Any, capacity_table: Mapping[str, float] | None = None) -> RoadGraph:
    """Convert an OSMnx-style ``MultiDiGraph`` (ROAD-02).

    Nodes: attributes ``x`` (lon), ``y`` (lat); node order = ascending OSM id, kept in ``osm_node_id``.
    Edges: ``length`` (m, required: missing -> ValueError), optional ``speed_kph``, ``highway``, ``lanes``,
    ``bridge``, ``ford``, ``geometry`` (object with ``.coords`` of lon/lat). Self-loops are dropped. Parallel
    edges (same u, v) keep only the one with the smallest free-flow time (tie: smallest key). Speeds are imputed
    (ROAD-03), bridges parsed (ROAD-08), ``crossing`` = ford tag (ROAD-07 waterway test is applied separately).
    """
    from outrun.roads.features import parse_bridge

    osm_ids = np.array(sorted(G.nodes), dtype=np.int64)
    index = {int(o): i for i, o in enumerate(osm_ids)}
    lon = np.array([float(G.nodes[o]["x"]) for o in osm_ids], dtype=np.float64)
    lat = np.array([float(G.nodes[o]["y"]) for o in osm_ids], dtype=np.float64)
    rows = []  # (u, v, key, data) without self-loops
    for u, v, k, d in G.edges(keys=True, data=True):
        if u == v:
            continue
        if d.get("length") is None:
            raise ValueError(f"edge ({u}, {v}, {k}) has no length")
        rows.append((index[int(u)], index[int(v)], k, d))
    hw_all = np.array([normalize_highway(d.get("highway")) for *_, d in rows], dtype=object)
    kph_all = np.array([_to_float(d.get("speed_kph")) for *_, d in rows], dtype=np.float64)
    mps_all = impute_speeds(hw_all, kph_all) if rows else np.zeros(0)
    len_all = np.array([float(d["length"]) for *_, d in rows], dtype=np.float64)
    fft_all = len_all / mps_all if rows else np.zeros(0)
    best: dict[tuple[int, int], int] = {}  # parallel edges: keep min free-flow time, tie -> smallest key
    for i, (u, v, k, _) in enumerate(rows):
        j = best.get((u, v))
        if j is None or (fft_all[i], _key_sort(k)) < (fft_all[j], _key_sort(rows[j][2])):
            best[(u, v)] = i
    keep = np.array(sorted(best.values()), dtype=np.int64)
    kept = [rows[i] for i in keep]
    src = np.array([r[0] for r in kept], dtype=np.int64)
    dst = np.array([r[1] for r in kept], dtype=np.int64)
    lanes = np.array([_parse_lanes(r[3].get("lanes")) for r in kept], dtype=np.int64)
    bridge = np.array([parse_bridge(r[3].get("bridge")) for r in kept], dtype=bool)
    ford = np.array([parse_bridge(r[3].get("ford")) for r in kept], dtype=bool)
    graph, order = from_edge_arrays(
        lon, lat, src, dst, len_all[keep], mps_all[keep], highway=hw_all[keep].astype(str), lanes=lanes,
        bridge=bridge, crossing=ford, capacity_table=capacity_table,
    )
    # polyline geometry (projected metres), endpoints included, in CSR order
    xs, ys, offsets = [], [], [0]
    for e in order:
        u, v, _, d = kept[e]
        geom = d.get("geometry")
        if geom is not None:
            coords = np.asarray(geom.coords, dtype=np.float64)
            gx, gy = _project(coords[:, 0], coords[:, 1])
        else:
            gx = np.array([graph.x_m[u], graph.x_m[v]])
            gy = np.array([graph.y_m[u], graph.y_m[v]])
        xs.append(gx)
        ys.append(gy)
        offsets.append(offsets[-1] + len(gx))
    return replace(
        graph,
        osm_node_id=osm_ids,
        geom_offsets=np.array(offsets, dtype=np.int64),
        geom_x=np.concatenate(xs) if xs else np.zeros(0),
        geom_y=np.concatenate(ys) if ys else np.zeros(0),
    )


def save_npz(graph: RoadGraph, path: str | Path) -> Path:
    """Write the graph to a compressed ``.npz`` (ROAD-04). Creates parent folders; stores ``format_version``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for f in fields(RoadGraph):
        val = getattr(graph, f.name)
        if f.name == "meta" or val is None:
            continue
        arrays[f.name] = val.astype(str) if f.name == "highway" else val
    arrays["meta"] = np.array(json.dumps(graph.meta, default=str))
    arrays["format_version"] = np.array(NPZ_FORMAT_VERSION, dtype=np.int64)
    np.savez_compressed(path, **arrays)
    return path


def load_npz(path: str | Path) -> RoadGraph:
    """Read a graph written by :func:`save_npz` (ROAD-04). Arrays round-trip exactly.

    Missing file -> FileNotFoundError; ``format_version`` != ``NPZ_FORMAT_VERSION`` -> ValueError.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        version = int(data["format_version"]) if "format_version" in data else -1
        if version != NPZ_FORMAT_VERSION:
            raise ValueError(f"{path}: format_version {version} != {NPZ_FORMAT_VERSION}")
        kwargs: dict[str, Any] = {}
        for f in fields(RoadGraph):
            if f.name == "meta":
                kwargs["meta"] = json.loads(str(data["meta"])) if "meta" in data else {}
            elif f.name in data:
                kwargs[f.name] = data[f.name]
    return RoadGraph(**kwargs)


def cache_key(place: str | None = None, bbox: tuple[float, float, float, float] | None = None) -> str:
    """Cache file stem (ROAD-06).

    place: lowercase, every run of non-alphanumeric characters -> "-", leading/trailing "-" stripped
    ("Tulsa, Oklahoma, USA" -> "tulsa-oklahoma-usa"). bbox (west, south, east, north):
    ``f"bbox_{w:.4f}_{s:.4f}_{e:.4f}_{n:.4f}"``. Exactly one of place/bbox, else ValueError.
    """
    if (place is None) == (bbox is None):
        raise ValueError("give exactly one of place or bbox")
    if place is not None:
        return re.sub(r"[^a-z0-9]+", "-", place.lower()).strip("-")
    w, s, e, n = bbox
    return f"bbox_{w:.4f}_{s:.4f}_{e:.4f}_{n:.4f}"


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
    key = cache_key(place, bbox)
    if bbox is not None:
        w, s, e, n = (float(b) for b in bbox)
        if w >= e or s >= n or not (-90 <= s <= 90 and -90 <= n <= 90):
            raise ValueError(f"invalid bbox {bbox}")
    path = Path(cache_dir) / f"{key}.npz"
    if path.exists() and not refresh:
        return load_npz(path)
    G = (loader or _osmnx_loader)(place, bbox)
    if G.number_of_edges() == 0:
        raise ValueError(f"no road edges for {place or key}")
    graph = from_networkx(G)
    graph = replace(graph, meta={**graph.meta, "source": place if place is not None else key})
    save_npz(graph, path)
    return graph


# ---------------------------------------------------------------- helpers
def _project(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """lon/lat (EPSG:4326) -> EPSG:5070 metres."""
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x, y = tr.transform(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def _default_capacity_table() -> dict[str, float]:
    """``configs/planning.yaml`` ``roads.capacity_veh_per_h_per_lane``."""
    import yaml

    cfg_path = Path(__file__).resolve().parents[3] / "configs" / "planning.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    return {str(k): float(v) for k, v in cfg["roads"]["capacity_veh_per_h_per_lane"].items()}


def _to_float(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        vals = [_to_float(v) for v in value]
        vals = [v for v in vals if math.isfinite(v)]
        return min(vals) if vals else math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _parse_lanes(value: Any) -> int:
    """int or numeric string ("2", "2;3"); a list takes its max; missing/unparsable -> 1."""
    if isinstance(value, (list, tuple)):
        return max((_parse_lanes(v) for v in value), default=1)
    if value is None:
        return 1
    nums = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", str(value))]
    return max(1, int(max(nums))) if nums else 1


def _key_sort(key: Any) -> tuple[int, Any]:
    return (0, key) if isinstance(key, (int, np.integer)) else (1, str(key))


def _osmnx_loader(place: str | None, bbox: tuple[float, float, float, float] | None) -> Any:
    """Default loader: OSMnx drive network (simplified), with speeds in km/h."""
    import osmnx as ox

    if place is not None:
        G = ox.graph_from_place(place, network_type="drive", simplify=True)
    else:
        G = ox.graph_from_bbox(bbox, network_type="drive", simplify=True)  # osmnx 2.x: (left, bottom, right, top)
    return ox.add_edge_speeds(G) if G.number_of_edges() else G
