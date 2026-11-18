# 03: Roads (real OSM road graph)

Modules: `src/outrun/roads/graph.py`, `src/outrun/roads/features.py`
Tests: `tests/unit/test_roads.py` (synthetic graphs and a fake OSMnx-style loader; no OSM download)
Design: DESIGN 4.1, 2.2 (`roads/graph.py`, `roads/features.py`). Proposal phrases: "real road maps" (ROAD-01..05),
"any US city" (ROAD-06).

Purpose: turn an OSM drive network (via OSMnx) into one plain-numpy `RoadGraph` in CSR form that every router
and optimizer uses. Real graphs are built in T4 (OKC) and on demand (`outrun route --place`); unit tests use tiny
synthetic graphs only.

Conventions: node arrays length N, edge arrays length E; distances in metres, times in seconds, speeds in m/s
internally (km/h only when reading OSM tags). Great-circle distances use the sphere of radius 6,371,008.8 m.

`Must` = needed for the approved proposal (coverage matrix in `00-overview.md`). `Should` = depth.

## Requirements

### ROAD-01 `RoadGraph` and `from_edge_arrays` build a CSR graph (Must)
`RoadGraph` is a frozen dataclass of numpy arrays (fields in `graph.py`): node `lon`, `lat`, `x_m`, `y_m`
(EPSG:5070), `osm_node_id`; CSR `indptr` (N+1); edge `src` (non-decreasing), `dst`, `length_m`, `speed_mps`,
`free_flow_s = length_m / speed_mps`, `lanes`, `capacity_vph`, `highway`, `bridge`, `crossing`; optional polyline
geometry (`geom_offsets`, `geom_x`, `geom_y`, projected metres) and `meta`. `out_edges(u)` is the CSR slice
`indptr[u]..indptr[u+1]-1`; `n_nodes`, `n_edges` give sizes.

`from_edge_arrays(lon, lat, src, dst, length_m, speed_mps, *, highway, lanes, bridge, crossing, x_m, y_m,
capacity_table) -> (graph, order)`:
- sorts edges by `src` with a **stable** sort; `order[k]` is the input index of CSR edge `k`;
- `indptr[u+1] - indptr[u]` = out-degree of `u`;
- defaults: highway `"other"`, lanes 1, bridge/crossing False, `x_m/y_m` projected from lon/lat with pyproj
  (EPSG:4326 -> EPSG:5070, `always_xy`), capacity per ROAD-03;
- raises `ValueError` for: zero nodes, mismatched array lengths, an endpoint outside `[0, N)`, a length or speed
  that is <= 0 or non-finite.

### ROAD-02 `from_networkx(G)` converts an OSMnx-style MultiDiGraph (Must)
- Node order = ascending OSM id (kept in `osm_node_id`); `lon = x`, `lat = y` node attributes.
- Edge attributes: `length` (metres, **required**; missing -> `ValueError`), optional `speed_kph` (imputed per
  ROAD-03 when missing), `highway` (normalized, ROAD-03), `lanes` (int or numeric string; a list takes its max;
  missing/unparsable -> 1), `bridge` (ROAD-08), `ford` (`crossing = parse_bridge`-style truthiness of `ford`),
  `geometry` (object with `.coords` in lon/lat; projected into `geom_*`; missing -> straight segment).
- Self-loops are dropped. Parallel edges (same `u`, `v`) collapse to the one with the smallest free-flow time
  (tie: smallest key). Output is in CSR order (ROAD-01) and capacity follows ROAD-03.

### ROAD-03 Highway classes, speed imputation, capacity (Must)
- `normalize_highway(value)`: a list takes its first element; a `_link` suffix is stripped; anything not a key of
  `DEFAULT_SPEED_KPH` (motorway, trunk, primary, secondary, tertiary, unclassified, residential, living_street,
  service, other), and None/NaN, map to `"other"`.
- `impute_speeds(highway, speed_kph) -> m/s`: known speeds (finite, > 0) are kept; a missing speed takes the median
  of the known speeds of the same normalized class in the same graph, else `DEFAULT_SPEED_KPH[class]`; result / 3.6.
- `edge_capacity_vph(highway, lanes, table=None)`: per-lane capacity x lanes; lanes < 1 or NaN count as 1;
  classes not in the table use `"other"`. Default table = `configs/planning.yaml`
  `roads.capacity_veh_per_h_per_lane` (motorway/trunk 1800, primary/secondary 900, other 600; DESIGN 4.1).

### ROAD-04 CSR `.npz` save / load round-trips exactly (Must)
`save_npz(graph, path) -> Path` writes a compressed `.npz` with every array field, `meta` as a JSON string and an
integer `format_version` = `NPZ_FORMAT_VERSION`; parent folders are created; no pickled objects (loadable with
`allow_pickle=False`; strings stored as unicode arrays). `load_npz(path)` returns a `RoadGraph` whose arrays are
identical to the saved ones. Missing file -> `FileNotFoundError`; a different `format_version` -> `ValueError`.

### ROAD-05 Facility snapping to the nearest node (Must)
`haversine_m(lon1, lat1, lon2, lat2)` is the vectorized great-circle distance (radius 6,371,008.8 m,
broadcasting). `snap_to_nodes(graph, lon, lat) -> (node_idx, dist_m)` returns, per point, the nearest node by
haversine distance and that distance; ties (within 1e-6 m) go to the lowest node index. Empty input -> two empty
arrays; lon/lat of different lengths -> `ValueError`. Used for shelters, hospitals, stations and schools
(DESIGN 4.1 Facilities).

### ROAD-06 Any US city: build and cache from a place name or bbox (Must, answer to the professor)
`cache_key(place=None, bbox=None)`: for a place, lowercase, each run of non-alphanumeric characters -> `-`,
leading/trailing `-` removed (`"Tulsa, Oklahoma, USA"` -> `tulsa-oklahoma-usa`); for a bbox
`(west, south, east, north)`, `f"bbox_{w:.4f}_{s:.4f}_{e:.4f}_{n:.4f}"`. Exactly one of the two, else `ValueError`.

`build_graph(place=None, bbox=None, *, cache_dir, loader=None, refresh=False) -> RoadGraph`
(`outrun route --place`, the app city box):
- cache file `cache_dir/<cache_key>.npz`; if it exists and `refresh` is False it is loaded and the loader is
  **not** called;
- otherwise `loader(place, bbox)` is called (default: `osmnx.graph_from_place` / `graph_from_bbox`,
  `network_type="drive"`, simplified; imported lazily), converted with `from_networkx`, saved, returned;
- `graph.meta["source"]` = the place string (bbox: the cache key);
- `ValueError` for: both or neither of place/bbox; bbox with west >= east, south >= north or a latitude outside
  [-90, 90] (checked before any loader call); a loaded graph with no edges.
Acceptance (spec 16) runs this on Tulsa and one non-Oklahoma city.

### ROAD-07 Low-water-crossing proxy (Should; feeds flood `edge_overrides`)
`low_water_crossing_flags(edge_lines, waterways, bridge, ford) -> bool array`: edge `e` is a crossing iff `ford[e]`
or (not `bridge[e]` and its polyline (n x 2, projected metres) intersects or touches any waterway polyline).
No waterways -> ford flags only. `edge_lines`, `bridge`, `ford` of different lengths -> `ValueError`. (DESIGN 2.2:
OSM ford + road-by-waterway without bridge.)

### ROAD-08 Bridge tag parsing (Should; feeds ice bridge multiplier)
`parse_bridge(value) -> bool`: False for None, NaN, False, and the strings `"no"`, `"0"`, `""` (case-insensitive);
a list is True iff any element is True; every other value (`"yes"`, `"viaduct"`, `"movable"`, True, `"1"`, ...) is
True.

### ROAD-09 Reverse adjacency (Should; used by reverse search in spec 07)
`RoadGraph.reverse_csr() -> (in_indptr (N+1,), in_edges (E,))`: the edge ids entering `v` are
`in_edges[in_indptr[v]:in_indptr[v+1]]`, in ascending edge-id order; every edge id appears exactly once.
