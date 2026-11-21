"""Greenville CA road graph for the WSTS 2021 Dixie Fire demo (T20); same recipe as build_okc_graph.py.

osmium extract -b -121.55,39.85,-120.60,40.60 norcal-latest.osm.pbf, then tags-filter highway=* (drivable classes) and
waterway=river,stream,canal into $PROC/roads/greenville_{drive,water}.osm; this script turns them into
$PROC/roads/greenville.npz.
"""

import json
import time
from collections import Counter
from dataclasses import replace

import numpy as np
import osmnx as ox

from outrun.roads.features import low_water_crossing_flags
from outrun.roads.graph import _project, build_graph, save_npz

# osmnx drops the ford tag unless it is listed (T4 left fords = 0)
ox.settings.useful_tags_way = list(dict.fromkeys([*ox.settings.useful_tags_way, "ford"]))
D = "/Users/milto/OutrunData/proc/roads"
bbox = (-121.50, 39.88, -120.65, 40.58)  # covers the Dixie Fire day-0 detections (T20)


def loader(place, bb):
    G = ox.graph_from_xml(f"{D}/greenville_drive.osm", simplify=True, retain_all=False)
    G = ox.truncate.truncate_graph_bbox(G, bb)
    G = ox.truncate.largest_component(G, strongly=False)
    return ox.add_edge_speeds(G)


t = time.time()
g = build_graph(bbox=bbox, cache_dir=D, loader=loader, refresh=True)
t1 = time.time()
print("graph", g.n_nodes, g.n_edges, int(g.bridge.sum()), int(g.crossing.sum()), round(t1 - t), flush=True)
w = ox.features_from_xml(f"{D}/greenville_water.osm", tags={"waterway": ["river", "stream", "canal"]})
# drains/ditches are not flash-flood crossings; culverted (tunnel) waterway stretches carry the road over them
w = w[w.geometry.geom_type.isin(["LineString", "MultiLineString"])]
if "tunnel" in w.columns:
    w = w[~w["tunnel"].astype(str).str.lower().isin(["culvert", "yes", "building_passage", "flooded"])]
w = w.explode(index_parts=False)
ways = []
for geom in w.geometry:
    c = np.asarray(geom.coords)
    x, y = _project(c[:, 0], c[:, 1])
    ways.append(np.c_[x, y])
lines = [
    np.c_[g.geom_x[a:b], g.geom_y[a:b]] for a, b in zip(g.geom_offsets[:-1], g.geom_offsets[1:], strict=True)
]
fords = int(g.crossing.sum())
flags = low_water_crossing_flags(lines, ways, g.bridge, g.crossing)
g = replace(
    g,
    crossing=flags,
    meta={
        **g.meta,
        "source": "GREENVILLE_BBOX",
        "bbox": list(bbox),
        "loader": "osmium tags-filter + ox.graph_from_xml",
        "n_waterways": len(ways),
    },
)
save_npz(g, f"{D}/greenville.npz")
print(
    json.dumps(
        dict(
            nodes=g.n_nodes,
            edges=g.n_edges,
            bridges=int(g.bridge.sum()),
            fords=fords,
            crossings=int(flags.sum()),
            waterways=len(ways),
            graph_s=round(t1 - t),
            total_s=round(time.time() - t),
            highway=dict(Counter(g.highway.tolist())),
            imputed_note="",
        )
    )
)
