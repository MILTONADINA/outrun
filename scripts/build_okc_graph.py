import json
import time
from dataclasses import replace

import numpy as np
import osmnx as ox

from outrun.roads.features import low_water_crossing_flags
from outrun.roads.graph import _project, build_graph, save_npz

D="/Users/milto/OutrunData/proc/roads"
bbox=(-98.10,35.10,-97.10,35.80)
def loader(place, bb):
    G=ox.graph_from_xml(f"{D}/okc_drive.osm", simplify=True, retain_all=False)
    G=ox.truncate.truncate_graph_bbox(G, bb)
    G=ox.truncate.largest_component(G, strongly=False)
    return ox.add_edge_speeds(G)
t=time.time()
g=build_graph(bbox=bbox, cache_dir=D, loader=loader, refresh=True)
t1=time.time()
print("graph",g.n_nodes,g.n_edges,int(g.bridge.sum()),int(g.crossing.sum()),round(t1-t),flush=True)
w=ox.features_from_xml(f"{D}/okc_water.osm", tags={"waterway":["river","stream","canal","drain","ditch"]})
w=w[w.geometry.geom_type.isin(["LineString","MultiLineString"])].explode(index_parts=False)
ways=[]
for geom in w.geometry:
    c=np.asarray(geom.coords); x,y=_project(c[:,0],c[:,1]); ways.append(np.c_[x,y])
lines=[np.c_[g.geom_x[a:b],g.geom_y[a:b]] for a,b in zip(g.geom_offsets[:-1],g.geom_offsets[1:])]
fords=int(g.crossing.sum())
flags=low_water_crossing_flags(lines,ways,g.bridge,g.crossing)
g=replace(g,crossing=flags,meta={**g.meta,"source":"OKC_BBOX","bbox":list(bbox),"loader":"osmium tags-filter + ox.graph_from_xml","n_waterways":len(ways)})
save_npz(g,f"{D}/okc.npz")
from collections import Counter

print(json.dumps(dict(nodes=g.n_nodes,edges=g.n_edges,bridges=int(g.bridge.sum()),fords=fords,crossings=int(flags.sum()),waterways=len(ways),graph_s=round(t1-t),total_s=round(time.time()-t),highway=dict(Counter(g.highway.tolist())),imputed_note="")))
