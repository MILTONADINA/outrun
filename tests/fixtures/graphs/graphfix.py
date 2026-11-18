"""Synthetic road graphs for routing tests (offline, numpy only).

``make_road_graph`` builds an ``outrun.roads.graph.RoadGraph`` directly (CSR computed here with numpy), so the
routing tests do not depend on the ROAD-01 builder. ``outrun`` is imported lazily inside the functions.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

R_EARTH = 6_371_008.8
OKC_LON, OKC_LAT = -97.52, 35.47


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def make_road_graph(lon, lat, src, dst, length_m, speed_mps, x_m=None, y_m=None):
    """RoadGraph from edge lists; edges re-ordered by a stable sort on src. Returns (graph, order)."""
    from outrun.roads.graph import RoadGraph

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    n = len(lon)
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    length = np.asarray(length_m, dtype=float)[order]
    speed = np.asarray(speed_mps, dtype=float)[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    indptr = np.cumsum(indptr)
    if x_m is None:  # local equirectangular metres (good enough for synthetic tests)
        x_m = (lon - OKC_LON) * 111_320.0 * math.cos(math.radians(OKC_LAT))
        y_m = (lat - OKC_LAT) * 111_320.0
    e = len(src)
    g = RoadGraph(
        lon=lon,
        lat=lat,
        x_m=np.asarray(x_m, dtype=float),
        y_m=np.asarray(y_m, dtype=float),
        indptr=indptr,
        src=src,
        dst=dst,
        length_m=length,
        speed_mps=speed,
        free_flow_s=length / speed,
        lanes=np.ones(e, dtype=np.int64),
        capacity_vph=np.full(e, 600.0),
        highway=np.array(["other"] * e),
        bridge=np.zeros(e, dtype=bool),
        crossing=np.zeros(e, dtype=bool),
        osm_node_id=np.arange(n, dtype=np.int64),
    )
    return g, order


def lattice(rows=5, cols=6, spacing_m=1000.0, speed_mps=15.0):
    """Bidirectional lattice (node id = r*cols + c). Edge lengths = spacing (>= great-circle span)."""
    dlat = spacing_m / 111_320.0
    dlon = spacing_m / (111_320.0 * math.cos(math.radians(OKC_LAT)))
    lon = np.array([OKC_LON + c * dlon for r in range(rows) for c in range(cols)])
    lat = np.array([OKC_LAT + r * dlat for r in range(rows) for c in range(cols)])
    src, dst = [], []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            for dr, dc in ((0, 1), (1, 0)):
                rr, cc = r + dr, c + dc
                if rr < rows and cc < cols:
                    v = rr * cols + cc
                    src += [u, v]
                    dst += [v, u]
    length = np.full(len(src), spacing_m)
    return make_road_graph(lon, lat, src, dst, length, np.full(len(src), speed_mps))[0]


def random_graph(n, k, seed, speed_choices=(10.0, 15.0, 20.0, 30.0), integer_minutes=False):
    """Random strongly-connected-ish directed graph: ring + k nearest neighbours, lengths >= great-circle span.

    With ``integer_minutes`` every free-flow time is a whole number of minutes (lengths are set from the time).
    """
    rng = np.random.default_rng(seed)
    lon = OKC_LON + rng.uniform(0, 0.08, n)
    lat = OKC_LAT + rng.uniform(0, 0.06, n)
    pairs = set()
    for i in range(n):  # ring guarantees strong connectivity
        pairs.add((i, (i + 1) % n))
        pairs.add(((i + 1) % n, i))
    d = haversine(lon[:, None], lat[:, None], lon[None, :], lat[None, :])
    for i in range(n):
        for j in np.argsort(d[i])[1 : k + 1]:
            pairs.add((i, int(j)))
            if rng.random() < 0.7:
                pairs.add((int(j), i))
    pairs = sorted(pairs)
    src = np.array([p[0] for p in pairs])
    dst = np.array([p[1] for p in pairs])
    span = haversine(lon[src], lat[src], lon[dst], lat[dst])
    speed = rng.choice(np.array(speed_choices), len(src))
    length = span * rng.uniform(1.0, 1.5, len(src)) + 1.0
    if integer_minutes:
        minutes = np.maximum(1, np.ceil(length / speed / 60.0))
        length = minutes * 60.0 * speed
    return make_road_graph(lon, lat, src, dst, length, speed)[0]


def dijkstra_costs(n, src, dst, w, origin):
    """Reference single-source Dijkstra over arrays (inf weights skipped)."""
    adj = [[] for _ in range(n)]
    for s, t, c in zip(src, dst, w, strict=True):
        if math.isfinite(c):
            adj[int(s)].append((int(t), float(c)))
    dist = [math.inf] * n
    dist[origin] = 0.0
    pq = [(0.0, origin)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, c in adj[u]:
            if d + c < dist[v]:
                dist[v] = d + c
                heapq.heappush(pq, (d + c, v))
    return dist


def brute_force_td(graph, cost_model, origin, goal, step_s=60.0, horizon_s=None):
    """Independent time-expanded Dijkstra over (node, minute) states, using only CostModel.clock/cost/wait_cost.

    Requires every clock time to be a multiple of ``step_s``. Returns min cost to goal (inf if unreachable).
    """
    if horizon_s is None:
        horizon_s = cost_model.n_slots * cost_model.slot_s + float(np.sum(graph.free_flow_s)) * 3.0
    kmax = int(math.ceil(horizon_s / step_s))
    best = {(origin, 0): 0.0}
    pq = [(0.0, origin, 0)]
    while pq:
        g, u, k = heapq.heappop(pq)
        if best.get((u, k), math.inf) < g:
            continue
        if u == goal:
            return g
        t = k * step_s
        moves = []
        if k + 1 <= kmax:
            moves.append((u, k + 1, cost_model.wait_cost(u, t, t + step_s)))
        for e in range(int(graph.indptr[u]), int(graph.indptr[u + 1])):
            c = cost_model.cost(e, t)
            if not math.isfinite(c):
                continue
            kk = k + int(round(cost_model.clock(e, t) / step_s))
            if kk <= kmax:
                moves.append((int(graph.dst[e]), kk, c))
        for v, kk, c in moves:
            ng = g + c
            if ng < best.get((v, kk), math.inf) - 1e-12:
                best[(v, kk)] = ng
                heapq.heappush(pq, (ng, v, kk))
    return math.inf
