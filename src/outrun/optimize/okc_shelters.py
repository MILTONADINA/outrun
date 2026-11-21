"""Real OKC shelter instance (T11): PL 94-171 block groups, HIFLD shelters, travel times on the OKC road graph.

``python -m outrun.optimize.okc_shelters`` builds the instance and prints min-cost flow vs nearest vs greedy.
Static free-flow travel times (no hazard layers yet); the time-dependent version is exercised by the unit tests.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from outrun.config import get_paths
from outrun.data.census import read_pl94171_block_groups
from outrun.data.facilities import load_hifld_facilities
from outrun.optimize.shelters import (
    Facility,
    ShelterSite,
    assign_greedy,
    assign_nearest,
    assignment_metrics,
    build_shelter_supply,
    shelter_demand,
    solve_min_cost_flow,
    travel_cost_matrix,
    unserved_cost,
)
from outrun.roads.graph import _project, load_npz

OKC_BBOX = (-98.1, 35.1, -97.1, 35.8)
OKC_CENTRE = (-97.52, 35.47)


@dataclass
class OkcInstance:
    demand: np.ndarray
    capacity: np.ndarray
    cost_s: np.ndarray
    zone_xy: np.ndarray
    site_xy: np.ndarray
    sites: list[ShelterSite]
    population: np.ndarray


def _largest_scc_nodes(graph) -> np.ndarray:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = graph.n_nodes
    adj = csr_matrix((np.ones(graph.n_edges), (graph.src, graph.dst)), shape=(n, n))
    _, lab = connected_components(adj, directed=True, connection="strong")
    return np.flatnonzero(lab == np.bincount(lab).argmax())


def build_okc_instance(
    radius_km: float | None = None, raw: str | Path | None = None, min_shelters: int | None = None
) -> OkcInstance:
    """Block groups with internal point in the OKC bbox (optionally within ``radius_km`` of the centre)."""
    from scipy.spatial import cKDTree

    paths = get_paths()
    raw = Path(raw or paths.raw)
    graph = load_npz(Path(paths.proc) / "roads" / "okc.npz")
    scc = _largest_scc_nodes(graph)
    tree = cKDTree(np.c_[graph.x_m[scc], graph.y_m[scc]])

    def snap(lon, lat):
        x, y = _project(np.asarray(lon, float), np.asarray(lat, float))
        _, idx = tree.query(np.c_[x, y])
        return scc[idx], x, y

    bg = read_pl94171_block_groups(raw / "census_2020_pl94171_ok" / "ok2020.pl.zip")
    lon0, lat0, lon1, lat1 = OKC_BBOX
    bg = bg[bg["lon"].between(lon0, lon1) & bg["lat"].between(lat0, lat1) & (bg["population"] > 0)]
    if radius_km is not None:
        cx, cy = _project(np.array([OKC_CENTRE[0]]), np.array([OKC_CENTRE[1]]))
        x, y = _project(bg["lon"].to_numpy(), bg["lat"].to_numpy())
        bg = bg[np.hypot(x - cx[0], y - cy[0]) <= radius_km * 1000]
    zone_nodes, zx, zy = snap(bg["lon"], bg["lat"])

    fac = load_hifld_facilities(raw, OKC_BBOX)
    f_nodes, fx, fy = snap(fac["lon"], fac["lat"])
    facilities = [
        Facility.from_record(rec, int(n), float(x), float(y))
        for rec, n, x, y in zip(fac.to_dict("records"), f_nodes, fx, fy, strict=True)
    ]
    sites = build_shelter_supply(facilities, min_shelters=min_shelters)

    g = nx.DiGraph()
    g.add_nodes_from(range(graph.n_nodes))
    g.add_weighted_edges_from(
        zip(graph.src.tolist(), graph.dst.tolist(), graph.free_flow_s.tolist(), strict=True), weight="time"
    )  # parallel edges: keep the smallest weight
    best = {}
    for u, v, w in zip(graph.src.tolist(), graph.dst.tolist(), graph.free_flow_s.tolist(), strict=True):
        if (u, v) not in best or w < best[(u, v)]:
            best[(u, v)] = w
    g.clear_edges()
    g.add_edges_from((u, v, {"time": w}) for (u, v), w in best.items())

    cost = travel_cost_matrix(g, zone_nodes.tolist(), [s.node for s in sites])
    pop = bg["population"].to_numpy()
    return OkcInstance(
        shelter_demand(pop),
        np.array([s.capacity for s in sites], dtype=np.int64),
        cost,
        np.c_[zx, zy],
        np.array([[s.x, s.y] for s in sites]),
        sites,
        pop,
    )


def compare(inst: OkcInstance) -> list[dict]:
    rows = []
    u = unserved_cost(inst.cost_s)
    for name, fn in (
        ("mcf", lambda: solve_min_cost_flow(inst.demand, inst.capacity, inst.cost_s)),
        (
            "nearest",
            lambda: assign_nearest(inst.demand, inst.capacity, inst.zone_xy, inst.site_xy, inst.cost_s),
        ),
        ("greedy", lambda: assign_greedy(inst.demand, inst.capacity, inst.cost_s)),
    ):
        t = time.perf_counter()
        a = fn()
        dt = time.perf_counter() - t
        m = assignment_metrics(a, inst.capacity, inst.cost_s)
        used = a.flow > 0
        obj = float((a.flow[used] * np.rint(inst.cost_s[used])).sum() + u * a.unserved.sum())
        rows.append({"method": name, "runtime_s": dt, "objective": obj, **m.__dict__})
    return rows


if __name__ == "__main__":
    for label, radius, min_sh in (
        ("metro bbox", None, None),
        ("10 km of centre", 10.0, None),
        ("metro bbox + assumed schools/fire stations", None, 10_000),
    ):
        t = time.perf_counter()
        inst = build_okc_instance(radius, min_shelters=min_sh)
        print(
            f"\n== {label}: {len(inst.demand)} zones, pop {inst.population.sum()}, demand {inst.demand.sum()}, "
            f"{len(inst.sites)} sites ({sum(s.assumed for s in inst.sites)} assumed), capacity {inst.capacity.sum()}, "
            f"unreachable cells {(~np.isfinite(inst.cost_s)).sum()}, build {time.perf_counter() - t:.1f}s"
        )
        for r in compare(inst):
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
