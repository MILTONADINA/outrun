"""Spec 03-roads (ROAD-*). Tiny synthetic graphs only; no OSM download, no network."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest
from tests.fixtures.graphs.graphfix import OKC_LAT, OKC_LON, haversine, lattice


def _edges():
    # 4 nodes; edges deliberately not sorted by src
    lon = np.array([OKC_LON, OKC_LON + 0.01, OKC_LON + 0.02, OKC_LON + 0.01])
    lat = np.array([OKC_LAT, OKC_LAT, OKC_LAT, OKC_LAT + 0.01])
    src = np.array([2, 0, 1, 0, 3, 1])
    dst = np.array([1, 1, 2, 3, 0, 0])
    length = np.array([900.0, 910.0, 920.0, 1200.0, 1210.0, 930.0])
    speed = np.array([10.0, 20.0, 10.0, 15.0, 30.0, 20.0])
    return lon, lat, src, dst, length, speed


def _osm_graph():
    """OSMnx-style MultiDiGraph with OSM ids, parallel edges, a self-loop and missing tags."""
    G = nx.MultiDiGraph()
    G.add_node(500, x=OKC_LON, y=OKC_LAT)
    G.add_node(100, x=OKC_LON + 0.01, y=OKC_LAT)
    G.add_node(300, x=OKC_LON + 0.01, y=OKC_LAT + 0.01)
    G.add_edge(500, 100, 0, length=1000.0, speed_kph=36.0, highway="primary", lanes="2", bridge="yes")
    G.add_edge(500, 100, 1, length=1000.0, speed_kph=72.0, highway="primary", lanes="3")  # faster parallel
    G.add_edge(100, 500, 0, length=1000.0, speed_kph=54.0, highway="primary")
    G.add_edge(100, 300, 0, length=1500.0, highway="residential", ford="yes")  # no speed -> imputed
    G.add_edge(300, 100, 0, length=1500.0, speed_kph=30.0, highway=["residential", "service"])
    G.add_edge(300, 300, 0, length=50.0, speed_kph=30.0, highway="service")  # self-loop dropped
    return G


# ---------------------------------------------------------------- ROAD-01
@pytest.mark.req("ROAD-01")
def test_from_edge_arrays_builds_csr():
    from outrun.roads.graph import from_edge_arrays

    lon, lat, src, dst, length, speed = _edges()
    g, order = from_edge_arrays(lon, lat, src, dst, length, speed)
    assert g.n_nodes == 4 and g.n_edges == 6
    assert np.all(np.diff(g.src) >= 0)
    assert g.indptr.tolist() == [0, 2, 4, 5, 6]
    np.testing.assert_array_equal(g.src, src[order])
    np.testing.assert_array_equal(g.dst, dst[order])
    np.testing.assert_allclose(g.length_m, length[order])
    np.testing.assert_allclose(g.free_flow_s, length[order] / speed[order])
    # stable sort: the two edges from node 0 keep input order (input idx 1 then 3)
    assert order[:2].tolist() == [1, 3]
    for u in range(4):
        assert all(g.src[e] == u for e in g.out_edges(u))
    assert g.lanes.tolist() == [1] * 6
    assert not g.bridge.any() and not g.crossing.any()
    assert set(g.highway.tolist()) == {"other"}


@pytest.mark.req("ROAD-01")
@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.update(src=a["src"][:-1]),  # length mismatch
        lambda a: a.update(dst=np.array([1, 1, 2, 3, 0, 4])),  # node out of range
        lambda a: a.update(src=np.array([2, 0, 1, 0, 3, -1])),  # negative node
        lambda a: a.update(length=np.array([900.0, 0.0, 920.0, 1200.0, 1210.0, 930.0])),
        lambda a: a.update(speed=np.array([10.0, 20.0, -1.0, 15.0, 30.0, 20.0])),
        lambda a: a.update(speed=np.array([10.0, 20.0, np.nan, 15.0, 30.0, 20.0])),
        lambda a: a.update(lon=np.array([]), lat=np.array([])),  # no nodes
    ],
)
def test_from_edge_arrays_rejects_bad_input(mutate):
    from outrun.roads.graph import from_edge_arrays

    lon, lat, src, dst, length, speed = _edges()
    a = dict(lon=lon, lat=lat, src=src, dst=dst, length=length, speed=speed)
    mutate(a)
    with pytest.raises(ValueError):
        from_edge_arrays(a["lon"], a["lat"], a["src"], a["dst"], a["length"], a["speed"])


@pytest.mark.req("ROAD-01")
def test_from_edge_arrays_projects_to_epsg5070():
    pyproj = pytest.importorskip("pyproj")
    from outrun.roads.graph import from_edge_arrays

    lon, lat, src, dst, length, speed = _edges()
    g, _ = from_edge_arrays(lon, lat, src, dst, length, speed)
    tr = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x, y = tr.transform(lon, lat)
    np.testing.assert_allclose(g.x_m, x, atol=0.01)
    np.testing.assert_allclose(g.y_m, y, atol=0.01)


# ---------------------------------------------------------------- ROAD-02
@pytest.mark.req("ROAD-02")
def test_from_networkx_nodes_edges_and_parallel_edges():
    from outrun.roads.graph import from_networkx

    g = from_networkx(_osm_graph(), capacity_table={"primary": 900, "other": 600})
    assert g.osm_node_id.tolist() == [100, 300, 500]  # ascending OSM id
    np.testing.assert_allclose(g.lon, [OKC_LON + 0.01, OKC_LON + 0.01, OKC_LON])
    assert g.n_edges == 4  # parallel pair collapsed, self-loop dropped
    pairs = {
        (int(g.osm_node_id[s]), int(g.osm_node_id[d])): e
        for e, (s, d) in enumerate(zip(g.src, g.dst, strict=True))
    }
    assert set(pairs) == {(500, 100), (100, 500), (100, 300), (300, 100)}
    e = pairs[(500, 100)]  # kept the faster parallel edge (72 km/h, 3 lanes, no bridge)
    assert g.speed_mps[e] == pytest.approx(20.0)
    assert g.free_flow_s[e] == pytest.approx(50.0)
    assert g.lanes[e] == 3 and not g.bridge[e]
    assert g.capacity_vph[e] == pytest.approx(2700.0)
    assert g.highway[pairs[(300, 100)]] == "residential"
    assert g.crossing[pairs[(100, 300)]] and not g.crossing[pairs[(300, 100)]]
    assert np.all(np.diff(g.src) >= 0)


@pytest.mark.req("ROAD-02")
def test_from_networkx_requires_length():
    from outrun.roads.graph import from_networkx

    G = _osm_graph()
    G.add_edge(100, 500, 1, speed_kph=50.0, highway="primary")  # no length
    with pytest.raises(ValueError):
        from_networkx(G)


# ---------------------------------------------------------------- ROAD-03
@pytest.mark.req("ROAD-03")
@pytest.mark.parametrize(
    "value,expected",
    [
        ("primary", "primary"),
        ("motorway_link", "motorway"),
        (["secondary", "tertiary"], "secondary"),
        ("track", "other"),
        (None, "other"),
        (float("nan"), "other"),
    ],
)
def test_normalize_highway(value, expected):
    from outrun.roads.graph import normalize_highway

    assert normalize_highway(value) == expected


@pytest.mark.req("ROAD-03")
def test_impute_speeds_class_median_then_default():
    from outrun.roads.graph import DEFAULT_SPEED_KPH, impute_speeds

    highway = np.array(["residential", "residential", "residential", "motorway", "primary"])
    kph = np.array([30.0, 50.0, np.nan, np.nan, 0.0])
    mps = impute_speeds(highway, kph)
    np.testing.assert_allclose(
        mps,
        np.array([30.0, 50.0, 40.0, DEFAULT_SPEED_KPH["motorway"], DEFAULT_SPEED_KPH["primary"]]) / 3.6,
    )


@pytest.mark.req("ROAD-03")
def test_edge_capacity_table_and_lanes():
    from outrun.roads.graph import edge_capacity_vph

    table = {"motorway": 1800, "trunk": 1800, "primary": 900, "secondary": 900, "other": 600}
    hw = np.array(["motorway", "primary", "residential", "trunk", "secondary"])
    lanes = np.array([3, 2, 1, 0, np.nan])
    np.testing.assert_allclose(edge_capacity_vph(hw, lanes, table), [5400, 1800, 600, 1800, 900])


@pytest.mark.req("ROAD-03")
def test_edge_capacity_default_from_config():
    yaml = pytest.importorskip("yaml")
    from pathlib import Path

    from outrun.roads.graph import edge_capacity_vph

    cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs" / "planning.yaml").read_text())
    table = cfg["roads"]["capacity_veh_per_h_per_lane"]
    hw = np.array(["motorway", "primary", "residential"])
    out = edge_capacity_vph(hw, np.array([1, 1, 1]))
    np.testing.assert_allclose(out, [table["motorway"], table["primary"], table["other"]])


# ---------------------------------------------------------------- ROAD-04
@pytest.mark.req("ROAD-04")
def test_npz_round_trip(tmp_path):
    from outrun.roads.graph import from_networkx, load_npz, save_npz

    g = from_networkx(_osm_graph())
    p = save_npz(g, tmp_path / "deep" / "dir" / "g.npz")
    assert p.exists()
    h = load_npz(p)
    for name in (
        "lon",
        "lat",
        "x_m",
        "y_m",
        "indptr",
        "src",
        "dst",
        "length_m",
        "speed_mps",
        "free_flow_s",
        "lanes",
        "capacity_vph",
        "highway",
        "bridge",
        "crossing",
        "osm_node_id",
    ):
        np.testing.assert_array_equal(getattr(h, name), getattr(g, name), err_msg=name)


@pytest.mark.req("ROAD-04")
def test_npz_errors(tmp_path):
    from outrun.roads.graph import NPZ_FORMAT_VERSION, from_networkx, load_npz, save_npz

    with pytest.raises(FileNotFoundError):
        load_npz(tmp_path / "missing.npz")
    p = save_npz(from_networkx(_osm_graph()), tmp_path / "g.npz")
    data = dict(np.load(p, allow_pickle=False))
    assert int(data["format_version"]) == NPZ_FORMAT_VERSION
    data["format_version"] = np.array(NPZ_FORMAT_VERSION + 99)
    bad = tmp_path / "bad.npz"
    np.savez(bad, **data)
    with pytest.raises(ValueError):
        load_npz(bad)


# ---------------------------------------------------------------- ROAD-05
@pytest.mark.req("ROAD-05")
def test_haversine_known_distance():
    from outrun.roads.features import haversine_m

    # one degree of latitude on the 6,371,008.8 m sphere
    assert float(haversine_m(0.0, 0.0, 0.0, 1.0)) == pytest.approx(math.radians(1) * 6_371_008.8, rel=1e-9)
    d = haversine_m(np.array([OKC_LON, OKC_LON]), np.array([OKC_LAT, OKC_LAT]), OKC_LON + 0.01, OKC_LAT)
    assert d.shape == (2,) and d[0] == pytest.approx(d[1])


@pytest.mark.req("ROAD-05")
def test_snap_to_nodes_nearest_and_ties():
    from outrun.roads.features import snap_to_nodes

    g = lattice(3, 3, spacing_m=1000.0)
    lon = np.array([g.lon[4] + 1e-5, g.lon[8], (g.lon[0] + g.lon[1]) / 2])
    lat = np.array([g.lat[4] + 1e-5, g.lat[8] + 1e-4, g.lat[0]])
    idx, dist = snap_to_nodes(g, lon, lat)
    assert idx.tolist() == [4, 8, 0]  # third point is equidistant from 0 and 1 -> lowest index
    np.testing.assert_allclose(dist, haversine(lon, lat, g.lon[idx], g.lat[idx]), rtol=1e-6, atol=1e-6)
    e_idx, e_dist = snap_to_nodes(g, np.array([]), np.array([]))
    assert e_idx.shape == (0,) and e_dist.shape == (0,)
    with pytest.raises(ValueError):
        snap_to_nodes(g, np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------- ROAD-06
class _Loader:
    def __init__(self, graph):
        self.graph, self.calls = graph, []

    def __call__(self, place, bbox):
        self.calls.append((place, bbox))
        return self.graph


@pytest.mark.req("ROAD-06")
@pytest.mark.parametrize(
    "place,key",
    [("Tulsa, Oklahoma, USA", "tulsa-oklahoma-usa"), ("  Paradise, CA!! ", "paradise-ca"), ("Ada", "ada")],
)
def test_cache_key_place(place, key):
    from outrun.roads.graph import cache_key

    assert cache_key(place=place) == key


@pytest.mark.req("ROAD-06")
def test_cache_key_bbox_and_errors():
    from outrun.roads.graph import cache_key

    assert cache_key(bbox=(-97.6, 35.3, -97.4, 35.55)) == "bbox_-97.6000_35.3000_-97.4000_35.5500"
    with pytest.raises(ValueError):
        cache_key()
    with pytest.raises(ValueError):
        cache_key(place="Tulsa", bbox=(-97.6, 35.3, -97.4, 35.55))


@pytest.mark.req("ROAD-06")
def test_build_graph_any_place_caches(tmp_path, no_network):
    from outrun.roads.graph import build_graph

    loader = _Loader(_osm_graph())
    g1 = build_graph("Tulsa, Oklahoma, USA", cache_dir=tmp_path, loader=loader)
    assert loader.calls == [("Tulsa, Oklahoma, USA", None)]
    assert (tmp_path / "tulsa-oklahoma-usa.npz").exists()
    assert g1.n_edges == 4 and g1.meta["source"] == "Tulsa, Oklahoma, USA"
    g2 = build_graph("Tulsa, Oklahoma, USA", cache_dir=tmp_path, loader=loader)
    assert len(loader.calls) == 1  # served from cache
    np.testing.assert_array_equal(g2.dst, g1.dst)
    build_graph("Tulsa, Oklahoma, USA", cache_dir=tmp_path, loader=loader, refresh=True)
    assert len(loader.calls) == 2


@pytest.mark.req("ROAD-06")
def test_build_graph_bbox_and_errors(tmp_path):
    from outrun.roads.graph import build_graph

    loader = _Loader(_osm_graph())
    bbox = (-97.6, 35.3, -97.4, 35.55)
    build_graph(bbox=bbox, cache_dir=tmp_path, loader=loader)
    assert loader.calls == [(None, bbox)]
    assert (tmp_path / "bbox_-97.6000_35.3000_-97.4000_35.5500.npz").exists()
    for kwargs in (
        {},
        {"place": "Tulsa", "bbox": bbox},
        {"bbox": (-97.4, 35.3, -97.6, 35.55)},  # west >= east
        {"bbox": (-97.6, 35.6, -97.4, 35.55)},  # south >= north
        {"bbox": (-97.6, 35.3, -97.4, 95.0)},  # latitude out of range
    ):
        with pytest.raises(ValueError):
            build_graph(cache_dir=tmp_path / "x", loader=loader, **kwargs)
    empty = nx.MultiDiGraph()
    empty.add_node(1, x=OKC_LON, y=OKC_LAT)
    with pytest.raises(ValueError):
        build_graph("Nowhere", cache_dir=tmp_path, loader=_Loader(empty))


# ---------------------------------------------------------------- ROAD-07
@pytest.mark.req("ROAD-07")
def test_low_water_crossing_proxy():
    from outrun.roads.features import low_water_crossing_flags

    edges = [
        np.array([[0.0, 0.0], [100.0, 0.0]]),  # crosses the creek, no bridge -> crossing
        np.array([[0.0, 10.0], [100.0, 10.0]]),  # crosses the creek on a bridge -> not
        np.array([[0.0, 500.0], [100.0, 500.0]]),  # far from water -> not
        np.array([[0.0, 900.0], [100.0, 900.0]]),  # ford tag -> crossing
        np.array([[0.0, -50.0], [50.0, -50.0], [50.0, -20.0]]),  # polyline touching creek end-point
    ]
    creek = [np.array([[50.0, -20.0], [50.0, 100.0]])]
    bridge = np.array([False, True, False, False, False])
    ford = np.array([False, False, False, True, False])
    out = low_water_crossing_flags(edges, creek, bridge, ford)
    assert out.dtype == bool
    assert out.tolist() == [True, False, False, True, True]
    assert low_water_crossing_flags(edges, [], bridge, ford).tolist() == [False, False, False, True, False]
    with pytest.raises(ValueError):
        low_water_crossing_flags(edges, creek, bridge[:2], ford)


# ---------------------------------------------------------------- ROAD-08
@pytest.mark.req("ROAD-08")
@pytest.mark.parametrize(
    "value,expected",
    [
        ("yes", True),
        ("viaduct", True),
        ("movable", True),
        (True, True),
        ("1", True),
        (["no", "yes"], True),
        ("no", False),
        ("No", False),
        ("0", False),
        ("", False),
        (None, False),
        (False, False),
        (float("nan"), False),
        (["no"], False),
    ],
)
def test_parse_bridge(value, expected):
    from outrun.roads.features import parse_bridge

    assert parse_bridge(value) is expected


# ---------------------------------------------------------------- ROAD-09
@pytest.mark.req("ROAD-09")
def test_reverse_csr():
    g = lattice(3, 4)
    in_indptr, in_edges = g.reverse_csr()
    assert in_indptr.shape == (g.n_nodes + 1,) and in_edges.shape == (g.n_edges,)
    assert sorted(in_edges.tolist()) == list(range(g.n_edges))
    for v in range(g.n_nodes):
        es = in_edges[in_indptr[v] : in_indptr[v + 1]]
        assert es.tolist() == sorted(es.tolist())
        assert set(es.tolist()) == {e for e in range(g.n_edges) if g.dst[e] == v}
