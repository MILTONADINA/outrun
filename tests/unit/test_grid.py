"""specs/02-grid-risk.md: GRID-01..08 (the shared 1 km EPSG:5070 grid, windows and regridding)."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from pyproj import Geod, Transformer

from outrun.core.grid import (
    GRID,
    GridWindow,
    aggregate_to_grid,
    bbox_to_window,
    domain_window,
    lonlat_to_xy,
    polar_to_grid,
    polar_to_xy,
    regrid_nearest,
    xy_to_lonlat,
)

ROOT = Path(__file__).resolve().parents[2]
FWD = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
INV = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
OKC_BBOX = (-98.10, 35.10, -97.10, 35.80)


def _win32() -> GridWindow:
    return GridWindow(x0=-150_000.0, y0=1_300_000.0, nx=32, ny=32, res=1000.0)


def _densified_boundary(bbox, n=200):
    lon0, lat0, lon1, lat1 = bbox
    t = np.linspace(0.0, 1.0, n)
    lon = np.concatenate(
        [lon0 + (lon1 - lon0) * t, np.full(n, lon1), lon1 - (lon1 - lon0) * t, np.full(n, lon0)]
    )
    lat = np.concatenate(
        [np.full(n, lat0), lat0 + (lat1 - lat0) * t, np.full(n, lat1), lat1 - (lat1 - lat0) * t]
    )
    return FWD.transform(lon, lat)


# ---------------------------------------------------------------- GRID-01
@pytest.mark.req("GRID-01")
def test_grid_constants_match_config():
    cfg = yaml.safe_load((ROOT / "configs" / "domains.yaml").read_text())
    assert GRID.crs == "EPSG:5070" == cfg["grid"]["crs"]
    assert GRID.resolution_m == 1000.0 == float(cfg["grid"]["resolution_m"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        GRID.resolution_m = 500.0  # type: ignore[misc]


@pytest.mark.req("GRID-01")
@pytest.mark.parametrize("kw", [{"nx": 0}, {"ny": 0}, {"res": 0.0}, {"res": -1.0}])
def test_grid_window_rejects_bad_sizes(kw):
    args = {"x0": 0.0, "y0": 0.0, "nx": 4, "ny": 4, "res": 1000.0} | kw
    with pytest.raises(ValueError):
        GridWindow(**args)


# ---------------------------------------------------------------- GRID-02
@pytest.mark.req("GRID-02")
def test_lonlat_to_xy_matches_pyproj(rng):
    lon = rng.uniform(-103, -94, size=(5, 7))
    lat = rng.uniform(33, 37, size=(5, 7))
    x, y = lonlat_to_xy(lon, lat)
    ex, ey = FWD.transform(lon, lat)
    assert np.asarray(x).shape == (5, 7)
    np.testing.assert_allclose(x, ex, atol=1e-3)
    np.testing.assert_allclose(y, ey, atol=1e-3)
    lo, la = xy_to_lonlat(x, y)
    np.testing.assert_allclose(lo, lon, atol=1e-7)
    np.testing.assert_allclose(la, lat, atol=1e-7)


@pytest.mark.req("GRID-02")
def test_lonlat_to_xy_scalar():
    x, y = lonlat_to_xy(-97.52, 35.47)
    ex, ey = FWD.transform(-97.52, 35.47)
    assert float(x) == pytest.approx(ex, abs=1e-3)
    assert float(y) == pytest.approx(ey, abs=1e-3)


# ---------------------------------------------------------------- GRID-03
@pytest.mark.req("GRID-03")
@pytest.mark.parametrize("buffer_km", [0.0, 50.0])
def test_bbox_to_window_snapped_covering_minimal(buffer_km):
    w = bbox_to_window(OKC_BBOX, buffer_km=buffer_km)
    res = 1000.0
    assert w.res == res and w.crs == "EPSG:5070"
    assert w.x0 % res == 0 and w.y0 % res == 0
    bx, by = _densified_boundary(OKC_BBOX)
    b = buffer_km * 1000.0
    xmin, xmax, ymin, ymax = bx.min() - b, bx.max() + b, by.min() - b, by.max() + b
    wxmin, wymax = w.x0, w.y0
    wxmax, wymin = w.x0 + w.nx * res, w.y0 - w.ny * res
    # covers (1 cm slack: the implementation may densify the boundary differently)
    assert wxmin <= xmin + 0.01 and wxmax >= xmax - 0.01 and wymin <= ymin + 0.01 and wymax >= ymax - 0.01
    # minimal (densified boundary is a slight inner approximation, allow 1 m)
    assert xmin - wxmin < res + 1 and wxmax - xmax < res + 1
    assert ymin - wymin < res + 1 and wymax - ymax < res + 1


@pytest.mark.req("GRID-03")
def test_bbox_to_window_okc_size_plausible():
    w = bbox_to_window(OKC_BBOX)
    assert 88 <= w.nx <= 100
    assert 76 <= w.ny <= 86


@pytest.mark.req("GRID-03")
@pytest.mark.parametrize(
    "bbox,buf",
    [
        ((-97.0, 35.0, -98.0, 36.0), 0.0),  # lon_min >= lon_max
        ((-98.0, 36.0, -97.0, 35.0), 0.0),  # lat_min >= lat_max
        ((-190.0, 35.0, -97.0, 36.0), 0.0),
        ((-98.0, 35.0, -97.0, 95.0), 0.0),
        ((-98.0, 35.0, -97.0, 36.0), -1.0),
    ],
)
def test_bbox_to_window_rejects_invalid(bbox, buf):
    with pytest.raises(ValueError):
        bbox_to_window(bbox, buffer_km=buf)


# ---------------------------------------------------------------- GRID-04
@pytest.mark.req("GRID-04")
def test_window_shape_bounds_centres():
    w = GridWindow(x0=-150_000.0, y0=1_300_000.0, nx=32, ny=20, res=1000.0)
    assert w.shape == (20, 32)
    assert w.bounds == (-150_000.0, 1_280_000.0, -118_000.0, 1_300_000.0)
    xc, yc = w.x_centers(), w.y_centers()
    assert xc.shape == (32,) and yc.shape == (20,)
    assert xc[0] == -149_500.0 and xc[-1] == -118_500.0
    assert yc[0] == 1_299_500.0 and yc[-1] == 1_280_500.0
    assert np.all(np.diff(xc) > 0) and np.all(np.diff(yc) < 0)


@pytest.mark.req("GRID-04")
def test_cell_of_centres_and_edges():
    w = _win32()
    xc, yc = w.x_centers(), w.y_centers()
    X, Y = np.meshgrid(xc, yc)
    r, c = w.cell_of(X, Y)
    assert r.shape == (32, 32)
    np.testing.assert_array_equal(r, np.repeat(np.arange(32)[:, None], 32, axis=1))
    np.testing.assert_array_equal(c, np.repeat(np.arange(32)[None, :], 32, axis=0))
    # west / north edges inside, east / south edges outside
    r, c = w.cell_of(
        np.array([-150_000.0, -118_000.0, -149_000.0, -149_000.0]),
        np.array([1_299_000.0, 1_299_000.0, 1_300_000.0, 1_268_000.0]),
    )
    assert (int(r[0]), int(c[0])) == (1, 0)
    assert (int(r[1]), int(c[1])) == (-1, -1)
    assert (int(r[2]), int(c[2])) == (0, 1)
    assert (int(r[3]), int(c[3])) == (-1, -1)
    assert r.dtype == np.int64 and c.dtype == np.int64
    np.testing.assert_array_equal(
        w.contains(np.array([-149_999.0, -200_000.0]), np.array([1_299_999.0, 0.0])), [True, False]
    )


# ---------------------------------------------------------------- GRID-05
def _okc_window(n=10):
    x, y = FWD.transform(-97.52, 35.47)
    return GridWindow(x0=math.floor(x / 1000) * 1000 - 5000, y0=math.ceil(y / 1000) * 1000 + 5000, nx=n, ny=n)


def _brute_nearest(values, lon2, lat2, w, max_dist, fill):
    sx, sy = FWD.transform(lon2.ravel(), lat2.ravel())
    X, Y = np.meshgrid(w.x_centers(), w.y_centers())
    d2 = (X.ravel()[:, None] - sx[None, :]) ** 2 + (Y.ravel()[:, None] - sy[None, :]) ** 2
    idx = d2.argmin(axis=1)
    dist = np.sqrt(d2[np.arange(len(idx)), idx])
    out = values.ravel()[idx].astype(np.float32)
    out[dist > max_dist] = fill
    return out.reshape(w.ny, w.nx)


@pytest.mark.req("GRID-05")
def test_regrid_nearest_1d_lonlat_matches_brute_force(rng):
    lon = np.round(np.arange(-97.60, -97.495, 0.01), 4)
    lat = np.round(np.arange(35.40, 35.505, 0.01), 4)
    values = rng.random((len(lat), len(lon))).astype(np.float32)
    w = _okc_window()
    out = regrid_nearest(values, lon, lat, w, max_dist_m=1000.0)
    lon2, lat2 = np.meshgrid(lon, lat)
    expected = _brute_nearest(values, lon2, lat2, w, 1000.0, np.nan)
    assert out.shape == (10, 10) and out.dtype == np.float32
    np.testing.assert_array_equal(out, expected)
    assert np.isnan(out).any() and (~np.isnan(out)).any()  # partial coverage exercised


@pytest.mark.req("GRID-05")
def test_regrid_nearest_2d_lonlat_and_fill(rng):
    lon = np.arange(-97.60, -97.495, 0.01)
    lat = np.arange(35.40, 35.505, 0.01)
    lon2, lat2 = np.meshgrid(lon, lat)
    values = rng.random(lon2.shape).astype(np.float32)
    w = _okc_window()
    out = regrid_nearest(values, lon2, lat2, w, max_dist_m=1000.0, fill=-1.0)
    np.testing.assert_array_equal(out, _brute_nearest(values, lon2, lat2, w, 1000.0, -1.0))
    with pytest.raises(ValueError):
        regrid_nearest(values[:, :-1], lon2, lat2, w, max_dist_m=1000.0)


# ---------------------------------------------------------------- GRID-06
@pytest.mark.req("GRID-06")
def test_aggregate_to_grid_all_aggs():
    w = _win32()
    xc, yc = w.x_centers(), w.y_centers()
    x = np.array([xc[0] - 200, xc[0] + 300, xc[2], -10_000_000.0, xc[3]])
    y = np.array([yc[0], yc[0] - 100, yc[1], 0.0, yc[3]])
    v = np.array([0.2, 0.6, 0.5, 0.9, np.nan])
    mx = aggregate_to_grid(v, x, y, w, agg="max")
    assert mx.shape == (32, 32) and mx.dtype == np.float32
    assert mx[0, 0] == pytest.approx(0.6) and mx[1, 2] == pytest.approx(0.5)
    assert mx[3, 3] == 0.0 and mx.sum() == pytest.approx(1.1)
    assert aggregate_to_grid(v, x, y, w, agg="mean")[0, 0] == pytest.approx(0.4)
    assert aggregate_to_grid(v, x, y, w, agg="sum")[0, 0] == pytest.approx(0.8)
    cnt = aggregate_to_grid(v, x, y, w, agg="count", fill=-5.0)
    assert cnt[0, 0] == 2 and cnt[1, 2] == 1 and cnt[3, 3] == 0 and cnt[10, 10] == 0
    filled = aggregate_to_grid(v, x, y, w, agg="max", fill=-1.0)
    assert filled[3, 3] == -1.0 and filled[10, 10] == -1.0 and filled[0, 0] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        aggregate_to_grid(v, x, y, w, agg="median")


# ---------------------------------------------------------------- GRID-07
@pytest.mark.req("GRID-07")
def test_polar_to_xy_matches_geodesic():
    rlon, rlat = -97.278, 35.333  # KTLX
    ranges = np.array([1_000.0, 10_000.0, 50_000.0, 150_000.0])
    az = np.array([0.0, 45.0, 90.0, 200.0, 315.0])
    x, y = polar_to_xy(ranges, az, rlon, rlat)
    assert x.shape == (5, 4) and y.shape == (5, 4)
    geod = Geod(ellps="WGS84")
    A, R = np.meshgrid(az, ranges, indexing="ij")
    lon, lat, _ = geod.fwd(np.full(A.shape, rlon), np.full(A.shape, rlat), A, R)
    ex, ey = FWD.transform(lon, lat)
    np.testing.assert_allclose(x, ex, atol=25.0)
    np.testing.assert_allclose(y, ey, atol=25.0)


@pytest.mark.req("GRID-07")
def test_polar_to_grid_equals_aggregate(rng):
    rlon, rlat = -97.52, 35.47
    rx, ry = FWD.transform(rlon, rlat)
    w = GridWindow(
        x0=math.floor(rx / 1000) * 1000 - 16_000, y0=math.ceil(ry / 1000) * 1000 + 16_000, nx=32, ny=32
    )
    ranges = np.arange(250.0, 20_000.0, 500.0)
    az = np.arange(0.0, 360.0, 2.0)
    vals = rng.random((len(az), len(ranges)))
    out = polar_to_grid(vals, ranges, az, rlon, rlat, w, agg="max")
    px, py = polar_to_xy(ranges, az, rlon, rlat)
    np.testing.assert_array_equal(out, aggregate_to_grid(vals, px, py, w, agg="max"))


# ---------------------------------------------------------------- GRID-08
@pytest.mark.req("GRID-08")
def test_domain_window_from_mapping_and_buffer():
    domains = {
        "OKC_BBOX": {"lon": [-98.10, -97.10], "lat": [35.10, 35.80]},
        "OK_BBOX": {"lon": [-103.1, -94.4], "lat": [33.6, 37.1], "buffer_km": 50},
    }
    assert domain_window("OKC_BBOX", domains) == bbox_to_window(OKC_BBOX)
    assert domain_window("OK_BBOX", domains) == bbox_to_window((-103.1, 33.6, -94.4, 37.1), buffer_km=50)
    with pytest.raises(KeyError):
        domain_window("NOWHERE", domains)


@pytest.mark.req("GRID-08")
def test_domain_window_default_reads_config():
    w = domain_window("OKC_BBOX")
    x, y = FWD.transform(-97.52, 35.47)
    assert bool(w.contains(np.array([x]), np.array([y]))[0])
    assert w == bbox_to_window(OKC_BBOX)
