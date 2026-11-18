"""The shared GRID (EPSG:5070, 1 km), grid windows and regridding (specs/02-grid-risk.md, GRID-01..08).

Cell ``(row, col)`` of a window: row 0 is the northern-most row, col 0 the western-most column. All x/y values are
EPSG:5070 metres. Sources (SEVIR LCC, HRRR LCC, MRMS lat/lon, polar radar) are brought onto a window either by
nearest-neighbour sampling (``regrid_nearest``) or by binning their pixels (``aggregate_to_grid``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pyproj import Geod, Transformer

Bbox = tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)
Agg = Literal["max", "mean", "sum", "count"]


@dataclass(frozen=True)
class Grid:
    """The project-wide grid definition (GRID-01). Values equal ``configs/domains.yaml`` ``grid``."""

    crs: str = "EPSG:5070"
    resolution_m: float = 1000.0


GRID = Grid()

# pyproj transformers are reusable and thread-safe for reading; build them once.
# always_xy=True means (lon, lat) / (x, y) argument order, never (lat, lon).
_TO_XY = Transformer.from_crs("EPSG:4326", GRID.crs, always_xy=True)
_TO_LONLAT = Transformer.from_crs(GRID.crs, "EPSG:4326", always_xy=True)
_GEOD = Geod(ellps="WGS84")
_AGGS: tuple[str, ...] = ("max", "mean", "sum", "count")


@dataclass(frozen=True)
class GridWindow:
    """A rectangular block of GRID cells (GRID-01, GRID-04).

    ``x0`` is the west edge and ``y0`` the north edge, in metres. The window covers
    ``[x0, x0 + nx*res) x (y0 - ny*res, y0]``: west and north edges inside, east and south edges outside.
    """

    x0: float
    y0: float
    nx: int
    ny: int
    res: float = 1000.0
    crs: str = "EPSG:5070"

    def __post_init__(self) -> None:
        # nx/ny count cells, so they must be whole numbers (1.5 cells would make shape and centres disagree).
        if self.nx != int(self.nx) or self.ny != int(self.ny):
            raise ValueError(f"GridWindow needs whole-number nx and ny, got nx={self.nx}, ny={self.ny}")
        if int(self.nx) < 1 or int(self.ny) < 1:
            raise ValueError(f"GridWindow needs nx >= 1 and ny >= 1, got nx={self.nx}, ny={self.ny}")
        if not self.res > 0:
            raise ValueError(f"GridWindow needs res > 0, got {self.res}")

    @property
    def shape(self) -> tuple[int, int]:
        """``(ny, nx)`` (GRID-04)."""
        return (int(self.ny), int(self.nx))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax) = (x0, y0 - ny*res, x0 + nx*res, y0)`` (GRID-04)."""
        return (self.x0, self.y0 - self.ny * self.res, self.x0 + self.nx * self.res, self.y0)

    def x_centers(self) -> np.ndarray:
        """Ascending cell-centre x, ``x0 + (c + 0.5) * res``, shape ``(nx,)`` (GRID-04)."""
        return self.x0 + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.res

    def y_centers(self) -> np.ndarray:
        """Descending cell-centre y, ``y0 - (r + 0.5) * res``, shape ``(ny,)`` (GRID-04)."""
        return self.y0 - (np.arange(self.ny, dtype=np.float64) + 0.5) * self.res

    def cell_of(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(row, col)`` int64 arrays of the input shape; ``(-1, -1)`` for points outside the window (GRID-04)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x, y = np.broadcast_arrays(x, y)
        with np.errstate(invalid="ignore"):  # NaN coordinates simply land outside
            col_f = np.floor((x - self.x0) / self.res)
            row_f = np.floor((self.y0 - y) / self.res)
            # West edge (x == x0) gives col 0, north edge (y == y0) gives row 0: inside.
            # East edge gives col == nx, south edge gives row == ny: outside.
            inside = (col_f >= 0) & (col_f < self.nx) & (row_f >= 0) & (row_f < self.ny)
        row = np.where(inside, row_f, -1).astype(np.int64)
        col = np.where(inside, col_f, -1).astype(np.int64)
        return row, col

    def contains(self, x: Any, y: Any) -> np.ndarray:
        """Boolean array: True where ``cell_of`` finds a cell (GRID-04)."""
        row, _ = self.cell_of(x, y)
        return row >= 0


def lonlat_to_xy(lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 lon/lat (degrees) -> EPSG:5070 x/y (metres), float64 arrays of the input shape (GRID-02)."""
    # Broadcast first, so a scalar lon with an array of lat works (pyproj needs equal sizes).
    lon, lat = np.broadcast_arrays(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
    x, y = _TO_XY.transform(lon, lat)
    return np.asarray(x, dtype=np.float64).reshape(lon.shape), np.asarray(y, dtype=np.float64).reshape(
        lat.shape
    )


def xy_to_lonlat(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """EPSG:5070 x/y (metres) -> WGS84 lon/lat (degrees), float64 arrays of the input shape (GRID-02)."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    lon, lat = _TO_LONLAT.transform(x, y)
    return np.asarray(lon, dtype=np.float64).reshape(x.shape), np.asarray(lat, dtype=np.float64).reshape(
        y.shape
    )


def bbox_to_window(bbox: Bbox, buffer_km: float = 0.0, res: float = GRID.resolution_m) -> GridWindow:
    """Smallest res-aligned window covering the densified, projected bbox grown by ``buffer_km`` (GRID-03).

    Raises ValueError for an invalid bbox (min >= max, lon outside [-180, 180], lat outside [-90, 90]) or a
    negative buffer.
    """
    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    if not (-180.0 <= lon_min < lon_max <= 180.0):
        raise ValueError(f"invalid bbox longitudes: {bbox!r}")
    if not (-90.0 <= lat_min < lat_max <= 90.0):
        raise ValueError(f"invalid bbox latitudes: {bbox!r}")
    if buffer_km < 0:
        raise ValueError(f"buffer_km must be >= 0, got {buffer_km}")
    if not res > 0:
        raise ValueError(f"res must be > 0, got {res}")

    # Densify the four sides (Albers bends lon/lat lines, so the corners alone are not enough).
    n = 200
    t = np.linspace(0.0, 1.0, n)
    lon = np.concatenate(
        [
            lon_min + (lon_max - lon_min) * t,
            np.full(n, lon_max),
            lon_max - (lon_max - lon_min) * t,
            np.full(n, lon_min),
        ]
    )
    lat = np.concatenate(
        [
            np.full(n, lat_min),
            lat_min + (lat_max - lat_min) * t,
            np.full(n, lat_max),
            lat_max - (lat_max - lat_min) * t,
        ]
    )
    x, y = lonlat_to_xy(lon, lat)

    b = buffer_km * 1000.0
    xmin, xmax = float(x.min()) - b, float(x.max()) + b
    ymin, ymax = float(y.min()) - b, float(y.max()) + b

    # Snap outward to multiples of res: west edge down, north edge up, then just enough cells to cover.
    x0 = math.floor(xmin / res) * res
    y0 = math.ceil(ymax / res) * res
    nx = max(1, math.ceil((xmax - x0) / res))
    ny = max(1, math.ceil((y0 - ymin) / res))
    return GridWindow(x0=float(x0), y0=float(y0), nx=int(nx), ny=int(ny), res=float(res), crs=GRID.crs)


def regrid_nearest(
    values: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    window: GridWindow,
    max_dist_m: float,
    fill: float = np.nan,
) -> np.ndarray:
    """Nearest source pixel (distance in EPSG:5070) for every target cell centre; ``fill`` beyond ``max_dist_m``.

    ``lon``/``lat`` are 2-D of ``values.shape`` or 1-D with ``values.shape == (len(lat), len(lon))`` (GRID-05).
    Returns float32 ``(ny, nx)``. Shape mismatch raises ValueError.
    """
    from scipy.spatial import cKDTree

    values = np.asarray(values)
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if lon.ndim == 1 and lat.ndim == 1:
        if values.shape != (len(lat), len(lon)):
            raise ValueError(f"values shape {values.shape} != (len(lat), len(lon)) = {(len(lat), len(lon))}")
        lon, lat = np.meshgrid(lon, lat)  # rows follow lat, columns follow lon
    elif lon.shape != values.shape or lat.shape != values.shape:
        raise ValueError(
            f"lon {lon.shape}, lat {lat.shape} and values {values.shape} must have the same shape"
        )

    # Project the source pixels to EPSG:5070 and find, for each target cell centre, the nearest one.
    sx, sy = lonlat_to_xy(lon.ravel(), lat.ravel())
    # Source pixels with NaN/inf coordinates (masked or off-domain pixels) are skipped; cKDTree rejects them.
    ok = np.isfinite(sx) & np.isfinite(sy)
    if not ok.any():
        return np.full(window.shape, fill, dtype=np.float32)
    src_values = values.ravel()[ok]
    tree = cKDTree(np.column_stack([sx[ok], sy[ok]]))
    cx, cy = np.meshgrid(window.x_centers(), window.y_centers())
    dist, idx = tree.query(np.column_stack([cx.ravel(), cy.ravel()]), k=1)

    out = src_values[idx].astype(np.float32)
    out[dist > max_dist_m] = fill  # too far from any source pixel: no data
    return out.reshape(window.shape)


def aggregate_to_grid(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    window: GridWindow,
    agg: Agg = "max",
    fill: float = 0.0,
) -> np.ndarray:
    """Bin points into window cells with max/mean/sum/count (GRID-06).

    Points outside the window and NaN values are ignored. Empty cells get ``fill`` (0 for ``count``).
    Returns float32 ``(ny, nx)``. Unknown ``agg`` raises ValueError.
    """
    if agg not in _AGGS:
        raise ValueError(f"agg must be one of {_AGGS}, got {agg!r}")
    values = np.asarray(values, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not (values.shape == x.shape == y.shape):
        raise ValueError(f"values {values.shape}, x {x.shape} and y {y.shape} must have the same shape")

    row, col = window.cell_of(x.ravel(), y.ravel())
    v = values.ravel()
    keep = (row >= 0) & ~np.isnan(v)
    flat = row[keep] * window.nx + col[keep]  # flat cell index of every usable point
    v = v[keep]
    n_cells = window.nx * window.ny

    count = np.bincount(flat, minlength=n_cells).astype(np.float64)
    if agg == "count":
        out = count
    elif agg == "sum":
        out = np.bincount(flat, weights=v, minlength=n_cells)
    elif agg == "mean":
        total = np.bincount(flat, weights=v, minlength=n_cells)
        out = np.divide(total, count, out=np.zeros(n_cells), where=count > 0)
    else:  # max
        out = np.full(n_cells, -np.inf)
        np.maximum.at(out, flat, v)
    if agg != "count":
        out = np.where(count > 0, out, fill)
    return out.astype(np.float32).reshape(window.shape)


def polar_to_xy(
    ranges_m: np.ndarray, azimuths_deg: np.ndarray, radar_lon: float, radar_lat: float
) -> tuple[np.ndarray, np.ndarray]:
    """EPSG:5070 x/y of radar gates, each ``(n_az, n_range)``; azimuth clockwise from true north (GRID-07)."""
    ranges_m = np.asarray(ranges_m, dtype=np.float64)
    azimuths_deg = np.asarray(azimuths_deg, dtype=np.float64)
    az, rng = np.meshgrid(azimuths_deg, ranges_m, indexing="ij")  # (n_az, n_range)
    # Geodesic forward problem on WGS84: start at the radar, go `range` metres along bearing `azimuth`.
    lon, lat, _ = _GEOD.fwd(np.full(az.shape, float(radar_lon)), np.full(az.shape, float(radar_lat)), az, rng)
    return lonlat_to_xy(np.asarray(lon), np.asarray(lat))


def polar_to_grid(
    values: np.ndarray,
    ranges_m: np.ndarray,
    azimuths_deg: np.ndarray,
    radar_lon: float,
    radar_lat: float,
    window: GridWindow,
    agg: Agg = "max",
    fill: float = 0.0,
) -> np.ndarray:
    """``aggregate_to_grid(values, *polar_to_xy(...), window, agg, fill)`` (GRID-07)."""
    x, y = polar_to_xy(ranges_m, azimuths_deg, radar_lon, radar_lat)
    return aggregate_to_grid(values, x, y, window, agg=agg, fill=fill)


def domain_window(name: str, domains: Mapping[str, Mapping[str, Any]] | None = None) -> GridWindow:
    """Window of a named domain (``domains:`` block of ``configs/domains.yaml`` when ``domains`` is None) (GRID-08).

    Unknown name raises KeyError.
    """
    if domains is None:
        from outrun.config import load_config

        domains = load_config("domains")["domains"]
    if name not in domains:
        raise KeyError(f"unknown domain {name!r}; known: {sorted(domains)}")
    d = domains[name]
    lon, lat = d["lon"], d["lat"]
    buffer_km = d.get("buffer_km") or 0.0
    return bbox_to_window((lon[0], lat[0], lon[1], lat[1]), buffer_km=float(buffer_km))
