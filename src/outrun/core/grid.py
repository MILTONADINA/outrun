"""The shared GRID (EPSG:5070, 1 km), grid windows and regridding (specs/02-grid-risk.md, GRID-01..08).

Cell ``(row, col)`` of a window: row 0 is the northern-most row, col 0 the western-most column. All x/y values are
EPSG:5070 metres. Sources (SEVIR LCC, HRRR LCC, MRMS lat/lon, polar radar) are brought onto a window either by
nearest-neighbour sampling (``regrid_nearest``) or by binning their pixels (``aggregate_to_grid``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Bbox = tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)
Agg = Literal["max", "mean", "sum", "count"]


@dataclass(frozen=True)
class Grid:
    """The project-wide grid definition (GRID-01). Values equal ``configs/domains.yaml`` ``grid``."""

    crs: str = "EPSG:5070"
    resolution_m: float = 1000.0


GRID = Grid()


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
        if int(self.nx) < 1 or int(self.ny) < 1:
            raise ValueError(f"GridWindow needs nx >= 1 and ny >= 1, got nx={self.nx}, ny={self.ny}")
        if not self.res > 0:
            raise ValueError(f"GridWindow needs res > 0, got {self.res}")

    @property
    def shape(self) -> tuple[int, int]:
        """``(ny, nx)`` (GRID-04)."""
        raise NotImplementedError

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax) = (x0, y0 - ny*res, x0 + nx*res, y0)`` (GRID-04)."""
        raise NotImplementedError

    def x_centers(self) -> np.ndarray:
        """Ascending cell-centre x, ``x0 + (c + 0.5) * res``, shape ``(nx,)`` (GRID-04)."""
        raise NotImplementedError

    def y_centers(self) -> np.ndarray:
        """Descending cell-centre y, ``y0 - (r + 0.5) * res``, shape ``(ny,)`` (GRID-04)."""
        raise NotImplementedError

    def cell_of(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(row, col)`` int64 arrays of the input shape; ``(-1, -1)`` for points outside the window (GRID-04)."""
        raise NotImplementedError

    def contains(self, x: Any, y: Any) -> np.ndarray:
        """Boolean array: True where ``cell_of`` finds a cell (GRID-04)."""
        raise NotImplementedError


def lonlat_to_xy(lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
    """WGS84 lon/lat (degrees) -> EPSG:5070 x/y (metres), float64 arrays of the input shape (GRID-02)."""
    raise NotImplementedError


def xy_to_lonlat(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """EPSG:5070 x/y (metres) -> WGS84 lon/lat (degrees), float64 arrays of the input shape (GRID-02)."""
    raise NotImplementedError


def bbox_to_window(bbox: Bbox, buffer_km: float = 0.0, res: float = GRID.resolution_m) -> GridWindow:
    """Smallest res-aligned window covering the densified, projected bbox grown by ``buffer_km`` (GRID-03).

    Raises ValueError for an invalid bbox (min >= max, lon outside [-180, 180], lat outside [-90, 90]) or a
    negative buffer.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def polar_to_xy(
    ranges_m: np.ndarray, azimuths_deg: np.ndarray, radar_lon: float, radar_lat: float
) -> tuple[np.ndarray, np.ndarray]:
    """EPSG:5070 x/y of radar gates, each ``(n_az, n_range)``; azimuth clockwise from true north (GRID-07)."""
    raise NotImplementedError


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
    raise NotImplementedError


def domain_window(name: str, domains: Mapping[str, Mapping[str, Any]] | None = None) -> GridWindow:
    """Window of a named domain (``domains:`` block of ``configs/domains.yaml`` when ``domains`` is None) (GRID-08).

    Unknown name raises KeyError.
    """
    raise NotImplementedError
