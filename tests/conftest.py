"""Shared fixtures. Everything here is offline and synthetic: no network, no real datasets.

Only numpy / stdlib are imported at module level, so a missing or stubbed ``outrun`` module can never break test
collection (the traceability tests must always run). Area-specific fixtures belong in the area's own test files.
"""

from __future__ import annotations

import math
import socket
from datetime import UTC, datetime

import numpy as np
import pytest

GRID_N = 32  # synthetic grids are 32 x 32 cells of 1 km
N_SLOTS = 12  # routing window: 12 slots x 15 min = 3 h
SLOT_MINUTES = 15
OKC_LON, OKC_LAT = -97.52, 35.47  # anchor for synthetic coordinates


def pytest_configure(config):
    config.addinivalue_line("markers", "req(id): links a test to a requirement ID in specs/")


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator (same numbers every run)."""
    return np.random.default_rng(0)


@pytest.fixture
def t0() -> datetime:
    """A fixed, timezone-aware issue time (the El Reno case evening, UTC)."""
    return datetime(2019, 5, 26, 3, 0, tzinfo=UTC)


def make_gaussian_cube(
    n_slots: int = N_SLOTS,
    n: int = GRID_N,
    start: tuple[float, float] = (8.0, 8.0),
    velocity: tuple[float, float] = (1.5, 1.0),
    sigma: float = 3.0,
    peak: float = 0.9,
) -> np.ndarray:
    """A Gaussian 'hazard' moving across an n x n grid: array (n_slots, n, n) of probabilities in [0, peak].

    Centre at slot s is ``start + s * velocity`` in (row, col) cell units.
    """
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    cube = np.empty((n_slots, n, n), dtype=np.float32)
    for s in range(n_slots):
        cy, cx = start[0] + s * velocity[0], start[1] + s * velocity[1]
        cube[s] = peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    return cube


@pytest.fixture
def grid_field(rng) -> np.ndarray:
    """One 32 x 32 float32 field of uniform random probabilities in [0, 1)."""
    return rng.random((GRID_N, GRID_N), dtype=np.float32)


@pytest.fixture
def risk_cube() -> np.ndarray:
    """(12, 32, 32) float32 moving-Gaussian probability cube (one hazard, all routing slots)."""
    return make_gaussian_cube()


@pytest.fixture
def risk_stack_array() -> np.ndarray:
    """(6, 12, 32, 32) float32: one moving Gaussian per hazard, in Hazard order
    (storm, tornado, flood, wildfire, ice, smoke), each starting at a different place."""
    starts = [(4, 4), (16, 4), (4, 16), (24, 24), (16, 16), (28, 8)]
    return np.stack([make_gaussian_cube(start=s) for s in starts])


@pytest.fixture
def grid_coords() -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates (x, y) in metres for a 32 x 32, 1 km grid (EPSG:5070-like numbers)."""
    x0, y0 = -150_000.0, 1_300_000.0
    x = x0 + 500.0 + 1000.0 * np.arange(GRID_N)
    y = y0 - 500.0 - 1000.0 * np.arange(GRID_N)  # north-up: y decreases with row
    return x, y


def make_lattice_graph(
    rows: int = 5, cols: int = 6, spacing_m: float = 1000.0, speed_mps: float = 15.0
) -> dict:
    """A tiny directed road lattice (rows x cols nodes, both directions on every edge) as plain numpy arrays.

    Returns a dict with ``lon``, ``lat`` (node coordinates near OKC), ``src``, ``dst`` (edge endpoints),
    ``length_m``, ``speed_mps`` and ``free_flow_s`` (edge arrays). Default: 30 nodes, 98 directed edges.
    """
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
    src_a, dst_a = np.array(src), np.array(dst)
    length = np.full(len(src_a), spacing_m)
    speed = np.full(len(src_a), speed_mps)
    return {
        "lon": lon,
        "lat": lat,
        "src": src_a,
        "dst": dst_a,
        "length_m": length,
        "speed_mps": speed,
        "free_flow_s": length / speed,
    }


@pytest.fixture
def lattice_graph() -> dict:
    """30-node, 98-edge directed lattice graph as numpy arrays (see ``make_lattice_graph``)."""
    return make_lattice_graph()


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch) -> dict:
    """Temporary RAW / PROC / scenarios / models / logs folders; OUTRUN_RAW and OUTRUN_PROC point at them."""
    dirs = {name: tmp_path / name for name in ("raw", "proc", "scenarios", "models", "logs")}
    for d in dirs.values():
        d.mkdir()
    monkeypatch.setenv("OUTRUN_RAW", str(dirs["raw"]))
    monkeypatch.setenv("OUTRUN_PROC", str(dirs["proc"]))
    return dirs


@pytest.fixture
def no_network(monkeypatch):
    """Make any attempt to open a non-local socket connection fail loudly (opt-in)."""
    real_connect = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "localhost", "::1") and not str(host).startswith("/"):
            raise RuntimeError(f"network access attempted in a test: {address!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    yield
