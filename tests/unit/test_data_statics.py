"""specs/01-data.md DATA-18 (DEM slope/TPI) and DATA-19 (census population, facility tables)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from outrun.data.census import block_group_population
from outrun.data.dem import slope_deg, tpi
from outrun.data.facilities import FACILITY_KINDS, normalize_facilities


# ---------------------------------------------------------------- DATA-18
@pytest.mark.req("DATA-18")
def test_slope_of_plane():
    yy, xx = np.mgrid[0:8, 0:10].astype(np.float64)
    elev = 3.0 * xx * 30.0 / 30.0 + 4.0 * yy  # dz/dx = 3 m per 30 m cell, dz/dy = 4 m per cell
    s = slope_deg(elev, cell_m=30.0)
    assert s.dtype == np.float32 and s.shape == elev.shape
    np.testing.assert_allclose(s, math.degrees(math.atan(math.hypot(3 / 30, 4 / 30))), rtol=1e-5)
    flat = slope_deg(np.full((5, 5), 100.0), 10.0)
    np.testing.assert_allclose(flat, 0.0)


@pytest.mark.req("DATA-18")
def test_slope_nan_propagates():
    e = np.arange(25, dtype=np.float64).reshape(5, 5)
    e[2, 2] = np.nan
    assert np.isnan(slope_deg(e, 1.0)[2, 1])


@pytest.mark.req("DATA-18")
def test_tpi_plane_peak_edges():
    yy, xx = np.mgrid[0:7, 0:7].astype(np.float64)
    plane = 2.0 * xx + yy
    t = tpi(plane, 1)
    assert t.dtype == np.float32
    np.testing.assert_allclose(t[1:-1, 1:-1], 0.0, atol=1e-5)
    # corner (0, 0): in-bounds window = 2x2 cells {0, 2, 1, 3} -> mean 1.5 -> tpi = -1.5
    assert t[0, 0] == pytest.approx(-1.5)
    peak = np.zeros((5, 5))
    peak[2, 2] = 9.0
    assert tpi(peak, 1)[2, 2] == pytest.approx(9.0 - 1.0)
    with pytest.raises(ValueError):
        tpi(plane, 0)


@pytest.mark.req("DATA-18")
def test_tpi_ignores_nan_neighbours():
    e = np.full((3, 3), 5.0)
    e[0, 0] = np.nan
    assert tpi(e, 1)[1, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------- DATA-19
@pytest.mark.req("DATA-19")
def test_block_group_population():
    t = pd.DataFrame({"GEOID": [401090001001, "401090001002"], "P1_001N": [1200, 0]})
    out = block_group_population(t)
    assert list(out.columns) == ["geoid", "population"]
    assert list(out["geoid"]) == ["401090001001", "401090001002"]
    assert out["population"].dtype == np.int64
    t2 = pd.DataFrame({"GEOID": ["60370001001"], "P1_001N": [5]})
    assert block_group_population(t2)["geoid"][0] == "060370001001"
    for bad in (
        pd.DataFrame({"GEOID": ["1", "1"], "P1_001N": [1, 2]}),
        pd.DataFrame({"GEOID": ["1"], "P1_001N": [-1]}),
        pd.DataFrame({"GEOID": ["1"], "P1_001N": [np.nan]}),
    ):
        with pytest.raises(ValueError):
            block_group_population(bad)
    with pytest.raises(KeyError):
        block_group_population(pd.DataFrame({"GEOID": ["1"]}))


@pytest.mark.req("DATA-19")
def test_normalize_facilities():
    df = pd.DataFrame(
        {"NAME": ["A", None, "C"], "Longitude": [-97.5, -97.4, None], "Latitude": [35.4, 35.5, 35.6],
         "capacity": [300, None, 10]}
    )  # fmt: skip
    df = df.rename(columns={"NAME": "name"})
    out = normalize_facilities(df, "shelter")
    assert list(out.columns) == ["facility_id", "kind", "name", "lon", "lat", "capacity"]
    assert list(out["facility_id"]) == ["shelter-0", "shelter-1"]
    assert list(out["name"]) == ["A", ""]
    assert out["capacity"][0] == 300.0 and math.isnan(out["capacity"][1])
    assert set(FACILITY_KINDS) == {"shelter", "hospital", "fire_station", "school"}
    out2 = normalize_facilities(pd.DataFrame({"lon": [-97.0], "lat": [35.0]}), "hospital")
    assert math.isnan(out2["capacity"][0]) and out2["kind"][0] == "hospital"
    with pytest.raises(ValueError):
        normalize_facilities(df, "mall")
    with pytest.raises(KeyError):
        normalize_facilities(pd.DataFrame({"name": ["x"]}), "school")
