"""specs/01-data.md DATA-03: HRRR .idx byte ranges, crop, f01 rule, cycle/lead rules, MASSDEN units. No Herbie."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr

from outrun.data import hrrr
from outrun.data.hrrr import (
    IdxEntry,
    byte_ranges,
    check_lead,
    check_massden_units,
    choose_ice_cycles,
    choose_smoke_leads,
    crop_bbox,
    fetch_fields,
    ice_cycle_times,
    massden_to_ugm3,
    parse_idx,
    range_header,
    select_fields,
)

IDX = """1:0:d=2021010100:REFC:entire atmosphere:1 hour fcst:
2:100:d=2021010100:TMP:surface:1 hour fcst:
3:250:d=2021010100:TMP:2 m above ground:1 hour fcst:
4:400:d=2021010100:DPT:2 m above ground:1 hour fcst:

5:600:d=2021010100:PRATE:surface:1 hour fcst:
6:700:d=2021010100:FRZR:surface:0-1 hour acc fcst:
7:800:d=2021010100:FRZR:surface:1 hour fcst:extra:field
8:900:d=2021010100:MASSDEN:8 m above ground:1 hour fcst:
"""
GRIB = bytes(i % 251 for i in range(1000))


# ---------------------------------------------------------------- idx / byte ranges
@pytest.mark.req("DATA-03")
def test_parse_idx():
    e = parse_idx(IDX)
    assert len(e) == 8
    assert e[0] == IdxEntry(
        1, 0, 99, datetime(2021, 1, 1, 0, tzinfo=UTC), "REFC", "entire atmosphere", "1 hour fcst"
    )
    assert e[2].var == "TMP" and e[2].level == "2 m above ground" and (e[2].start, e[2].end) == (250, 399)
    assert e[6].fcst == "1 hour fcst" and e[6].end == 899
    assert e[-1].start == 900 and e[-1].end is None
    with pytest.raises(ValueError):
        parse_idx("not an idx line\n")


@pytest.mark.req("DATA-03")
def test_select_fields_exact_match():
    e = parse_idx(IDX)
    sel = select_fields(e, ["DPT:2 m above ground", "TMP:2 m above ground"])
    assert [x.num for x in sel] == [3, 4]  # idx order
    assert [x.num for x in select_fields(e, ["FRZR:surface"])] == [6, 7]
    assert [x.num for x in select_fields(e, ["TMP:surface", "TMP:surface"])] == [2]
    with pytest.raises(KeyError, match="CFRZR"):
        select_fields(e, ["CFRZR:surface"])


@pytest.mark.req("DATA-03")
def test_byte_ranges_merge_contiguous():
    e = parse_idx(IDX)
    sel = select_fields(e, ["MASSDEN:8 m above ground", "TMP:2 m above ground", "DPT:2 m above ground"])
    assert byte_ranges(sel) == [(250, 599), (900, None)]
    assert byte_ranges(select_fields(e, ["TMP:surface", "PRATE:surface"])) == [(100, 249), (600, 699)]
    assert range_header(250, 599) == "bytes=250-599"
    assert range_header(900, None) == "bytes=900-"


@pytest.mark.req("DATA-03")
def test_fetch_fields_uses_one_request_per_range():
    calls = []

    def range_get(url, start, end):
        calls.append((url, start, end))
        return GRIB[start : None if end is None else end + 1]

    out = fetch_fields(
        "s3://hrrr/x.grib2",
        IDX,
        ["TMP:2 m above ground", "DPT:2 m above ground", "MASSDEN:8 m above ground"],
        range_get,
    )
    assert calls == [("s3://hrrr/x.grib2", 250, 599), ("s3://hrrr/x.grib2", 900, None)]
    assert out == GRIB[250:600] + GRIB[900:]


# ---------------------------------------------------------------- crop
@pytest.mark.req("DATA-03")
def test_crop_bbox_2d_coords_and_0_360_lon():
    ny, nx = 10, 12
    yy, xx = np.mgrid[0:ny, 0:nx]
    lat = 30.0 + 0.5 * yy + 0.1 * xx
    lon = 260.0 + 0.5 * xx  # 0..360 convention, i.e. -100 + 0.5 x
    da = xr.DataArray(
        np.arange(ny * nx, dtype=np.float32).reshape(ny, nx),
        dims=("y", "x"),
        coords={"latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon)},
    )
    bbox = (-99.0, 31.0, -97.0, 33.0)
    lon180 = lon - 360.0
    inside = (lon180 >= bbox[0]) & (lon180 <= bbox[2]) & (lat >= bbox[1]) & (lat <= bbox[3])
    rows, cols = np.where(inside.any(axis=1))[0], np.where(inside.any(axis=0))[0]
    out = crop_bbox(da, bbox)
    np.testing.assert_array_equal(
        out.values, da.values[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    )
    with pytest.raises(ValueError):
        crop_bbox(da, (10.0, 10.0, 11.0, 11.0))


# ---------------------------------------------------------------- f01 / cycles / leads
@pytest.mark.req("DATA-03")
def test_check_lead_f01_rule():
    for var in ("PRATE", "FRZR", "CFRZR", "CICEP", "CSNOW", "CRAIN", "APCP"):
        with pytest.raises(ValueError):
            check_lead(var, 0)
        assert check_lead(var, 1) is None
    assert check_lead("TMP", 0) is None
    with pytest.raises(ValueError):
        check_lead("TMP", -1)
    assert len(hrrr.HRRR_ICE_FIELDS) == 9


@pytest.mark.req("DATA-03")
def test_ice_cycle_times_window_and_v4_start():
    t = ice_cycle_times(2020, 2020)
    assert t[0] == datetime(2020, 12, 3, 0, tzinfo=UTC)
    assert t[-1] == datetime(2021, 3, 15, 21, tzinfo=UTC)
    assert len(t) == 103 * 8 and t == sorted(set(t))
    leap = ice_cycle_times(2023, 2023, cycles=(0, 6, 12, 18))
    assert leap[0] == datetime(2023, 11, 15, 0, tzinfo=UTC) and len(leap) == 122 * 4
    assert len(ice_cycle_times()) == (103 + 121 + 121 + 122 + 121) * 8


@pytest.mark.req("DATA-03")
def test_cycle_and_lead_caps():
    assert choose_ice_cycles(1.0) == (0, 3, 6, 9, 12, 15, 18, 21)
    assert choose_ice_cycles(1.9) == (0, 3, 6, 9, 12, 15, 18, 21)
    assert choose_ice_cycles(2.0) == (0, 6, 12, 18)
    assert choose_smoke_leads(1.0) == (1, 6)  # ~35 GB
    assert choose_smoke_leads(2.2) == (6,)  # ~77 GB > 60


# ---------------------------------------------------------------- MASSDEN
@pytest.mark.req("DATA-03")
def test_massden_conversion():
    before = datetime(2021, 12, 20, 23, tzinfo=UTC)
    after = datetime(2022, 6, 1, tzinfo=UTC)
    v = np.array([1.0, 25.0, np.nan])
    out = massden_to_ugm3(v, before)
    np.testing.assert_allclose(out, v)
    assert out.dtype == np.float32
    kg = np.array([1e-9, 2.5e-8, np.nan])
    np.testing.assert_allclose(massden_to_ugm3(kg, after), [1.0, 25.0, np.nan], rtol=1e-5)
    np.testing.assert_allclose(massden_to_ugm3(np.array([3e-9]), hrrr.MASSDEN_UNIT_CHANGE), [3.0], rtol=1e-5)
    with pytest.raises(ValueError):
        massden_to_ugm3(np.array([5.0]), after)  # still ug/m3 after the change
    with pytest.raises(ValueError):
        massden_to_ugm3(np.array([-1e-9]), after)
    with pytest.raises(ValueError):
        massden_to_ugm3(v, datetime(2022, 6, 1))


@pytest.mark.req("DATA-03")
def test_massden_units_check():
    after = datetime(2022, 6, 1, tzinfo=UTC)
    before = datetime(2021, 6, 1, tzinfo=UTC)
    assert check_massden_units("kg m**-3", after) is None
    assert check_massden_units("ug m-3", before) is None
    with pytest.raises(ValueError):
        check_massden_units("kg m**-3", before)
    with pytest.raises(ValueError):
        check_massden_units("ug m-3", after)
