"""specs/01-data.md DATA-04 (MRMS decode/crop/accumulate) and DATA-09 (FLASH event hours). Decoder is injected."""

from __future__ import annotations

import gzip
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from outrun.data import mrms
from outrun.data.mrms import (
    accumulate,
    clean_mrms,
    crop_latlon,
    flash_hours,
    mrms_key,
    parse_mrms_name,
    read_mrms,
)

H = timedelta(hours=1)


# ---------------------------------------------------------------- DATA-04 names
@pytest.mark.req("DATA-04")
def test_key_roundtrip_and_parse():
    t = datetime(2021, 6, 1, 12, 0, tzinfo=UTC)
    key = mrms_key(mrms.QPE_PRODUCT, t)
    assert key == (
        "CONUS/MultiSensor_QPE_01H_Pass2_00.00/20210601/"
        "MRMS_MultiSensor_QPE_01H_Pass2_00.00_20210601-120000.grib2.gz"
    )
    assert parse_mrms_name(key) == (mrms.QPE_PRODUCT, t)
    assert parse_mrms_name("MRMS_FLASH_QPE_FFG01H_00.00_20230715-031500.grib2") == (
        "FLASH_QPE_FFG01H_00.00",
        datetime(2023, 7, 15, 3, 15, tzinfo=UTC),
    )
    with pytest.raises(ValueError):
        parse_mrms_name("something_else.grib2")


@pytest.mark.req("DATA-04")
def test_clean_negative_to_nan():
    out = clean_mrms(np.array([-3.0, -999.0, 0.0, 5.5]))
    assert out.dtype == np.float32
    np.testing.assert_array_equal(np.isnan(out), [True, True, False, False])
    np.testing.assert_allclose(out[2:], [0.0, 5.5])


@pytest.mark.req("DATA-04")
def test_read_mrms_gunzips_to_temp_and_cleans(tmp_path):
    raw = tmp_path / "MRMS_MultiSensor_QPE_01H_Pass2_00.00_20210601-120000.grib2.gz"
    raw.write_bytes(gzip.compress(b"GRIB-PAYLOAD"))
    original = raw.read_bytes()
    seen = {}

    def decoder(p):
        seen["path"] = p
        seen["bytes"] = Path(p).read_bytes()
        return xr.DataArray(np.array([[-3.0, 2.0]]), dims=("latitude", "longitude"))

    da = read_mrms(raw, decoder=decoder)
    assert seen["bytes"] == b"GRIB-PAYLOAD"
    assert str(seen["path"]) != str(raw)
    assert not os.path.exists(seen["path"])  # temp file removed
    assert raw.read_bytes() == original  # raw untouched
    assert np.isnan(da.values[0, 0]) and da.values[0, 1] == 2.0


@pytest.mark.req("DATA-04")
def test_crop_latlon_descending_lat_and_0_360_lon():
    lat = np.arange(37.0, 32.9, -0.5)  # 37.0 .. 33.0 descending
    lon = np.arange(260.0, 264.01, 0.5)  # -100 .. -96 in 0-360
    da = xr.DataArray(
        np.arange(lat.size * lon.size, dtype=np.float32).reshape(lat.size, lon.size),
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )
    out = crop_latlon(da, (-99.0, 34.0, -97.0, 36.0))
    np.testing.assert_allclose(out["latitude"].values, [36.0, 35.5, 35.0, 34.5, 34.0])
    np.testing.assert_allclose(out["longitude"].values, [-99.0, -98.5, -98.0, -97.5, -97.0])
    np.testing.assert_array_equal(out.values, da.values[2:7, 2:7])
    with pytest.raises(ValueError):
        crop_latlon(da, (0.0, 0.0, 1.0, 1.0))


def _hourly(times, values):
    return xr.DataArray(
        np.asarray(values, dtype=np.float32),
        dims=("time", "y", "x"),
        coords={"time": pd.DatetimeIndex(times)},
    )


@pytest.mark.req("DATA-04")
def test_accumulate_windows_and_missing_hours():
    t0 = pd.Timestamp("2021-06-01 00:00")
    hours = [0, 1, 2, 4, 5, 6]  # 03 missing
    times = [t0 + pd.Timedelta(hours=h) for h in hours]
    vals = np.stack([np.full((2, 2), h + 1.0) for h in hours])  # value = hour + 1
    vals[4, 0, 0] = np.nan  # 05Z cell (0,0) missing
    ds = accumulate(_hourly(times, vals), windows_h=(1, 3))
    assert set(ds.data_vars) == {"qpe_1h", "qpe_3h"}
    np.testing.assert_array_equal(ds["time"].values, pd.DatetimeIndex(times).values)
    q1, q3 = ds["qpe_1h"].values, ds["qpe_3h"].values
    np.testing.assert_allclose(q1[1], 2.0)
    assert np.isnan(q3[0]).all() and np.isnan(q3[1]).all()  # window reaches before the data
    np.testing.assert_allclose(q3[2], 1.0 + 2.0 + 3.0)  # 00+01+02
    assert np.isnan(q3[3]).all()  # 04Z needs 03Z (absent)
    assert np.isnan(q3[5, 0, 0])  # 06Z window includes the NaN cell at 05Z
    np.testing.assert_allclose(q3[5, 1, 1], 5.0 + 6.0 + 7.0)


@pytest.mark.req("DATA-04")
def test_accumulate_rejects_bad_time_axis():
    t0 = pd.Timestamp("2021-06-01 00:00")
    with pytest.raises(ValueError):
        accumulate(_hourly([t0, t0 + pd.Timedelta(minutes=30)], np.zeros((2, 1, 1))))
    with pytest.raises(ValueError):
        accumulate(_hourly([t0 + pd.Timedelta(hours=1), t0], np.zeros((2, 1, 1))))


# ---------------------------------------------------------------- DATA-09
REPORTS = [datetime(2021, 6, 1, 12, 30, tzinfo=UTC), datetime(2021, 6, 1, 14, 10, tzinfo=UTC)]
EVENT = [datetime(2021, 6, 1, 6, tzinfo=UTC) + i * H for i in range(15)]  # 06Z..20Z
RAINY = [datetime(2021, 6, 2, 0, tzinfo=UTC) + i * H for i in range(24)] + EVENT[:3]


@pytest.mark.req("DATA-09")
def test_flash_hours_event_windows_plus_random():
    out = flash_hours(REPORTS, RAINY, pad_h=6, random_frac=0.2, seed=0)
    assert out == sorted(set(out))
    assert set(EVENT) <= set(out)
    extra = sorted(set(out) - set(EVENT))
    assert len(extra) == 3  # round(0.2 * 15)
    avail = sorted(set(RAINY) - set(EVENT))
    idx = np.random.default_rng(0).choice(len(avail), 3, replace=False)
    assert set(extra) == {avail[i] for i in idx}
    assert flash_hours(REPORTS, RAINY, seed=0) == out


@pytest.mark.req("DATA-09")
def test_flash_hours_single_report_and_caps():
    one = flash_hours(REPORTS[:1], [], pad_h=6)
    assert one == [datetime(2021, 6, 1, 6, tzinfo=UTC) + i * H for i in range(13)]
    assert flash_hours(REPORTS, RAINY, max_hours=15) == EVENT
    capped = flash_hours(REPORTS, RAINY, max_hours=16)
    assert len(capped) == 16 and set(EVENT) <= set(capped)
    assert flash_hours(REPORTS, RAINY, max_hours=10) == EVENT[:10]
    with pytest.raises(ValueError):
        flash_hours([datetime(2021, 6, 1, 12)], RAINY)
