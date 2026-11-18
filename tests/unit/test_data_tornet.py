"""specs/01-data.md DATA-02: TorNet catalog, official split rule, NetCDF chips to fp16. Synthetic chips in tmp_path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from outrun.data import tornet
from outrun.data.tornet import SplitRule, convert_chips, official_split, read_chip, read_tornet_catalog

A, R, T = 6, 8, 3


def _catalog(tmp_path, with_type=True):
    df = pd.DataFrame(
        {
            "filename": [f"f{i}.nc" for i in range(6)],
            "category": ["TOR", "NUL", "WRN", "TOR", "NUL", "TOR"],
            "ef_number": [2, -1, -1, 1, -1, 3],
            "start_time": [
                "2019-05-20 22:00:00",
                "2019-05-20 23:30:00",
                "2015-01-17 10:00:00",
                "2016-12-31 05:00:00",
                "2018-04-01 00:00:00",
                "2015-01-18 00:00:00",
            ],
            "lat": [35.3] * 6,
            "lon": [-97.3] * 6,
        }
    )
    if with_type:
        df["type"] = ["train", "train", " Test ", "train", "test", "TRAIN"]
    p = tmp_path / "catalog.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------- catalog + split
@pytest.mark.req("DATA-02")
def test_read_catalog_adds_year_julian_label(tmp_path):
    cat = read_tornet_catalog(_catalog(tmp_path))
    assert str(cat["start_time"].dt.tz) == "UTC"
    assert list(cat["year"]) == [2019, 2019, 2015, 2016, 2018, 2015]
    assert list(cat["julian_day"]) == [140, 140, 17, 366, 91, 18]
    assert list(cat["label"]) == [1, 0, 0, 1, 0, 1]
    pd.read_csv(_catalog(tmp_path)).drop(columns=["start_time"]).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(KeyError):
        read_tornet_catalog(tmp_path / "bad.csv")


@pytest.mark.req("DATA-02")
def test_official_split_column_mode(tmp_path):
    cat = read_tornet_catalog(_catalog(tmp_path))
    out = official_split(cat, SplitRule(mode="column", column="type"))
    assert list(out["filename"]) == ["f0.nc", "f1.nc", "f2.nc", "f3.nc", "f5.nc"]  # 2018 dropped
    assert list(out["split"]) == ["train", "train", "test", "train", "train"]
    assert list(out.index) == list(range(5))
    bad = cat.copy()
    bad.loc[0, "type"] = "val"
    with pytest.raises(ValueError):
        official_split(bad, SplitRule(mode="column", column="type"))
    with pytest.raises(KeyError):
        official_split(cat.drop(columns=["type"]), SplitRule(mode="column", column="type"))


@pytest.mark.req("DATA-02")
def test_official_split_julian_mode(tmp_path):
    cat = read_tornet_catalog(_catalog(tmp_path, with_type=False))
    out = official_split(
        cat, SplitRule(mode="julian", modulus=20, train_below=17), years=(2015, 2016, 2019, 2018)
    )
    expected = ["train" if d % 20 < 17 else "test" for d in cat["julian_day"]]
    assert list(out["split"]) == expected
    # 140 % 20 = 0 -> train ; 17 -> test ; 366 % 20 = 6 -> train ; 18 -> test
    assert expected == ["train", "train", "test", "train", "train", "test"]
    # one (year, day) never spans two splits
    assert out.groupby(["year", "julian_day"])["split"].nunique().max() == 1


# ---------------------------------------------------------------- chips
def _write_chip(path, rng, nan_at=None, n_time=T):
    data = {}
    for v in tornet.VARIABLES:
        arr = rng.normal(size=(n_time, A, R, 2)).astype(np.float32)
        if nan_at is not None and v == nan_at[0]:
            arr[-1, nan_at[1], nan_at[2], nan_at[3]] = np.nan
        data[v] = (("time", "azimuth", "range", "sweep"), arr)
    ds = xr.Dataset(
        data,
        coords={"azimuth": np.linspace(0.0, 5.0, A), "range": 2000.0 + 250.0 * np.arange(R)},
    )
    ds.to_netcdf(path)
    return ds


@pytest.mark.req("DATA-02")
def test_read_chip_channel_layout(tmp_path, rng):
    ds = _write_chip(tmp_path / "c.nc", rng, nan_at=("KDP", 2, 3, 1))
    x = read_chip(tmp_path / "c.nc")
    assert x.shape == (27, A, R) and x.dtype == np.float16
    fill = tornet.FILL_VALUES
    c = 0
    for v in tornet.VARIABLES:
        for tilt in (0, 1):
            last = np.nan_to_num(ds[v].values[-1, :, :, tilt], nan=fill[v])
            first = np.nan_to_num(ds[v].values[0, :, :, tilt], nan=fill[v])
            np.testing.assert_allclose(x[c], last.astype(np.float16), rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(x[12 + c], (last - first).astype(np.float16), rtol=1e-3, atol=1e-2)
            c += 1
    kdp_tilt1 = 2 * tornet.VARIABLES.index("KDP") + 1
    assert float(x[kdp_tilt1, 2, 3]) == pytest.approx(fill["KDP"], abs=1e-3)
    mask = np.zeros((A, R))
    mask[2, 3] = 1.0
    np.testing.assert_array_equal(x[24], mask)
    np.testing.assert_allclose(
        x[25], np.broadcast_to((2000.0 + 250.0 * np.arange(R)) / 1000.0, (A, R)), rtol=1e-3
    )
    np.testing.assert_allclose(x[26], np.broadcast_to(np.linspace(0.0, 5.0, A)[:, None], (A, R)), atol=1e-2)


@pytest.mark.req("DATA-02")
def test_read_chip_single_frame_diff_zero(tmp_path, rng):
    _write_chip(tmp_path / "c.nc", rng, n_time=1)
    x = read_chip(tmp_path / "c.nc")
    assert np.all(x[12:24] == 0)


@pytest.mark.req("DATA-02")
def test_read_chip_missing_variable(tmp_path, rng):
    ds = _write_chip(tmp_path / "c.nc", rng)
    ds.drop_vars("ZDR").to_netcdf(tmp_path / "d.nc")
    with pytest.raises(KeyError):
        read_chip(tmp_path / "d.nc")


@pytest.mark.req("DATA-02")
def test_convert_chips_stacks(tmp_path, rng):
    paths = [tmp_path / f"c{i}.nc" for i in range(3)]
    for p in paths:
        _write_chip(p, rng)
    out = tmp_path / "chips.npy"
    convert_chips(paths, out)
    arr = np.load(out, mmap_mode="r")
    assert arr.shape == (3, 27, A, R) and arr.dtype == np.float16
    np.testing.assert_array_equal(arr[1], read_chip(paths[1]))
