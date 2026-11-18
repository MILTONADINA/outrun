"""specs/01-data.md DATA-06: WSTS GeoTIFF reader, NDWS 12-channel mapper, shift statistics."""

from __future__ import annotations

import numpy as np
import pytest

from outrun.data import wsts
from outrun.data.ndws import NDWS_INPUT_FEATURES
from outrun.data.wsts import channel_shift_stats, map_to_ndws, read_wsts_day

H = W = 16  # 16 px x 375 m = 6 km -> 6 x 6 at 1 km


def _bands(value=2.0):
    b = {desc: np.full((H, W), value, np.float32) for desc in wsts.WSTS_TO_NDWS.values()}
    b[wsts.WSTS_TO_NDWS["PrevFireMask"]] = np.zeros((H, W), np.float32)
    b[wsts.WSTS_TO_NDWS["th"]] = np.full((H, W), 90.0, np.float32)
    return b


def _conv(channel, v):
    scale, offset = wsts.UNIT_CONVERSIONS.get(channel, (1.0, 0.0))
    return v * scale + offset


@pytest.mark.req("DATA-06")
def test_read_wsts_day_band_descriptions(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    descs = ["Elevation", " Wind direction ", "Active fire"]
    data = np.stack([np.full((H, W), i + 1.0, np.float32) for i in range(3)])
    p = tmp_path / "2021-08-01.tif"
    with rasterio.open(
        p, "w", driver="GTiff", height=H, width=W, count=3, dtype="float32",
        crs="EPSG:32611", transform=from_origin(500000, 4000000, 375, 375),
    ) as dst:  # fmt: skip
        dst.write(data)
        for i, d in enumerate(descs, start=1):
            dst.set_band_description(i, d)
    bands, res = read_wsts_day(p)
    assert res == pytest.approx(375.0)
    assert set(bands) == {"Elevation", "Wind direction", "Active fire"}
    np.testing.assert_array_equal(bands["Active fire"], np.full((H, W), 3.0, np.float32))

    p2 = tmp_path / "nodesc.tif"
    with rasterio.open(
        p2, "w", driver="GTiff", height=H, width=W, count=1, dtype="float32",
        crs="EPSG:32611", transform=from_origin(500000, 4000000, 375, 375),
    ) as dst:  # fmt: skip
        dst.write(data[:1])
    with pytest.raises(ValueError):
        read_wsts_day(p2)


@pytest.mark.req("DATA-06")
def test_map_constant_fields_shape_and_order():
    pop = np.full((6, 6), 7.0, np.float32)
    out = map_to_ndws(_bands(2.0), 375.0, population=pop)
    assert out.shape == (12, 6, 6) and out.dtype == np.float32
    for c, name in enumerate(NDWS_INPUT_FEATURES):
        if name == "population":
            np.testing.assert_allclose(out[c], 7.0)
        elif name == "th":
            np.testing.assert_allclose(out[c], _conv("th", 90.0), atol=1e-3)
        elif name == "PrevFireMask":
            np.testing.assert_array_equal(out[c], 0.0)
        else:
            np.testing.assert_allclose(out[c], _conv(name, 2.0), rtol=1e-4)


@pytest.mark.req("DATA-06")
def test_map_wind_direction_circular_mean():
    b = _bands()
    th = np.full((H, W), 350.0, np.float32)
    th[:, 1::2] = 10.0  # alternating columns: every 1 km cell sees both 350 and 10
    b[wsts.WSTS_TO_NDWS["th"]] = th
    out = map_to_ndws(b, 375.0)
    c = NDWS_INPUT_FEATURES.index("th")
    d = np.asarray(out[c], np.float64)
    if "th" not in wsts.UNIT_CONVERSIONS:
        assert np.all((d >= 0) & (d <= 360))  # float32 rounding may give 360.0
        assert np.all(np.minimum(d, 360 - d) < 3.0)  # ~0 deg, never ~180


@pytest.mark.req("DATA-06")
def test_map_fire_mask_any_and_population_rules():
    b = _bands()
    fire = np.zeros((H, W), np.float32)
    fire[1, 1] = 1.0  # source px 375-750 m lies inside output cell (0, 0)
    b[wsts.WSTS_TO_NDWS["PrevFireMask"]] = fire
    out = map_to_ndws(b, 375.0)
    c = NDWS_INPUT_FEATURES.index("PrevFireMask")
    assert out[c, 0, 0] == 1.0 and out[c].sum() == 1.0
    assert np.isnan(out[NDWS_INPUT_FEATURES.index("population")]).all()
    with pytest.raises(ValueError):
        map_to_ndws(b, 375.0, population=np.zeros((5, 5), np.float32))
    b.pop(wsts.WSTS_TO_NDWS["erc"])
    with pytest.raises(KeyError):
        map_to_ndws(b, 375.0)


@pytest.mark.req("DATA-06")
def test_map_ignores_nan_pixels():
    b = _bands(4.0)
    elev = b[wsts.WSTS_TO_NDWS["elevation"]]
    elev[0, 0] = np.nan
    elev[8:, 8:] = np.nan  # make a whole bottom-right region NaN
    out = map_to_ndws(b, 375.0)
    e = out[NDWS_INPUT_FEATURES.index("elevation")]
    np.testing.assert_allclose(e[0, 0], _conv("elevation", 4.0), rtol=1e-4)
    assert np.isnan(e[5, 5])  # footprint 5000-6000 m = px 13.3-16, all NaN


@pytest.mark.req("DATA-06")
def test_channel_shift_stats(rng):
    nd = rng.normal(0, 1, size=(40, 12, 8, 8)).astype(np.float32)
    ws = rng.normal(0, 1, size=(30, 12, 8, 8)).astype(np.float32)
    ws[:, 1] += 5.0  # wind direction strongly shifted
    ws[:, 9] = np.nan  # population unavailable
    df = channel_shift_stats(ws, nd, threshold=0.2)
    assert list(df.index) == list(NDWS_INPUT_FEATURES)
    for col in ("ks", "shifted", "wsts_q05", "ndws_q05", "wsts_q50", "ndws_q95"):
        assert col in df.columns
    assert df.loc["elevation", "ks"] < 0.1 and not df.loc["elevation", "shifted"]
    assert df.loc["th", "ks"] > 0.9 and df.loc["th", "shifted"]
    assert np.isnan(df.loc["population", "ks"]) and df.loc["population", "shifted"]
    assert df.loc["elevation", "ndws_q50"] == pytest.approx(np.median(nd[:, 0]), abs=1e-6)
