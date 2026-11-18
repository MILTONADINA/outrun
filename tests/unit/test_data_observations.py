"""specs/01-data.md DATA-07 (Storm Events, SPC, AQS, HMS) and DATA-17 (NWS zones). Synthetic files in tmp_path."""

from __future__ import annotations

import math
import zipfile

import numpy as np
import pandas as pd
import pytest

from outrun.data import aqs, hms, nwszones, spc, stormevents

# ---------------------------------------------------------------- Storm Events
SE_COLS = [
    "BEGIN_YEARMONTH", "BEGIN_DAY", "BEGIN_TIME", "END_YEARMONTH", "END_DAY", "END_TIME", "EPISODE_ID", "EVENT_ID",
    "STATE", "STATE_FIPS", "EVENT_TYPE", "CZ_TYPE", "CZ_FIPS", "CZ_TIMEZONE", "BEGIN_LAT", "BEGIN_LON", "END_LAT",
    "END_LON", "TOR_WIDTH", "TOR_LENGTH", "EPISODE_NARRATIVE", "EVENT_NARRATIVE",
]  # fmt: skip


def _se_rows():
    return [
        [201905, 20, 1530, 201905, 20, 1605, 10, 100, "OKLAHOMA", 40, "Tornado", "C", 17, "CST-6",
         35.5, -98.0, 35.6, -97.9, 100, 2.0, "Supercells.", "EF2 tornado."],
        [202102, 15, 5, 202102, 15, 600, 11, 101, "OKLAHOMA", 40, "Ice Storm", "Z", 25, "CST-6",
         None, None, None, None, None, None, None, "Freezing rain accumulated."],
        [202307, 4, 2330, 202307, 5, 100, 12, 102, "VIRGINIA", 51, "Flash Flood ", "C", 59, "EST-5",
         38.8, -77.3, None, None, None, None, "Storms.", None],
        [202001, 2, 1200, 202001, 2, 1300, 13, 103, "TEXAS", 48, "Winter Weather", "Z", 7, "CST",
         None, None, None, None, None, None, None, None],
        [202001, 3, 900, 202001, 3, 1000, 14, 104, "PUERTO RICO", 72, "Flash Flood", "Z", 1, "AST-4",
         None, None, None, None, None, None, None, None],
    ]  # fmt: skip


@pytest.mark.req("DATA-07")
def test_read_storm_events_times_zones_units(tmp_path):
    p = tmp_path / "StormEvents_details_a.csv"
    pd.DataFrame(_se_rows()[:2], columns=SE_COLS).to_csv(p, index=False)
    p2 = tmp_path / "StormEvents_details_b.csv.gz"
    pd.DataFrame(_se_rows()[2:], columns=SE_COLS).to_csv(p2, index=False, compression="gzip")
    df = stormevents.read_storm_events([p, p2])
    assert list(df.index) == [0, 1, 2, 3, 4]
    assert list(df["event_id"]) == [100, 101, 102, 103, 104]
    ts = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    assert df.loc[0, "begin"] == ts("2019-05-20 21:30") and df.loc[0, "end"] == ts("2019-05-20 22:05")
    assert df.loc[1, "begin"] == ts("2021-02-15 06:05")
    assert df.loc[2, "begin"] == ts("2023-07-05 04:30") and df.loc[2, "end"] == ts("2023-07-05 06:00")
    assert df.loc[3, "begin"] == ts("2020-01-02 18:00")  # "CST" without offset
    assert df.loc[2, "event_type"] == "Flash Flood"
    assert df.loc[1, "zone_id"] == "OKZ025" and df.loc[3, "zone_id"] == "TXZ007"
    assert pd.isna(df.loc[0, "zone_id"]) and pd.isna(df.loc[4, "zone_id"])  # county row; territory zone
    assert df.loc[4, "begin"] == ts("2020-01-03 13:00")  # AST-4
    assert df.loc[0, "tor_width_m"] == pytest.approx(91.44)
    assert df.loc[0, "tor_length_m"] == pytest.approx(3218.688)
    assert math.isnan(df.loc[1, "tor_width_m"]) and math.isnan(df.loc[1, "begin_lat"])
    assert df.loc[0, "narrative"] == "EF2 tornado. Supercells."
    assert df.loc[1, "narrative"] == "Freezing rain accumulated."
    assert df.loc[3, "narrative"] == ""
    for c in ("episode_id", "state_fips", "cz_type", "cz_fips", "begin_lon", "end_lat", "end_lon"):
        assert c in df.columns


@pytest.mark.req("DATA-07")
def test_storm_events_bad_timezone(tmp_path):
    rows = _se_rows()[:1]
    rows[0][13] = "XYZ"
    p = tmp_path / "bad.csv"
    pd.DataFrame(rows, columns=SE_COLS).to_csv(p, index=False)
    with pytest.raises(ValueError):
        stormevents.read_storm_events(p)


@pytest.mark.req("DATA-07")
def test_filter_event_types(tmp_path):
    p = tmp_path / "a.csv"
    pd.DataFrame(_se_rows(), columns=SE_COLS).to_csv(p, index=False)
    df = stormevents.read_storm_events(p)
    out = stormevents.filter_event_types(df, ["flash flood", " ICE STORM"])
    assert list(out["event_id"]) == [101, 102, 104] and list(out.index) == [0, 1, 2]


# ---------------------------------------------------------------- SPC
SPC_COLS = ["om", "yr", "mo", "dy", "date", "time", "tz", "st", "stf", "mag", "slat", "slon", "elat", "elon",
            "len", "wid", "sg"]  # fmt: skip


def _spc_rows():
    return [
        [1, 2019, 5, 20, "2019-05-20", "15:30:00", 3, "OK", 40, 2, 35.0, -97.0, 35.1, -96.9, 8.0, 200, 1],
        [2, 2019, 5, 21, "2019-05-21", "01:00:00", 9, "OK", 40, 0, 35.2, -97.2, 0.0, 0.0, 0.1, 10, 1],
        [3, 2019, 5, 21, "2019-05-21", "02:00:00", 3, "KS", 20, 1, 37.0, -97.0, 37.1, -96.9, 3.0, 50, 2],
        [4, 2019, 5, 22, "2019-05-22", "02:00:00", 3, "KS", 20, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1],
    ]


@pytest.mark.req("DATA-07")
def test_read_spc_tornadoes(tmp_path):
    p = tmp_path / "1950-2023_actual_tornadoes.csv"
    pd.DataFrame(_spc_rows(), columns=SPC_COLS).to_csv(p, index=False)
    df = spc.read_spc_tornadoes(p)
    assert list(df["om"]) == [1, 2] and list(df.index) == [0, 1]
    assert df.loc[0, "begin"] == pd.Timestamp("2019-05-20 21:30", tz="UTC")
    assert df.loc[1, "begin"] == pd.Timestamp("2019-05-21 01:00", tz="UTC")
    assert (df.loc[1, "elat"], df.loc[1, "elon"]) == (35.2, -97.2)
    assert df.loc[0, "len_m"] == pytest.approx(8.0 * 1609.344)
    assert df.loc[0, "wid_m"] == pytest.approx(200 * 0.9144)
    rows = _spc_rows()[:1]
    rows[0][6] = 6
    pd.DataFrame(rows, columns=SPC_COLS).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError):
        spc.read_spc_tornadoes(tmp_path / "bad.csv")


@pytest.mark.req("DATA-07")
def test_read_spc_tornadoes_year_range_skips_old_tz_codes(tmp_path):
    rows = _spc_rows()[:2]
    old = [9, 1970, 4, 3, "1970-04-03", "16:00:00", 6, "OK", 40, 1, 35.0, -97.0, 35.1, -96.9, 1.0, 50, 1]
    new = [5, 2015, 5, 6, "2015-05-06", "18:00:00", 3, "OK", 40, 1, 35.3, -97.4, 35.4, -97.3, 2.0, 100, 1]
    p = tmp_path / "mixed.csv"
    pd.DataFrame([old, *rows, new], columns=SPC_COLS).to_csv(p, index=False)
    with pytest.raises(ValueError):
        spc.read_spc_tornadoes(p)  # default keeps every row, so the 1970 tz=6 row still raises
    df = spc.read_spc_tornadoes(p, years=(2015, 2025))
    assert list(df["om"]) == [1, 2, 5] and list(df.index) == [0, 1, 2]
    assert df.loc[2, "begin"] == pd.Timestamp("2015-05-07 00:00", tz="UTC")
    assert list(spc.read_spc_tornadoes(p, years=(2015, 2015))["om"]) == [5]  # both bounds inclusive
    with pytest.raises(ValueError):
        spc.read_spc_tornadoes(p, years=(1970, 2025))  # the strict tz check still applies inside the range


@pytest.mark.req("DATA-07")
def test_swath_polygons_area():
    from pyproj import Transformer

    df = pd.DataFrame(
        {"slat": [35.4, 35.3], "slon": [-97.5, -97.2], "elat": [35.4, 35.3], "elon": [-97.4, -97.2],
         "wid_m": [200.0, 10.0], "om": [1, 2]}
    )  # fmt: skip
    g = spc.swath_polygons(df, crs="EPSG:5070", min_width_m=50.0)
    assert len(g) == 2 and list(g["om"]) == [1, 2]
    assert g.crs.to_epsg() == 5070
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x0, y0 = tr.transform(-97.5, 35.4)
    x1, y1 = tr.transform(-97.4, 35.4)
    length = math.hypot(x1 - x0, y1 - y0)
    assert g.geometry.iloc[0].area == pytest.approx(length * 200 + math.pi * 100**2, rel=0.01)
    assert g.geometry.iloc[1].area == pytest.approx(math.pi * 25**2, rel=0.02)  # disc, min width 50 m


@pytest.mark.req("DATA-07")
def test_rasterize_mask_centre_rule():
    from shapely.geometry import box

    x = np.array([0.5, 1.5, 2.0, 3.0])
    y = np.array([2.5, 1.5, 0.5])
    m = spc.rasterize_mask([box(0, 0, 2, 2)], x, y)
    assert m.shape == (3, 4) and m.dtype == bool
    np.testing.assert_array_equal(m, [[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0]])
    assert not spc.rasterize_mask([], x, y).any()


# ---------------------------------------------------------------- AQS
AQS_COLS = ["State Code", "County Code", "Site Num", "Parameter Code", "POC", "Latitude", "Longitude", "Date GMT",
            "Time GMT", "Sample Measurement", "Units of Measure"]  # fmt: skip


@pytest.mark.req("DATA-07")
def test_read_aqs_hourly_zip(tmp_path):
    rows = [
        [40, 109, 1037, 88101, 1, 35.47, -97.5, "2023-06-07", "14:00", 10.0, "Micrograms/cubic meter (LC)"],
        [40, 109, 1037, 88101, 2, 35.47, -97.5, "2023-06-07", "14:00", 14.0, "Micrograms/cubic meter (LC)"],
        [40, 109, 1037, 88101, 1, 35.47, -97.5, "2023-06-07", "13:00", -2.0, "Micrograms/cubic meter (LC)"],
        [6, 37, 1103, 88101, 1, 34.07, -118.2, "2023-06-07", "00:00", 30.0, "Micrograms/cubic meter (LC)"],
        [6, 37, 1103, 88502, 1, 34.07, -118.2, "2023-06-07", "01:00", 99.0, "Micrograms/cubic meter (LC)"],
    ]
    csv = pd.DataFrame(rows, columns=AQS_COLS).to_csv(index=False)
    p = tmp_path / "hourly_88101_2023.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("hourly_88101_2023.csv", csv)
    df = aqs.read_aqs_hourly(p)
    assert list(df["site_id"]) == ["06-037-1103", "40-109-1037", "40-109-1037"]
    assert list(df["time"]) == [
        pd.Timestamp("2023-06-07 00:00", tz="UTC"),
        pd.Timestamp("2023-06-07 13:00", tz="UTC"),
        pd.Timestamp("2023-06-07 14:00", tz="UTC"),
    ]
    np.testing.assert_allclose(df["pm25"], [30.0, 0.0, 12.0])
    assert df.loc[1, "lat"] == pytest.approx(35.47) and df.loc[1, "lon"] == pytest.approx(-97.5)
    (tmp_path / "plain.csv").write_text(csv)
    assert len(aqs.read_aqs_hourly(tmp_path / "plain.csv")) == 3


# ---------------------------------------------------------------- HMS
@pytest.mark.req("DATA-07")
def test_read_hms_shapefile(tmp_path):
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon, box

    geoms = [
        box(-98, 35, -97, 36),
        box(-100, 30, -99, 31),
        MultiPolygon([box(-90, 40, -89, 41), box(-88, 40, -87, 41)]),
    ]
    gdf = gpd.GeoDataFrame({"Density": ["Light", " medium", "HEAVY"]}, geometry=geoms, crs="EPSG:4326")
    p = tmp_path / "hms_smoke20230607.shp"
    gdf.to_file(p)
    polys = hms.read_hms(p)
    assert [d for _, d in polys] == ["Light", "Medium", "Heavy", "Heavy"]
    assert all(isinstance(g, Polygon) for g, _ in polys)
    assert abs(polys[0][0].bounds[0]) > 1e4  # metres in EPSG:5070, not degrees
    bad = gpd.GeoDataFrame({"Density": ["Thick"]}, geometry=[box(-98, 35, -97, 36)], crs="EPSG:4326")
    bad.to_file(tmp_path / "bad.shp")
    with pytest.raises(ValueError):
        hms.read_hms(tmp_path / "bad.shp")


@pytest.mark.req("DATA-07")
def test_hms_date():
    from datetime import date

    assert hms.hms_date("hms_smoke20230607.zip") == date(2023, 6, 7)
    with pytest.raises(ValueError):
        hms.hms_date("hms_smoke.zip")


# ---------------------------------------------------------------- DATA-17 zones
@pytest.mark.req("DATA-17")
def test_zone_id():
    assert nwszones.zone_id(40, 25) == "OKZ025"
    assert nwszones.zone_id(6, 1) == "CAZ001"
    assert len(nwszones.STATE_FIPS_TO_POSTAL) == 51
    with pytest.raises(ValueError):
        nwszones.zone_id(99, 1)


@pytest.mark.req("DATA-17")
def test_zones_from_frame_and_masks():
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {"STATE": ["OK", "OK", "TX"], "ZONE": ["25", "025", "007"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 5, 6, 6)],
        crs="EPSG:5070",
    )
    z = nwszones.zones_from_frame(gdf)
    assert sorted(z["zone_id"]) == ["OKZ025", "TXZ007"]
    assert z.set_index("zone_id").geometry["OKZ025"].area == pytest.approx(2.0)
    x = np.array([0.5, 1.5, 2.5, 5.5])
    y = np.array([0.5, 5.5])
    masks = nwszones.zone_masks(z, x, y)
    np.testing.assert_array_equal(masks["OKZ025"], [[1, 1, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(masks["TXZ007"], [[0, 0, 0, 0], [0, 0, 0, 1]])
