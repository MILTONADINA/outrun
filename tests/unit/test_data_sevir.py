"""specs/01-data.md DATA-01 (SEVIR reader) and DATA-08 (IR107 60 GB rule). Tiny synthetic h5 files in tmp_path."""

from __future__ import annotations

from datetime import date

import h5py
import numpy as np
import pandas as pd
import pytest

from outrun.data import sevir
from outrun.data.sevir import (
    GB,
    apply_ir107_cap,
    build_sevir_memmap,
    decode_ir,
    filter_catalog,
    ir_to_uint8,
    parse_sevir_filename,
    pool_vil_2x2,
    read_catalog,
)


def _catalog_rows():
    """Two storm events with vil+ir107, one with vil only, one random event, one 2017 event."""
    rows = []

    def add(eid, t, img, fname, idx, box=(-98.0, 35.0, -96.0, 37.0)):
        rows.append(
            dict(
                id=eid, file_name=fname, file_index=idx, img_type=img, time_utc=t, proj="lcc",
                llcrnrlon=box[0], llcrnrlat=box[1], urcrnrlon=box[2], urcrnrlat=box[3],
            )
        )  # fmt: skip

    add("S100", "2018-05-01 20:00:00", "vil", "vil.h5", 0)
    add("S100", "2018-05-01 20:00:00", "ir107", "ir.h5", 0)
    add("S200", "2019-06-03 21:00:00", "vil", "vil.h5", 1, box=(-90.0, 40.0, -88.0, 42.0))
    add("S200", "2019-06-03 21:00:00", "ir107", "ir.h5", 1, box=(-90.0, 40.0, -88.0, 42.0))
    add("S300", "2018-07-01 00:00:00", "vil", "vil.h5", 2)  # no IR -> dropped
    add("R400", "2018-08-01 00:00:00", "vil", "vil.h5", 3)
    add("R400", "2018-08-01 00:00:00", "ir107", "ir.h5", 3)  # random event -> dropped
    add("S500", "2017-08-01 00:00:00", "vil", "vil.h5", 4)
    add("S500", "2017-08-01 00:00:00", "ir107", "ir.h5", 4)  # 2017 -> dropped
    return rows


@pytest.fixture
def catalog_csv(tmp_path):
    p = tmp_path / "CATALOG.csv"
    pd.DataFrame(_catalog_rows()).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------- DATA-01
@pytest.mark.req("DATA-01")
def test_read_catalog_parses_utc(catalog_csv, tmp_path):
    cat = read_catalog(catalog_csv)
    assert str(cat["time_utc"].dt.tz) == "UTC"
    assert cat.loc[0, "time_utc"] == pd.Timestamp("2018-05-01 20:00:00", tz="UTC")
    pd.DataFrame(_catalog_rows()).drop(columns=["proj"]).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(KeyError):
        read_catalog(tmp_path / "bad.csv")


@pytest.mark.req("DATA-01")
def test_filter_catalog_years_types_storm_both_types(catalog_csv):
    out = filter_catalog(read_catalog(catalog_csv))
    assert list(zip(out["id"], out["img_type"], strict=True)) == [
        ("S100", "ir107"),
        ("S100", "vil"),
        ("S200", "ir107"),
        ("S200", "vil"),
    ]
    assert list(out.index) == [0, 1, 2, 3]


@pytest.mark.req("DATA-01")
def test_filter_catalog_bbox_intersection(catalog_csv):
    cat = read_catalog(catalog_csv)
    okc = (-98.10, 35.10, -97.10, 35.80)
    assert set(filter_catalog(cat, bbox=okc)["id"]) == {"S100"}
    touching = (-88.0, 30.0, -80.0, 40.0)  # touches S200's corner (closed intervals)
    assert set(filter_catalog(cat, bbox=touching)["id"]) == {"S200"}
    nowhere = (0.0, 0.0, 1.0, 1.0)
    assert len(filter_catalog(cat, bbox=nowhere)) == 0
    assert set(filter_catalog(cat, img_types=("vil",))["id"]) == {"S100", "S200", "S300"}


@pytest.mark.req("DATA-01")
def test_pool_vil_average_round_half_up():
    a = np.array([[0, 1, 255, 255], [0, 1, 255, 254], [10, 20, 3, 3], [30, 40, 3, 4]], dtype=np.uint8)
    out = pool_vil_2x2(a)
    # (0+1+0+1+2)//4 = 1 ; (255*3+254+2)//4 = 255 ; (100+2)//4 = 25 ; (13+2)//4 = 3
    np.testing.assert_array_equal(out, np.array([[1, 255], [25, 3]], dtype=np.uint8))
    assert out.dtype == np.uint8
    batch = np.stack([a, a])[:, None]  # (2, 1, 4, 4)
    assert pool_vil_2x2(batch).shape == (2, 1, 2, 2)
    with pytest.raises(ValueError):
        pool_vil_2x2(np.zeros((3, 4), np.uint8))


@pytest.mark.req("DATA-01")
def test_pool_is_average_not_max():
    a = np.array([[200, 0], [0, 0]], dtype=np.uint8)
    assert pool_vil_2x2(a)[0, 0] == 50


@pytest.mark.req("DATA-01")
def test_ir_encoding():
    tb = np.array([100.0, 180.0, 250.0, 320.0, 400.0, np.nan])
    np.testing.assert_array_equal(ir_to_uint8(tb), np.array([0, 0, 128, 255, 255, 0], dtype=np.uint8))
    assert ir_to_uint8(tb).dtype == np.uint8
    raw = np.array([-2315, 0], dtype=np.int16)
    np.testing.assert_allclose(decode_ir(raw, scale=0.01, offset=273.15), [250.0, 273.15], atol=1e-4)
    assert decode_ir(raw).dtype == np.float32


def _write_h5(path, key, data, ids):
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=data)
        f.create_dataset(sevir.ID_KEY, data=np.array([i.encode() for i in ids]))


@pytest.fixture
def sevir_files(tmp_path, rng):
    S, T, n = 4, 5, 2
    vil = rng.integers(0, 256, size=(n, 2 * S, 2 * S, T), dtype=np.uint8)
    ir_raw = rng.integers(-9000, 4000, size=(n, S, S, T)).astype(np.int16)
    _write_h5(tmp_path / "vil.h5", sevir.VIL_KEY, vil, ["S100", "S200"])
    _write_h5(tmp_path / "ir.h5", sevir.IR107_KEY, ir_raw, ["S100", "S200"])
    rows = pd.DataFrame(_catalog_rows()[:4])
    rows["time_utc"] = pd.to_datetime(rows["time_utc"], utc=True)
    return rows, vil, ir_raw, S, T


@pytest.mark.req("DATA-01")
def test_build_sevir_memmap_layout(sevir_files, tmp_path):
    rows, vil, ir_raw, S, T = sevir_files
    out = tmp_path / "proc" / "sevir_train.npy"
    out.parent.mkdir()
    meta = build_sevir_memmap(rows, tmp_path, out, ir_scale=0.01, ir_offset=273.15)
    arr = np.load(out, mmap_mode="r")
    assert arr.shape == (2, T, 2, S, S) and arr.dtype == np.uint8
    for i in range(2):
        exp_vil = pool_vil_2x2(np.moveaxis(vil[i], -1, 0))
        exp_ir = ir_to_uint8(decode_ir(np.moveaxis(ir_raw[i], -1, 0), 0.01, 273.15))
        np.testing.assert_array_equal(arr[i, :, 0], exp_vil)
        np.testing.assert_array_equal(arr[i, :, 1], exp_ir)
    assert list(meta["id"]) == ["S100", "S200"] and list(meta["row"]) == [0, 1]
    for c in ("time_utc", "proj", "llcrnrlat", "llcrnrlon", "urcrnrlat", "urcrnrlon"):
        assert c in meta.columns
    assert meta.loc[1, "llcrnrlon"] == -90.0
    saved = pd.read_parquet(out.with_suffix(".meta.parquet"))
    assert list(saved["id"]) == ["S100", "S200"]


@pytest.mark.req("DATA-01")
def test_build_sevir_memmap_id_mismatch(sevir_files, tmp_path):
    rows, vil, ir_raw, S, T = sevir_files
    _write_h5(tmp_path / "ir.h5", sevir.IR107_KEY, ir_raw, ["S200", "S100"])  # swapped ids
    with pytest.raises(ValueError):
        build_sevir_memmap(rows, tmp_path, tmp_path / "o.npy")


@pytest.mark.req("DATA-01")
def test_build_sevir_memmap_shape_mismatch_and_missing(sevir_files, tmp_path):
    rows, vil, ir_raw, S, T = sevir_files
    _write_h5(tmp_path / "ir.h5", sevir.IR107_KEY, ir_raw[..., :-1], ["S100", "S200"])  # T differs
    with pytest.raises(ValueError):
        build_sevir_memmap(rows, tmp_path, tmp_path / "o.npy")
    (tmp_path / "vil.h5").unlink()
    with pytest.raises(FileNotFoundError):
        build_sevir_memmap(rows, tmp_path, tmp_path / "o2.npy")


# ---------------------------------------------------------------- DATA-08
@pytest.mark.req("DATA-08")
def test_parse_sevir_filename():
    assert parse_sevir_filename("SEVIR_IR107_STORMEVENTS_2018_0101_0630.h5") == (
        "ir107",
        2018,
        date(2018, 1, 1),
        date(2018, 6, 30),
    )
    with pytest.raises(ValueError):
        parse_sevir_filename("random.h5")


FILES = [
    ("SEVIR_IR107_STORMEVENTS_2018_0101_0331.h5", 10 * GB),
    ("SEVIR_IR107_STORMEVENTS_2018_0401_0630.h5", 10 * GB),
    ("SEVIR_IR107_STORMEVENTS_2018_0701_1231.h5", 20 * GB),
    ("SEVIR_IR107_STORMEVENTS_2019_0101_0630.h5", 15 * GB),
    ("SEVIR_IR107_STORMEVENTS_2019_0701_1231.h5", 15 * GB),
]  # total 70 GB


@pytest.mark.req("DATA-08")
def test_ir107_cap_under_cap_keeps_all():
    r = apply_ir107_cap(FILES, cap_bytes=70 * GB)
    assert r.kept == [f for f, _ in FILES] and r.dropped == [] and r.total_bytes == 70 * GB and not r.over_cap


@pytest.mark.req("DATA-08")
def test_ir107_cap_drops_earliest_2018_h1_first():
    r = apply_ir107_cap(FILES, cap_bytes=60 * GB)
    assert r.dropped == [FILES[0][0]]
    assert r.kept == [f for f, _ in FILES[1:]] and r.total_bytes == 60 * GB and not r.over_cap
    r = apply_ir107_cap(FILES, cap_bytes=55 * GB)
    assert r.dropped == [FILES[0][0], FILES[1][0]] and r.total_bytes == 50 * GB


@pytest.mark.req("DATA-08")
def test_ir107_cap_never_drops_2019_or_late_2018():
    r = apply_ir107_cap(FILES, cap_bytes=30 * GB)
    assert r.dropped == [FILES[0][0], FILES[1][0]]
    assert r.total_bytes == 50 * GB and r.over_cap
