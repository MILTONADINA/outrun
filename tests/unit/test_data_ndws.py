"""specs/01-data.md DATA-05: NDWS TFRecord -> npy, percentile clip/normalize. Tiny TFRecords written in tmp_path."""

from __future__ import annotations

import numpy as np
import pytest

from outrun.data.ndws import (
    NDWS_INPUT_FEATURES,
    NDWS_LABEL,
    convert_ndws,
    fit_clip_stats,
    normalize,
    parse_ndws_example,
    read_ndws_tfrecord,
)

S = 4  # tiny patches; production size is 64


def _example(rng, k):
    feats = {
        name: (rng.random(S * S) + 10 * i + k).astype(np.float32)
        for i, name in enumerate(NDWS_INPUT_FEATURES)
    }
    lab = np.zeros(S * S, np.float32)
    lab[0], lab[1] = 1.0, -1.0
    feats[NDWS_LABEL] = lab
    return feats


def _write_tfrecord(path, examples):
    from tfrecord import TFRecordWriter

    w = TFRecordWriter(str(path))
    for ex in examples:
        w.write({k: (v.tolist(), "float") for k, v in ex.items()})
    w.close()


@pytest.mark.req("DATA-05")
def test_parse_example_layout(rng):
    ex = _example(rng, 0)
    x, y = parse_ndws_example(ex, size=S)
    assert x.shape == (12, S, S) and x.dtype == np.float32
    assert y.shape == (S, S) and y.dtype == np.int8
    for c, name in enumerate(NDWS_INPUT_FEATURES):
        np.testing.assert_array_equal(x[c], ex[name].reshape(S, S))
    assert y[0, 0] == 1 and y[0, 1] == -1 and y[1, 1] == 0


@pytest.mark.req("DATA-05")
def test_parse_example_errors(rng):
    ex = _example(rng, 0)
    bad = dict(ex)
    bad.pop("erc")
    with pytest.raises(KeyError):
        parse_ndws_example(bad, size=S)
    bad = dict(ex, pr=np.zeros(S * S + 1, np.float32))
    with pytest.raises(ValueError):
        parse_ndws_example(bad, size=S)
    lab = ex[NDWS_LABEL].copy()
    lab[3] = 0.5
    with pytest.raises(ValueError):
        parse_ndws_example(dict(ex, **{NDWS_LABEL: lab}), size=S)


@pytest.mark.req("DATA-05")
def test_read_tfrecord_in_file_order(tmp_path, rng):
    exs = [_example(rng, k) for k in range(3)]
    _write_tfrecord(tmp_path / "a.tfrecord", exs)
    X, y = read_ndws_tfrecord(tmp_path / "a.tfrecord", size=S)
    assert X.shape == (3, 12, S, S) and y.shape == (3, S, S)
    for k in range(3):
        np.testing.assert_allclose(X[k], parse_ndws_example(exs[k], size=S)[0], rtol=1e-6)


@pytest.mark.req("DATA-05")
def test_convert_ndws_writes_split_files(tmp_path, rng):
    exs = [_example(rng, k) for k in range(4)]
    _write_tfrecord(tmp_path / "train_01.tfrecord", exs[2:])
    _write_tfrecord(tmp_path / "train_00.tfrecord", exs[:2])
    _write_tfrecord(tmp_path / "eval_00.tfrecord", exs[:1])
    out = tmp_path / "out"
    out.mkdir()
    res = convert_ndws(
        {
            "train": [tmp_path / "train_01.tfrecord", tmp_path / "train_00.tfrecord"],
            "eval": [tmp_path / "eval_00.tfrecord"],
        },
        out,
        size=S,
    )
    assert res["train"] == (out / "ndws_train.npy", out / "ndws_train_labels.npy")
    X = np.load(out / "ndws_train.npy")
    y = np.load(out / "ndws_train_labels.npy")
    assert X.shape == (4, 12, S, S) and y.dtype == np.int8
    for k in range(4):  # sorted path order: train_00 then train_01
        np.testing.assert_allclose(X[k], parse_ndws_example(exs[k], size=S)[0], rtol=1e-6)
    assert np.load(out / "ndws_eval.npy").shape[0] == 1
    with pytest.raises(ValueError):
        convert_ndws({"validation": [tmp_path / "eval_00.tfrecord"]}, out, size=S)


@pytest.mark.req("DATA-05")
def test_clip_stats_and_normalize(rng):
    X = rng.normal(5.0, 2.0, size=(50, 12, S, S)).astype(np.float32)
    X[0, 0, 0, 0] = 1e6  # outlier, clipped
    stats = fit_clip_stats(X, lo_pct=1.0, hi_pct=99.0)
    assert set(stats) == set(NDWS_INPUT_FEATURES) - {"th", "PrevFireMask"}
    assert stats["elevation"]["hi"] < 1e6
    Z = normalize(X, stats)
    assert Z.dtype == np.float32 and Z.shape == X.shape
    th, fire = NDWS_INPUT_FEATURES.index("th"), NDWS_INPUT_FEATURES.index("PrevFireMask")
    np.testing.assert_array_equal(Z[:, th], X[:, th])
    np.testing.assert_array_equal(Z[:, fire], X[:, fire])
    c = NDWS_INPUT_FEATURES.index("vs")
    assert abs(float(Z[:, c].mean())) < 1e-3 and abs(float(Z[:, c].std()) - 1.0) < 1e-2
    const = np.ones((3, 12, S, S), np.float32)
    assert np.isfinite(normalize(const, fit_clip_stats(const))).all()
