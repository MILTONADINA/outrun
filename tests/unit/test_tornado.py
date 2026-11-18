"""Tornado detection + swath tests (specs/09-tornado.md). Synthetic chips only; outrun is imported inside tests."""

from __future__ import annotations

import math
from datetime import UTC, date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
VARS = ("DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH")


def make_chip(t: int = 2, a: int = 16, r: int = 32, seed: int = 0) -> dict[str, np.ndarray]:
    """Random chip: variable -> (t, 2 tilts, a azimuths, r gates) float32, no NaNs."""
    rng = np.random.default_rng(seed)
    return {v: rng.normal(size=(t, 2, a, r)).astype(np.float32) for v in VARS}


def couplet_chip(dv: float, a: int = 16, r: int = 32) -> dict[str, np.ndarray]:
    """Quiet chip with one azimuthal velocity couplet of strength ``dv`` m/s at tilt 0 of the last frame."""
    chip = {v: np.zeros((2, 2, a, r), np.float32) for v in VARS}
    chip["DBZ"][:] = 20.0
    chip["RHOHV"][:] = 0.98
    chip["VEL"][-1, 0, 5, 10] = -dv / 2
    chip["VEL"][-1, 0, 6, 10] = dv / 2
    return chip


def axes(a: int = 16, r: int = 32) -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(20.0, 51.0, r), np.linspace(100.0, 115.0, a)  # range_km, azimuth_deg


def make_scenario(t0):
    """The adapter reads only ``t0`` from a scenario (TORN-10), so a namespace stands in for Scenario."""
    return SimpleNamespace(id="synthetic", t0=t0)


def tornado_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "hazards.yaml").read_text())["hazards"]["tornado"]


# ---------------------------------------------------------------- TORN-01 chip tensor
@pytest.mark.req("TORN-01")
def test_chip_tensor_channels():
    from outrun.hazards.tornado.features import N_CHANNELS, VARIABLES, build_chip_tensor

    assert VARIABLES == VARS and N_CHANNELS == 27
    chip = make_chip()
    rng_km, az = axes()
    x = build_chip_tensor(chip, rng_km, az)
    assert x.shape == (27, 16, 32) and x.dtype == np.float32
    for v, var in enumerate(VARS):
        for t in range(2):
            np.testing.assert_allclose(x[2 * v + t], chip[var][-1, t], atol=1e-6)
            np.testing.assert_allclose(x[12 + 2 * v + t], chip[var][-1, t] - chip[var][0, t], atol=1e-5)
    np.testing.assert_array_equal(x[24], 0.0)  # no missing gates
    np.testing.assert_allclose(x[25], np.broadcast_to(rng_km / 300.0, (16, 32)), atol=1e-6)
    np.testing.assert_allclose(x[26][:, 0], np.linspace(-1, 1, 16), atol=1e-5)
    np.testing.assert_allclose(x[26], np.broadcast_to(x[26][:, :1], (16, 32)))


@pytest.mark.req("TORN-01")
def test_chip_tensor_fill_and_missing_mask():
    from outrun.hazards.tornado.features import FILL_VALUES, build_chip_tensor

    chip = make_chip()
    chip["ZDR"][-1, 1, 3, 4] = np.nan
    chip["DBZ"][-1, 0, 7, 8] = np.nan
    chip["DBZ"][0, 0, 0, 0] = np.nan  # first frame only: affects the difference, not the mask
    x = build_chip_tensor(chip, *axes())
    assert np.isfinite(x).all()
    assert x[9, 3, 4] == pytest.approx(FILL_VALUES["ZDR"])  # ZDR is variable 4, tilt 1 -> channel 9
    assert x[0, 7, 8] == pytest.approx(FILL_VALUES["DBZ"]) == pytest.approx(-20.0)
    assert x[12 + 9, 3, 4] == pytest.approx(FILL_VALUES["ZDR"] - chip["ZDR"][0, 1, 3, 4], abs=1e-5)
    assert x[12, 0, 0] == pytest.approx(chip["DBZ"][-1, 0, 0, 0] - FILL_VALUES["DBZ"], abs=1e-5)
    mask = np.zeros((16, 32), np.float32)
    mask[3, 4] = mask[7, 8] = 1.0
    np.testing.assert_array_equal(x[24], mask)
    assert np.isnan(chip["ZDR"][-1, 1, 3, 4])  # input not modified


@pytest.mark.req("TORN-01")
def test_chip_tensor_no_range_single_frame_and_errors():
    from outrun.hazards.tornado.features import N_CHANNELS_NO_RANGE, build_chip_tensor

    chip = make_chip(t=1)
    rng_km, az = axes()
    x = build_chip_tensor(chip, rng_km, az, include_range=False)
    assert x.shape == (N_CHANNELS_NO_RANGE, 16, 32) == (26, 16, 32)
    np.testing.assert_array_equal(x[12:24], 0.0)  # T == 1 -> zero differences
    np.testing.assert_allclose(x[25][:, 0], np.linspace(-1, 1, 16), atol=1e-5)  # azimuth is last
    bad = dict(chip)
    del bad["KDP"]
    with pytest.raises(KeyError):
        build_chip_tensor(bad, rng_km, az)
    with pytest.raises(ValueError):
        build_chip_tensor(chip, rng_km[:-1], az)
    bad = dict(chip)
    bad["VEL"] = np.zeros((1, 2, 16, 30), np.float32)
    with pytest.raises(ValueError):
        build_chip_tensor(bad, rng_km, az)


@pytest.mark.req("TORN-01")
def test_channel_stats_and_normalize():
    from outrun.hazards.tornado.features import fit_channel_stats, normalize

    rng = np.random.default_rng(0)
    X = rng.normal(3.0, 2.0, size=(10, 27, 4, 6)).astype(np.float32)
    X[:, 5] = 7.0  # constant channel -> std replaced by 1
    mean, std = fit_channel_stats(X)
    assert mean.shape == (27,) and std.shape == (27,)
    np.testing.assert_allclose(mean, X.mean(axis=(0, 2, 3)), rtol=1e-5, atol=1e-5)
    assert std[5] == pytest.approx(1.0)
    Z = normalize(X, mean, std)
    assert Z.dtype == np.float32 and Z is not X
    np.testing.assert_allclose(Z[:, :24].mean(axis=(0, 2, 3)), 0.0, atol=1e-4)
    np.testing.assert_array_equal(Z[:, 24:], X[:, 24:])
    np.testing.assert_allclose(Z[:, 5], 0.0, atol=1e-6)


# ---------------------------------------------------------------- TORN-02 split
def _catalog() -> pd.DataFrame:
    days = [date(2015, 1, 1) + timedelta(days=k) for k in range(0, 365, 2)]
    days += [date(2016, 1, 1) + timedelta(days=k) for k in range(0, 366, 3)]
    rows = [d for d in days for _ in range(2)]  # two chips per day
    types = ["test" if d.month == 12 else "train" for d in rows]
    return pd.DataFrame({"date": rows, "chip": range(len(rows)), "type": types})


MIXED_DAY = date(2015, 6, 15)


def _daily_catalog() -> pd.DataFrame:
    """Two chips on every calendar day of 2015, so val, train and test days are consecutive somewhere.

    The official split is the catalog ``type`` column: December is test, and MIXED_DAY holds one train and one test
    chip (TorNet assigns by storm episode, so one UTC day can hold both). Some values need strip/lower-casing.
    """
    days = [date(2015, 1, 1) + timedelta(days=k) for k in range(365)]
    rows = [d for d in days for _ in range(2)]
    cat = pd.DataFrame({"date": rows, "chip": range(len(rows))})
    cat["type"] = ["test" if d.month == 12 else "train" for d in rows]
    first_mixed = cat.index[cat["date"] == MIXED_DAY][0]
    cat.loc[first_mixed, "type"] = " Test "
    cat.loc[cat["date"] == date(2015, 3, 3), "type"] = "TRAIN"
    return cat


@pytest.mark.req("TORN-02")
def test_split_catalog_rules():
    from datetime import datetime

    from outrun.eval.splits import Split, SplitItem, tornet_split, validate_split
    from outrun.hazards.tornado.dataset import split_catalog

    cat = _daily_catalog()
    official = cat["type"].str.strip().str.lower()
    quarantine = {date(2015, 5, 1), date(2015, 5, 3), date(2015, 9, 9)}
    lab = split_catalog(cat, val_frac=0.1, seed=0, quarantine=quarantine)
    assert isinstance(lab, pd.Series) and list(lab.index) == list(cat.index)
    assert set(lab.unique()) <= {"train", "val", "test", "excluded"}
    test_mask = official == "test"
    assert (lab[test_mask] == "test").all()  # quarantine never changes test rows
    assert (lab[~test_mask] != "test").all()
    q_mask = cat["date"].isin(quarantine) & ~test_mask
    assert q_mask.any() and (lab[q_mask] == "excluded").all()
    assert (lab[cat["date"] == date(2015, 11, 30)] == "excluded").all()  # 1-day gap before the first test day
    assert (lab == "train").any() and (lab == "val").any()

    # the mixed day: the label comes from the column, not the date; its train chip falls in the test-day gap
    mixed = cat["date"] == MIXED_DAY
    assert list(lab[mixed & test_mask]) == ["test"]
    assert list(lab[mixed & ~test_mask]) == ["excluded"]
    near = cat["date"].isin({MIXED_DAY - timedelta(days=1), MIXED_DAY + timedelta(days=1)})
    assert (lab[near] == "excluded").all()

    # labels equal the frozen SPLIT-02 rule on the same samples
    samples = [
        (str(i), datetime(d.year, d.month, d.day, tzinfo=UTC), o)
        for i, d, o in zip(cat.index, cat["date"], official, strict=True)
    ]
    ref = tornet_split(samples, quarantine=quarantine, val_frac=0.1, seed=0)
    for part in ("train", "val", "test"):
        assert {it.id for it in getattr(ref, part)} == {str(i) for i in lab.index[lab == part]}

    # and they are leak-free with the 1-day gap (val and train days are consecutive in this catalog)
    def items(part: str) -> tuple[SplitItem, ...]:
        return tuple(SplitItem(str(i), cat.at[i, "date"]) for i in lab.index[lab == part])

    split = Split("tornet", items("train"), items("val"), items("test"))
    assert validate_split(split, gap_days=1) == []
    val_days = {it.day for it in split.val}
    train_days = {it.day for it in split.train}
    assert not any(abs((v - t).days) <= 1 for v in val_days for t in train_days)


@pytest.mark.req("TORN-02")
def test_split_catalog_deterministic_and_errors():
    from outrun.hazards.tornado.dataset import split_catalog

    cat = _catalog()
    a = split_catalog(cat, seed=3)
    b = split_catalog(cat, seed=3)
    pd.testing.assert_series_equal(a, b)
    renamed = cat.rename(columns={"type": "official"})
    pd.testing.assert_series_equal(split_catalog(renamed, seed=3, column="official"), a)
    small = pd.DataFrame({"date": [date(2015, 1, 1), date(2015, 1, 2)], "type": ["train", "train"]})
    assert (split_catalog(small, val_frac=0.1) == "val").sum() == 1  # at least one group
    with pytest.raises(KeyError):
        split_catalog(cat.rename(columns={"date": "day"}))
    with pytest.raises(KeyError):
        split_catalog(cat.drop(columns="type"))
    bad = small.assign(type=["train", "val"])
    with pytest.raises(ValueError):
        split_catalog(bad)


# ---------------------------------------------------------------- TORN-03 hand features
def _feature_chip() -> dict[str, np.ndarray]:
    a, r = 4, 5
    chip = {v: np.zeros((2, 2, a, r), np.float32) for v in VARS}
    chip["DBZ"][:] = 10.0
    chip["DBZ"][-1, 0, 1, 2] = 55.0
    chip["DBZ"][-1, 0, 2, 2] = 35.0
    chip["DBZ"][-1, 1, 0, 0] = 70.0  # tilt 1: ignored
    chip["DBZ"][0, 0, 0, 0] = 80.0  # first frame: ignored
    chip["RHOHV"][:] = 0.98
    chip["RHOHV"][-1, 0, 2, 2] = 0.7
    chip["RHOHV"][-1, 0, 1, 2] = 0.9
    chip["RHOHV"][-1, 0, 0, 0] = 0.5  # DBZ 10 there: not a debris candidate
    chip["VEL"][-1, 0, 1, 3] = -20.0
    chip["VEL"][-1, 0, 2, 3] = 25.0
    chip["VEL"][-1, 0, 3, 3] = np.nan  # pair (2,3)-(3,3) skipped
    chip["VEL"][-1, 1, 0, 0] = 100.0  # tilt 1: ignored
    chip["VEL"][0, 0, 0, 0] = 100.0  # first frame: ignored
    chip["ZDR"][-1, 0, 3, 4] = 3.5
    chip["ZDR"][-1, 0, 0, 1] = np.nan
    chip["ZDR"][-1, 1, 0, 0] = 9.0
    chip["KDP"][-1, 0] = np.nan  # all missing -> fill value
    chip["WIDTH"][-1, 0, 2, 1] = 7.0
    return chip


@pytest.mark.req("TORN-03")
def test_hand_features_values():
    from outrun.hazards.tornado.features import FILL_VALUES, HAND_FEATURES, hand_features, max_dv

    assert HAND_FEATURES == (
        "max_dbz",
        "max_dv",
        "min_rhohv_dbz30",
        "max_zdr",
        "max_kdp",
        "max_width",
        "range_km",
    )
    chip = _feature_chip()
    f = hand_features(chip, 42.0)
    assert f.shape == (7,) and f.dtype == np.float64
    expected = [55.0, 45.0, 0.7, 3.5, FILL_VALUES["KDP"], 7.0, 42.0]
    np.testing.assert_allclose(f, expected, atol=1e-5)
    assert max_dv(chip) == pytest.approx(45.0)
    assert max_dv(chip, tilt=1) == pytest.approx(100.0)


@pytest.mark.req("TORN-03")
def test_hand_features_empty_cases():
    from outrun.hazards.tornado.features import hand_features, max_dv

    chip = {v: np.zeros((1, 2, 3, 3), np.float32) for v in VARS}
    chip["DBZ"][:] = 10.0  # no gate above 30 dBZ
    chip["VEL"][:] = np.nan  # no valid velocity pair
    f = hand_features(chip, 80.0)
    assert f[1] == pytest.approx(0.0)
    assert f[2] == pytest.approx(1.0)
    assert max_dv(chip) == pytest.approx(0.0)


# ---------------------------------------------------------------- TORN-04 CNN
@pytest.mark.req("TORN-04")
def test_cnn_shapes_size_agnostic_and_errors():
    from outrun.hazards.tornado.model import TornadoCNN

    torch.manual_seed(0)
    m = TornadoCNN(in_channels=27, widths=(4, 8, 8, 8), hidden=8).eval()
    assert m(torch.rand(3, 27, 16, 32)).shape == (3,)
    assert m(torch.rand(2, 27, 24, 40)).shape == (2,)
    with pytest.raises(ValueError):
        m(torch.rand(1, 26, 16, 32))
    m26 = TornadoCNN(in_channels=26, widths=(4, 8), hidden=8).eval()  # no-range ablation
    assert m26(torch.rand(1, 26, 8, 8)).shape == (1,)


@pytest.mark.req("TORN-04")
def test_cnn_default_parameter_count():
    from outrun.hazards.tornado.model import TornadoCNN

    n = sum(p.numel() for p in TornadoCNN().parameters() if p.requires_grad)
    assert 500_000 <= n <= 3_000_000


@pytest.mark.req("TORN-04")
def test_cnn_predict_proba_range_determinism_batch_independence():
    from outrun.hazards.tornado.model import TornadoCNN, predict_proba

    x = torch.randn(4, 27, 16, 32, generator=torch.Generator().manual_seed(2))
    torch.manual_seed(11)
    a = TornadoCNN(widths=(4, 8, 8, 8), hidden=8)
    torch.manual_seed(11)
    b = TornadoCNN(widths=(4, 8, 8, 8), hidden=8)
    p = predict_proba(a, x)
    assert p.shape == (4,)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0
    torch.testing.assert_close(p, predict_proba(a, x))
    torch.testing.assert_close(p, predict_proba(b, x))
    torch.testing.assert_close(predict_proba(a, x[1:2]), p[1:2], atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------- TORN-05 focal loss, calibration
@pytest.mark.req("TORN-05")
def test_focal_loss_values():
    from outrun.hazards.tornado.model import focal_loss

    g = torch.Generator().manual_seed(0)
    logits = torch.randn(32, generator=g) * 3
    targets = (torch.rand(32, generator=g) > 0.7).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    torch.testing.assert_close(focal_loss(logits, targets, gamma=0.0), bce, rtol=1e-5, atol=1e-6)
    assert float(focal_loss(torch.zeros(1), torch.ones(1))) == pytest.approx(0.25 * math.log(2), rel=1e-5)
    got = focal_loss(torch.zeros(2), torch.tensor([1.0, 0.0]), gamma=2.0, alpha=0.25)
    assert float(got) == pytest.approx(0.125 * math.log(2), rel=1e-5)
    easy = focal_loss(torch.tensor([5.0]), torch.tensor([1.0]))
    easy_bce = torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([5.0]), torch.tensor([1.0]))
    p = torch.sigmoid(torch.tensor(5.0))
    assert float(easy) == pytest.approx(float((1 - p) ** 2 * easy_bce), rel=1e-4)
    extreme = focal_loss(torch.tensor([100.0, -100.0]), torch.tensor([0.0, 1.0]))
    assert torch.isfinite(extreme) and float(extreme) > 0


@pytest.mark.req("TORN-05")
def test_isotonic_calibrator_base_rate_and_monotone():
    from outrun.hazards.tornado.calibration import IsotonicCalibrator

    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(int)
    p = np.clip(0.5 + 0.3 * y + rng.normal(0, 0.2, 2000), 0, 1)  # scores far above the 10% base rate
    cal = IsotonicCalibrator().fit(p, y)
    q = cal.transform(p)
    assert abs(q.mean() - y.mean()) < 1e-6
    grid = np.linspace(-0.5, 1.5, 41)
    out = cal.transform(grid)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(np.diff(out) >= -1e-12)
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().transform(p)


# ---------------------------------------------------------------- TORN-06 baselines
@pytest.mark.req("TORN-06")
def test_delta_v_rule():
    from outrun.hazards.tornado.baselines import delta_v_rule

    assert delta_v_rule(couplet_chip(50.0)) == 1.0
    assert delta_v_rule(couplet_chip(30.0)) == 0.0
    assert delta_v_rule(couplet_chip(40.0)) == 1.0  # inclusive
    assert delta_v_rule(couplet_chip(30.0), threshold=25.0) == 1.0
    chip = couplet_chip(10.0)
    chip["VEL"][-1, 1, 5, 10], chip["VEL"][-1, 1, 6, 10] = -40.0, 40.0  # strong couplet at tilt 1 only
    assert delta_v_rule(chip) == 0.0


@pytest.mark.req("TORN-06")
def test_gbm_baseline_separable_and_deterministic():
    from outrun.hazards.tornado.baselines import gbm_predict_proba, train_gbm

    rng = np.random.default_rng(0)
    n = 300
    y = (rng.random(n) < 0.3).astype(int)
    X = rng.normal(size=(n, 7))
    X[:, 1] += 4.0 * y  # shear separates the classes
    X[rng.random(n) < 0.05, 3] = np.nan  # missing values are allowed
    m1 = train_gbm(X, y, seed=0)
    m2 = train_gbm(X, y, seed=0)
    p1 = gbm_predict_proba(m1, X)
    assert p1.shape == (n,) and p1.min() >= 0 and p1.max() <= 1
    assert roc_auc_score(y, p1) > 0.95
    np.testing.assert_array_equal(p1, gbm_predict_proba(m2, X))


@pytest.mark.req("TORN-06")
def test_range_only_model_runs():
    from outrun.hazards.tornado.baselines import range_only_predict_proba, train_range_only

    rng = np.random.default_rng(1)
    r = rng.uniform(5, 200, 200)
    y = (rng.random(200) < 0.2).astype(int)
    p = range_only_predict_proba(train_range_only(r, y), r)
    assert p.shape == (200,) and p.min() >= 0 and p.max() <= 1


# ---------------------------------------------------------------- TORN-07 scores
@pytest.mark.req("TORN-07")
def test_tornado_scores_hand_computed():
    from outrun.hazards.tornado.metrics import tornado_scores

    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    cat = np.array(["TOR"] * 4 + ["WRN"] * 4 + ["NUL"] * 4)
    p = np.array([0.9, 0.8, 0.6, 0.4, 0.7, 0.5, 0.85, 0.3, 0.05, 0.1, 0.02, 0.0])
    rng_km = np.array([10, 20, 60, 70, 30, 40, 80, 90, 110, 120, 130, 140], float)
    s = tornado_scores(y, p, cat, rng_km, threshold=0.5)
    assert s["auc"] == pytest.approx(26 / 32)
    assert s["auc_warned"] == pytest.approx(10 / 16)  # random nulls excluded: the hard case
    assert s["auprc"] == pytest.approx(average_precision_score(y, p))
    assert s["auc_by_range"]["0-50"] == pytest.approx(1.0)
    assert s["auc_by_range"]["50-100"] == pytest.approx(0.5)
    assert math.isnan(s["auc_by_range"]["100+"])  # negatives only
    assert s["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert s["pod"] == pytest.approx(0.75)
    assert s["far"] == pytest.approx(0.5)
    assert s["csi"] == pytest.approx(3 / 7)


@pytest.mark.req("TORN-07")
def test_tornado_scores_degenerate_and_errors():
    from outrun.hazards.tornado.metrics import tornado_scores

    y = np.zeros(4, int)
    s = tornado_scores(y, np.array([0.1, 0.2, 0.3, 0.4]), np.array(["NUL"] * 4), np.full(4, 30.0))
    assert math.isnan(s["auc"]) and math.isnan(s["pod"]) and math.isnan(s["far"])
    with pytest.raises(ValueError):
        tornado_scores(y, np.zeros(3), np.array(["NUL"] * 4), np.full(4, 30.0))


# ---------------------------------------------------------------- TORN-08 swath
GX = np.arange(-60_000.0, 60_001.0, 1000.0)  # 121 cells, includes 0
GY = GX[::-1].copy()  # north-up: y decreases with row
IX0 = IY0 = 60  # index of coordinate 0


def _cell(xm: float, ym: float) -> tuple[int, int]:
    return int(np.argmin(np.abs(GY - ym))), int(np.argmin(np.abs(GX - xm)))


@pytest.mark.req("TORN-08")
def test_swath_basic_properties():
    from outrun.hazards.tornado.swath import project_swath

    s = project_swath(0.8, 0.0, 0.0, (10.0, 0.0), GX, GY)
    assert s.shape == (12, 121, 121) and s.dtype == np.float32
    assert s.min() >= 0.0 and s.max() <= 0.8 + 1e-6
    assert s[0, IY0, IX0] == pytest.approx(0.8, abs=1e-5)
    np.testing.assert_array_equal(s[2:], 0.0)  # beyond the 30 min horizon
    np.testing.assert_allclose(s, s[:, ::-1, :], atol=1e-6)  # symmetric across an eastward track
    i, j = np.unravel_index(np.argmax(s[1]), s[1].shape)
    assert GY[i] == 0.0 and 9_000 <= GX[j] <= 18_000  # moved downstream in slot 1
    assert s[:, :, :5].max() < 1e-6  # more than 50 km upstream
    assert s[:, :5, :].max() < 1e-6  # more than 50 km across track


@pytest.mark.req("TORN-08")
def test_swath_widens_with_time_and_is_anisotropic():
    from outrun.hazards.tornado.swath import project_swath

    s = project_swath(1.0, 0.0, 0.0, (10.0, 0.0), GX, GY)
    early = s[0][_cell(3_000, 6_000)]  # 6 km across track from the 5 min position
    late = s[1][_cell(12_000, 6_000)]  # 6 km across track from the 20 min position
    assert late > early
    still = project_swath(1.0, 0.0, 0.0, (0.0, 0.0), GX, GY)  # zero motion: along-track axis is east
    assert still[1][_cell(0, 6_000)] > still[1][_cell(6_000, 0)]


@pytest.mark.req("TORN-08")
def test_swath_errors_and_combine():
    from outrun.hazards.tornado.swath import combine_swaths, project_swath

    for bad in (-0.1, 1.2):
        with pytest.raises(ValueError):
            project_swath(bad, 0.0, 0.0, (10.0, 0.0), GX, GY)
    a = np.zeros((12, 3, 3), np.float32)
    b = np.zeros((12, 3, 3), np.float32)
    a[0, 1, 1], b[0, 1, 1], b[1, 0, 0] = 0.3, 0.6, 0.2
    c = combine_swaths([a, b])
    assert c[0, 1, 1] == pytest.approx(0.6) and c[1, 0, 0] == pytest.approx(0.2)
    np.testing.assert_array_equal(combine_swaths([a]), a)


# ---------------------------------------------------------------- TORN-09 motion fallback
@pytest.mark.req("TORN-09")
def test_resolve_motion_order():
    from outrun.hazards.tornado.swath import resolve_motion

    assert resolve_motion((3.0, -4.0), (10.0, 0.0)) == pytest.approx((3.0, -4.0, "storm"))
    u, v, src = resolve_motion(None, (10.0, 0.0))
    assert src == "hrrr"
    assert (u, v) == pytest.approx((0.75 * 10 * math.cos(math.radians(30)), -0.75 * 10 * 0.5))
    u, v, src = resolve_motion((float("nan"), 1.0), (0.0, 8.0))  # non-finite storm motion -> HRRR
    assert src == "hrrr" and (u, v) == pytest.approx((0.75 * 8 * 0.5, 0.75 * 8 * math.cos(math.radians(30))))
    u, v, src = resolve_motion(None, None)
    assert src == "default"
    assert (u, v) == pytest.approx((15 * math.sin(math.radians(60)), 15 * math.cos(math.radians(60))))
    assert u > 0 and v > 0  # from 240 deg (WSW) means moving toward the ENE


# ---------------------------------------------------------------- TORN-10 adapter
AX = np.arange(-20_000.0, 20_001.0, 1000.0)  # 41 cells
AY = AX[::-1].copy()


def _inputs(dvs=(50.0,), observed=True, motion=(10.0, 0.0), held_out=True):
    from outrun.hazards.tornado.adapter import TornadoDetection, TornadoInputs

    rng_km, az = axes()
    dets = [
        TornadoDetection(
            chip=couplet_chip(dv), range_km=rng_km, azimuth_deg=az, center_range_km=35.0, x_m=0.0, y_m=0.0
        )
        for dv in dvs
    ]
    obs = np.zeros((12, 41, 41), np.float32) if observed else None
    if obs is not None:
        obs[0:2, 20, 20:25] = 1.0
    return TornadoInputs(
        detections=dets,
        x=AX,
        y=AY,
        storm_motion=(lambda x, y: motion) if motion is not None else None,
        hrrr_wind=None,
        observed=obs,
        held_out=held_out,
    )


def _check_layer(layer, t0, kind, inputs):
    from outrun.core.risk import Hazard

    assert layer.hazard == Hazard.TORNADO
    assert layer.kind == kind
    assert layer.issued == t0
    assert layer.valid_until == t0 + timedelta(minutes=tornado_cfg()["native_horizon_min"])
    assert layer.valid_until == t0 + timedelta(minutes=30)
    assert layer.source.startswith("tornado.")
    assert layer.held_out == inputs.held_out
    assert tuple(layer.prob.dims) == ("slot", "y", "x")
    assert layer.prob.shape == (12, 41, 41)
    assert layer.prob.dtype == np.float16
    v = layer.prob.values.astype(np.float32)
    assert np.isfinite(v).all() and v.min() >= 0.0 and v.max() <= 1.0
    return v


@pytest.mark.req("TORN-10")
def test_adapter_policy_matches_config():
    from outrun.core.policy import Profile
    from outrun.core.risk import Hazard
    from outrun.hazards.tornado.adapter import TornadoAdapter

    cfg = tornado_cfg()
    ad = TornadoAdapter(provider=lambda scn: _inputs())
    assert ad.hazard is Hazard.TORNADO
    pol = ad.policy
    for prof in Profile:
        assert pol.block_at[prof] == pytest.approx(cfg["block_at"][prof.name.lower()])
        assert pol.block_at[prof] == pytest.approx(0.5)
        assert pol.exposure_weight[prof] == pytest.approx(cfg["exposure_weight"][prof.name.lower()])
    assert pol.exposure_weight[Profile.CIVILIAN] == pytest.approx(20.0)
    assert pol.beyond_horizon == "drop" == cfg["beyond_horizon"]


@pytest.mark.req("TORN-10")
def test_adapter_baseline_delta_v_swath(t0):
    from outrun.hazards.tornado.adapter import TornadoAdapter

    inp = _inputs(dvs=(50.0,))
    v = _check_layer(
        TornadoAdapter(provider=lambda scn: inp).baseline(make_scenario(t0)), t0, "baseline", inp
    )
    assert v[0, 20, 20] == pytest.approx(1.0, abs=1e-3)  # rule fires at the chip centre
    assert v[1].max() > 0.5 and v[1, 20, 30] > v[1, 20, 10]  # carried east by the storm motion
    np.testing.assert_array_equal(v[3:], 0.0)  # dropped after the horizon (+15 min linger at most)
    quiet = _inputs(dvs=(10.0,))
    vq = TornadoAdapter(provider=lambda scn: quiet).baseline(make_scenario(t0)).prob.values
    np.testing.assert_array_equal(vq, 0.0)
    empty = _inputs(dvs=())
    ve = TornadoAdapter(provider=lambda scn: empty).baseline(make_scenario(t0)).prob.values
    np.testing.assert_array_equal(ve, 0.0)


@pytest.mark.req("TORN-10")
def test_adapter_baseline_uses_motion_fallback(t0):
    from outrun.hazards.tornado.adapter import TornadoAdapter

    inp = _inputs(dvs=(50.0,), motion=None)  # no storm motion, no HRRR -> 15 m/s from 240 deg (toward ENE)
    v = TornadoAdapter(provider=lambda scn: inp).baseline(make_scenario(t0)).prob.values.astype(np.float32)
    i, j = np.unravel_index(np.argmax(v[1]), v[1].shape)
    assert AX[j] > 0 and AY[i] > 0


@pytest.mark.req("TORN-10")
def test_adapter_predict_with_model(t0):
    from outrun.hazards.tornado.adapter import TornadoAdapter
    from outrun.hazards.tornado.model import TornadoCNN

    inp = _inputs(dvs=(50.0, 20.0))
    with pytest.raises(RuntimeError):
        TornadoAdapter(provider=lambda scn: inp).predict(make_scenario(t0))
    torch.manual_seed(0)
    ad = TornadoAdapter(provider=lambda scn: inp, model=TornadoCNN(widths=(4, 8, 8, 8), hidden=8))
    a = ad.predict(make_scenario(t0))
    v = _check_layer(a, t0, "forecast", inp)
    np.testing.assert_array_equal(v[3:], 0.0)
    np.testing.assert_array_equal(a.prob.values, ad.predict(make_scenario(t0)).prob.values)


@pytest.mark.req("TORN-10")
def test_adapter_observed(t0):
    from outrun.hazards.tornado.adapter import TornadoAdapter

    inp = _inputs()
    v = _check_layer(
        TornadoAdapter(provider=lambda scn: inp).observed(make_scenario(t0)), t0, "observed", inp
    )
    np.testing.assert_array_equal(v, inp.observed)
    none = _inputs(observed=False)
    assert TornadoAdapter(provider=lambda scn: none).observed(make_scenario(t0)) is None


@pytest.mark.req("TORN-10")
def test_tornado_adapter_zero_argument_construction_and_default_factory():
    from outrun.config import get_paths
    from outrun.core.risk import Hazard
    from outrun.hazards.tornado.adapter import TornadoAdapter, default_adapter

    ad = TornadoAdapter()  # zero-argument factory for the registry (RISK-13); nothing is loaded
    assert ad.hazard is Hazard.TORNADO and ad.policy.beyond_horizon == "drop"
    d = default_adapter()
    assert isinstance(d, TornadoAdapter) and d.checkpoint == get_paths().models / "tornado" / "best.pt"
