"""specs/11-wildfire.md: channels, normalisation, satellite channels, augmentation, model, loss, baselines,
metrics, hourly growth, observed layer, dataset. Synthetic arrays only (offline, CPU, seconds).
Adapter/RiskLayer tests are in test_wildfire_adapter.py.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from outrun.hazards.wildfire import baselines, dataset, growth, metrics, model

TH = 1  # index of wind direction in NDWS_CHANNELS
PREV = 11  # index of PrevFireMask


def _sample(h=16, w=16, th=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((12, h, w)).astype(np.float32)
    x[TH] = th
    x[PREV] = 0.0
    return x


# ---------------------------------------------------------------- FIRE-01
@pytest.mark.req("FIRE-01")
def test_ndws_channel_order():
    assert dataset.NDWS_CHANNELS == (
        "elevation",
        "th",
        "vs",
        "tmmn",
        "tmmx",
        "sph",
        "pr",
        "pdsi",
        "NDVI",
        "population",
        "erc",
        "PrevFireMask",
    )


@pytest.mark.req("FIRE-01")
def test_norm_stats_standardise_train_and_identity_mask():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, size=(6, 12, 8, 8)).astype(np.float32)
    x[:, PREV] = rng.integers(-1, 2, size=(6, 8, 8))
    x[:, 9] = 7.0  # constant channel
    st = dataset.fit_norm_stats(x, lo_pct=0.0, hi_pct=100.0)
    for arr in (st.lo, st.hi, st.mean, st.std):
        assert arr.shape == (12,)
    assert (st.lo[PREV], st.hi[PREV], st.mean[PREV], st.std[PREV]) == (-1, 1, 0, 1)
    assert st.std[9] == 1.0
    z = dataset.normalize(x, st)
    assert z.dtype == np.float32 and z.shape == x.shape
    for c in [i for i in range(12) if i not in (9, PREV)]:
        assert st.lo[c] == pytest.approx(x[:, c].min()) and st.hi[c] == pytest.approx(x[:, c].max())
        assert z[:, c].mean() == pytest.approx(0.0, abs=1e-4)
        assert z[:, c].std() == pytest.approx(1.0, abs=1e-3)
    np.testing.assert_array_equal(z[:, PREV], x[:, PREV])
    np.testing.assert_allclose(z[:, 9], 0.0)


@pytest.mark.req("FIRE-01")
def test_normalize_clips_handles_nan_and_does_not_mutate():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 12, 8, 8)).astype(np.float32)
    st = dataset.fit_norm_stats(x)  # default 0.1 / 99.9 percentiles
    for c in range(11):
        assert st.lo[c] == pytest.approx(np.percentile(x[:, c], 0.1), rel=1e-4, abs=1e-5)
        assert st.hi[c] == pytest.approx(np.percentile(x[:, c], 99.9), rel=1e-4, abs=1e-5)
    s = x[0].copy()
    s[0, 0, 0] = 1e6
    s[2, 1, 1] = np.nan
    before = s.copy()
    z = dataset.normalize(s, st)
    assert z.shape == (12, 8, 8)
    assert z[0, 0, 0] == pytest.approx((st.hi[0] - st.mean[0]) / st.std[0], rel=1e-5)
    assert z[2, 1, 1] == 0.0
    np.testing.assert_array_equal(np.isnan(s), np.isnan(before))
    np.testing.assert_array_equal(np.nan_to_num(s), np.nan_to_num(before))
    with pytest.raises(ValueError):
        dataset.normalize(np.zeros((11, 8, 8), dtype=np.float32), st)


# ---------------------------------------------------------------- FIRE-02
@pytest.mark.req("FIRE-02")
def test_satellite_channels_are_model_inputs():
    assert dataset.SATELLITE_CHANNELS == ("NDVI", "PrevFireMask")
    assert set(dataset.SATELLITE_CHANNELS) <= set(dataset.NDWS_CHANNELS)
    assert dataset.channel_indices(dataset.SATELLITE_CHANNELS) == [8, 11]
    assert dataset.channel_indices(["elevation", "erc"]) == [0, 10]
    with pytest.raises(KeyError):
        dataset.channel_indices(["MODIS_band_31"])
    torch.manual_seed(0)
    net = model.FireUNet()
    assert net(torch.randn(1, len(dataset.NDWS_CHANNELS), 32, 32)).shape == (1, 1, 32, 32)


@pytest.mark.req("FIRE-02")
def test_check_channel_order_guards_wsts_arrays():
    assert dataset.check_channel_order(list(dataset.NDWS_CHANNELS)) is None
    swapped = list(dataset.NDWS_CHANNELS)
    swapped[8], swapped[9] = swapped[9], swapped[8]
    with pytest.raises(ValueError):
        dataset.check_channel_order(swapped)
    with pytest.raises(ValueError):
        dataset.check_channel_order(dataset.NDWS_CHANNELS[:-1])
    with pytest.raises(ValueError):
        dataset.check_channel_order((*dataset.NDWS_CHANNELS, "extra"))


@pytest.mark.req("FIRE-02")
def test_ablate_satellite_channels():
    x = np.random.default_rng(0).random((2, 12, 8, 8)).astype(np.float32) + 1.0
    before = x.copy()
    a = dataset.ablate_channels(x, dataset.SATELLITE_CHANNELS)
    assert a.shape == x.shape
    assert np.all(a[:, [8, 11]] == 0)
    keep = [i for i in range(12) if i not in (8, 11)]
    np.testing.assert_array_equal(a[:, keep], x[:, keep])
    np.testing.assert_array_equal(x, before)
    single = dataset.ablate_channels(x[0], ["NDVI"])
    assert np.all(single[8] == 0) and np.all(single[11] == x[0, 11])


# ---------------------------------------------------------------- FIRE-03
def _downwind_offset(th_deg: float, scale: int = 3) -> tuple[int, int]:
    """Image (d_row, d_col) of a step of length `scale` toward bearing th + 180 (row 0 = north)."""
    b = np.radians(th_deg + 180.0)
    return int(np.rint(-scale * np.cos(b))), int(np.rint(scale * np.sin(b)))


DIHEDRAL = list(itertools.product([False, True], [False, True], [0, 1, 2, 3]))


@pytest.mark.req("FIRE-03")
@pytest.mark.parametrize("th", [0.0, 45.0, 90.0, 135.0, 225.0, 270.0])
@pytest.mark.parametrize("hflip,vflip,k", DIHEDRAL)
def test_augment_keeps_wind_consistent_with_image(th, hflip, vflip, k):
    x = np.zeros((12, 32, 32), dtype=np.float32)
    x[TH] = th
    fire = (12, 14)
    dr, dc = _downwind_offset(th)
    x[PREV][fire] = 1.0
    x[0][fire[0] + dr, fire[1] + dc] = 1.0  # marker one step downwind (elevation channel)
    xa, _ = dataset.augment(x, x[PREV].copy(), hflip=hflip, vflip=vflip, k=k)
    th_new = np.unique(xa[TH])
    assert th_new.size == 1 and 0.0 <= th_new[0] < 360.0
    f = np.argwhere(xa[PREV] == 1.0)[0]
    m = np.argwhere(xa[0] == 1.0)[0]
    assert tuple(m - f) == _downwind_offset(float(th_new[0]))


@pytest.mark.req("FIRE-03")
def test_augment_formulas_and_spatial_transform():
    rng = np.random.default_rng(0)
    x = rng.random((12, 8, 8)).astype(np.float32)
    x[TH] = 30.0
    y = rng.integers(-1, 2, size=(8, 8)).astype(np.int8)
    xb, yb = x.copy(), y.copy()
    xa, ya = dataset.augment(x, y, hflip=True)
    np.testing.assert_allclose(xa[TH], 330.0)
    np.testing.assert_array_equal(xa[0], x[0][:, ::-1])
    np.testing.assert_array_equal(ya, y[:, ::-1])
    xa, ya = dataset.augment(x, y, vflip=True)
    np.testing.assert_allclose(xa[TH], 150.0)
    np.testing.assert_array_equal(xa[5], x[5][::-1, :])
    np.testing.assert_array_equal(ya, y[::-1, :])
    xa, ya = dataset.augment(x, y, k=1)
    np.testing.assert_allclose(xa[TH], 300.0)
    np.testing.assert_array_equal(xa[3], np.rot90(x[3], 1))
    np.testing.assert_array_equal(ya, np.rot90(y, 1))
    xa, ya = dataset.augment(x, y, hflip=True, vflip=True, k=3)
    np.testing.assert_allclose(xa[TH], (180.0 - (360.0 - 30.0) - 270.0) % 360.0)
    np.testing.assert_array_equal(ya, np.rot90(y[::-1, ::-1], 3))
    x5, _ = dataset.augment(x, y, k=5)
    x1, _ = dataset.augment(x, y, k=1)
    np.testing.assert_array_equal(x5, x1)
    np.testing.assert_array_equal(x, xb)
    np.testing.assert_array_equal(y, yb)


@pytest.mark.req("FIRE-03")
def test_random_augment_is_seeded():
    x = _sample(th=60.0)
    y = np.random.default_rng(3).integers(0, 2, size=(16, 16)).astype(np.int8)
    a = dataset.random_augment(x, y, np.random.default_rng(42))
    b = dataset.random_augment(x, y, np.random.default_rng(42))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    candidates = [dataset.augment(x, y, hf, vf, k) for hf, vf, k in DIHEDRAL]
    assert any(np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1]) for c in candidates)


# ---------------------------------------------------------------- FIRE-04
@pytest.mark.req("FIRE-04")
def test_fire_unet_shapes_params_errors():
    torch.manual_seed(0)
    net = model.FireUNet()
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert 500_000 <= n_params <= 5_000_000
    assert net(torch.randn(2, 12, 64, 64)).shape == (2, 1, 64, 64)
    assert net(torch.randn(1, 12, 32, 32)).shape == (1, 1, 32, 32)
    with pytest.raises(ValueError):
        net(torch.randn(1, 11, 32, 32))


@pytest.mark.req("FIRE-04")
def test_fire_unet_gradients_and_eval_determinism():
    torch.manual_seed(0)
    net = model.FireUNet()
    net(torch.randn(2, 12, 32, 32)).mean().backward()
    for name, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, name
    net.eval()
    x = torch.randn(1, 12, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(net(x), net(x), rtol=0, atol=0)


# ---------------------------------------------------------------- FIRE-05
def _rand_lt(seed, shape=(2, 1, 8, 8)):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(*shape, generator=g)
    target = (torch.rand(*shape, generator=g) > 0.7).float()
    target[0, 0, 0, :] = -1
    return logits, target


@pytest.mark.req("FIRE-05")
def test_fire_loss_formula():
    logits, target = _rand_lt(0)
    v = target != -1
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[v], target[v], pos_weight=torch.tensor(3.0)
    )
    assert float(model.fire_loss(logits, target, dice_weight=0.0)) == pytest.approx(float(bce), rel=1e-5)
    p = torch.sigmoid(logits[v])
    t = target[v]
    dice = 1 - (2 * (p * t).sum() + 1.0) / (p.sum() + t.sum() + 1.0)
    full = model.fire_loss(logits, target)
    assert float(full) == pytest.approx(float(bce + dice), rel=1e-5)
    half = model.fire_loss(logits, target, pos_weight=1.0, dice_weight=0.5)
    bce1 = torch.nn.functional.binary_cross_entropy_with_logits(logits[v], t)
    assert float(half) == pytest.approx(float(bce1 + 0.5 * dice), rel=1e-5)


@pytest.mark.req("FIRE-05")
def test_fire_loss_ignores_uncertain_pixels():
    logits, target = _rand_lt(1)
    logits.requires_grad_(True)
    loss = model.fire_loss(logits, target)
    loss.backward()
    assert torch.all(logits.grad[0, 0, 0, :] == 0)
    changed = logits.detach().clone()
    changed[0, 0, 0, :] = -40.0
    assert float(model.fire_loss(changed, target)) == pytest.approx(float(loss), rel=1e-6)


@pytest.mark.req("FIRE-05")
def test_fire_loss_all_ignored_and_shape_mismatch():
    logits = torch.randn(1, 1, 4, 4, requires_grad=True)
    out = model.fire_loss(logits, torch.full((1, 1, 4, 4), -1.0))
    assert torch.isfinite(out) and float(out) == 0.0
    out.backward()
    with pytest.raises(ValueError):
        model.fire_loss(torch.randn(1, 1, 4, 4), torch.zeros(1, 1, 5, 4))


# ---------------------------------------------------------------- FIRE-06
@pytest.mark.req("FIRE-06")
def test_persistence():
    x = _sample()
    x[PREV, 2, 3] = 1.0
    x[PREV, 4, 4] = -1.0
    p = baselines.persistence(x)
    assert p.shape == (16, 16) and p.dtype == np.float32
    expected = np.zeros((16, 16))
    expected[2, 3] = 1
    np.testing.assert_array_equal(p, expected)
    assert baselines.persistence(np.stack([x, x])).shape == (2, 16, 16)


@pytest.mark.req("FIRE-06")
@pytest.mark.parametrize(
    "th,neighbour",
    [(270.0, (5, 6)), (0.0, (6, 5)), (90.0, (5, 4)), (180.0, (4, 5)), (45.0, (6, 4)), (100.0, (5, 4))],
)
def test_downwind_persistence_direction(th, neighbour):
    x = _sample(th=th)
    x[PREV, 5, 5] = 1.0
    out = baselines.downwind_persistence(x)
    assert out.dtype == np.float32
    expected = np.zeros((16, 16))
    expected[5, 5] = 1
    expected[neighbour] = 1
    np.testing.assert_array_equal(out, expected)


@pytest.mark.req("FIRE-06")
def test_downwind_persistence_circular_mean_edges_batch():
    x = _sample(th=180.0)  # background wind from the south
    x[PREV, 5, 5] = x[PREV, 5, 8] = 1.0
    x[TH, 5, 5], x[TH, 5, 8] = 350.0, 10.0  # circular mean 0 (from north) -> spread south
    out = baselines.downwind_persistence(x)
    assert out[6, 5] == 1 and out[6, 8] == 1 and out[4, 5] == 0 and out.sum() == 4
    edge = _sample(th=270.0)
    edge[PREV, 3, 15] = 1.0
    e = baselines.downwind_persistence(edge)
    assert e.sum() == 1 and e[3, 15] == 1 and e[3, 0] == 0  # no wrap-around
    none = _sample(th=90.0)
    assert baselines.downwind_persistence(none).sum() == 0
    batch = baselines.downwind_persistence(np.stack([x, edge]))
    assert batch.shape == (2, 16, 16)
    np.testing.assert_array_equal(batch[0], out)


# ---------------------------------------------------------------- FIRE-07
@pytest.mark.req("FIRE-07")
def test_pixel_logistic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 12, 8, 8)).astype(np.float32)
    y = (x[:, 4] + 0.5 * x[:, 2] > 0).astype(np.int8)
    ym = y.copy()
    ym[:, 0, :] = -1
    lr = baselines.PixelLogistic(seed=0).fit(x, ym)
    p = lr.predict_proba(x)
    assert p.shape == (4, 8, 8) and p.dtype == np.float32
    assert p.min() >= 0 and p.max() <= 1
    assert roc_auc_score(y.ravel(), p.ravel()) > 0.95
    np.testing.assert_array_equal(p, baselines.PixelLogistic(seed=0).fit(x, ym).predict_proba(x))
    with pytest.raises(ValueError):
        baselines.PixelLogistic().fit(x, np.ones_like(y))
    with pytest.raises(RuntimeError):
        baselines.PixelLogistic().predict_proba(x)


# ---------------------------------------------------------------- FIRE-08
@pytest.mark.req("FIRE-08")
def test_pr_auc_ignores_uncertain():
    rng = np.random.default_rng(0)
    prob = rng.random((16, 16))
    label = (rng.random((16, 16)) > 0.8).astype(np.int8)
    label[0, :] = -1
    keep = label != -1
    expected = average_precision_score(label[keep], prob[keep])
    assert metrics.pr_auc(prob, label) == pytest.approx(expected)
    perfect = np.where(label == 1, 0.9, 0.1)
    perfect[0, :] = 0.99  # ignored pixels with high scores do not hurt
    assert metrics.pr_auc(perfect, label) == pytest.approx(1.0)
    assert np.isnan(metrics.pr_auc(prob, np.zeros((16, 16), dtype=np.int8)))
    with pytest.raises(ValueError):
        metrics.pr_auc(prob, label[:-1])


@pytest.mark.req("FIRE-08")
def test_new_burn_pr_auc_exposes_persistence():
    prev = np.zeros((10, 10), dtype=np.int8)
    prev[2:5, 2:5] = 1
    prev[9, 9] = -1
    label = np.zeros((10, 10), dtype=np.int8)
    label[2:5, 2:5] = 1  # fire continues
    label[5, 2:6] = 1  # 4 new-burn pixels
    label[9, 9] = 1  # prev unknown: excluded from new-burn
    label[0, 0] = -1  # uncertain: excluded everywhere
    prob = np.clip(prev, 0, 1).astype(np.float64)  # persistence
    out = metrics.fire_pr_auc(prob, label, prev)
    assert set(out) == {"all", "new_burn"}
    assert out["all"] == pytest.approx(metrics.pr_auc(prob, label))
    nb = (prev == 0) & (label != -1)
    assert out["new_burn"] == pytest.approx(label[nb].mean())  # constant score -> AP = positive rate
    assert out["new_burn"] == pytest.approx(4 / nb.sum())


@pytest.mark.req("FIRE-08")
def test_per_fire_macro():
    assert metrics.per_fire_macro([0.2, np.nan, 0.4]) == pytest.approx(0.3)
    assert np.isnan(metrics.per_fire_macro([np.nan]))
    assert np.isnan(metrics.per_fire_macro([]))


# ---------------------------------------------------------------- FIRE-09
@pytest.mark.req("FIRE-09")
def test_hourly_growth_steps(t0):
    m0 = np.zeros((4, 4), dtype=np.float32)
    p1 = np.ones((4, 4), dtype=np.float32)
    g = growth.hourly_growth(m0, p1, day0_time=t0, issued=t0)
    assert g.shape == (12, 4, 4) and g.dtype == np.float32
    expected = [0.0] * 4 + [1 / 24] * 4 + [2 / 24] * 4
    np.testing.assert_allclose(g[:, 0, 0], expected, rtol=1e-6)


@pytest.mark.req("FIRE-09")
def test_hourly_growth_hold_after_day_and_mask_cleaning(t0):
    m0 = np.zeros((2, 2), dtype=np.float32)
    m0[0, 0] = -1.0
    m0[0, 1] = np.nan
    m0[1, 1] = 1.0
    p1 = np.full((2, 2), 0.6, dtype=np.float32)
    p1[1, 1] = 0.0  # shrinking fire: plain interpolation decreases
    g = growth.hourly_growth(m0, p1, day0_time=t0, issued=t0 + timedelta(hours=23, minutes=30))
    a = np.array([23 / 24, 23 / 24, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    np.testing.assert_allclose(g[:, 0, 0], a * 0.6, rtol=1e-5)
    np.testing.assert_allclose(g[:, 0, 1], a * 0.6, rtol=1e-5)
    np.testing.assert_allclose(g[:, 1, 1], 1 - a, rtol=1e-5, atol=1e-7)
    mid = growth.hourly_growth(
        np.ones((2, 2)), np.zeros((2, 2)), day0_time=t0, issued=t0 + timedelta(hours=12)
    )
    assert mid[0, 0, 0] == pytest.approx(0.5)


@pytest.mark.req("FIRE-09")
def test_hourly_growth_errors(t0):
    z = np.zeros((2, 2))
    with pytest.raises(ValueError):
        growth.hourly_growth(z, z, day0_time=t0, issued=t0 - timedelta(hours=1))
    with pytest.raises(ValueError):
        growth.hourly_growth(z, z, day0_time=datetime(2019, 5, 26), issued=datetime(2019, 5, 26, 1))
    with pytest.raises(ValueError):
        growth.hourly_growth(z, np.zeros((3, 2)), day0_time=t0, issued=t0)


# ---------------------------------------------------------------- FIRE-11
@pytest.mark.req("FIRE-11")
def test_observed_from_mask():
    m = np.array([[1, 0], [-1, np.nan]], dtype=np.float32)
    obs = growth.observed_from_mask(m)
    assert obs.shape == (12, 2, 2) and obs.dtype == np.float32
    for s in range(12):
        np.testing.assert_array_equal(obs[s], [[1, 0], [0, 0]])
    assert growth.observed_from_mask(m, n_slots=3).shape == (3, 2, 2)


# ---------------------------------------------------------------- FIRE-12
@pytest.mark.req("FIRE-12")
def test_ndws_dataset(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.random((3, 12, 16, 16)).astype(np.float32)
    x[:, TH] = 90.0
    x[:, PREV] = rng.integers(0, 2, size=(3, 16, 16))
    y = rng.integers(-1, 2, size=(3, 16, 16)).astype(np.int8)
    np.save(tmp_path / "ndws_train_x.npy", x)
    np.save(tmp_path / "ndws_train_y.npy", y)
    st = dataset.NormStats(
        lo=np.zeros(12, np.float32),
        hi=np.full(12, 400.0, np.float32),
        mean=np.full(12, 0.5, np.float32),
        std=np.full(12, 2.0, np.float32),
    )
    ds = dataset.NdwsDataset(tmp_path / "ndws_train_x.npy", tmp_path / "ndws_train_y.npy", st)
    assert len(ds) == 3
    xi, yi = ds[2]
    assert xi.dtype == torch.float32 and xi.shape == (12, 16, 16)
    assert yi.dtype == torch.float32 and yi.shape == (1, 16, 16)
    np.testing.assert_allclose(xi.numpy(), dataset.normalize(x[2], st), rtol=1e-6)
    np.testing.assert_array_equal(yi.numpy()[0], y[2].astype(np.float32))
    np.save(tmp_path / "bad_y.npy", y[:2])
    with pytest.raises(ValueError):
        dataset.NdwsDataset(tmp_path / "ndws_train_x.npy", tmp_path / "bad_y.npy", st)


@pytest.mark.req("FIRE-12")
def test_ndws_dataset_augments_before_normalising(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.random((2, 12, 16, 16)).astype(np.float32)
    x[:, TH] = 30.0
    y = rng.integers(-1, 2, size=(2, 16, 16)).astype(np.int8)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "y.npy", y)
    st = dataset.NormStats(
        lo=np.zeros(12, np.float32),
        hi=np.full(12, 400.0, np.float32),
        mean=np.zeros(12, np.float32),
        std=np.ones(12, np.float32),
    )
    ds = dataset.NdwsDataset(tmp_path / "x.npy", tmp_path / "y.npy", st, augment=True, seed=0)
    xi, yi = ds[0]
    options = []
    for hf, vf, k in DIHEDRAL:
        xa, ya = dataset.augment(x[0], y[0], hf, vf, k)
        options.append((dataset.normalize(xa, st), ya.astype(np.float32)))
    assert any(np.allclose(xi.numpy(), xo) and np.array_equal(yi.numpy()[0], yo) for xo, yo in options)


@pytest.mark.req("FIRE-12")
@pytest.mark.req("SPLIT-07")
def test_ndws_dataset_from_split(tmp_path):
    from datetime import date

    from outrun.eval.splits import Split, SplitItem

    split = Split(
        "toy",
        train=tuple(SplitItem(i) for i in ("ndws:a", "ndws:c")),
        val=tuple(SplitItem(i) for i in ("ndws:b",)),
        test=tuple(SplitItem(i) for i in ("ndws:d",)),
        extra={"case_holdout": (SplitItem("wsts:q", date(2024, 5, 6)),)},
    )
    rng = np.random.default_rng(2)
    x = rng.random((4, 12, 8, 8)).astype(np.float32)
    y = rng.integers(-1, 2, size=(4, 8, 8)).astype(np.int8)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "y.npy", y)
    st = dataset.NormStats(
        lo=np.zeros(12, np.float32),
        hi=np.full(12, 400.0, np.float32),
        mean=np.zeros(12, np.float32),
        std=np.ones(12, np.float32),
    )
    ids = ["ndws:a", "ndws:b", "ndws:c", "ndws:d"]
    tr = dataset.NdwsDataset.from_split(tmp_path / "x.npy", tmp_path / "y.npy", ids, split, "train", st)
    va = dataset.NdwsDataset.from_split(tmp_path / "x.npy", tmp_path / "y.npy", ids, split, "val", st)
    assert tuple(tr.ids) == ("ndws:a", "ndws:c") and len(tr) == 2
    assert tuple(va.ids) == ("ndws:b",)
    assert "ndws:d" not in tr.ids and "ndws:d" not in va.ids
    np.testing.assert_allclose(tr[1][0].numpy(), dataset.normalize(x[2], st), rtol=1e-6)
    with pytest.raises(ValueError):
        dataset.NdwsDataset.from_split(tmp_path / "x.npy", tmp_path / "y.npy", ids[:2], split, "train", st)
