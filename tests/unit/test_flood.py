"""specs/10-flood.md: features, labels, sampling, model, loss, baselines, crossings, metrics, dataset.

Synthetic arrays only (offline, CPU, seconds). Adapter/RiskLayer tests are in test_flood_adapter.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from sklearn.metrics import roc_auc_score

from outrun.hazards.flood import baselines, crossings, dataset, metrics, model
from outrun.hazards.flood.crossings import CrossingKind
from outrun.hazards.flood.dataset import FloodReport, SampleKind

ROOT = Path(__file__).resolve().parents[2]
H = W = 16


def _inputs(h: int = H, w: int = W) -> list[np.ndarray]:
    """Ten (h, w) inputs; input i is the constant i + 1."""
    return [np.full((h, w), float(i + 1), dtype=np.float32) for i in range(10)]


def _hours(t: datetime, h: float) -> datetime:
    return t + timedelta(hours=h)


# ---------------------------------------------------------------- FLOOD-01
@pytest.mark.req("FLOOD-01")
def test_feature_channel_order_and_values():
    assert dataset.FLOOD_CHANNELS == (
        "qpe_1h",
        "qpe_3h",
        "qpe_6h",
        "qpe_24h",
        "ffg_ratio_1h",
        "ffg_ratio_3h",
        "ari_max",
        "elevation",
        "slope",
        "tpi",
        "valid",
    )
    out = dataset.build_features(*_inputs())
    assert out.shape == (11, H, W) and out.dtype == np.float32
    for i in range(10):
        assert np.all(out[i] == i + 1)
    assert np.all(out[10] == 1.0)


@pytest.mark.req("FLOOD-01")
def test_feature_valid_mask_and_nonfinite_zeroed():
    ins = _inputs()
    ins[4][0, 0] = np.nan  # ffg_ratio_1h
    ins[7][3, 5] = np.inf  # elevation
    before = [a.copy() for a in ins]
    out = dataset.build_features(*ins)
    assert np.isfinite(out).all()
    assert out[4, 0, 0] == 0.0 and out[7, 3, 5] == 0.0
    assert out[10, 0, 0] == 0.0 and out[10, 3, 5] == 0.0
    assert out[10].sum() == H * W - 2
    assert out[0, 0, 0] == 1.0  # other channels at the invalid pixel keep their value
    for a, b in zip(ins, before, strict=True):
        np.testing.assert_array_equal(a, b)  # inputs not modified


@pytest.mark.req("FLOOD-01")
@pytest.mark.parametrize("which", [0, 1, 2, 3])
def test_feature_missing_hour_window_raises(which):
    ins = _inputs()
    ins[which][:] = np.nan
    with pytest.raises(dataset.MissingHoursError):
        dataset.build_features(*ins)
    assert issubclass(dataset.MissingHoursError, ValueError)


@pytest.mark.req("FLOOD-01")
def test_feature_shape_errors():
    ins = _inputs()
    ins[8] = np.zeros((H, W + 1), dtype=np.float32)
    with pytest.raises(ValueError):
        dataset.build_features(*ins)
    ins = _inputs()
    ins[0] = np.zeros((2, H, W), dtype=np.float32)
    with pytest.raises(ValueError):
        dataset.build_features(*ins)


# ---------------------------------------------------------------- FLOOD-02
@pytest.mark.req("FLOOD-02")
def test_distance_to_reports(t0):
    x = np.array([3000.0, 0.0, 20000.0])
    y = np.array([4000.0, 0.0, 0.0])
    r = FloodReport(t0, t0, 0.0, 0.0, 10000.0, 0.0)  # segment along x from 0 to 10 km
    d = dataset.distance_to_reports(x, y, [r])
    np.testing.assert_allclose(d, [4000.0, 0.0, 10000.0])
    assert np.all(np.isinf(dataset.distance_to_reports(x, y, [])))


@pytest.mark.req("FLOOD-02")
def test_ring_labels_distance_bands(t0):
    x = np.array([0.0, 3000.0, 5000.0, 10000.0, 15000.0, 20000.0])
    y = np.zeros_like(x)
    r = FloodReport(_hours(t0, 1), _hours(t0, 2), 0.0, 0.0, 0.0, 0.0)
    lab = dataset.ring_labels(x, y, [r], t0)
    assert lab.dtype == np.int8 and lab.shape == x.shape
    np.testing.assert_array_equal(lab, [1, 1, 1, -1, -1, 0])


@pytest.mark.req("FLOOD-02")
def test_ring_labels_segment_and_2d_grid(t0):
    xx, yy = np.meshgrid(np.array([5000.0, 20000.0]), np.array([4000.0, 16000.0]))
    r = FloodReport(t0, _hours(t0, 1), 0.0, 0.0, 10000.0, 0.0)
    lab = dataset.ring_labels(xx, yy, [r], t0)
    # (5 km, 4 km): 4 km from segment -> 1; (20 km, 4 km): ~10.8 km -> -1
    # (5 km, 16 km): 16 km -> 0; (20 km, 16 km): ~18.9 km -> 0
    np.testing.assert_array_equal(lab, [[1, -1], [0, 0]])


@pytest.mark.req("FLOOD-02")
def test_ring_labels_time_activity(t0):
    x = np.array([0.0, 10000.0])
    y = np.zeros(2)
    point = dict(x0=0.0, y0=0.0, x1=0.0, y1=0.0)
    # begins 6.5 h after issue: [begin - 1 h, end] starts at 5.5 h <= 6 h -> active
    late_ok = FloodReport(_hours(t0, 6.5), _hours(t0, 7), **point)
    np.testing.assert_array_equal(dataset.ring_labels(x, y, [late_ok], t0), [1, -1])
    # begins 7.5 h after issue: starts at 6.5 h > 6 h -> inactive, no positives and no ring
    too_late = FloodReport(_hours(t0, 7.5), _hours(t0, 8), **point)
    np.testing.assert_array_equal(dataset.ring_labels(x, y, [too_late], t0), [0, 0])
    # ended before issue -> inactive; ending exactly at issue -> active (closed intervals)
    ended = FloodReport(_hours(t0, -2), t0 - timedelta(minutes=1), **point)
    np.testing.assert_array_equal(dataset.ring_labels(x, y, [ended], t0), [0, 0])
    at_issue = FloodReport(_hours(t0, -2), t0, **point)
    np.testing.assert_array_equal(dataset.ring_labels(x, y, [at_issue], t0), [1, -1])


@pytest.mark.req("FLOOD-02")
def test_ring_labels_inactive_report_creates_no_ring(t0):
    x = np.array([0.0, 30000.0])
    y = np.zeros(2)
    inactive_near = FloodReport(_hours(t0, 10), _hours(t0, 11), 8000.0, 0.0, 8000.0, 0.0)
    active_far = FloodReport(t0, _hours(t0, 1), 30000.0, 0.0, 30000.0, 0.0)
    np.testing.assert_array_equal(dataset.ring_labels(x, y, [inactive_near, active_far], t0), [0, 1])
    np.testing.assert_array_equal(dataset.ring_labels(x, y, [], t0), [0, 0])


@pytest.mark.req("FLOOD-02")
def test_ring_labels_errors(t0):
    x = np.zeros(3)
    with pytest.raises(ValueError):
        dataset.ring_labels(x, x, [], datetime(2019, 5, 26, 3))  # naive
    with pytest.raises(ValueError):
        dataset.ring_labels(x, np.zeros(4), [], t0)


# ---------------------------------------------------------------- FLOOD-03
@pytest.mark.req("FLOOD-03")
def test_near_report(t0):
    x = np.array([[0.0, 1000.0]])
    y = np.zeros_like(x)

    def rep(dist_m, b_h, e_h):
        return FloodReport(_hours(t0, b_h), _hours(t0, e_h), dist_m, 0.0, dist_m, 0.0)

    assert dataset.near_report(x, y, [rep(20000.0, -1, 1)], t0) is True
    assert dataset.near_report(x, y, [rep(30000.0, -1, 1)], t0) is False  # 29 km from nearest cell
    assert dataset.near_report(x, y, [rep(20000.0, -6, -4)], t0) is False  # ended 4 h before
    assert dataset.near_report(x, y, [rep(20000.0, 2.5, 3)], t0) is True  # begins within 3 h
    assert dataset.near_report(x, y, [rep(20000.0, 3.5, 4)], t0) is False
    assert dataset.near_report(x, y, [], t0) is False


@pytest.mark.req("FLOOD-03")
def test_sample_kind_precedence():
    assert {k.value for k in SampleKind} == {"positive", "hard_negative", "wet", "dry"}
    zeros = np.zeros((8, 8), dtype=np.int8)
    pos = zeros.copy()
    pos[2, 2] = 1
    heavy = np.zeros((8, 8))
    heavy[4, 4] = 30.0
    light = np.full((8, 8), 5.0)
    dry = np.full((8, 8), 0.5)
    assert dataset.sample_kind(pos, dry, report_near=False) is SampleKind.POSITIVE
    assert dataset.sample_kind(pos, heavy, report_near=True) is SampleKind.POSITIVE
    assert dataset.sample_kind(zeros, heavy, report_near=False) is SampleKind.HARD_NEGATIVE
    assert dataset.sample_kind(zeros, heavy, report_near=True) is SampleKind.WET
    assert dataset.sample_kind(zeros, light, report_near=False) is SampleKind.WET
    assert dataset.sample_kind(zeros, dry, report_near=False) is SampleKind.DRY
    ring_only = np.full((8, 8), -1, dtype=np.int8)
    assert dataset.sample_kind(ring_only, heavy, report_near=False) is SampleKind.HARD_NEGATIVE


@pytest.mark.req("FLOOD-03")
def test_sample_kind_nan_handling():
    zeros = np.zeros((4, 4), dtype=np.int8)
    q = np.full((4, 4), np.nan)
    assert dataset.sample_kind(zeros, q, report_near=False) is SampleKind.DRY
    q[1, 1] = 26.0
    assert dataset.sample_kind(zeros, q, report_near=False) is SampleKind.HARD_NEGATIVE


# ---------------------------------------------------------------- FLOOD-13
def _kinds(npos, nhard, nwet, ndry):
    return (
        [SampleKind.POSITIVE] * npos
        + [SampleKind.HARD_NEGATIVE] * nhard
        + [SampleKind.WET] * nwet
        + [SampleKind.DRY] * ndry
    )


def _count(kinds, idx):
    sel = [kinds[i] for i in idx]
    return {k: sel.count(k) for k in SampleKind}


@pytest.mark.req("FLOOD-13")
@pytest.mark.parametrize("n,expected", [(9, (3, 3, 3)), (10, (4, 3, 3)), (11, (4, 4, 3)), (0, (0, 0, 0))])
def test_balanced_indices_quotas(n, expected):
    kinds = _kinds(10, 10, 10, 10)
    idx = dataset.balanced_indices(kinds, n, np.random.default_rng(0))
    assert idx.dtype == np.int64
    c = _count(kinds, idx)
    assert (c[SampleKind.POSITIVE], c[SampleKind.HARD_NEGATIVE], c[SampleKind.WET]) == expected
    assert c[SampleKind.DRY] == 0
    assert np.all(np.diff(idx) > 0)  # sorted, no duplicates


@pytest.mark.req("FLOOD-13")
def test_balanced_indices_shortfall_and_determinism():
    kinds = _kinds(2, 10, 10, 30)
    idx = dataset.balanced_indices(kinds, 9, np.random.default_rng(1))
    c = _count(kinds, idx)
    assert (c[SampleKind.POSITIVE], c[SampleKind.HARD_NEGATIVE], c[SampleKind.WET], c[SampleKind.DRY]) == (
        2,
        3,
        3,
        0,
    )
    a = dataset.balanced_indices(kinds, 9, np.random.default_rng(7))
    b = dataset.balanced_indices(kinds, 9, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        dataset.balanced_indices(kinds, -1, np.random.default_rng(0))


# ---------------------------------------------------------------- FLOOD-04
@pytest.mark.req("FLOOD-04")
def test_unet_shapes_and_parameter_count():
    torch.manual_seed(0)
    net = model.FloodUNet()
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    assert 500_000 <= n_params <= 5_000_000
    assert net(torch.randn(2, 11, 32, 32)).shape == (2, 1, 32, 32)
    assert net(torch.randn(1, 11, 64, 64)).shape == (1, 1, 64, 64)


@pytest.mark.req("FLOOD-04")
def test_unet_wrong_channels_and_ablation_variant():
    torch.manual_seed(0)
    with pytest.raises(ValueError):
        model.FloodUNet()(torch.randn(1, 8, 32, 32))
    assert model.FloodUNet(in_channels=8)(torch.randn(1, 8, 32, 32)).shape == (1, 1, 32, 32)


@pytest.mark.req("FLOOD-04")
def test_unet_gradients_and_eval_determinism():
    torch.manual_seed(0)
    net = model.FloodUNet()
    net(torch.randn(2, 11, 32, 32)).mean().backward()
    for name, p in net.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, name
    net.eval()
    x = torch.randn(1, 11, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(net(x), net(x), rtol=0, atol=0)


# ---------------------------------------------------------------- FLOOD-05
def _focal_ref(logits, target, gamma, alpha):
    p = 1 / (1 + np.exp(-logits))
    pt = np.where(target == 1, p, 1 - p)
    at = np.ones_like(p) if alpha is None else np.where(target == 1, alpha, 1 - alpha)
    return -at * (1 - pt) ** gamma * np.log(pt)


@pytest.mark.req("FLOOD-05")
def test_focal_loss_matches_formula_on_valid_pixels():
    logits = np.array([[[2.0, -1.0], [0.5, 3.0]]])
    target = np.array([[[1.0, 0.0], [-1.0, 0.0]]])
    out = model.focal_loss(torch.tensor(logits), torch.tensor(target), gamma=2.0, alpha=0.25)
    keep = target != -1
    expected = _focal_ref(logits, target, 2.0, 0.25)[keep].mean()
    assert float(out) == pytest.approx(expected, rel=1e-5)


@pytest.mark.req("FLOOD-05")
def test_focal_loss_gamma0_equals_bce():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 1, 8, 8, generator=g)
    target = (torch.rand(2, 1, 8, 8, generator=g) > 0.5).float()
    target[0, 0, :2] = -1
    valid = target != -1
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits[valid], target[valid])
    out = model.focal_loss(logits, target, gamma=0.0, alpha=None)
    assert float(out) == pytest.approx(float(bce), rel=1e-5)


@pytest.mark.req("FLOOD-05")
def test_focal_loss_ignored_pixels_have_no_effect():
    g = torch.Generator().manual_seed(1)
    target = (torch.rand(1, 1, 8, 8, generator=g) > 0.5).float()
    target[0, 0, 0, :] = -1
    logits = torch.randn(1, 1, 8, 8, generator=g, requires_grad=True)
    loss = model.focal_loss(logits, target)
    loss.backward()
    assert torch.all(logits.grad[0, 0, 0, :] == 0)
    changed = logits.detach().clone()
    changed[0, 0, 0, :] = 50.0
    assert float(model.focal_loss(changed, target)) == pytest.approx(float(loss), rel=1e-6)


@pytest.mark.req("FLOOD-05")
def test_focal_loss_all_ignored_and_shape_mismatch():
    logits = torch.randn(1, 1, 4, 4, requires_grad=True)
    out = model.focal_loss(logits, torch.full((1, 1, 4, 4), -1.0))
    assert torch.isfinite(out) and float(out) == 0.0
    out.backward()
    with pytest.raises(ValueError):
        model.focal_loss(torch.randn(1, 1, 4, 4), torch.zeros(1, 1, 4, 5))


# ---------------------------------------------------------------- FLOOD-06
@pytest.mark.req("FLOOD-06")
def test_ffg_ratio_baseline():
    r1 = np.array([[0.5, 1.0], [0.2, np.nan]])
    r3 = np.array([[0.9, 0.1], [1.3, np.nan]])
    out = baselines.ffg_ratio_baseline(r1, r3)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, [[0, 1], [1, 0]])


@pytest.mark.req("FLOOD-06")
def test_ari_baselines():
    np.testing.assert_array_equal(baselines.ari_baseline(np.array([2.0, 10.0, 25.0, np.nan])), [0, 1, 1, 0])
    q3 = np.array([10.0, 50.0, 10.0, np.nan])
    q6 = np.array([20.0, 20.0, 75.0, np.nan])
    out = baselines.ari_rule_fallback(q3, q6)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, [0, 1, 1, 0])


@pytest.mark.req("FLOOD-06")
def test_rule_baseline_from_features():
    f = np.zeros((11, 3, 3), dtype=np.float32)
    f[4, 0, 0] = 1.2  # ffg_ratio_1h
    f[5, 1, 1] = 1.0  # ffg_ratio_3h
    f[6, 2, 2] = 50.0  # ari_max: not used by this rule
    expected = np.zeros((3, 3))
    expected[0, 0] = expected[1, 1] = 1
    np.testing.assert_array_equal(baselines.rule_baseline_from_features(f), expected)
    batch = np.stack([f, np.zeros_like(f)])
    out = baselines.rule_baseline_from_features(batch)
    assert out.shape == (2, 3, 3)
    np.testing.assert_array_equal(out[0], expected)
    assert out[1].sum() == 0


# ---------------------------------------------------------------- FLOOD-07
def _separable(seed=0, n=4, h=8, w=8):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 11, h, w)).astype(np.float32)
    X[:, 10] = 1.0  # valid
    y = (X[:, 0] > 0).astype(np.int8)
    return X, y


@pytest.mark.req("FLOOD-07")
def test_logistic_baseline_ranks_separable_data():
    X, y = _separable()
    y_masked = y.copy()
    y_masked[:, 0, :] = -1
    lr = baselines.LogisticBaseline(seed=0).fit(X, y_masked)
    p = lr.predict_proba(X)
    assert p.shape == (4, 8, 8) and p.dtype == np.float32
    assert p.min() >= 0 and p.max() <= 1
    assert roc_auc_score(y.ravel(), p.ravel()) > 0.95


@pytest.mark.req("FLOOD-07")
def test_logistic_baseline_valid_mask_determinism_errors():
    X, y = _separable()
    Xi = X.copy()
    Xi[0, 10, 0, 0] = 0.0
    a = baselines.LogisticBaseline(seed=0).fit(Xi, y).predict_proba(Xi)
    b = baselines.LogisticBaseline(seed=0).fit(Xi, y).predict_proba(Xi)
    np.testing.assert_array_equal(a, b)
    assert a[0, 0, 0] == 0.0
    with pytest.raises(ValueError):
        baselines.LogisticBaseline().fit(X, np.zeros_like(y))
    with pytest.raises(RuntimeError):
        baselines.LogisticBaseline().predict_proba(X)


# ---------------------------------------------------------------- FLOOD-08
@pytest.mark.req("FLOOD-08")
@pytest.mark.parametrize("depth,expected", [(-3.0, True), (-2.0, False), (-1.0, False)])
def test_in_depression_threshold(depth, expected):
    dem = np.zeros((61, 61))
    dem[30, 30] = depth
    assert crossings.in_depression(dem, 10.0, 30, 30) is expected


@pytest.mark.req("FLOOD-08")
def test_in_depression_uses_radius_border_and_nan():
    dem = np.full((201, 201), -100.0)
    yy, xx = np.mgrid[0:201, 0:201]
    disk = (yy - 100) ** 2 + (xx - 100) ** 2 <= 30**2  # 300 m at 10 m cells
    dem[disk] = 0.0
    dem[100, 100] = -3.0
    assert crossings.in_depression(dem, 10.0, 100, 100) is True  # median of the disk, not of the whole DEM
    bowl = np.full((61, 61), 5.0)
    bowl[30, 30] = 2.9
    assert crossings.in_depression(bowl, 10.0, 30, 30) is True
    corner = np.zeros((61, 61))
    corner[0, 0] = -3.0
    assert crossings.in_depression(corner, 10.0, 0, 0) is True
    nan_half = np.zeros((61, 61))
    nan_half[:, :30] = np.nan
    nan_half[30, 30] = -3.0
    assert crossings.in_depression(nan_half, 10.0, 30, 30) is True
    with pytest.raises(ValueError):
        crossings.in_depression(corner, 10.0, 61, 0)


@pytest.mark.req("FLOOD-08")
def test_crossing_overrides_multipliers():
    assert {k.value for k in CrossingKind} == {"low_water", "depression", "bridge"}
    p = np.array([0.1, 0.5, 0.8])
    edge_prob = {1: p.copy(), 2: p.copy(), 3: p.copy(), 4: p.copy()}
    kinds = {1: CrossingKind.LOW_WATER, 2: CrossingKind.DEPRESSION, 3: CrossingKind.BRIDGE}
    out = crossings.crossing_overrides(edge_prob, kinds)
    assert set(out) == {1, 2, 3}
    for v in out.values():
        assert v.dtype == np.float32 and v.shape == (3,)
    np.testing.assert_allclose(out[1], [0.15, 0.75, 1.0], rtol=1e-6)
    np.testing.assert_allclose(out[2], [0.2, 1.0, 1.0], rtol=1e-6)
    np.testing.assert_array_equal(out[3], [0.0, 0.0, 0.0])
    for v in edge_prob.values():
        np.testing.assert_array_equal(v, p)  # not modified
    with pytest.raises(KeyError):
        crossings.crossing_overrides({1: p}, {1: CrossingKind.LOW_WATER, 9: CrossingKind.BRIDGE})


@pytest.mark.req("FLOOD-08")
def test_crossing_defaults_match_config():
    cfg = yaml.safe_load((ROOT / "configs" / "hazards.yaml").read_text())["hazards"]["flood"]
    assert cfg["crossing_multiplier"] == 1.5 and cfg["depression_multiplier"] == 2.0
    out = crossings.crossing_overrides({0: np.array([0.2])}, {0: CrossingKind.LOW_WATER})
    assert float(out[0][0]) == pytest.approx(0.2 * cfg["crossing_multiplier"], rel=1e-6)


# ---------------------------------------------------------------- FLOOD-09
@pytest.mark.req("FLOOD-09")
def test_select_channels_no_flash():
    assert dataset.FLASH_CHANNELS == ("ffg_ratio_1h", "ffg_ratio_3h", "ari_max")
    names = dataset.channel_names(use_flash=False)
    assert names == tuple(c for c in dataset.FLOOD_CHANNELS if c not in dataset.FLASH_CHANNELS)
    assert dataset.channel_names(use_flash=True) == dataset.FLOOD_CHANNELS
    x = np.arange(11, dtype=np.float32)[:, None, None] * np.ones((11, 4, 4), dtype=np.float32)
    np.testing.assert_array_equal(dataset.select_channels(x, use_flash=True), x)
    sub = dataset.select_channels(x, use_flash=False)
    assert sub.shape == (8, 4, 4)
    np.testing.assert_array_equal(sub[:, 0, 0], [0, 1, 2, 3, 7, 8, 9, 10])
    batch = np.stack([x, x])
    assert dataset.select_channels(batch, use_flash=False).shape == (2, 8, 4, 4)
    with pytest.raises(ValueError):
        dataset.select_channels(np.zeros((10, 4, 4)), use_flash=False)


# ---------------------------------------------------------------- FLOOD-11
@pytest.mark.req("FLOOD-11")
def test_observed_from_reports_slots_and_radius(t0):
    x = 500.0 + 1000.0 * np.arange(32)
    y = 500.0 + 1000.0 * np.arange(32)
    r = FloodReport(
        t0 + timedelta(minutes=30), t0 + timedelta(minutes=60), 10500.0, 10500.0, 10500.0, 10500.0
    )
    obs = dataset.observed_from_reports([r], x, y, t0)
    assert obs.shape == (12, 32, 32) and obs.dtype == np.float32
    active = [s for s in range(12) if obs[s].max() > 0]
    assert active == [2, 3, 4]
    assert obs[2, 10, 10] == 1.0  # report location (row = y index, col = x index)
    assert obs[2, 10, 14] == 1.0  # 4 km away
    assert obs[2, 10, 16] == 0.0  # 6 km away
    assert set(np.unique(obs)) <= {0.0, 1.0}
    assert dataset.observed_from_reports([], x, y, t0).sum() == 0


# ---------------------------------------------------------------- FLOOD-12
@pytest.mark.req("FLOOD-12")
def test_event_pod():
    prob = np.zeros((4, 4))
    prob[0, 0] = 0.9
    prob[3, 3] = 0.5
    m_hit = np.zeros((4, 4), bool)
    m_hit[0:2, 0:2] = True
    m_miss = np.zeros((4, 4), bool)
    m_miss[2, 0] = True
    m_edge = np.zeros((4, 4), bool)
    m_edge[3, 3] = True
    assert metrics.event_pod(prob, [m_hit, m_miss]) == pytest.approx(0.5)
    assert metrics.event_pod(prob, [m_edge]) == pytest.approx(1.0)  # >= threshold counts
    assert metrics.event_pod(prob, [np.zeros((4, 4), bool)]) == pytest.approx(0.0)
    assert np.isnan(metrics.event_pod(prob, []))
    with pytest.raises(ValueError):
        metrics.event_pod(prob, [np.zeros((3, 3), bool)])


# ---------------------------------------------------------------- FLOOD-14
@pytest.mark.req("FLOOD-14")
def test_flood_patch_dataset(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.random((3, 11, 16, 16)).astype(np.float16)
    y = rng.integers(-1, 2, size=(3, 16, 16)).astype(np.int8)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "y.npy", y)
    ds = dataset.FloodPatchDataset(tmp_path / "x.npy", tmp_path / "y.npy")
    assert len(ds) == 3
    xi, yi = ds[1]
    assert xi.dtype == torch.float32 and xi.shape == (11, 16, 16)
    assert yi.dtype == torch.float32 and yi.shape == (1, 16, 16)
    np.testing.assert_allclose(xi.numpy(), x[1].astype(np.float32))
    np.testing.assert_array_equal(yi.numpy()[0], y[1].astype(np.float32))
    assert (yi == -1).any()
    ds8 = dataset.FloodPatchDataset(tmp_path / "x.npy", tmp_path / "y.npy", use_flash=False)
    x8, _ = ds8[1]
    assert x8.shape == (8, 16, 16)
    np.testing.assert_allclose(x8.numpy(), x[1, [0, 1, 2, 3, 7, 8, 9, 10]].astype(np.float32))
    np.save(tmp_path / "y2.npy", y[:2])
    with pytest.raises(ValueError):
        dataset.FloodPatchDataset(tmp_path / "x.npy", tmp_path / "y2.npy")


@pytest.mark.req("FLOOD-14")
@pytest.mark.req("SPLIT-07")
def test_flood_patch_dataset_from_split(tmp_path):
    from datetime import date

    from outrun.eval.splits import Split, SplitItem

    split = Split(
        "toy",
        train=tuple(SplitItem(i) for i in ("s0", "s3")),
        val=tuple(SplitItem(i) for i in ("s1",)),
        test=tuple(SplitItem(i) for i in ("s2",)),
        extra={"case_holdout": (SplitItem("s4", date(2024, 5, 6)),)},
    )
    rng = np.random.default_rng(0)
    x = rng.random((5, 11, 8, 8)).astype(np.float16)
    y = rng.integers(-1, 2, size=(5, 8, 8)).astype(np.int8)
    np.save(tmp_path / "x.npy", x)
    np.save(tmp_path / "y.npy", y)
    ids = ["s0", "s1", "s2", "s3", "s4"]
    tr = dataset.FloodPatchDataset.from_split(tmp_path / "x.npy", tmp_path / "y.npy", ids, split, "train")
    va = dataset.FloodPatchDataset.from_split(tmp_path / "x.npy", tmp_path / "y.npy", ids, split, "val")
    assert tuple(tr.ids) == ("s0", "s3") and len(tr) == 2
    assert tuple(va.ids) == ("s1",) and len(va) == 1
    for ds in (tr, va):
        assert "s2" not in ds.ids and "s4" not in ds.ids  # test and quarantined never reach train/val
    np.testing.assert_allclose(tr[1][0].numpy(), x[3].astype(np.float32))
    with pytest.raises(ValueError):
        dataset.FloodPatchDataset.from_split(tmp_path / "x.npy", tmp_path / "y.npy", ids[:3], split, "train")
