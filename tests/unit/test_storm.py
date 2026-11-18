"""Storm nowcast tests (specs/08-storm.md). Synthetic tensors only; outrun is imported inside tests."""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
H = W = 32


def make_events(n: int = 2, h: int = H, w: int = W, seed: int = 0) -> np.ndarray:
    """Fake SEVIR event array (n, 49, 2, h, w) uint8."""
    return np.random.default_rng(seed).integers(0, 256, size=(n, 49, 2, h, w), dtype=np.uint8)


def make_scenario(t0):
    """The adapter reads only ``t0`` from a scenario (STORM-08), so a namespace stands in for Scenario."""
    return SimpleNamespace(id="synthetic", t0=t0)


def storm_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "hazards.yaml").read_text())["hazards"]["storm"]


def blob(h: int, w: int, cy: float, cx: float, sigma: float = 3.0, peak: float = 200.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return (peak * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))).astype(np.float32)


# ---------------------------------------------------------------- STORM-01 windows
@pytest.mark.req("STORM-01")
def test_window_index_event_major():
    from outrun.hazards.storm.dataset import window_index

    idx = window_index(2)
    assert idx.shape == (6, 2)
    assert idx.tolist() == [[0, 0], [0, 12], [0, 24], [1, 0], [1, 12], [1, 24]]
    assert window_index(0).shape == (0, 2)


@pytest.mark.req("STORM-01")
def test_dataset_items_frames_and_scaling():
    from outrun.hazards.storm.dataset import SevirWindowDataset

    ev = make_events(3)
    ds = SevirWindowDataset(ev, event_ids=[2, 0], use_ir=True)
    assert len(ds) == 6
    x, y = ds[4]  # event_ids[1] = 0, offset 12
    assert x.dtype == torch.float32 and y.dtype == torch.float32
    assert x.shape == (26, H, W) and y.shape == (12, H, W)
    np.testing.assert_allclose(x[:13].numpy(), ev[0, 12:25, 0] / 255.0, atol=1e-6)
    np.testing.assert_allclose(y.numpy(), ev[0, 25:37, 0].astype(np.float32))
    x0, y0 = ds[2]  # event 2, offset 24
    np.testing.assert_allclose(y0.numpy(), ev[2, 37:49, 0].astype(np.float32))
    assert float(x0.min()) >= 0.0 and float(x0.max()) <= 1.0


@pytest.mark.req("STORM-01")
def test_dataset_index_and_shape_errors():
    from outrun.hazards.storm.dataset import SevirWindowDataset

    ds = SevirWindowDataset(make_events(1))
    assert len(ds) == 3
    with pytest.raises(IndexError):
        ds[3]
    with pytest.raises(IndexError):
        ds[-1]
    with pytest.raises(ValueError):
        SevirWindowDataset(np.zeros((1, 49, H, W), np.uint8))  # ndim 4
    with pytest.raises(ValueError):
        SevirWindowDataset(np.zeros((1, 40, 2, H, W), np.uint8))  # too few frames for offset 24
    with pytest.raises(ValueError):
        SevirWindowDataset(np.zeros((1, 49, 1, H, W), np.uint8), use_ir=True)  # no IR channel


# ---------------------------------------------------------------- STORM-02 satellite input
@pytest.mark.req("STORM-02")
def test_ir_channels_and_vil_only_ablation():
    from outrun.hazards.storm.dataset import SevirWindowDataset, in_channels

    ev = make_events(2, seed=1)
    with_ir = SevirWindowDataset(ev, use_ir=True)
    vil_only = SevirWindowDataset(ev, use_ir=False)
    assert len(with_ir) == len(vil_only) == 6
    assert in_channels(True) == 26 and in_channels(False) == 13
    for i in range(len(with_ir)):
        xa, ya = with_ir[i]
        xb, yb = vil_only[i]
        assert xb.shape == (13, H, W)
        torch.testing.assert_close(xb, xa[:13])
        torch.testing.assert_close(yb, ya)
    xa, _ = with_ir[1]  # event 0, offset 12
    np.testing.assert_allclose(xa[13:].numpy(), ev[0, 12:25, 1] / 255.0, atol=1e-6)


@pytest.mark.req("STORM-02")
def test_events_with_both_vil_and_ir():
    from outrun.hazards.storm.dataset import events_with_both

    cat = pd.DataFrame(
        {
            "id": ["S3", "S1", "S1", "S2", "S3", "S3", "S4"],
            "img_type": ["ir107", "vil", "ir107", "vil", "vil", "vis", "ir069"],
        }
    )
    assert events_with_both(cat) == ["S1", "S3"]
    with pytest.raises(KeyError):
        events_with_both(cat.drop(columns=["img_type"]))


@pytest.mark.req("STORM-02")
def test_model_accepts_vil_only_ablation_input():
    from outrun.hazards.storm.model import StormUNet

    torch.manual_seed(0)
    m = StormUNet(in_channels=13, base=4, levels=2).eval()
    vil, logits = m(torch.rand(1, 13, 16, 16))
    assert vil.shape == (1, 12, 16, 16) and logits.shape == (1, 12, 3, 16, 16)


# ---------------------------------------------------------------- STORM-03 targets and weights
@pytest.mark.req("STORM-03")
def test_exceedance_targets_inclusive_thresholds():
    from outrun.hazards.storm.model import (
        RISK_THRESHOLD,
        RISK_THRESHOLD_INDEX,
        VIL_THRESHOLDS,
        exceedance_targets,
    )

    assert VIL_THRESHOLDS == (74, 133, 181)
    assert VIL_THRESHOLDS[RISK_THRESHOLD_INDEX] == RISK_THRESHOLD == 133
    y = torch.zeros(2, 12, 1, 4)
    y[..., 0, :] = torch.tensor([73.0, 74.0, 133.0, 181.0])
    t = exceedance_targets(y)
    assert t.shape == (2, 12, 3, 1, 4) and t.dtype == torch.float32
    expected = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.float32)
    torch.testing.assert_close(t[0, 0, :, 0, :], expected)
    torch.testing.assert_close(t[1, 11, :, 0, :], expected)


@pytest.mark.req("STORM-03")
def test_intensity_weights_breaks():
    from outrun.hazards.storm.model import intensity_weights

    y = torch.tensor([0.0, 15.0, 16.0, 73.0, 74.0, 132.0, 133.0, 159.0, 160.0, 180.0, 181.0, 255.0])
    w = intensity_weights(y)
    assert w.shape == y.shape
    torch.testing.assert_close(w, torch.tensor([1.0, 1, 2, 2, 5, 5, 10, 10, 20, 20, 30, 30]))


# ---------------------------------------------------------------- STORM-04 model
@pytest.mark.req("STORM-04")
def test_unet_shapes_and_attributes():
    from outrun.hazards.storm.model import StormUNet

    torch.manual_seed(0)
    m = StormUNet(in_channels=26, n_out=12, n_thresholds=3, base=4, levels=2).eval()
    assert (m.in_channels, m.n_out, m.n_thresholds, m.levels) == (26, 12, 3, 2)
    vil, logits = m(torch.rand(2, 26, 32, 16))
    assert vil.shape == (2, 12, 32, 16)
    assert logits.shape == (2, 12, 3, 32, 16)


@pytest.mark.req("STORM-04")
def test_unet_rejects_bad_input():
    from outrun.hazards.storm.model import StormUNet

    m = StormUNet(base=4, levels=2).eval()
    with pytest.raises(ValueError):
        m(torch.rand(1, 26, 18, 16))  # 18 not divisible by 4
    with pytest.raises(ValueError):
        m(torch.rand(1, 13, 16, 16))  # wrong channel count


@pytest.mark.req("STORM-04")
def test_unet_default_parameter_count():
    from outrun.hazards.storm.model import StormUNet

    n = sum(p.numel() for p in StormUNet().parameters() if p.requires_grad)
    assert 4_000_000 <= n <= 12_000_000


@pytest.mark.req("STORM-04")
def test_unet_predict_proba_range_and_determinism():
    from outrun.hazards.storm.model import StormUNet, predict_proba

    x = torch.rand(2, 26, 16, 16, generator=torch.Generator().manual_seed(1))
    torch.manual_seed(7)
    a = StormUNet(base=4, levels=2)
    torch.manual_seed(7)
    b = StormUNet(base=4, levels=2)
    for pa, pb in zip(a.state_dict().values(), b.state_dict().values(), strict=True):
        torch.testing.assert_close(pa, pb)
    p1 = predict_proba(a, x)
    p2 = predict_proba(a, x)
    assert p1.shape == (2, 12, 3, 16, 16)
    assert float(p1.min()) >= 0.0 and float(p1.max()) <= 1.0
    torch.testing.assert_close(p1, p2)
    torch.testing.assert_close(p1, predict_proba(b, x))


# ---------------------------------------------------------------- STORM-05 loss and training step
@pytest.mark.req("STORM-05")
def test_storm_loss_matches_formula():
    from outrun.hazards.storm.model import exceedance_targets, intensity_weights, storm_loss

    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 256, (2, 12, 4, 4), generator=g).float()
    vil = torch.rand(2, 12, 4, 4, generator=g)
    logits = torch.randn(2, 12, 3, 4, 4, generator=g)
    t, w = y / 255.0, intensity_weights(y)
    expected = (
        (w * (vil - t) ** 2).mean()
        + (w * (vil - t).abs()).mean()
        + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(logits, exceedance_targets(y))
    )
    got = storm_loss(vil, logits, y, bce_weight=0.5)
    assert got.ndim == 0 and torch.isfinite(got) and float(got) >= 0
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.req("STORM-05")
def test_storm_loss_near_zero_for_perfect_prediction():
    from outrun.hazards.storm.model import exceedance_targets, storm_loss

    y = torch.randint(0, 256, (1, 12, 4, 4), generator=torch.Generator().manual_seed(3)).float()
    logits = exceedance_targets(y) * 40.0 - 20.0
    assert float(storm_loss(y / 255.0, logits, y)) < 1e-3


@pytest.mark.req("STORM-05")
@pytest.mark.slow
def test_train_step_overfits_one_batch():
    from outrun.hazards.storm.model import StormUNet, train_step

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, 26, 16, 16, generator=g)
    y = torch.randint(0, 256, (2, 12, 16, 16), generator=g).float()
    m = StormUNet(base=4, levels=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    losses = [train_step(m, (x, y), opt) for _ in range(40)]
    assert all(isinstance(v, float) and math.isfinite(v) for v in losses)
    assert losses[-1] < 0.7 * losses[0]


# ---------------------------------------------------------------- STORM-06 baselines
@pytest.mark.req("STORM-06")
def test_persistence_and_mask():
    from outrun.hazards.storm.baselines import persistence, persistence_mask

    frames = np.random.default_rng(0).integers(0, 256, (13, 8, 8)).astype(np.uint8)
    out = persistence(frames, n_out=12)
    assert out.shape == (12, 8, 8) and out.dtype == np.float32
    for k in range(12):
        np.testing.assert_array_equal(out[k], frames[-1].astype(np.float32))
    mask = persistence_mask(frames, threshold=133, n_out=12)
    assert mask.shape == (12, 8, 8) and mask.dtype == np.float32
    np.testing.assert_array_equal(mask[5], (frames[-1] >= 133).astype(np.float32))
    with pytest.raises(ValueError):
        persistence(np.zeros((0, 8, 8)))


@pytest.mark.req("STORM-06")
@pytest.mark.parametrize("shift", [(1.0, 0.0), (0.0, 2.0), (1.0, -1.0)])
def test_estimate_motion_recovers_translation(shift):
    from outrun.hazards.storm.baselines import estimate_motion

    dx, dy = shift
    frames = np.stack([blob(48, 48, 22 + k * dy, 20 + k * dx, sigma=4.0) for k in range(4)])
    m = estimate_motion(frames)
    assert m.shape == (2, 48, 48) and m.dtype == np.float32
    sel = frames[-1] > 0.1 * frames[-1].max()
    assert abs(np.median(m[0][sel]) - dx) <= 0.35
    assert abs(np.median(m[1][sel]) - dy) <= 0.35
    with pytest.raises(ValueError):
        estimate_motion(frames[:1])


@pytest.mark.req("STORM-06")
def test_extrapolate_integer_shift_is_exact():
    from outrun.hazards.storm.baselines import extrapolate

    frame = np.random.default_rng(1).random((10, 12)).astype(np.float32)
    motion = np.zeros((2, 10, 12), np.float32)
    motion[0] = 1.0  # dx = +1 column per frame
    out = extrapolate(frame, motion, 3)
    assert out.shape == (3, 10, 12) and out.dtype == np.float32
    np.testing.assert_allclose(out[0][:, 1:], frame[:, :-1], atol=1e-5)
    np.testing.assert_allclose(out[0][:, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(out[2][:, 3:], frame[:, :-3], atol=1e-5)
    motion[0], motion[1] = 0.0, -2.0  # dy = -2 rows per frame
    out = extrapolate(frame, motion, 1)
    np.testing.assert_allclose(out[0][:-2, :], frame[2:, :], atol=1e-5)
    np.testing.assert_allclose(out[0][-2:, :], 0.0, atol=1e-5)


@pytest.mark.req("STORM-06")
def test_extrapolation_nowcast_tracks_moving_blob():
    from outrun.hazards.storm.baselines import extrapolation_nowcast

    frames = np.stack([blob(40, 64, 20, 10 + k, sigma=3.0) for k in range(13)])  # 1 px/frame to the right
    out = extrapolation_nowcast(frames, n_out=12)
    assert out.shape == (12, 40, 64) and out.dtype == np.float32
    assert float(out.min()) >= 0.0
    r, c = np.unravel_index(np.argmax(out[5]), out[5].shape)  # step 6 -> centre at col 10 + 12 + 6
    assert abs(c - 28) <= 1.5 and abs(r - 20) <= 1.5


# ---------------------------------------------------------------- STORM-07 calibration and slotting
@pytest.mark.req("STORM-07")
def test_per_lead_calibrator_independent_monotone():
    from outrun.hazards.storm.calibration import PerLeadCalibrator

    rng = np.random.default_rng(0)
    probs = rng.random((200, 2, 4)).astype(np.float32)
    truth = np.zeros_like(probs, dtype=bool)
    truth[:, 0] = probs[:, 0] > 0.5  # lead 0 informative, lead 1 never happens
    cal = PerLeadCalibrator().fit(probs, truth)
    grid = np.linspace(0, 1, 11, dtype=np.float32)
    test = np.stack([np.stack([grid, grid], axis=0)], axis=0)  # (1, 2, 11)
    out = cal.transform(test)
    assert out.shape == test.shape and out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    assert np.all(np.diff(out[0, 0]) >= -1e-7)
    assert out[0, 0, -1] > out[0, 0, 0]
    np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(
        cal.transform(np.array([[[-0.5], [2.0]]], np.float32))[0, 0], out[0, 0, 0], atol=1e-6
    )


@pytest.mark.req("STORM-07")
def test_per_lead_calibrator_errors():
    from outrun.hazards.storm.calibration import PerLeadCalibrator

    with pytest.raises(RuntimeError):
        PerLeadCalibrator().transform(np.zeros((1, 2, 3)))
    rng = np.random.default_rng(1)
    p = rng.random((50, 2, 3))
    cal = PerLeadCalibrator().fit(p, p > 0.5)
    with pytest.raises(ValueError):
        cal.transform(np.zeros((1, 3, 3)))


@pytest.mark.req("STORM-07")
def test_frames_to_slots_max():
    from outrun.hazards.storm.calibration import frames_to_slots

    p = np.zeros((12, 2, 2), np.float32)
    p[1, 0, 0] = 0.7
    p[2, 0, 0] = 0.4
    p[5, 1, 1] = 0.9
    p[9, 0, 1] = 0.2
    s = frames_to_slots(p)
    assert s.shape == (4, 2, 2)
    assert s[0, 0, 0] == pytest.approx(0.7)
    assert s[1, 1, 1] == pytest.approx(0.9)
    assert s[3, 0, 1] == pytest.approx(0.2)
    assert s[2].max() == 0.0
    with pytest.raises(ValueError):
        frames_to_slots(np.zeros((10, 2, 2)))


# ---------------------------------------------------------------- STORM-08 adapter
def _frames(with_truth: bool = True, with_ir: bool = True, held_out: bool = True):
    from outrun.hazards.storm.adapter import StormFrames

    rng = np.random.default_rng(5)
    vil = rng.integers(0, 256, (13, H, W)).astype(np.uint8)
    ir = rng.integers(0, 256, (13, H, W)).astype(np.uint8) if with_ir else None
    fut = rng.integers(0, 256, (12, H, W)).astype(np.uint8) if with_truth else None
    x = -150_000.0 + 500.0 + 1000.0 * np.arange(W)
    y = 1_300_000.0 - 500.0 - 1000.0 * np.arange(H)
    return StormFrames(vil=vil, ir=ir, future_vil=fut, x=x, y=y, regrid=lambda a: a, held_out=held_out)


def _check_layer(layer, t0, kind, frames):
    from outrun.core.risk import Hazard

    assert layer.hazard == Hazard.STORM
    assert layer.kind == kind
    assert layer.issued == t0
    assert layer.valid_until == t0 + timedelta(minutes=storm_cfg()["native_horizon_min"])
    assert layer.valid_until == t0 + timedelta(minutes=60)
    assert layer.source.startswith("storm.")
    assert layer.held_out == frames.held_out
    assert layer.edge_overrides is None
    assert tuple(layer.prob.dims) == ("slot", "y", "x")
    assert layer.prob.shape == (12, H, W)
    assert layer.prob.dtype == np.float16
    v = layer.prob.values.astype(np.float32)
    assert np.isfinite(v).all() and v.min() >= 0.0 and v.max() <= 1.0
    # beyond-horizon decay (HORIZON-01 via core): never above slot 3, non-increasing over slots 4..11
    tail = v[4:]
    assert np.all(tail <= v[3][None] + 1e-3)
    assert np.all(np.diff(v[3:], axis=0) <= 1e-3)
    return v


@pytest.mark.req("STORM-08")
def test_adapter_policy_matches_config():
    from outrun.core.policy import Profile
    from outrun.core.risk import Hazard
    from outrun.hazards.storm.adapter import StormAdapter

    cfg = storm_cfg()
    ad = StormAdapter(provider=lambda scn: _frames())
    assert ad.hazard is Hazard.STORM
    pol = ad.policy
    for prof in Profile:
        assert pol.block_at[prof] is None
        assert pol.exposure_weight[prof] == pytest.approx(cfg["exposure_weight"][prof.name.lower()])
    assert pol.exposure_weight[Profile.CIVILIAN] == pytest.approx(2.0)
    for p in (0.0, 0.5, 1.0):
        assert pol.speed_factor(p) == pytest.approx(1 - cfg["speed_factor_slope"] * p)
    assert pol.speed_factor(1.0) == pytest.approx(0.7)
    assert pol.beyond_horizon == "decay" == cfg["beyond_horizon"]
    assert pol.decay_tau_min == pytest.approx(30)


@pytest.mark.req("STORM-08")
def test_adapter_baseline_is_persistence_mask(t0):
    from outrun.hazards.storm.adapter import StormAdapter

    fr = _frames()
    layer = StormAdapter(provider=lambda scn: fr).baseline(make_scenario(t0))
    v = _check_layer(layer, t0, "baseline", fr)
    mask = (fr.vil[-1] >= 133).astype(np.float32)
    for s in range(4):
        np.testing.assert_array_equal(v[s], mask)


@pytest.mark.req("STORM-08")
def test_adapter_predict_with_model(t0):
    from outrun.hazards.storm.adapter import StormAdapter
    from outrun.hazards.storm.model import StormUNet

    fr = _frames()
    torch.manual_seed(0)
    ad = StormAdapter(provider=lambda scn: fr, model=StormUNet(base=4, levels=2))
    a = ad.predict(make_scenario(t0))
    b = ad.predict(make_scenario(t0))
    _check_layer(a, t0, "forecast", fr)
    np.testing.assert_array_equal(a.prob.values, b.prob.values)


@pytest.mark.req("STORM-08")
def test_adapter_predict_errors(t0):
    from outrun.hazards.storm.adapter import StormAdapter
    from outrun.hazards.storm.model import StormUNet

    with pytest.raises(RuntimeError):
        StormAdapter(provider=lambda scn: _frames()).predict(make_scenario(t0))
    no_ir = _frames(with_ir=False)
    with pytest.raises(ValueError):
        StormAdapter(provider=lambda scn: no_ir, model=StormUNet(base=4, levels=2)).predict(make_scenario(t0))
    # the VIL-only ablation model needs no IR
    layer = StormAdapter(
        provider=lambda scn: no_ir, model=StormUNet(in_channels=13, base=4, levels=2)
    ).predict(make_scenario(t0))
    assert layer.kind == "forecast"


@pytest.mark.req("STORM-08")
def test_adapter_observed(t0):
    from outrun.hazards.storm.adapter import StormAdapter

    fr = _frames()
    layer = StormAdapter(provider=lambda scn: fr).observed(make_scenario(t0))
    v = _check_layer(layer, t0, "observed", fr)
    exc = (fr.future_vil >= 133).astype(np.float32)
    for s in range(4):
        np.testing.assert_array_equal(v[s], exc[3 * s : 3 * s + 3].max(axis=0))
    no_truth = _frames(with_truth=False)
    assert StormAdapter(provider=lambda scn: no_truth).observed(make_scenario(t0)) is None


# ---------------------------------------------------------------- STORM-09 motion export
@pytest.mark.req("STORM-09")
def test_motion_to_mps_flips_row_axis():
    from outrun.hazards.storm.baselines import motion_at, motion_to_mps

    m = np.zeros((2, 3, 4), np.float32)
    m[0], m[1] = 1.0, 1.0  # one pixel right and one pixel down (south) per frame
    mps = motion_to_mps(m, pixel_m=2000.0, frame_s=300.0)
    assert mps.shape == (2, 3, 4)
    u, v = motion_at(mps, 1, 2)
    assert isinstance(u, float) and isinstance(v, float)
    assert u == pytest.approx(2000 / 300)
    assert v == pytest.approx(-2000 / 300)
    with pytest.raises(IndexError):
        motion_at(mps, 3, 0)


# ---------------------------------------------------------------- STORM-10 scores
@pytest.mark.req("STORM-10")
def test_csi_counts():
    from outrun.hazards.storm.metrics import CSI_THRESHOLDS, csi

    assert CSI_THRESHOLDS == (16, 74, 133, 160, 181, 219)
    truth = np.array([200, 200, 200, 0, 0], float)
    pred = np.array([200, 200, 0, 200, 0], float)  # hits 2, misses 1, false alarms 1
    assert csi(pred, truth, 133) == pytest.approx(0.5)
    assert csi(truth, truth, 133) == pytest.approx(1.0)
    assert csi(np.zeros(5), truth, 133) == pytest.approx(0.0)
    assert math.isnan(csi(np.zeros(5), np.zeros(5), 133))
    assert csi(np.array([133.0]), np.array([133.0]), 133) == pytest.approx(1.0)  # inclusive
    with pytest.raises(ValueError):
        csi(np.zeros(4), np.zeros(5), 133)


@pytest.mark.req("STORM-10")
def test_fss_properties():
    from outrun.hazards.storm.metrics import fss

    a = np.zeros((9, 9))
    a[4, 4] = 200
    b = np.zeros((9, 9))
    b[4, 5] = 200  # shifted by one pixel
    assert fss(a, a, 133, 3) == pytest.approx(1.0)
    assert fss(a, b, 133, 1) == pytest.approx(0.0)
    assert fss(a, b, 133, 3) > fss(a, b, 133, 1)
    # window 3, shifted single pixel: 9 cells of 1/9 each side, overlap 6 -> 1 - (6 * (1/9)**2) / (18 * (1/9)**2)
    assert fss(a, b, 133, 3) == pytest.approx(1 - 6 / 18)
    assert math.isnan(fss(np.zeros((9, 9)), np.zeros((9, 9)), 133, 3))
    with pytest.raises(ValueError):
        fss(a, a, 133, 2)


@pytest.mark.req("STORM-10")
def test_storm_scores_per_lead_and_crop():
    from outrun.hazards.storm.metrics import storm_scores

    truth = np.zeros((2, 12, 16, 16))
    truth[:, :, 6:10, 6:10] = 200
    pred = truth.copy()
    pred[:, 6:, :, :] = 0  # leads 7..12 miss everything
    pred[:, :, 0, 0] = 200  # a false alarm at the corner (outside the central 8x8)
    s = storm_scores(pred, truth)
    assert set(s) >= {"csi", "fss", "mse"}
    assert set(s["csi"]) == {16, 74, 133, 160, 181, 219}
    assert set(s["fss"]) == {3, 7}
    c133 = s["csi"][133]
    assert c133.shape == (12,) and s["mse"].shape == (12,) and s["fss"][3].shape == (12,)
    assert c133[0] == pytest.approx(16 / 17)  # 16 hits, 1 false alarm per sample -> pooled 32 / 34
    assert c133[11] == pytest.approx(0.0)
    assert math.isnan(s["csi"][219][0])
    assert s["mse"][11] > s["mse"][0]
    cropped = storm_scores(pred, truth, crop=8)
    assert cropped["csi"][133][0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        storm_scores(pred, truth, crop=32)


@pytest.mark.req("STORM-01")
@pytest.mark.req("SPLIT-07")
def test_dataset_from_split_excludes_test_and_quarantined_events():
    from datetime import date

    from outrun.eval.splits import Split, SplitItem
    from outrun.hazards.storm.dataset import SevirWindowDataset

    split = Split(
        "toy",
        train=tuple(SplitItem(i) for i in ("E0", "E2")),
        val=tuple(SplitItem(i) for i in ("E1",)),
        test=tuple(SplitItem(i) for i in ("E3",)),
        extra={"case_holdout": (SplitItem("E4", date(2024, 5, 6)),)},
    )
    ev = make_events(5)
    ids = ["E0", "E1", "E2", "E3", "E4"]
    tr = SevirWindowDataset.from_split(ev, ids, split, "train")
    va = SevirWindowDataset.from_split(ev, ids, split, "val")
    te = SevirWindowDataset.from_split(ev, ids, split, "test")
    assert tuple(tr.ids) == ("E0", "E2") and len(tr) == 2 * 3
    assert tuple(va.ids) == ("E1",) and tuple(te.ids) == ("E3",)
    for ds in (tr, va):
        assert "E3" not in ds.ids and "E4" not in ds.ids
    ref = SevirWindowDataset(ev, event_ids=[0, 2])
    for i in range(len(tr)):
        np.testing.assert_array_equal(tr[i][0].numpy(), ref[i][0].numpy())
    with pytest.raises(ValueError):
        SevirWindowDataset.from_split(ev, ids[:4], split, "train")


@pytest.mark.req("STORM-08")
def test_storm_adapter_zero_argument_construction_and_default_factory():
    from outrun.config import get_paths
    from outrun.core.risk import Hazard
    from outrun.hazards.storm.adapter import StormAdapter, default_adapter

    ad = StormAdapter()  # the registry stores zero-argument factories (RISK-13); nothing is loaded
    assert ad.hazard is Hazard.STORM and ad.policy.beyond_horizon == "decay"
    d = default_adapter()
    assert isinstance(d, StormAdapter) and d.checkpoint == get_paths().models / "storm" / "best.pt"
    assert StormAdapter(checkpoint=None).checkpoint is None
