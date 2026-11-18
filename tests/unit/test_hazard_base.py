"""specs/02-grid-risk.md: RISK-13 (adapter registry and conformance), RISK-14..16 (shared training loop)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import torch
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset

from outrun.core import risk as core_risk
from outrun.core.grid import GridWindow
from outrun.core.policy import HazardPolicy, Profile
from outrun.core.risk import Hazard, RiskLayer
from outrun.core.scenario import Scenario
from outrun.hazards import base
from outrun.hazards.base import (
    AdapterRegistry,
    EpochTiming,
    TrainConfig,
    load_checkpoint,
    register_adapter,
    save_checkpoint,
    select_device,
    time_one_epoch,
    train_model,
    validate_adapter_output,
)

N = 8
WIN = GridWindow(x0=-150_000.0, y0=1_300_000.0, nx=N, ny=N)
XC = -150_000.0 + 500.0 + 1000.0 * np.arange(N)
YC = 1_300_000.0 - 500.0 - 1000.0 * np.arange(N)
T0 = datetime(2019, 5, 26, 3, 0, tzinfo=UTC)
POLICY = HazardPolicy(
    block_at={p: None for p in Profile},
    speed_factor=lambda p: 1.0,
    exposure_weight={p: 1.0 for p in Profile},
    beyond_horizon="hold",
    decay_tau_min=None,
)


def _scn():
    return Scenario(
        id="unit", t0=T0, bbox=(-98.1, 35.1, -97.1, 35.8), city="Test", hazards=tuple(Hazard), window=WIN
    )


def _layer(hazard, kind, x=XC, issued=T0, valid_until=T0 + timedelta(hours=3)):
    da = xr.DataArray(
        np.full((12, N, N), 0.2, dtype=np.float16),
        dims=("slot", "y", "x"),
        coords={"slot": np.arange(12), "y": YC, "x": x},
    )
    return RiskLayer(
        hazard=hazard,
        prob=da,
        edge_overrides=None,
        kind=kind,
        issued=issued,
        valid_until=valid_until,
        source="fake",
        held_out=False,
    )


class GoodAdapter:
    hazard = Hazard.SMOKE
    policy = POLICY

    def baseline(self, scn):
        return _layer(self.hazard, "baseline")

    def predict(self, scn):
        return _layer(self.hazard, "forecast")

    def observed(self, scn):
        return None


# ---------------------------------------------------------------- RISK-13
@pytest.mark.req("RISK-13")
def test_registry_register_get_cache_order():
    reg = AdapterRegistry()
    calls = []

    def make_smoke():
        calls.append(1)
        return GoodAdapter()

    class Storm(GoodAdapter):
        hazard = Hazard.STORM

    reg.register(Hazard.SMOKE, make_smoke)
    reg.register(Hazard.STORM, Storm)
    assert reg.hazards() == [Hazard.STORM, Hazard.SMOKE]
    a1 = reg.get(Hazard.SMOKE)
    a2 = reg.get("smoke")
    assert a1 is a2 and len(calls) == 1
    assert isinstance(reg.get(Hazard.STORM), Storm)
    with pytest.raises(ValueError):
        reg.register(Hazard.SMOKE, make_smoke)
    reg.register(Hazard.SMOKE, make_smoke, replace=True)
    with pytest.raises(KeyError):
        reg.get(Hazard.ICE)
    with pytest.raises(KeyError):
        reg.get("lava")


@pytest.mark.req("RISK-13")
def test_registry_hazard_mismatch_and_protocol_reexport():
    assert base.HazardAdapter is core_risk.HazardAdapter
    reg = AdapterRegistry()
    reg.register(Hazard.FLOOD, GoodAdapter)  # adapter says SMOKE
    with pytest.raises(ValueError):
        reg.get(Hazard.FLOOD)
    assert isinstance(GoodAdapter(), base.HazardAdapter)


@pytest.mark.req("RISK-13")
def test_register_adapter_decorator_uses_module_registry(monkeypatch):
    monkeypatch.setattr(
        base, "REGISTRY", AdapterRegistry(), raising=False
    )  # keep the real registry untouched

    class Ice(GoodAdapter):
        hazard = Hazard.ICE

    out = register_adapter(Hazard.ICE)(Ice)
    assert out is Ice
    assert Hazard.ICE in base.REGISTRY.hazards()
    assert isinstance(base.get_adapter(Hazard.ICE), Ice)


@pytest.mark.req("RISK-13")
def test_validate_adapter_output_accepts_good():
    out = validate_adapter_output(GoodAdapter(), _scn())
    assert set(out) == {"baseline", "predict", "observed"}
    assert out["observed"] is None and out["predict"].kind == "forecast"


@pytest.mark.req("RISK-13")
@pytest.mark.parametrize(
    "method,layer",
    [
        ("predict", lambda: _layer(Hazard.STORM, "forecast")),  # wrong hazard
        ("predict", lambda: _layer(Hazard.SMOKE, "baseline")),  # wrong kind
        ("baseline", lambda: _layer(Hazard.SMOKE, "baseline", x=XC + 1000.0)),  # wrong grid
        (
            "predict",
            lambda: _layer(
                Hazard.SMOKE,
                "forecast",
                issued=T0 + timedelta(minutes=5),
                valid_until=T0 + timedelta(hours=3),
            ),
        ),  # issued after t0
        (
            "observed",
            lambda: _layer(Hazard.SMOKE, "observed", issued=T0 - timedelta(hours=2), valid_until=T0),
        ),  # expires at t0
        ("baseline", lambda: "not a layer"),
    ],
)
def test_validate_adapter_output_rejects(method, layer):
    ad = GoodAdapter()
    setattr(ad, method, lambda scn: layer())
    with pytest.raises(ValueError):
        validate_adapter_output(ad, _scn())


# ---------------------------------------------------------------- training fixtures
def _linear(w0=0.0):
    m = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(w0)
    return m


def _loader(x, y, bs):
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=False)


# ---------------------------------------------------------------- RISK-14
@pytest.mark.req("RISK-14")
def test_early_stopping_restores_best(tmp_path):
    # train target y = 2x pushes w 0 -> 0.4 -> 0.72 -> ...; val target y = 0.5x is best at epoch 1 (w = 0.4)
    x = torch.ones(8, 1)
    train = _loader(x, 2.0 * x, 8)
    val = _loader(x, 0.5 * x, 8)
    model = _linear()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = TrainConfig(checkpoint_dir=tmp_path / "ckpt", max_epochs=20, patience=2)
    res = train_model(model, train, val, torch.nn.MSELoss(), cfg, optimizer=opt)
    assert res.best_epoch == 1 and res.epochs_run == 3 and res.stopped_early is True
    assert res.best_val_loss == pytest.approx(0.01, abs=1e-6)
    assert float(model.weight) == pytest.approx(0.4, abs=1e-6)  # best weights restored
    assert [h["epoch"] for h in res.history] == [1, 2, 3]
    assert all({"epoch", "train_loss", "val_loss", "seconds"} <= set(h) for h in res.history)
    assert res.history[0]["train_loss"] == pytest.approx(4.0, abs=1e-5)  # (0 - 2)^2 before the first step
    assert res.history[1]["val_loss"] == pytest.approx((0.72 - 0.5) ** 2, abs=1e-5)


@pytest.mark.req("RISK-14")
def test_runs_to_max_epochs_without_early_stop_and_weighted_losses(tmp_path):
    x = torch.ones(10, 1)
    train = _loader(x, 2.0 * x, 4)  # batches of 4, 4, 2
    val = _loader(torch.cat([x[:4], 2 * x[:2]]), torch.zeros(6, 1), 4)  # batches: 4 ones, then 2 twos
    model = _linear(0.5)
    cfg = TrainConfig(checkpoint_dir=tmp_path, max_epochs=3, patience=10, lr=0.0)  # lr 0: weights fixed
    res = train_model(model, train, val, torch.nn.MSELoss(), cfg)
    assert res.epochs_run == 3 and res.stopped_early is False
    # val loss weighted by batch size: (4 * 0.25 + 2 * 1.0) / 6 = 0.5
    assert res.history[0]["val_loss"] == pytest.approx(0.5, abs=1e-6)
    assert res.history[0]["train_loss"] == pytest.approx(1.5**2, abs=1e-6)
    assert res.best_epoch == 1  # no strict improvement after epoch 1


@pytest.mark.req("RISK-14")
def test_nan_loss_and_empty_val_raise(tmp_path):
    x = torch.ones(4, 1)
    train = _loader(x, x, 4)

    def nan_loss(out, y):
        return (out - y).mean() * float("nan")

    with pytest.raises(FloatingPointError):
        train_model(_linear(), train, _loader(x, x, 4), nan_loss, TrainConfig(checkpoint_dir=tmp_path / "a"))
    empty = _loader(torch.zeros(0, 1), torch.zeros(0, 1), 4)
    with pytest.raises(ValueError):
        train_model(_linear(), train, empty, torch.nn.MSELoss(), TrainConfig(checkpoint_dir=tmp_path / "b"))


# ---------------------------------------------------------------- RISK-15
def _data():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 3, generator=g)
    y = x @ torch.tensor([[1.0], [-2.0], [0.5]]) + 0.1 * torch.randn(32, 1, generator=g)
    return _loader(x[:24], y[:24], 8), _loader(x[24:], y[24:], 8)


def _model():
    torch.manual_seed(123)
    return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))


@pytest.mark.req("RISK-15")
def test_resume_equals_uninterrupted(tmp_path):
    train, val = _data()
    full = _model()
    r_full = train_model(
        full,
        train,
        val,
        torch.nn.MSELoss(),
        TrainConfig(checkpoint_dir=tmp_path / "full", max_epochs=4, patience=100, lr=0.05),
    )
    part = _model()
    train_model(
        part,
        train,
        val,
        torch.nn.MSELoss(),
        TrainConfig(checkpoint_dir=tmp_path / "part", max_epochs=2, patience=100, lr=0.05),
    )
    assert (tmp_path / "part" / "last.pt").exists() and (tmp_path / "part" / "best.pt").exists()
    resumed = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))  # fresh init
    r_res = train_model(
        resumed,
        train,
        val,
        torch.nn.MSELoss(),
        TrainConfig(checkpoint_dir=tmp_path / "part", max_epochs=4, patience=100, lr=0.05, resume=True),
    )
    assert r_res.epochs_run == 4 == r_full.epochs_run
    assert [h["epoch"] for h in r_res.history] == [1, 2, 3, 4]
    np.testing.assert_allclose(
        [h["val_loss"] for h in r_res.history], [h["val_loss"] for h in r_full.history], rtol=1e-6
    )
    for a, b in zip(full.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-6)
    assert r_res.best_checkpoint == tmp_path / "part" / "best.pt"
    assert r_res.last_checkpoint == tmp_path / "part" / "last.pt"


@pytest.mark.req("RISK-15")
def test_resume_without_checkpoint_starts_fresh(tmp_path):
    train, val = _data()
    r = train_model(
        _model(),
        train,
        val,
        torch.nn.MSELoss(),
        TrainConfig(checkpoint_dir=tmp_path / "new" / "dir", max_epochs=2, patience=5, resume=True),
    )
    assert r.epochs_run == 2 and len(r.history) == 2
    assert (tmp_path / "new" / "dir" / "last.pt").exists()


@pytest.mark.req("RISK-15")
def test_save_load_checkpoint_round_trip(tmp_path):
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    save_checkpoint(tmp_path / "c.pt", m, opt, epoch=7, note="hi")
    other = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    extra = load_checkpoint(tmp_path / "c.pt", other, torch.optim.AdamW(other.parameters(), lr=1e-3))
    assert extra["epoch"] == 7 and extra["note"] == "hi"
    for a, b in zip(m.parameters(), other.parameters(), strict=True):
        assert torch.equal(a, b)


# ---------------------------------------------------------------- RISK-16
@pytest.mark.req("RISK-16")
def test_time_one_epoch(tmp_path):
    train, _ = _data()
    m = _model()
    before = [p.detach().clone() for p in m.parameters()]
    t = time_one_epoch(m, train, torch.nn.MSELoss())
    assert isinstance(t, EpochTiming)
    assert t.seconds > 0 and t.batches == 3 and t.samples == 24
    assert t.estimate_total_s(20) == pytest.approx(t.seconds * 20)
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters(), strict=True))  # it trained
    t2 = time_one_epoch(m, train, torch.nn.MSELoss(), max_batches=2)
    assert t2.batches == 2 and t2.samples == 16
    with pytest.raises(FloatingPointError):
        time_one_epoch(m, train, lambda o, y: (o - y).sum() * float("inf") * 0.0)


@pytest.mark.req("RISK-16")
def test_select_device():
    assert select_device("cpu") == torch.device("cpu")
    d = select_device("mps")
    assert d.type in {"mps", "cpu"}
    if not torch.backends.mps.is_available():
        assert d.type == "cpu"
    with pytest.raises(ValueError):
        select_device("cuda9")


@pytest.mark.req("RISK-13")
def test_load_default_adapters_registers_all_six(monkeypatch):
    from outrun.core.risk import Hazard
    from outrun.hazards import base

    monkeypatch.delenv("OUTRUN_CONFIG_DIR", raising=False)
    reg = base.load_default_adapters()
    assert reg is base.REGISTRY
    assert reg.hazards() == list(Hazard)
    for h in Hazard:
        assert base.get_adapter(h).hazard is h  # constructed with no data or model loaded
    assert base.load_default_adapters() is base.REGISTRY  # idempotent
    assert reg.hazards() == list(Hazard)
