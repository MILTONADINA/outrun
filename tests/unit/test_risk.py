"""specs/02-grid-risk.md: RISK-01..08 (RiskLayer, RiskStack, zarr IO, sampling, overlays, risk-source switch)."""

from __future__ import annotations

import dataclasses
import io
import math
from dataclasses import fields
from datetime import timedelta

import numpy as np
import pytest
import xarray as xr
from PIL import Image

from outrun.core.grid import GridWindow
from outrun.core.policy import HazardPolicy, Profile
from outrun.core.risk import (
    HAZARD_COLORS,
    RISK_SOURCES,
    Hazard,
    HazardAdapter,
    LayerMeta,
    RiskLayer,
    RiskStack,
    build_stack,
    decode_overlay,
)
from outrun.core.scenario import Scenario

N = 32
WIN = GridWindow(x0=-150_000.0, y0=1_300_000.0, nx=N, ny=N, res=1000.0)
XC = -150_000.0 + 500.0 + 1000.0 * np.arange(N)
YC = 1_300_000.0 - 500.0 - 1000.0 * np.arange(N)


def _policy(rule="hold", tau=None, block=None):
    return HazardPolicy(
        block_at={p: block for p in Profile},
        speed_factor=lambda p: 1.0,
        exposure_weight={p: 1.0 for p in Profile},
        beyond_horizon=rule,
        decay_tau_min=tau,
    )


HOLD = {h: _policy() for h in Hazard}


def _da(arr):
    arr = np.asarray(arr)
    return xr.DataArray(
        arr, dims=("slot", "y", "x"), coords={"slot": np.arange(arr.shape[0]), "y": YC, "x": XC}
    )


def _layer(hazard, arr, t0, kind="forecast", horizon_min=180, overrides=None, source="test", held_out=False):
    return RiskLayer(
        hazard=hazard,
        prob=_da(np.asarray(arr, dtype=np.float16)),
        edge_overrides=overrides,
        kind=kind,
        issued=t0,
        valid_until=t0 + timedelta(minutes=horizon_min),
        source=source,
        held_out=held_out,
    )


def _scenario(t0):
    return Scenario(
        id="unit", t0=t0, bbox=(-98.1, 35.1, -97.1, 35.8), city="Test", hazards=tuple(Hazard), window=WIN
    )


# ---------------------------------------------------------------- RISK-01
@pytest.mark.req("RISK-01")
def test_hazard_enum_order_and_values():
    assert [h.name for h in Hazard] == ["STORM", "TORNADO", "FLOOD", "WILDFIRE", "ICE", "SMOKE"]
    assert [h.value for h in Hazard] == ["storm", "tornado", "flood", "wildfire", "ice", "smoke"]
    assert Hazard("storm") is Hazard.STORM


@pytest.mark.req("RISK-01")
def test_risk_layer_fields_exact_and_frozen(t0, risk_cube):
    assert [f.name for f in fields(RiskLayer)] == [
        "hazard",
        "prob",
        "edge_overrides",
        "kind",
        "issued",
        "valid_until",
        "source",
        "held_out",
    ]
    lay = _layer(Hazard.STORM, risk_cube, t0, overrides={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        lay.source = "x"  # type: ignore[misc]
    assert _layer(Hazard.FLOOD, risk_cube, t0, overrides={7: np.full(12, 0.3)}).edge_overrides[7].shape == (
        12,
    )


@pytest.mark.req("RISK-01")
@pytest.mark.parametrize(
    "bad", ["hazard", "dims", "noslot", "nan", "range", "kind", "naive", "order", "override"]
)
def test_risk_layer_validation(bad, t0, risk_cube):
    kw = dict(
        hazard=Hazard.STORM,
        prob=_da(risk_cube.astype(np.float16)),
        edge_overrides=None,
        kind="forecast",
        issued=t0,
        valid_until=t0 + timedelta(hours=1),
        source="s",
        held_out=False,
    )
    if bad == "hazard":
        kw["hazard"] = "storm"
    elif bad == "dims":
        kw["prob"] = kw["prob"].transpose("y", "slot", "x")
    elif bad == "noslot":
        kw["prob"] = _da(np.zeros((0, N, N), dtype=np.float16))
    elif bad == "nan":
        a = risk_cube.copy()
        a[0, 0, 0] = np.nan
        kw["prob"] = _da(a)
    elif bad == "range":
        kw["prob"] = _da(risk_cube + 0.5)
    elif bad == "kind":
        kw["kind"] = "nowcast"
    elif bad == "naive":
        kw["issued"] = t0.replace(tzinfo=None)
    elif bad == "order":
        kw["valid_until"] = t0
    elif bad == "override":
        kw["edge_overrides"] = {3: np.full(5, 0.2)}
    with pytest.raises(ValueError):
        RiskLayer(**kw)


# ---------------------------------------------------------------- RISK-02
@pytest.mark.req("RISK-02")
def test_from_layers_shape_order_coords_meta(t0, risk_cube):
    lay = _layer(Hazard.FLOOD, risk_cube, t0, source="flood.test", held_out=True)
    st = RiskStack.from_layers([lay], t0, WIN, policies=HOLD)
    d = st.data
    assert d.dims == ("hazard", "slot", "y", "x")
    assert d.shape == (6, 12, N, N) and d.dtype == np.float16
    assert list(d.coords["hazard"].values) == [h.value for h in Hazard]
    np.testing.assert_array_equal(d.coords["slot"].values, np.arange(12))
    np.testing.assert_allclose(d.coords["x"].values, XC)
    np.testing.assert_allclose(d.coords["y"].values, YC)
    assert st.t0 == t0 and st.window == WIN and st.n_slots == 12
    np.testing.assert_array_equal(d.sel(hazard="flood").values, risk_cube.astype(np.float16))
    assert float(d.sel(hazard="storm").max()) == 0.0
    assert set(st.meta) == {Hazard.FLOOD}
    assert st.meta[Hazard.FLOOD] == LayerMeta(
        kind="forecast",
        source="flood.test",
        issued=t0,
        valid_until=t0 + timedelta(minutes=180),
        held_out=True,
    )


@pytest.mark.req("RISK-02")
def test_from_layers_applies_beyond_horizon(t0):
    arr = np.full((12, N, N), 0.8, dtype=np.float32)
    lay = _layer(Hazard.STORM, arr, t0, horizon_min=60, overrides={5: np.full(12, 0.8)})
    pol = dict(HOLD)
    pol[Hazard.STORM] = _policy("decay", tau=30.0)
    st = RiskStack.from_layers([lay], t0, WIN, policies=pol)
    s = st.data.sel(hazard="storm").values.astype(np.float32)
    np.testing.assert_allclose(s[3], 0.8, atol=1e-3)
    np.testing.assert_allclose(s[5], 0.8 * math.exp(-0.5), atol=1e-3)
    assert float(np.abs(s[8:]).max()) == 0.0
    ov = st.edge_overrides[Hazard.STORM][5]
    assert ov.shape == (12,)
    assert ov[5] == pytest.approx(0.8 * math.exp(-0.5), abs=1e-3) and ov[9] == 0.0


@pytest.mark.req("RISK-02")
def test_from_layers_errors(t0, risk_cube):
    good = _layer(Hazard.ICE, risk_cube, t0)
    with pytest.raises(ValueError):
        RiskStack.from_layers([good, good], t0, WIN, policies=HOLD)
    shifted = RiskLayer(
        hazard=Hazard.ICE,
        prob=_da(risk_cube).assign_coords(x=XC + 1000.0),
        edge_overrides=None,
        kind="forecast",
        issued=t0,
        valid_until=t0 + timedelta(hours=3),
        source="s",
        held_out=False,
    )
    with pytest.raises(ValueError):
        RiskStack.from_layers([shifted], t0, WIN, policies=HOLD)
    with pytest.raises(ValueError):
        RiskStack.from_layers([good], t0.replace(tzinfo=None), WIN, policies=HOLD)


@pytest.mark.req("RISK-02")
@pytest.mark.parametrize("drop", [("x",), ("y",), ("x", "y")])
def test_from_layers_rejects_coordless_layer(t0, risk_cube, drop):
    bare = xr.DataArray(np.asarray(risk_cube, dtype=np.float16), dims=("slot", "y", "x"))
    if len(drop) == 1:  # keep the other coordinate
        keep = "y" if drop == ("x",) else "x"
        bare = bare.assign_coords({keep: YC if keep == "y" else XC})
    lay = RiskLayer(
        hazard=Hazard.ICE,
        prob=bare,
        edge_overrides=None,
        kind="forecast",
        issued=t0,
        valid_until=t0 + timedelta(hours=3),
        source="s",
        held_out=False,
    )
    with pytest.raises(ValueError):
        RiskStack.from_layers([lay], t0, WIN, policies=HOLD)


@pytest.mark.req("RISK-02")
def test_zeros_stack(t0):
    st = RiskStack.zeros(t0, WIN)
    assert st.data.shape == (6, 12, N, N) and float(st.data.max()) == 0.0 and st.meta == {}


# ---------------------------------------------------------------- RISK-03
@pytest.mark.req("RISK-03")
def test_slot_timing(t0):
    st = RiskStack.zeros(t0, WIN)
    for s in range(12):
        start = st.slot_start(s)
        assert start == t0 + timedelta(minutes=15 * s)
        assert st.slot_of(start) == s
        assert st.slot_of(start + timedelta(minutes=14, seconds=59)) == s
    assert st.slot_of(t0 + timedelta(hours=5)) == 11
    with pytest.raises(IndexError):
        st.slot_start(12)
    with pytest.raises(ValueError):
        st.slot_of(t0 - timedelta(seconds=1))
    with pytest.raises(ValueError):
        st.slot_of(t0.replace(tzinfo=None))


# ---------------------------------------------------------------- RISK-04
@pytest.mark.req("RISK-04")
def test_zarr_round_trip(t0, risk_stack_array, tmp_path):
    layers = [
        _layer(
            h,
            risk_stack_array[i],
            t0,
            kind="baseline",
            source=f"src.{h.value}",
            held_out=(i % 2 == 0),
            overrides={11: np.linspace(0, 1, 12)} if h is Hazard.FLOOD else None,
        )
        for i, h in enumerate(Hazard)
        if h is not Hazard.SMOKE
    ]
    st = RiskStack.from_layers(layers, t0, WIN, policies=HOLD)
    path = tmp_path / "risk.zarr"
    st.to_zarr(path)
    back = RiskStack.from_zarr(path)
    assert back.data.dims == st.data.dims and back.data.dtype == np.float16
    np.testing.assert_array_equal(back.data.values.view(np.uint16), st.data.values.view(np.uint16))
    for c in ("hazard", "slot", "y", "x"):
        np.testing.assert_array_equal(back.data.coords[c].values, st.data.coords[c].values)
    assert back.t0 == t0 and back.t0.tzinfo is not None
    assert back.window == WIN
    assert back.meta == st.meta
    assert set(back.edge_overrides) == set(st.edge_overrides)
    np.testing.assert_array_equal(back.edge_overrides[Hazard.FLOOD][11], st.edge_overrides[Hazard.FLOOD][11])
    with pytest.raises(FileNotFoundError):
        RiskStack.from_zarr(tmp_path / "missing.zarr")


# ---------------------------------------------------------------- RISK-05
@pytest.fixture
def stack(t0, risk_stack_array):
    layers = [_layer(h, risk_stack_array[i], t0) for i, h in enumerate(Hazard)]
    return RiskStack.from_layers(layers, t0, WIN, policies=HOLD)


@pytest.mark.req("RISK-05")
def test_get_and_sample(stack, risk_stack_array):
    stored = risk_stack_array.astype(np.float16).astype(np.float32)
    g = stack.get(Hazard.TORNADO, 3)
    assert g.shape == (N, N) and g.dtype == np.float32
    np.testing.assert_array_equal(g, stored[1, 3])
    np.testing.assert_array_equal(stack.get("tornado", 3), g)
    # points anywhere inside a cell return that cell's value
    rows, cols = np.array([0, 5, 31, 17]), np.array([0, 9, 31, 2])
    x = XC[cols] + np.array([-499.0, 0.0, 499.0, 250.0])
    y = YC[rows] + np.array([499.0, 0.0, -499.0, -250.0])
    v = stack.sample(Hazard.ICE, 7, x, y)
    assert v.dtype == np.float32 and v.shape == (4,)
    np.testing.assert_array_equal(v, stored[4, 7, rows, cols])
    out = stack.sample(Hazard.ICE, 7, np.array([-10_000_000.0]), np.array([0.0]))
    assert out[0] == 0.0


@pytest.mark.req("RISK-05")
def test_sample_points_matches_sample(stack, rng):
    x = rng.uniform(-152_000, -116_000, 50)
    y = rng.uniform(1_266_000, 1_302_000, 50)
    sp = stack.sample_points(x, y)
    assert sp.shape == (6, 12, 50) and sp.dtype == np.float32
    for i, h in enumerate(Hazard):
        for s in (0, 6, 11):
            np.testing.assert_array_equal(sp[i, s], stack.sample(h, s, x, y))


# ---------------------------------------------------------------- RISK-06
@pytest.mark.req("RISK-06")
def test_overlay_png_format(stack, tmp_path):
    path = tmp_path / "o.png"
    png = stack.overlay_png(Hazard.WILDFIRE, 4, path=path)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and path.read_bytes() == png
    im = Image.open(io.BytesIO(png))
    assert im.mode == "RGBA" and im.size == (N, N)
    a = np.asarray(im)
    p = stack.get(Hazard.WILDFIRE, 4)
    np.testing.assert_array_equal(a[..., 3], np.clip(np.floor(255.0 * p + 0.5), 0, 255).astype(np.uint8))
    on = a[..., 3] > 0
    assert on.any()
    assert np.all(a[on][:, :3] == np.array(HAZARD_COLORS[Hazard.WILDFIRE], dtype=np.uint8))
    np.testing.assert_allclose(decode_overlay(png), a[..., 3] / 255.0, atol=1e-7)
    assert len(set(HAZARD_COLORS.values())) == 6 and set(HAZARD_COLORS) == set(Hazard)


@pytest.mark.req("RISK-06")
def test_overlay_corners(stack):
    from pyproj import Transformer

    inv = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    corners = stack.overlay_corners()
    exp = [
        inv.transform(x, y)
        for x, y in [
            (-150_000, 1_300_000),
            (-118_000, 1_300_000),
            (-118_000, 1_268_000),
            (-150_000, 1_268_000),
        ]
    ]
    assert len(corners) == 4
    np.testing.assert_allclose(np.array(corners, dtype=float), np.array(exp), atol=1e-7)


# ---------------------------------------------------------------- RISK-07
@pytest.mark.req("RISK-07")
def test_overlay_equals_sampled_values(stack):
    X, Y = np.meshgrid(XC, YC)
    for h in Hazard:
        for s in range(12):
            img = (
                np.asarray(Image.open(io.BytesIO(stack.overlay_png(h, s))))[..., 3].astype(np.float32) / 255.0
            )
            sampled = stack.sample(h, s, X, Y)
            np.testing.assert_array_equal(img, np.floor(255.0 * sampled + 0.5).astype(np.float32) / 255.0)
            assert float(np.abs(img - sampled).max()) <= 0.5 / 255 + 1e-7


@pytest.mark.req("RISK-07")
def test_overlay_decode_tracks_stack_data_after_layers(t0, rng):
    arr = rng.random((12, N, N)).astype(np.float32)
    st = RiskStack.from_layers([_layer(Hazard.SMOKE, arr, t0)], t0, WIN, policies=HOLD)
    X, Y = np.meshgrid(XC, YC)
    dec = decode_overlay(st.overlay_png("smoke", 9))
    assert float(np.abs(dec - st.sample("smoke", 9, X, Y)).max()) <= 0.5 / 255 + 1e-7


# ---------------------------------------------------------------- RISK-08
class FakeAdapter:
    def __init__(self, hazard, observed_none=False):
        self.hazard = hazard
        self.policy = _policy()
        self.observed_none = observed_none
        self.calls = []

    def _l(self, scn, kind, values):
        arr = np.broadcast_to(np.asarray(values, dtype=np.float32)[:, None, None], (12, N, N))
        return _layer(self.hazard, arr, scn.t0, kind=kind, source=f"fake.{kind}")

    def predict(self, scn):
        self.calls.append("predict")
        return self._l(scn, "forecast", 0.1 + np.arange(12) / 20)

    def baseline(self, scn):
        self.calls.append("baseline")
        return self._l(scn, "baseline", np.full(12, 0.3))

    def observed(self, scn):
        self.calls.append("observed")
        return None if self.observed_none else self._l(scn, "observed", np.full(12, 0.6))


@pytest.mark.req("RISK-08")
def test_build_stack_sources(t0):
    scn = _scenario(t0)
    ads = [FakeAdapter(Hazard.STORM), FakeAdapter(Hazard.FLOOD, observed_none=True)]
    assert isinstance(ads[0], HazardAdapter)
    assert RISK_SOURCES == ("model", "baseline", "observed", "none", "current-only")

    def vals(st, h):
        return st.data.sel(hazard=h).values.astype(np.float32)[:, 0, 0]

    m = build_stack(ads, scn, "model", policies=HOLD)
    np.testing.assert_allclose(vals(m, "storm"), 0.1 + np.arange(12) / 20, atol=1e-3)
    assert m.meta[Hazard.STORM].kind == "forecast"
    b = build_stack(ads, scn, "baseline", policies=HOLD)
    np.testing.assert_allclose(vals(b, "flood"), 0.3, atol=1e-3)
    o = build_stack(ads, scn, "observed", policies=HOLD)
    np.testing.assert_allclose(vals(o, "storm"), 0.6, atol=1e-3)
    assert float(o.data.sel(hazard="flood").max()) == 0.0 and Hazard.FLOOD not in o.meta
    c = build_stack(ads, scn, "current-only", policies=HOLD)
    np.testing.assert_allclose(vals(c, "storm"), 0.1, atol=1e-3)
    assert c.meta[Hazard.STORM].source.endswith("+current-only")
    for a in ads:
        a.calls.clear()
    n = build_stack(ads, scn, "none", policies=HOLD)
    assert float(n.data.max()) == 0.0 and all(not a.calls for a in ads)
    assert n.t0 == t0 and n.window == WIN
    with pytest.raises(ValueError):
        build_stack(ads, scn, "oracle", policies=HOLD)
