"""Storm provider: SEVIR frames for a scenario, mode selection and source labels (STORM-08). Synthetic data only."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

T0 = datetime(2019, 5, 26, 3, 0, tzinfo=UTC)
BBOX = (-98.0, 35.2, -97.8, 35.4)
S = 16  # pooled pixels per side


def make_world(tmp_path: Path, with_ir: bool = True, vil: int = 100, bad_input: bool = False):
    """A one-event fake SEVIR (raw CATALOG + processed memmap) covering the OKC area, plus a models dir."""
    from outrun.data.sevir import CATALOG_REQUIRED

    raw, proc = tmp_path / "raw", tmp_path / "proc"
    (raw / "sevir").mkdir(parents=True)
    (proc / "sevir").mkdir(parents=True)
    offsets = ":".join(str(o) for o in range(-118, 123, 5))  # 49 frames, frame 18 is 03:00 for time_utc 03:28
    row = dict.fromkeys(CATALOG_REQUIRED, "")
    row.update(
        id="S900001", file_name="vil/2019/x.h5", file_index=0, img_type="vil", time_utc="2019-05-26 03:28:00",
        minute_offsets=offsets, proj="laea", llcrnrlat=33.75, llcrnrlon=-100.03, urcrnrlat=37.2, urcrnrlon=-95.79,
    )  # fmt: skip
    pd.DataFrame([row]).to_csv(raw / "sevir" / "CATALOG.csv", index=False)
    if not with_ir:  # raw VIL file only (2x the pooled size, frames last), as for El Reno
        import h5py

        (raw / "sevir_vil_2018_2019" / "2019").mkdir(parents=True)
        with h5py.File(raw / "sevir_vil_2018_2019" / "2019" / "x.h5", "w") as f:
            f["id"] = np.array([b"S900001"])
            f["vil"] = np.full((1, 2 * S, 2 * S, 49), vil, np.uint8)
    if with_ir:
        events = np.zeros((1, 49, 2, S, S), np.uint8)
        events[:, :, 0] = vil
        events[0, 5:8, 0] = 255 if bad_input else vil
        np.save(proc / "sevir" / "sevir.npy", events)
        pd.DataFrame({"row": [0], "id": ["S900001"]}).to_parquet(proc / "sevir" / "sevir.meta.parquet")
    return raw, proc, tmp_path / "models"


def make_scenario(**case):
    from outrun.core.risk import Hazard
    from outrun.core.scenario import Scenario

    return Scenario(
        id="t", t0=T0, bbox=BBOX, city="x", hazards=(Hazard.STORM,), held_out={Hazard.STORM: True},
        case=case or {"sevir_id": "S900001"},
    )  # fmt: skip


def save_checkpoint(models: Path, folder: str, in_channels: int) -> None:
    from outrun.hazards.storm.model import StormUNet

    (models / folder).mkdir(parents=True)
    net = StormUNet(in_channels=in_channels, base=4, levels=1)
    torch.save({"model": net.state_dict()}, models / folder / "best.pt")


@pytest.mark.req("STORM-08")
def test_provider_frames_and_regrid(tmp_path):
    from outrun.hazards.storm.provider import real_provider

    raw, proc, _ = make_world(tmp_path)
    scn = make_scenario()
    frames = real_provider(raw=raw, proc=proc)(scn)
    assert frames.vil.shape == (13, S, S) and frames.ir is not None and frames.future_vil.shape == (12, S, S)
    window = scn.grid_window()
    out = frames.regrid(frames.vil[:2])
    assert out.shape == (2, *window.shape) and np.all(out == 100)  # whole bbox lies inside the event extent
    assert frames.held_out and frames.x.shape == (window.nx,) and frames.y.shape == (window.ny,)


@pytest.mark.req("STORM-08")
def test_provider_finds_event_by_time_and_bbox(tmp_path):
    from outrun.hazards.storm.provider import real_provider

    raw, proc, _ = make_world(tmp_path)
    ref = real_provider(raw=raw, proc=proc).resolve(make_scenario(note="no sevir_id"))
    assert ref.event_id == "S900001" and ref.t0_index == 18 and ref.has_ir


@pytest.mark.req("STORM-08")
def test_provider_rejects_no_data_frames_and_bad_time(tmp_path):
    from outrun.core.risk import Hazard
    from outrun.core.scenario import Scenario
    from outrun.hazards.storm.provider import real_provider

    raw, proc, _ = make_world(tmp_path, bad_input=True)
    prov = real_provider(raw=raw, proc=proc)
    with pytest.raises(LookupError, match="no-data"):
        prov(make_scenario())
    late = Scenario(id="t", t0=datetime(2019, 5, 27, tzinfo=UTC), bbox=BBOX, city="x", hazards=(Hazard.STORM,),
                    case={"sevir_id": "S900001"})  # fmt: skip
    with pytest.raises(LookupError):
        prov.resolve(late)


@pytest.mark.req("STORM-08")
def test_dl_mode_needs_ir(tmp_path):
    from outrun.hazards.storm.provider import real_provider

    raw, proc, _ = make_world(tmp_path, with_ir=False)  # event only in the CATALOG: VIL-only like El Reno
    with pytest.raises(ValueError):
        real_provider("nonsense")
    with pytest.raises(LookupError, match="IR107"):
        real_provider("dl", raw=raw, proc=proc)(make_scenario())


@pytest.mark.req("STORM-08")
def test_auto_mode_selection(tmp_path):
    from outrun.hazards.storm.provider import SevirRef, choose_mode

    models = tmp_path / "models"
    ir, vil_only = SevirRef("S1", 18, {}, 0, False), SevirRef("S2", 18, {}, None, False)
    assert choose_mode("auto", ir, models) == "baseline"  # no checkpoints
    save_checkpoint(models, "storm_vil_only", 13)
    assert choose_mode("auto", ir, models) == "dl_vil_only"
    assert choose_mode("auto", vil_only, models) == "dl_vil_only"
    save_checkpoint(models, "storm", 26)
    assert choose_mode("auto", ir, models) == "dl"  # IR present and the VIL+IR checkpoint exists
    assert choose_mode("auto", vil_only, models) == "dl_vil_only"  # El Reno case: no IR
    assert choose_mode("baseline", ir, models) == "baseline"
    with pytest.raises(ValueError):
        choose_mode("fast", ir, models)


@pytest.mark.req("STORM-08")
@pytest.mark.parametrize(
    ("mode", "with_ir", "prefix", "kind"),
    [
        ("baseline", True, "storm.", "baseline"),
        ("dl_vil_only", False, "storm.unet_vil_only@S900001", "forecast"),
        ("dl", True, "storm.unet_vil_ir@S900001", "forecast"),
        ("auto", False, "storm.unet_vil_only@S900001", "forecast"),
    ],
)
def test_adapter_per_mode_labels_layer(tmp_path, mode, with_ir, prefix, kind):
    from outrun.hazards.storm.provider import real_provider, storm_adapter

    raw, proc, models = make_world(tmp_path, with_ir=with_ir)
    save_checkpoint(models, "storm_vil_only", 13)
    if with_ir:
        save_checkpoint(models, "storm", 26)
    scn = make_scenario()
    adapter = storm_adapter(scn, mode, real_provider(mode, raw=raw, proc=proc), models_dir=models)
    layer = adapter.predict(scn)
    assert layer.hazard.value == "storm" and layer.kind == kind and layer.source.startswith(prefix)
    assert layer.prob.shape == (12, *scn.grid_window().shape) and layer.held_out
    assert float(layer.prob.min()) >= 0.0 and float(layer.prob.max()) <= 1.0


@pytest.mark.req("STORM-08")
def test_baseline_mode_extrapolates_and_observed_labelled(tmp_path):
    from outrun.hazards.storm.provider import real_provider, storm_adapter

    raw, proc, models = make_world(tmp_path, vil=200)  # whole area above VIL 133
    scn = make_scenario()
    adapter = storm_adapter(scn, "baseline", real_provider(raw=raw, proc=proc), models_dir=models)
    base = adapter.baseline(scn)
    assert base.source.startswith("storm.") and "extrapolation" in base.source
    assert float(base.prob[0].min()) == 1.0 and float(base.prob[11].max()) <= 1.0  # in-horizon slot, decayed later
    obs = adapter.observed(scn)
    assert obs.kind == "observed" and obs.source.startswith("storm.")
