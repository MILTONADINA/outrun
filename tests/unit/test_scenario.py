"""specs/02-grid-risk.md: RISK-10 (Scenario)."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta, timezone

import pytest

from outrun.core.grid import GridWindow, bbox_to_window
from outrun.core.risk import Hazard
from outrun.core.scenario import Scenario

T0 = datetime(2019, 5, 26, 3, 0, tzinfo=UTC)
BBOX = (-98.10, 35.10, -97.10, 35.80)


def _scn(**kw):
    args = dict(
        id="el-reno-2019",
        t0=T0,
        bbox=BBOX,
        city="Oklahoma City",
        hazards=(Hazard.STORM, Hazard.TORNADO),
        held_out={Hazard.TORNADO: True},
        title="El Reno EF3",
        case={"date": "2019-05-25", "sources": ["SEVIR", "TorNet"]},
    )
    args.update(kw)
    return Scenario(**args)


@pytest.mark.req("RISK-10")
def test_scenario_fields_and_defaults():
    s = Scenario(id="s1", t0=T0, bbox=BBOX, city="X", hazards=(Hazard.FLOOD,))
    assert s.held_out == {} and s.title == "" and s.case == {} and s.synthetic is False and s.window is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.city = "Y"  # type: ignore[misc]


@pytest.mark.req("RISK-10")
@pytest.mark.parametrize(
    "kw",
    [
        {"id": "El Reno"},
        {"id": "-x"},
        {"id": ""},
        {"t0": T0.replace(tzinfo=None)},
        {"t0": T0.astimezone(timezone(timedelta(hours=-5)))},
        {"bbox": (-97.0, 35.0, -98.0, 36.0)},
        {"bbox": (-98.0, 36.0, -97.0, 35.0)},
        {"bbox": (-98.0, 35.0, -97.0)},
        {"hazards": (Hazard.STORM, Hazard.STORM)},
        {"hazards": ("storm",)},
        {"held_out": {Hazard.SMOKE: True}},
    ],
)
def test_scenario_validation(kw):
    with pytest.raises(ValueError):
        _scn(**kw)


@pytest.mark.req("RISK-10")
def test_grid_window_and_held_out():
    s = _scn()
    assert s.grid_window() == bbox_to_window(BBOX)
    w = GridWindow(x0=-150_000.0, y0=1_300_000.0, nx=32, ny=32)
    assert _scn(window=w).grid_window() == w
    assert s.is_held_out(Hazard.TORNADO) is True
    assert s.is_held_out(Hazard.STORM) is False


@pytest.mark.req("RISK-10")
@pytest.mark.parametrize("window", [None, GridWindow(x0=-150_000.0, y0=1_300_000.0, nx=32, ny=32)])
def test_to_from_dict_and_json_round_trip(window, tmp_path):
    s = _scn(window=window, synthetic=window is not None)
    d = s.to_dict()
    text = json.dumps(d)  # JSON-serialisable
    assert d["hazards"] == ["storm", "tornado"]
    assert datetime.fromisoformat(d["t0"]) == T0 and datetime.fromisoformat(d["t0"]).tzinfo is not None
    assert Scenario.from_dict(json.loads(text)) == s
    p = tmp_path / "scenario.json"
    s.save(p)
    assert Scenario.load(p) == s
