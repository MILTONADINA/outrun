"""T20: HMS fire-point replay layers (causal persistence + observed) on a synthetic scenario; no data, no network."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from outrun.core.grid import bbox_to_window
from outrun.core.risk import Hazard
from outrun.core.scenario import Scenario
from outrun.hazards.wildfire.provider import hms_replay_layers

T0 = datetime(2025, 3, 14, 20, 0, tzinfo=UTC)
BBOX = (-97.6, 35.4, -97.4, 35.6)


def _scn() -> Scenario:
    return Scenario(
        id="replay-test", t0=T0, bbox=BBOX, city="x", hazards=(Hazard.WILDFIRE,),
        window=bbox_to_window(BBOX),
    )  # fmt: skip


def test_replay_is_causal_and_observed_accumulates():
    pts = pd.DataFrame(
        {
            "lon": [-97.5, -97.5, -97.45],
            "lat": [35.5, 35.5, 35.45],
            "t": [T0 - timedelta(hours=1), T0 + timedelta(minutes=20), T0 - timedelta(hours=5)],
        }
    )
    out = hms_replay_layers(_scn(), pts)
    persist, seen = out["model"], out["observed"]
    assert persist.kind == "baseline" and "persistence_replay" in persist.source
    assert seen.kind == "observed"
    p = persist.prob.values.astype(np.float32)
    assert p.shape[0] == 12 and np.all(p == p[0])  # held constant
    assert (
        p[0] == 1.0
    ).sum() == 1  # only the detection of the last 3 h; the old one and the future one are out
    assert p[0][(p[0] > 0) & (p[0] < 1)].min() > 0.39 and p[0].max() == 1.0
    o = seen.prob.values.astype(np.float32)
    assert (
        o[0].sum() == 0 and o[1].sum() == 1 and o[11].sum() == 1
    )  # detection at t0+20 min shows from slot 1
