"""Real tornado provider for a scenario (TORN-10): TorNet chips near the scenario plus storm motion.

Detections are processed TorNet chips (radar volumes), placed on GRID from the radar site plus the chip's centre
azimuth/range (never from the SPC track). Storm motion comes from the storm provider (STORM-09): pySTEPS/LK motion of
the SEVIR VIL frames, averaged over the active cells and converted to m/s. If that is unavailable, ``resolve_motion``
(TORN-09) falls back to HRRR wind and then the 15 m/s default.

TorNet chips run 15 min, and the one El Reno volume that exists ends after the storm ``t0``; chips whose end is
within ``LAG_MIN`` of ``t0`` are used (a demo-only timing slack, recorded in the bundle sources).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

from outrun.core.grid import lonlat_to_xy
from outrun.core.risk import Hazard
from outrun.core.scenario import Scenario

from .adapter import TornadoDetection, TornadoInputs
from .dataset import tornet_root
from .features import chip_from_stored

# NEXRAD site (lat, lon); only the Oklahoma-area radars the demo scenarios can reach.
SITES = {
    "KTLX": (35.3331, -97.2778),
    "KVNX": (36.7408, -98.1278),
    "KFDR": (34.3622, -98.9767),
    "KINX": (36.1750, -95.5644),
    "KSRX": (35.2904, -94.3619),
}
LAG_MIN = 35.0
ACTIVE_VIL = 100  # uint8 VIL counts that mark an active cell for the motion average
_GEOD = Geod(ellps="WGS84")


def storm_motion_fn(scn: Scenario) -> Callable[[float, float], tuple[float, float] | None] | None:
    """Mean ``(u, v)`` m/s of the active cells in the scenario's SEVIR VIL frames, or None if no frames/cells."""
    from outrun.hazards.storm import baselines
    from outrun.hazards.storm.provider import StormProvider

    try:
        frames = StormProvider("baseline")(scn)
        motion = baselines.motion_to_mps(baselines.estimate_motion(frames.vil[-6:].astype(np.float32)))
        cells = frames.vil[-1] >= ACTIVE_VIL
        if not cells.any():
            return None
        uv = (float(motion[0][cells].mean()), float(motion[1][cells].mean()))
    except (LookupError, OSError, ValueError):
        return None
    if not np.isfinite(uv).all():
        return None
    return lambda x, y: uv


def real_provider(root: Path | None = None, split: str = "case_holdout", use_storm_motion: bool = True):
    """``provider(scn) -> TornadoInputs`` over the processed TorNet ``split`` (default: the case chips)."""
    root = Path(root) if root is not None else tornet_root()

    def provider(scn: Scenario) -> TornadoInputs:
        meta = pd.read_csv(root / f"{split}_meta.csv")
        end = pd.to_datetime(meta["end_time"]).dt.tz_localize("UTC")
        start = pd.to_datetime(meta["start_time"]).dt.tz_localize("UTC")
        lag = pd.Timedelta(minutes=LAG_MIN)
        pick = np.flatnonzero(
            ((start >= scn.t0 - pd.Timedelta(minutes=15)) & (end <= scn.t0 + lag)).to_numpy()
            & meta["site_name"].isin(SITES).to_numpy()
        )
        window = scn.grid_window()
        x, y = window.x_centers(), window.y_centers()
        rng, az = np.load(root / f"{split}_range_km.npy"), np.load(root / f"{split}_azimuth.npy")
        xs = np.load(root / f"{split}_x.npy", mmap_mode="r")
        dets = []
        for i in pick:
            lat0, lon0 = SITES[meta["site_name"].iloc[i]]
            r_c, a_c = float(rng[i].mean()), float(az[i].mean())
            lon, lat, _ = _GEOD.fwd(lon0, lat0, a_c, r_c * 1000.0)
            cx, cy = (float(v) for v in lonlat_to_xy(lon, lat))
            if not (x.min() <= cx <= x.max() and y.min() <= cy <= y.max()):
                continue
            dets.append(TornadoDetection(chip_from_stored(np.asarray(xs[i])), rng[i], az[i], r_c, cx, cy))
        return TornadoInputs(
            detections=dets, x=x, y=y,
            storm_motion=storm_motion_fn(scn) if use_storm_motion else None,
            hrrr_wind=None, observed=None, held_out=scn.is_held_out(Hazard.TORNADO),
        )  # fmt: skip

    return provider
