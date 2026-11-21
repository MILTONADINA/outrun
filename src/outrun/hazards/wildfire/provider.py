"""WSTS provider for the wildfire adapter: one fire-day GeoTIFF (+ the next day) -> ``WildfireInputs`` (FIRE-10).

The 375 m WSTS bands are mapped to the 12 NDWS channels at 1 km (``outrun.data.wsts.map_to_ndws``), normalised with
the NDWS-train statistics saved next to the checkpoint, and regridded (nearest cell) onto the scenario's GRID window.
Population is not in WSTS, so that channel is NaN (-> 0 after normalisation, the NDWS-train mean).

Replay (T20): ``hms_replay_layers`` turns NOAA HMS fire-detection points (observed satellite detections, no model) into
a persistence layer and an observed layer for the labelled OK March 2025 scenario.

Demo fire (T13): ``DEMO`` below, picked from the 2021 catalog (``wsts_shift catalog``) by size alone.
"""

from __future__ import annotations

import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

from outrun.config import get_paths
from outrun.core.grid import lonlat_to_xy, regrid_nearest
from outrun.core.risk import RiskLayer
from outrun.core.scenario import Scenario
from outrun.data.wsts import map_to_ndws, read_wsts_day

from .adapter import N_SLOTS, WildfireInputs, to_risk_layer
from .dataset import NDWS_CHANNELS, NormStats, check_channel_order, normalize
from .train import load_stats

FIRE_CH = NDWS_CHANNELS.index("PrevFireMask")


@dataclass(frozen=True)
class DemoFire:
    """The headline WSTS 2021 fire: largest 2021 fire by VIIRS pixels; no Southern Plains fire is large (max 354 px)."""

    fire: str = (
        "fire_25294714"  # Dixie Fire, Plumas County CA (centre -121.06, 40.21); 72 days, 72 k fire pixels
    )
    day0: date = date(
        2021, 8, 5
    )  # day with the most active-fire pixels (4550); Aug 4 has a VIIRS gap (0 pixels)
    town: str = (
        "Greenville, CA"  # nearby populated place whose OSM roads T20 routes on (centre -120.95, 40.14)
    )


DEMO = DemoFire()


def _read(zip_path: Path, fire: str, day: date, tmp: Path) -> tuple[np.ndarray, dict]:
    """Mapped 1 km 12-channel array and the geometry (lon/lat of each mapped cell centre) of one WSTS day."""
    import rasterio
    from pyproj import Transformer

    name = f"{day.year}/{fire}/{day.isoformat()}.tif"
    with zipfile.ZipFile(zip_path) as z:
        path = Path(z.extract(name, tmp))
    bands, res = read_wsts_day(path)
    x = map_to_ndws(bands, res)
    with rasterio.open(path) as src:
        t, crs = src.transform, src.crs
    path.unlink()
    rows, cols = np.mgrid[0 : x.shape[1], 0 : x.shape[2]]
    xs = t.c + (cols + 0.5) * 1000.0  # mapped cell k covers [k, k + 1) km from the image's top-left corner
    ys = t.f - (rows + 0.5) * 1000.0
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xs, ys)
    return x, {"lon": lon, "lat": lat}


def wsts_provider(
    zip_path: Path | None = None,
    fire: str = DEMO.fire,
    day0: date = DEMO.day0,
    stats: NormStats | None = None,
    max_dist_m: float = 1500.0,
) -> Callable[[Scenario], WildfireInputs]:
    """Provider for ``WildfireAdapter``: day-0 inputs of ``fire`` on ``day0``, next-day mask as truth (held out)."""
    zip_path = zip_path or get_paths().raw / "wsts" / "WildfireSpreadTS.zip"
    stats = stats or load_stats(get_paths().models / "wildfire" / "norm_stats.npz")
    check_channel_order(NDWS_CHANNELS)
    with tempfile.TemporaryDirectory() as tmp:
        x_raw, geo = _read(zip_path, fire, day0, Path(tmp))
        nxt, _ = _read(zip_path, fire, day0 + timedelta(days=1), Path(tmp))
    day0_time = datetime.combine(day0 + timedelta(days=1), datetime.min.time(), tzinfo=UTC)  # end of day 0
    day0_mask, day1_mask = x_raw[FIRE_CH].copy(), nxt[FIRE_CH].copy()

    def provide(scn: Scenario) -> WildfireInputs:
        win = scn.grid_window()

        def regrid(a: np.ndarray) -> np.ndarray:
            return np.stack(
                [
                    regrid_nearest(
                        np.asarray(s, np.float32), geo["lon"], geo["lat"], win, max_dist_m, fill=0.0
                    )
                    for s in a
                ]
            )

        return WildfireInputs(
            x_model=normalize(x_raw, stats),
            x_raw=x_raw,
            day0_mask=day0_mask,
            day0_time=day0_time,
            day1_mask=day1_mask,
            x=win.x_centers(),
            y=win.y_centers(),
            regrid=regrid,
            held_out=True,  # WSTS 2021 is the out-of-time test set
        )

    return provide


def real_provider() -> Callable[[Scenario], WildfireInputs]:
    """The default provider (``WildfireAdapter(provider=None)``): the demo fire on the raw WSTS archive."""
    return wsts_provider()


# ---------------------------------------------------------------------------------------------------------------
# OK March 2025 replay: HMS fire points (satellite detections), no weather model behind it.
REPLAY_SOURCE = "wildfire.hms_persistence_replay"


def read_hms_fire_points(days: list[date], raw: Path | None = None):
    """Rows (lon, lat, utc time) of the HMS fire-point shapefiles of ``days`` (``hms_fireYYYYMMDD.zip``).

    ``Time`` is the UTC HHMM of the detection (GOES FDC, VIIRS, ...); returns a pandas DataFrame with columns
    ``lon``, ``lat`` and ``t`` (tz-aware UTC).
    """
    import geopandas as gpd
    import pandas as pd

    raw = raw or get_paths().raw / "hms_fire_points"
    frames = []
    for d in days:
        path = next(raw.rglob(f"hms_fire{d:%Y%m%d}.zip"), None)
        if path is None:
            raise FileNotFoundError(
                f"no hms_fire{d:%Y%m%d}.zip under {raw}; run `outrun fetch --source hms_fire_points`"
            )
        g = gpd.read_file(f"zip://{path}")
        hhmm = g["Time"].astype(str).str.zfill(4)
        t = pd.to_datetime(f"{d:%Y-%m-%d} " + hhmm.str[:2] + ":" + hhmm.str[2:], utc=True)
        frames.append(pd.DataFrame({"lon": g["Lon"].astype(float), "lat": g["Lat"].astype(float), "t": t}))
    return pd.concat(frames, ignore_index=True)


def _rasterise(points, scn: Scenario, t_from: datetime, t_to: datetime) -> np.ndarray:
    """(ny, nx) float32 1.0 in cells with a detection in ``(t_from, t_to]``."""
    win = scn.grid_window()
    out = np.zeros(win.shape, np.float32)
    sel = points[(points["t"] > t_from) & (points["t"] <= t_to)]
    if len(sel):
        x, y = lonlat_to_xy(sel["lon"].to_numpy(), sel["lat"].to_numpy())
        row, col = win.cell_of(x, y)
        ok = row >= 0
        out[row[ok], col[ok]] = 1.0
    return out


def hms_replay_layers(
    scn: Scenario, points, lookback_h: float = 3.0, ring_p: float = 0.4, slot_minutes: int = 15
) -> dict[str, RiskLayer]:
    """Persistence + observed RiskLayers from HMS detections, causal at ``scn.t0``.

    ``model``: detections of the last ``lookback_h`` hours before t0 (nothing after t0), P = 1 in a detected cell and
    ``ring_p`` in its 8 neighbours (the 1 km cell is smaller than a GOES pixel, so the neighbours are uncertain),
    held constant over the 12 slots (kind "baseline": this is persistence, not a forecast model).
    ``observed``: slot s holds every detection in ``(t0, t0 + (s + 1) * slot_minutes]``, P = 1 (what really happened).
    """
    t0 = scn.t0
    past = _rasterise(points, scn, t0 - timedelta(hours=lookback_h), t0)
    ring = np.zeros_like(past)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            ring = np.maximum(ring, np.roll(np.roll(past, dr, axis=0), dc, axis=1))
    persist = np.where(past > 0, 1.0, ring_p * ring).astype(np.float32)
    seen = np.stack(
        [_rasterise(points, scn, t0, t0 + timedelta(minutes=slot_minutes * (s + 1))) for s in range(N_SLOTS)]
    )
    win = scn.grid_window()
    kw = {"issued": t0, "held_out": True, "x": win.x_centers(), "y": win.y_centers()}
    return {
        "model": to_risk_layer(
            np.repeat(persist[None], N_SLOTS, 0),
            kind="baseline",
            source=f"{REPLAY_SOURCE}@{t0:%Y-%m-%d}",
            **kw,
        ),  # fmt: skip
        "observed": to_risk_layer(seen, kind="observed", source="wildfire.hms_fire_points", **kw),
    }
