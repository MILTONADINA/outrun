"""Real ice provider (ICE-08): HRRR f01 store + NWS zones + labelled bags -> ``IceInputs`` for one scenario.

Needs ``PROC/hrrr/ice`` (``data.zarr``, ``bags/zones.npz``, ``bags/elevation.npy``, ``bags/bags.parquet``). ``scn.t0``
must be an HRRR cycle time (3-hourly UTC). One 64x64 crop on the 3 km HRRR grid is centred on the scenario bbox; its
zone-mask channel is the union of the NWS zones that touch the bbox (the bag the model was trained on). The step at
``t0`` uses cycle t0 as time t and t0 - 3 h as t-3 h, like ``build.make_windows``. ``truth`` is the labelled zones
of the 6-h window containing t0 (display only, never scored here).
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd

from outrun.config import get_paths
from outrun.core.grid import regrid_nearest
from outrun.hazards.ice.adapter import IceInputs
from outrun.hazards.ice.build import ARRAY
from outrun.hazards.ice.dataset import ICE_CROP, ICE_FIELDS, build_bag_input

TEST_FROM = pd.Timestamp("2024-07-01", tz="UTC")  # ice test season = winter 2024-25 (splits/ice.json)


def real_provider():
    import zarr

    d = get_paths().proc / "hrrr" / "ice"
    root = zarr.open_group(d / "data.zarr", mode="r")
    lat, lon = np.asarray(root["latitude"][:]), np.asarray(root["longitude"][:])
    times = np.asarray(root["time"][:])
    z = np.load(d / "bags" / "zones.npz")
    elev = np.load(d / "bags" / "elevation.npy").astype(np.float32)
    zone_ids, cells, offs = z["zone_ids"], z["cells"], z["offsets"]
    arrays = [root[ARRAY[f]] for f in ICE_FIELDS]
    cache: dict = {}

    def frame(ts: pd.Timestamp) -> np.ndarray:
        hit = np.flatnonzero(times == int(ts.timestamp()))
        if not len(hit):
            raise LookupError(f"no HRRR f01 cycle at {ts.isoformat()}")
        return np.stack([np.asarray(a[int(hit[0])], dtype=np.float32) for a in arrays])

    def provider(scn) -> IceInputs:
        t = pd.Timestamp(scn.t0.astimezone(UTC))
        lon0, lat0, lon1, lat1 = scn.bbox
        inbox = (lon >= lon0) & (lon <= lon1) & (lat >= lat0) & (lat <= lat1)
        flat_in = np.flatnonzero(inbox.ravel())
        touching = [i for i in range(len(zone_ids)) if np.isin(cells[offs[i] : offs[i + 1]], flat_in).any()]
        mask = np.zeros(lat.size, bool)
        for i in touching:
            mask[cells[offs[i] : offs[i + 1]]] = True
        mask = mask.reshape(lat.shape)
        ft, ftm3 = frame(t), frame(t - pd.Timedelta(hours=3))
        x, m = build_bag_input(ft, ftm3, elev, mask)
        # crop origin = the one build_bag_input chose (centroid of the mask, clamped), recovered the same way
        rows, cols = np.nonzero(mask)
        cr, cc = round(float(rows.mean())), round(float(cols.mean()))
        r0 = min(max(cr - ICE_CROP // 2, 0), max(0, lat.shape[0] - ICE_CROP))
        c0 = min(max(cc - ICE_CROP // 2, 0), max(0, lat.shape[1] - ICE_CROP))
        sl = (slice(r0, r0 + ICE_CROP), slice(c0, c0 + ICE_CROP))
        clon, clat = lon[sl], lat[sl]
        win = scn.grid_window()

        def regrid(a: np.ndarray) -> np.ndarray:
            return np.stack([regrid_nearest(a[k], clon, clat, win, 4500.0, fill=0.0) for k in range(len(a))])

        fields = {f: ft[j][sl] for j, f in enumerate(ICE_FIELDS)}
        truth = None
        ws = t.floor("6h")
        bags = cache.get("bags")
        if bags is None:
            bags = cache["bags"] = pd.read_parquet(
                d / "bags" / "bags.parquet", columns=["zone_id", "window_start", "label"]
            )
        lab = bags[(bags["window_start"] == ws) & bags["zone_id"].isin(zone_ids[touching])]
        if len(lab):
            pos = np.zeros(lat.size, bool)
            for zid in lab.loc[lab["label"] == 1, "zone_id"]:
                i = int(np.flatnonzero(zone_ids == zid)[0])
                pos[cells[offs[i] : offs[i + 1]]] = True
            truth = np.repeat(regrid(pos.reshape(lat.shape)[sl][None].astype(np.float32)), 12, axis=0)
        return IceInputs(
            x_steps=x[None], fields_steps=[fields], truth=truth, x=win.x_centers(), y=win.y_centers(),
            regrid=regrid, held_out=bool(t >= TEST_FROM),
        )  # fmt: skip

    return provider
