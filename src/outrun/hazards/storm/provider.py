"""Real storm data provider: SEVIR frames for one scenario, and the adapter for each storm mode (STORM-08).

``real_provider()`` is what ``StormAdapter`` imports lazily. Given a scenario it finds the SEVIR event (``scn.case``
key ``sevir_id``, else the CATALOG VIL event whose frames cover ``scn.t0`` and whose extent contains the scenario
bbox), cuts the 13 input and 12 future frames around ``t0`` and returns ``StormFrames`` with a ``regrid`` that
nearest-neighbour samples the 2 km SEVIR pixels onto the scenario's EPSG:5070 1 km window.

Data sources, in order: the processed VIL+IR memmap (``PROC/sevir``) when the event is in it, else the raw SEVIR VIL
file (VIL only; this is the El Reno 2019-05-26 case, which has no IR107 in SEVIR).

Modes (``storm_adapter``) and the layer source label each one gives (always starts with ``storm.``):

* ``dl``           VIL+IR U-Net ``models/storm/best.pt``            -> ``storm.unet_vil_ir@<event>``
* ``dl_vil_only``  VIL-only U-Net ``models/storm_vil_only/best.pt`` -> ``storm.unet_vil_only@<event>``
* ``baseline``     pySTEPS-style Lucas-Kanade extrapolation        -> ``storm.pysteps_extrapolation@<event>``
* ``auto``         ``dl`` if the event has IR and its checkpoint exists, else ``dl_vil_only`` if that checkpoint
                   exists, else ``baseline``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from outrun.config import get_paths
from outrun.core.grid import regrid_nearest
from outrun.core.risk import RiskLayer
from outrun.core.scenario import Scenario

from . import baselines
from .adapter import StormAdapter, StormFrames
from .calibration import frames_to_slots
from .dataset import BAD_FRAME_FRACTION, N_IN, N_OUT, NO_DATA_VIL
from .model import RISK_THRESHOLD

MODES = ("auto", "dl", "dl_vil_only", "baseline")
SEVIR_PROJ = "+proj=laea +lat_0=38 +lon_0=-98 +units=m +a=6370997.0 +ellps=sphere"
CHECKPOINTS = {"dl": ("storm", "storm.unet_vil_ir"), "dl_vil_only": ("storm_vil_only", "storm.unet_vil_only")}


@dataclass(frozen=True)
class SevirRef:
    """Where one scenario's frames come from."""

    event_id: str
    t0_index: int  # frame index of scn.t0 (the last input frame)
    row: dict  # CATALOG VIL row
    memmap_row: int | None  # row in PROC/sevir/sevir.npy (VIL+IR) or None -> raw VIL file only
    held_out: bool

    @property
    def has_ir(self) -> bool:
        return self.memmap_row is not None


def _pool(a: np.ndarray) -> np.ndarray:
    from outrun.data.sevir import pool_vil_2x2

    return pool_vil_2x2(a)


def _is_bad(frames: np.ndarray) -> bool:
    return bool(((frames == NO_DATA_VIL).mean(axis=(-2, -1)) > BAD_FRAME_FRACTION).any())


class StormProvider:
    """Callable ``provider(scn) -> StormFrames``; also ``resolve(scn) -> SevirRef`` (no pixel data is read)."""

    def __init__(self, mode: str = "auto", raw: Path | None = None, proc: Path | None = None) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        paths = get_paths()
        self.mode = mode
        self.raw = Path(raw) if raw is not None else Path(paths.raw)
        self.proc = Path(proc) if proc is not None else Path(paths.proc)
        self._splits = paths.splits / "sevir.json"
        self._catalog: pd.DataFrame | None = None
        self._memmap_ids: dict[str, int] | None = None
        self._events: np.ndarray | None = None

    # ------------------------------------------------------------ lookup
    def _vil_catalog(self) -> pd.DataFrame:
        if self._catalog is None:
            from outrun.data.sevir import read_catalog

            cat = read_catalog(self.raw / "sevir" / "CATALOG.csv")
            vil = cat[(cat["img_type"] == "vil") & cat["id"].astype(str).str.startswith("S")]
            self._catalog = vil.drop_duplicates("id").set_index("id", drop=False)
        return self._catalog

    def _memmap_row(self, event_id: str) -> int | None:
        if self._memmap_ids is None:
            meta_path = self.proc / "sevir" / "sevir.meta.parquet"
            ids = pd.read_parquet(meta_path)["id"].astype(str).tolist() if meta_path.exists() else []
            self._memmap_ids = {eid: r for r, eid in enumerate(ids)}
        return self._memmap_ids.get(event_id)

    def _held_out(self, scn: Scenario, event_id: str) -> bool:
        from outrun.core.risk import Hazard

        if Hazard.STORM in scn.held_out:
            return scn.is_held_out(Hazard.STORM)
        if not self._splits.exists():
            return False
        d = json.loads(self._splits.read_text())
        held = {i["id"] for i in d.get("test", [])} | {i["id"] for v in d.get("extra", {}).values() for i in v}
        return event_id in held

    def resolve(self, scn: Scenario) -> SevirRef:
        cat = self._vil_catalog()
        event_id = scn.case.get("sevir_id")
        if event_id is None:
            lon0, lat0, lon1, lat1 = scn.bbox
            inside = (
                (cat["llcrnrlon"] <= lon0) & (cat["urcrnrlon"] >= lon1)
                & (cat["llcrnrlat"] <= lat0) & (cat["urcrnrlat"] >= lat1)
            )  # fmt: skip
            near = (cat["time_utc"] - scn.t0).abs() <= pd.Timedelta(hours=2)
            hits = cat[inside & near]
            if hits.empty:
                raise LookupError(f"no SEVIR event covers bbox {scn.bbox} at {scn.t0:%Y-%m-%d %H:%M} UTC")
            event_id = str((hits["time_utc"] - scn.t0).abs().idxmin())
        if event_id not in cat.index:
            raise LookupError(f"SEVIR event {event_id!r} is not in the CATALOG")
        row = cat.loc[event_id].to_dict()
        offsets = np.array([int(v) for v in str(row["minute_offsets"]).split(":")])
        times = row["time_utc"] + pd.to_timedelta(offsets, unit="m")
        gap = np.abs((times - scn.t0).total_seconds())
        i = int(np.argmin(gap))
        if gap[i] > 300:
            raise LookupError(f"{event_id} frames {times[0]:%H:%M}-{times[-1]:%H:%M} UTC do not cover t0 {scn.t0}")
        if i < N_IN - 1:
            raise LookupError(f"{event_id}: t0 is frame {i}, need {N_IN - 1} earlier frames")
        return SevirRef(event_id, i, row, self._memmap_row(event_id), self._held_out(scn, event_id))

    # ------------------------------------------------------------ pixels
    def _read(self, ref: SevirRef) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Pooled uint8 ``vil (13, 192, 192)``, ``ir`` or None, ``future_vil (12, ...)`` or None."""
        lo, hi = ref.t0_index - (N_IN - 1), ref.t0_index + 1
        end = min(hi + N_OUT, 49)
        if ref.has_ir:
            if self._events is None:
                self._events = np.load(self.proc / "sevir" / "sevir.npy", mmap_mode="r")
            ev = self._events[ref.memmap_row, lo:end]
            vil_all, ir = np.asarray(ev[:, 0]), np.asarray(ev[:N_IN, 1])
        else:
            import h5py

            name = Path(str(ref.row["file_name"])).name
            year = Path(str(ref.row["file_name"])).parent.name
            path = self.raw / "sevir_vil_2018_2019" / year / name
            with h5py.File(path, "r") as f:
                idx = int(ref.row["file_index"])
                got = f["id"][idx]
                if (got.decode() if isinstance(got, bytes) else str(got)) != ref.event_id:
                    raise ValueError(f"{path}: row {idx} is not {ref.event_id}")
                vil_all = _pool(np.moveaxis(f["vil"][idx, :, :, lo:end], -1, 0))
            ir = None
        vil, future = vil_all[:N_IN], vil_all[N_IN:]
        if _is_bad(vil):
            raise LookupError(f"{ref.event_id}: no-data (255) input frames around t0, window is not usable")
        if len(future) < N_OUT or _is_bad(future):
            future = None
        return vil, ir, future

    def _regrid(self, ref: SevirRef, scn: Scenario, size: int):
        """``(k, h, w) -> (k, Y, X)``: nearest 2 km SEVIR pixel for every 1 km window cell (0 outside the event)."""
        from pyproj import Transformer

        r = ref.row
        to_laea = Transformer.from_crs("EPSG:4326", SEVIR_PROJ, always_xy=True)
        (x0, x1), (y0, y1) = (
            to_laea.transform([r["llcrnrlon"], r["urcrnrlon"]], [r["llcrnrlat"], r["urcrnrlat"]])
        )
        xs = x0 + (np.arange(size) + 0.5) * (x1 - x0) / size
        ys = y1 - (np.arange(size) + 0.5) * (y1 - y0) / size  # row 0 is the northern edge
        lon, lat = Transformer.from_crs(SEVIR_PROJ, "EPSG:4326", always_xy=True).transform(
            *np.meshgrid(xs, ys)
        )
        window = scn.grid_window()
        index = regrid_nearest(
            np.arange(size * size, dtype=np.float32).reshape(size, size), lon, lat, window,
            max_dist_m=1.5 * max((x1 - x0), (y1 - y0)) / size, fill=-1.0,
        ).astype(np.int64)  # fmt: skip
        inside = index >= 0

        def regrid(a: np.ndarray) -> np.ndarray:
            flat = np.asarray(a, dtype=np.float32).reshape(a.shape[0], -1)
            return np.where(inside[None], flat[:, np.where(inside, index, 0)], 0.0).astype(np.float32)

        return regrid

    def __call__(self, scn: Scenario) -> StormFrames:
        ref = self.resolve(scn)
        if self.mode == "dl" and not ref.has_ir:
            raise LookupError(f"mode 'dl' needs IR107 but {ref.event_id} has no VIL+IR memmap row")
        vil, ir, future = self._read(ref)
        window = scn.grid_window()
        return StormFrames(
            vil=vil, ir=ir, future_vil=future, x=window.x_centers(), y=window.y_centers(),
            regrid=self._regrid(ref, scn, vil.shape[-1]), held_out=ref.held_out,
        )  # fmt: skip


def real_provider(mode: str = "auto", **kwargs) -> StormProvider:
    """The provider ``StormAdapter`` builds lazily (STORM-08); ``mode`` only matters for validation here."""
    return StormProvider(mode, **kwargs)


class ExtrapolationStormAdapter(StormAdapter):
    """No-model storm layer: Lucas-Kanade motion + semi-Lagrangian extrapolation of the 13 VIL frames, then
    ``VIL >= 133`` (the pySTEPS-style baseline of STORM-06). ``baseline`` and ``predict`` both return it, kind
    "baseline", because there is no learned model in this mode."""

    def __init__(self, provider: StormProvider, *, source_prefix: str | None = None, **kwargs) -> None:
        super().__init__(provider=provider, **kwargs)
        self._prefix = source_prefix or (
            "storm.pysteps_extrapolation" if baselines.PYSTEPS_AVAILABLE else "storm.lk_extrapolation"
        )

    def baseline(self, scn: Scenario) -> RiskLayer:
        frames = self._frames(scn)
        mask = (baselines.extrapolation_nowcast(frames.vil, N_OUT) >= RISK_THRESHOLD).astype(np.float32)
        return self._layer(scn, "baseline", frames_to_slots(mask), frames, f"{self._prefix}@{self._event(scn)}")

    def predict(self, scn: Scenario) -> RiskLayer:
        return self.baseline(scn)

    def _event(self, scn: Scenario) -> str:
        return self._provider.resolve(scn).event_id


class _Labelled(StormAdapter):
    """``StormAdapter`` whose forecast source label carries the SEVIR event id."""

    def predict(self, scn: Scenario) -> RiskLayer:
        layer = super().predict(scn)
        event = self._provider.resolve(scn).event_id
        return RiskLayer(
            hazard=layer.hazard, prob=layer.prob, edge_overrides=layer.edge_overrides, kind=layer.kind,
            issued=layer.issued, valid_until=layer.valid_until, source=f"{self.source_tag}@{event}",
            held_out=layer.held_out,
        )  # fmt: skip


def choose_mode(mode: str, ref: SevirRef, models_dir: Path | None = None) -> str:
    """Resolve ``auto`` for one event (see module docstring); explicit modes are validated, not changed."""
    models = Path(models_dir) if models_dir is not None else get_paths().models
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode != "auto":
        return mode
    if ref.has_ir and (models / CHECKPOINTS["dl"][0] / "best.pt").exists():
        return "dl"
    if (models / CHECKPOINTS["dl_vil_only"][0] / "best.pt").exists():
        return "dl_vil_only"
    return "baseline"


def storm_adapter(
    scn: Scenario | None = None,
    mode: str = "auto",
    provider: StormProvider | None = None,
    models_dir: Path | None = None,
) -> StormAdapter:
    """Adapter for one storm mode. ``auto`` needs ``scn`` to see whether the event has IR (El Reno has none)."""
    provider = provider or real_provider(mode)
    models = Path(models_dir) if models_dir is not None else get_paths().models
    if mode == "auto":
        if scn is None:
            raise ValueError("mode 'auto' needs the scenario to pick a model")
        mode = choose_mode(mode, provider.resolve(scn), models)
    if mode == "baseline":
        return ExtrapolationStormAdapter(provider)
    folder, tag = CHECKPOINTS[mode]
    return _Labelled(provider=provider, checkpoint=models / folder / "best.pt", source_tag=tag)
