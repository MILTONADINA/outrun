"""SEVIR training windows (STORM-01) and the IR107 satellite input / VIL-only ablation (STORM-02).

Reads the processed event memmap written by DATA-01: shape ``(N, 49, 2, H, W)`` uint8, channel 0 = VIL,
channel 1 = IR107 (180-320 K mapped linearly to 0..255).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from outrun.eval.splits import Split

N_FRAMES = 49
N_IN = 13
N_OUT = 12
OFFSETS: tuple[int, ...] = (0, 12, 24)
VIL_CH = 0
IR_CH = 1
NO_DATA_VIL = (
    255  # SEVIR VIL fill value: whole frames or blocks come back 255 where the radar mosaic has no data
)
BAD_FRAME_FRACTION = 0.02  # a frame with more than this share of 255 pixels is no-data, not weather


def bad_frames(events: np.ndarray, rows: Sequence[int] | None = None) -> np.ndarray:
    """Bool ``(len(rows), F)``: VIL frames that are mostly the 255 no-data fill (real storms saturate a few pixels).

    On SEVIR 2018-19 this is 0.5% of train frames but 12% of val frames (April-May 2019), so it must be handled.
    """
    rows = range(events.shape[0]) if rows is None else rows
    out = np.zeros((len(rows), events.shape[1]), dtype=bool)
    for k, r in enumerate(rows):
        out[k] = (np.asarray(events[r, :, VIL_CH]) == NO_DATA_VIL).mean(axis=(1, 2)) > BAD_FRAME_FRACTION
    return out


def window_index(n_events: int, offsets: Sequence[int] = OFFSETS) -> np.ndarray:
    """Int array ``(n_events * len(offsets), 2)`` of ``(event, offset)`` rows, event-major (STORM-01)."""
    ev = np.repeat(np.arange(n_events, dtype=np.int64), len(offsets))
    off = np.tile(np.asarray(offsets, dtype=np.int64), n_events)
    return np.stack([ev, off], axis=1).reshape(-1, 2)


def in_channels(use_ir: bool) -> int:
    """26 with IR107 channels, 13 for the VIL-only ablation (STORM-02)."""
    return 2 * N_IN if use_ir else N_IN


def events_with_both(catalog: pd.DataFrame) -> list[str]:
    """Sorted unique CATALOG ``id``s having both a ``vil`` and an ``ir107`` row (STORM-02).

    Raises KeyError if ``id`` or ``img_type`` is missing.
    """
    for col in ("id", "img_type"):
        if col not in catalog.columns:
            raise KeyError(col)
    vil = set(catalog.loc[catalog["img_type"] == "vil", "id"])
    ir = set(catalog.loc[catalog["img_type"] == "ir107", "id"])
    return sorted(vil & ir)


class SevirWindowDataset(Dataset):
    """Windows of 13 input frames -> 12 target frames (STORM-01, STORM-02).

    ``ds[i] -> (x, y)``: ``x`` float32 ``(26|13, H, W)`` = VIL/255 (+ IR/255); ``y`` float32 ``(12, H, W)`` in
    VIL digital units. Raises ValueError on a bad event array shape, IndexError on an out-of-range index.
    """

    def __init__(
        self,
        events: np.ndarray,
        event_ids: Sequence[int] | None = None,
        use_ir: bool = True,
        offsets: Sequence[int] = OFFSETS,
        drop_bad: bool = False,
    ) -> None:
        if events.ndim != 5:
            raise ValueError(f"events must be (N, F, 2, H, W), got ndim {events.ndim}")
        offsets = tuple(int(o) for o in offsets)
        if events.shape[1] < max(offsets) + N_IN + N_OUT:
            raise ValueError(f"need at least {max(offsets) + N_IN + N_OUT} frames, got {events.shape[1]}")
        if use_ir and events.shape[2] < 2:
            raise ValueError("use_ir=True needs an IR107 channel (events.shape[2] >= 2)")
        self.events = events
        self.use_ir = use_ir
        self.offsets = offsets
        self.event_ids = (
            np.arange(events.shape[0]) if event_ids is None else np.asarray(event_ids, dtype=np.int64)
        )
        self.ids: list[str] | None = None
        # (event row, frame offset) per window; with ``drop_bad`` any window touching a no-data frame is skipped
        pos_off = window_index(len(self.event_ids), offsets)  # (position in event_ids, offset)
        if drop_bad:
            bad = bad_frames(events, [int(e) for e in self.event_ids])
            span = N_IN + N_OUT
            keep = [w for w in pos_off if not bad[w[0], w[1] : w[1] + span].any()]
            pos_off = np.array(keep, dtype=np.int64).reshape(-1, 2)
        self.windows = np.stack([self.event_ids[pos_off[:, 0]], pos_off[:, 1]], axis=1)

    @classmethod
    def from_split(
        cls,
        events: np.ndarray,
        event_ids: Sequence[str],
        split: Split,
        part: str,
        use_ir: bool = True,
    ) -> SevirWindowDataset:
        """STORM-01 / SPLIT-07. Only events whose id is in ``split.<part>`` (``eval.splits.split_mask``).

        Sets ``ds.ids`` (selected event ids, row order). Windows with a no-data (255-filled) frame are dropped. ``len(event_ids) != len(events)`` -> ValueError.
        """
        from outrun.eval.splits import split_mask

        if len(event_ids) != len(events):
            raise ValueError(f"{len(event_ids)} event ids for {len(events)} event rows")
        rows = np.flatnonzero(split_mask(event_ids, split, part))
        ds = cls(events, event_ids=rows, use_ir=use_ir, drop_bad=True)
        ds.ids = [event_ids[r] for r in rows]
        return ds

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= i < len(self):
            raise IndexError(i)
        e, o = (int(v) for v in self.windows[i])
        ev = self.events
        chans = [np.asarray(ev[e, o : o + N_IN, VIL_CH], dtype=np.float32) / 255.0]
        if self.use_ir:
            chans.append(np.asarray(ev[e, o : o + N_IN, IR_CH], dtype=np.float32) / 255.0)
        y = np.asarray(ev[e, o + N_IN : o + N_IN + N_OUT, VIL_CH], dtype=np.float32)
        return torch.from_numpy(np.concatenate(chans, axis=0)), torch.from_numpy(y)
