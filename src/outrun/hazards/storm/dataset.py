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


def window_index(n_events: int, offsets: Sequence[int] = OFFSETS) -> np.ndarray:
    """Int array ``(n_events * len(offsets), 2)`` of ``(event, offset)`` rows, event-major (STORM-01)."""
    raise NotImplementedError


def in_channels(use_ir: bool) -> int:
    """26 with IR107 channels, 13 for the VIL-only ablation (STORM-02)."""
    raise NotImplementedError


def events_with_both(catalog: pd.DataFrame) -> list[str]:
    """Sorted unique CATALOG ``id``s having both a ``vil`` and an ``ir107`` row (STORM-02).

    Raises KeyError if ``id`` or ``img_type`` is missing.
    """
    raise NotImplementedError


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
    ) -> None:
        raise NotImplementedError

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

        Sets ``ds.ids`` (selected event ids, row order). ``len(event_ids) != len(events)`` -> ValueError.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
