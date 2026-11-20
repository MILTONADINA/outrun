"""No-data (255-filled) VIL frames: detection and window dropping (specs/08-storm.md STORM-01). Synthetic only."""

from __future__ import annotations

import numpy as np
import pytest

H = W = 32


def clean_events(n: int = 2) -> np.ndarray:
    """(n, 49, 2, H, W) uint8 with VIL values 0..200, so no pixel equals the 255 fill."""
    return np.random.default_rng(0).integers(0, 200, size=(n, 49, 2, H, W), dtype=np.uint8)


@pytest.mark.req("STORM-01")
def test_bad_frames_flags_only_frames_above_two_percent_fill():
    from outrun.hazards.storm.dataset import bad_frames

    ev = clean_events(2)
    ev[0, 5, 0] = 255  # whole frame is no-data
    ev[0, 6, 0, :, :16] = 255  # half a frame
    ev[0, 7, 0].flat[: int(0.01 * H * W)] = 255  # 1% of pixels: real saturation, not flagged
    ev[1, 3, 1] = 255  # IR channel is ignored
    bad = bad_frames(ev)
    assert bad.shape == (2, 49)
    assert bad[0].nonzero()[0].tolist() == [5, 6]
    assert not bad[1].any()
    assert bad_frames(ev, [1]).shape == (1, 49) and not bad_frames(ev, [1]).any()


@pytest.mark.req("STORM-01")
def test_drop_bad_removes_windows_touching_a_no_data_frame():
    from outrun.hazards.storm.dataset import SevirWindowDataset

    ev = clean_events(2)
    ev[0, 30, 0] = 255  # frame 30 lies in the offset-12 window (12..36) and the offset-24 window (24..48)
    kept = SevirWindowDataset(ev, drop_bad=True)
    assert [tuple(w) for w in kept.windows] == [(0, 0), (1, 0), (1, 12), (1, 24)]
    assert len(SevirWindowDataset(ev)) == 6  # default keeps everything


@pytest.mark.req("STORM-01")
@pytest.mark.req("SPLIT-07")
def test_from_split_drops_bad_windows_and_keeps_ids():
    from outrun.eval.splits import Split, SplitItem
    from outrun.hazards.storm.dataset import SevirWindowDataset

    split = Split(
        "toy",
        train=(SplitItem("E0"),),
        val=(SplitItem("E1"), SplitItem("E2")),
        test=(SplitItem("E3"),),
    )
    ev = clean_events(4)
    ev[1, 0, 0] = 255  # E1 frame 0: only the offset-0 window touches it
    ev[3, :, 0] = 255  # test event all bad, must not affect val
    ids = ["E0", "E1", "E2", "E3"]
    va = SevirWindowDataset.from_split(ev, ids, split, "val")
    assert tuple(va.ids) == ("E1", "E2")
    assert len(va) == 2 + 3
    assert all(int(e) in (1, 2) for e, _ in va.windows)
    assert len(SevirWindowDataset.from_split(ev, ids, split, "train")) == 3
    assert len(SevirWindowDataset.from_split(ev, ids, split, "test")) == 0
