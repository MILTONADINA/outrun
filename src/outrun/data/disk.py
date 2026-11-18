"""Disk budget (specs/01-data.md DATA-14; DESIGN 7.3): HDD <= 375 GB, SSD <= 225 GB."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from outrun.data.manifest import Manifest

GB = 10**9


class DiskSpaceError(OSError):
    """Not enough free space, or no cut combination fits the cap."""


@dataclass(frozen=True)
class DiskBudget:
    hdd_cap_bytes: int = 375 * GB
    ssd_cap_bytes: int = 225 * GB


def load_disk_budget(config_dir: Path | str | None = None) -> DiskBudget:
    """DATA-14. ``disk:`` {hdd_cap_gb, ssd_cap_gb} from configs/disk.yaml if it exists, else DESIGN defaults."""
    raise NotImplementedError


def planned_bytes(manifest: Manifest) -> dict[str, int]:
    """DATA-14. Sum of approx_bytes per target; keys exactly hdd, ssd, stream."""
    raise NotImplementedError


def check_budget(
    planned: Mapping[str, int], budget: DiskBudget, used: Mapping[str, int] | None = None
) -> list[str]:
    """DATA-14. Violation messages (naming 'hdd'/'ssd' and the GB overshoot); empty when within budget."""
    raise NotImplementedError


def used_bytes(path: Path | str) -> int:
    """DATA-14. Total size of regular files under ``path`` (no symlink following; missing path -> 0)."""
    raise NotImplementedError


def check_free_space(
    path: Path | str,
    need_bytes: int,
    disk_usage: Callable = shutil.disk_usage,
    reserve_bytes: int = 0,
) -> int:
    """DATA-14. Return free bytes; DiskSpaceError if free - reserve_bytes < need_bytes."""
    raise NotImplementedError


def cut_plan(hdd_bytes: int, cap_bytes: int, cuts: Sequence[tuple[str, int]]) -> list[str]:
    """DATA-14. Apply cuts in order until within cap; return names applied; DiskSpaceError if never enough."""
    raise NotImplementedError
