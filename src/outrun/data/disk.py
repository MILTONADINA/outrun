"""Disk budget (specs/01-data.md DATA-14; DESIGN 7.3): HDD <= 375 GB, SSD <= 225 GB by default.

``configs/disk.yaml`` overrides the caps (currently 300 / 150 GB: RAW and PROC share the one internal SSD).
"""

from __future__ import annotations

import os
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
    import yaml

    if config_dir is None:
        from outrun.config import get_paths

        config_dir = get_paths().configs
    path = Path(config_dir) / "disk.yaml"
    defaults = DiskBudget()
    if not path.exists():
        return defaults
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: top level must be a mapping")
    disk = data.get("disk") or {}
    if not isinstance(disk, Mapping):
        raise ValueError(f"{path}: 'disk' must be a mapping")
    hdd = disk.get("hdd_cap_gb")
    ssd = disk.get("ssd_cap_gb")
    return DiskBudget(
        hdd_cap_bytes=int(round(float(hdd) * GB)) if hdd is not None else defaults.hdd_cap_bytes,
        ssd_cap_bytes=int(round(float(ssd) * GB)) if ssd is not None else defaults.ssd_cap_bytes,
    )


def planned_bytes(manifest: Manifest) -> dict[str, int]:
    """DATA-14. Sum of approx_bytes per target; keys exactly hdd, ssd, stream."""
    totals = {"hdd": 0, "ssd": 0, "stream": 0}
    for s in manifest.sources:
        totals[s.target] += s.approx_bytes
    return totals


def check_budget(
    planned: Mapping[str, int], budget: DiskBudget, used: Mapping[str, int] | None = None
) -> list[str]:
    """DATA-14. Violation messages (naming 'hdd'/'ssd' and the GB overshoot); empty when within budget."""
    used = used or {}
    violations = []
    for disk, cap in (("hdd", budget.hdd_cap_bytes), ("ssd", budget.ssd_cap_bytes)):
        total = planned.get(disk, 0) + used.get(disk, 0)
        if total > cap:
            violations.append(
                f"{disk}: planned + used = {total / GB:.1f} GB exceeds the cap of {cap / GB:.1f} GB "
                f"by {(total - cap) / GB:.1f} GB"
            )
    return violations


def used_bytes(path: Path | str) -> int:
    """DATA-14. Total size of regular files under ``path`` (no symlink following; missing path -> 0)."""
    root = Path(path)
    if root.is_symlink() or not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            f = Path(dirpath) / name
            if f.is_symlink() or not f.is_file():
                continue
            total += f.stat().st_size
    return total


def check_free_space(
    path: Path | str,
    need_bytes: int,
    disk_usage: Callable = shutil.disk_usage,
    reserve_bytes: int = 0,
) -> int:
    """DATA-14. Return free bytes; DiskSpaceError if free - reserve_bytes < need_bytes."""
    free = int(disk_usage(path).free)
    if free - reserve_bytes < need_bytes:
        raise DiskSpaceError(
            f"{path}: need {need_bytes / GB:.2f} GB but only {free / GB:.2f} GB free "
            f"(reserve {reserve_bytes / GB:.2f} GB)"
        )
    return free


def cut_plan(hdd_bytes: int, cap_bytes: int, cuts: Sequence[tuple[str, int]]) -> list[str]:
    """DATA-14. Apply cuts in order until within cap; return names applied; DiskSpaceError if never enough."""
    applied: list[str] = []
    saved = 0
    for name, nbytes in cuts:
        if hdd_bytes - saved <= cap_bytes:
            return applied
        applied.append(name)
        saved += nbytes
    if hdd_bytes - saved <= cap_bytes:
        return applied
    raise DiskSpaceError(
        f"hdd: {hdd_bytes / GB:.1f} GB minus every cut ({saved / GB:.1f} GB) still exceeds the cap of "
        f"{cap_bytes / GB:.1f} GB"
    )
