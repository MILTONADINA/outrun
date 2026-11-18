"""Paths, YAML config loading, seeds and device (specs/01-data.md DATA-15, DATA-16).

RAW (external HDD) and PROC (SSD) roots come from the environment; every tunable value comes from ``configs/*.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_RAW = Path("/Volumes/OutrunRaw")


@dataclass(frozen=True)
class Paths:
    """Absolute project paths (DATA-15)."""

    root: Path
    raw: Path
    proc: Path
    configs: Path
    splits: Path
    scenarios: Path
    models: Path
    logs: Path


def get_paths(env: Mapping[str, str] | None = None) -> Paths:
    """DATA-15. raw = OUTRUN_RAW or /Volumes/OutrunRaw; proc = OUTRUN_PROC or REPO_ROOT/data/proc;
    configs = OUTRUN_CONFIG_DIR or REPO_ROOT/configs; splits/scenarios/models/logs = REPO_ROOT/<name>.
    Values are ~-expanded and made absolute; empty string = unset; nothing is created."""
    raise NotImplementedError


def load_config(name: str, config_dir: Path | str | None = None) -> dict:
    """DATA-16. Read ``<config_dir or get_paths().configs>/<name>.yaml`` (suffix optional) as a fresh dict.
    Missing file -> FileNotFoundError; non-mapping top level -> ValueError."""
    raise NotImplementedError


def seed_everything(seed: int) -> None:
    """DATA-16. Seed ``random``, NumPy's global generator and torch (if importable)."""
    raise NotImplementedError


def get_device(env: Mapping[str, str] | None = None) -> str:
    """DATA-16. OUTRUN_DEVICE in {"cpu", "mps"} returned as given; other non-empty value -> ValueError;
    unset -> "mps" if torch MPS is available else "cpu"."""
    raise NotImplementedError
