"""Paths, YAML config loading, seeds and device (specs/01-data.md DATA-15, DATA-16).

RAW and PROC roots come from the environment (OUTRUN_RAW / OUTRUN_PROC; both on the internal SSD, see README);
every tunable value comes from ``configs/*.yaml``.
"""

from __future__ import annotations

import os
import random
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
    if env is None:
        env = os.environ

    def from_env(name: str, default: Path) -> Path:
        value = env.get(name, "")
        if not value:
            return default
        return Path(value).expanduser().resolve()

    return Paths(
        root=REPO_ROOT,
        raw=from_env("OUTRUN_RAW", DEFAULT_RAW),
        proc=from_env("OUTRUN_PROC", REPO_ROOT / "data" / "proc"),
        configs=from_env("OUTRUN_CONFIG_DIR", REPO_ROOT / "configs"),
        splits=REPO_ROOT / "splits",
        scenarios=REPO_ROOT / "scenarios",
        models=REPO_ROOT / "models",
        logs=REPO_ROOT / "logs",
    )


def load_config(name: str, config_dir: Path | str | None = None) -> dict:
    """DATA-16. Read ``<config_dir or get_paths().configs>/<name>.yaml`` (suffix optional) as a fresh dict.
    Missing file -> FileNotFoundError; non-mapping top level -> ValueError."""
    import yaml

    base = Path(config_dir) if config_dir is not None else get_paths().configs
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    path = base / filename
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data  # parsed fresh on every call, so callers may mutate it


def seed_everything(seed: int) -> None:
    """DATA-16. Seed ``random``, NumPy's global generator and torch (if importable)."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)


def get_device(env: Mapping[str, str] | None = None) -> str:
    """DATA-16. OUTRUN_DEVICE in {"cpu", "mps"} returned as given; other non-empty value -> ValueError;
    unset -> "mps" if torch MPS is available else "cpu"."""
    if env is None:
        env = os.environ
    value = env.get("OUTRUN_DEVICE", "")
    if value:
        if value not in ("cpu", "mps"):
            raise ValueError(f"OUTRUN_DEVICE must be 'cpu' or 'mps', got {value!r}")
        return value
    try:
        import torch
    except ImportError:
        return "cpu"
    return "mps" if torch.backends.mps.is_available() else "cpu"
