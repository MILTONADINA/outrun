"""Scenario bundle reader (spec 14: APP-01, APP-08, APP-14).

Layout of ``<scenarios_dir>/<id>/`` (scenario.json, overlays/<source>/<hazard>/<slot:02d>.png, results/*.json) is
defined in specs/14-app.md. Bundles are written by ``outrun scenario`` (T22) and read here, by the server and by
``export_static``. Nothing here touches the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REQUIRED_KEYS: tuple[str, ...] = (
    "id",
    "title",
    "t0",
    "city",
    "bbox",
    "hazards",
    "held_out",
    "synthetic",
    "sources",
)
DEFAULTS: dict[str, Any] = {"n_slots": 12, "slot_minutes": 15}


def is_valid_scenario_id(scenario_id: str) -> bool:
    """True iff ``scenario_id`` matches ``^[a-z0-9][a-z0-9_-]{0,63}$`` (APP-14)."""
    raise NotImplementedError


def scenario_dir(root: Path, scenario_id: str) -> Path:
    """``root / scenario_id`` for a valid id; ``ValueError`` otherwise (APP-14)."""
    raise NotImplementedError


def load_scenario_meta(folder: Path) -> dict[str, Any] | None:
    """Parsed ``scenario.json`` of one bundle with defaults filled in and ``overlay_sources`` added, or ``None``
    if the folder is not a valid bundle (missing/invalid JSON, missing key, id != folder name, bad id, bad bbox).
    (APP-01)
    """
    raise NotImplementedError


def list_scenarios(scenarios_dir: Path) -> list[dict[str, Any]]:
    """All valid bundles' metadata, sorted by id; ``[]`` for a missing or empty folder (APP-01)."""
    raise NotImplementedError


def read_metrics(reports_dir: Path) -> dict[str, Any]:
    """``{"metrics": {stem: json}, "invalid": [file names]}`` from ``<reports_dir>/metrics/*.json`` (APP-08)."""
    raise NotImplementedError
