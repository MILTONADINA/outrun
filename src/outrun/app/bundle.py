"""Scenario bundle reader (spec 14: APP-01, APP-08, APP-14).

Layout of ``<scenarios_dir>/<id>/`` (scenario.json, overlays/<source>/<hazard>/<slot:02d>.png, results/*.json) is
defined in specs/14-app.md. Bundles are written by ``outrun scenario`` (T22) and read here, by the server and by
``export_static``. Nothing here touches the network.
"""

from __future__ import annotations

import json
import math
import re
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
OVERLAY_SOURCES: tuple[str, ...] = ("model", "baseline", "observed")
_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")


def is_valid_scenario_id(scenario_id: str) -> bool:
    """True iff ``scenario_id`` matches ``^[a-z0-9][a-z0-9_-]{0,63}$`` (APP-14)."""
    return isinstance(scenario_id, str) and _ID_RE.fullmatch(scenario_id) is not None


def scenario_dir(root: Path, scenario_id: str) -> Path:
    """``root / scenario_id`` for a valid id; ``ValueError`` otherwise (APP-14)."""
    if not is_valid_scenario_id(scenario_id):
        raise ValueError(f"invalid scenario id: {scenario_id!r}")
    return Path(root) / scenario_id


def load_scenario_meta(folder: Path) -> dict[str, Any] | None:
    """Parsed ``scenario.json`` of one bundle with defaults filled in and ``overlay_sources`` added, or ``None``
    if the folder is not a valid bundle (missing/invalid JSON, missing key, id != folder name, bad id, bad bbox).
    (APP-01)
    """
    try:
        meta = json.loads((Path(folder) / "scenario.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or any(k not in meta for k in REQUIRED_KEYS):
        return None
    if not is_valid_scenario_id(meta["id"]) or meta["id"] != Path(folder).name:
        return None
    if not _valid_bbox(meta["bbox"]):
        return None
    meta = {**DEFAULTS, **meta}
    overlays = Path(folder) / "overlays"
    meta["overlay_sources"] = sorted(s for s in OVERLAY_SOURCES if (overlays / s).is_dir())
    return meta


def list_scenarios(scenarios_dir: Path) -> list[dict[str, Any]]:
    """All valid bundles' metadata, sorted by id; ``[]`` for a missing or empty folder (APP-01)."""
    root = Path(scenarios_dir)
    if not root.is_dir():
        return []
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            meta = load_scenario_meta(child)
            if meta is not None:
                out.append(meta)
    return sorted(out, key=lambda m: m["id"])


def read_metrics(reports_dir: Path) -> dict[str, Any]:
    """``{"metrics": {stem: json}, "invalid": [file names]}`` from ``<reports_dir>/metrics/*.json`` (APP-08)."""
    metrics: dict[str, Any] = {}
    invalid: list[str] = []
    folder = Path(reports_dir) / "metrics"
    if folder.is_dir():
        for f in sorted(folder.glob("*.json")):
            try:
                metrics[f.stem] = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                invalid.append(f.name)
    return {"metrics": metrics, "invalid": sorted(invalid)}


def _valid_bbox(bbox: Any) -> bool:
    """``[west, south, east, north]`` finite numbers with west < east and south < north."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bbox):
        return False
    return bbox[0] < bbox[2] and bbox[1] < bbox[3]
