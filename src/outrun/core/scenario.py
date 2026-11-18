"""Scenario: one routing case (id, t0, bbox, city, case metadata, held-out flags) (specs/02-grid-risk.md, RISK-10)."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from outrun.core.grid import GridWindow, bbox_to_window
from outrun.core.risk import Hazard

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class Scenario:
    """A routing case. Construction validates the fields and raises ValueError on bad input (RISK-10).

    ``bbox`` is ``(lon_min, lat_min, lon_max, lat_max)``; ``window`` (optional) overrides the bbox-derived window.
    """

    id: str
    t0: datetime
    bbox: tuple[float, float, float, float]
    city: str
    hazards: tuple[Hazard, ...]
    held_out: Mapping[Hazard, bool] = field(default_factory=dict)
    title: str = ""
    case: Mapping[str, Any] = field(default_factory=dict)
    synthetic: bool = False
    window: GridWindow | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ID_RE.match(self.id):
            raise ValueError(f"scenario id must match {_ID_RE.pattern}, got {self.id!r}")
        if not isinstance(self.t0, datetime) or self.t0.tzinfo is None or self.t0.utcoffset() != timedelta(0):
            raise ValueError("t0 must be a timezone-aware UTC datetime")
        try:
            lon_min, lat_min, lon_max, lat_max = (float(v) for v in self.bbox)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bbox must be 4 numbers, got {self.bbox!r}") from exc
        if not (-180 <= lon_min < lon_max <= 180 and -90 <= lat_min < lat_max <= 90):
            raise ValueError(f"invalid bbox {self.bbox!r}")
        if any(not isinstance(h, Hazard) for h in self.hazards) or len(set(self.hazards)) != len(
            self.hazards
        ):
            raise ValueError("hazards must be distinct Hazard members")
        if any(h not in self.hazards for h in self.held_out):
            raise ValueError("held_out keys must be scenario hazards")

    def grid_window(self) -> GridWindow:
        """``window`` if given, else ``bbox_to_window(bbox)`` (RISK-10)."""
        return self.window if self.window is not None else bbox_to_window(self.bbox)

    def is_held_out(self, hazard: Hazard) -> bool:
        """``held_out.get(hazard, False)`` (RISK-10)."""
        return bool(self.held_out.get(hazard, False))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict (ISO t0, string hazard values, window fields or None) (RISK-10)."""
        return {
            "id": self.id,
            "t0": self.t0.isoformat(),
            "bbox": [float(v) for v in self.bbox],
            "city": self.city,
            "hazards": [h.value for h in self.hazards],
            "held_out": {h.value: bool(v) for h, v in self.held_out.items()},
            "title": self.title,
            "case": dict(self.case),
            "synthetic": bool(self.synthetic),
            "window": None if self.window is None else asdict(self.window),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Scenario:
        """Exact inverse of ``to_dict`` (RISK-10)."""
        window = d.get("window")
        return cls(
            id=d["id"],
            t0=datetime.fromisoformat(d["t0"]),
            bbox=tuple(float(v) for v in d["bbox"]),
            city=d["city"],
            hazards=tuple(Hazard(h) for h in d["hazards"]),
            held_out={Hazard(h): bool(v) for h, v in (d.get("held_out") or {}).items()},
            title=d.get("title", ""),
            case=dict(d.get("case") or {}),
            synthetic=bool(d.get("synthetic", False)),
            window=None if window is None else GridWindow(**window),
        )

    def save(self, path: str | Path) -> None:
        """Write ``to_dict()`` as JSON (RISK-10)."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        """Read a JSON file written by ``save`` (RISK-10)."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
