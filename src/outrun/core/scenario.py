"""Scenario: one routing case (id, t0, bbox, city, case metadata, held-out flags) (specs/02-grid-risk.md, RISK-10)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from outrun.core.grid import GridWindow
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
        raise NotImplementedError

    def is_held_out(self, hazard: Hazard) -> bool:
        """``held_out.get(hazard, False)`` (RISK-10)."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict (ISO t0, string hazard values, window fields or None) (RISK-10)."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Scenario:
        """Exact inverse of ``to_dict`` (RISK-10)."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Write ``to_dict()`` as JSON (RISK-10)."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        """Read a JSON file written by ``save`` (RISK-10)."""
        raise NotImplementedError
