"""Pydantic request models for the app API (spec 14: APP-03, APP-06). Pure data, fully defined.

Field-level checks live here (coordinate ranges, lambda >= 0, choice lists, 1..3 incidents). Checks that need the
scenario (origin inside the scenario bbox) are done in ``outrun.app.server``. Profile names, risk sources and
hazard names are plain strings in the wire format; the server validates profile names against
``outrun.core.policy.Profile`` (see specs/14-app.md "Shared definitions").
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from outrun.core.risk import RiskSource  # defined once in core (RISK-08)

Lon = Annotated[float, Field(ge=-180.0, le=180.0)]
Lat = Annotated[float, Field(ge=-90.0, le=90.0)]
LonLat = tuple[Lon, Lat]

OVERLAY_SOURCES: tuple[str, ...] = ("model", "baseline", "observed")
SHELTER_METHODS: tuple[str, ...] = ("mcf", "nearest", "greedy")
EVAC_METHODS: tuple[str, ...] = ("sa", "msa", "shortest", "safest")
SCENARIO_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"


class RouteRequest(BaseModel):
    """Body of ``POST /api/route`` (APP-03). ``lambda`` is a Python keyword, so the field is ``lam``."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    scenario: str
    origin: LonLat
    dest: LonLat | Literal["nearest_shelter"]
    profile: str = "civilian"
    lam: Annotated[float, Field(ge=0.0)] | None = Field(default=None, alias="lambda")
    t0: datetime | None = None
    risk_source: RiskSource = RiskSource.MODEL


class DispatchRequest(BaseModel):
    """Body of ``POST /api/dispatch`` (APP-06): 1 to 3 incidents."""

    model_config = ConfigDict(extra="forbid")

    scenario: str
    incidents: Annotated[list[LonLat], Field(min_length=1, max_length=3)]
    t0: datetime | None = None
