"""Observed-hazard readers for the routing evaluation (DESIGN 2.2: SEVIR future VIL, SPC tracks, Storm Events,
VIIRS/WSTS masks, NWS zones, AQS).

Placeholder. Each hazard adapter's ``observed()`` (``outrun.core.risk.HazardAdapter``, owned by specs 08-13)
returns the observed ``RiskLayer``; this module is where shared truth-rasterisation helpers will live if more
than one adapter needs them. It defines no requirement of its own (see specs/15-evaluation.md, "Open points").
"""

from __future__ import annotations

__all__: list[str] = []
