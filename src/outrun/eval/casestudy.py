"""Case-study registry (specs/15-evaluation.md, SPLIT-11; DESIGN 6.4).

A case study is a real date that is either quarantined from train/val (``held_out="quarantine"``, e.g. the El
Reno primary case on SEVIR's train/val side) or already inside an official test split
(``held_out="official_test"``). Only quarantine cases feed ``splits/quarantine.json``. Case-study scenario runs
themselves (risk.zarr, overlays, results) are built by ``outrun scenario`` from ``core.scenario.Scenario``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from typing import Literal

__all__ = ["CaseStudy", "case_quarantine"]


@dataclass(frozen=True)
class CaseStudy:
    """One real case-study date (pure data)."""

    id: str
    hazards: tuple[str, ...]  # lower-case hazard names: storm, tornado, flood, wildfire, ice, smoke
    day: date  # UTC calendar day
    city: str
    held_out: Literal["quarantine", "official_test"]
    primary: bool = False


def case_quarantine(cases: Iterable[CaseStudy]) -> dict[str, date]:
    """SPLIT-11. ``{case id: day}`` for the quarantine cases only; ``ValueError`` on duplicate ids or on more
    than one primary case."""
    cases = list(cases)
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case-study ids")
    if sum(1 for c in cases if c.primary) > 1:
        raise ValueError("more than one primary case study")
    return {c.id: c.day for c in cases if c.held_out == "quarantine"}
