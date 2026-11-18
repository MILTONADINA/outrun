"""Profiles, hazard policies and the beyond-horizon rule (specs/02-grid-risk.md, RISK-09, HORIZON-01..04).

``HazardPolicy`` fields match DESIGN 1.2 exactly; values come from ``configs/hazards.yaml``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from outrun.core.risk import Hazard

BeyondHorizon = Literal["hold", "decay", "drop"]
BEYOND_HORIZON_RULES: tuple[str, ...] = ("hold", "decay", "drop")


class Profile(Enum):
    """Routing profiles (RISK-09)."""

    CIVILIAN = "civilian"
    SENSITIVE = "sensitive"
    EV = "ev"


@dataclass(frozen=True)
class HazardPolicy:
    """How one hazard's risk turns into routing cost (DESIGN 1.2, RISK-09).

    - ``block_at[p]``: P >= value makes an edge impassable for profile p (None = never blocks).
    - ``speed_factor(p)``: in [1/3, 1]; the clock divides by it.
    - ``exposure_weight[p]``: w_h^p >= 0 in ``c = tau * (1 + sum_h w_h^p r_h)``.
    - ``beyond_horizon`` / ``decay_tau_min``: what happens after the layer's ``valid_until`` (HORIZON-01..03).
    """

    block_at: dict[Profile, float | None]
    speed_factor: Callable[[float], float]
    exposure_weight: dict[Profile, float]
    beyond_horizon: BeyondHorizon
    decay_tau_min: float | None


def make_speed_factor(slope: float) -> Callable[[Any], Any]:
    """``f(p) = clip(1 - slope * p, 1/3, 1)``, for floats or arrays (same shape out). ``slope < 0`` -> ValueError.

    RISK-09.
    """
    raise NotImplementedError


def blocks(policy: HazardPolicy, profile: Profile, p: Any) -> bool | np.ndarray:
    """False everywhere if ``policy.block_at[profile] is None``, else ``p >= block_at`` (RISK-09)."""
    raise NotImplementedError


def load_hazard_config(path: str | Path | None = None) -> dict[str, Any]:
    """Parsed ``hazards.yaml`` from ``outrun.config.get_paths().configs`` (env ``OUTRUN_CONFIG_DIR``) (RISK-09)."""
    raise NotImplementedError


def load_policies(path: str | Path | None = None) -> dict[Hazard, HazardPolicy]:
    """All six ``HazardPolicy`` objects from ``configs/hazards.yaml`` (RISK-09).

    A missing hazard or an invalid ``beyond_horizon`` raises ValueError.
    """
    raise NotImplementedError


def apply_beyond_horizon(
    prob: np.ndarray,
    policy: HazardPolicy,
    *,
    t0: datetime,
    issued: datetime,
    valid_until: datetime,
    n_slots: int = 12,
    slot_minutes: int = 15,
    zero_after_min: float | None = 120.0,
    linger_min: float = 15.0,
) -> np.ndarray:
    """Extend/replace slots after the layer horizon with the policy's hold / decay / drop rule.

    Slot k starts at ``t0 + k * slot_minutes``; it is valid iff ``k < prob.shape[0]`` and its start is before
    ``valid_until``. Valid slots (a prefix) are copied; ``L`` is the last valid slot and
    ``dt_k = max(0, minutes(t_k - valid_until))``.

    - hold (HORIZON-03): beyond slots = L.
    - decay (HORIZON-01): ``L * exp(-dt_k / decay_tau_min)``; 0 once ``t_k - issued >= zero_after_min``.
    - drop (HORIZON-02): 0, except cells with ``L >= min(non-None block_at)`` keep L while ``dt_k < linger_min``.

    Returns a new float32 array ``(n_slots, *prob.shape[1:])``; the input is not modified (HORIZON-04).
    Raises ValueError for no valid slot, naive datetimes, an unknown rule, or decay without a positive tau.
    """
    raise NotImplementedError
