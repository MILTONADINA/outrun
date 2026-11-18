"""Profiles, hazard policies and the beyond-horizon rule (specs/02-grid-risk.md, RISK-09, HORIZON-01..04).

``HazardPolicy`` fields match DESIGN 1.2 exactly; values come from ``configs/hazards.yaml``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    if slope < 0:
        raise ValueError(f"speed_factor slope must be >= 0, got {slope}")

    def speed_factor(p: Any) -> Any:
        out = np.clip(1.0 - slope * np.asarray(p, dtype=np.float64), 1.0 / 3.0, 1.0)
        return float(out) if out.ndim == 0 else out

    return speed_factor


def blocks(policy: HazardPolicy, profile: Profile, p: Any) -> bool | np.ndarray:
    """False everywhere if ``policy.block_at[profile] is None``, else ``p >= block_at`` (RISK-09)."""
    threshold = policy.block_at.get(profile)
    p_arr = np.asarray(p)
    result = np.zeros(p_arr.shape, dtype=bool) if threshold is None else p_arr >= threshold
    return bool(result) if result.ndim == 0 else result


def load_hazard_config(path: str | Path | None = None) -> dict[str, Any]:
    """Parsed ``hazards.yaml`` from ``outrun.config.get_paths().configs`` (env ``OUTRUN_CONFIG_DIR``) (RISK-09)."""
    from outrun.config import load_config

    if path is None:
        return load_config("hazards")
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def load_policies(path: str | Path | None = None) -> dict[Hazard, HazardPolicy]:
    """All six ``HazardPolicy`` objects from ``configs/hazards.yaml`` (RISK-09).

    A missing hazard or an invalid ``beyond_horizon`` raises ValueError.
    """
    from outrun.core.risk import Hazard

    cfg = load_hazard_config(path).get("hazards") or {}
    policies: dict[Hazard, HazardPolicy] = {}
    for hazard in Hazard:
        if hazard.value not in cfg:
            raise ValueError(f"hazards.yaml has no entry for {hazard.value!r}")
        h = cfg[hazard.value]
        try:
            block_at = {
                prof: (None if h["block_at"][prof.value] is None else float(h["block_at"][prof.value]))
                for prof in Profile
            }
            weights = {prof: float(h["exposure_weight"][prof.value]) for prof in Profile}
            slope = float(h["speed_factor_slope"])
            rule = h["beyond_horizon"]
        except KeyError as exc:
            raise ValueError(f"hazards.yaml entry {hazard.value!r} is missing {exc}") from exc
        if rule not in BEYOND_HORIZON_RULES:
            raise ValueError(
                f"{hazard.value}: beyond_horizon must be one of {BEYOND_HORIZON_RULES}, got {rule!r}"
            )
        tau = h.get("decay_tau_min")
        policies[hazard] = HazardPolicy(
            block_at=block_at,
            speed_factor=make_speed_factor(slope),
            exposure_weight=weights,
            beyond_horizon=rule,
            decay_tau_min=None if tau is None else float(tau),
        )
    return policies


def _is_aware(t: Any) -> bool:
    return isinstance(t, datetime) and t.tzinfo is not None and t.utcoffset() is not None


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
    if not (_is_aware(t0) and _is_aware(issued) and _is_aware(valid_until)):
        raise ValueError("t0, issued and valid_until must be timezone-aware datetimes")
    rule = policy.beyond_horizon
    if rule not in BEYOND_HORIZON_RULES:
        raise ValueError(f"unknown beyond_horizon rule {rule!r}")
    tau = policy.decay_tau_min
    if rule == "decay" and (tau is None or tau <= 0):
        raise ValueError(f"decay needs decay_tau_min > 0, got {tau!r}")

    prob = np.asarray(prob)
    slot_starts = [t0 + timedelta(minutes=k * slot_minutes) for k in range(n_slots)]

    # Valid slots: the layer has data for them and they start before the horizon ends (always a prefix).
    n_valid = 0
    while n_valid < min(n_slots, prob.shape[0]) and slot_starts[n_valid] < valid_until:
        n_valid += 1
    if n_valid == 0:
        raise ValueError("the layer has no valid slot in the routing window (horizon ends at or before t0)")

    out = np.zeros((n_slots, *prob.shape[1:]), dtype=np.float32)  # new array: the input is never modified
    out[:n_valid] = prob[:n_valid]
    last = out[n_valid - 1].copy()  # L, the last valid slot

    if rule == "drop":
        thresholds = [b for b in policy.block_at.values() if b is not None]
        lingering = last >= min(thresholds) if thresholds else np.zeros(last.shape, dtype=bool)

    for k in range(n_valid, n_slots):
        dt_min = max(0.0, (slot_starts[k] - valid_until).total_seconds() / 60.0)
        if rule == "hold":
            out[k] = last
        elif rule == "decay":
            since_issue_min = (slot_starts[k] - issued).total_seconds() / 60.0
            if zero_after_min is not None and since_issue_min >= zero_after_min:
                out[k] = 0.0
            else:
                out[k] = last * np.exp(-dt_min / tau)
        else:  # drop: zero, except blocked cells that linger for linger_min after the horizon
            out[k] = np.where(lingering, last, 0.0) if dt_min < linger_min else 0.0
    return np.clip(out, 0.0, 1.0, out=out)
