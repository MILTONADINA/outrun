"""Chip tensors (TORN-01) and hand-crafted radar features (TORN-03).

A chip maps each variable in ``VARIABLES`` to a float array ``(T, 2, A, R)`` (frame, tilt, azimuth, range gate);
missing gates are NaN. Tilt 0 is the lowest tilt.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

VARIABLES: tuple[str, ...] = ("DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH")
N_TILTS = 2
FILL_VALUES: dict[str, float] = {"DBZ": -20.0, "VEL": 0.0, "KDP": 0.0, "RHOHV": 0.0, "ZDR": 0.0, "WIDTH": 0.0}
N_CHANNELS = 27
N_CHANNELS_NO_RANGE = 26
N_NORMALIZED = 24
NORM_CLIP = 10.0  # standardised channels are clipped to +-10 sigma (aliased velocity outliers)
HAND_FEATURES: tuple[str, ...] = (
    "max_dbz", "max_dv", "min_rhohv_dbz30", "max_zdr", "max_kdp", "max_width", "range_km",
)  # fmt: skip


def coord_channels(range_km: np.ndarray, azimuth_deg: np.ndarray, include_range: bool = True) -> np.ndarray:
    """float32 ``(1|2, A, R)`` CoordConv channels: ``range_km / 300`` (if ``include_range``) then azimuth in [-1, 1]."""
    rng = np.asarray(range_km, dtype=np.float32)
    az = np.asarray(azimuth_deg, dtype=np.float64)
    n_a, n_r = az.shape[0], rng.shape[0]
    if az.size > 1 and az.max() > az.min():
        az_scaled = 2.0 * (az - az.min()) / (az.max() - az.min()) - 1.0
    else:
        az_scaled = np.zeros_like(az)
    out = []
    if include_range:
        out.append(np.broadcast_to(rng[None, :] / 300.0, (n_a, n_r)))
    out.append(np.broadcast_to(az_scaled.astype(np.float32)[:, None], (n_a, n_r)))
    return np.stack(out).astype(np.float32)


def build_chip_tensor(
    chip: Mapping[str, np.ndarray],
    range_km: np.ndarray,
    azimuth_deg: np.ndarray,
    include_range: bool = True,
) -> np.ndarray:
    """float32 ``(27|26, A, R)``: 12 last-frame, 12 last-minus-first, missing mask, range/300, azimuth in [-1, 1].

    KeyError on a missing variable, ValueError on inconsistent shapes (TORN-01). Does not modify ``chip``.
    """
    arrays = {
        v: np.asarray(chip[v], dtype=np.float32) for v in VARIABLES
    }  # KeyError if a variable is missing
    shape = arrays[VARIABLES[0]].shape
    if len(shape) != 4 or shape[1] != N_TILTS:
        raise ValueError(f"chip arrays must be (T, {N_TILTS}, A, R), got {shape}")
    if any(a.shape != shape for a in arrays.values()):
        raise ValueError("all chip variables must have the same shape")
    _, _, n_a, n_r = shape
    if np.shape(range_km) != (n_r,) or np.shape(azimuth_deg) != (n_a,):
        raise ValueError("range_km / azimuth_deg lengths must match the chip's range gates / azimuths")

    last, diff = [], []
    missing = np.zeros((n_a, n_r), dtype=bool)
    for var in VARIABLES:
        a = arrays[var]
        for tilt in range(N_TILTS):
            a_last, a_first = a[-1, tilt], a[0, tilt]
            missing |= np.isnan(a_last)
            fill = FILL_VALUES[var]
            f_last = np.where(np.isnan(a_last), fill, a_last)
            f_first = np.where(np.isnan(a_first), fill, a_first)
            last.append(f_last)
            diff.append(f_last - f_first if a.shape[0] > 1 else np.zeros_like(f_last))
    coords = coord_channels(range_km, azimuth_deg, include_range)
    return np.concatenate([np.stack(last + diff), missing[None].astype(np.float32), coords]).astype(
        np.float32
    )


def fit_channel_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and std over (N, A, R); std 0 -> 1 (TORN-01)."""
    mean = X.mean(axis=(0, 2, 3), dtype=np.float64)
    std = X.std(axis=(0, 2, 3), dtype=np.float64)
    std = np.where(std <= 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray, n_norm: int = N_NORMALIZED, clip: float | None = None
) -> np.ndarray:
    """float32 copy with the first ``n_norm`` channels standardised, the rest unchanged (TORN-01).

    ``clip`` (optional, default off) limits the standardised channels to ``[-clip, clip]``: TorNet velocity has
    aliasing outliers (+-129 m/s), so training and the adapter pass the same ``clip``.
    """
    out = np.array(X, dtype=np.float32, copy=True)
    m = np.asarray(mean, dtype=np.float32)[:n_norm].reshape(1, n_norm, 1, 1)
    s = np.asarray(std, dtype=np.float32)[:n_norm].reshape(1, n_norm, 1, 1)
    out[:, :n_norm] = (out[:, :n_norm] - m) / s
    if clip is not None:
        np.clip(out[:, :n_norm], -clip, clip, out=out[:, :n_norm])
    return out


def max_dv(chip: Mapping[str, np.ndarray], tilt: int = 0) -> float:
    """Max |VEL[a+1, r] - VEL[a, r]| in the last frame at ``tilt``, NaN pairs skipped, 0 if none (TORN-03)."""
    vel = np.asarray(chip["VEL"], dtype=np.float64)[-1, tilt]
    d = np.abs(vel[1:] - vel[:-1])
    d = d[np.isfinite(d)]
    return float(d.max()) if d.size else 0.0


def _max_or_fill(a: np.ndarray, var: str) -> float:
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else FILL_VALUES[var]


def hand_features(chip: Mapping[str, np.ndarray], range_km_center: float) -> np.ndarray:
    """float64 ``(7,)`` in ``HAND_FEATURES`` order from the last frame, tilt 0 (TORN-03)."""
    last = {v: np.asarray(chip[v], dtype=np.float64)[-1, 0] for v in VARIABLES}
    debris = (last["DBZ"] > 30.0) & np.isfinite(last["RHOHV"])  # NaN DBZ compares False
    min_rhohv = float(last["RHOHV"][debris].min()) if debris.any() else 1.0
    return np.array(
        [
            _max_or_fill(last["DBZ"], "DBZ"),
            max_dv(chip, tilt=0),
            min_rhohv,
            _max_or_fill(last["ZDR"], "ZDR"),
            _max_or_fill(last["KDP"], "KDP"),
            _max_or_fill(last["WIDTH"], "WIDTH"),
            float(range_km_center),
        ],
        dtype=np.float64,
    )


def chip_from_stored(x: np.ndarray) -> dict[str, np.ndarray]:
    """Rebuild a last-frame chip (T = 1, NaN for missing) from a stored ``(>=25, A, R)`` processed chip.

    The processed arrays hold fill values plus the union missing mask (channel 24), not per-variable NaNs. A gate is
    treated as missing for a variable when the mask is set there and the value equals that variable's fill value.
    """
    x = np.asarray(x, dtype=np.float32)
    miss = x[24] > 0
    chip = {}
    for v, var in enumerate(VARIABLES):
        a = x[2 * v : 2 * v + 2].copy()  # (2 tilts, A, R)
        a[(a == FILL_VALUES[var]) & miss[None]] = np.nan
        chip[var] = a[None]
    return chip
