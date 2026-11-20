"""Persistence and optical-flow extrapolation baselines (STORM-06) and motion export (STORM-09).

pySTEPS (Lucas-Kanade + semi-Lagrangian) is used when installed; otherwise an own numpy/scipy fallback.
Image axes: dx + toward increasing column, dy + toward increasing row (rows run north to south).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage

try:  # optional backend
    from pysteps.motion.lucaskanade import dense_lucaskanade as _dense_lk

    PYSTEPS_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    PYSTEPS_AVAILABLE = False


def _last(frames: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError(f"frames must be (T >= 1, H, W), got {frames.shape}")
    return frames[-1]


def persistence(frames: np.ndarray, n_out: int = 12) -> np.ndarray:
    """float32 ``(n_out, H, W)``, every slice = ``frames[-1]``. ValueError if no frames (STORM-06)."""
    last = _last(frames).astype(np.float32)
    return np.repeat(last[None], n_out, axis=0)


def persistence_mask(frames: np.ndarray, threshold: float = 133, n_out: int = 12) -> np.ndarray:
    """float32 ``(n_out, H, W)`` of ``frames[-1] >= threshold`` (STORM-06)."""
    mask = (_last(frames) >= threshold).astype(np.float32)
    return np.repeat(mask[None], n_out, axis=0)


def _global_shift(prev: np.ndarray, cur: np.ndarray, iters: int = 8) -> tuple[float, float]:
    """Own fallback: one global (dx, dy) in px/frame from iterated Lucas-Kanade (Gauss-Newton on a warped frame)."""
    dx = dy = 0.0
    gy, gx = np.gradient(cur)
    rows, cols = np.mgrid[0 : cur.shape[0], 0 : cur.shape[1]].astype(np.float64)
    for _ in range(iters):
        # prev(r, c) should equal cur(r + dy, c + dx): sample cur at the shifted position
        pts = [rows + dy, cols + dx]
        warped = ndimage.map_coordinates(cur, pts, order=1, mode="nearest")
        wgx = ndimage.map_coordinates(gx, pts, order=1, mode="nearest")
        wgy = ndimage.map_coordinates(gy, pts, order=1, mode="nearest")
        err = prev - warped
        a = np.array([[np.sum(wgx * wgx), np.sum(wgx * wgy)], [np.sum(wgx * wgy), np.sum(wgy * wgy)]])
        b = np.array([np.sum(wgx * err), np.sum(wgy * err)])
        if abs(np.linalg.det(a)) < 1e-9:
            break
        step = np.linalg.solve(a, b)
        dx, dy = dx + step[0], dy + step[1]
        if np.abs(step).max() < 1e-4:
            break
    return float(dx), float(dy)


def _fallback_motion(frames: np.ndarray) -> np.ndarray:
    """Uniform field: mean over consecutive frame pairs of the global shift."""
    f = frames.astype(np.float64)
    shifts = [_global_shift(f[k], f[k + 1]) for k in range(len(f) - 1)]
    dx, dy = np.mean(shifts, axis=0)
    out = np.empty((2,) + f.shape[1:], np.float32)
    out[0], out[1] = dx, dy
    return out


def estimate_motion(frames: np.ndarray) -> np.ndarray:
    """float32 ``(2, H, W)`` motion (dx, dy) in pixels/frame; ValueError if fewer than 2 frames (STORM-06)."""
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[0] < 2:
        raise ValueError(f"need frames (T >= 2, H, W), got {frames.shape}")
    f = frames.astype(np.float32)
    if PYSTEPS_AVAILABLE and f.max() > f.min():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = np.asarray(_dense_lk(f), dtype=np.float32)
            if m.shape == (2,) + f.shape[1:] and np.isfinite(m).all():
                return m
        except Exception:  # LK finds no features (e.g. blank frames) or OpenCV is missing
            pass
    if f.max() == f.min():
        return np.zeros((2,) + f.shape[1:], np.float32)
    return _fallback_motion(f)


def extrapolate(frame: np.ndarray, motion: np.ndarray, n_steps: int) -> np.ndarray:
    """Backward semi-Lagrangian advection, bilinear, 0 outside: float32 ``(n_steps, H, W)`` (STORM-06)."""
    frame = np.asarray(frame, dtype=np.float32)
    h, w = frame.shape
    rows, cols = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = np.asarray(motion[0], np.float32), np.asarray(motion[1], np.float32)
    out = np.empty((n_steps, h, w), np.float32)
    for k in range(1, n_steps + 1):
        sr, sc = rows - k * dy, cols - k * dx
        vals = ndimage.map_coordinates(frame, [sr, sc], order=1, mode="nearest")
        inside = (sr >= 0) & (sr <= h - 1) & (sc >= 0) & (sc <= w - 1)
        out[k - 1] = np.where(inside, vals, 0.0)
    return out


def extrapolation_nowcast(frames: np.ndarray, n_out: int = 12) -> np.ndarray:
    """``extrapolate(frames[-1], estimate_motion(frames), n_out)`` clipped >= 0 (STORM-06)."""
    out = extrapolate(_last(frames), estimate_motion(frames), n_out)
    return np.clip(out, 0.0, None).astype(np.float32)


def motion_to_mps(motion_px: np.ndarray, pixel_m: float = 2000.0, frame_s: float = 300.0) -> np.ndarray:
    """Image-axis px/frame -> map-axis (u east, v north) m/s; ``v = -dy * pixel_m / frame_s`` (STORM-09)."""
    m = np.asarray(motion_px, dtype=np.float32)
    k = pixel_m / frame_s
    return np.stack([m[0] * k, -m[1] * k]).astype(np.float32)


def motion_at(motion_mps: np.ndarray, row: int, col: int) -> tuple[float, float]:
    """``(u, v)`` at one pixel as Python floats; IndexError out of range (STORM-09)."""
    h, w = motion_mps.shape[-2:]
    if not (0 <= row < h and 0 <= col < w):
        raise IndexError((row, col))
    return float(motion_mps[0, row, col]), float(motion_mps[1, row, col])
