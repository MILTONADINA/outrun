"""Persistence and optical-flow extrapolation baselines (STORM-06) and motion export (STORM-09).

pySTEPS (Lucas-Kanade + semi-Lagrangian) is used when installed; otherwise an own numpy/scipy fallback.
Image axes: dx + toward increasing column, dy + toward increasing row (rows run north to south).
"""

from __future__ import annotations

import numpy as np

try:  # optional backend
    import pysteps  # noqa: F401

    PYSTEPS_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    PYSTEPS_AVAILABLE = False


def persistence(frames: np.ndarray, n_out: int = 12) -> np.ndarray:
    """float32 ``(n_out, H, W)``, every slice = ``frames[-1]``. ValueError if no frames (STORM-06)."""
    raise NotImplementedError


def persistence_mask(frames: np.ndarray, threshold: float = 133, n_out: int = 12) -> np.ndarray:
    """float32 ``(n_out, H, W)`` of ``frames[-1] >= threshold`` (STORM-06)."""
    raise NotImplementedError


def estimate_motion(frames: np.ndarray) -> np.ndarray:
    """float32 ``(2, H, W)`` motion (dx, dy) in pixels/frame; ValueError if fewer than 2 frames (STORM-06)."""
    raise NotImplementedError


def extrapolate(frame: np.ndarray, motion: np.ndarray, n_steps: int) -> np.ndarray:
    """Backward semi-Lagrangian advection, bilinear, 0 outside: float32 ``(n_steps, H, W)`` (STORM-06)."""
    raise NotImplementedError


def extrapolation_nowcast(frames: np.ndarray, n_out: int = 12) -> np.ndarray:
    """``extrapolate(frames[-1], estimate_motion(frames), n_out)`` clipped >= 0 (STORM-06)."""
    raise NotImplementedError


def motion_to_mps(motion_px: np.ndarray, pixel_m: float = 2000.0, frame_s: float = 300.0) -> np.ndarray:
    """Image-axis px/frame -> map-axis (u east, v north) m/s; ``v = -dy * pixel_m / frame_s`` (STORM-09)."""
    raise NotImplementedError


def motion_at(motion_mps: np.ndarray, row: int, col: int) -> tuple[float, float]:
    """``(u, v)`` at one pixel as Python floats; IndexError out of range (STORM-09)."""
    raise NotImplementedError
