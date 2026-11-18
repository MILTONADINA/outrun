"""Adapter registry, adapter conformance check and the shared training loop (specs/02-grid-risk.md, RISK-13..16).

The training loop is a plain PyTorch loop: timed epoch, checkpoint every epoch with resume, early stopping on the
validation loss only (never test data).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from outrun.core.risk import Hazard, HazardAdapter, RiskLayer
from outrun.core.scenario import Scenario

__all__ = [
    "HazardAdapter",
    "AdapterRegistry",
    "REGISTRY",
    "register_adapter",
    "get_adapter",
    "load_default_adapters",
    "validate_adapter_output",
    "TrainConfig",
    "TrainResult",
    "EpochTiming",
    "train_model",
    "time_one_epoch",
    "save_checkpoint",
    "load_checkpoint",
    "select_device",
]


class AdapterRegistry:
    """Maps each Hazard to a zero-argument adapter factory; instances are created lazily and cached (RISK-13)."""

    def __init__(self) -> None:
        raise NotImplementedError

    def register(
        self, hazard: Hazard, factory: Callable[[], HazardAdapter], *, replace: bool = False
    ) -> None:
        """Duplicate registration without ``replace=True`` raises ValueError."""
        raise NotImplementedError

    def get(self, hazard: Hazard | str) -> HazardAdapter:
        """Cached adapter; KeyError if unknown/unregistered; ValueError if ``adapter.hazard`` != key."""
        raise NotImplementedError

    def hazards(self) -> list[Hazard]:
        """Registered hazards in Hazard order."""
        raise NotImplementedError


REGISTRY: AdapterRegistry  # module-level registry, created at import time by the implementation


def register_adapter(hazard: Hazard, *, replace: bool = False) -> Callable[[type], type]:
    """Class decorator: registers the class as ``hazard``'s factory in ``REGISTRY``; returns it unchanged (RISK-13)."""
    raise NotImplementedError


def get_adapter(hazard: Hazard | str) -> HazardAdapter:
    """``REGISTRY.get(hazard)`` (RISK-13)."""
    raise NotImplementedError


def load_default_adapters() -> AdapterRegistry:
    """Import the six ``hazards/<h>/adapter.py`` modules so each registers its zero-argument ``default_adapter``
    into ``REGISTRY``; return ``REGISTRY``. Idempotent; loads no data or model (RISK-13)."""
    raise NotImplementedError


def validate_adapter_output(adapter: HazardAdapter, scn: Scenario) -> dict[str, RiskLayer | None]:
    """Call baseline/predict/observed and check hazard, kind, grid coords and times (RISK-13).

    Returns ``{"baseline": ..., "predict": ..., "observed": ...}``; raises ValueError on any violation.
    """
    raise NotImplementedError


@dataclass
class TrainConfig:
    """Training-loop settings (RISK-14..15)."""

    checkpoint_dir: Path
    max_epochs: int = 20
    patience: int = 3
    min_delta: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    resume: bool = False
    seed: int = 0
    max_batches_per_epoch: int | None = None
    grad_clip: float | None = None


@dataclass
class TrainResult:
    """Outcome of ``train_model`` (RISK-14). ``history``: one dict per epoch (epoch, train_loss, val_loss, seconds)."""

    best_epoch: int
    best_val_loss: float
    epochs_run: int
    stopped_early: bool
    history: list[dict[str, float]] = field(default_factory=list)
    best_checkpoint: Path | None = None
    last_checkpoint: Path | None = None


@dataclass(frozen=True)
class EpochTiming:
    """Result of ``time_one_epoch`` (RISK-16)."""

    seconds: float
    batches: int
    samples: int

    def estimate_total_s(self, epochs: int) -> float:
        """``seconds * epochs``."""
        raise NotImplementedError


def train_model(
    model: torch.nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer | None = None,
) -> TrainResult:
    """Train with early stopping on val loss, checkpoint every epoch, optional resume; best weights restored.

    RISK-14 (loop, early stop, NaN guard -> FloatingPointError, empty val -> ValueError), RISK-15 (checkpoint/resume).
    """
    raise NotImplementedError


def time_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    device: str = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> EpochTiming:
    """Train one epoch (or ``max_batches``) and time it; non-finite loss -> FloatingPointError (RISK-16)."""
    raise NotImplementedError


def save_checkpoint(
    path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, **extra: Any
) -> None:
    """Save model (and optimizer) state plus extra entries (RISK-15)."""
    raise NotImplementedError


def load_checkpoint(
    path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    """Restore model (and optimizer) state in place; return the extra entries (RISK-15)."""
    raise NotImplementedError


def select_device(prefer: str = "mps") -> torch.device:
    """``mps`` if preferred and available, else ``cpu``; other values -> ValueError (RISK-16)."""
    raise NotImplementedError
