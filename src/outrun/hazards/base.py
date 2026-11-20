"""Adapter registry, adapter conformance check and the shared training loop (specs/02-grid-risk.md, RISK-13..16).

The training loop is a plain PyTorch loop: timed epoch, checkpoint every epoch with resume, early stopping on the
validation loss only (never test data).
"""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
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
        self._factories: dict[Hazard, Callable[[], HazardAdapter]] = {}
        self._instances: dict[Hazard, HazardAdapter] = {}

    def register(
        self, hazard: Hazard, factory: Callable[[], HazardAdapter], *, replace: bool = False
    ) -> None:
        """Duplicate registration without ``replace=True`` raises ValueError."""
        hazard = _to_hazard(hazard)
        if hazard in self._factories and not replace:
            raise ValueError(f"an adapter for {hazard.value} is already registered (use replace=True)")
        self._factories[hazard] = factory
        self._instances.pop(hazard, None)  # a replaced factory must build a fresh instance

    def get(self, hazard: Hazard | str) -> HazardAdapter:
        """Cached adapter; KeyError if unknown/unregistered; ValueError if ``adapter.hazard`` != key."""
        hazard = _to_hazard(hazard)
        if hazard not in self._factories:
            raise KeyError(f"no adapter registered for {hazard.value}")
        if hazard not in self._instances:
            adapter = self._factories[hazard]()
            if getattr(adapter, "hazard", None) is not hazard:
                raise ValueError(
                    f"adapter registered for {hazard.value} reports hazard {getattr(adapter, 'hazard', None)!r}"
                )
            self._instances[hazard] = adapter
        return self._instances[hazard]

    def hazards(self) -> list[Hazard]:
        """Registered hazards in Hazard order."""
        return [h for h in Hazard if h in self._factories]


def _to_hazard(hazard: Hazard | str) -> Hazard:
    """``Hazard`` or its string value; anything else is a KeyError (it can never be registered)."""
    if isinstance(hazard, Hazard):
        return hazard
    try:
        return Hazard(hazard)
    except ValueError as exc:
        raise KeyError(f"unknown hazard {hazard!r}") from exc


REGISTRY: AdapterRegistry = AdapterRegistry()


def register_adapter(hazard: Hazard, *, replace: bool = False) -> Callable[[type], type]:
    """Class decorator: registers the class as ``hazard``'s factory in ``REGISTRY``; returns it unchanged (RISK-13)."""

    def decorator(cls: type) -> type:
        REGISTRY.register(hazard, cls, replace=replace)  # module global looked up at call time
        return cls

    return decorator


def get_adapter(hazard: Hazard | str) -> HazardAdapter:
    """``REGISTRY.get(hazard)`` (RISK-13)."""
    return REGISTRY.get(hazard)


def load_default_adapters() -> AdapterRegistry:
    """Import the six ``hazards/<h>/adapter.py`` modules so each registers its zero-argument ``default_adapter``
    into ``REGISTRY``; return ``REGISTRY``. Idempotent; loads no data or model (RISK-13)."""
    for hazard in Hazard:
        # Python imports a module only once, so a second call registers nothing new.
        importlib.import_module(f"outrun.hazards.{hazard.value}.adapter")
    return REGISTRY


_EXPECTED_KIND = {"baseline": "baseline", "predict": "forecast", "observed": "observed"}


def validate_adapter_output(adapter: HazardAdapter, scn: Scenario) -> dict[str, RiskLayer | None]:
    """Call baseline/predict/observed and check hazard, kind, grid coords and times (RISK-13).

    Returns ``{"baseline": ..., "predict": ..., "observed": ...}``; raises ValueError on any violation.
    """
    window = scn.grid_window()
    results: dict[str, RiskLayer | None] = {}
    for method, kind in _EXPECTED_KIND.items():
        layer = getattr(adapter, method)(scn)
        results[method] = layer
        if layer is None and method == "observed":
            continue  # no ground truth for this scenario is allowed
        where = f"{type(adapter).__name__}.{method}"
        if not isinstance(layer, RiskLayer):
            raise ValueError(f"{where} returned {type(layer).__name__}, not a RiskLayer")
        if layer.hazard is not adapter.hazard:
            raise ValueError(f"{where} returned hazard {layer.hazard.value}, expected {adapter.hazard.value}")
        if layer.kind != kind:
            raise ValueError(f"{where} returned kind {layer.kind!r}, expected {kind!r}")
        for name, centres in (("y", window.y_centers()), ("x", window.x_centers())):
            if name not in layer.prob.coords:
                raise ValueError(f"{where}: prob has no {name!r} coordinate")
            c = np.asarray(layer.prob.coords[name].values, dtype=np.float64)
            if c.shape != centres.shape or not np.allclose(c, centres, rtol=0.0, atol=1e-6):
                raise ValueError(f"{where}: {name!r} coords differ from the scenario window centres")
        if layer.valid_until <= scn.t0:
            raise ValueError(f"{where}: valid_until {layer.valid_until} is not after t0 {scn.t0}")
        if method != "observed" and layer.issued > scn.t0:
            raise ValueError(f"{where}: issued {layer.issued} is after t0 {scn.t0} (would use future data)")
    return results


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
        return self.seconds * epochs


def _check_finite(loss: torch.Tensor, what: str) -> None:
    if not math.isfinite(float(loss.detach())):
        raise FloatingPointError(f"non-finite {what}: {float(loss.detach())}")


def _train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
    grad_clip: float | None = None,
) -> tuple[float, int, int]:
    """One pass over ``loader`` in train mode. Returns (mean loss weighted by batch size, batches, samples)."""
    model.train()
    total, batches, samples = 0.0, 0, 0
    for x, y in loader:
        if max_batches is not None and batches >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        _check_finite(loss, "training loss")  # the fp16/NaN guard: stop instead of training on garbage
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.detach()) * len(x)
        batches += 1
        samples += len(x)
    return (total / samples if samples else float("nan")), batches, samples


def _evaluate(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> float:
    """Validation loss (eval mode, no gradients), batch losses weighted by batch size."""
    model.eval()
    total, samples = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            _check_finite(loss, "validation loss")
            total += float(loss) * len(x)
            samples += len(x)
    if samples == 0:
        raise ValueError("the validation loader is empty")
    return total / samples


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
    device = torch.device(cfg.device)
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"

    model.to(device)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Training state. Everything needed to continue lives in last.pt.
    start_epoch, best_val, best_epoch, bad_epochs = 1, math.inf, 0, 0
    history: list[dict[str, float]] = []
    if cfg.resume and last_path.exists():
        state = load_checkpoint(last_path, model, optimizer)
        start_epoch = int(state["epoch"]) + 1
        best_val, best_epoch = float(state["best_val_loss"]), int(state["best_epoch"])
        bad_epochs, history = int(state["bad_epochs"]), list(state["history"])
        torch.set_rng_state(state["torch_rng_state"])
    else:
        torch.manual_seed(cfg.seed)

    stopped_early = start_epoch > 1 and bad_epochs >= cfg.patience  # a resumed run may already have stopped
    first_epoch = cfg.max_epochs + 1 if stopped_early else start_epoch  # nothing left to run if stopped
    for epoch in range(first_epoch, cfg.max_epochs + 1):
        t_start = time.perf_counter()
        train_loss, _, _ = _train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, cfg.max_batches_per_epoch, cfg.grad_clip
        )
        val_loss = _evaluate(model, val_loader, loss_fn, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "seconds": time.perf_counter() - t_start,
            }
        )

        print(f"epoch {epoch}: {history[-1]}", flush=True)
        if val_loss < best_val - cfg.min_delta:  # strict improvement on validation only
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            save_checkpoint(best_path, model, epoch=epoch, val_loss=val_loss)
        else:
            bad_epochs += 1
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch=epoch,
            best_val_loss=best_val,
            best_epoch=best_epoch,
            bad_epochs=bad_epochs,
            history=history,
            torch_rng_state=torch.get_rng_state(),
        )
        if bad_epochs >= cfg.patience and epoch < cfg.max_epochs:
            stopped_early = True
            break

    if best_path.exists():  # end with the best-val weights, not the last ones
        load_checkpoint(best_path, model)
    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val,
        epochs_run=len(history),
        stopped_early=stopped_early,
        history=history,
        best_checkpoint=best_path if best_path.exists() else None,
        last_checkpoint=last_path if last_path.exists() else None,
    )


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
    dev = torch.device(device)
    model.to(dev)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t_start = time.perf_counter()
    _, batches, samples = _train_one_epoch(model, loader, loss_fn, optimizer, dev, max_batches)
    if dev.type == "mps":
        torch.mps.synchronize()  # GPU work is asynchronous: wait for it before stopping the clock
    seconds = time.perf_counter() - t_start
    return EpochTiming(seconds=seconds, batches=batches, samples=samples)


def save_checkpoint(
    path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, **extra: Any
) -> None:
    """Save model (and optimizer) state plus extra entries (RISK-15)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": model.state_dict(), "extra": extra}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never leaves a half-written checkpoint


def load_checkpoint(
    path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    """Restore model (and optimizer) state in place; return the extra entries (RISK-15)."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return dict(payload.get("extra", {}))


def select_device(prefer: str = "mps") -> torch.device:
    """``mps`` if preferred and available, else ``cpu``; other values -> ValueError (RISK-16)."""
    if prefer not in ("mps", "cpu"):
        raise ValueError(f"prefer must be 'mps' or 'cpu', got {prefer!r}")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
