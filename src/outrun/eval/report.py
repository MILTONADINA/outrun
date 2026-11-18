"""Report writer, one-shot test ledger and latency timing (specs/15-evaluation.md, EVAL-16..EVAL-21).

Files written under ``reports/`` (DESIGN 2.1):

* ``reports/metrics/<hazard>.json``          per-model metrics with CIs (EVAL-17)
* ``reports/metrics/routing_<scenario>.json`` routing evaluation (EVAL-18)
* ``reports/metrics/latency.json``           query/run timings (EVAL-21)
* ``reports/metrics.md``                     human report with honest DL-vs-baseline verdicts (EVAL-19)
* ``reports/test_ledger.json``               one entry per test-split evaluation, with git sha (EVAL-20)

The acceptance suite (specs/16-acceptance.md) reads exactly these files.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from outrun.eval.bootstrap import CI
from outrun.eval.routing_eval import RoutingEvalResult

__all__ = [
    "HAZARDS",
    "HeldOutReuseError",
    "LatencyStats",
    "ModelMetrics",
    "RunLedger",
    "format_ci",
    "load_group_metrics",
    "load_model_metrics",
    "time_calls",
    "write_group_metrics",
    "write_latency",
    "write_metrics_md",
    "write_model_metrics",
    "write_routing_metrics",
]

#: Hazard names used as file stems and JSON values (``core.risk.Hazard`` member names, lower-cased).
HAZARDS: tuple[str, ...] = ("storm", "tornado", "flood", "wildfire", "ice", "smoke")

SCHEMA_VERSION = 1


class HeldOutReuseError(RuntimeError):
    """A model was evaluated on a held-out split a second time without a recorded reason (EVAL-20)."""


@dataclass(frozen=True)
class ModelMetrics:
    """Metrics of one model (or baseline) on one split; every value is a bootstrap ``CI``."""

    name: str  # e.g. "storm.unet_vil_ir", "persistence", "pysteps"
    kind: Literal["dl", "baseline", "reference"]  # "reference" = published number, never the best baseline
    metrics: dict[str, CI] = field(default_factory=dict)


@dataclass(frozen=True)
class LatencyStats:
    """Wall-clock timings of ``n`` timed calls, in seconds (EVAL-21)."""

    n: int
    median_s: float
    p95_s: float
    max_s: float


def format_ci(ci: CI | Mapping[str, Any], digits: int = 3) -> str:
    """EVAL-16. ``"0.512 [0.480, 0.540]"``; ``"n/a"`` if the estimate is NaN; ``"0.512 [n/a]"`` if a bound is."""
    raise NotImplementedError


def write_model_metrics(
    out_dir: str | Path,
    hazard: str,
    results: Mapping[str, Sequence[ModelMetrics]],
    primary_metric: str,
    higher_is_better: bool,
    git_sha: str,
    n_resamples: int = 1000,
    ablations: Mapping[str, Sequence[ModelMetrics]] | None = None,
) -> Path:
    """EVAL-17. Validate and write ``<out_dir>/<hazard>.json``; returns its path."""
    raise NotImplementedError


def load_model_metrics(path: str | Path) -> dict[str, Any]:
    """EVAL-17. Read a ``<hazard>.json`` file back as a dict (schema in the spec)."""
    raise NotImplementedError


def write_routing_metrics(
    out_dir: str | Path, scenario_id: str, result: RoutingEvalResult, git_sha: str, primary: bool = False
) -> Path:
    """EVAL-18. Write ``<out_dir>/routing_<scenario_id>.json``; returns its path."""
    raise NotImplementedError


def write_group_metrics(
    out_dir: str | Path,
    scenario_id: str,
    *,
    shelters: Mapping[str, Any],
    evacuation: Mapping[str, Sequence[Mapping[str, float]]],
    lower_bound: float,
    ems: Mapping[str, Any],
    git_sha: str,
    seeds: Sequence[int],
    primary: bool = False,
) -> Path:
    """EVAL-22. ``group_<scenario_id>.json`` with shelter (mcf/nearest/greedy), evacuation (sa/msa/shortest/safest/
    sa_lambda0, one EVAC-10 dict per seed, lower bound) and EMS (hungarian/reroute_baseline/pareto) metrics."""
    raise NotImplementedError


def load_group_metrics(path: str | Path) -> dict[str, Any]:
    """EVAL-22. Read a group metrics file back (NaN-aware)."""
    raise NotImplementedError


def write_metrics_md(metrics_dir: str | Path, out_path: str | Path) -> Path:
    """EVAL-19. Render every metrics JSON in ``metrics_dir`` into one Markdown report at ``out_path``."""
    raise NotImplementedError


class RunLedger:
    """EVAL-20. Append-only JSON ledger of held-out (test) evaluations: one run per (model, split)."""

    def __init__(self, path: str | Path) -> None:
        raise NotImplementedError

    def record(
        self, model: str, git_sha: str, split: str = "test", rerun_reason: str | None = None
    ) -> dict[str, Any]:
        """Append ``{"model", "split", "git_sha", "utc", "rerun_reason"}``; ``HeldOutReuseError`` on a repeat
        (model, split) without a non-empty ``rerun_reason``; ``ValueError`` on an empty git sha."""
        raise NotImplementedError

    def runs(self, model: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
        """Recorded entries in record order, optionally filtered."""
        raise NotImplementedError


def time_calls(fn: Callable[[Any], Any], inputs: Sequence[Any], warmup: int = 3) -> LatencyStats:
    """EVAL-21. Time ``fn(x)`` for every ``x`` in ``inputs`` after ``warmup`` untimed calls."""
    raise NotImplementedError


def write_latency(out_dir: str | Path, name: str, stats: LatencyStats, git_sha: str) -> Path:
    """EVAL-21. Merge ``{name: {"n", "median_s", "p95_s", "max_s", "git_sha", "created_utc"}}`` into
    ``<out_dir>/latency.json`` (other names kept); returns its path."""
    raise NotImplementedError
