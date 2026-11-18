"""Command-line interface: ``outrun fetch|prep|train|eval|scenario|route|serve|export-static|disk``
(spec 14: CLI-01..CLI-08; DESIGN 2.2).

``python -m outrun`` and the ``outrun`` console script both reach ``main`` through ``outrun.__main__``.
Hazard and profile choices are derived from ``outrun.core.risk.Hazard`` / ``outrun.core.policy.Profile``
(``name.lower()``), never duplicated as literals. Heavy modules are imported inside the handlers only.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

COMMANDS: tuple[str, ...] = (
    "fetch",
    "prep",
    "train",
    "eval",
    "scenario",
    "route",
    "serve",
    "export-static",
    "disk",
)

GB = 1e9


@dataclass(frozen=True)
class DiskStatus:
    """Result of ``check_disk`` (CLI-08). Sizes in GB (1e9 bytes)."""

    path: Path
    total_gb: float
    used_gb: float
    free_gb: float
    need_gb: float
    max_used_gb: float
    ok: bool


def build_parser() -> argparse.ArgumentParser:
    """Parser with prog ``outrun``, ``--version`` and the nine sub-commands; options and defaults per CLI-02,
    route argument groups per CLI-05. ``args.command`` is the sub-command name. (CLI-01, CLI-02, CLI-05)
    """
    raise NotImplementedError


def check_disk(path: Path, need_gb: float, max_used_gb: float) -> DiskStatus:
    """Disk check from ``shutil.disk_usage(path)``: ``ok = used + need <= max_used and need <= free``.
    ``ValueError`` if ``need_gb < 0``; ``FileNotFoundError`` if ``path`` is missing. (CLI-08)
    """
    raise NotImplementedError


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download manifest sources (delegates to ``outrun.data.fetch``)."""
    raise NotImplementedError


def cmd_prep(args: argparse.Namespace) -> int:
    """Preprocess one dataset to PROC (delegates to ``outrun.data.<dataset>``)."""
    raise NotImplementedError


def cmd_train(args: argparse.Namespace) -> int:
    """Train one hazard model (delegates to ``outrun.hazards.base`` training loop)."""
    raise NotImplementedError


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate on val (default) or, with ``--final``, the one-time test run (delegates to ``outrun.eval``)."""
    raise NotImplementedError


def cmd_scenario(args: argparse.Namespace) -> int:
    """Precompute scenario bundles (layout in specs/14-app.md) into ``args.out``."""
    raise NotImplementedError


def cmd_route(args: argparse.Namespace) -> int:
    """Plan one safe route on a place or bbox graph (delegates to ``outrun.routing.planner``; ROAD-06)."""
    raise NotImplementedError


def cmd_serve(args: argparse.Namespace) -> int:
    """``outrun.app.server.create_app(scenarios_dir=..., reports_dir=..., demo=args.demo or None)`` then
    ``uvicorn.run(app, host=args.host, port=args.port)``; returns 0. (CLI-06)
    """
    raise NotImplementedError


def cmd_export_static(args: argparse.Namespace) -> int:
    """``outrun.app.export.export_static(scenarios, reports, out)``; prints the file count, returns 0;
    ``FileExistsError`` -> message on stderr, returns 1. (CLI-07)
    """
    raise NotImplementedError


def cmd_disk(args: argparse.Namespace) -> int:
    """Check RAW (HDD limit) and PROC (SSD limit) with ``check_disk``; 0 if both ok else 1. (CLI-08)"""
    raise NotImplementedError


HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "fetch": cmd_fetch,
    "prep": cmd_prep,
    "train": cmd_train,
    "eval": cmd_eval,
    "scenario": cmd_scenario,
    "route": cmd_route,
    "serve": cmd_serve,
    "export-static": cmd_export_static,
    "disk": cmd_disk,
}


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` (default ``sys.argv[1:]``), apply the cross-argument checks (CLI-04 test-split guard,
    CLI-05 bbox order; violations exit with status 2 before any handler runs), then call
    ``HANDLERS[args.command](args)`` (looked up at call time) and return its int. (CLI-03, CLI-04, CLI-05)
    """
    raise NotImplementedError
