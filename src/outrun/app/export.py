"""Static recruiter demo export (spec 14: APP-13), used by ``outrun export-static`` -> docs/ -> GitHub Pages."""

from __future__ import annotations

from pathlib import Path

MANIFEST_NAME = ".outrun-static.json"


def export_static(scenarios_dir: Path, reports_dir: Path, out_dir: Path) -> list[str]:
    """Write index.html, static/**, config.json, .nojekyll, data/scenarios.json, data/metrics.json, per-scenario
    overlays and results, and the ``.outrun-static.json`` manifest under ``out_dir``. Returns the sorted relative
    posix paths written (manifest excluded).

    Never touches files it does not own: raises ``FileExistsError`` (writing nothing) if a target exists and is not
    in the previous manifest; removes only previously-owned files that are no longer targeted. Idempotent. (APP-13)
    """
    raise NotImplementedError
