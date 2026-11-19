"""Static recruiter demo export (spec 14: APP-13), used by ``outrun export-static`` -> docs/ -> GitHub Pages."""

from __future__ import annotations

import json
from pathlib import Path

from outrun.app.bundle import list_scenarios, read_metrics

MANIFEST_NAME = ".outrun-static.json"
STATIC_DIR = Path(__file__).parent / "static"


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")


def export_static(scenarios_dir: Path, reports_dir: Path, out_dir: Path) -> list[str]:
    """Write index.html, static/**, config.json, .nojekyll, data/scenarios.json, data/metrics.json, per-scenario
    overlays and results, and the ``.outrun-static.json`` manifest under ``out_dir``. Returns the sorted relative
    posix paths written (manifest excluded).

    Never touches files it does not own: raises ``FileExistsError`` (writing nothing) if a target exists and is not
    in the previous manifest; removes only previously-owned files that are no longer targeted. Idempotent. (APP-13)
    """
    scenarios_dir, out_dir = Path(scenarios_dir), Path(out_dir)
    scenarios = list_scenarios(scenarios_dir)
    plan: dict[str, Path | bytes] = {}  # relative posix path -> source file or literal bytes
    plan["index.html"] = STATIC_DIR / "index.html"
    for f in sorted(STATIC_DIR.rglob("*")):
        rel = f.relative_to(STATIC_DIR)
        if (
            f.is_file()
            and rel.as_posix() != "index.html"
            and "__pycache__" not in rel.parts
            and f.name != ".DS_Store"
        ):
            plan["static/" + rel.as_posix()] = f
    plan["config.json"] = _json_bytes({"mode": "static"})
    plan[".nojekyll"] = b""
    plan["data/scenarios.json"] = _json_bytes(scenarios)
    plan["data/metrics.json"] = _json_bytes(read_metrics(reports_dir))
    for meta in scenarios:
        sid, folder = meta["id"], scenarios_dir / meta["id"]
        for png in sorted((folder / "overlays").rglob("*.png")):
            plan[f"data/scenarios/{sid}/overlay/{png.relative_to(folder / 'overlays').as_posix()}"] = png
        for res in sorted((folder / "results").glob("*.json")):
            plan[f"data/scenarios/{sid}/{res.name}"] = res

    manifest_path = out_dir / MANIFEST_NAME
    previous: set[str] = set()
    if manifest_path.is_file():
        try:
            previous = set(json.loads(manifest_path.read_text(encoding="utf-8")).get("files", []))
        except (OSError, ValueError):
            previous = set()
    for rel in plan:
        if (out_dir / rel).exists() and rel not in previous:
            raise FileExistsError(f"{out_dir / rel} exists and was not written by a previous export")

    for rel in sorted(previous - set(plan)):
        stale = out_dir / rel
        if stale.is_file():
            stale.unlink()
    for rel, src in plan.items():
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src if isinstance(src, bytes) else src.read_bytes())
    written = sorted(plan)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(_json_bytes({"files": written}))
    return written
