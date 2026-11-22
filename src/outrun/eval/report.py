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

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

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


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    return path


def _ci_dict(ci: CI | Mapping[str, Any]) -> dict[str, float]:
    return ci.to_dict() if isinstance(ci, CI) else dict(ci)


def format_ci(ci: CI | Mapping[str, Any], digits: int = 3) -> str:
    """EVAL-16. ``"0.512 [0.480, 0.540]"``; ``"n/a"`` if the estimate is NaN; ``"0.512 [n/a]"`` if a bound is."""
    d = _ci_dict(ci)
    est, low, high = float(d["estimate"]), float(d["low"]), float(d["high"])
    if math.isnan(est):
        return "n/a"
    if math.isnan(low) or math.isnan(high):
        return f"{est:.{digits}f} [n/a]"
    return f"{est:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _split_ok(name: str) -> bool:
    return name in ("val", "test", "case_holdout") or (name.startswith("test_") and len(name) > 5)


def _models_dict(models: Sequence[ModelMetrics]) -> dict[str, Any]:
    return {
        m.name: {"kind": m.kind, "metrics": {k: _ci_dict(v) for k, v in m.metrics.items()}} for m in models
    }


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
    if hazard not in HAZARDS:
        raise ValueError(f"unknown hazard {hazard!r}")
    if not git_sha:
        raise ValueError("git_sha must not be empty")
    if "test" not in results:
        raise ValueError("results must contain a 'test' split")
    for split, models in results.items():
        if not _split_ok(split):
            raise ValueError(f"invalid split name {split!r}")
        kinds = {m.kind for m in models}
        if "dl" not in kinds or "baseline" not in kinds:
            raise ValueError(f"split {split!r} needs at least one dl and one baseline model")
        names = [m.name for m in models]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate model names in split {split!r}")
        for m in models:
            if primary_metric not in m.metrics:
                raise ValueError(
                    f"model {m.name!r} in split {split!r} lacks primary metric {primary_metric!r}"
                )
    obj = {
        "schema": SCHEMA_VERSION,
        "hazard": hazard,
        "git_sha": git_sha,
        "created_utc": _utc_now(),
        "primary_metric": primary_metric,
        "higher_is_better": bool(higher_is_better),
        "n_resamples": int(n_resamples),
        "splits": {split: _models_dict(models) for split, models in results.items()},
        "ablations": {name: _models_dict(models) for name, models in (ablations or {}).items()},
    }
    return _write_json(Path(out_dir) / f"{hazard}.json", obj)


def load_model_metrics(path: str | Path) -> dict[str, Any]:
    """EVAL-17. Read a ``<hazard>.json`` file back as a dict (schema in the spec)."""
    return json.loads(Path(path).read_text())


def write_routing_metrics(
    out_dir: str | Path, scenario_id: str, result: RoutingEvalResult, git_sha: str, primary: bool = False
) -> Path:
    """EVAL-18. Write ``<out_dir>/routing_<scenario_id>.json``; returns its path."""
    if not git_sha:
        raise ValueError("git_sha must not be empty")
    obj = {
        "schema": SCHEMA_VERSION,
        "scenario": scenario_id,
        "primary": bool(primary),
        "git_sha": git_sha,
        "created_utc": _utc_now(),
        "n_origins": int(result.n_origins),
        "n_paired": int(result.n_paired),
        "planners": {str(k.value): asdict(v) for k, v in result.summaries.items()},
        "gap_closed": result.gap_closed.to_dict(),
        "c_minus_b": result.c_minus_b.to_dict(),
        "c_minus_a_prime": result.c_minus_a_prime.to_dict(),
    }
    return _write_json(Path(out_dir) / f"routing_{scenario_id}.json", obj)


_SHELTER_REQUIRED = ("mcf", "nearest", "greedy")
_EVAC_REQUIRED = ("sa", "msa", "shortest", "safest", "sa_lambda0")
_EMS_REQUIRED = ("hungarian", "reroute_baseline", "pareto")


def _plain(x: Any) -> Any:
    """Convert dataclasses / numpy scalars to plain JSON types (NaN kept)."""
    if is_dataclass(x) and not isinstance(x, type):
        return _plain(asdict(x))
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


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
    if not git_sha:
        raise ValueError("git_sha must not be empty")
    seeds = [int(x) for x in seeds]
    if not seeds:
        raise ValueError("seeds must not be empty")
    if not math.isfinite(float(lower_bound)):
        raise ValueError("lower_bound must be finite")
    for name, have, need in (("shelters", shelters, _SHELTER_REQUIRED), ("evacuation", evacuation, _EVAC_REQUIRED),
                             ("ems", ems, _EMS_REQUIRED)):  # fmt: skip
        missing = [m for m in need if m not in have]
        if missing:
            raise ValueError(f"{name} is missing {missing}")
    for method, rows in evacuation.items():
        if len(rows) != len(seeds):
            raise ValueError(f"evacuation[{method!r}] has {len(rows)} entries for {len(seeds)} seeds")
    obj = {
        "schema": SCHEMA_VERSION,
        "scenario": scenario_id,
        "primary": bool(primary),
        "git_sha": git_sha,
        "created_utc": _utc_now(),
        "shelters": {m: _plain(v) for m, v in shelters.items()},
        "evacuation": {
            "seeds": seeds,
            "lower_bound": float(lower_bound),
            "methods": {m: [_plain(r) for r in rows] for m, rows in evacuation.items()},
        },
        "ems": _plain(ems),
    }
    return _write_json(Path(out_dir) / f"group_{scenario_id}.json", obj)


def load_group_metrics(path: str | Path) -> dict[str, Any]:
    """EVAL-22. Read a group metrics file back (NaN-aware)."""
    return json.loads(Path(path).read_text())


def _best(models: dict[str, Any], kind: str, primary: str, higher: bool) -> tuple[str, float] | None:
    cands = [(n, float(m["metrics"][primary]["estimate"])) for n, m in models.items()
             if m["kind"] == kind and primary in m["metrics"]]  # fmt: skip
    cands = [c for c in cands if not math.isnan(c[1])]
    if not cands:
        return None
    return max(cands, key=lambda c: c[1]) if higher else min(cands, key=lambda c: c[1])


def _hazard_section(d: dict[str, Any]) -> list[str]:
    primary, higher = d["primary_metric"], d["higher_is_better"]
    out = [f"## {d['hazard']}", "", f"git sha: {d['git_sha']}", ""]
    splits = d["splits"]
    order = (["test"] if "test" in splits else []) + sorted(s for s in splits if s != "test")
    for split in order:
        models = splits[split]
        cols = [primary] + sorted({k for m in models.values() for k in m["metrics"]} - {primary})
        out += [
            f"### {split}",
            "",
            "| model | kind | " + " | ".join(cols) + " |",
            "|---|---|" + "---|" * len(cols),
        ]
        for name, m in models.items():
            cells = [format_ci(m["metrics"][c]) if c in m["metrics"] else "n/a" for c in cols]
            out.append(f"| {name} | {m['kind']} | " + " | ".join(cells) + " |")
        out.append("")
        dl = _best(models, "dl", primary, higher)
        bl = _best(models, "baseline", primary, higher)
        dl_all = [n for n, m in models.items() if m["kind"] == "dl"]
        if bl is not None and (dl is not None or dl_all):
            if dl is None:  # all DL estimates are NaN
                dl_name, dl_x = dl_all[0], float("nan")
                beats = False
            else:
                dl_name, dl_x = dl
                beats = dl_x > bl[1] if higher else dl_x < bl[1]
            verb = "beats" if beats else "does not beat"
            out += [f"**{split}:** DL model `{dl_name}` {verb} the best baseline `{bl[0]}` on {primary}: "
                    f"{dl_x:.3f} vs {bl[1]:.3f}.", ""]  # fmt: skip
    if d.get("ablations"):
        out += ["### ablations", ""]
        for abl, models in d["ablations"].items():
            for name, m in models.items():
                cell = format_ci(m["metrics"][primary]) if primary in m["metrics"] else "n/a"
                out.append(f"- {abl}: `{name}` {primary} {cell}")
        out.append("")
    return out


def _routing_section(d: dict[str, Any]) -> list[str]:
    cols = ["n_routed", "n_failed", "pct_intersect", "mean_exposure_risk_min", "mean_travel_time_ratio",
            "blocked_crossings", "mean_predicted_exposure"]  # fmt: skip

    def fmt(v: Any) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, int):
            return str(v)
        return "n/a" if math.isnan(v) else f"{v:.3f}"

    out = [f"## Routing: {d['scenario']}", "", f"git sha: {d['git_sha']}", "",
           f"origins: {d['n_origins']}, paired: {d['n_paired']}", "",
           "| planner | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]  # fmt: skip
    for k in ("a", "a_prime", "b", "c", "d"):
        s = d["planners"][k]
        out.append(f"| {k} | " + " | ".join(fmt(s[c]) for c in cols) + " |")
    g = d["gap_closed"]
    if math.isnan(g["estimate"]):
        head = "DL planning (c) closes n/a of the observed-exposure gap between (a) and (d)."
    else:
        head = (f"DL planning (c) closes {100 * g['estimate']:.1f}% [{100 * g['low']:.1f}%, {100 * g['high']:.1f}%] "
                "of the observed-exposure gap between (a) and (d).")  # fmt: skip
    out += ["", head, "",
            f"c minus b (mean observed exposure, negative = DL better): {format_ci(d['c_minus_b'])}", "",
            f"c minus a': {format_ci(d['c_minus_a_prime'])}", ""]  # fmt: skip
    return out


def _mean_over_seeds(rows: Sequence[Mapping[str, float]], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    return float(np.mean(vals)) if vals else float("nan")


def _group_section(d: dict[str, Any]) -> list[str]:
    out = [f"## Group: {d['scenario']}", "", f"git sha: {d['git_sha']}", "", "### Shelters", "",
           "| method | overflow_persons | unserved_persons | max_minutes |", "|---|---|---|---|"]  # fmt: skip
    shel = d["shelters"]
    for m in [*_SHELTER_REQUIRED, *sorted(set(shel) - set(_SHELTER_REQUIRED))]:
        s = shel[m]
        out.append(
            f"| {m} | {round(s['overflow_persons'])} | {round(s['unserved_persons'])} | {s['max_minutes']:.3f} |"
        )
    evac = d["evacuation"]
    cols = ["vehicle_hours", "exposure_veh_h", "clearance_min", "max_vc", "gap_to_lb_pct"]
    out += [
        "",
        "### Evacuation",
        "",
        "| method | seeds | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for m, rows in evac["methods"].items():
        out.append(
            f"| {m} | {len(rows)} | " + " | ".join(f"{_mean_over_seeds(rows, c):.3f}" for c in cols) + " |"
        )
    out += ["", f"lower bound: {evac['lower_bound']:g}", "", "### Emergency vehicles", "",
            "| dispatch | mean_eta_min | mean_detour_min | pct_hit_block |", "|---|---|---|---|"]  # fmt: skip
    for m in ("hungarian", "reroute_baseline"):
        e = d["ems"][m]
        out.append(
            f"| {m} | {e['mean_eta_min']:.3f} | {e['mean_detour_min']:.3f} | {e['pct_hit_block']:.3f} |"
        )
    out += ["", f"Pareto points: {len(d['ems']['pareto'])}", ""]
    return out


def write_metrics_md(metrics_dir: str | Path, out_path: str | Path) -> Path:
    """EVAL-19. Render every metrics JSON in ``metrics_dir`` into one Markdown report at ``out_path``."""
    mdir = Path(metrics_dir)
    if not mdir.is_dir():
        raise FileNotFoundError(mdir)
    lines = ["# Outrun metrics", ""]
    for hz in HAZARDS:
        if (mdir / f"{hz}.json").exists():
            lines += _hazard_section(load_model_metrics(mdir / f"{hz}.json"))
    for p in sorted(mdir.glob("routing_*.json")):
        lines += _routing_section(json.loads(p.read_text()))
    for p in sorted(mdir.glob("group_*.json")):
        lines += _group_section(load_group_metrics(p))
    lat = mdir / "latency.json"
    if lat.exists():
        lines += [
            "## Latency",
            "",
            "| name | n | median_s | p95_s | max_s | git sha |",
            "|---|---|---|---|---|---|",
        ]
        for name, v in json.loads(lat.read_text()).items():
            lines.append(
                f"| {name} | {v['n']} | {v['median_s']:.3f} | {v['p95_s']:.3f} | {v['max_s']:.3f} | {v['git_sha']} |"
            )
        lines.append("")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


class RunLedger:
    """EVAL-20. Append-only JSON ledger of held-out (test) evaluations: one run per (model, split)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> list[dict[str, Any]]:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def record(
        self, model: str, git_sha: str, split: str = "test", rerun_reason: str | None = None
    ) -> dict[str, Any]:
        """Append ``{"model", "split", "git_sha", "utc", "rerun_reason"}``; ``HeldOutReuseError`` on a repeat
        (model, split) without a non-empty ``rerun_reason``; ``ValueError`` on an empty git sha."""
        if not git_sha:
            raise ValueError("git_sha must not be empty")
        entries = self._load()
        if any(e["model"] == model and e["split"] == split for e in entries) and not rerun_reason:
            raise HeldOutReuseError(f"{model!r} was already evaluated on {split!r}; give a rerun_reason")
        entry = {
            "model": model,
            "split": split,
            "git_sha": git_sha,
            "utc": _utc_now(),
            "rerun_reason": rerun_reason,
        }
        entries.append(entry)
        _write_json(self.path, entries)
        return entry

    def runs(self, model: str | None = None, split: str | None = None) -> list[dict[str, Any]]:
        """Recorded entries in record order, optionally filtered."""
        return [e for e in self._load()
                if (model is None or e["model"] == model) and (split is None or e["split"] == split)]  # fmt: skip


def time_calls(fn: Callable[[Any], Any], inputs: Sequence[Any], warmup: int = 3) -> LatencyStats:
    """EVAL-21. Time ``fn(x)`` for every ``x`` in ``inputs`` after ``warmup`` untimed calls."""
    if len(inputs) == 0:
        raise ValueError("inputs must not be empty")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    times = []
    for x in inputs:
        t = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - t)
    return LatencyStats(
        len(times), float(np.median(times)), float(np.percentile(times, 95)), float(np.max(times))
    )


def write_latency(out_dir: str | Path, name: str, stats: LatencyStats, git_sha: str) -> Path:
    """EVAL-21. Merge ``{name: {"n", "median_s", "p95_s", "max_s", "git_sha", "created_utc"}}`` into
    ``<out_dir>/latency.json`` (other names kept); returns its path."""
    path = Path(out_dir) / "latency.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[name] = {"n": stats.n, "median_s": stats.median_s, "p95_s": stats.p95_s, "max_s": stats.max_s,
                  "git_sha": git_sha, "created_utc": _utc_now()}  # fmt: skip
    return _write_json(path, data)
