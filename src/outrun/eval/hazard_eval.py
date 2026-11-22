"""Contract between the one-shot evaluation driver (T24) and the per-hazard evaluators.

Each hazard has ``outrun.hazards.<hazard>.final_eval.evaluate(split, n_resamples=1000) -> HazardEval``. ``split``
is ``"val"`` (dry run while developing; never touches test) or ``"test"`` (the one-shot run: it must return the
official test split under ``results["test"]`` and the quarantined case hold-out under
``results["case_holdout"]``, plus ``"test_wsts2021"`` for wildfire). With ``split="val"`` the same code path runs
on the val ids and the key is ``"val"``.

Frozen artefacts only: models, calibrators, thresholds and filters are loaded from ``models/`` and the val-fit
files; nothing is refit on val/test and nothing is tuned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from outrun.eval.report import ModelMetrics

__all__ = ["HazardEval"]


@dataclass
class HazardEval:
    hazard: str  # one of eval.report.HAZARDS
    primary_metric: str
    higher_is_better: bool
    results: dict[str, list[ModelMetrics]]  # split name -> models (dl, baseline, reference rows)
    ablations: dict[str, list[ModelMetrics]] = field(default_factory=dict)  # ablation name -> models
    dl_models: list[str] = field(default_factory=list)  # model names to record in the test ledger
    notes: list[str] = field(default_factory=list)  # honest caveats for NOTES.md / metrics.md
    figures: dict[str, Any] = field(
        default_factory=dict
    )  # JSON-able curve data (reliability bins, per-lead CSI)
