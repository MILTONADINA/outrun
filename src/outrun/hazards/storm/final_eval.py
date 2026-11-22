"""One-shot held-out evaluation of the storm hazard (T24, DESIGN 6.1, EVAL-xx).

``evaluate(split)`` returns a ``HazardEval``. ``split="val"`` is the dry run; ``split="test"`` scores the official
test events under ``results["test"]`` and the quarantined case hold-out under ``results["case_holdout"]``, through
the very same code path (only the split part names differ).

Everything is frozen: checkpoints and per-lead isotonic calibrators come from ``models/storm`` and
``models/storm_vil_only``; the baselines (persistence, pySTEPS-style extrapolation) have no trainable part. The
no-data-frame filter is the frozen rule inside ``SevirWindowDataset.from_split``.

Representation (same as the val runs logged in NOTES.md): categorical scores (CSI, FSS, MSE) are computed on the
U-Net's VIL regression head x 255 (clipped to 0..255) against the true future VIL; Brier and ECE use the calibrated
P(VIL >= 133) head. For speed, every metric is computed from per-window sufficient statistics (contingency counts,
FSS sums, squared errors, reliability-bin sums), summed per UTC day. The day-block bootstrap then resamples days
and recomputes the metric from the summed statistics, which is exactly the pooled metric on the resampled days.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter

from outrun.config import get_paths
from outrun.eval.bootstrap import CI, day_block_ci
from outrun.eval.hazard_eval import HazardEval
from outrun.eval.report import ModelMetrics
from outrun.eval.splits import read_split

from .adapter import StormAdapter
from .baselines import extrapolation_nowcast
from .dataset import N_IN, N_OUT, SevirWindowDataset
from .metrics import CSI_THRESHOLDS
from .model import RISK_THRESHOLD, RISK_THRESHOLD_INDEX

PRIMARY_THRESHOLDS = (74, 133)
FSS_THRESHOLD = 133
FSS_WINDOWS = (3, 7)
N_BINS = 10
BATCH = 8
DL_NAME = "storm.unet_vil_ir"
VIL_ONLY_NAME = "storm.unet_vil_only"
PERSISTENCE_NAME = "persistence"
EXTRAPOLATION_NAME = "pysteps_extrapolation"

# split argument -> {result key: part of splits/sevir.json}
_PARTS = {
    "val": {"val": "val"},
    "test": {"test": "test", "case_holdout": "case_holdout"},
}


# ------------------------------------------------------------------ per-window sufficient statistics
def window_stats(
    pred: np.ndarray, truth: np.ndarray, prob: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Per-window statistics of ``(B, 12, H, W)`` VIL forecasts ``pred`` (and probabilities ``prob``) vs ``truth``.

    ``cont`` (B, 12, n_thr, 3) = hits / misses / false alarms; ``fss`` (B, 12, n_win, 2) = sum (pf - po)^2 and
    sum (pf^2 + po^2) of the 133 exceedance fractions; ``sse`` (B, 12); with ``prob``: ``brier`` (B, 12) sums and
    ``bins`` (B, N_BINS, 3) = count / sum p / sum y pooled over leads.
    """
    pred, truth = pred.astype(np.float32), truth.astype(np.float32)
    n = len(pred)
    cont = np.empty((n, N_OUT, len(CSI_THRESHOLDS), 3))
    for k, thr in enumerate(CSI_THRESHOLDS):
        p, o = pred >= thr, truth >= thr
        hits = (p & o).sum(axis=(2, 3))
        cont[:, :, k, 0] = hits
        cont[:, :, k, 1] = o.sum(axis=(2, 3)) - hits
        cont[:, :, k, 2] = p.sum(axis=(2, 3)) - hits
    p133, o133 = (pred >= FSS_THRESHOLD).astype(np.float64), (truth >= FSS_THRESHOLD).astype(np.float64)
    fss = np.empty((n, N_OUT, len(FSS_WINDOWS), 2))
    for k, win in enumerate(FSS_WINDOWS):
        size = (1, 1, win, win)
        pf = uniform_filter(p133, size=size, mode="constant", cval=0.0)
        po = uniform_filter(o133, size=size, mode="constant", cval=0.0)
        fss[:, :, k, 0] = ((pf - po) ** 2).sum(axis=(2, 3))
        fss[:, :, k, 1] = (pf**2 + po**2).sum(axis=(2, 3))
    out = {"cont": cont, "fss": fss, "sse": ((pred - truth) ** 2).sum(axis=(2, 3), dtype=np.float64)}
    if prob is not None:
        y = (truth >= RISK_THRESHOLD).astype(np.float64)
        prob = prob.astype(np.float64)
        out["brier"] = ((prob - y) ** 2).sum(axis=(2, 3))
        idx = np.minimum((prob * N_BINS).astype(np.int64), N_BINS - 1)  # last bin includes 1.0
        bins = np.zeros((n, N_BINS, 3))
        for i in range(n):
            bins[i, :, 0] = np.bincount(idx[i].ravel(), minlength=N_BINS)
            bins[i, :, 1] = np.bincount(idx[i].ravel(), weights=prob[i].ravel(), minlength=N_BINS)
            bins[i, :, 2] = np.bincount(idx[i].ravel(), weights=y[i].ravel(), minlength=N_BINS)
        out["bins"] = bins
    return out


def _by_day(
    stats: dict[str, np.ndarray], window_days: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Sum the per-window statistics inside each UTC day (rows ordered by sorted day)."""
    days, inverse = np.unique(window_days, return_inverse=True)
    out = {}
    for key, arr in stats.items():
        summed = np.zeros((len(days),) + arr.shape[1:])
        np.add.at(summed, inverse, arr)
        out[key] = summed
    n_win = np.bincount(inverse, minlength=len(days)).astype(np.float64)
    out["n_win"] = n_win
    return out, days


# ------------------------------------------------------------------ metrics from summed statistics
def _csi_per_lead(cont: np.ndarray) -> np.ndarray:
    """``(D, 12, n_thr, 3)`` day sums -> CSI ``(12, n_thr)`` of the pooled counts; nan where 0/0."""
    c = cont.sum(axis=0)
    denom = c.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, c[..., 0] / denom, np.nan)


def _lead_mean(curves: np.ndarray) -> np.ndarray:
    """nan-mean over the lead axis (axis 0), nan if every lead is nan (as ``csi_over_leads``)."""
    ok = np.isfinite(curves)
    return np.where(ok.any(axis=0), np.nansum(curves, axis=0) / np.maximum(ok.sum(axis=0), 1), np.nan)


def _csi_lead_mean(thr: float) -> Callable[..., float]:
    k = CSI_THRESHOLDS.index(thr)
    return lambda cont: float(_lead_mean(_csi_per_lead(cont))[k])


def _csi_primary(cont: np.ndarray) -> float:
    means = _lead_mean(_csi_per_lead(cont))
    ks = [CSI_THRESHOLDS.index(t) for t in PRIMARY_THRESHOLDS]
    return float(np.mean(means[ks]))


def _fss_lead_mean(k: int) -> Callable[..., float]:
    def fn(fss: np.ndarray) -> float:
        s = fss.sum(axis=0)[:, k]  # (12, 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            per_lead = np.where(s[:, 1] > 0, 1.0 - s[:, 0] / s[:, 1], np.nan)
        return float(_lead_mean(per_lead[:, None])[0])

    return fn


def _mse(sse: np.ndarray, n_win: np.ndarray, n_pix: int) -> float:
    """Mean over leads of the per-lead pooled MSE (``storm_scores`` convention)."""
    return float((sse.sum(axis=0) / (n_win.sum() * n_pix)).mean())


def _brier(brier: np.ndarray, n_win: np.ndarray, n_pix: int) -> float:
    return float(brier.sum() / (n_win.sum() * N_OUT * n_pix))


def _ece(bins: np.ndarray) -> float:
    b = bins.sum(axis=0)
    total = b[:, 0].sum()
    ok = b[:, 0] > 0
    gap = np.abs(b[ok, 1] / b[ok, 0] - b[ok, 2] / b[ok, 0])
    return float((b[ok, 0] / total * gap).sum()) if total else float("nan")


def _metrics(s: dict[str, np.ndarray], days: np.ndarray, n_pix: int, n_resamples: int) -> dict[str, CI]:
    """Day-block bootstrap CI (seed 0) of every metric from the per-day statistics ``s``."""

    def ci(fn: Callable[..., float], keys: Sequence[str]) -> CI:
        return day_block_ci(fn, [s[k] for k in keys], days, n_resamples=n_resamples, seed=0)

    out = {"csi_mean_74_133": ci(_csi_primary, ["cont"])}
    for thr in CSI_THRESHOLDS:
        out[f"csi_{thr}"] = ci(_csi_lead_mean(thr), ["cont"])
    for k, win in enumerate(FSS_WINDOWS):
        out[f"fss_{FSS_THRESHOLD}_w{win}"] = ci(_fss_lead_mean(k), ["fss"])
    out["mse"] = ci(lambda sse, n: _mse(sse, n, n_pix), ["sse", "n_win"])
    if "brier" in s:
        out["brier_133"] = ci(lambda b, n: _brier(b, n, n_pix), ["brier", "n_win"])
        out["ece_133"] = ci(_ece, ["bins"])
    return out


def _figures(s: dict[str, np.ndarray], n_pix: int) -> dict:
    """Point-estimate curves: per-lead CSI per threshold, and (calibrated models) reliability bins + Brier per lead."""
    curves = _csi_per_lead(s["cont"])
    fig: dict = {"per_lead_csi": {str(t): curves[:, k].tolist() for k, t in enumerate(CSI_THRESHOLDS)}}
    if "bins" in s:
        b = s["bins"].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            fig["reliability"] = {
                "bin_edges": np.linspace(0, 1, N_BINS + 1).tolist(),
                "mean_pred": np.where(b[:, 0] > 0, b[:, 1] / b[:, 0], np.nan).tolist(),
                "obs_freq": np.where(b[:, 0] > 0, b[:, 2] / b[:, 0], np.nan).tolist(),
                "counts": b[:, 0].tolist(),
            }
        fig["brier_per_lead"] = (s["brier"].sum(axis=0) / (s["n_win"].sum() * n_pix)).tolist()
    return fig


# ------------------------------------------------------------------ models
class _DLModel:
    """A frozen checkpoint with its frozen calibrator (``calibrator.json`` next to it, if present)."""

    def __init__(self, name: str, checkpoint: Path, device: str) -> None:
        adapter = StormAdapter(checkpoint=checkpoint)
        model = adapter._load_model()
        assert model is not None
        self.name = name
        self.device = device
        self.model = model.to(device).eval()
        self.calibrator = adapter.calibrator
        self.n_channels = model.in_channels

    def predict(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """``x`` (B, 26, H, W) -> (VIL ``(B, 12, H, W)`` in 0..255, calibrated P(VIL >= 133) same shape)."""
        x = x if self.n_channels == 2 * N_IN else x[:, :N_IN]
        with torch.no_grad():
            vil, logits = self.model(x.to(self.device))
        prob = torch.sigmoid(logits)[:, :, RISK_THRESHOLD_INDEX].float().cpu().numpy()
        if self.calibrator is not None:
            prob = self.calibrator.transform(prob)
        return np.clip(vil.float().cpu().numpy() * 255.0, 0.0, 255.0), prob


def _device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _score_part(
    ds: SevirWindowDataset, window_days: np.ndarray, dl_models: Sequence[_DLModel], n_resamples: int
) -> tuple[dict[str, ModelMetrics], dict[str, dict]]:
    """One pass over the windows of ``ds``: every DL model and both baselines; returns metrics and figure data."""
    names = [m.name for m in dl_models] + [PERSISTENCE_NAME, EXTRAPOLATION_NAME]
    acc: dict[str, list[dict[str, np.ndarray]]] = {name: [] for name in names}
    n_pix = 0
    for start in range(0, len(ds), BATCH):
        xs, ys = zip(*(ds[i] for i in range(start, min(start + BATCH, len(ds)))), strict=True)
        x, truth = torch.stack(xs), torch.stack(ys).numpy()
        n_pix = truth.shape[-1] * truth.shape[-2]
        for m in dl_models:
            vil, prob = m.predict(x)
            acc[m.name].append(window_stats(vil, truth, prob))
        vil_in = np.rint(x[:, :N_IN].numpy() * 255.0)  # back to VIL digital units, (B, 13, H, W)
        last = np.repeat(vil_in[:, -1:], N_OUT, axis=1)
        acc[PERSISTENCE_NAME].append(window_stats(last, truth))
        extrap = np.stack([extrapolation_nowcast(f, N_OUT) for f in vil_in])
        acc[EXTRAPOLATION_NAME].append(window_stats(extrap, truth))
    metrics: dict[str, ModelMetrics] = {}
    figures: dict[str, dict] = {}
    for name in names:
        stats = {k: np.concatenate([b[k] for b in acc[name]]) for k in acc[name][0]}
        by_day, days = _by_day(stats, window_days)
        kind = "baseline" if name in (PERSISTENCE_NAME, EXTRAPOLATION_NAME) else "dl"
        metrics[name] = ModelMetrics(name, kind, _metrics(by_day, days, n_pix, n_resamples))
        figures[name] = _figures(by_day, n_pix)
    return metrics, figures


# ------------------------------------------------------------------ public entry point
def evaluate(split: str, n_resamples: int = 1000) -> HazardEval:
    """Frozen-artefact evaluation of the storm hazard on ``split`` ("val" dry run, or "test" one-shot)."""
    if split not in _PARTS:
        raise ValueError(f"split must be one of {sorted(_PARTS)}, got {split!r}")
    paths = get_paths()
    meta = pd.read_parquet(paths.proc / "sevir" / "sevir.meta.parquet")
    event_ids = meta["id"].astype(str).tolist()
    events = np.load(paths.proc / "sevir" / "sevir.npy", mmap_mode="r")
    sevir_split = read_split(paths.splits / "sevir.json")
    device = _device()
    full = _DLModel(DL_NAME, paths.models / "storm" / "best.pt", device)
    vil_only = _DLModel(VIL_ONLY_NAME, paths.models / "storm_vil_only" / "best.pt", device)

    results: dict[str, list[ModelMetrics]] = {}
    ablations: dict[str, list[ModelMetrics]] = {}
    figures: dict = {}
    n_windows: dict[str, int] = {}
    for result_key, part in _PARTS[split].items():
        items = getattr(sevir_split, part) if part in ("train", "val", "test") else sevir_split.extra[part]
        day_of = {i.id: i.day for i in items}
        ds = SevirWindowDataset.from_split(events, event_ids, sevir_split, part, use_ir=True)
        window_days = np.array([day_of[event_ids[e]] for e in ds.windows[:, 0]], dtype=object)
        n_windows[result_key] = len(ds)
        metrics, figs = _score_part(ds, window_days, [full, vil_only], n_resamples)
        results[result_key] = [
            metrics[DL_NAME],
            metrics[PERSISTENCE_NAME],
            metrics[EXTRAPOLATION_NAME],
        ]
        # the ablation sits next to the main split under "vil_only", other splits as "vil_only@<split>"
        ablation_key = "vil_only" if result_key == split else f"vil_only@{result_key}"
        ablations[ablation_key] = [metrics[VIL_ONLY_NAME]]
        figures[result_key] = {
            "per_lead_csi": {n: f["per_lead_csi"] for n, f in figs.items()},
            "reliability": {n: f["reliability"] for n, f in figs.items() if "reliability" in f},
            "brier_per_lead": {n: f["brier_per_lead"] for n, f in figs.items() if "brier_per_lead" in f},
            "lead_minutes": [5 * (k + 1) for k in range(N_OUT)],
        }

    notes = [
        f"Windows scored per split (after the frozen no-data-frame filter): {json.dumps(n_windows)}; "
        "3 windows per SEVIR event at offsets 0/12/24 frames.",
        "Primary csi_mean_74_133 = mean over thresholds 74 and 133 of the lead-averaged CSI (12 leads, 5-60 min), "
        "contingency counts pooled over all windows and pixels per lead. CSI/FSS/MSE use the U-Net VIL regression "
        "head x 255 (clipped); persistence = last observed VIL frame; pysteps_extrapolation = Lucas-Kanade motion + "
        "semi-Lagrangian advection of the last frame (baselines.extrapolation_nowcast), VIL-only inputs.",
        "CIs: day-block bootstrap over UTC event days (seed 0, same resamples for every model, so models are "
        f"paired), {n_resamples} resamples. Sample = window; days are SEVIR event start days.",
        "Brier and ECE (10 equal-width bins, pooled over leads) are for the calibrated P(VIL >= 133) head of the "
        "U-Nets only (the frozen per-lead isotonic calibrators were fit on val; val calibration numbers are "
        "therefore optimistic). Baselines have no probabilities, so no Brier/ECE.",
        "Ablation 'vil_only' = models/storm_vil_only (13 VIL channels, no IR107), evaluated on the same windows "
        "and with its own frozen val calibrator; extra splits are keyed 'vil_only@<split>'.",
        "Not done: the 1 km resolution check and the storm-windows subset listed in DESIGN 6.1; pooling means "
        "windows with little weather weigh little in CSI (pooled counts), as in the val runs.",
        "models/storm_run1 (earlier run) is ignored.",
    ]
    return HazardEval(
        hazard="storm",
        primary_metric="csi_mean_74_133",
        higher_is_better=True,
        results=results,
        ablations=ablations,
        dl_models=[DL_NAME],
        notes=notes,
        figures=figures,
    )


def self_check(n_windows: int = 24) -> None:
    """Counts-based statistics must equal ``outrun.eval.metrics`` on a handful of VAL windows (never test)."""
    from outrun.eval import metrics as em

    paths = get_paths()
    meta = pd.read_parquet(paths.proc / "sevir" / "sevir.meta.parquet")
    event_ids = meta["id"].astype(str).tolist()
    events = np.load(paths.proc / "sevir" / "sevir.npy", mmap_mode="r")
    ds = SevirWindowDataset.from_split(
        events, event_ids, read_split(paths.splits / "sevir.json"), "val", use_ir=True
    )
    model = _DLModel(DL_NAME, paths.models / "storm" / "best.pt", _device())
    xs, ys = zip(*(ds[i] for i in range(0, len(ds), max(1, len(ds) // n_windows))), strict=True)
    vil, prob = model.predict(torch.stack(xs))
    truth = torch.stack(ys).numpy()
    s = window_stats(vil, truth, prob)
    s["n_win"] = np.array([len(vil)], dtype=np.float64)
    s = {k: v.sum(axis=0, keepdims=True) if k != "n_win" else v for k, v in s.items()}
    ref = em.csi_over_leads(vil.transpose(1, 0, 2, 3), truth.transpose(1, 0, 2, 3), PRIMARY_THRESHOLDS)
    assert abs(_csi_primary(s["cont"]) - ref["mean_all"]) < 1e-9
    n_pix = vil.shape[-1] * vil.shape[-2]
    y = truth >= RISK_THRESHOLD
    assert abs(_brier(s["brier"], s["n_win"], n_pix) - em.brier(y, prob)) < 1e-9
    assert abs(_ece(s["bins"]) - em.ece(y, prob, N_BINS)) < 1e-9
    fss_ref = np.nanmean([em.fss(vil[:, k], truth[:, k], FSS_THRESHOLD, 3) for k in range(N_OUT)])
    assert abs(_fss_lead_mean(0)(s["fss"]) - fss_ref) < 1e-9
    print("storm final_eval self-check OK")


if __name__ == "__main__":
    self_check()
