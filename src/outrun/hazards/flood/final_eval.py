"""Flood one-shot held-out evaluation (T24): ``evaluate(split, n_resamples) -> HazardEval``.

Same code path for ``"val"`` (dry run) and ``"test"`` (official test = ``split.test``, plus the quarantined case
hold-out = ``split.extra["case_holdout"]``, empty for flood). Only frozen artefacts are used: ``models/flood``,
``models/flood_noflash`` checkpoints and their ``calibrator.json`` (fit on val). The logistic regression is fit on
the TRAIN split only (seed 0). Rule baselines have no parameters.

Sample set: ``train.balanced_part`` (FLOOD-13 balanced sampler, seed 0), applied to each part exactly as to val.
Scored pixels: ``y != -1`` (uncertain ring dropped) and ``valid == 1``.

Bootstrap: day-block over the UTC issue day of each patch. Pixels never leave per-day sufficient statistics
(score histograms, contingency counts, Brier sums, reliability bins), so ``day_block_ci`` runs on per-day rows,
which is a true day-block bootstrap. AUPRC / ROC-AUC estimates come from 2000 sqrt-spaced score bins (ties inside a
bin are merged), which agrees with the exact sklearn value to about 1e-4.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
from scipy.ndimage import maximum_filter

from outrun.config import get_paths
from outrun.core.calibrate import calibrator_from_dict
from outrun.eval.bootstrap import CI, day_block_ci
from outrun.eval.hazard_eval import HazardEval
from outrun.eval.report import ModelMetrics
from outrun.eval.splits import read_split

from . import baselines as B
from .dataset import FLOOD_CHANNELS, SampleKind, select_channels
from .model import FloodUNet
from .train import balanced_part

N_BINS = 2000  # score bins (sqrt spacing: fine resolution at low probabilities)
THRESHOLD = 0.5
NB_RADIUS = 2
CHUNK = 256
N_REL = 10
_STATS = (
    "tp",
    "fp",
    "fn",
    "nb_hit",
    "nb_miss",
    "nb_fa",
    "pe_hit",
    "pe_n",
    "hn_alarm",
    "hn_n",
    "sse",
    "n_pix",
)
_S = {k: i for i, k in enumerate(_STATS)}
_NAN_CI = CI(float("nan"), float("nan"), float("nan"), 0, 0)


class _Acc:
    """Per-day sufficient statistics of one model on one part."""

    def __init__(self, n_days: int) -> None:
        self.hist = np.zeros(
            (3, n_days, N_BINS), dtype=np.int64
        )  # positive px, negative px, hard-neg-patch neg px
        self.stats = np.zeros((n_days, len(_STATS)))
        self.rel = np.zeros((n_days, N_REL, 3))  # per reliability bin: count, sum p, sum y
        self.n_days = n_days

    def update(self, p: np.ndarray, X: np.ndarray, Y: np.ndarray, kinds: np.ndarray, day: np.ndarray) -> None:
        valid = X[:, -1] > 0
        mask = (Y != -1) & valid
        pos, neg = mask & (Y == 1), mask & (Y == 0)
        hn = neg & (kinds == SampleKind.HARD_NEGATIVE.value)[:, None, None]
        d3 = np.broadcast_to(day[:, None, None], p.shape)
        pb = np.minimum((np.sqrt(np.clip(p, 0, 1)) * N_BINS).astype(np.int64), N_BINS - 1)
        flat = d3 * N_BINS + pb
        for k, sel in enumerate((pos, neg, hn)):
            self.hist[k] += np.bincount(flat[sel], minlength=self.n_days * N_BINS).reshape(
                self.n_days, N_BINS
            )
        # per-patch contingency / neighbourhood / event counts, then added to their day
        pred = (p >= THRESHOLD) & mask
        size = (1, 2 * NB_RADIUS + 1, 2 * NB_RADIUS + 1)
        f_near = maximum_filter(pred.astype(np.uint8), size=size, mode="constant").astype(bool)
        o_near = maximum_filter(pos.astype(np.uint8), size=size, mode="constant").astype(bool)
        foot = (Y == 1) & valid
        is_pos, is_hn = kinds == SampleKind.POSITIVE.value, kinds == SampleKind.HARD_NEGATIVE.value
        row = np.zeros((len(p), len(_STATS)))
        row[:, _S["tp"]] = (pred & pos).sum((1, 2))
        row[:, _S["fp"]] = (pred & neg).sum((1, 2))
        row[:, _S["fn"]] = (~pred & pos).sum((1, 2))
        row[:, _S["nb_hit"]] = (pos & f_near).sum((1, 2))
        row[:, _S["nb_miss"]] = (pos & ~f_near).sum((1, 2))
        row[:, _S["nb_fa"]] = (pred & ~o_near).sum((1, 2))
        row[:, _S["pe_hit"]] = is_pos & (((p >= THRESHOLD) & foot).any((1, 2)))
        row[:, _S["pe_n"]] = is_pos
        row[:, _S["hn_alarm"]] = is_hn & (((p >= THRESHOLD) & valid).any((1, 2)))
        row[:, _S["hn_n"]] = is_hn
        t = (Y == 1).astype(np.float64)
        row[:, _S["sse"]] = (((p - t) ** 2) * mask).sum((1, 2))
        row[:, _S["n_pix"]] = mask.sum((1, 2))
        np.add.at(self.stats, day, row)
        rb = np.minimum((np.clip(p, 0, 1) * N_REL).astype(np.int64), N_REL - 1)
        rflat = (d3 * N_REL + rb)[mask]
        size_ = self.n_days * N_REL
        for c, w in enumerate((None, p[mask], t[mask])):
            self.rel[:, :, c] += np.bincount(rflat, weights=w, minlength=size_).reshape(self.n_days, N_REL)


# ------------------------------------------------------------------ metrics from per-day sums
def _ap(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos.sum(0)[::-1], neg.sum(0)[::-1]  # highest score first
    n_pos = pos.sum()
    if n_pos == 0:
        return float("nan")
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    prec = tp / np.maximum(tp + fp, 1)
    return float(np.sum(pos / n_pos * prec))


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos, neg = pos.sum(0)[::-1], neg.sum(0)[::-1]
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    tpr = np.concatenate([[0.0], np.cumsum(pos) / pos.sum()])
    fpr = np.concatenate([[0.0], np.cumsum(neg) / neg.sum()])
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def _ratio(num: str, den: tuple[str, ...]) -> Callable[[np.ndarray], float]:
    def fn(stats: np.ndarray) -> float:
        s = stats.sum(0)
        d = sum(s[_S[k]] for k in den)
        return float(s[_S[num]] / d) if d else float("nan")

    return fn


def _brier(stats: np.ndarray) -> float:
    s = stats.sum(0)
    return float(s[_S["sse"]] / s[_S["n_pix"]])


def _ece_from_rel(rel: np.ndarray) -> float:
    r = rel.sum(0)
    n, ok = r[:, 0].sum(), r[:, 0] > 0
    if n == 0:
        return float("nan")
    c = r[ok, 0]
    return float(np.sum(c / n * np.abs(r[ok, 1] / c - r[ok, 2] / c)))


def _compact(hist: np.ndarray) -> np.ndarray:
    """Drop score bins that are empty in all three histograms (keeps order; far fewer columns)."""
    keep = hist.sum((0, 1)) > 0
    return hist[:, :, keep].astype(np.int32)


def _model_metrics(
    name: str, kind: str, acc: _Acc, days: np.ndarray, n: int, calibrated: bool
) -> ModelMetrics:
    h = _compact(acc.hist)
    pos, neg, hn = h[0], h[1], h[2]
    ci = lambda fn, arrs: day_block_ci(fn, arrs, days, n_resamples=n, seed=0)  # noqa: E731
    m: dict[str, CI] = {
        "auprc": ci(lambda a, b: _ap(a, b), [pos, neg]),
        "roc_auc": ci(lambda a, b: _auc(a, b), [pos, neg]),
        "auc_vs_hard_negative": ci(lambda a, b: _auc(a, b), [pos, hn]),
        "brier": ci(_brier, [acc.stats]),
        "csi_pixel_p50": ci(_ratio("tp", den=("tp", "fp", "fn")), [acc.stats]),
        "pod_pixel_p50": ci(_ratio("tp", den=("tp", "fn")), [acc.stats]),
        "far_pixel_p50": ci(_ratio("fp", den=("tp", "fp")), [acc.stats]),
        "neighbourhood_csi_r2_p50": ci(_ratio("nb_hit", den=("nb_hit", "nb_miss", "nb_fa")), [acc.stats]),
        "event_pod_p50": ci(_ratio("pe_hit", den=("pe_n",)), [acc.stats]),
        "hard_neg_alarm_rate_p50": ci(_ratio("hn_alarm", den=("hn_n",)), [acc.stats]),
    }
    if calibrated:
        m["ece"] = ci(_ece_from_rel, [acc.rel])
    return ModelMetrics(name, kind, m)  # type: ignore[arg-type]


# ------------------------------------------------------------------ data access
def _chunks(sub, full):
    """Yield (X, Y, kinds-slice-start) chunks of the balanced subset, in order; X (n,11,H,W), Y (n,H,W)."""
    base = full.base
    idx = [int(r) for r in sub.indices]
    for s in range(0, len(idx), CHUNK):
        part = idx[s : s + CHUNK]
        X = np.stack([np.asarray(base.x[int(base.rows[i])], dtype=np.float32) for i in part])
        Y = np.stack([np.asarray(base.y[int(base.rows[i])]) for i in part])
        yield X, Y, part, s


def _fit_logistic(split) -> B.LogisticBaseline:
    """LR on TRAIN only: uniform random pixel subsample (2 M kept pixels, seed 0) streamed over chunks."""
    full, sub, _ = balanced_part(split, "train")
    n_kept = 0
    for X, Y, _, _ in _chunks(sub, full):
        n_kept += int(((Y != -1) & (X[:, -1] > 0)).sum())
    frac = min(1.0, 2_000_000 / max(n_kept, 1))
    rng = np.random.default_rng(0)
    xs, ys = [], []
    for X, Y, _, _ in _chunks(sub, full):
        keep = (Y != -1) & (X[:, -1] > 0) & (rng.random(Y.shape) < frac)
        xs.append(np.moveaxis(X, 1, -1)[keep])  # (k, 11)
        ys.append(Y[keep])
    px = np.concatenate(xs)[:, :, None, None]
    py = np.concatenate(ys)[:, None, None]
    return B.LogisticBaseline(seed=0, max_pixels=len(py) + 1).fit(px, py)


def _load_net(name: str, use_flash: bool, device: str):
    import torch

    from outrun.hazards.base import load_checkpoint

    d = get_paths().models / name
    net = FloodUNet(in_channels=11 if use_flash else 8)
    load_checkpoint(d / "best.pt", net)
    cal = calibrator_from_dict(json.loads((d / "calibrator.json").read_text()))
    return net.to(device).eval(), cal, torch


def _score_part(split, part: str, n_resamples: int, device: str, lr: B.LogisticBaseline) -> dict:
    items = split.test if part == "test" else split.extra[part] if part != "val" else split.val
    full, sub, kinds = balanced_part(split, part)
    day_of = {it.id: it.day for it in items}
    ids = [full.base.ids[int(i)] for i in sub.indices]
    day_labels = sorted({day_of[i] for i in ids})
    day_idx = np.array([day_labels.index(day_of[i]) for i in ids])
    kind_arr = np.array([k.value for k in kinds])
    nets = {"main": _load_net("flood", True, device), "noflash": _load_net("flood_noflash", False, device)}
    i_ari = FLOOD_CHANNELS.index("ari_max")
    names = [
        "unet_cal",
        "unet_raw",
        "noflash_cal",
        "noflash_raw",
        "ffg_ge1",
        "ari_ge10",
        "ari_rule",
        "logreg",
    ]
    acc = {k: _Acc(len(day_labels)) for k in names}
    for X, Y, part_idx, s in _chunks(sub, full):
        n = len(part_idx)
        probs: dict[str, np.ndarray] = {}
        for _key, (net, cal, torch), flash, tag in (
            ("main", nets["main"], True, "unet"),
            ("noflash", nets["noflash"], False, "noflash"),
        ):
            xb = torch.from_numpy(np.ascontiguousarray(select_channels(X, flash))).to(device)
            with torch.no_grad():
                p = torch.sigmoid(net(xb))[:, 0].cpu().numpy().astype(np.float64)
            probs[f"{tag}_raw"] = p
            probs[f"{tag}_cal"] = cal.transform(p).astype(np.float64)
        probs["ffg_ge1"] = B.rule_baseline_from_features(X)
        probs["ari_ge10"] = B.ari_baseline(X[:, i_ari])
        probs["ari_rule"] = B.ari_rule_fallback(X[:, 1], X[:, 2])
        probs["logreg"] = lr.predict_proba(X)
        for k, p in probs.items():
            acc[k].update(np.asarray(p, dtype=np.float64), X, Y, kind_arr[s : s + n], day_idx[s : s + n])
    days = np.array(day_labels, dtype=object)
    spec = {
        "unet_cal": ("flood.unet", "dl", True),
        "unet_raw": ("flood.unet_raw", "dl", True),
        "noflash_cal": ("flood.unet_noflash", "dl", True),
        "noflash_raw": ("flood.unet_noflash_raw", "dl", True),
        "ffg_ge1": ("ffg_ratio_ge_1", "baseline", False),
        "ari_ge10": ("qpe_ari_ge_10yr", "baseline", False),
        "ari_rule": ("ari_rule_3h50_6h75", "baseline", False),
        "logreg": ("logistic_regression", "baseline", True),
    }
    models = {k: _model_metrics(v[0], v[1], acc[k], days, n_resamples, v[2]) for k, v in spec.items()}
    s0 = acc["unet_cal"].stats.sum(0)
    pos_px, neg_px = int(acc["unet_cal"].hist[0].sum()), int(acc["unet_cal"].hist[1].sum())
    info = {
        "n_patches": len(ids),
        "n_days": len(day_labels),
        "n_pixels": pos_px + neg_px,
        "pixel_prevalence": pos_px / max(pos_px + neg_px, 1),
        "kinds": {k.value: int(sum(1 for x in kinds if x == k)) for k in SampleKind},
        "n_positive_patches": int(s0[_S["pe_n"]]),
    }
    rel = {}
    for k in ("unet_cal", "unet_raw"):
        r = acc[k].rel.sum(0)
        c = np.maximum(r[:, 0], 1)
        rel[spec[k][0]] = {
            "bin_edges": np.linspace(0, 1, N_REL + 1).tolist(),
            "count": r[:, 0].astype(int).tolist(),
            "mean_pred": np.where(r[:, 0] > 0, r[:, 1] / c, np.nan).tolist(),
            "obs_freq": np.where(r[:, 0] > 0, r[:, 2] / c, np.nan).tolist(),
        }
    return {"models": models, "info": info, "reliability": rel}


def evaluate(split: str, n_resamples: int = 1000) -> HazardEval:
    """Flood evaluation on ``"val"`` (dry run) or ``"test"`` (official test + case hold-out)."""
    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    fsplit = read_split(get_paths().splits / "flood.json")
    parts = ["val"] if split == "val" else ["test", "case_holdout"]
    from outrun.hazards.base import select_device

    device = str(select_device("mps"))
    lr = _fit_logistic(fsplit)
    results: dict[str, list[ModelMetrics]] = {}
    ablations: dict[str, list[ModelMetrics]] = {"no_flash": []}
    figures: dict = {}
    notes: list[str] = []
    for part in parts:
        items = fsplit.test if part == "test" else fsplit.val if part == "val" else fsplit.extra[part]
        if not items:
            results[part] = []
            notes.append(f"{part}: empty set in splits/flood.json (nothing to evaluate).")
            continue
        out = _score_part(fsplit, part, n_resamples, device, lr)
        m = out["models"]
        results[part] = [m[k] for k in ("unet_cal", "unet_raw", "ffg_ge1", "ari_ge10", "ari_rule", "logreg")]
        if part == parts[0]:
            ablations["no_flash"] = [m["noflash_cal"], m["noflash_raw"]]
        figures[part] = {"info": out["info"], "reliability": out["reliability"]}
    notes += [
        "Sample set is the FLOOD-13 balanced sample (1/3 positive, 1/3 hard-negative, 1/3 wet patches; seed 0), "
        "built exactly as for val from the frozen split ids; pixel prevalence is ~5 % and is NOT the true base rate. "
        "AUPRC and Brier are only comparable within this sample; calibrated probabilities are calibrated to the "
        "balanced val sample (isotonic, frozen), not to climatology.",
        "Scored pixels: ring (y == -1) and valid == 0 pixels dropped. Thresholded metrics (pixel CSI/POD/FAR, "
        f"neighbourhood CSI radius {NB_RADIUS}, event POD, hard-negative alarm rate) use a fixed p >= {THRESHOLD}, "
        "not tuned; event POD = fraction of positive patches with an alarm pixel in the positive footprint "
        "(patch footprints, not individual reports).",
        "CI = day-block bootstrap over UTC issue days of the patches (per-day sufficient statistics); AUPRC/ROC "
        "from 2000 sqrt-spaced score bins (error ~1e-4). Single training seed: CI covers test sampling only, "
        "not training variance.",
        "flood.unet = frozen isotonic-calibrated U-Net (shipped model); flood.unet_raw = same net uncalibrated. "
        "Isotonic ties cost about 0.003 AUPRC on val. FLASH ablation (no_flash) was within seed noise on val.",
        "Logistic regression refit on TRAIN only from a streamed uniform 2 M-pixel subsample (seed 0); its random "
        "subset differs from logs/flood_baselines_val.json, so small differences are expected.",
        "Rule baselines are {0,1} predictors: AUPRC is a single operating point; Brier/ECE are not reported "
        "meaningfully for them (ECE omitted).",
        "No published reference number exists for this patch set; no reference row.",
    ]
    if split == "test":
        notes.append("case_holdout is empty for flood (quarantine touches no flood sample, T14).")
    return HazardEval(
        hazard="flood",
        primary_metric="auprc",
        higher_is_better=True,
        results=results,
        ablations=ablations,
        dl_models=["flood.unet"],
        notes=notes,
        figures=figures,
    )


if __name__ == "__main__":  # val-only dry run
    import sys
    import time

    t0 = time.time()
    ev = evaluate("val", n_resamples=int(sys.argv[1]) if len(sys.argv) > 1 else 50)
    for grp, lst in [*ev.results.items(), *ev.ablations.items()]:
        for mm in lst:
            print(
                grp,
                mm.name,
                {k: (round(v.estimate, 4), round(v.low, 4), round(v.high, 4)) for k, v in mm.metrics.items()},
            )
    print("seconds", round(time.time() - t0))
