"""One-shot held-out evaluation of the wildfire U-Net and baselines (T24; contract in ``eval/hazard_eval.py``).

``evaluate("val")`` is the dry run (NDWS eval file, plus a few WSTS 2020 fires for the WSTS code path; 2021 is never
read). ``evaluate("test")`` runs the SAME code on the NDWS test file ids (``split.test``), the WSTS 2021 fires
(``split.extra["test_wsts2021"]``) and the quarantined case hold-out (``split.extra["case_holdout"]``).

Frozen artefacts only: ``models/wildfire/best.pt``, ``norm_stats.npz``, ``calibrator.json``. The pixel logistic
baseline is fit on NDWS train only (seeded). Nothing is tuned here.

Scores are streamed into per-group score histograms (fixed log-spaced bins) so the bootstrap is a true cluster
bootstrap that costs a sum over histograms instead of a pass over millions of pixels.

``python -m outrun.hazards.wildfire.final_eval`` runs a val-only self-check (binned vs exact metrics).
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path

import numpy as np

from outrun.config import get_paths
from outrun.data.wsts import map_to_ndws, read_wsts_day
from outrun.eval import metrics as M  # noqa: N812
from outrun.eval.bootstrap import CI, day_block_ci
from outrun.eval.hazard_eval import HazardEval
from outrun.eval.report import ModelMetrics
from outrun.eval.splits import Split, read_split, split_mask

from .baselines import PixelLogistic, downwind_persistence, persistence
from .dataset import normalize
from .metrics import per_fire_macro
from .train import load_stats, ndws_files
from .wsts_shift import TIF, zip_index

PREV = 11  # PrevFireMask channel
N_BINS = 4096  # score histogram bins (edge 0, then uniform in logit from -14 to 14)
EDGES = np.concatenate([[0.0], 1.0 / (1.0 + np.exp(-np.linspace(-14.0, 14.0, N_BINS - 1)))])
REL_BINS = 10
PUBLISHED_NDWS_PRAUC = 0.284
DL_NAME = "wildfire.unet"
RAW_NAME = "wildfire.unet_uncalibrated"
BASELINES = ("persistence", "downwind_persistence", "pixel_logistic")
ALL_MODELS = (DL_NAME, RAW_NAME, *BASELINES)
BOOT_SEED = 0
LR_MAX_PIXELS = 2_000_000  # same as PixelLogistic default / val_eval


# ------------------------------------------------------------------ frozen model and baselines
class _Predictor:
    """All model scores for a batch of raw NDWS-order inputs (N, 12, H, W): {name: (N, H, W) float32}."""

    def __init__(self) -> None:
        import torch

        from .model import FireUNet, TemperatureScaler

        model_dir = get_paths().models / "wildfire"
        self.stats = load_stats(model_dir / "norm_stats.npz")
        payload = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
        self.net = FireUNet.from_state_dict(payload["model"]).eval()
        self.cal = TemperatureScaler.load(model_dir / "calibrator.json")  # frozen val fit
        self.lr = _fit_logistic(self.stats)

    def logits(self, z: np.ndarray, batch: int = 64) -> np.ndarray:
        """U-Net logits for normalised (N, 12, H, W); zero-padded bottom/right to a multiple of 2**levels."""
        import torch
        from torch.nn import functional as F  # noqa: N812

        mult = 2**self.net.levels
        h, w = z.shape[-2:]
        pad_h, pad_w = (-h) % mult, (-w) % mult
        out = []
        with torch.no_grad():
            for i in range(0, len(z), batch):
                t = torch.from_numpy(z[i : i + batch])
                if pad_h or pad_w:
                    t = F.pad(t, (0, pad_w, 0, pad_h))
                out.append(self.net(t)[:, 0, :h, :w].numpy())
        return np.concatenate(out)

    def __call__(self, x: np.ndarray) -> dict[str, np.ndarray]:
        z = normalize(x, self.stats)
        raw = (1.0 / (1.0 + np.exp(-self.logits(z).astype(np.float64)))).astype(np.float32)
        return {
            DL_NAME: self.cal.transform(raw),
            RAW_NAME: raw,
            "persistence": persistence(x),
            "downwind_persistence": downwind_persistence(x),
            "pixel_logistic": self.lr.predict_proba(z),
        }


def _fit_logistic(stats) -> PixelLogistic:
    """Pixel logistic on NDWS TRAIN only, same pixel subsample and seed as ``val_eval`` (never val/test).

    The 2 M labelled pixels are drawn exactly as ``PixelLogistic.fit`` would (seeded choice over all labelled
    train pixels) but gathered chunk by chunk, so the 15 k x 12 x 64 x 64 train array is never held in memory.
    """
    global _LR
    if _LR is not None:
        return _LR
    split = read_split(get_paths().splits / "ndws.json")
    x_path, y_path, ids = ndws_files("train")
    rows = np.flatnonzero(split_mask(ids, split, "train"))
    x = np.load(x_path, mmap_mode="r")
    y = np.asarray(np.load(y_path, mmap_mode="r")[rows]).reshape(len(rows), -1)
    keep = np.flatnonzero(y.reshape(-1) != -1)
    if keep.size > LR_MAX_PIXELS:
        keep = np.sort(np.random.default_rng(0).choice(keep, LR_MAX_PIXELS, replace=False))
    hw = y.shape[1]
    feats = np.empty((keep.size, x.shape[1]), np.float32)
    labels = y.reshape(-1)[keep]
    for a in range(0, len(rows), 512):
        lo, hi = np.searchsorted(keep, [a * hw, min(a + 512, len(rows)) * hw])
        if lo == hi:
            continue
        z = normalize(np.asarray(x[rows[a : a + 512]]), stats)
        z = np.moveaxis(z, 1, -1).reshape(-1, z.shape[1])
        feats[lo:hi] = z[keep[lo:hi] - a * hw]
    _LR = PixelLogistic(seed=0).fit(feats[:, :, None, None], labels[:, None, None])
    return _LR


_LR: PixelLogistic | None = None


# ------------------------------------------------------------------ streaming group statistics
class _Acc:
    """Per-group sufficient statistics of one model's scores (group = NDWS sample or WSTS fire)."""

    def __init__(self, n_groups: int) -> None:
        self.hist = np.zeros((2, n_groups, 2 * N_BINS), np.float32)  # [all, new-burn]: neg bins | pos bins
        self.sse = np.zeros(n_groups)
        self.n = np.zeros(n_groups)
        self.rel = np.zeros((n_groups, REL_BINS, 3))  # count, sum p, sum y
        self.fire_ap = np.full(n_groups, np.nan)

    def add(self, g: int, prob: np.ndarray, label: np.ndarray, prev: np.ndarray) -> None:
        keep = label.reshape(-1) != -1
        p = prob.reshape(-1)[keep].astype(np.float64)
        y = (label.reshape(-1)[keep] == 1).astype(np.int64)
        new = prev.reshape(-1)[keep] == 0
        b = np.searchsorted(EDGES, p, side="right") - 1
        for k, sel in enumerate((slice(None), new)):
            self.hist[k, g] += np.bincount(y[sel] * N_BINS + b[sel], minlength=2 * N_BINS)
        self.sse[g] += ((p - y) ** 2).sum()
        self.n[g] += len(p)
        rb = np.minimum(np.searchsorted(np.linspace(0, 1, REL_BINS + 1), p, side="right") - 1, REL_BINS - 1)
        for j, w in enumerate((np.ones_like(p), p, y.astype(np.float64))):
            self.rel[g, :, j] += np.bincount(rb, weights=w, minlength=REL_BINS)


def _ap(hist: np.ndarray) -> float:
    """Step-wise average precision from one (neg bins | pos bins) histogram (sklearn definition, ties pooled)."""
    neg, pos = hist[:N_BINS][::-1].astype(np.float64), hist[N_BINS:][::-1].astype(np.float64)
    if pos.sum() == 0:
        return float("nan")
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum(pos / pos.sum() * precision))


def _ece_from(rel: np.ndarray) -> float:
    c, sp, sy = rel[..., 0], rel[..., 1], rel[..., 2]
    ok = c > 0
    n = c.sum()
    return float(np.sum(np.abs(sp[ok] - sy[ok])) / n) if n else float("nan")


def _ci(fn: Callable[..., float], arrays: list[np.ndarray], n_resamples: int) -> CI:
    """Cluster bootstrap over groups (``day_block_ci`` with one block per group); NaN bounds if < 2 groups."""
    g = len(arrays[0])
    if g < 2:
        return CI(float(fn(*arrays)), float("nan"), float("nan"), 0, g)
    return day_block_ci(fn, arrays, np.arange(g), n_resamples=n_resamples, seed=BOOT_SEED)


def _metrics(acc: _Acc, n_resamples: int, per_fire: bool) -> dict[str, CI]:
    out = {
        "prauc": _ci(lambda h: _ap(h.sum(0)), [acc.hist[0]], n_resamples),
        "prauc_new_burn": _ci(lambda h: _ap(h.sum(0)), [acc.hist[1]], n_resamples),
        "brier": _ci(lambda s, n: float(s.sum() / n.sum()), [acc.sse, acc.n], n_resamples),
        "ece": _ci(lambda r: _ece_from(r.sum(0)), [acc.rel], n_resamples),
    }
    if per_fire:
        out["prauc_per_fire_macro"] = _ci(per_fire_macro, [acc.fire_ap], n_resamples)
    return out


def _rows(accs: dict[str, _Acc], n_resamples: int, per_fire: bool) -> dict[str, list[ModelMetrics]]:
    """{"main": [dl, baselines..., reference], "uncalibrated": [raw U-Net, same baselines]}."""
    base = [ModelMetrics(m, "baseline", _metrics(accs[m], n_resamples, per_fire)) for m in BASELINES]
    dl = ModelMetrics(DL_NAME, "dl", _metrics(accs[DL_NAME], n_resamples, per_fire))
    raw = ModelMetrics(RAW_NAME, "dl", _metrics(accs[RAW_NAME], n_resamples, per_fire))
    return {"main": [dl, *base], "no_calibration": [raw, *base]}


def _reliability(acc: _Acc) -> dict:
    rel = acc.rel.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "bin_edges": np.linspace(0, 1, REL_BINS + 1).tolist(),
            "mean_pred": (rel[:, 1] / rel[:, 0]).tolist(),
            "obs_freq": (rel[:, 2] / rel[:, 0]).tolist(),
            "counts": rel[:, 0].astype(int).tolist(),
        }


# ------------------------------------------------------------------ NDWS
def _ndws_eval(file: str, part: str, split: Split, predict: _Predictor, chunk: int = 256) -> dict[str, _Acc]:
    """Stream the NDWS file rows whose id is in ``split.<part>``; one bootstrap group per sample."""
    x_path, y_path, ids = ndws_files(file)
    rows = np.flatnonzero(split_mask(ids, split, part))
    x, y = np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r")
    accs = {m: _Acc(len(rows)) for m in ALL_MODELS}
    for a in range(0, len(rows), chunk):
        sel = rows[a : a + chunk]
        xb, yb = np.asarray(x[sel], dtype=np.float32), np.asarray(y[sel])
        probs = predict(xb)
        for i in range(len(sel)):
            for m in ALL_MODELS:
                accs[m].add(a + i, probs[m][i], yb[i], xb[i, PREV])
    return accs


# ------------------------------------------------------------------ WSTS
def _wsts_fires(split: Split, split_name: str, zip_path: Path) -> dict[str, list[str]]:
    """{fire id: sorted tif names}. test: ``extra["test_wsts2021"]``; val dry run: 6 seeded 2020 fires (unused by
    train, val and test)."""
    idx = zip_index(zip_path)
    if split_name == "val":
        keys = sorted(k for k in idx if k[0] == "2020")
        pick = np.random.default_rng(0).choice(len(keys), 6, replace=False)
        return {keys[i][1]: idx[keys[i]] for i in sorted(pick)}
    return _wsts_named(idx, [i.id for i in split.extra["test_wsts2021"]], "2021")


def _wsts_named(idx: dict, item_ids: list[str], year: str) -> dict[str, list[str]]:
    fires = [i.removeprefix("wsts:") for i in item_ids if i.startswith("wsts:")]
    return {f: idx[(year, f)] for f in sorted(fires) if (year, f) in idx}


def _tif_date(name: str) -> date:
    return date.fromisoformat(TIF.match(name)["day"])


def _fire_pairs(z: zipfile.ZipFile, names: list[str], tmp: Path) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """(mapped day-d inputs (12, H, W), day d+1 fire mask) for consecutive days, through ``map_to_ndws``."""
    prev_x, prev_day = None, None
    for name in names:
        path = Path(z.extract(name, tmp))
        bands, res = read_wsts_day(path)
        path.unlink()
        x = map_to_ndws(bands, res)
        day = _tif_date(name)
        if prev_x is not None and (day - prev_day).days == 1 and x.shape == prev_x.shape:
            yield prev_x, x[PREV]
        prev_x, prev_day = x, day


def _wsts_eval(fires: dict[str, list[str]], zip_path: Path, predict: _Predictor) -> dict[str, _Acc]:
    """One bootstrap group per fire; every consecutive-day pair of the fire is a sample."""
    accs = {m: _Acc(len(fires)) for m in ALL_MODELS}
    with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory() as tmp:
        for g, names in enumerate(fires.values()):
            pairs = list(_fire_pairs(z, names, Path(tmp)))
            if not pairs:
                continue
            x = np.stack([p[0] for p in pairs])
            y = np.stack([p[1] for p in pairs])
            probs = predict(x)
            for m in ALL_MODELS:
                for i in range(len(x)):
                    accs[m].add(g, probs[m][i], y[i], x[i, PREV])
                accs[m].fire_ap[g] = M.auprc(y.reshape(-1), probs[m].reshape(-1))
    return accs


# ------------------------------------------------------------------ entry point
def _unit(split_name: str) -> tuple[str, str]:
    if split_name == "val":
        return "eval", "val"
    if split_name == "test":
        return "test", "test"
    raise ValueError(f"split must be 'val' or 'test', got {split_name!r}")


def evaluate(split: str, n_resamples: int = 1000) -> HazardEval:
    file, part = _unit(split)
    sp = read_split(get_paths().splits / "ndws.json")
    zip_path = get_paths().raw / "wsts" / "WildfireSpreadTS.zip"
    predict = _Predictor()
    results: dict[str, list[ModelMetrics]] = {}
    ablations: dict[str, list[ModelMetrics]] = {}
    figures: dict = {}

    def collect(key: str, accs: dict[str, _Acc], per_fire: bool, reference: bool) -> None:
        rows = _rows(accs, n_resamples, per_fire)
        results[key] = rows["main"] + ([_reference()] if reference else [])
        ablations.setdefault("no_calibration", []).extend(rows["no_calibration"][:1])
        figures[f"reliability_{key}"] = _reliability(accs[DL_NAME])

    collect(split, _ndws_eval(file, part, sp, predict), per_fire=False, reference=True)
    notes = [
        "NDWS has no dates: the bootstrap resamples individual samples (64x64 patches; an unknown number are "
        "neighbours from the same fire) with replacement, not day blocks, so the NDWS CIs are probably "
        "optimistic. WSTS CIs resample whole fires (a fire's days are strongly dependent).",
        "PR-AUC and the CIs are computed from per-group score histograms (4096 fixed logit-spaced bins, ties "
        "pooled), exact for the {0,1} baselines; differences from sklearn AP are ~1e-4 (see --selfcheck). "
        "Labels -1 (uncertain) are ignored everywhere. prauc_new_burn = pixels with PrevFireMask == 0.",
        "Pixel logistic is refit here on NDWS train only (seed 0, the same 2 M-pixel subsample as val_eval). "
        "U-Net, normalisation stats and calibrator (T=2.18, bias=3.07, fitted on val) are frozen from models/wildfire.",
        "Published NDWS PR-AUC 0.284 (reference row) is not a like-for-like number: different training and "
        "evaluation protocol; it is context, never the best baseline.",
        "No ablation checkpoint exists for wildfire (e.g. no-satellite); the only ablation row is the "
        "uncalibrated U-Net (same PR-AUC, worse Brier/ECE).",
        "WSTS: day-d inputs mapped to the 12 NDWS channels at 1 km with data.wsts.map_to_ndws (population NaN -> "
        "NDWS-train mean), label = day d+1 VIIRS any-fire mask (no uncertain class), pairs only for "
        "consecutive calendar days; images are zero-padded to a multiple of 8 for the U-Net and cropped back. "
        "WSTS fires are western/higher/drier than NDWS (KS shift table, NOTES T13): expect a domain-shift drop.",
    ]
    if split == "val":
        fires = _wsts_fires(sp, "val", zip_path)
        accs = _wsts_eval(fires, zip_path, predict)
        rows = _rows(accs, n_resamples, True)["main"]
        figures["wsts_dry_run_2020_fires"] = {
            "fires": list(fires),
            "models": {m.name: {k: v.to_dict() for k, v in m.metrics.items()} for m in rows},
        }
        notes.append(
            "val dry run: the WSTS code path ran on 6 seeded WSTS 2020 fires (not part of any split) "
            "and is under figures['wsts_dry_run_2020_fires']; WSTS 2021 was not read."
        )
    else:
        wsts = _wsts_fires(sp, "test", zip_path)
        collect("test_wsts2021", _wsts_eval(wsts, zip_path, predict), per_fire=True, reference=False)
        notes.append(f"WSTS 2021: {len(wsts)} fires (split.extra['test_wsts2021']).")
        holdout = {i.id for i in sp.extra.get("case_holdout", ())}
        accs = _ndws_eval("test", "case_holdout", sp, predict)
        n_case = int(accs[DL_NAME].n.astype(bool).sum())
        w_case = _wsts_named(zip_index(zip_path), sorted(holdout), "2021")
        if n_case:
            collect("case_holdout", accs, per_fire=False, reference=False)
        elif w_case:
            collect("case_holdout", _wsts_eval(w_case, zip_path, predict), per_fire=True, reference=False)
        else:
            notes.append(
                "case_holdout is empty for wildfire (splits/ndws.json extra['case_holdout'] has 0 items), so "
                "results has no 'case_holdout' key."
            )
    return HazardEval(
        hazard="wildfire",
        primary_metric="prauc",
        higher_is_better=True,
        results=results,
        ablations=ablations,
        dl_models=[DL_NAME],
        notes=notes,
        figures=figures,
    )


def _reference() -> ModelMetrics:
    nan = float("nan")
    return ModelMetrics("ndws_published", "reference", {"prauc": CI(PUBLISHED_NDWS_PRAUC, nan, nan, 0, 0)})


# ------------------------------------------------------------------ self-check (val only)
def _selfcheck() -> None:
    """Binned metrics vs exact ``eval.metrics`` / sklearn on the NDWS val rows (val only)."""
    sp = read_split(get_paths().splits / "ndws.json")
    predict = _Predictor()
    accs = _ndws_eval("eval", "val", sp, predict)
    x_path, y_path, ids = ndws_files("eval")
    rows = np.flatnonzero(split_mask(ids, sp, "val"))
    y = np.asarray(np.load(y_path, mmap_mode="r")[rows])
    x = np.asarray(np.load(x_path, mmap_mode="r")[rows], dtype=np.float32)
    probs = {
        m: np.concatenate([predict(x[i : i + 256])[m] for i in range(0, len(x), 256)]) for m in ALL_MODELS
    }
    for m in ALL_MODELS:
        a = accs[m]
        exact = (
            M.auprc(y.reshape(-1), probs[m].reshape(-1)),
            M.brier(y.reshape(-1), probs[m].reshape(-1), mask=(y.reshape(-1) != -1)),
            M.ece(y.reshape(-1), probs[m].reshape(-1)),
        )
        binned = (_ap(a.hist[0].sum(0)), float(a.sse.sum() / a.n.sum()), _ece_from(a.rel.sum(0)))
        print(m, "exact", np.round(exact, 5), "binned", np.round(binned, 5))
        assert abs(exact[0] - binned[0]) < 2e-3 and abs(exact[1] - binned[1]) < 1e-6
    print(json.dumps({"ok": True}))


if __name__ == "__main__":
    _selfcheck()
