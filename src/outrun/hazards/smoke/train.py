"""Smoke CNN+MLP training (T18): fit on the frozen train samples, early-stop on val pinball loss. Test is never read.

``.venv/bin/python -m outrun train smoke [--no-hms]`` (needs OUTRUN_PROC; PYTORCH_ENABLE_MPS_FALLBACK=1).
Everything comes from ``hazards.smoke.build`` (PROC/smoke): the site patches (cycle, site, 8, 9, 9) stay in RAM and a
batch is an index lookup; tabular features are built once per split. ``SmokeData.load`` applies the frozen
``splits/smoke.json`` (``samples_for_split``), so held-out monitors, quarantine and 2024 never reach train or val.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from outrun.config import get_paths
from outrun.eval.splits import read_split
from outrun.hazards.base import load_checkpoint, save_checkpoint, select_device
from outrun.hazards.smoke.build import smoke_dir
from outrun.hazards.smoke.dataset import (
    SmokeFeatureConfig,
    is_smoke_episode,
    n_tabular,
    samples_for_split,
    tabular_features_batch,
)
from outrun.hazards.smoke.model import SmokeNet, pinball_loss

QUANTILES = (0.5, 0.9)


@dataclass
class SmokePart:
    """One split part: row indices into the shared patch store plus its tabular features and labels."""

    cycle: np.ndarray  # (N,) cycle index
    site: np.ndarray  # (N,) site index
    tab: np.ndarray  # (N, n_tabular) float32
    hist: np.ndarray  # (N, 6) raw PM2.5 history
    target: np.ndarray  # (N,) log1p PM2.5 at +6 h
    pm_target: np.ndarray  # (N,)
    background: np.ndarray  # (N,) 30-day median before issue (NaN early 2022)
    episode: np.ndarray  # (N,) HMS medium/heavy at the site at the target time (evaluation only)
    month: np.ndarray  # (N,) calendar month of the issue time
    site_id: np.ndarray  # (N,)

    def __len__(self) -> int:
        return len(self.cycle)


class SmokeData:
    """Patch store in RAM + parts for the requested splits."""

    def __init__(
        self, parts: tuple[str, ...] = ("train", "val"), cfg: SmokeFeatureConfig | None = None
    ) -> None:
        if any(p.startswith("test") for p in parts):
            raise ValueError("test parts are scored once, in T24, not here")
        d = smoke_dir()
        cfg = cfg or SmokeFeatureConfig()
        self.cfg = cfg
        self.patches = np.load(d / "patches.npy", mmap_mode="r")
        self.patches = np.ascontiguousarray(self.patches)  # ~5 GB float16 in RAM
        hms = np.load(d / "hms.npy")
        table = pd.read_parquet(d / "samples.parquet")
        table["row"] = np.arange(len(table))
        split = read_split(get_paths().splits / "smoke.json")
        self.parts: dict[str, SmokePart] = {}
        for part in parts:
            t = samples_for_split(table, split, part)
            cyc, site = t["cycle_idx"].to_numpy(), t["site_idx"].to_numpy()
            hist = t[[f"h{k}" for k in range(6)]].to_numpy(dtype=np.float32)
            times = pd.DatetimeIndex(t["issue_time"])
            tab = tabular_features_batch(hist, hms[cyc, site], times, cfg)
            self.parts[part] = SmokePart(
                cyc, site, tab, hist, t["target"].to_numpy(dtype=np.float32), t["pm_target"].to_numpy(dtype=np.float32),
                t["background"].to_numpy(dtype=np.float32), is_smoke_episode(hms[cyc + 1, site]), times.month.to_numpy(),
                t["site_id"].to_numpy(),
            )  # fmt: skip

    def batch(self, part: SmokePart, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.patches[part.cycle[idx], part.site[idx]].astype(np.float32)
        return torch.from_numpy(p), torch.from_numpy(part.tab[idx]), torch.from_numpy(part.target[idx])


def make_net(cfg: SmokeFeatureConfig, data: SmokeData | None = None, seed: int = 0) -> SmokeNet:
    torch.manual_seed(seed)
    net = SmokeNet(n_tabular=n_tabular(cfg))
    if data is not None:  # standardisation statistics from TRAIN only
        tr = data.parts["train"]
        idx = np.random.default_rng(seed).choice(len(tr), size=min(200_000, len(tr)), replace=False)
        p = data.patches[tr.cycle[idx], tr.site[idx]].astype(np.float32)
        net.set_scaler(
            p.mean(axis=(0, 2, 3)), p.std(axis=(0, 2, 3)), tr.tab[idx].mean(axis=0), tr.tab[idx].std(axis=0)
        )
    return net


def predict_log(
    net: SmokeNet, data: SmokeData, part: SmokePart, device: str = "cpu", batch_size: int = 8192
) -> np.ndarray:
    net.eval().to(device)
    out = []
    with torch.no_grad():
        for lo in range(0, len(part), batch_size):
            idx = np.arange(lo, min(lo + batch_size, len(part)))
            p, t, _ = data.batch(part, idx)
            out.append(net(p.to(device), t.to(device)).cpu().numpy())
    return np.concatenate(out)


def train_smoke(
    out_dir: Path,
    *,
    use_hms: bool = True,
    batch_size: int = 2048,
    device: str = "cpu",
    resume: bool = False,
    max_epochs: int = 25,
    patience: int = 4,
    lr: float = 2e-3,
    seed: int = 0,
):
    """Plain PyTorch loop: Adam, pinball loss on log1p PM2.5, best.pt on val loss, last.pt every epoch (resume)."""
    cfg = SmokeFeatureConfig(use_hms=use_hms)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = SmokeData(("train", "val"), cfg)
    tr, va = data.parts["train"], data.parts["val"]
    print(f"smoke: {len(tr)} train / {len(va)} val samples, {len(set(tr.site_id))} / {len(set(va.site_id))} monitors, "
          f"use_hms={use_hms}, device={device}", flush=True)  # fmt: skip
    net = make_net(cfg, data, seed)
    # epoch-1 reference: constant train quantiles of the log target, scored on val
    const = torch.tensor(np.quantile(tr.target, QUANTILES), dtype=torch.float32)
    ref = float(pinball_loss(const.repeat(len(va), 1), torch.from_numpy(va.target), QUANTILES))
    print(f"constant-quantile pinball on val: {ref:.5f}", flush=True)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    start, best, best_epoch, bad, history = 1, math.inf, 0, 0, []
    last = out_dir / "last.pt"
    if resume and last.exists():
        st = load_checkpoint(last, net, opt)
        start, best, best_epoch, bad, history = (
            int(st["epoch"]) + 1,
            float(st["best_val"]),
            int(st["best_epoch"]),
            int(st["bad"]),
            list(st["history"]),
        )
    rng = np.random.default_rng(seed)
    for epoch in range(start, max_epochs + 1):
        t0 = time.perf_counter()
        net.train()
        order = rng.permutation(len(tr))
        tot, n = 0.0, 0
        for lo in range(0, len(order), batch_size):
            idx = np.sort(order[lo : lo + batch_size])
            p, t, y = (a.to(device) for a in data.batch(tr, idx))
            opt.zero_grad()
            loss = pinball_loss(net(p, t), y, QUANTILES)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            opt.step()
            tot, n = tot + float(loss.detach()) * len(idx), n + len(idx)
        pv = predict_log(net, data, va, device)
        val = float(pinball_loss(torch.from_numpy(pv), torch.from_numpy(va.target), QUANTILES))
        net.to(device)
        history.append(
            {"epoch": epoch, "train_loss": tot / n, "val_loss": val, "seconds": time.perf_counter() - t0}
        )
        print(f"epoch {epoch}: {history[-1]}", flush=True)
        if val < best:
            best, best_epoch, bad = val, epoch, 0
            save_checkpoint(out_dir / "best.pt", net, epoch=epoch, val_loss=val, use_hms=use_hms)
        else:
            bad += 1
        save_checkpoint(
            last, net, opt, epoch=epoch, best_val=best, best_epoch=best_epoch, bad=bad, history=history
        )
        if bad >= patience:
            break
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best,
                "constant_quantile_val": ref,
                "use_hms": use_hms,
                "history": history,
            },
            indent=2,
        )
    )
    return net, {"best_epoch": best_epoch, "best_val_loss": best, "epochs": len(history)}


def load_best(out_dir: Path, use_hms: bool = True) -> SmokeNet:
    net = SmokeNet(n_tabular=n_tabular(SmokeFeatureConfig(use_hms=use_hms)))
    load_checkpoint(out_dir / "best.pt", net)
    return net.eval()


__all__ = ["SmokeData", "train_smoke", "load_best", "predict_log", "select_device"]
