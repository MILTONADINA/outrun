"""Wildfire U-Net training on NDWS train; early stopping on NDWS val loss only (BCE + Dice, uncertain pixels ignored).

The NDWS test files and every WSTS fire are never read here (held-out discipline, DESIGN 6.3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from outrun.config import get_paths

from .dataset import NdwsDataset, NormStats, fit_norm_stats
from .model import FireUNet, fire_loss

PREVALENCE_FLOOR = 1e-3


def ndws_files(split_file: str) -> tuple[Path, Path, list[str]]:
    """(x path, label path, row ids) of one NDWS file split in PROC (``train``, ``eval`` or ``test``)."""
    d = get_paths().proc / "ndws"
    ids = json.loads((d / f"ndws_{split_file}_ids.json").read_text())
    return d / f"ndws_{split_file}.npy", d / f"ndws_{split_file}_labels.npy", ids


def save_stats(stats: NormStats, path: Path) -> None:
    np.savez(path, lo=stats.lo, hi=stats.hi, mean=stats.mean, std=stats.std)


def load_stats(path: Path) -> NormStats:
    with np.load(path) as z:
        return NormStats(lo=z["lo"], hi=z["hi"], mean=z["mean"], std=z["std"])


def fit_train_stats(x_path: Path, ids: list[str]) -> NormStats:
    """Percentile clip + mean/std from the NDWS TRAIN rows (all of them, read once)."""
    from outrun.eval.splits import read_split, split_mask

    split = read_split(get_paths().splits / "ndws.json")
    rows = np.flatnonzero(split_mask(ids, split, "train"))
    return fit_norm_stats(np.load(x_path, mmap_mode="r")[rows])


def train_wildfire(
    out_dir: str | Path,
    *,
    batch_size: int = 32,
    device: str = "cpu",
    resume: bool = False,
    max_epochs: int = 30,
    patience: int = 5,
    seed: int = 0,
    lr: float = 1e-3,
    num_workers: int = 4,
    max_batches_per_epoch: int | None = None,
    base: int = 32,
):
    """Train ``FireUNet`` with the shared loop in ``outrun.hazards.base`` (checkpoint every epoch, early stop on val)."""
    from torch.utils.data import DataLoader

    from outrun.eval.splits import read_split
    from outrun.hazards.base import TrainConfig, train_model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = read_split(get_paths().splits / "ndws.json")
    x_tr, y_tr, ids_tr = ndws_files("train")
    x_va, y_va, ids_va = ndws_files(
        "eval"
    )  # the official "eval" file is our validation set; "test" is not opened
    stats_path = out_dir / "norm_stats.npz"
    if resume and stats_path.exists():
        stats = load_stats(stats_path)
    else:
        stats = fit_train_stats(x_tr, ids_tr)
        save_stats(stats, stats_path)
    train_ds = NdwsDataset.from_split(x_tr, y_tr, ids_tr, split, "train", stats, augment=True, seed=seed)
    val_ds = NdwsDataset.from_split(x_va, y_va, ids_va, split, "val", stats)
    ytr = np.asarray(train_ds.y[train_ds.rows[::10]])
    yva = np.asarray(val_ds.y[val_ds.rows])
    prev_tr = float((ytr[ytr != -1] == 1).mean())
    print(
        f"wildfire: {len(train_ds)} train / {len(val_ds)} val samples; fire prevalence train {prev_tr:.4f} "
        f"val {float((yva[yva != -1] == 1).mean()):.4f}; unknown pixels train {(ytr == -1).mean():.4f} "
        f"val {(yva == -1).mean():.4f}; device={device}",
        flush=True,
    )
    torch.manual_seed(seed)
    model = FireUNet(base=base)
    with torch.no_grad():  # start at the base rate (pos_weight 3 shifts it up a little)
        p = min(max(prev_tr, PREVALENCE_FLOOR), 1 - PREVALENCE_FLOOR)
        model.head.bias.fill_(float(np.log(p / (1 - p))))
    cfg = TrainConfig(
        checkpoint_dir=out_dir,
        max_epochs=max_epochs,
        device=device,
        resume=resume,
        patience=patience,
        seed=seed,
        lr=lr,
        weight_decay=1e-4,
        grad_clip=1.0,
        max_batches_per_epoch=max_batches_per_epoch,
    )
    kw = {"num_workers": num_workers, "persistent_workers": num_workers > 0}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, **kw)
    result = train_model(model, train_loader, val_loader, fire_loss, cfg)
    return model, result
