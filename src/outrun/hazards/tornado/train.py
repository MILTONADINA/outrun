"""Tornado CNN training: train on train, early-stop on val loss only (focal loss). Test chips are never read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .dataset import TornetChips, keep_indices, load_meta, tornet_root
from .features import fit_channel_stats
from .model import TornadoCNN, focal_loss

PREVALENCE_FLOOR = 1e-3


def fit_norm_stats(
    n_chips: int = 1200, seed: int = 0, root: Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Channel mean/std from a random subset of TRAIN chips (25 stored channels; coordinate channels unused)."""
    root = Path(root) if root is not None else tornet_root()
    meta = load_meta("train", root)
    keep = keep_indices("train", meta)
    pick = np.sort(np.random.default_rng(seed).choice(keep, size=min(n_chips, len(keep)), replace=False))
    x = np.load(root / "train_x.npy", mmap_mode="r")
    sample = np.stack([np.asarray(x[i], dtype=np.float32) for i in pick])
    return fit_channel_stats(sample)


def train_tornado(
    out_dir: str | Path,
    *,
    include_range: bool = True,
    batch_size: int = 32,
    device: str = "cpu",
    resume: bool = False,
    max_epochs: int = 20,
    patience: int = 5,
    seed: int = 0,
    lr: float = 1e-4,
    num_workers: int = 6,
    max_batches_per_epoch: int | None = None,
    gamma: float = 2.0,
    alpha: float | None = None,
    root: Path | None = None,
):
    """Train the CNN with the shared loop in ``outrun.hazards.base`` (checkpoint every epoch, early stop on val loss).

    Saves ``norm_stats.npz`` (train statistics) next to ``best.pt``. Returns ``(model, TrainResult)``.
    """
    from torch.utils.data import DataLoader

    from outrun.hazards.base import TrainConfig, train_model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / "norm_stats.npz"
    if resume and stats_path.exists():
        with np.load(stats_path) as z:
            mean, std = z["mean"], z["std"]
    else:
        mean, std = fit_norm_stats(root=root, seed=seed)
        np.savez(stats_path, mean=mean, std=std)
    train_ds = TornetChips("train", mean, std, include_range=include_range, root=root)
    val_ds = TornetChips("val", mean, std, include_range=include_range, root=root)
    prevalence = float(train_ds.labels[train_ds.indices].mean())
    print(
        f"tornado: {len(train_ds)} train (prevalence {prevalence:.4f}) / {len(val_ds)} val chips "
        f"(prevalence {val_ds.labels.mean():.4f}), include_range={include_range}, device={device}",
        flush=True,
    )
    torch.manual_seed(seed)
    model = TornadoCNN(in_channels=27 if include_range else 26)
    with torch.no_grad():  # start at the base rate so the first epochs are not spent learning the offset
        p = min(max(prevalence, PREVALENCE_FLOOR), 1 - PREVALENCE_FLOOR)
        model.head[-1].bias.fill_(float(np.log(p / (1 - p))))
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
    loader_kw = {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 4 if num_workers else None,
    }
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, **loader_kw)

    def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return focal_loss(logits, y, gamma=gamma, alpha=alpha)

    result = train_model(model, train_loader, val_loader, loss_fn, cfg)
    return model, result
