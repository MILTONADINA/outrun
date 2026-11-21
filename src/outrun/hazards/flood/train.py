"""Flood U-Net training on the frozen train split; early stopping on val focal loss only (FLOOD-04, FLOOD-05, FLOOD-13).

Test samples are never read: ``FloodPatchDataset.from_split`` keeps only rows of the requested part. Train and val are
both drawn with the FLOOD-13 balanced sampler (1/3 positive, 1/3 hard negative, 1/3 wet; DRY never), once, with a
fixed seed, so the epoch-1 train-vs-val losses are comparable. Pixels with valid == 0 are ignored like the ring.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from outrun.config import get_paths

from .dataset import FloodPatchDataset, SampleKind, balanced_indices
from .model import FloodUNet, focal_loss

PREVALENCE_FLOOR = 1e-3


class MaskedPatches(Dataset):
    """Wrap a FloodPatchDataset: target -1 (ignored) where the ``valid`` channel (always last) is 0."""

    def __init__(self, base: FloodPatchDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        return x, torch.where(x[-1:] > 0, y, torch.full_like(y, -1.0))


def flood_files() -> tuple[Path, Path, list[str], list[str]]:
    d = get_paths().proc / "flood"
    return (
        d / "x.npy",
        d / "y.npy",
        json.loads((d / "ids.json").read_text()),
        json.loads((d / "kinds.json").read_text()),
    )


def balanced_part(split, part: str, use_flash: bool = True, seed: int = 0):
    """(dataset restricted to ``part`` and re-balanced by FLOOD-13, kinds of its rows)."""
    x, y, ids, kinds = flood_files()
    ds = FloodPatchDataset.from_split(x, y, ids, split, part, use_flash)
    k = [SampleKind(kinds[r]) for r in ds.rows]
    pick = balanced_indices(k, len(k), np.random.default_rng(seed))
    return MaskedPatches(ds), Subset(MaskedPatches(ds), pick.tolist()), [k[i] for i in pick]


def train_flood(
    out_dir: str | Path,
    *,
    batch_size: int = 64,
    device: str = "cpu",
    resume: bool = False,
    max_epochs: int = 30,
    patience: int = 5,
    seed: int = 0,
    lr: float = 1e-3,
    num_workers: int = 4,
    max_batches_per_epoch: int | None = None,
    base: int = 32,
    use_flash: bool = True,
):
    from outrun.eval.splits import read_split
    from outrun.hazards.base import TrainConfig, train_model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split = read_split(get_paths().splits / "flood.json")
    _, train_ds, ktr = balanced_part(split, "train", use_flash, seed)
    _, val_ds, kva = balanced_part(split, "val", use_flash, seed)
    ytr = np.concatenate([train_ds[i][1].numpy().ravel() for i in range(0, len(train_ds), 10)])
    yva = np.concatenate([val_ds[i][1].numpy().ravel() for i in range(0, len(val_ds), 10)])
    prev_tr, prev_va = float((ytr[ytr != -1] == 1).mean()), float((yva[yva != -1] == 1).mean())

    def count(ks):
        return {
            k.value: sum(1 for x in ks if x == k)
            for k in (SampleKind.POSITIVE, SampleKind.HARD_NEGATIVE, SampleKind.WET)
        }

    print(
        f"flood: {len(train_ds)} train {count(ktr)} / {len(val_ds)} val {count(kva)} patches; pixel prevalence "
        f"train {prev_tr:.4f} val {prev_va:.4f}; ignored pixels train {(ytr == -1).mean():.4f} val {(yva == -1).mean():.4f}; "
        f"device={device} use_flash={use_flash}",
        flush=True,
    )
    torch.manual_seed(seed)
    model = FloodUNet(in_channels=11 if use_flash else 8, base=base)
    with torch.no_grad():  # start at the base rate
        p = min(max(prev_tr, PREVALENCE_FLOOR), 1 - PREVALENCE_FLOOR)
        model.head.bias.fill_(float(np.log(p / (1 - p))))
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "use_flash": use_flash,
                "base": base,
                "seed": seed,
                "lr": lr,
                "batch_size": batch_size,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "prev_train": prev_tr,
                "prev_val": prev_va,
            },
            indent=1,
        )
    )
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
    result = train_model(model, train_loader, val_loader, focal_loss, cfg)
    return model, result
