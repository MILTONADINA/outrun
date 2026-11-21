"""Ice MIL-CNN training (T16): fit on the frozen train bags, early-stop on val loss. Test bags are never read.

``.venv/bin/python -m outrun train ice`` (needs OUTRUN_PROC; run with PYTORCH_ENABLE_MPS_FALLBACK=1).
The whole HRRR f01 crop (9 fields, float16, about 15 GB) is held in RAM; a bag is two 64x64 windows of it
(cycle ws+3 h as time t, cycle ws as t-3 h) around the zone centroid, plus elevation and the zone mask.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from outrun.config import get_paths
from outrun.hazards.ice.build import ARRAY
from outrun.hazards.ice.dataset import (
    ICE_CROP,
    ICE_FIELDS,
    N_ICE_CHANNELS,
    bags_for_split,
    normalize_bag_input,
)
from outrun.hazards.ice.model import IceMILNet, mil_pool


class ZoneLogitNet(nn.Module):
    """IceMILNet + LogSumExp over the in-zone cells (mask = input channel 19): one zone logit per bag."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.net = IceMILNet(N_ICE_CHANNELS, base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return mil_pool(self.net(x), x[:, 19] > 0.5)


class BagStore:
    """HRRR crop in RAM + per-zone crop windows; ``batch(bags, rows)`` -> normalised (B, 20, 64, 64) float32."""

    def __init__(self, bags_dir: Path) -> None:
        import zarr

        root = zarr.open_group(get_paths().proc / "hrrr" / "ice" / "data.zarr", mode="r")
        self.fields = np.stack([np.asarray(root[ARRAY[f]][:]) for f in ICE_FIELDS])  # (9, T, H, W) float16
        zones = np.load(bags_dir / "zones.npz")
        h, w = (int(v) for v in zones["grid_shape"])
        elev = np.load(bags_dir / "elevation.npy").astype(np.float32)
        self.zindex = {z: i for i, z in enumerate(zones["zone_ids"])}
        self.crops, self.masks = [], []
        c = ICE_CROP
        for k in range(len(zones["zone_ids"])):
            cells = zones["cells"][zones["offsets"][k] : zones["offsets"][k + 1]]
            r, col = np.divmod(cells, w)
            cr, cc = round(float(r.mean())), round(float(col.mean()))
            r0, c0 = min(max(cr - c // 2, 0), max(0, h - c)), min(max(cc - c // 2, 0), max(0, w - c))
            m = np.zeros((h, w), bool)
            m[r, col] = True
            self.crops.append((r0, c0))
            self.masks.append((m[r0 : r0 + c, c0 : c0 + c], elev[r0 : r0 + c, c0 : c0 + c]))

    def batch(self, bags: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
        c = ICE_CROP
        x = np.zeros((len(rows), N_ICE_CHANNELS, c, c), np.float32)
        z, it, im = (bags[k].to_numpy()[rows] for k in ("zone_index", "i_t", "i_tm3"))
        for j in range(len(rows)):
            r0, c0 = self.crops[z[j]]
            x[j, :9] = self.fields[:, it[j], r0 : r0 + c, c0 : c0 + c]
            x[j, 9:18] = self.fields[:, im[j], r0 : r0 + c, c0 : c0 + c]
            x[j, 18] = self.masks[z[j]][1]
            x[j, 19] = self.masks[z[j]][0]
        return normalize_bag_input(x)


def load_bags(part_names=("train", "val")) -> tuple[Path, dict[str, pd.DataFrame]]:
    from outrun.eval.splits import read_split

    d = get_paths().proc / "hrrr" / "ice" / "bags"
    bags = pd.read_parquet(d / "bags.parquet")
    zones = np.load(d / "zones.npz")
    bags["zone_index"] = bags["zone_id"].map({z: i for i, z in enumerate(zones["zone_ids"])})
    split = read_split(Path("splits/ice.json"))
    return d, {p: bags_for_split(bags, split, p) for p in part_names}  # test is never requested here


class _Loader:
    """Re-iterable batches of (x, label); ``shuffle`` reshuffles on every pass (seeded)."""

    def __init__(self, store, bags, batch_size, shuffle, seed=0, max_rows=None):
        self.store, self.bags, self.bs, self.shuffle = store, bags, batch_size, shuffle
        self.rng, self.max_rows = np.random.default_rng(seed), max_rows
        self.y = bags["label"].to_numpy(np.float32)

    def __iter__(self):
        n = len(self.bags)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        if self.max_rows:
            order = order[: self.max_rows]
        for i in range(0, len(order), self.bs):
            rows = order[i : i + self.bs]
            yield torch.from_numpy(self.store.batch(self.bags, rows)), torch.from_numpy(self.y[rows])


def zone_probabilities(model: nn.Module, loader, device: str) -> np.ndarray:
    """sigmoid(zone logit) for every bag of an unshuffled loader (eval mode)."""
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
    return np.concatenate(out)


def train_ice(
    out_dir: str | Path,
    *,
    batch_size: int = 64,
    device: str = "cpu",
    resume: bool = False,
    max_epochs: int = 12,
    patience: int = 3,
    seed: int = 0,
    lr: float = 1e-3,
    max_batches_per_epoch: int | None = None,
    base: int = 32,
):
    import torch.nn.functional as F

    from outrun.hazards.base import TrainConfig, train_model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d, parts = load_bags()
    print(
        f"loading HRRR crop into RAM ... train {len(parts['train'])} val {len(parts['val'])} bags", flush=True
    )
    store = BagStore(d)
    torch.manual_seed(seed)
    model = ZoneLogitNet(base)
    train_loader = _Loader(store, parts["train"], batch_size, True, seed)
    val_loader = _Loader(store, parts["val"], 256, False)
    loss_fn = lambda logit, y: F.binary_cross_entropy_with_logits(logit, y)  # noqa: E731
    cfg = TrainConfig(
        checkpoint_dir=out_dir, max_epochs=max_epochs, patience=patience, lr=lr, weight_decay=1e-4,
        device=device, resume=resume, seed=seed, max_batches_per_epoch=max_batches_per_epoch, grad_clip=5.0,
    )  # fmt: skip
    result = train_model(model, train_loader, val_loader, loss_fn, cfg)
    (out_dir / "train_history.json").write_text(json.dumps(result.history, indent=2) + "\n")
    return model, result


if __name__ == "__main__":
    import sys

    _, r = train_ice(get_paths().models / "ice", device=sys.argv[1] if len(sys.argv) > 1 else "cpu")
    print(r.best_epoch, r.best_val_loss)
