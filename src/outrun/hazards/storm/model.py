"""Storm U-Net (STORM-04), exceedance targets / intensity weights (STORM-03), loss and training step (STORM-05)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

VIL_THRESHOLDS: tuple[int, ...] = (74, 133, 181)
RISK_THRESHOLD = 133
RISK_THRESHOLD_INDEX = 1


_WEIGHT_BREAKS = (16, 74, 133, 160, 181)  # SEVIR VIL breakpoints for the intensity weights
_WEIGHT_VALUES = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0)


def exceedance_targets(y: torch.Tensor) -> torch.Tensor:
    """``(..., 12, H, W)`` VIL units -> float32 ``(..., 12, 3, H, W)`` of ``y >= VIL_THRESHOLDS[k]`` (STORM-03)."""
    thresholds = torch.as_tensor(VIL_THRESHOLDS, dtype=y.dtype, device=y.device)
    # insert a threshold axis after the lead axis: (..., 12, 1, H, W) >= (3, 1, 1)
    return (y.unsqueeze(-3) >= thresholds.view(-1, 1, 1)).to(torch.float32)


def intensity_weights(y: torch.Tensor) -> torch.Tensor:
    """SEVIR-style weights 1/2/5/10/20/30 at VIL breaks 16/74/133/160/181 (STORM-03)."""
    # number of breaks that y has reached (0..5) indexes straight into the weight table
    breaks = torch.as_tensor(_WEIGHT_BREAKS, dtype=y.dtype, device=y.device)
    level = (y.unsqueeze(-1) >= breaks).sum(dim=-1)
    table = torch.as_tensor(_WEIGHT_VALUES, dtype=torch.float32, device=y.device)
    return table[level]


def _double_conv(c_in: int, c_out: int) -> nn.Sequential:
    """Two (3x3 conv, batch norm, ReLU) layers: the basic U-Net building block."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class StormUNet(nn.Module):
    """2D U-Net, time as channels (STORM-04).

    ``forward(x (B, in_channels, H, W)) -> (vil (B, n_out, H, W), logits (B, n_out, n_thresholds, H, W))``.
    ValueError if H/W not divisible by ``2**levels`` or the channel count is wrong.

    Layout: ``inc`` (width ``base``), ``levels`` encoder stages that each halve H, W and double the width
    (``base * 2**k``), then ``levels`` decoder stages that upsample, concatenate the skip and halve the width.
    """

    def __init__(
        self, in_channels: int = 26, n_out: int = 12, n_thresholds: int = 3, base: int = 32, levels: int = 4
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_out = n_out
        self.n_thresholds = n_thresholds
        self.levels = levels
        widths = [base * 2**k for k in range(levels + 1)]  # base, 2 base, ..., base * 2**levels
        self.inc = _double_conv(in_channels, widths[0])
        self.down = nn.ModuleList(_double_conv(widths[k], widths[k + 1]) for k in range(levels))
        self.up_conv = nn.ModuleList(
            nn.ConvTranspose2d(widths[k + 1], widths[k], 2, stride=2) for k in reversed(range(levels))
        )
        self.up = nn.ModuleList(_double_conv(2 * widths[k], widths[k]) for k in reversed(range(levels)))
        self.vil_head = nn.Conv2d(base, n_out, 1)
        self.logit_head = nn.Conv2d(base, n_out * n_thresholds, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (B, {self.in_channels}, H, W), got {tuple(x.shape)}")
        step = 2**self.levels
        if x.shape[2] % step or x.shape[3] % step:
            raise ValueError(f"H and W must be divisible by {step}, got {tuple(x.shape[2:])}")
        h = self.inc(x)
        skips = [h]
        for block in self.down:
            h = block(F.max_pool2d(h, 2))
            skips.append(h)
        skips.pop()  # the deepest feature map is h itself
        for upsample, block in zip(self.up_conv, self.up, strict=True):
            h = block(torch.cat([skips.pop(), upsample(h)], dim=1))
        vil = self.vil_head(h)
        logits = self.logit_head(h).unflatten(1, (self.n_out, self.n_thresholds))
        return vil, logits


def predict_proba(model: StormUNet, x: torch.Tensor) -> torch.Tensor:
    """Eval mode, no grad: ``sigmoid(logits)`` ``(B, n_out, n_thresholds, H, W)`` in [0, 1] (STORM-04)."""
    model.eval()
    with torch.no_grad():
        _, logits = model(x)
    return torch.sigmoid(logits)


def storm_loss(
    vil_pred: torch.Tensor, logits: torch.Tensor, y: torch.Tensor, bce_weight: float = 1.0
) -> torch.Tensor:
    """Weighted MSE + MAE on ``y/255`` plus BCE on exceedance targets (STORM-05)."""
    target = y / 255.0
    w = intensity_weights(y)
    err = vil_pred - target
    regression = (w * err**2).mean() + (w * err.abs()).mean()
    bce = F.binary_cross_entropy_with_logits(logits, exceedance_targets(y))
    return regression + bce_weight * bce


def train_step(
    model: StormUNet, batch: tuple[torch.Tensor, torch.Tensor], optimizer: torch.optim.Optimizer
) -> float:
    """One optimisation step; returns the loss before the step (STORM-05)."""
    x, y = batch
    model.train()
    optimizer.zero_grad()
    vil, logits = model(x)
    loss = storm_loss(vil, logits, y)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def train_storm(
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    out_dir: str | Path,
    *,
    use_ir: bool = True,
    batch_size: int = 16,
    device: str = "cpu",
    resume: bool = False,
    max_epochs: int = 20,
    patience: int = 3,
    seed: int = 0,
    log: Callable[[str], None] = print,
):
    """Train the storm U-Net with the shared loop in ``outrun.hazards.base`` (val-loss early stopping, checkpoint
    every epoch). ``train_ds``/``val_ds`` are ``SevirWindowDataset``s built from the train/val split only.
    Returns ``(model, TrainResult)``; the best weights are at ``out_dir/best.pt``."""
    from torch.utils.data import DataLoader

    from outrun.hazards.base import TrainConfig, train_model

    model = StormUNet(in_channels=26 if use_ir else 13)
    cfg = TrainConfig(
        checkpoint_dir=Path(out_dir),
        max_epochs=max_epochs,
        device=device,
        resume=resume,
        patience=patience,
        seed=seed,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    def loss_fn(output: tuple[torch.Tensor, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
        return storm_loss(output[0], output[1], y)

    result = train_model(model, train_loader, val_loader, loss_fn, cfg)
    for row in result.history:
        log(str(row))
    return model, result
