# 08: Severe storms (SEVIR VIL + GOES-16 IR107 nowcast)

Modules: `src/outrun/hazards/storm/` (`dataset.py`, `model.py`, `baselines.py`, `calibration.py`, `metrics.py`,
`adapter.py`). Unit tests: `tests/unit/test_storm*.py`. Design: DESIGN 3.1, 1.2, 6.1.

Purpose: a 0-60 min nowcast of storm intensity (VIL) from 13 past frames of radar VIL and satellite IR107, turned
into a calibrated probability layer P(VIL >= 133) on GRID for routing. The baselines are persistence and an
optical-flow extrapolation (pySTEPS when installed). The VIL+IR vs VIL-only ablation measures what the satellite
channel adds.

Conventions used below:
- **Event array**: the processed SEVIR memmap written by `DATA-01`, shape `(N, 49, 2, H, W)` uint8 (event,
  5-min frame, channel, row, col). Channel 0 = VIL (digital units 0..255), channel 1 = IR107 (brightness
  temperature 180-320 K mapped linearly to 0..255). Rows run north to south (north-up image).
- **VIL units**: thresholds 16/74/133/160/181/219 are in VIL digital units (0..255), as in SEVIR.
- **Leads**: 12 output frames at t+5..t+60 min; frame `i` (0-based) is lead `5 (i+1)` min.
- Only `adapter.py` imports `outrun.core`. The package `__init__.py` imports nothing.

"Must" marks requirements the proposal depends on (all of `STORM-01..STORM-08` are in the proposal coverage
matrix). Other requirements are "Should" (evaluation/support); they are still traced and tested.

## Requirements

### STORM-01 Training windows from the SEVIR event array (Must)
`dataset.py` constants: `N_FRAMES = 49`, `N_IN = 13`, `N_OUT = 12`, `OFFSETS = (0, 12, 24)`, `VIL_CH = 0`,
`IR_CH = 1`.

`window_index(n_events: int, offsets=OFFSETS) -> np.ndarray` returns an int array of shape
`(n_events * len(offsets), 2)` with rows `(event, offset)` in event-major order
(`[(0, 0), (0, 12), (0, 24), (1, 0), ...]`). `n_events = 0` gives shape `(0, 2)`.

`SevirWindowDataset(events, event_ids=None, use_ir=True, offsets=OFFSETS)` is a `torch.utils.data.Dataset`:
- `events`: array-like `(N, F, 2, H, W)` uint8 (an `np.memmap` in production). `event_ids` (default
  `range(N)`) selects the events that belong to this split.
- `len(ds) == len(event_ids) * len(offsets)`. Item `i` uses event `event_ids[i // len(offsets)]` and offset
  `o = offsets[i % len(offsets)]`.
- `ds[i]` returns `(x, y)`, both `torch.float32`:
  - `x[0:13] = events[e, o:o+13, VIL_CH] / 255` (inputs t-60..t, scaled to [0, 1]);
  - `y = events[e, o+13:o+25, VIL_CH]` as float32 **in VIL digital units** (not scaled), shape `(12, H, W)`.
- `ds[i]` for `i < 0` or `i >= len(ds)` raises `IndexError`.
- Construction raises `ValueError` if `events.ndim != 5`, if `events.shape[1] < max(offsets) + 25`, or if
  `use_ir=True` and `events.shape[2] < 2`.
- `SevirWindowDataset.from_split(events, event_ids: Sequence[str], split, part, use_ir=True)` (classmethod) is the
  training entry point: `event_ids[k]` is the SEVIR event id of row `k` of `events` (length `N`, else
  `ValueError`). It equals `SevirWindowDataset(events, event_ids=np.flatnonzero(split_mask(event_ids, split, part)),
  use_ir=use_ir)` (SPLIT-07) and sets `ds.ids` = the selected event ids in row order. Quarantined and test events
  never appear in a train or val dataset.

### STORM-02 Satellite input: GOES-16 IR107 channels and the VIL-only ablation switch (Must)
- With `use_ir=True`, `x` has `26` channels: `x[0:13]` VIL as in STORM-01 and
  `x[13:26] = events[e, o:o+13, IR_CH] / 255`.
- With `use_ir=False` (the VIL-only ablation), `x` has `13` channels and equals `x[0:13]` of the `use_ir=True`
  item for the same index; `y` is identical. Both arms built from the same `event_ids` have the same length (the
  ablation trains on the identical event subset).
- `in_channels(use_ir: bool) -> int` returns `26` or `13`.
- `events_with_both(catalog: pd.DataFrame) -> list[str]` takes SEVIR `CATALOG.csv` rows (columns `id`,
  `img_type`) and returns the sorted unique `id`s that have both an `img_type == "vil"` row and an
  `img_type == "ir107"` row. Events missing either are dropped from both arms. Missing columns raise `KeyError`.

### STORM-03 Exceedance targets and intensity weights (Must)
`model.py` constants: `VIL_THRESHOLDS = (74, 133, 181)`, `RISK_THRESHOLD = 133`, `RISK_THRESHOLD_INDEX = 1`
(the index of 133 in `VIL_THRESHOLDS`).
- `exceedance_targets(y: torch.Tensor) -> torch.Tensor`: `y` has shape `(..., 12, H, W)` in VIL units; returns
  float32 of shape `(..., 12, 3, H, W)` with `out[..., k, :, :] = (y >= VIL_THRESHOLDS[k])` as 0.0/1.0
  (threshold inclusive).
- `intensity_weights(y: torch.Tensor) -> torch.Tensor` (same shape as `y`, float32), SEVIR-style weights that
  emphasise intense cells: `1` for `y < 16`, `2` for `16 <= y < 74`, `5` for `74 <= y < 133`, `10` for
  `133 <= y < 160`, `20` for `160 <= y < 181`, `30` for `y >= 181`.

### STORM-04 Storm U-Net (PyTorch deep-learning model) (Must)
`StormUNet(in_channels=26, n_out=12, n_thresholds=3, base=32, levels=4)` is a `torch.nn.Module`: a 2D U-Net with
`levels` down/up-sampling stages (channel widths `base * 2**k`), time stacked as channels.
- Attributes `in_channels`, `n_out`, `n_thresholds`, `levels` keep the constructor values.
- `forward(x)` takes `(B, in_channels, H, W)` float32 and returns a tuple `(vil, logits)`:
  `vil` `(B, n_out, H, W)` is the VIL regression head in scaled units (VIL / 255); `logits`
  `(B, n_out, n_thresholds, H, W)` are exceedance logits for `VIL_THRESHOLDS`.
- `H` or `W` not divisible by `2**levels`, or `x.shape[1] != in_channels`, raises `ValueError`.
- `StormUNet(in_channels=13)` is the VIL-only ablation model.
- The default configuration (`base=32, levels=4, in_channels=26`) has between 4 and 12 million trainable
  parameters (DESIGN: about 8M).
- `predict_proba(model, x) -> torch.Tensor` puts the model in eval mode, runs without gradients and returns
  `sigmoid(logits)`, shape `(B, n_out, n_thresholds, H, W)`, every value in [0, 1].
- Determinism: two models built after the same `torch.manual_seed(s)` have identical weights, and in eval mode
  the same input gives identical outputs on repeated calls.

### STORM-05 Loss and training step (Must)
- `storm_loss(vil_pred, logits, y, bce_weight=1.0) -> torch.Tensor` (scalar):
  with `t = y / 255`, `w = intensity_weights(y)`:
  `mean(w * (vil_pred - t)**2) + mean(w * |vil_pred - t|) + bce_weight * BCEWithLogits_mean(logits, exceedance_targets(y))`.
  It is finite and `>= 0`; for `vil_pred == t` and logits of `+-20` matching the targets it is below `1e-3`.
- `train_step(model, batch, optimizer) -> float`: one optimisation step on `batch = (x, y)` (model in train
  mode, zero grads, `storm_loss`, backward, step); returns the loss before the step as a Python float.
- Repeating `train_step` on one fixed tiny batch drives the loss down (a sound model and loss can overfit one
  batch): after 40 steps of Adam (lr 1e-2) on a 2-sample 16x16 batch with `base=4, levels=2`, the loss is below
  70% of the first step's loss.

### STORM-06 Baselines: persistence and optical-flow extrapolation (pySTEPS) (Must)
`baselines.py`:
- `persistence(frames, n_out=12) -> np.ndarray`: `frames` `(T, H, W)`; returns float32 `(n_out, H, W)`, every
  slice equal to `frames[-1]`. `T == 0` raises `ValueError`.
- `persistence_mask(frames, threshold=133, n_out=12) -> np.ndarray`: float32 `(n_out, H, W)` of
  `frames[-1] >= threshold` as 0.0/1.0 (the skeleton rule baseline layer).
- `estimate_motion(frames) -> np.ndarray`: `frames` `(T >= 2, H, W)`; returns float32 `(2, H, W)` motion in
  pixels per frame in image axes: `[0]` = dx (+ toward increasing column), `[1]` = dy (+ toward increasing row).
  Backend: pySTEPS Lucas-Kanade when `pysteps` imports (`PYSTEPS_AVAILABLE`), otherwise an own numpy/scipy method
  (no error). For a rigidly translating smooth pattern (shift up to 2 px/frame), the median motion over pixels
  where `frames[-1] > 0.1 * frames[-1].max()` is within 0.35 px of the true shift in each axis. `T < 2` raises
  `ValueError`.
- `extrapolate(frame, motion, n_steps) -> np.ndarray`: backward semi-Lagrangian advection with bilinear
  interpolation; `out[k-1][r, c] = frame(r - k * dy, c - k * dx)` for `k = 1..n_steps` (motion constant in time),
  0 where the source point lies outside the frame. Returns float32 `(n_steps, H, W)`. With constant integer motion
  the result is an exact shift (within 1e-5).
- `extrapolation_nowcast(frames, n_out=12) -> np.ndarray` = `extrapolate(frames[-1], estimate_motion(frames),
  n_out)` clipped to `>= 0`, float32 `(n_out, H, W)`. A Gaussian blob moving 1 px/frame to the right is
  forecast with its peak within 1.5 px of the true position at step 6.

### STORM-07 Per-lead calibration and 15-minute slotting (Must)
`calibration.py` (may delegate to `outrun.core.calibrate` isotonic regression):
- `PerLeadCalibrator().fit(probs, truth) -> self`: `probs` float `(N, L, ...)`, `truth` bool/0-1 of the same
  shape; fits one isotonic map per lead `l` (axis 1) on the flattened values of that lead, clipped to [0, 1]
  outside the fitted range.
- `transform(probs) -> np.ndarray` (same shape, float32, values in [0, 1]), applying lead `l`'s map to
  `probs[:, l]`. Each map is non-decreasing. Leads are independent (a lead whose truth is all 0 maps everything to
  0 while other leads are unaffected).
- `transform` before `fit` raises `RuntimeError`; a lead count different from the fitted one raises `ValueError`.
- `frames_to_slots(p, frames_per_slot=3) -> np.ndarray`: `p` `(F, ...)` (12 five-minute frames) returns
  `(F // frames_per_slot, ...)` where slot `s` is the element-wise **max** over frames
  `s*frames_per_slot .. (s+1)*frames_per_slot - 1`. `F % frames_per_slot != 0` raises `ValueError`.
  For 12 frames this gives 4 slots (0-15, 15-30, 30-45, 45-60 min).

### STORM-08 `StormAdapter`: RiskLayer, policy and beyond-horizon decay (Must)
`adapter.py`:
- `StormFrames` (dataclass): `vil` `(13, h, w)` uint8 (t-60..t), `ir` `(13, h, w)` uint8 or `None`,
  `future_vil` `(12, h, w)` uint8 or `None` (truth, t+5..t+60), `x` `(X,)` and `y` `(Y,)` GRID cell-centre
  coordinates in metres, `regrid: Callable[[np.ndarray], np.ndarray]` mapping `(k, h, w)` model-grid arrays to
  `(k, Y, X)` on GRID, `held_out: bool`.
- `StormAdapter(provider=None, model=None, calibrator=None, policy=None, source_tag="storm.unet_vil_ir@dev", *,
  checkpoint=None)` implements `outrun.core.risk.HazardAdapter`. `provider: Callable[[Scenario], StormFrames]`
  supplies the data (None: the real SEVIR/MRMS provider, imported lazily on first use); the adapter reads only
  `scn.t0` from the scenario itself. Every argument is optional and construction loads nothing, so `StormAdapter()`
  works offline (the registry stores zero-argument factories, RISK-13). With `model=None` and a `checkpoint` path,
  a `StormUNet` is loaded lazily on the first `predict`; with neither, `predict` raises `RuntimeError`.
- Module-level `default_adapter() -> StormAdapter` (zero arguments, loads nothing; the path is kept as `adapter.checkpoint`, a `Path`; `checkpoint =
  get_paths().models / "storm" / "best.pt"`) is registered with `REGISTRY.register(Hazard.STORM, default_adapter)`
  when `hazards.base` is implemented.
- `hazard is Hazard.STORM`. `policy` (default: `outrun.core.policy.load_policies()[Hazard.STORM]`, i.e. `configs/hazards.yaml` `storm`) has
  `block_at[p] is None` for every `Profile`, `speed_factor(p) = 1 - 0.3 p`, exposure weights from the config
  (civilian 2), `beyond_horizon == "decay"`, `decay_tau_min == 30`.
- `baseline(scn)`: persistence of the last-frame mask `vil[-1] >= 133` for all 4 in-horizon slots;
  `kind == "baseline"`.
- `predict(scn)`: `x = vil/255` (+ `ir/255` when `model.in_channels == 26`), `p = sigmoid(logits)[:, :,
  RISK_THRESHOLD_INDEX]` -> `calibrator.transform` (identity if `None`) -> `frames_to_slots` -> regrid;
  `kind == "forecast"`. No model raises `RuntimeError`; a 26-channel model with `ir is None` raises `ValueError`.
- `observed(scn)`: `None` if `future_vil is None`; else `frames_to_slots(future_vil >= 133)` regridded;
  `kind == "observed"`.
- Every returned `RiskLayer`: `hazard == Hazard.STORM`, `issued == scn.t0`,
  `valid_until == scn.t0 + 60 min` (native horizon from config), `source` starts with `"storm."`,
  `held_out == frames.held_out`, `edge_overrides is None`, `prob` an `xr.DataArray` with dims
  `("slot", "y", "x")`, shape `(12, Y, X)`, dtype float16, values in [0, 1].
- Slots 0-3 hold the in-horizon values. Slots 4-11 (after `valid_until`) are filled with the storm decay rule
  (`HORIZON-01`): the adapter passes its 4 valid slots to `outrun.core.policy.apply_beyond_horizon(prob, policy,
  t0=scn.t0, issued=scn.t0, valid_until=..., n_slots=12)` and does not re-implement the rule. Unlike tornado
  `observed`, storm `observed` gets the decay too (truth only exists for 60 min). `RiskStack.from_layers`
  re-applies the rule (RISK-02); applying it twice changes nothing. The
  test checks only the properties every reading of that rule satisfies: each cell in slots 4-11 is `>= 0`,
  `<=` its slot-3 value, and non-increasing from slot to slot.

### STORM-09 Motion export for the tornado swath (Should)
`baselines.py`:
- `motion_to_mps(motion_px, pixel_m=2000.0, frame_s=300.0) -> np.ndarray`: converts a `(2, H, W)` image-axis
  field (STORM-06) to metres per second in map axes `(u east, v north)`: `u = dx * pixel_m / frame_s`,
  `v = -dy * pixel_m / frame_s` (rows run north to south, so the row axis is flipped).
- `motion_at(motion_mps, row, col) -> tuple[float, float]` returns `(u, v)` at that pixel as Python floats;
  out-of-range indices raise `IndexError`.

### STORM-10 Nowcast scores: CSI, FSS, MSE per lead (Should)
`metrics.py` (may delegate to `outrun.eval.metrics`; the formulas here are the contract):
- `CSI_THRESHOLDS = (16, 74, 133, 160, 181, 219)`.
- `csi(pred, truth, threshold) -> float`: events are `>= threshold`; `hits / (hits + misses + false_alarms)`
  summed over all elements; `nan` when the denominator is 0. Shape mismatch raises `ValueError`.
- `fss(pred, truth, threshold, window) -> float`: binary fields `>= threshold` over the last two axes, fractions
  = mean over a `window x window` neighbourhood (zero padding outside); `FSS = 1 - sum((Pf - Po)**2) /
  (sum(Pf**2) + sum(Po**2))` over all elements; `nan` when the denominator is 0. Even or `< 1` window raises
  `ValueError`.
- `storm_scores(pred, truth, thresholds=CSI_THRESHOLDS, fss_threshold=133, fss_windows=(3, 7), crop=None)
  -> dict`: `pred`, `truth` `(N, 12, H, W)` in VIL units. Returns `{"csi": {thr: (12,) array}, "fss":
  {window: (12,) array}, "mse": (12,) array}`, one value per lead, counts pooled over `N`. `crop=c` scores only the
  central `c x c` pixels (edge-inflow check); `c` larger than `H` or `W` raises `ValueError`.
