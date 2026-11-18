# 11: Wildfire spread (NDWS training, WSTS 2021 out-of-time test)

Modules: `src/outrun/hazards/wildfire/{dataset,model,baselines,metrics,growth,adapter}.py`
Tests: `tests/unit/test_wildfire*.py`
Design: DESIGN 3.4 (data, preprocessing, model, evaluation, risk layer), 1.2 (interfaces, hold rule), 4.2 (costs).

Purpose: next-day fire-spread probability from 12 NDWS channels (GRIDMET weather, elevation, population, MODIS/VIIRS
NDVI and the previous active-fire mask) with a U-Net trained on NDWS and tested out of time on WildfireSpreadTS 2021.
For routing, the day-0 observed mask grows linearly hour by hour into the day-1 prediction.

## Shared conventions (this spec)

- A sample is `x (12, H, W)` in `NDWS_CHANNELS` order and a label `y (H, W)` with values `1` (fire next day),
  `0` (no fire) and `-1` (uncertain: ignored in loss, fitting and metrics). Batches add a leading `N`. NDWS uses
  64 x 64; every function also works on other sizes (tests use 16 x 16 and 32 x 32).
- **Image orientation:** row 0 is north, rows increase southward, columns increase eastward.
- **Wind direction** channel `th` is in degrees, meteorological "from" convention: the direction the wind blows
  **from**, clockwise from north, in `[0, 360)`. The wind blows **toward** `th + 180`. A unit step toward compass
  bearing `b` is `(d_row, d_col) = (-cos b, sin b)`.
- `PrevFireMask` values: `1` burning, `0` not burning, `-1` unknown (treated as not burning by baselines).
- The WSTS 12-channel mapper (`DATA-06`, `data/wsts.py`) must emit arrays in `NDWS_CHANNELS` order with the same
  units, so WSTS 2021 samples go through exactly the same code as NDWS samples. This spec only states that contract.
- Times are timezone-aware UTC `datetime`; a naive `datetime` raises `ValueError` wherever one is accepted.
- Policy numbers come from `configs/hazards.yaml` (`hazards.wildfire`); tests restate the DESIGN values.
- Must = required for the proposal ("wildfire spread", "deep learning", "real satellite data"). All FIRE-01..FIRE-08
  are Must.

## Requirements

### FIRE-01 NDWS channel layout and train-percentile normalisation (Must)
`dataset.NDWS_CHANNELS` is the tuple
`("elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "NDVI", "population", "erc", "PrevFireMask")`.

`dataset.NormStats` is a frozen dataclass of four `float32 (12,)` arrays `lo, hi, mean, std`.
`fit_norm_stats(x_train, lo_pct=0.1, hi_pct=99.9) -> NormStats` with `x_train (N, 12, H, W)`: per channel,
`lo`/`hi` are the `lo_pct`/`hi_pct` percentiles of the finite training values (numpy default linear interpolation, `np.percentile`); `mean`/`std` are the mean and
population std of the values clipped to `[lo, hi]`; a `std` of `0` is stored as `1`.
For `PrevFireMask` the stats are fixed to `lo=-1, hi=1, mean=0, std=1` (identity).

`normalize(x, stats) -> np.ndarray` (`(12, H, W)` or `(N, 12, H, W)`) returns a new `float32` array
`(clip(x, lo, hi) - mean) / std` per channel; the `PrevFireMask` channel is therefore passed through unchanged;
non-finite values become `0` after normalisation. The input is not modified. A channel axis other than 12 raises
`ValueError`. Normalisation is applied **after** augmentation (FIRE-03), because augmentation needs raw degrees.

### FIRE-02 Satellite input channels: MODIS/VIIRS NDVI and active-fire mask (Must)
`dataset.SATELLITE_CHANNELS = ("NDVI", "PrevFireMask")` (MODIS/VIIRS vegetation index and the satellite active-fire
mask in NDWS; VIIRS bands in WSTS). They are part of the model input:
- `channel_indices(names) -> list[int]` returns positions in `NDWS_CHANNELS` (`[8, 11]` for `SATELLITE_CHANNELS`);
  an unknown name raises `KeyError`.
- `check_channel_order(names)` returns `None` if `tuple(names) == NDWS_CHANNELS` and raises `ValueError` otherwise
  (wrong order, missing or extra channels): the guard every WSTS-mapped array passes before inference.
- `ablate_channels(x, names) -> np.ndarray` returns a copy of `(12, H, W)` / `(N, 12, H, W)` `x` with the named
  channels set to `0` (their normalised mean) and every other channel unchanged; the input is not modified.
  `ablate_channels(x, SATELLITE_CHANNELS)` is the no-satellite ablation.
- `model.FireUNet()` defaults to `in_channels == len(NDWS_CHANNELS) == 12`.

### FIRE-03 Flip/rot90 augmentation with wind direction rotated consistently (Must)
`dataset.augment(x, y, hflip=False, vflip=False, k=0) -> (x_aug, y_aug)` on a raw (un-normalised) sample
`x (12, H, W)`, `y (H, W)`, applied in this order: horizontal flip (columns reversed, `x[..., ::-1]`) if `hflip`,
then vertical flip (rows reversed) if `vflip`, then `np.rot90(·, k, axes=(-2, -1))` (counter-clockwise by
`90 k` degrees). Every channel and the label get the same spatial transform. The `th` channel values are also
transformed so that the wind keeps pointing the same way relative to the image content:
- horizontal flip: `th -> (360 - th) mod 360`;
- vertical flip: `th -> (180 - th) mod 360`;
- counter-clockwise rotation by `90 k`: `th -> (th - 90 k) mod 360`.
Output `th` is in `[0, 360)`. Consequence (tested for all 8 dihedral transforms): a feature one step downwind of a
burning pixel before augmentation is one step downwind (per the new `th`) of that pixel after augmentation.
Inputs are not modified; `k` is taken mod 4.

`random_augment(x, y, rng)` draws `hflip`, `vflip` (fair coins) and `k` in `{0,1,2,3}` from the
`np.random.Generator` and calls `augment`; same seed gives the same output.

### FIRE-04 Fire U-Net (PyTorch deep-learning model) (Must)
`model.FireUNet(in_channels=12, base=32, levels=3)` is a `torch.nn.Module` U-Net (encoder-decoder with skip
connections; `levels` downsampling steps by 2; `base` channels in the first stage, doubling per level).
- `forward(x)` maps `(B, in_channels, H, W)` float32 to logits `(B, 1, H, W)` for any `H`, `W` divisible by
  `2**levels`.
- With defaults it has between 0.5 M and 5 M trainable parameters (DESIGN: about 2 M).
- An input whose channel count differs from `in_channels` raises `ValueError`.
- Gradients reach every trainable parameter; in `eval()` mode repeated passes are identical.

### FIRE-05 BCE + Dice loss with uncertain pixels ignored (Must)
`model.fire_loss(logits, target, pos_weight=3.0, dice_weight=1.0, ignore_index=-1, smooth=1.0) -> torch.Tensor`:
with `V` the pixels where `target != ignore_index` and `p = sigmoid(logits)`,
`loss = BCE + dice_weight * Dice`, where
`BCE` = mean over `V` of `BCEWithLogits(logits, target, pos_weight=pos_weight)` and
`Dice = 1 - (2 * sum_V(p * t) + smooth) / (sum_V(p) + sum_V(t) + smooth)`.
- Ignored pixels contribute nothing (loss unchanged when their logits change; gradient exactly zero there).
- If `V` is empty it returns a finite `0.0` that supports `.backward()`.
- Shapes must match (`(B, 1, H, W)` or `(B, H, W)`), else `ValueError`.

### FIRE-06 Persistence and downwind-dilated persistence baselines (Must)
In `baselines.py`, for `x (12, H, W)` (returns `(H, W)`) or `(N, 12, H, W)` (returns `(N, H, W)`), `float32`:
- `persistence(x)`: the `PrevFireMask` channel with values `< 0` set to `0` (p = 1 where burning, else 0).
- `downwind_persistence(x)`: the persistence mask OR the same mask shifted by one pixel downwind. Per sample the
  wind direction is the circular mean of `th` over burning pixels (over all pixels if none burn); the downwind
  bearing `th + 180` is rounded to the nearest multiple of 45 degrees and mapped to one of the 8 neighbour steps
  `(d_row, d_col) = (-round(cos b), round(sin b))`. Pixels shifted off the grid are dropped (no wrap-around).
  This is also the adapter's rule baseline.

### FIRE-07 Per-pixel logistic-regression baseline (Must)
`baselines.PixelLogistic(C=1.0, seed=0)`:
- `fit(x, y) -> self` with `x (N, 12, H, W)` (normalised) and `y (N, H, W)`: each pixel is a sample with its 12
  channel values as features; pixels with `y == -1` are excluded; fewer than two classes raises `ValueError`.
- `predict_proba(x) -> np.ndarray` returns `float32 (N, H, W)` in `[0, 1]`; before `fit` raises `RuntimeError`.
- Deterministic for a fixed `seed`; on linearly separable synthetic data pixel AUC > 0.95.

### FIRE-08 PR-AUC on all pixels and on new-burn pixels (Must)
In `metrics.py`:
- `pr_auc(prob, label) -> float`: average precision (`sklearn.metrics.average_precision_score`) over pixels with
  `label != -1`, positives `label == 1`; `nan` if there is no positive pixel. Shapes must match, else `ValueError`.
- `fire_pr_auc(prob, label, prev_mask) -> dict` with keys `"all"` (= `pr_auc(prob, label)`) and `"new_burn"`:
  `pr_auc` restricted to pixels with `prev_mask == 0` (not burning before; `prev_mask == -1` excluded) and
  `label != -1`. This exposes previous-mask copying: persistence scores a constant `0` on those pixels, so its
  new-burn PR-AUC equals the new-burn positive rate.
- `per_fire_macro(values) -> float`: the mean of the finite values in a sequence (per-fire macro average on WSTS);
  `nan` if none is finite.

### FIRE-09 Hourly linear growth from the day-0 mask to the day-1 prediction
`growth.hourly_growth(day0_mask, day1_prob, day0_time, issued, n_slots=12, slot_minutes=15) -> np.ndarray`
returns `float32 (n_slots, H, W)`: for slot `s` with start `t_s = issued + s * slot_minutes`,
`a_s = clip(floor((t_s - day0_time) / 1 h) / 24, 0, 1)` and `P_s = (1 - a_s) * m0 + a_s * day1_prob`,
with `m0` = `day0_mask` with values `< 0` or NaN set to `0`. Growth is stepped hourly (piecewise constant within
the hour) and held at `day1_prob` after 24 h (hold rule). It is plain linear interpolation: if the prediction is
below the day-0 mask somewhere (a shrinking fire), P decreases there. `issued < day0_time`, naive datetimes, or
mismatched shapes raise `ValueError`.

### FIRE-10 Adapter: hazard, policy and RiskLayer construction
`adapter.WildfireAdapter(provider=None, model=None, calibrator=None, policy=None, source_tag="wildfire.unet@dev",
*, checkpoint=None)` implements the `HazardAdapter` protocol from `outrun.core.risk`. Every argument is optional and
construction loads nothing heavy, so `WildfireAdapter()` works offline (zero-argument factory, RISK-13).
- `provider: Callable[[Scenario], WildfireInputs] | None` (None: the real WSTS/NDWS provider, imported lazily).
  `WildfireInputs` (frozen dataclass): `x_model` `(12, h, w)` float32 normalised model input (FIRE-12 `normalize`),
  `x_raw` `(12, h, w)` raw NDWS-order channels, `day0_mask` `(h, w)`, `day0_time` (aware UTC), `day1_mask`
  `(h, w)` or `None` (truth), `x (X,)`/`y (Y,)` GRID centres, `regrid` `(k, h, w) -> (k, Y, X)`,
  `held_out: bool = False`. The adapter reads only `scn.t0` and `scn.grid_window()`.
- `model: nn.Module | None`; when None and `checkpoint` is a path, a `FireUNet` is loaded lazily from it; with
  neither, `predict` raises `RuntimeError`. `calibrator` (`.transform(p)`) is identity when None.
- Module-level `default_adapter() -> WildfireAdapter` (zero arguments, loads nothing; the path is kept as `adapter.checkpoint`, a `Path`; `checkpoint =
  get_paths().models / "wildfire" / "best.pt"`) is the factory registered with `REGISTRY.register(Hazard.WILDFIRE,
  default_adapter)` when `hazards.base` is implemented.
- `baseline(scn)`: `hourly_growth(day0_mask, downwind_persistence(x_raw), day0_time, scn.t0)` (FIRE-06, FIRE-09),
  regridded; `kind="baseline"`, source starting `"wildfire.persistence"`.
- `predict(scn)`: `p1 = calibrator.transform(sigmoid(model(x_model[None]))[0, 0])`, then
  `hourly_growth(day0_mask, p1, day0_time, scn.t0)`, regridded; `kind="forecast"`, source = `source_tag` (starts
  with `"wildfire."`).
- `observed(scn)`: `None` when `day1_mask is None`; else `observed_from_mask(day1_mask)` (FIRE-11) regridded;
  `kind="observed"`, source starting `"wildfire.viirs"`.
- Every layer: built with `to_risk_layer(..., issued=scn.t0, held_out=inputs.held_out,
  x=scn.grid_window().x_centers(), y=scn.grid_window().y_centers())`: shape `(12, Y, X)`, window coords,
  `edge_overrides=None`, `valid_until = scn.t0 + 24 h`. Provider `x`/`y` that differ from those centres (1e-6 m) raise `ValueError`.
- `hazard is Hazard.WILDFIRE`.
- `policy` is an `outrun.core.policy.HazardPolicy` with `block_at = {CIVILIAN: 0.5, SENSITIVE: 0.5, EV: 0.7}`,
  `speed_factor(p) == 1` for all p, `exposure_weight = {CIVILIAN: 20, SENSITIVE: 20, EV: 5}`,
  `beyond_horizon == "hold"`, `decay_tau_min is None`.

`adapter.to_risk_layer(prob, *, kind, issued, source, held_out, valid_until=None, x=None, y=None)` returns a
`RiskLayer` with `hazard=Hazard.WILDFIRE`, `edge_overrides=None`, `prob` an `xr.DataArray` with dims
`("slot", "y", "x")`, dtype `float16`, values clipped to `[0, 1]` and NaN as `0`, coords attached when given (the adapter always gives them).
`prob` must be `(12, ny, nx)`, else `ValueError`. `valid_until` defaults to `issued + 24 h` (next-day horizon).
A `kind` outside `{"forecast", "baseline", "observed"}` or a naive `issued` raises `ValueError`.

### FIRE-11 Observed fire layer from the next-day VIIRS mask
`growth.observed_from_mask(day1_mask, n_slots=12) -> np.ndarray` returns `float32 (n_slots, H, W)` with every slot
equal to `1.0` where `day1_mask == 1` and `0.0` elsewhere (`-1` and NaN become `0`). `WildfireAdapter.observed`
wraps it with `kind="observed"` (ground truth = next-day WSTS VIIRS mask).

### FIRE-12 NDWS dataset for training
`dataset.NdwsDataset(x_path, y_path, stats, augment=False, seed=0)` is a `torch.utils.data.Dataset` over
`ndws_{split}.npy` files (`x (N, 12, H, W)`, `y (N, H, W)` with `-1` = uncertain) opened with memory mapping.
`len(ds) == N`; `ds[i]` returns `(x, y)` torch tensors: `x` float32 `(12, H, W)` equal to
`normalize(x_raw[i], stats)` when `augment=False` (with `augment=True`, `random_augment` is applied to the raw
sample before normalising), and `y` float32 `(1, H, W)` with `-1` preserved. Mismatched `N` raises `ValueError`.
`NdwsDataset.from_split(x_path, y_path, sample_ids, split, part, stats, augment=False, seed=0)` (classmethod) is
the training entry point: `sample_ids[k]` names row `k` with its split id (`"ndws:<id>"`, length `N`, else
`ValueError`); the dataset holds only the rows where `split_mask(sample_ids, split, part)` (SPLIT-07) is True, in
row order (`ds[j]` is the `j`-th selected row), and `ds.ids` lists their ids. Test samples never reach a train or
val dataset.
