# 09: Tornadoes (TorNet radar detection + projected swath)

Modules: `src/outrun/hazards/tornado/` (`features.py`, `dataset.py`, `model.py`, `calibration.py`,
`baselines.py`, `metrics.py`, `swath.py`, `adapter.py`). Unit tests: `tests/unit/test_tornado*.py`.
Design: DESIGN 3.2, 1.2, 6.1. The prefix is `TORN` (DESIGN wrote `TOR-*`).

Purpose: a CNN that scores TorNet NEXRAD Level-II chips as tornadic or not, calibrated on val, and a projection
of each detection into a 30-minute swath on GRID for routing. Baselines: gradient boosting on hand features and
a delta-V (gate-to-gate shear) rule. The headline metric is ROC-AUC on tornadic vs warned-null chips only.

Conventions used below:
- **Chip**: a mapping `variable -> float array (T, 2, A, R)` (time frame, tilt, azimuth, range gate) for the six
  variables `VARIABLES`; missing gates are `NaN`. Tilt 0 is the lowest tilt. Chip size is [CANDIDATE] 120 x 240;
  nothing below depends on it.
- The official TorNet split is the `type` column of `catalog.csv` (verified in T1 from `mit-ll/tornet`; chosen by
  Milton on Nov 18), read per row by TORN-02. The category labels are [CANDIDATE] parameters with defaults `"TOR"`
  (tornadic) and `"WRN"` (warned null).
- Map motion vectors are `(u east, v north)` in m/s. Grid coordinates are EPSG:5070 metres (GRID).
- Only `adapter.py` imports `outrun.core`. The package `__init__.py` imports nothing.

"Must" marks requirements the proposal depends on (all of `TORN-01..TORN-08` are in the proposal coverage
matrix). `TORN-10` (the adapter that ships the layer) is also Must. `TORN-09` is "Should"; it is still traced and
tested.

## Requirements

### TORN-01 Chip tensor: last frame, temporal difference, missing mask, CoordConv channels (Must)
`features.py` constants: `VARIABLES = ("DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH")`, `N_TILTS = 2`,
`FILL_VALUES = {"DBZ": -20.0, "VEL": 0.0, "KDP": 0.0, "RHOHV": 0.0, "ZDR": 0.0, "WIDTH": 0.0}`,
`N_CHANNELS = 27`, `N_CHANNELS_NO_RANGE = 26`, `N_NORMALIZED = 24`.

`build_chip_tensor(chip, range_km, azimuth_deg, include_range=True) -> np.ndarray` returns float32
`(N_CHANNELS, A, R)` (or `(N_CHANNELS_NO_RANGE, A, R)` without range). With `v` the index in `VARIABLES` and
`t` the tilt, after replacing `NaN` by `FILL_VALUES[var]`:
- channel `2v + t` (0..11): last frame `chip[var][-1, t]`;
- channel `12 + 2v + t` (12..23): last minus first frame `chip[var][-1, t] - chip[var][0, t]` (zeros when
  `T == 1`);
- channel 24: 1.0 where any variable/tilt of the last frame was `NaN` at that gate, else 0.0;
- channel 25 (only if `include_range`): `range_km / 300` broadcast along azimuth;
- last channel: azimuth scaled linearly to [-1, 1] (`min -> -1`, `max -> +1`; all 0 when `A == 1`), broadcast
  along range.
A missing variable raises `KeyError`; variables with different shapes, or `range_km`/`azimuth_deg` lengths not
matching `R`/`A`, raise `ValueError`. The input chip is not modified.

`fit_channel_stats(X) -> (mean, std)`: per-channel mean and std over `(N, A, R)` of `X` `(N, C, A, R)` (train
only), std 0 replaced by 1. `normalize(X, mean, std, n_norm=N_NORMALIZED)` returns a float32 copy in which the
first `n_norm` channels are `(X - mean) / std` and the rest (mask and coordinates) are unchanged.

### TORN-02 Split labels come from the TorNet catalog `type` column (Must)
`dataset.py`: `split_catalog(catalog, val_frac=0.1, seed=0, quarantine=(), column="type") -> pd.Series` labels each
row of `catalog` (a DataFrame with a `date` column of `datetime.date` or timestamps, and the official split column
`column`, by default TorNet's `type`) as `"train"`, `"val"`, `"test"` or `"excluded"`, indexed like `catalog`. It does
not run its own algorithm: it is a thin adapter over `outrun.eval.splits.tornet_split` (SPLIT-02), so the tornado
trainer and the frozen `splits/tornet.json` agree.
- Official split per row: `catalog[column].str.strip().str.lower()` (the DATA-02 `official_split` `mode="column"`
  normalisation); a value not in `{"train", "test"}` raises `ValueError`. It is never derived from the date: TorNet
  assigns chips by storm episode, and in 2015/16/19 35 UTC calendar days hold both train and test chips (4,525 rows,
  NOTES T1), so no date rule can reproduce it.
- It builds one sample per row, `(str(index), datetime(date, 00:00 UTC), official)`, calls
  `tornet_split(samples, quarantine=quarantine, val_frac=val_frac, seed=seed)` and labels each row with the set
  (`train`/`val`/`test`) its id lands in. Rows in none of the three (quarantined, `SPLIT-07`, or dropped by the
  SPLIT-02 1-day gap around test and val days) are `"excluded"`.
- Consequences (tested): every official-test row is `"test"` (quarantine never changes test rows) and no
  official-train row is `"test"`; on a mixed day the official-train rows are `"excluded"` (their day is a test day,
  so the SPLIT-02 gap drops them), as are train rows on the days just before and after; val is a set of calendar
  days; the labels, read back as `SplitItem(str(index), date)`, pass `validate_split(..., gap_days=1)` with no
  violations, and equal the sets of `tornet_split` on the same samples.
- Same inputs and seed give the same labels. A catalog without `date` or without `column` raises `KeyError`.
- **Deviation from DESIGN 3.2** ("Val = 10% of train Julian days (grouped)"): val is drawn from whole calendar
  days (one date = one group), not from day-of-year values pooled across years. This is a decision recorded
  here for the owner to confirm; DESIGN 3.2 was updated to match on Nov 18.

### TORN-03 Hand-crafted radar features (Must)
`features.py`: `HAND_FEATURES = ("max_dbz", "max_dv", "min_rhohv_dbz30", "max_zdr", "max_kdp", "max_width",
"range_km")`. `hand_features(chip, range_km_center) -> np.ndarray` float64 of shape `(7,)` in that order, all
from the **last frame, tilt 0**, ignoring `NaN`s:
- `max_dbz`, `max_zdr`, `max_kdp`, `max_width`: maxima (all-`NaN` -> the `FILL_VALUES` entry);
- `max_dv`: max over gates of `|VEL[a+1, r] - VEL[a, r]|` (azimuthally adjacent gates at the same range; pairs
  with a `NaN` skipped; 0.0 if no valid pair) - the gate-to-gate shear proxy;
- `min_rhohv_dbz30`: min of `RHOHV` over gates with `DBZ > 30` and valid `RHOHV` (debris proxy); 1.0 if none;
- `range_km`: `range_km_center`.
`max_dv(chip, tilt=0) -> float` exposes the shear proxy alone.

### TORN-04 Tornado CNN (PyTorch deep-learning model) (Must)
`model.py`: `TornadoCNN(in_channels=N_CHANNELS, widths=(32, 64, 128, 256), hidden=128)`, a port of the TorNet
baseline: one VGG block per width (two 3x3 conv + ReLU, then 2x2 max-pool), CoordConv inputs, global max-pool,
MLP head.
- `forward(x)` takes `(B, in_channels, A, R)` float32 and returns logits of shape `(B,)`.
- Size-agnostic: any `A, R >= 2**len(widths)` works (global pooling). `x.shape[1] != in_channels` raises
  `ValueError`.
- The default configuration has between 0.5 and 3 million trainable parameters (DESIGN: about 1.5M).
- `predict_proba(model, x) -> torch.Tensor` (eval mode, no grad) returns `sigmoid(logits)`, shape `(B,)`,
  values in [0, 1]. In eval mode a sample's output does not depend on the rest of the batch.
- Determinism: same `torch.manual_seed` -> identical weights; repeated eval-mode calls give identical outputs.

### TORN-05 Focal loss and isotonic calibration (Must)
- `model.focal_loss(logits, targets, gamma=2.0, alpha=None) -> torch.Tensor` (scalar):
  `mean(-a_t * (1 - p_t)**gamma * log(p_t))` with `p = sigmoid(logits)`, `p_t = p` if target 1 else `1 - p`,
  `a_t = alpha` if target 1 else `1 - alpha` (`a_t = 1` when `alpha is None`). Computed stably from logits.
  With `gamma = 0, alpha = None` it equals `BCEWithLogits` (mean). `logits = 0, target = 1, gamma = 2` gives
  `0.25 * ln 2`.
- `calibration.IsotonicCalibrator().fit(p, y) -> self` and `transform(p) -> np.ndarray` (float64, [0, 1],
  non-decreasing in `p`, clipped outside the fitted range). On the fitting data the mean calibrated probability
  equals the base rate `mean(y)` (within 1e-6), which corrects the catalog's inflated base rate. `transform`
  before `fit` raises `RuntimeError`.

### TORN-06 Baselines: gradient boosting on hand features, delta-V rule, range-only shortcut check (Must)
`baselines.py`:
- `delta_v_rule(chip, threshold=40.0) -> float`: `1.0` if `max_dv(chip, tilt=0) >= threshold` m/s, else `0.0`.
- `train_gbm(features, labels, seed=0)` fits and returns a `sklearn` `HistGradientBoostingClassifier`
  (`random_state=seed`) on `(N, 7)` hand features (`NaN` allowed). `gbm_predict_proba(model, features)` returns
  `(N,)` float probabilities in [0, 1]. Same data and seed give identical probabilities.
- `train_range_only(range_km, labels)` fits a `LogisticRegression` on the single range feature;
  `range_only_predict_proba(model, range_km)` returns `(N,)` probabilities (the range-shortcut check: it must be
  weak on real data; here it only has to run and return valid probabilities).

### TORN-07 Detection scores incl. warned-null AUC and range bins (Must)
`metrics.py` (may delegate to `outrun.eval.metrics`; the definitions here are the contract):
`tornado_scores(y_true, p, category, range_km, threshold=0.5, positive="TOR", warned_null="WRN") -> dict` with
keys:
- `"auc"`, `"auprc"` (average precision): all rows;
- `"auc_warned"`: ROC-AUC on rows with `y_true == 1` or `category == warned_null` only (the headline);
- `"auc_by_range"`: `{"0-50": ..., "50-100": ..., "100+": ...}` over `[0, 50)`, `[50, 100)`, `[100, inf)` km;
- `"brier"`: `mean((p - y_true)**2)`;
- `"pod"`, `"far"`, `"csi"` with `p >= threshold` as a detection: `pod = hits / (hits + misses)`,
  `far = false_alarms / (hits + false_alarms)`, `csi = hits / (hits + misses + false_alarms)`.
Any score whose subset has a single class (AUC) or a zero denominator is `nan`. Length mismatch raises
`ValueError`. `positive` documents the label of `y_true == 1` rows and is not used to compute labels.

### TORN-08 Swath projection of a detection (Must)
`swath.py`: `project_swath(p, x0, y0, motion, grid_x, grid_y, n_slots=12, slot_minutes=15, horizon_min=30,
sigma_along_km=2.0, sigma_across0_km=4.0, sigma_across_end_km=8.0, step_min=1.0) -> np.ndarray` returns float32
`(n_slots, Y, X)` (`Y = len(grid_y)`, `X = len(grid_x)`; index `[k, i, j]` is cell `(grid_x[j], grid_y[i])`).
- For each `tau` in `0, step_min, 2 step_min, ...` with `tau < horizon_min`, the detection is at
  `(x0 + u*60*tau, y0 + v*60*tau)` with `motion = (u, v)` m/s. The point kernel is
  `p * exp(-0.5 * (a**2 / sigma_along**2 + c**2 / sigma_across(tau)**2))`, `a`/`c` the along/across-track
  distances in km and `sigma_across(tau) = sigma_across0 + (sigma_across_end - sigma_across0) * tau / horizon_min`
  (4 km widening to 8 km at 30 min). With zero motion the along-track axis is east.
- Slot `k` is the max of the kernels with `tau` in `[k*slot_minutes, (k+1)*slot_minutes)`; slots with no such
  `tau` are 0 (beyond the 30 min horizon, so slots 2..11 are 0 by default).
- Every value is in `[0, p]`; the cell at the detection gets about `p` in slot 0; the field is symmetric across
  the track; it is wider later (larger across-track values in slot 1 than slot 0 at the same across-track offset
  from the track); it is below 1e-6 farther than 50 km from the whole track.
- `p` outside [0, 1] raises `ValueError`.
- `combine_swaths(swaths) -> np.ndarray`: element-wise max of a non-empty sequence of equal-shape arrays.

### TORN-09 Storm-motion fallback order (Should)
`swath.py`: `resolve_motion(storm_motion, hrrr_wind) -> tuple[float, float, str]` returns `(u, v, source)`:
1. `storm_motion` (pySTEPS motion from SEVIR VIL, `STORM-09`) if not `None` and finite -> `source = "storm"`;
2. else `hrrr_wind` (HRRR 700/500 hPa mean wind) if not `None` and finite, taken at 75% speed and rotated 30
   degrees clockwise (right-mover): `u' = 0.75 (u cos30 + v sin30)`, `v' = 0.75 (-u sin30 + v cos30)` ->
   `source = "hrrr"`;
3. else 15 m/s **from** 240 degrees (blowing toward 60 degrees): `u = 15 sin 60 deg`, `v = 15 cos 60 deg` ->
   `source = "default"`.
The SPC track (scoring truth) is never an input.

### TORN-10 `TornadoAdapter`: RiskLayer, policy and drop beyond the horizon (Must)
`adapter.py`:
- `TornadoDetection` (dataclass): `chip` (mapping as above), `range_km` `(R,)`, `azimuth_deg` `(A,)`,
  `center_range_km: float`, `x_m: float`, `y_m: float` (chip centre on GRID).
- `TornadoInputs` (dataclass): `detections: Sequence[TornadoDetection]`, `x` `(X,)`, `y` `(Y,)` GRID
  coordinates, `storm_motion: Callable[[float, float], tuple[float, float] | None] | None`,
  `hrrr_wind: Callable[[float, float], tuple[float, float] | None] | None`, `observed: np.ndarray | None`
  (`(12, Y, X)` SPC tracks buffered by width, rasterized by `DATA-07`), `held_out: bool`.
- `TornadoAdapter(provider=None, model=None, calibrator=None, norm_stats=None, policy=None,
  source_tag="tornado.cnn@dev", *, checkpoint=None)` implements `outrun.core.risk.HazardAdapter`; `provider:
  Callable[[Scenario], TornadoInputs]` (None: the real NEXRAD/TorNet provider, imported lazily on first use); the
  adapter reads only `scn.t0` from the scenario. Every argument is optional and construction loads nothing, so
  `TornadoAdapter()` works offline (zero-argument factory, RISK-13). With `model=None` and a `checkpoint` path, a
  `TornadoCNN` is loaded lazily; with neither, `predict` raises `RuntimeError`.
- Module-level `default_adapter() -> TornadoAdapter` (zero arguments, loads nothing; the path is kept as `adapter.checkpoint`, a `Path`; `checkpoint =
  get_paths().models / "tornado" / "best.pt"`) is registered with `REGISTRY.register(Hazard.TORNADO,
  default_adapter)` when `hazards.base` is implemented.
- `hazard is Hazard.TORNADO`. `policy` (default: `outrun.core.policy.load_policies()[Hazard.TORNADO]`): `block_at[p] == 0.5` for
  every `Profile`, exposure weights from the config (civilian 20), `beyond_horizon == "drop"`.
- `baseline(scn)`: each detection's `delta_v_rule` (0/1) projected with `project_swath` along
  `resolve_motion(...)` at the chip centre, combined by max; `kind == "baseline"`.
- `predict(scn)`: `build_chip_tensor` -> `normalize` (if `norm_stats`) -> `predict_proba` -> calibrator (identity
  if `None`) -> swath, combined by max; `kind == "forecast"`. No model raises `RuntimeError`.
- `observed(scn)`: `None` if `inputs.observed is None`, else that array **exactly as supplied** (all 12 slots are
  truth, so no beyond-horizon rule is applied by the adapter); `kind == "observed"`.
- No detections give an all-zero layer. Every returned `RiskLayer`: `hazard == Hazard.TORNADO`,
  `issued == scn.t0`, `valid_until == scn.t0 + 30 min`, `source` starts with `"tornado."`,
  `held_out == inputs.held_out`, `prob` dims `("slot", "y", "x")`, shape `(12, Y, X)`, float16, values in
  [0, 1]. Slots after `valid_until` follow the drop rule (`HORIZON-02`): the adapter passes its 2 valid slots to
  `outrun.core.policy.apply_beyond_horizon(...)` (blocked cells may linger 15 min), so slots starting at or after `t0 + 45 min` (index >= 3) are all 0 in forecast and baseline
  layers. `RiskStack.from_layers` re-applies the same rule (RISK-02); applying it twice changes nothing.
