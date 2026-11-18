# 10: Flash floods (MRMS QPE + FLASH + terrain to road crossings)

Modules: `src/outrun/hazards/flood/{dataset,model,baselines,crossings,metrics,adapter}.py`
Tests: `tests/unit/test_flood*.py`
Design: DESIGN 3.3 (data, labels, sampling, model, baselines, risk layer), 1.2 (interfaces), 4.2 (costs).

Purpose: a 0-6 h flash-flood probability on the native MRMS 0.01 degree grid, from radar QPE, the operational
FLASH flood-guidance products and terrain, mapped to GRID and sharpened at low-water road crossings through
`RiskLayer.edge_overrides`. The model must beat three honest baselines: FLASH QPE/FFG ratio >= 1, QPE ARI >= 10 yr,
and logistic regression on the same features.

## Shared conventions (this spec)

- Patch arrays are `(C, H, W)`; a batch is `(N, C, H, W)`. Training patches are 64 x 64; every function here also
  works on any `H, W` (tests use 16 x 16 and 32 x 32).
- Labels are `int8` with values `1` (flash flood), `0` (no flood) and `-1` (uncertain ring: ignored in loss,
  baseline fitting and metrics).
- Times are timezone-aware UTC `datetime`; a naive `datetime` raises `ValueError` wherever one is accepted.
- Distances and coordinates in this module are planar metres (the caller projects MRMS lon/lat or GRID cells to a
  local metric frame before calling). Distance to a report is the Euclidean distance to its begin-to-end segment.
- Inputs that other areas produce (hourly QPE accumulations `DATA-04`, FLASH fields `DATA-04`, terrain slope/TPI
  `data/dem.py`, Storm Events segments `DATA-07`, low-water crossings `roads/features.py`, calibration `CAL-*`) are
  taken here as plain arrays/dataclasses; this spec never re-implements them.
- Policy numbers come from `configs/hazards.yaml` (`hazards.flood`); tests restate the DESIGN values.
- Must = required for the proposal ("flash floods", "deep learning"). All FLOOD-01..FLOOD-08 are Must.

## Requirements

### FLOOD-01 Feature stack: 11 channels with a valid mask (Must)
`dataset.FLOOD_CHANNELS` is the tuple
`("qpe_1h", "qpe_3h", "qpe_6h", "qpe_24h", "ffg_ratio_1h", "ffg_ratio_3h", "ari_max", "elevation", "slope", "tpi", "valid")`.

`build_features(qpe_1h, qpe_3h, qpe_6h, qpe_24h, ffg_ratio_1h, ffg_ratio_3h, ari_max, elevation, slope, tpi) -> np.ndarray`
takes ten `(H, W)` arrays (QPE accumulations in mm ending at the issue time, FLASH QPE/FFG plain ratios (DATA-04
converts the raw percent), max QPE ARI in
years, elevation m, slope degrees, TPI m) and returns a `float32` array `(11, H, W)` in `FLOOD_CHANNELS` order.
- Channel `valid` is `1.0` where all ten inputs are finite at that pixel, else `0.0`.
- Every non-finite input value is written as `0.0` in the output (no NaN or inf in the output).
- If any of the four QPE accumulations is entirely non-finite (a missing MRMS hour makes the whole window missing),
  it raises `MissingHoursError` (a subclass of `ValueError`): such windows are skipped, never filled.
- Arrays of different shapes, or not 2-D, raise `ValueError`.
- Inputs are not modified.

### FLOOD-02 Uncertain-ring labels from Storm Events segments (Must)
`dataset.FloodReport` is a frozen dataclass `(begin: datetime, end: datetime, x0, y0, x1, y1: float)`: a Storm
Events "Flash Flood" report with its begin and end location in metres (a point report has `x0 == x1, y0 == y1`).

`distance_to_reports(x, y, reports) -> np.ndarray` returns, for cell-centre coordinate arrays `x`, `y` of the same
shape, the minimum Euclidean distance (m) to any report segment; `inf` everywhere if `reports` is empty.

`ring_labels(x, y, reports, issued, horizon_h=6.0, pre_h=1.0, pos_m=5000.0, ring_m=15000.0) -> np.ndarray[int8]`
(same shape as `x`):
- A report is **active** if the closed interval `[begin - pre_h, end]` intersects `[issued, issued + horizon_h]`.
  Inactive reports are ignored completely (they create neither positives nor rings).
- Label `1` where the distance to the nearest active report is `<= pos_m`.
- Label `-1` where it is `> pos_m` and `<= ring_m` (the uncertain 5-15 km ring).
- Label `0` elsewhere, and everywhere if no report is active.
- A naive `issued` raises `ValueError`; `x` and `y` of different shapes raise `ValueError`.

### FLOOD-03 Sample kinds and hard negatives (Must)
`dataset.SampleKind` is a `StrEnum`: `POSITIVE`, `HARD_NEGATIVE`, `WET`, `DRY` (values `"positive"`,
`"hard_negative"`, `"wet"`, `"dry"`).

`near_report(x, y, reports, issued, radius_m=25000.0, window_h=3.0) -> bool` is `True` iff some report has
`[begin - window_h, end + window_h]` containing `issued` and a segment distance `<= radius_m` to at least one cell.

`sample_kind(labels, qpe_1h, report_near, hard_neg_mm=25.0, wet_mm=1.0) -> SampleKind` classifies a patch, in this
order of precedence:
1. `POSITIVE` if any label equals `1`;
2. `HARD_NEGATIVE` if the patch maximum of `qpe_1h` (NaN ignored) is `>= hard_neg_mm` and `report_near` is `False`
   (heavy rain, no report within 25 km and +-3 h);
3. `WET` if the patch maximum of `qpe_1h` is `>= wet_mm`;
4. `DRY` otherwise (also when `qpe_1h` is all NaN).

Dry hours are never used for training (FLOOD-13), so "rain vs no rain" is not a shortcut.

### FLOOD-04 Flood U-Net (PyTorch deep-learning model) (Must)
`model.FloodUNet(in_channels=11, base=32, levels=3)` is a `torch.nn.Module` U-Net (encoder-decoder with skip
connections; `levels` downsampling steps by 2; first stage has `base` channels, doubling per level).
- `forward(x)` maps `(B, in_channels, H, W)` float32 to logits `(B, 1, H, W)` for any `H`, `W` divisible by
  `2**levels` (64 x 64 in training, 32 x 32 in tests).
- With defaults it has between 0.5 M and 5 M trainable parameters (DESIGN: about 2 M).
- An input whose channel count differs from `in_channels` raises `ValueError`.
- Gradients reach every trainable parameter from a scalar loss on the output.
- In `eval()` mode two forward passes on the same input give identical outputs.
- `FloodUNet(in_channels=8)` builds the no-FLASH ablation model (FLOOD-09).

### FLOOD-05 Focal loss with the uncertain ring masked (Must)
`model.focal_loss(logits, target, gamma=2.0, alpha=0.25, ignore_index=-1) -> torch.Tensor` (scalar):
- `p = sigmoid(logits)`, `p_t = p` where `target == 1` else `1 - p`; `alpha_t = alpha` where `target == 1` else
  `1 - alpha` (`alpha=None` means `alpha_t = 1`). Per-pixel loss `-alpha_t * (1 - p_t)**gamma * log(p_t)`, computed
  in a numerically stable way from the logits.
- Result = mean over pixels with `target != ignore_index`.
- Pixels with `target == ignore_index` contribute nothing: changing their logits leaves the loss unchanged and their
  gradient is exactly zero.
- With `gamma=0, alpha=None` it equals `BCEWithLogits` averaged over the non-ignored pixels.
- If every pixel is ignored it returns a finite `0.0` (no NaN) that still supports `.backward()`.
- `logits` and `target` shapes must match (`(B, 1, H, W)` or `(B, H, W)`), else `ValueError`.

### FLOOD-06 Rule baselines: FLASH FFG ratio, QPE ARI, ARI-style fallback (Must)
In `baselines.py`, each returns a `float32` probability array of the input shape with values in `{0.0, 1.0}`;
NaN inputs give `0.0`:
- `ffg_ratio_baseline(ffg_ratio_1h, ffg_ratio_3h, threshold=1.0)`: `1` where `max(ratio_1h, ratio_3h) >= threshold`
  (operational FLASH QPE/FFG ratio >= 1; also the adapter's rule baseline, DESIGN "FFG ratio >= 1 gives p = 1").
  The ratios are **plain ratios** (1.0 = QPE equals flash-flood guidance): MRMS codes them in percent and DATA-04
  `read_mrms` divides by 100, so threshold 1.0 is 100 % in the raw file.
- `ari_baseline(ari_max_years, threshold_years=10.0)`: `1` where max QPE ARI `>= threshold_years`.
- `ari_rule_fallback(qpe_3h, qpe_6h, mm_3h=50.0, mm_6h=75.0)`: `1` where `qpe_3h >= 50` or `qpe_6h >= 75`
  (used only if FLASH is absent from the bucket; documented in NOTES).
- `rule_baseline_from_features(features)` applies `ffg_ratio_baseline` to the `ffg_ratio_1h`, `ffg_ratio_3h`
  channels of an `(11, H, W)` or `(N, 11, H, W)` FLOOD-01 stack, returning `(H, W)` or `(N, H, W)`.

### FLOOD-07 Logistic-regression baseline on the same features (Must)
`baselines.LogisticBaseline(C=1.0, seed=0)`:
- `fit(X, y) -> self` with `X (N, 11, H, W)` FLOOD-01 stacks and `y (N, H, W)` labels. Each pixel is one sample
  whose features are its channels. Pixels with `y == -1` or `valid == 0` are excluded. Fewer than two classes among
  the kept pixels raises `ValueError`.
- `predict_proba(X) -> np.ndarray` returns `float32 (N, H, W)` values in `[0, 1]`; pixels with `valid == 0` get `0`.
- Deterministic for a fixed `seed`; on linearly separable synthetic data it ranks positives above negatives
  (pixel AUC > 0.95).
- `predict_proba` before `fit` raises `RuntimeError`.

### FLOOD-08 Low-water crossing edge overrides (Must)
In `crossings.py`: `CrossingKind` is a `StrEnum` `LOW_WATER`, `DEPRESSION`, `BRIDGE` (values `"low_water"`,
`"depression"`, `"bridge"`). A `DEPRESSION` crossing is a low-water crossing in a local depression.

`in_depression(dem, cell_m, row, col, radius_m=300.0, drop_m=2.0) -> bool` (3DEP 10 m DEM): `True` iff
`dem[row, col] < median(dem cells whose centres are within radius_m of (row, col), centre included) - drop_m`
(strictly more than 2 m below). The neighbourhood is clipped at the DEM border; NaN cells are ignored; a
`(row, col)` outside the DEM raises `ValueError`.

`crossing_overrides(edge_prob, kinds, crossing_mult=1.5, depression_mult=2.0) -> dict[int, np.ndarray]`:
`edge_prob` maps edge id to its sampled per-slot probability `(n_slots,)`; `kinds` maps edge id to a `CrossingKind`.
For every edge in `kinds` (and only those) the output holds a new `float32 (n_slots,)` array:
`LOW_WATER`: `min(1, 1.5 p)`; `DEPRESSION`: `min(1, 2 p)`; `BRIDGE`: `0`. An edge in `kinds` but missing from
`edge_prob` raises `KeyError`. Inputs are not modified. The defaults equal `crossing_multiplier` and
`depression_multiplier` in `configs/hazards.yaml`.

### FLOOD-09 No-FLASH ablation channels
`dataset.FLASH_CHANNELS = ("ffg_ratio_1h", "ffg_ratio_3h", "ari_max")` and
`dataset.select_channels(x, use_flash=True) -> np.ndarray`: for an `(11, H, W)` or `(N, 11, H, W)` stack returns it
unchanged in content when `use_flash` is `True`; when `False` returns the 8 channels of `FLOOD_CHANNELS` without
`FLASH_CHANNELS`, order preserved (`(8, H, W)` / `(N, 8, H, W)`). `dataset.channel_names(use_flash)` returns the
matching names. A channel axis other than 11 raises `ValueError`. (FFG is both an input and a baseline, so the
report ablates it.)

### FLOOD-10 Adapter: hazard, policy and RiskLayer construction
`adapter.FloodAdapter(provider=None, model=None, calibrator=None, policy=None, source_tag="flood.unet@dev", *,
graph=None, checkpoint=None, use_flash=True)` implements the `HazardAdapter` protocol from `outrun.core.risk`
(`hazard`, `policy`, `baseline`, `predict`, `observed`). Every argument is optional and construction loads nothing
heavy, so `FloodAdapter()` works offline (the registry needs a zero-argument factory, RISK-13).
- `provider: Callable[[Scenario], FloodInputs] | None` supplies the data (None: the real MRMS/FLASH/Storm Events
  provider, imported lazily on first use). `FloodInputs` (frozen dataclass): `features` `(11, h, w)` float32
  FLOOD-01 stack at issue time on the model grid, `reports: Sequence[FloodReport] | None` (truth; None = no truth),
  `x (X,)` and `y (Y,)` GRID cell centres of `scn.grid_window()`, `regrid: Callable[[np.ndarray], np.ndarray]`
  mapping `(k, h, w)` to `(k, Y, X)`, `held_out: bool = False`. The adapter reads only `scn.t0` and
  `scn.grid_window()` from the scenario.
- `model: nn.Module | None`; when None and `checkpoint` is a path, a `FloodUNet` is loaded lazily from it on the
  first `predict`; with neither, `predict` raises `RuntimeError`. `calibrator` (anything with `.transform(p)`,
  e.g. `outrun.core.calibrate.IsotonicCalibrator`) is identity when None.
- `graph: RoadGraph | Callable[[Scenario], RoadGraph] | None` is the road graph whose CSR edge ids key
  `edge_overrides`. Without a graph, every layer has `edge_overrides=None`. **It must be the same graph the router plans on**
  (same CSR edge ids): the CLI/backend pass the graph they route on (`outrun route --place` included); the
  default's `build_graph(bbox=scn.bbox, ...)` is only right when the router also uses the scenario-bbox graph.
  ASTAR-18 rejects override ids `>= n_edges`, but a same-size different graph cannot be detected, so callers own
  this.
- Module-level `default_adapter() -> FloodAdapter` (zero arguments, loads nothing; the path is kept as `adapter.checkpoint`, a `Path`) is the registered factory:
  `checkpoint = get_paths().models / "flood" / "best.pt"`, `graph = lambda scn: build_graph(bbox=scn.bbox,
  cache_dir=get_paths().proc / "graphs")`. The module registers it with `REGISTRY.register(Hazard.FLOOD,
  default_adapter)` when `hazards.base` is implemented.
- `baseline(scn)`: `p = rule_baseline_from_features(features)` (FLOOD-06), held over the 12 slots, regridded;
  `kind="baseline"`, source starting `"flood.ffg"`.
- `predict(scn)`: model input = `features` (or `select_channels(features, use_flash=False)` when `use_flash` is
  False), `p = calibrator.transform(sigmoid(model(x[None]))[0, 0])`, held over the 12 slots, regridded;
  `kind="forecast"`, source = `source_tag` (starts with `"flood."`).
- `observed(scn)`: `None` when `reports is None`; else `observed_from_reports(reports, x, y, scn.t0)` (FLOOD-11,
  already on GRID); `kind="observed"`, source starting `"flood.stormevents"`, `edge_overrides=None`.
- Crossing overrides (baseline and predict, only with a graph): `kinds` = `LOW_WATER` for every edge with
  `graph.crossing`, `BRIDGE` for every edge with `graph.bridge` (a bridge flag wins); `edge_prob[e] =
  EdgeRiskSampler.from_window(graph, scn.grid_window()).sample(p_slots[None])[e, 0]` (ASTAR-14) on the layer's own
  `(12, Y, X)` probabilities; `edge_overrides = crossing_overrides(edge_prob, kinds)` (FLOOD-08), keys in
  `range(graph.n_edges)`.
- Every layer: built with `to_risk_layer(..., issued=scn.t0, held_out=inputs.held_out, x=scn.grid_window().x_centers(),
  y=scn.grid_window().y_centers())`, so `prob` has shape `(12, Y, X)` and the window coords (RISK-02, RISK-13);
  `valid_until = scn.t0 + 6 h`. Provider `x`/`y` that differ from those centres (1e-6 m) raise `ValueError`.
- `hazard is Hazard.FLOOD`.
- `policy` is an `outrun.core.policy.HazardPolicy` with `block_at = {CIVILIAN: 0.4, SENSITIVE: 0.4, EV: 0.7}`
  (applied at crossings), `speed_factor(p) = 1 - 0.4 p`, `exposure_weight = {CIVILIAN: 10, SENSITIVE: 10, EV: 3}`,
  `beyond_horizon == "hold"`, `decay_tau_min is None`.

`adapter.to_risk_layer(prob, *, kind, issued, source, held_out, valid_until=None, edge_overrides=None, x=None, y=None)`
returns a `RiskLayer` with `hazard=Hazard.FLOOD`:
- `prob` is `(12, ny, nx)` (12 routing slots); any other number of dimensions or slots raises `ValueError`.
- `layer.prob` is an `xr.DataArray` with dims `("slot", "y", "x")`, dtype `float16`, values clipped to `[0, 1]`,
  NaN written as `0`; coordinates `x`, `y` attached when given (the adapter always gives them; a coordless layer is
  rejected by `RiskStack.from_layers`, RISK-02).
- `valid_until` defaults to `issued + 6 h` (native horizon; the layer holds beyond it, HORIZON-03).
- `kind` not in `{"forecast", "baseline", "observed"}`, or a naive `issued`, raises `ValueError`.
- `edge_overrides`, `source`, `held_out` are stored unchanged.

### FLOOD-11 Observed flood layer from Storm Events segments
`dataset.observed_from_reports(reports, x, y, issued, n_slots=12, slot_minutes=15, radius_m=5000.0) -> np.ndarray`
with GRID cell-centre axes `x (nx,)`, `y (ny,)` returns `float32 (n_slots, ny, nx)`: slot `s` covers
`[issued + s*15 min, issued + (s+1)*15 min)`; a report is active in slot `s` if `begin < slot_end` and
`end >= slot_start`; value `1.0` where the distance to an active report's segment is `<= radius_m`, else `0.0`.
No reports gives all zeros. `FloodAdapter.observed` wraps this with `kind="observed"`.

### FLOOD-12 Event POD
`metrics.event_pod(prob, report_masks, threshold=0.5) -> float`: the fraction of reports (each a boolean footprint
mask of `prob`'s shape) whose footprint contains at least one pixel with `prob >= threshold`. Empty `report_masks`
returns `nan`; a mask of a different shape raises `ValueError`; an all-False mask counts as not covered.

### FLOOD-13 Balanced patch sampler (1/3 positive, 1/3 hard negative, 1/3 wet)
`dataset.balanced_indices(kinds, n, rng) -> np.ndarray[int64]`: quotas `n // 3` for each of `POSITIVE`,
`HARD_NEGATIVE`, `WET`, with the remainder `n % 3` adding one each to `POSITIVE` then `HARD_NEGATIVE`. For each kind
it draws `min(quota, available)` distinct indices without replacement using `rng` (a `np.random.Generator`).
`DRY` indices are never returned. The result is sorted, has no duplicates, and is identical for the same seed.
`n < 0` raises `ValueError`.

### FLOOD-14 Patch dataset for training
`dataset.FloodPatchDataset(x_path, y_path, use_flash=True)` is a `torch.utils.data.Dataset` over `.npy` files
(`x (N, 11, H, W)` float16/float32, `y (N, H, W)` int8) opened with memory mapping.
`len(ds) == N`; `ds[i]` returns `(x, y)` torch tensors: `x` float32 `(11, H, W)` (or `(8, H, W)` with
`use_flash=False`, via FLOOD-09), `y` float32 `(1, H, W)` with `-1` preserved. Mismatched `N` raises `ValueError`.
`FloodPatchDataset.from_split(x_path, y_path, sample_ids, split, part, use_flash=True)` (classmethod) is the
training entry point: `sample_ids[k]` names patch row `k` (length `N`, else `ValueError`); the dataset holds only
the rows where `split_mask(sample_ids, split, part)` (SPLIT-07) is True, in row order (`ds[j]` is the `j`-th
selected row), and `ds.ids` lists their ids. Quarantined and test samples never reach a train or val dataset.
