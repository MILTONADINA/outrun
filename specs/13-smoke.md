# 13: Smoke (PM2.5 +6 h from AQS + HRRR-Smoke + HMS)

Modules: `src/outrun/hazards/smoke/dataset.py`, `src/outrun/hazards/smoke/model.py`,
`src/outrun/hazards/smoke/baselines.py`, `src/outrun/hazards/smoke/adapter.py`.
Tests: `tests/unit/test_smoke*.py`. Design: DESIGN 3.6 (source of truth), 1.2 (interfaces), 4.2 (costs).

Purpose: forecast PM2.5 six hours ahead at EPA AQS monitors from HRRR-Smoke fields, recent station PM2.5 and NOAA HMS
satellite smoke density with a small CNN + MLP quantile model; compare with persistence, calibrated raw HRRR-Smoke and
linear regression on a time split plus a spatial monitor holdout; apply it on a 3 km grid and expose it to routing as
an exposure-only `SMOKE` `RiskLayer`.

Shared conventions for this spec:
- PM2.5 is in ug/m3. All datetimes are timezone-aware UTC; a naive datetime raises `ValueError` wherever a function
  accepts one. Issue times are 00, 06, 12 or 18 UTC. Coordinates are EPSG:5070 metres.
- Core types (`Hazard`, `RiskLayer`, `HazardPolicy`, `Profile`, `Scenario`) are imported from `outrun.core.*`.
- Policy values come from `configs/hazards.yaml` (`hazards.smoke`), never hard-coded duplicates.

All requirements below are **Must** (the proposal phrase "smoke" maps to SMOKE-01..SMOKE-08; SMOKE-02 is the "real
satellite data" row; SMOKE-04 is the "deep learning" row). SMOKE-09 is Must for the report (T24 metrics).

## Requirements

### SMOKE-01 Station samples and targets (`dataset.build_station_samples`)
`build_station_samples(aqs: pd.DataFrame, issue_times: Sequence[datetime], horizon_h: int = 6, history_h: int = 6)
-> pd.DataFrame`. `aqs` columns: `site_id` (str), `time` (tz-aware UTC, hourly), `pm25` (float, may be NaN).
One output row per (site, issue time `t`) with columns `site_id`, `issue_time`, `pm_hist` (float32 array of length
`history_h`: hours `t-5 h .. t`, oldest first, NaN where missing), `target` = `log1p(max(0, pm25 at t + 6 h))`
(negative PM2.5 values are clipped to 0 before the log), and `pm_target` = the clipped PM2.5 at t + 6 h.
Rows are dropped when the target hour is missing/NaN or when all `history_h` history values are NaN.
An issue time not at 00/06/12/18 UTC raises `ValueError`. Output sorted by (`issue_time`, `site_id`), index reset.
`samples_for_split(samples: pd.DataFrame, split, part) -> pd.DataFrame` is the training entry point: it keeps the
rows of `samples` (the output above) whose id `f"{site_id}|{issue_time.isoformat()}"` (the SPLIT-06 id) satisfies
`split_mask(..., split, part)` (SPLIT-07), in input order, index reset. Quarantined samples, test samples and
samples of spatially held-out monitors never reach the train or val samples.

### SMOKE-02 HMS satellite smoke density input + no-HMS ablation switch (`dataset`)
- `HMS_DENSITY = {"Light": 1, "Medium": 2, "Heavy": 3}` (case-insensitive keys on input).
- `rasterize_hms(polygons: Sequence[tuple[shapely.Polygon, str]], x: np.ndarray, y: np.ndarray) -> np.ndarray[uint8]`
  of shape `(len(y), len(x))`: a cell whose centre `(x[j], y[i])` lies inside (or on the boundary of) a polygon gets
  that polygon's density; overlapping polygons give the **maximum**; cells outside every polygon are 0. An unknown
  density string raises `ValueError`. An empty polygon list gives all zeros.
- `is_smoke_episode(density) -> bool | np.ndarray[bool]` = density >= 2 (HMS medium or heavy at the site).
- `SmokeFeatureConfig(use_hms: bool = True)` (frozen dataclass). `tabular_features(pm_hist, hms_density, issue_time,
  cfg) -> np.ndarray[float32]` returns, in order: `log1p` of the 6 history values (each first clipped at 0; NaN replaced by the
  mean of the non-NaN clipped history values), the HMS density (only when `cfg.use_hms`), `sin`/`cos` of hour-of-day (period 24 h) and
  `sin`/`cos` of day-of-year (angle `2*pi*doy/365.25`, `doy` = 1-based `tm_yday`); hour angle is
  `2*pi*(hour + minute/60)/24`. Length is 11 with HMS and 10 without (`n_tabular(cfg)` returns it).

### SMOKE-03 MASSDEN unit change asserted and converted (`dataset.massden_to_ugm3`)
`MASSDEN_UNIT_CHANGE = datetime(2021, 12, 21, 0, 0, tzinfo=UTC)`. `massden_to_ugm3(values, valid_time) -> np.ndarray`
(float32, same shape): for `valid_time < MASSDEN_UNIT_CHANGE` the values are already ug/m3 and returned unchanged; for
`valid_time >= MASSDEN_UNIT_CHANGE` they are kg/m3 and multiplied by 1e9. Asserted plausibility:
- after the change, any raw value > 1e-3 (1e6 ug/m3, i.e. data still in ug/m3) raises `ValueError`;
- negative values raise `ValueError` (NaN is allowed and passed through);
- a naive `valid_time` raises `ValueError`.
The smoke package may re-export `outrun.data.hrrr.massden_to_ugm3` (DATA-03 defines the same semantics); the
constant and function must be importable from `outrun.hazards.smoke.dataset`.

### SMOKE-04 CNN + MLP quantile model with pinball loss (`model.SmokeNet`, `model.pinball_loss`)
`SmokeNet(in_channels: int = 8, patch: int = 9, n_tabular: int = 11, quantiles: tuple[float, ...] = (0.5, 0.9))` is a
`torch.nn.Module`: a CNN branch (2 conv layers + pooling) on the HRRR patch (4 fields x 2 leads = 8 channels, 9x9 at
3 km), flattened and concatenated with an MLP on the tabular features, followed by 3 dense layers.
`forward(patch: Tensor[B, 8, 9, 9], tab: Tensor[B, n_tabular]) -> Tensor[B, len(quantiles)]` (log1p-PM2.5 space).
Default parameter count between 20 k and 500 k (DESIGN: about 0.1 M). Wrong input shapes raise an error.
`pinball_loss(pred: Tensor[B, Q], target: Tensor[B], quantiles) -> Tensor[]` =
`mean over b, q of max(q * (y - pred), (q - 1) * (y - pred))`; it is 0 when `pred == target` everywhere.
`to_pm(pred_log: np.ndarray[B, Q]) -> np.ndarray[B, Q]` = `expm1`, clipped at 0, and made non-decreasing across the
quantile axis (q90 >= q50 via a running max). A few Adam steps on a synthetic set where the target is a linear function
of one tabular feature (64 samples, 60 Adam steps, lr 3e-3) bring the pinball loss below half its initial value.

### SMOKE-05 Baselines: persistence, calibrated raw HRRR-Smoke, linear regression (`baselines`)
- `persistence(pm_hist: np.ndarray[N, 6]) -> np.ndarray[N]` = the most recent non-NaN history value per row (NaN if
  the whole row is NaN).
- `background_median(times, pm, at: datetime, days: int = 30) -> float` = median of `pm` over `[at - days, at)`
  ignoring NaN (the target hour itself is excluded); NaN if there is no value in the window.
- `MassdenCalibrated().fit(massden_f06, background, pm_target)` fits `pm_target - background = a * massden_f06 + b`
  by ordinary least squares and stores the slope and intercept as float attributes `.a` and `.b`; `.predict(massden_f06, background) -> np.ndarray` returns `max(0, a*m + b + background)`.
  On data generated exactly by that model it recovers `a` and `b` to 1e-6. `predict` before `fit` raises `RuntimeError`.
- `LinearSmokeBaseline().fit(X, y)` / `.predict(X)`: ordinary least squares on the same (flattened) inputs as the model,
  target in log1p space; exact on noiseless linear data.

### SMOKE-06 Spatial monitor holdout by 200 km blocks (`dataset.spatial_holdout`)
`spatial_holdout(sites: pd.DataFrame, frac: float = 0.2, block_m: float = 200_000, seed: int = 0) -> set[str]`.
`sites` columns: `site_id`, `x_m`, `y_m`. It does not run its own algorithm: it returns
`set(outrun.eval.splits.spatial_holdout_monitors({site_id: (x_m, y_m)}, frac, block_m, seed))` (SPLIT-06), so the
smoke model and the frozen `splits/smoke.json` hold out the same monitors: for the same monitors and seed,
`spatial_holdout(sites, seed=s) == set(smoke_split(samples, monitors, seed=s).meta["holdout_monitors"])` when every
site has a sample. Block key = `(floor(x_m / block_m), floor(y_m / block_m))`; the sorted blocks are reordered by
`default_rng(seed).permutation(n_blocks)` and taken whole, in that order, until the held-out sites reach at least
`ceil(frac * n_sites)`. At training time the held-out set is read from the frozen split
(`read_split(splits/smoke.json).meta["holdout_monitors"]`), never recomputed with other arguments.
Guarantees: no block is split between held-out and kept sites; held-out share >= `frac` and less than `frac` +
(largest block size / number of sites); the same inputs and seed give the same set; at least one site is kept
whenever `n_sites - (largest block size) >= ceil(frac * n_sites)` (then every order stops before its last block). `frac` outside (0, 1) raises `ValueError`.

### SMOKE-07 Gridded application: IDW observation channel, PM to risk, hold over slots (`adapter`)
- `idw(station_xy: np.ndarray[N, 2], values: np.ndarray[N], x: np.ndarray, y: np.ndarray, power: float = 2.0)
  -> np.ndarray[len(y), len(x)]`: inverse-distance weighting to cell centres; a cell within 1 m of a station takes that
  station's value exactly; every output lies within [min(values), max(values)]; NaN station values are ignored; no
  valid station raises `ValueError`.
- `pm_to_risk(pm) = clip((pm - 35.4) / (150.4 - 35.4), 0, 1)` (thresholds from config), elementwise: 0 at or below
  35.4, 0.5 at 92.9, 1 at or above 150.4.
- `hold_to_slots(field: (Y, X), n_slots: int = 12) -> (n_slots, Y, X)`: the +6 h forecast covers the whole 3 h window,
  so every slot equals the field (HORIZON-03 hold).
- `shelter_deprioritized(r) -> bool | np.ndarray[bool]` = `r > 0.5` (`shelter_deprioritize_r` from config).

### SMOKE-08 Smoke `HazardAdapter` and its exposure-only policy (`adapter.SmokeAdapter`, `adapter.smoke_policy`, `adapter.make_smoke_layer`)
- `smoke_policy(cfg: Mapping | None = None) -> HazardPolicy` from `hazards.smoke` of `configs/hazards.yaml`:
  `block_at` is `None` for every `Profile` (exposure only); `exposure_weight` = civilian 0.5, sensitive 5, ev 0;
  `speed_factor(p) == 1.0` for every p in [0, 1]; `beyond_horizon == "hold"`; `decay_tau_min is None`. With `cfg is
  None` it equals `outrun.core.policy.load_policies()[Hazard.SMOKE]` value for value.
- `make_smoke_layer(pm_slots, x, y, *, kind, issued, valid_until, source, held_out) -> RiskLayer`: `hazard ==
  Hazard.SMOKE`; `prob = pm_to_risk(pm_slots)` as an `xr.DataArray` with dims `("slot", "y", "x")`, dtype float16,
  coords `x`, `y`; `edge_overrides is None`. `kind` not in `{"forecast", "baseline", "observed"}`, a shape mismatch, or
  `valid_until <= issued` raise `ValueError` (the `RiskLayer` constructor's own validation, RISK-01, applies too).
- `SmokeAdapter(provider=None, model=None, source_tag="smoke.cnnmlp@dev", *, checkpoint=None)` satisfies the
  `HazardAdapter` protocol; every argument is optional and construction loads nothing, so `SmokeAdapter()` works
  offline (zero-argument factory, RISK-13): `hazard == Hazard.SMOKE`; `policy` has the same values as
  `smoke_policy()`.
- `provider: Callable[[Scenario], SmokeInputs] | None` (None: the real HRRR/AQS/HMS provider, imported lazily).
  `SmokeInputs` (frozen dataclass): `patches` `(h*w, 8, 9, 9)` float32 and `tabs` `(h*w, n_tabular)` float32, one row
  per 3 km model-grid cell in row-major order (the observation channel = IDW of current monitors, SMOKE-07);
  `grid_shape (h, w)`; `hrrr_pm` `(h, w)` calibrated raw HRRR-Smoke PM2.5 at t0+6 h (SMOKE-05 `MassdenCalibrated`,
  applied by the provider); `obs_pm` `(Y, X)` AQS IDW at t0+6 h on GRID or `None` (no monitors in the bbox);
  `x (X,)`/`y (Y,)` GRID centres; `regrid` `(k, h, w) -> (k, Y, X)` (production: `regrid_nearest`, GRID-05);
  `held_out: bool = False`. The adapter reads only `scn.t0` and `scn.grid_window()`.
- `model: nn.Module | None`; when None and `checkpoint` is a path, a `SmokeNet` is loaded lazily; with neither,
  `predict` raises `RuntimeError`.
- Module-level `default_adapter() -> SmokeAdapter` (zero arguments, loads nothing; the path is kept as `adapter.checkpoint`, a `Path`; `checkpoint =
  get_paths().models / "smoke" / "best.pt"`) is registered with `REGISTRY.register(Hazard.SMOKE, default_adapter)`
  when `hazards.base` is implemented.
- `predict(scn)`: `pm = to_pm(model(patches, tabs))[:, 0]` (column 0 = the median, quantiles `(0.5, 0.9)`),
  reshaped to `grid_shape`, `hold_to_slots`, regridded; `kind="forecast"`, source = `source_tag` (starts with
  `"smoke.cnnmlp"`).
- `baseline(scn)`: `hold_to_slots(hrrr_pm)` regridded; `kind="baseline"`, source starting `"smoke.hrrr"`.
- `observed(scn)`: `None` when `obs_pm is None`; else `hold_to_slots(obs_pm)`; `kind="observed"`, source starting
  `"smoke.aqs"`.
- Every layer: built with `make_smoke_layer(pm_slots, x=scn.grid_window().x_centers(),
  y=scn.grid_window().y_centers(), issued=scn.t0, valid_until=scn.t0 + 6 h, held_out=inputs.held_out)`: shape
  `(12, Y, X)`, `prob = pm_to_risk(pm)`, `edge_overrides=None`. Provider `x`/`y` that differ from those centres (1e-6 m) raise `ValueError`.

### SMOKE-09 Evaluation metrics (`baselines.smoke_metrics`)
`smoke_metrics(pred_q50, pred_q90, obs, episode, threshold: float = 35.4) -> dict` (all PM2.5 in ug/m3). `rmse`, `mae`,
the episode errors and `csi`/`pod`/`far` use `pred_q50`; `q90_coverage` uses `pred_q90`. Keys:
`rmse`, `mae`, `rmse_episode`, `mae_episode` (on rows where `episode` is True; NaN if none), `csi`, `pod`, `far` (for
exceeding `threshold`, i.e. `> threshold`), and `q90_coverage` (share of rows with `obs <= pred_q90`). Perfect
predictions give `rmse == mae == 0`; CSI/POD/FAR with a zero denominator are 0.0. Mismatched lengths or empty input
raise `ValueError`.
