# 12: Icy roads (HRRR f01 + weak zone labels, MIL-CNN)

Modules: `src/outrun/hazards/ice/dataset.py`, `src/outrun/hazards/ice/model.py`,
`src/outrun/hazards/ice/baselines.py`, `src/outrun/hazards/ice/adapter.py`.
Tests: `tests/unit/test_ice*.py`. Design: DESIGN 3.5 (source of truth), 1.2 (interfaces), 4.2 (costs).

Purpose: turn zone-level weak labels (NCEI Storm Events by NWS forecast zone) and HRRR f01 surface fields into a
cell-level icy-road probability map with a multiple-instance-learning (MIL) CNN, compare it with a physics rule,
zone-month climatology and logistic regression, and expose it to routing as an `ICE` `RiskLayer` with a bridge prior.

Shared conventions for this spec:
- Temperatures from HRRR are kelvin. 0 C = 273.15 K. A "6-h window" starts at 00, 06, 12 or 18 UTC.
- All datetimes are timezone-aware UTC; a naive datetime raises `ValueError` wherever a function accepts one.
- Core types (`Hazard`, `RiskLayer`, `HazardPolicy`, `Profile`, `Scenario`) are imported from `outrun.core.*`.
- Policy values come from `configs/hazards.yaml` (`hazards.ice`), never hard-coded duplicates.

All requirements below are **Must** (the proposal phrase "icy roads" maps to ICE-01..ICE-08; ICE-04 is also the
"deep learning" row). ICE-09 is Must for the report (T24 metrics) but not in the proposal matrix.

## Requirements

### ICE-01 Weak-label keyword filter (`dataset.label_storm_events`)
`label_storm_events(events: pd.DataFrame) -> pd.DataFrame`. Input columns: `zone_id` (str), `event_type` (str),
`begin` and `end` (tz-aware UTC timestamps), `narrative` (str, may be empty or NaN). Output: the same rows plus an
int column `label`, with rows dropped according to these rules:
- `event_type` in `ICE_EVENT_TYPES = ("Ice Storm", "Sleet", "Freezing Fog")` -> `label = 1`.
- `event_type` in `("Winter Storm", "Winter Weather")` and the narrative matches `ICE_KEYWORDS`
  (regex `freezing rain|freezing drizzle|ice accum|glaze|sleet`, case-insensitive, searched anywhere) -> `label = 1`.
- `Winter Storm` / `Winter Weather` rows without a keyword match (including NaN narrative) are **kept with
  `label = -1`** ("ambiguous": they are excluded from training, never treated as negatives).
- Every other event type is dropped.
Event-type matching is exact after stripping whitespace, case-insensitive. Output index is reset (0..n-1); input order
is preserved. Missing required columns raise `KeyError`.

### ICE-02 Bags: (zone, 6-h window) labels and hard negatives (`dataset.assign_bag_labels`)
`window_start(t: datetime) -> datetime` floors `t` to the 6-h window start (00/06/12/18 UTC).
`assign_bag_labels(windows: pd.DataFrame, events: pd.DataFrame, t2m_hard_neg_c: float = 3.0) -> pd.DataFrame`.
`windows` has columns `zone_id`, `window_start` (tz-aware, each a 6-h window start; others raise `ValueError`) and
`t2m_min_k` (minimum 2 m temperature over the zone and window, kelvin). `events` is the output of ICE-01.
An event overlaps a window `[ws, ws + 6 h)` of the same zone when `begin < ws + 6 h` and `end >= ws`.
Per window:
- any overlapping `label = 1` event -> bag `label = 1` (`hard_negative = False`);
- else any overlapping `label = -1` event -> the window is **dropped** (ambiguous);
- else, if `t2m_min_k < 273.15 + t2m_hard_neg_c` -> `label = 0` and `hard_negative = True`;
- else `label = 0` and `hard_negative = False`.
Output columns: `zone_id`, `window_start`, `label` (int), `hard_negative` (bool), with the kept windows in input order.
`bags_for_split(bags: pd.DataFrame, split, part) -> pd.DataFrame` is the training entry point: it keeps the rows of
`bags` (columns `zone_id`, `window_start`, others carried) whose id `f"{zone_id}|{window_start.isoformat()}"`
(the SPLIT-05 id) satisfies `split_mask(..., split, part)` (SPLIT-07), in input order, index reset. Quarantined and
test bags never reach the train or val bags.

### ICE-03 Bag input tensor: 20 channels on a 64x64 crop (`dataset.build_bag_input`)
`ICE_FIELDS` is the tuple, in this order: `TMP:surface`, `TMP:2 m`, `DPT:2 m`, `CFRZR`, `CICEP`, `CSNOW`, `CRAIN`,
`PRATE`, `FRZR` (9 HRRR `wrfsfc` f01 fields). `ICE_LEAD_H = 1`; `check_lead(lead_h: int)` raises `ValueError` for any
lead other than 1 (f00 precipitation/accumulations are degenerate; this is the ice-specific "f01 only" rule, stricter
than the per-field `data.hrrr.check_lead` of DATA-03). The HRRR `VAR:LEVEL` strings are `data.hrrr.HRRR_ICE_FIELDS`.
`build_bag_input(fields_t, fields_tm3, elevation, zone_mask, crop=64) -> (x, mask_crop)`:
- `fields_t`, `fields_tm3`: `(9, H, W)` arrays (time t and t-3 h); `elevation`: `(H, W)`; `zone_mask`: `(H, W)` bool.
- `x` is float32 `(20, crop, crop)`: channels 0-8 = `fields_t`, 9-17 = `fields_tm3`, 18 = elevation,
  19 = zone mask as 0/1. No date, time or lat/lon channel exists (seasonality shortcut).
- The crop is centred on the zone mask's centroid `(cr, cc)` (mean row/col, rounded with Python `round`): rows
  `[cr - crop//2, cr - crop//2 + crop)`, likewise for columns, then shifted so it stays inside the grid (start
  clamped to `[0, max(0, H - crop)]`). If `H` or `W` < `crop`, the missing rows/cols are zero-padded at the bottom/right.
- `mask_crop` is the bool `(crop, crop)` zone mask (equal to `x[19] > 0.5`).
- An empty `zone_mask`, mismatched shapes, or a field axis not of length 9 raise `ValueError`.

### ICE-04 MIL-CNN: U-Net cell logits + LogSumExp zone pooling (`model.IceMILNet`, `model.mil_pool`, `model.mil_loss`)
`IceMILNet(in_channels: int = 20, base: int = 32)` is a `torch.nn.Module` U-Net encoder-decoder with three 2x
down-sampling stages (channels `base, 2*base, 4*base, 8*base`, skip connections).
`forward(x: Tensor[B, in_channels, H, W]) -> Tensor[B, 1, H, W]` (per-cell logits, same H and W) for any H, W that are
multiples of 8 (64x64 in training; 32x32 must also work). Default parameter count is between 0.5 M and 5 M (DESIGN:
about 2 M). A wrong channel count raises an error (any exception) rather than silently running.
`mil_pool(cell_logits: Tensor[B, 1, H, W], mask: Tensor[B, H, W] bool) -> Tensor[B]` = LogSumExp of the logits over the
in-zone cells only (`log(sum_i exp(l_i))`, numerically stable). It ignores out-of-zone cells entirely, so
`max_i l_i <= zone_logit <= max_i l_i + log(n_in_zone)`. A bag whose mask is empty raises `ValueError`.
`mil_loss(cell_logits, mask, labels: Tensor[B] float in {0,1}) -> Tensor[]` = mean binary cross-entropy with logits of
`mil_pool(...)` against `labels`. Gradients flow to the cells; 40 Adam steps (lr 1e-3, train mode) on a synthetic separable
toy set (positives have a hot spot inside the zone) reduce the loss and rank positive bags above negative ones on
average (both measured in train mode).
`cell_probabilities(model, x) -> np.ndarray[B, H, W]` = sigmoid of the cell logits in eval mode under `torch.no_grad()`,
values in [0, 1].

### ICE-05 Baselines: physics rule, zone-month climatology, logistic regression (`baselines`)
- `physics_rule(fields: Mapping[str, np.ndarray]) -> np.ndarray[bool]`, keys from `ICE_FIELDS`:
  `((CFRZR == 1) | (CICEP == 1) | ((PRATE > 0) & (TMP:surface <= 273.15))) & (TMP:surface <= 273.15)`, elementwise,
  same shape as the inputs. A missing key raises `KeyError`. `physics_rule_zone(fields, mask) -> float` = 1.0 if the
  rule is true in any in-zone cell, else 0.0.
- `ZoneMonthClimatology().fit(zone_ids, months, labels)`; `.predict(zone_ids, months) -> np.ndarray` = the training
  positive rate of that (zone, month); a (zone, month) unseen in training falls back to the global training rate.
  Values in [0, 1]. `predict` before `fit` raises `RuntimeError`.
- `zone_features(x: (20, H, W), mask: (H, W) bool) -> np.ndarray[36]` = per field channel 0-17 the in-zone mean
  (first 18 values) then the in-zone max (next 18). `LogisticZoneBaseline(seed=0).fit(X, y)` /
  `.predict_proba(X) -> np.ndarray` (P(label=1), in [0, 1]) wraps a standardised sklearn logistic regression.

### ICE-06 Zone-level isotonic calibration (`model.ZoneCalibrator`)
`ZoneCalibrator().fit(zone_probs: np.ndarray, labels: np.ndarray)` fits isotonic regression (out-of-bounds clipped) on
validation zone probabilities. `.transform(p: np.ndarray) -> np.ndarray` keeps the input shape (works on cell maps of
any shape), is monotone non-decreasing in `p`, and returns values in [0, 1]. On a perfectly separable fit set,
probabilities below every negative map to 0 and above every positive map to 1. `transform` before `fit` raises
`RuntimeError`. `save(path)` / `ZoneCalibrator.load(path)` round-trip to identical outputs.

### ICE-07 Risk layer: 3-h steps to slots (hold), bridge prior, speed factor (`adapter`)
- `steps_to_slots(step_probs: (n_steps, Y, X), step_hours: float = 3.0, n_slots: int = 12, slot_minutes: int = 15)
  -> (n_slots, Y, X)`: slot `s` uses step `min(floor(s * slot_minutes / (60 * step_hours)), n_steps - 1)`; beyond the
  last step the last step is **held** (HORIZON-03). `n_steps == 0` raises `ValueError`.
- `bridge_overrides(edge_probs: Mapping[int, np.ndarray], is_bridge: Mapping[int, bool], multiplier: float = 1.5)
  -> dict[int, np.ndarray]`: only bridge edges are returned, each with `p' = min(1, multiplier * p)` per slot;
  non-bridge edges are absent (the grid value applies). `multiplier = 1.0` (ablation) returns the bridge values
  unchanged. Inputs are not mutated.
- `ice_speed_factor(p) = clip(1 - 0.5 p, 1/3, 1)` (slope `speed_factor_slope` from config), elementwise on floats or
  arrays; identical to `outrun.core.policy.make_speed_factor(0.5)` (RISK-09): `1.0` at p = 0, `0.5` at p = 1, `1/3`
  for p >= 4/3, `1.0` for p <= 0.

### ICE-08 Ice `HazardAdapter` and its policy (`adapter.IceAdapter`, `adapter.ice_policy`, `adapter.make_ice_layer`)
- `ice_policy(cfg: Mapping | None = None) -> HazardPolicy` from `hazards.ice` of `configs/hazards.yaml` (or `cfg`, the
  same mapping shape): `block_at` is `None` for every `Profile`; `exposure_weight` = civilian 3, sensitive 3, ev 1;
  `speed_factor` behaves as `ice_speed_factor`; `beyond_horizon == "hold"`; `decay_tau_min is None`. With `cfg is None`
  it equals `outrun.core.policy.load_policies()[Hazard.ICE]` value for value (delegation is the intended implementation).
- `make_ice_layer(prob_slots, x, y, *, kind, issued, valid_until, source, held_out, edge_overrides=None) -> RiskLayer`:
  `hazard == Hazard.ICE`; `prob` is an `xr.DataArray` with dims `("slot", "y", "x")`, dtype float16, coords `x`, `y`
  equal to the inputs, values clipped to [0, 1]; the other fields are passed through. `kind` not in
  `{"forecast", "baseline", "observed"}`, a shape mismatch, or `valid_until <= issued` raise `ValueError` (the
  `RiskLayer` constructor's own validation, RISK-01, applies too).
- `IceAdapter(provider=None, model=None, calibrator=None, source_tag="ice.milcnn@dev", *, graph=None,
  checkpoint=None)` satisfies the `HazardAdapter` protocol; every argument is optional and construction loads
  nothing, so `IceAdapter()` works offline (zero-argument factory, RISK-13): `hazard == Hazard.ICE`; `policy` has
  the same values as `ice_policy()`.
- `provider: Callable[[Scenario], IceInputs] | None` (None: the real HRRR provider, imported lazily). `IceInputs`
  (frozen dataclass): `x_steps` `(n_steps, 20, h, w)` float32 ICE-03 inputs on the 3 km HRRR grid, one per 3-h step
  from `scn.t0`; `fields_steps`: one ICE-05 field mapping per step; `truth` `(12, Y, X)` labelled Storm Events zones
  on GRID or `None` (no coverage); `x (X,)`/`y (Y,)` GRID centres; `regrid` `(k, h, w) -> (k, Y, X)` (in production
  `outrun.core.grid.regrid_nearest`, GRID-05); `held_out: bool = False`. The adapter reads only `scn.t0` and
  `scn.grid_window()`.
- `model: nn.Module | None`; when None and `checkpoint` is a path, an `IceMILNet` (and a saved `ZoneCalibrator`
  next to it) is loaded lazily; with neither, `predict` raises `RuntimeError`. `calibrator` is identity when None.
- `graph: RoadGraph | Callable[[Scenario], RoadGraph] | None` keys `edge_overrides` by its CSR edge ids; without a
  graph, `edge_overrides=None`. **It must be the same graph the router plans on**
  (same CSR edge ids): the CLI/backend pass the graph they route on (`outrun route --place` included); the
  default's `build_graph(bbox=scn.bbox, ...)` is only right when the router also uses the scenario-bbox graph.
  ASTAR-18 rejects override ids `>= n_edges`, but a same-size different graph cannot be detected, so callers own
  this.
- Module-level `default_adapter() -> IceAdapter` (zero arguments, loads nothing; the path is kept as `adapter.checkpoint`, a `Path`; `checkpoint =
  get_paths().models / "ice" / "best.pt"`, `graph = lambda scn: build_graph(bbox=scn.bbox, cache_dir=get_paths().proc
  / "graphs")`) is registered with `REGISTRY.register(Hazard.ICE, default_adapter)` when `hazards.base` is
  implemented.
- `predict(scn)`: `steps_to_slots(calibrator.transform(cell_probabilities(model, x_steps)))`, regridded;
  `kind="forecast"`, source = `source_tag` (starts with `"ice.milcnn"`). With a graph: `edge_probs[e] =
  EdgeRiskSampler.from_window(graph, scn.grid_window()).sample(p_slots[None])[e, 0]` (ASTAR-14) for every bridge
  edge, `edge_overrides = bridge_overrides(edge_probs, {e: True for bridge edges})` (ICE-07), keys in
  `range(graph.n_edges)`.
- `baseline(scn)`: `steps_to_slots(stack of physics_rule(fields).astype(float32) per step)` (ICE-05), regridded;
  `kind="baseline"`, source starting `"ice.rule"`, `edge_overrides=None`.
- `observed(scn)`: `None` when `truth is None`; else the truth slots; `kind="observed"`, source starting
  `"ice.stormevents"`, `edge_overrides=None`.
- Every layer: built with `make_ice_layer(..., x=scn.grid_window().x_centers(), y=scn.grid_window().y_centers(),
  issued=scn.t0, valid_until=scn.t0 + 180 min, held_out=inputs.held_out)`, shape `(12, Y, X)`. Provider `x`/`y` that differ from those centres (1e-6 m) raise `ValueError`.

### ICE-09 Zone-window evaluation metrics (`baselines.zone_metrics`)
`zone_metrics(p: np.ndarray, y: np.ndarray, climatology: np.ndarray, threshold: float) -> dict` with keys `auprc`
(average precision), `roc_auc`, `brier`, `brier_skill` (`1 - brier / brier(climatology)`), `csi`, `pod`, `far` (at
`p >= threshold`). Perfect predictions give `auprc == roc_auc == 1`, `brier == 0`, `csi == pod == 1`, `far == 0`;
predicting the climatology itself gives `brier_skill == 0`. Empty input or a single class in `y` raises `ValueError`.
CSI/POD/FAR with a zero denominator are 0.0.
