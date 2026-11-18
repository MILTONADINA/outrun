# 02: Grid, risk stack, policies, scenarios, calibration, adapter base

Modules (under `src/outrun/`):

| Module | Contents |
|---|---|
| `core/grid.py` | `Grid`, `GRID`, `GridWindow`, lon/lat <-> EPSG:5070, bbox -> window, regridding (nearest, point aggregation, polar) |
| `core/risk.py` | `Hazard`, `RiskLayer`, `LayerMeta`, `RiskStack` (zarr IO, sampling, overlay PNG), `HazardAdapter`, `build_stack` |
| `core/policy.py` | `Profile`, `HazardPolicy`, `make_speed_factor`, `blocks`, `load_policies`, `load_hazard_config`, `apply_beyond_horizon` |
| `core/scenario.py` | `Scenario` |
| `core/synthetic.py` | Moving Gaussian hazards, `synthetic_window`, `synthetic_scenario`, `synthetic_layer`, `SyntheticAdapter`, `synthetic_stack` |
| `core/calibrate.py` | Brier, Brier skill, reliability bins, ECE, `IsotonicCalibrator`, `TemperatureCalibrator`, serialisation, reliability plot |
| `hazards/base.py` | `AdapterRegistry`, `REGISTRY`, `register_adapter`, `get_adapter`, `validate_adapter_output`, shared training loop |

Tests: `tests/unit/test_grid.py`, `test_risk.py`, `test_policy.py`, `test_scenario.py`, `test_synthetic.py`,
`test_calibrate.py`, `test_hazard_base.py`.

Design references: DESIGN 1.2 (interfaces, beyond-horizon table), 2.2 (core rows, `hazards/base.py`), 3 (shared
definitions, calibration), 4.2 (slots, sampling), 5 (overlay API). Config values: `configs/hazards.yaml`,
`configs/domains.yaml`. The default config folder is `outrun.config.get_paths().configs` (DATA-15:
`OUTRUN_CONFIG_DIR` if set, else `<repo>/configs`); `load_policies(path=None)` and `load_hazard_config(path=None)`
resolve their default file through it, so every config reader honours the same override.

**Priority.** Every requirement marked **Must** is required to ship the proposal (the router, optimizers, app and
all six adapters depend on it). **Should** items are depth.

## Shared conventions

- Units: metres (EPSG:5070) for x/y; seconds internally for clock time, minutes only where a name says `_min`;
  probabilities in [0, 1]; all `datetime`s timezone-aware UTC. A naive `datetime` is rejected with `ValueError`.
- Routing window: `N_SLOTS = 12` slots of `SLOT_MINUTES = 15` minutes from the scenario issue time `t0`.
  Slot `s` covers `[t0 + 15 s min, t0 + 15 (s + 1) min)`; layers are piecewise constant per slot.
- Cell `(row, col)` of a window: row 0 is the northern-most row, col 0 the western-most column.
- Validation errors raise `ValueError` unless a requirement says otherwise.

## Interfaces (DESIGN 1.2, exact)

```python
class Hazard(Enum): STORM="storm", TORNADO="tornado", FLOOD="flood", WILDFIRE="wildfire", ICE="ice", SMOKE="smoke"

@dataclass(frozen=True)
class RiskLayer:
    hazard: Hazard
    prob: xr.DataArray                              # (slot, y, x) float16, calibrated P on GRID
    edge_overrides: dict[int, np.ndarray] | None    # per-edge P(slot) for point features
    kind: Literal["forecast", "baseline", "observed"]
    issued: datetime
    valid_until: datetime
    source: str
    held_out: bool

class Profile(Enum): CIVILIAN="civilian", SENSITIVE="sensitive", EV="ev"

@dataclass(frozen=True)
class HazardPolicy:
    block_at: dict[Profile, float | None]
    speed_factor: Callable[[float], float]
    exposure_weight: dict[Profile, float]
    beyond_horizon: Literal["hold", "decay", "drop"]
    decay_tau_min: float | None

class HazardAdapter(Protocol):                      # runtime_checkable
    hazard: Hazard; policy: HazardPolicy
    def baseline(self, scn: Scenario) -> RiskLayer: ...
    def predict(self, scn: Scenario) -> RiskLayer: ...
    def observed(self, scn: Scenario) -> RiskLayer | None: ...
```

## Requirements: grid (`core/grid.py`)

### GRID-01 The shared GRID and windows (Must)
- `GRID` is a frozen `Grid(crs="EPSG:5070", resolution_m=1000.0)`; both values equal `configs/domains.yaml`
  `grid.crs` and `grid.resolution_m`.
- `GridWindow(x0, y0, nx, ny, res=1000.0, crs="EPSG:5070")` is a frozen dataclass. `x0` is the west edge and `y0`
  the north edge in metres. `nx < 1`, `ny < 1` or `res <= 0` raises `ValueError` at construction.

### GRID-02 lon/lat <-> EPSG:5070 (Must)
- `lonlat_to_xy(lon, lat) -> (x, y)` and `xy_to_lonlat(x, y) -> (lon, lat)` accept scalars or arrays (any shape) and
  return float64 `np.ndarray`s of the input shape.
- Results equal `pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)` (and its inverse) within
  1 mm. The round trip `xy_to_lonlat(*lonlat_to_xy(lon, lat))` returns the input within 1e-7 degrees.

### GRID-03 Bounding box to window (Must)
`bbox_to_window(bbox, buffer_km=0.0, res=GRID.resolution_m) -> GridWindow`, `bbox = (lon_min, lat_min, lon_max,
lat_max)`.
- The bbox boundary is densified (at least 100 points per side) and projected; its x/y extent is grown by
  `buffer_km * 1000` m on every side and snapped **outward** to multiples of `res`: `x0 = floor(xmin / res) * res`,
  `y0 = ceil(ymax / res) * res`, and `nx`, `ny` are the smallest counts whose window covers `xmax`, `ymin`.
- So: `x0 % res == 0`, `y0 % res == 0`; every densified boundary point (grown by the buffer) lies inside the window
  bounds; and the window is minimal (each side is less than `res` beyond the grown extent).
- `lon_min >= lon_max`, `lat_min >= lat_max`, a longitude outside [-180, 180], a latitude outside [-90, 90] or
  `buffer_km < 0` raises `ValueError`.

### GRID-04 Cell geometry (Must)
`GridWindow` methods:
- `shape -> (ny, nx)`; `bounds -> (xmin, ymin, xmax, ymax) = (x0, y0 - ny*res, x0 + nx*res, y0)`.
- `x_centers() -> (nx,)` ascending, `x0 + (c + 0.5) res`; `y_centers() -> (ny,)` descending, `y0 - (r + 0.5) res`.
- `cell_of(x, y) -> (row, col)` int64 arrays of the input shape: `col = floor((x - x0)/res)`,
  `row = floor((y0 - y)/res)`. A point outside the half-open window `[x0, x0+nx*res) x (y0-ny*res, y0]` gets
  `(-1, -1)`. The west and north edges are inside; the east and south edges are outside.
- `contains(x, y) -> bool array` is `row >= 0` of `cell_of`.

### GRID-05 Nearest-neighbour regridding (Must)
`regrid_nearest(values, lon, lat, window, max_dist_m, fill=np.nan) -> np.ndarray (ny, nx) float32` for any source
grid given by lon/lat (MRMS lat/lon, SEVIR and HRRR Lambert grids).
- `lon`, `lat` are either 2-D arrays of `values.shape`, or 1-D arrays with `values.shape == (len(lat), len(lon))`.
- Each target cell centre takes the value of the source pixel nearest to it, measured in EPSG:5070 metres. If that
  distance is greater than `max_dist_m`, the cell gets `fill`.
- Mismatched shapes raise `ValueError`.

### GRID-06 Point aggregation (Must)
`aggregate_to_grid(values, x, y, window, agg="max", fill=0.0) -> np.ndarray (ny, nx) float32`.
- Points (EPSG:5070 metres, any equal shapes) are binned with `cell_of`; points outside the window and NaN values
  are ignored.
- `agg` is `"max"`, `"mean"`, `"sum"` or `"count"`. Cells without any point get `fill`, except for `"count"`, where
  they are 0. Any other `agg` raises `ValueError`.

### GRID-07 Polar radar data to the grid (Should)
- `polar_to_xy(ranges_m, azimuths_deg, radar_lon, radar_lat) -> (x, y)`, each `(n_az, n_range)` EPSG:5070 metres.
  Azimuth is degrees clockwise from true north. Each gate lies at the geodesic distance `range` and azimuth from the
  radar (WGS84), within 25 m for ranges up to 150 km.
- `polar_to_grid(values, ranges_m, azimuths_deg, radar_lon, radar_lat, window, agg="max", fill=0.0)` equals
  `aggregate_to_grid(values, *polar_to_xy(...), window, agg, fill)`.

### GRID-08 Named domains (Must)
`domain_window(name, domains=None) -> GridWindow`.
- `domains` maps names to `{"lon": [min, max], "lat": [min, max], "buffer_km": optional}` (the `domains:` block of
  `configs/domains.yaml`, which is read when `domains is None`).
- Returns `bbox_to_window((lon[0], lat[0], lon[1], lat[1]), buffer_km=buffer_km or 0)`. An unknown name raises
  `KeyError`.

## Requirements: risk layers and the stack (`core/risk.py`)

### RISK-01 Hazard and RiskLayer validation (Must)
- `Hazard` members in this order: STORM, TORNADO, FLOOD, WILDFIRE, ICE, SMOKE, with lowercase string values
  (`Hazard("storm") is Hazard.STORM`).
- `RiskLayer` is frozen and has exactly the fields of DESIGN 1.2, in that order. Construction raises `ValueError`
  when: `hazard` is not a `Hazard`; `prob` is not an `xr.DataArray` with dims exactly `("slot", "y", "x")` and at
  least one slot; any `prob` value is NaN or outside [0, 1]; `kind` is not one of the three literals; `issued` or
  `valid_until` is naive; `valid_until <= issued`; or an `edge_overrides` value is not a 1-D array of length
  `prob.sizes["slot"]` with values in [0, 1]. `edge_overrides=None` and `{}` are valid.

### RISK-02 Building a stack from layers (Must)
`RiskStack.from_layers(layers, t0, window, policies=None, n_slots=12) -> RiskStack`.
- `stack.data` is an `xr.DataArray` with dims `("hazard", "slot", "y", "x")`, shape `(6, n_slots, ny, nx)`, dtype
  float16, coords `hazard = [h.value for h in Hazard]` (enum order), `slot = 0..n_slots-1`,
  `y = window.y_centers()`, `x = window.x_centers()`. `stack.t0 == t0`, `stack.window == window`.
- A hazard without a layer is all zeros and has no entry in `stack.meta`. For a hazard with a layer,
  `stack.meta[h]` is a `LayerMeta(kind, source, issued, valid_until, held_out)` copied from the layer.
- Each layer's `prob` is first passed through `apply_beyond_horizon` (HORIZON-01..04) with the hazard's policy
  (`policies[h]`, or `load_policies()[h]` when `policies is None`) and the hazard's `decay_zero_after_min` /
  `blocked_linger_min` from `configs/hazards.yaml` (defaults 120 and 15). Its `edge_overrides` get the same treatment
  and are kept in `stack.edge_overrides[h]` (length `n_slots` each).
- A layer whose `y`/`x` coords differ from the window centres (within 1e-6 m), a layer **without** `y` or `x`
  coords (a coordless layer is never accepted, so mis-gridded data cannot slip through), two layers for the same
  hazard, or a naive `t0` raise `ValueError`. Every adapter therefore builds its layers with
  `x=scn.grid_window().x_centers()`, `y=scn.grid_window().y_centers()` (or provider coords equal to them).
- `RiskStack.zeros(t0, window, n_slots=12)` is the all-zero stack with empty `meta`.

### RISK-03 Slot timing (Must)
- `stack.slot_start(s) == t0 + timedelta(minutes=15 * s)` for `0 <= s < n_slots`; other `s` raise `IndexError`.
- `stack.slot_of(t) = floor((t - t0) / 15 min)`, clamped to `n_slots - 1` for times past the window. `t < t0` or a
  naive `t` raises `ValueError`. `slot_of(slot_start(s)) == s` and `slot_of(slot_start(s) + 14 min 59 s) == s`.

### RISK-04 Zarr round trip (Must)
`stack.to_zarr(path)` then `RiskStack.from_zarr(path)` restores, exactly: `data` values (float16, bitwise), coords,
dims, `t0`, `window`, `meta` (all fields, tz-aware datetimes) and `edge_overrides` (edge ids and arrays).
`from_zarr` of a missing path raises `FileNotFoundError`.

### RISK-05 Sampling (Must)
- `stack.get(hazard, slot) -> np.ndarray (ny, nx) float32`, the stored values.
- `stack.sample(hazard, slot, x, y) -> np.ndarray float32` of the input shape: the stored value of the cell that
  contains each point (GRID-04 `cell_of`); points outside the window give 0.0.
- `stack.sample_points(x, y) -> np.ndarray (6, n_slots, n_points) float32` for 1-D `x`, `y`;
  `sample_points(x, y)[i, s] == sample(list(Hazard)[i], s, x, y)`.
- `hazard` may be a `Hazard` or its string value.

### RISK-06 Overlay PNG format (Must)
`stack.overlay_png(hazard, slot, path=None) -> bytes`, a PNG (written to `path` too when given):
- mode RGBA, width `nx`, height `ny`, pixel `(row, col)` = cell `(row, col)` (row 0 north).
- alpha `= floor(255 * p + 0.5)` (clipped to 0..255) where `p` is the stored float16 value as float32; RGB is the
  hazard's colour `HAZARD_COLORS[hazard]` (an `(r, g, b)` tuple, one distinct colour per hazard) wherever alpha > 0.
- `decode_overlay(png_bytes) -> np.ndarray (ny, nx) float32 = alpha / 255`.
- `stack.overlay_corners() -> [[lon, lat]] * 4`, the window corners in the order top-left, top-right, bottom-right,
  bottom-left (MapLibre image-source order), via `xy_to_lonlat`.

### RISK-07 What you see is what routes (Must)
For every hazard and slot, the overlay pixel of each cell equals the value the stack's sampling returns for any
point in that cell, up to the 8-bit quantisation of RISK-06:
`decode_overlay(stack.overlay_png(h, s)) == floor(255 * stack.sample(h, s, xc, yc) + 0.5) / 255` exactly at all cell
centres `(xc, yc)`, so `|overlay - sampled| <= 0.5 / 255`. Overlays are rendered from `stack.data` itself, never from
a separately resampled array.

### RISK-08 Risk-source switch (Must)
`RiskSource` is a `StrEnum` defined here once (`MODEL="model"`, `BASELINE="baseline"`, `OBSERVED="observed"`,
`NONE="none"`, `CURRENT_ONLY="current-only"`); `RISK_SOURCES = tuple(m.value for m in RiskSource)`. The planner
(PLAN-02), the app models (APP-03) and the CLI import it from `outrun.core.risk` and never redefine it.
`build_stack(adapters, scn, source, policies=None, n_slots=12) -> RiskStack` for `source` in `RISK_SOURCES` (a
`RiskSource` or its string value); `adapters` is a sequence of
`HazardAdapter`s; the stack uses `scn.t0` and `scn.grid_window()`.
- `"model"`: each adapter's `predict(scn)`. `"baseline"`: `baseline(scn)`. `"observed"`: `observed(scn)`; a `None`
  result leaves that hazard at zero with no meta. `"none"`: all zeros, no adapter method is called.
- `"current-only"` (planner a', risk frozen at t0): each adapter's `predict(scn)` slot 0 is held for all `n_slots`
  slots (no beyond-horizon rule; its meta keeps the layer's kind and has `source + "+current-only"`). This stack is
  for **overlays only** (what the map shows for planner a'); routing never uses it. The router's current-only
  freeze is done once, at the CostModel level at `slot(t0_s)` (PLAN-02), so a request issued after the scenario
  t0 freezes its own slot.
- An unknown `source` raises `ValueError`.

## Requirements: policies and the beyond-horizon rule (`core/policy.py`)

### RISK-09 Hazard policies from config (Must)
- `Profile` members CIVILIAN, SENSITIVE, EV with values `"civilian"`, `"sensitive"`, `"ev"`.
- `HazardPolicy` is frozen and has exactly the DESIGN 1.2 fields, in order.
- `make_speed_factor(slope) -> Callable`: `f(p) = clip(1 - slope * p, 1/3, 1)`; accepts a float or an array and
  returns the same shape. `slope < 0` raises `ValueError`.
- `blocks(policy, profile, p) -> bool | np.ndarray`: `False` everywhere if `policy.block_at[profile] is None`,
  else `p >= block_at`.
- `load_policies(path=None) -> dict[Hazard, HazardPolicy]` builds all six policies from `configs/hazards.yaml`
  (`block_at`, `exposure_weight` keyed by `Profile`, `speed_factor = make_speed_factor(speed_factor_slope)`,
  `beyond_horizon`, `decay_tau_min`). With the shipped config: tornado civilian weight 20 and `block_at` 0.5; flood
  civilian `block_at` 0.4 and EV 0.7; smoke sensitive weight 5 and civilian 0.5; storm `decay` with tau 30; tornado
  `drop`; flood, wildfire, ice, smoke `hold`; ice `speed_factor(1.0) == 0.5`. A hazard missing from the file, or an
  invalid `beyond_horizon`, raises `ValueError`.
- `load_hazard_config(path=None) -> dict` returns the parsed YAML mapping (used for hazard-specific extras such as
  `decay_zero_after_min` and `blocked_linger_min`).

### HORIZON-01 Decay beyond the horizon (storm) (Must)
`apply_beyond_horizon(prob, policy, *, t0, issued, valid_until, n_slots=12, slot_minutes=15, zero_after_min=120.0,
linger_min=15.0) -> np.ndarray` returns float32 of shape `(n_slots, *prob.shape[1:])`. Slot `k` starts at
`t_k = t0 + k * slot_minutes`. Slot `k` is **valid** iff `k < prob.shape[0]` and `t_k < valid_until`; valid slots
form a prefix of length `n_valid` and are copied unchanged. `L = prob[n_valid - 1]`, and for a beyond slot
`dt_k = max(0, (t_k - valid_until) in minutes)`.
For `beyond_horizon == "decay"`: slot `k` = `L * exp(-dt_k / decay_tau_min)`, and 0 if `zero_after_min` is not None
and `(t_k - issued) >= zero_after_min` minutes. Storm example (t0 = issued, horizon 60 min, tau 30): slots 0-3 copied,
slot 4 = L, slot 5 = L e^-0.5, slot 6 = L e^-1, slot 7 = L e^-1.5, slots 8-11 = 0.
`decay` with `decay_tau_min` None or <= 0 raises `ValueError`.

### HORIZON-02 Drop beyond the horizon (tornado) (Must)
For `beyond_horizon == "drop"`: beyond slots are 0, except cells where `L >= block_threshold` and
`dt_k < linger_min` keep `L` (blocked debris cells keep blocking for 15 min). `block_threshold` is the minimum of the
non-None `policy.block_at` values; if all are None nothing lingers. Tornado example (horizon 30 min, block 0.5):
slots 0-1 copied, slot 2 keeps only cells with `L >= 0.5`, slots 3-11 are 0.

### HORIZON-03 Hold beyond the horizon (flood, wildfire, ice, smoke) (Must)
For `beyond_horizon == "hold"`: every beyond slot equals `L`. A layer with fewer slots than `n_slots` (for example
an ice layer with 3 hourly-held steps covering 45 min) is extended by holding its last valid slot.

### HORIZON-04 Beyond-horizon edge cases (Must)
- Works for any trailing shape, including 1-D per-edge override arrays `(S,)`.
- The input array is never modified; the output is a new float32 array with values in [0, 1].
- `n_valid == 0` (horizon ends at or before `t0`, or an empty slot axis) raises `ValueError`; naive datetimes raise
  `ValueError`; an unknown `beyond_horizon` value raises `ValueError`.

## Requirements: scenarios (`core/scenario.py`)

### RISK-10 Scenario (Must)
`Scenario` is a frozen dataclass: `id: str`, `t0: datetime`, `bbox: tuple[float, float, float, float]` (lon_min,
lat_min, lon_max, lat_max), `city: str`, `hazards: tuple[Hazard, ...]`, `held_out: Mapping[Hazard, bool]` (default
empty), `title: str = ""`, `case: Mapping[str, Any]` (default empty; event date, sources, notes), `synthetic: bool =
False`, `window: GridWindow | None = None`.
- Construction raises `ValueError` when: `id` does not match `^[a-z0-9][a-z0-9_-]*$`; `t0` is naive or its UTC offset
  is not zero; the bbox is invalid (GRID-03 rules); `hazards` contains duplicates or non-`Hazard` items; a `held_out`
  key is not in `hazards`.
- `grid_window()` returns `window` when given, else `bbox_to_window(bbox)`.
- `is_held_out(h)` returns `held_out.get(h, False)`.
- `to_dict()` returns JSON-serialisable data (t0 as ISO 8601 with offset, hazards and held-out keys as string values,
  window as its fields or null); `Scenario.from_dict(d)` inverts it exactly (`from_dict(to_dict(s)) == s`).
  `save(path)` / `Scenario.load(path)` write and read that JSON.

## Requirements: synthetic hazards (`core/synthetic.py`)

### RISK-11 Moving Gaussian hazard (Must)
`moving_gaussian(n_slots=12, shape=(32, 32), start=(8.0, 8.0), velocity=(1.5, 1.0), sigma=3.0, peak=0.9)` returns
float32 `(n_slots, ny, nx)`, value `peak * exp(-((r - cy)^2 + (c - cx)^2) / (2 sigma^2))` with centre
`(cy, cx) = start + s * velocity` in (row, col) cell units. Deterministic (no randomness). `sigma <= 0`, `peak`
outside (0, 1] or `n_slots < 1` raises `ValueError`.

### RISK-12 Synthetic window, scenario, layers, adapters and stack (Must)
- `SYNTHETIC_STARTS`: storm (4, 4), tornado (16, 4), flood (4, 16), wildfire (24, 24), ice (16, 16), smoke (28, 8).
- `synthetic_window(n=32) == GridWindow(x0=-150_000.0, y0=1_300_000.0, nx=n, ny=n, res=1000.0)`.
- `synthetic_scenario(t0=None, n=32)`: id `"synthetic"`, `synthetic=True`, all six hazards, none held out,
  `window=synthetic_window(n)`, `bbox` = the lon/lat extent of that window, `t0` default 2019-05-26 03:00 UTC.
- `synthetic_layer(hazard, window, t0, *, kind="forecast", start=None, velocity=(1.5, 1.0), sigma=3.0, peak=0.9,
  n_slots=12, horizon_min=180, source=None) -> RiskLayer`: `prob` float16 dims `(slot, y, x)` with the window's
  centre coords and values `moving_gaussian(n_slots, window.shape, start or SYNTHETIC_STARTS[hazard], ...)`;
  `issued = t0`, `valid_until = t0 + horizon_min`, `source` default `f"synthetic.{hazard.value}.{kind}"`,
  `held_out=False`, `edge_overrides=None`.
- `SyntheticAdapter(hazard, policy=None)` satisfies `isinstance(_, HazardAdapter)`; `policy` defaults to
  `load_policies()[hazard]`. `predict(scn)` is `synthetic_layer(kind="forecast")` on `scn.grid_window()` and
  `scn.t0`; `baseline(scn)` repeats `predict` slot 0 in every slot (persistence, kind `"baseline"`);
  `observed(scn)` is the Gaussian with start shifted by (+1, +1) cells (kind `"observed"`).
- `synthetic_stack(t0=None, n=32) -> RiskStack` = `build_stack` of the six adapters' `"model"` source on
  `synthetic_scenario(t0, n)`; for n = 32 it equals the `risk_stack_array` test fixture within float16 precision.

## Requirements: calibration (`core/calibrate.py`)

All metric functions take `p` (probabilities) and `y` (0/1 labels) of equal shape (any shape, flattened), and an
optional boolean `mask` of the same shape (True = use; for ignored/uncertain pixels). Shape mismatch, no usable
elements, NaN or out-of-[0, 1] `p`, or labels other than 0/1 raise `ValueError`.

### CAL-01 Brier score and Brier skill (Must)
- `brier_score(p, y, mask=None) = mean((p - y)^2)`.
- `brier_skill_score(p, y, climatology=None, mask=None) = 1 - BS / BS_ref`, `BS_ref = mean((c - y)^2)` with
  `c = climatology` or `mean(y)`. `BS_ref == 0` raises `ValueError`.

### CAL-02 Reliability bins (Must)
`reliability_bins(p, y, n_bins=10, mask=None) -> ReliabilityBins(edges, counts, mean_pred, frac_pos)`: equal-width
bins on [0, 1], `edges` of length `n_bins + 1`; element bin = `min(floor(p * n_bins), n_bins - 1)` (so p = 1 is in
the last bin); `counts` int; `mean_pred` and `frac_pos` are NaN for empty bins. `n_bins < 1` raises `ValueError`.

### CAL-03 Expected calibration error (Must)
`expected_calibration_error(p, y, n_bins=10, mask=None) = sum_b (counts_b / N) * |frac_pos_b - mean_pred_b|` over
non-empty bins of CAL-02.

### CAL-04 Isotonic calibration (Must)
`IsotonicCalibrator().fit(p, y, mask=None) -> self`; `transform(p) -> np.ndarray` float32, input shape kept.
- The mapping is non-decreasing, outputs lie in [0, 1], inputs outside the fitted range are clipped to it.
- On its fit data the calibrated Brier score is <= the raw Brier score; on a large sample with true frequency `p^2`
  it recovers `p^2` (within 0.07 at p = 0.5 and 0.9).
- `transform` before `fit` raises `RuntimeError`.

### CAL-05 Temperature scaling (Must)
`TemperatureCalibrator().fit(p, y, mask=None) -> self` fits one `T > 0` (attribute `temperature`) minimising the
negative log-likelihood of `sigmoid(logit(p) / T)`, with `p` clipped to [1e-6, 1 - 1e-6] before `logit`;
`transform(p) = sigmoid(logit(p) / T)` as float32, input shape kept. It is monotone and maps 0.5 to 0.5. For
overconfident scores `p = sigmoid(2 z)` with labels drawn from `sigmoid(z)` it recovers `T` within 0.25 of 2.
`transform` before `fit` raises `RuntimeError`.

### CAL-06 Calibrator serialisation (Must)
`calibrator.to_dict()` returns JSON-serialisable data with a `"type"` key (`"isotonic"` or `"temperature"`);
`calibrator_from_dict(d)` rebuilds a calibrator whose `transform` is identical (to float32 precision) on any input.
An unknown type raises `ValueError`.

### CAL-07 Reliability diagram (Should)
`plot_reliability(bins, path, title="")` writes a PNG reliability diagram (predicted vs observed frequency, diagonal
reference, bin counts) with matplotlib's Agg backend and returns `path`.

## Requirements: adapter base and shared training loop (`hazards/base.py`)

### RISK-13 Adapter registry and conformance (Must)
- `hazards.base` re-exports `HazardAdapter` from `core.risk` (same object).
- `AdapterRegistry()`: `register(hazard, factory, *, replace=False)` stores a zero-argument factory (an adapter class
  or function); registering a hazard twice without `replace=True` raises `ValueError`. `get(hazard)` accepts a
  `Hazard` or its string value, calls the factory once and caches the instance (later calls return the same object);
  an unknown or unregistered hazard raises `KeyError`; an adapter whose `.hazard` differs from the key raises
  `ValueError`. `hazards()` lists registered hazards in `Hazard` order.
- `REGISTRY` is the module-level registry; `register_adapter(hazard, *, replace=False)` is a class decorator that
  registers into it and returns the class unchanged; `get_adapter(hazard)` is `REGISTRY.get(hazard)`.
- `validate_adapter_output(adapter, scn) -> dict[str, RiskLayer | None]` calls `baseline`, `predict` and `observed`
  and returns `{"baseline": ..., "predict": ..., "observed": ...}`. It raises `ValueError` if a result (other than an
  `observed` None) is not a `RiskLayer`, has `hazard != adapter.hazard`, has the wrong kind (`baseline` ->
  `"baseline"`, `predict` -> `"forecast"`, `observed` -> `"observed"`), has `y`/`x` coords missing or different
  from `scn.grid_window()` centres, has `valid_until <= scn.t0`, or (baseline and forecast only) `issued > scn.t0`.
- `load_default_adapters() -> AdapterRegistry` imports the six `hazards/<h>/adapter.py` modules (each registers its
  zero-argument `default_adapter` factory into `REGISTRY` on import, see STORM-08, TORN-10, FLOOD-10, FIRE-10,
  ICE-08, SMOKE-08) and returns `REGISTRY`. It is idempotent (a second call registers nothing new and does not
  raise). Afterwards `REGISTRY.hazards() == list(Hazard)` and `get_adapter(h).hazard is h` for every hazard, with
  no data or model loaded (the CLI and `build_stack` callers get all six adapters this way).

### RISK-14 Training loop with early stopping on validation (Must)
`train_model(model, train_loader, val_loader, loss_fn, cfg, optimizer=None) -> TrainResult`.
`TrainConfig(checkpoint_dir, max_epochs=20, patience=3, min_delta=0.0, lr=1e-3, weight_decay=0.0, device="cpu",
resume=False, seed=0, max_batches_per_epoch=None, grad_clip=None)`.
- Loaders yield `(x, y)` batches. Epochs are numbered from 1. Each epoch trains (model in train mode) then computes
  the validation loss (eval mode, no gradients). Epoch losses are batch losses averaged with weights `len(x)`.
- The default optimizer is `torch.optim.AdamW(lr, weight_decay)`. `torch.manual_seed(cfg.seed)` is called at the start
  of a fresh run.
- A val loss below `best - min_delta` is an improvement. Training stops early after `patience` consecutive epochs
  without improvement, or at `max_epochs`.
- At the end the model holds the **best-val** weights. `TrainResult(best_epoch, best_val_loss, epochs_run,
  stopped_early, history, best_checkpoint, last_checkpoint)`; `history` has one dict per epoch with keys `epoch`,
  `train_loss`, `val_loss`, `seconds`.
- A non-finite training loss raises `FloatingPointError` (the fp16 autocast NaN guard). An empty validation loader
  raises `ValueError`.

### RISK-15 Checkpoint and resume (Must)
- Every epoch writes `checkpoint_dir/last.pt` (model and optimizer state, epoch, best val loss, best epoch, bad-epoch
  counter, history) and, on improvement, `checkpoint_dir/best.pt` (model state, epoch, val loss). The directory is
  created if missing.
- With `cfg.resume=True` and an existing `last.pt`, training restores that state and continues from the next epoch;
  the result (weights, history, `epochs_run`) equals an uninterrupted run with the same data and config. Without
  `last.pt` it starts fresh.
- `save_checkpoint(path, model, optimizer=None, **extra)` and `load_checkpoint(path, model, optimizer=None) -> dict`
  (returns the extra entries) round-trip model weights exactly.

### RISK-16 Timed epoch and device selection (Must)
- `time_one_epoch(model, loader, loss_fn, *, device="cpu", optimizer=None, max_batches=None) -> EpochTiming(seconds,
  batches, samples)` trains for one epoch (or `max_batches` batches) and reports wall-clock seconds (> 0), batches
  and samples seen. `EpochTiming.estimate_total_s(epochs) == seconds * epochs`. A non-finite loss raises
  `FloatingPointError`.
- `select_device(prefer="mps") -> torch.device`: `mps` if preferred and available, else `cpu`; `prefer="cpu"` always
  gives `cpu`; any other value raises `ValueError`.
