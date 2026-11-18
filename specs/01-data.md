# 01: Data (fetch, manifest, readers, croppers, disk budget, config)

Modules: `src/outrun/config.py`, `src/outrun/data/*.py`
(`manifest`, `fetch`, `disk`, `sevir`, `tornet`, `hrrr`, `mrms`, `ndws`, `wsts`, `stormevents`, `spc`, `aqs`, `hms`,
`nwszones`, `dem`, `census`, `facilities`).
Tests: `tests/unit/test_data_*.py`, `tests/unit/test_config.py` (fixtures generated in `tmp_path`; no real data, no
network).

Purpose: get every raw source listed in `data/manifest.yaml` onto the right disk, resumably and checksummed; read each
raw format once (SEVIR HDF5, TorNet NetCDF, HRRR/MRMS GRIB2, NDWS TFRecord, WSTS GeoTIFF, CSV/zip tables, shapefiles)
and turn it into the arrays and tables the hazard packages train on; keep both disks inside the DESIGN 7.3 budget.

## Conventions for this spec

- All datetimes are timezone-aware UTC. A function that takes a datetime raises `ValueError` for a naive one.
  Exception: xarray `time` coordinates are `datetime64` values, which are UTC by convention (numpy has no tz).
- A bbox is a tuple `(lon_min, lat_min, lon_max, lat_max)` in degrees (EPSG:4326). "Inside a bbox" means
  `lon_min <= lon <= lon_max and lat_min <= lat <= lat_max` (closed intervals). Longitudes in `[0, 360)` (GRIB
  convention) are converted with `lon - 360 if lon > 180` before any comparison.
- `GB = 10**9` bytes (decimal, matching `aws s3 ls --summarize` and DESIGN 7.3).
- **`[CANDIDATE]` format facts** (h5 keys, NetCDF variable names, TFRecord feature names, GeoTIFF band names, file
  name patterns, fill values, IR encoding) are module-level constants, verified in T1/T3/T10/T13 against the real
  files. Tests import the constants, so correcting a constant after verification never requires a test change.
- Heavy optional libraries (`herbie`, `cfgrib`, `boto3`, `tfrecord`, `geopandas`, `rasterio`, `torch`) are imported
  inside function bodies, never at module top level, so importing any `outrun.data.*` module is fast.
- Network and GRIB decoding are behind injected callables (a `Transport`, a `range_get`, a `decoder`). Unit tests pass
  fakes; the defaults (requests/boto3/Herbie/cfgrib) are exercised only by acceptance runs (T1/T3).

Priority: **Must (proposal)** = listed in the proposal coverage matrix of `00-overview.md` ("real weather data",
"real radar data"). **Must (pipeline)** = needed by later tasks; not graded on its own.

## Requirements

### DATA-01 SEVIR reader: CATALOG filter, VIL + IR107 h5 to 2 km uint8 memmap with projection metadata
**Must (proposal: real radar + satellite data).** Module `data/sevir.py`. Radar = NEXRAD VIL mosaic, satellite =
GOES-16 IR107.

`[CANDIDATE]` constants: `VIL_KEY = "vil"`, `IR107_KEY = "ir107"`, `ID_KEY = "id"` (h5 dataset names; data arrays are
`(N_events, H, W, T)` with time last, `id` is a byte-string array of event ids); `IR_RAW_SCALE = 0.01`,
`IR_RAW_OFFSET = 273.15` (raw int16 -> kelvin `raw * scale + offset`); storm-event ids start with `"S"`, random-event
ids with `"R"`. Fixed (not candidate): `IR_TB_MIN_K = 180.0`, `IR_TB_MAX_K = 320.0`.

- `read_catalog(path) -> pd.DataFrame`: reads `CATALOG.csv`; `time_utc` becomes tz-aware UTC timestamps; all other
  columns are kept as read. Columns used downstream: `id`, `file_name`, `file_index`, `img_type`, `time_utc`, `proj`,
  `llcrnrlat`, `llcrnrlon`, `urcrnrlat`, `urcrnrlon`. A missing one of these raises `KeyError`.
- `filter_catalog(catalog, years=(2018, 2019), img_types=("vil", "ir107"), bbox=None) -> pd.DataFrame` keeps a row when
  all of: `time_utc.year in years`; `img_type in img_types`; `id` starts with `"S"` (STORMEVENTS only); and, when
  `bbox` is given, the event rectangle `[llcrnrlon, urcrnrlon] x [llcrnrlat, urcrnrlat]` **intersects** the bbox
  (closed intervals: touching counts). Then it keeps only ids that still have a row for **every** type in `img_types`
  (events missing IR are dropped from both arms). Output sorted by `(id, img_type)`, index reset. An empty result is a
  valid (empty) DataFrame.
- `pool_vil_2x2(vil: np.ndarray[uint8, (..., 2S, 2S)]) -> np.ndarray[uint8, (..., S, S)]`: 2x2 **average** pooling of
  the last two axes, rounded half up: `(a + b + c + d + 2) // 4` computed without uint8 overflow. Odd last-two
  dimensions raise `ValueError`.
- `decode_ir(raw, scale=IR_RAW_SCALE, offset=IR_RAW_OFFSET) -> np.ndarray[float32]` = `raw * scale + offset` (kelvin).
- `ir_to_uint8(tb_k) -> np.ndarray[uint8]`: clip to `[180, 320]` K, then `floor((tb - 180) / 140 * 255 + 0.5)`
  (180 K -> 0, 320 K -> 255, 250 K -> 128, values below/above clip to 0/255). NaN maps to 0.
- `build_sevir_memmap(rows, raw_dir, out_path, ir_scale=IR_RAW_SCALE, ir_offset=IR_RAW_OFFSET) -> pd.DataFrame`:
  `rows` is a `filter_catalog` result containing both `vil` and `ir107` rows per id. For each id in sorted order it reads
  the VIL array `h5[VIL_KEY][file_index]` from `raw_dir / file_name` of the `vil` row (shape `(2S, 2S, T)`) and the IR
  array `h5[IR107_KEY][file_index]` from the `ir107` row (shape `(S, S, T)`). It checks `h5[ID_KEY][file_index]`
  (decoded) equals the id in both files (mismatch -> `ValueError`). It writes a NumPy `.npy` file at `out_path`
  (`np.lib.format.open_memmap`, so `np.load(out_path, mmap_mode="r")` works) of dtype uint8 and shape
  `(N, T, 2, S, S)`: channel 0 = `pool_vil_2x2` of VIL, channel 1 = `ir_to_uint8(decode_ir(IR))`, time moved to axis 1.
  S and T are taken from the files (production: S = 192, T = 49). VIL spatial size != 2 x IR size, or VIL T != IR T,
  raises `ValueError`; a missing h5 file raises `FileNotFoundError`.
  Returns per-event metadata in memmap row order, also written to `out_path.with_suffix(".meta.parquet")`: columns
  `row` (0..N-1), `id`, `time_utc`, `proj`, `llcrnrlat`, `llcrnrlon`, `urcrnrlat`, `urcrnrlon` (taken from the `vil` row),
  used later for reprojection to GRID.

### DATA-02 TorNet reader: catalog, NetCDF Level-II chips to fp16 arrays, official split rule
**Must (proposal: real radar data).** Module `data/tornet.py`.

`[CANDIDATE]` constants (T1/T10 verify against `mit-ll/tornet` and real files): `VARIABLES = ("DBZ", "VEL", "KDP",
"RHOHV", "ZDR", "WIDTH")`; each is a NetCDF variable with dims `("time", "azimuth", "range", "sweep")`; 1-D coordinate
variables `azimuth` (degrees) and `range` (metres); `N_TILTS = 2` (sweep 0 and 1 are used); `FILL_VALUES`: a dict giving
one constant per variable used for missing (NaN) gates.

- `read_tornet_catalog(path) -> pd.DataFrame`: requires columns `filename`, `category`, `start_time` (missing ->
  `KeyError`). `start_time` becomes tz-aware UTC. Adds `year` (UTC year), `julian_day` (UTC day of year, 1..366) and,
  if no `label` column exists, `label = 1` where `category == "TOR"` else 0.
- `SplitRule` (frozen dataclass): `mode: Literal["column", "julian"] = "column"`, `column: str = "type"`,
  `modulus: int = 20`, `train_below: int = 17`. The default is `[CANDIDATE]`; T1 records the verified rule in NOTES.md
  and only the defaults change.
- `official_split(catalog, rule=SplitRule(), years=(2015, 2016, 2019)) -> pd.DataFrame`: the rows whose `year` is in
  `years` (input order kept, index reset), plus a column `split` in `{"train", "test"}`:
  - `mode="column"`: `split = catalog[rule.column].str.strip().str.lower()`; any value not in `{"train", "test"}`
    raises `ValueError`; a missing column raises `KeyError`.
  - `mode="julian"`: `"train"` if `julian_day % modulus < train_below` else `"test"`. All rows of one
    `(year, julian_day)` therefore share a split.
  Validation (10 % of train Julian days) is not part of this rule; it is built by the split builder (spec 15, SPLIT-02).
- `read_chip(path, fill_values=FILL_VALUES) -> np.ndarray[float16, (27, A, R)]`. Channel layout:
  - 0-11: last time frame (`time = -1`), ordered `for var in VARIABLES for tilt in (0, 1)`, NaN replaced by
    `fill_values[var]`;
  - 12-23: last minus first frame (`time = -1` minus `time = 0`), same order, both after filling (all zeros if the file
    has one frame);
  - 24: missing mask, 1.0 where any of the 12 raw last-frame values was NaN, else 0.0;
  - 25: range coordinate in km (`range / 1000`) broadcast along azimuth;
  - 26: azimuth coordinate in degrees broadcast along range.
  A missing variable raises `KeyError`.
- `convert_chips(paths, out_path, fill_values=FILL_VALUES) -> np.memmap`: writes `.npy` of shape `(len(paths), 27, A, R)`
  float16 (row i = `read_chip(paths[i])`); chips with differing `(A, R)` raise `ValueError`.

### DATA-03 HRRR reader: `.idx` byte-range field fetch, in-memory crop, f01 rule, MASSDEN unit handling
**Must (proposal: real weather data).** Module `data/hrrr.py`.

- `IdxEntry` (frozen dataclass): `num: int`, `start: int`, `end: int | None` (inclusive), `init: datetime`, `var: str`,
  `level: str`, `fcst: str`.
- `parse_idx(text) -> list[IdxEntry]`: one line per GRIB message, `num:start:d=YYYYMMDDHH:VAR:LEVEL:FCST:` (extra
  trailing `:`-fields are ignored, blank lines skipped). `end` = next line's `start - 1`; the last entry has
  `end = None`. `init` is the `d=` time in UTC. A malformed line raises `ValueError`.
- `select_fields(entries, fields) -> list[IdxEntry]`: each field is `"VAR:LEVEL"`, matched exactly against
  `f"{e.var}:{e.level}"`. Returns every matching entry, in idx order, no duplicates. A field with no match raises
  `KeyError` naming the field.
- `byte_ranges(entries) -> list[tuple[int, int | None]]`: one `(start, end)` per entry, sorted by start, with contiguous
  ranges (`end + 1 == next start`) merged.
- `range_header(start, end) -> str`: `"bytes=start-end"`, or `"bytes=start-"` when `end is None`.
- `fetch_fields(grib_url, idx_text, fields, range_get) -> bytes`: parses, selects, merges, then calls
  `range_get(grib_url, start, end)` **once per merged range** in ascending order and returns the concatenated bytes.
  It never downloads a whole file.
- `crop_bbox(da, bbox) -> xr.DataArray`: `da` has dims `(y, x)` (optionally leading dims) with 2-D coordinates
  `latitude` and `longitude` on `(y, x)`. Returns the smallest index rectangle `da.isel(y=slice(y0, y1),
  x=slice(x0, x1))` that contains every cell whose (lon, lat) is inside the bbox. No cell inside raises `ValueError`.
- f01 rule: `ACCUM_FIELDS = frozenset({"PRATE", "FRZR", "APCP", "CFRZR", "CICEP", "CSNOW", "CRAIN"})`;
  `check_lead(var, lead_h)` raises `ValueError` if `lead_h < 0`, or if `lead_h == 0` and `var in ACCUM_FIELDS`
  (precipitation/accumulated fields are zero or degenerate at f00). Otherwise returns `None`.
- `HRRR_ICE_FIELDS` `[CANDIDATE level strings, T3]`: the 9 `"VAR:LEVEL"` strings for TMP surface, TMP 2 m, DPT 2 m,
  CFRZR, CICEP, CSNOW, CRAIN, PRATE, FRZR. `HRRR_V4_START = datetime(2020, 12, 3, tzinfo=UTC)`.
- `ice_cycle_times(first_winter=2020, last_winter=2024, cycles=(0, 3, 6, 9, 12, 15, 18, 21)) -> list[datetime]`: winter
  `w` covers Nov 15 of year `w` through Mar 15 of year `w + 1` (both days inclusive); one time per day per cycle hour;
  times before `HRRR_V4_START` are dropped; sorted ascending, no duplicates.
- `choose_ice_cycles(mean_mb_per_message, threshold_mb=1.9) -> tuple[int, ...]`: `(0, 6, 12, 18)` if
  `mean_mb_per_message > threshold_mb`, else the 3-hourly `(0, 3, ..., 21)` (DESIGN 3.5 80 GB cap).
- `choose_smoke_leads(mean_mb_per_message, n_days=1096, cap_gb=60.0) -> tuple[int, ...]`: streamed volume
  `n_days * 4 cycles * len(leads) * 4 fields * mb / 1000` GB; returns `(1, 6)` if that is `<= cap_gb` for `(1, 6)`,
  else `(6,)` (DESIGN 3.6: drop f01).
- MASSDEN: `MASSDEN_UNIT_CHANGE = datetime(2021, 12, 21, tzinfo=UTC)`. `massden_to_ugm3(values, valid_time)` has exactly
  the SMOKE-03 semantics (before the change values are ug/m3 and returned unchanged as float32; at or after it they are
  kg/m3 and multiplied by 1e9; after the change a raw value > 1e-3 raises `ValueError`; negative values raise
  `ValueError`; NaN passes through; naive time raises `ValueError`). The smoke package may delegate to this function.
  `check_massden_units(units, valid_time)`: the GRIB `units` string must contain `"kg"` (case-insensitive) when
  `valid_time >= MASSDEN_UNIT_CHANGE` and must not contain it before; a mismatch raises `ValueError`.

### DATA-04 MRMS reader: QPE 01H Pass2 + FLASH GRIB2 decode, crop, 1/3/6/24 h accumulations
**Must (proposal: real radar + weather data).** Module `data/mrms.py`.

`[CANDIDATE]` constants: `QPE_PRODUCT = "MultiSensor_QPE_01H_Pass2_00.00"`; `FLASH_PRODUCTS` (tuple of product names,
T3 lists `s3://noaa-mrms-pds/CONUS/`); key layout `CONUS/{product}/{YYYYMMDD}/MRMS_{product}_{YYYYMMDD}-{HHMMSS}.grib2.gz`.

- `mrms_key(product, valid_time) -> str` builds that key. `parse_mrms_name(name) -> tuple[str, datetime]` accepts a
  key or a bare file name (`.grib2` or `.grib2.gz`) and returns `(product, valid_time)` (UTC);
  `parse_mrms_name(mrms_key(p, t)) == (p, t)`. A name not matching `MRMS_<product>_<YYYYMMDD>-<HHMMSS>.grib2[.gz]`
  raises `ValueError`.
- `read_mrms(path, decoder=None) -> xr.DataArray`: if `path` ends with `.gz` it is gunzipped to a temporary file first
  (the raw file is never modified); `decoder(path_to_grib2) -> xr.DataArray` (default: cfgrib via xarray) decodes it.
  The result goes through `clean_mrms`. The temporary file is removed afterwards.
- `clean_mrms(values) -> float32` (array or DataArray): negative values (MRMS -3 = no coverage, -999 = missing) become
  NaN; everything else unchanged.
- `crop_latlon(da, bbox) -> xr.DataArray`: `da` has 1-D dimension coordinates `latitude` (ascending or descending) and
  `longitude` (`[0, 360)` or `[-180, 180)`). Returns the cells inside the bbox with longitudes converted to
  `[-180, 180)`; the original latitude order is kept. No cell inside raises `ValueError`.
- `accumulate(hourly, windows_h=(1, 3, 6, 24)) -> xr.Dataset`: `hourly` is a `(time, y, x)` DataArray of 1-h QPE (mm)
  whose `time` values are on exact hours and strictly increasing (else `ValueError`). Variable `qpe_{h}h` at time `t` is
  the sum of `hourly` at `t - (h-1) h, ..., t`; it is NaN (per cell) if any of those hours is absent from `time` or NaN
  at that cell ("windows with missing hours are skipped"). Output keeps the input `time` coordinate.

### DATA-05 NDWS TFRecord reader to npy
**Must (proposal: real weather data — GRIDMET weather, MODIS/VIIRS channels).** Module `data/ndws.py`.

`[CANDIDATE]` constants: `NDWS_INPUT_FEATURES = ("elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "NDVI",
"population", "erc", "PrevFireMask")` (channel order), `NDWS_LABEL = "FireMask"`, `NDWS_SIZE = 64`; every feature is a
float list of length `size * size` in row-major order.

- `parse_ndws_example(features: Mapping[str, Sequence[float] | np.ndarray], size=NDWS_SIZE) -> tuple[x, y]`:
  `x` float32 `(12, size, size)` in `NDWS_INPUT_FEATURES` order, `y` int8 `(size, size)` from `NDWS_LABEL`. A missing
  feature raises `KeyError`; a wrong length raises `ValueError`; a label value not in `{-1, 0, 1}` raises `ValueError`.
  `-1` (uncertain) is preserved.
- `read_ndws_tfrecord(path, size=NDWS_SIZE) -> tuple[X, y]`: all examples in file order, `X` `(N, 12, size, size)`
  float32, `y` `(N, size, size)` int8 (uses the pure-Python `tfrecord` package, never TensorFlow).
- `convert_ndws(split_files: Mapping[str, Sequence[Path]], out_dir, size=NDWS_SIZE) -> dict[str, tuple[Path, Path]]`:
  split names must be in `{"train", "eval", "test"}` (else `ValueError`). For each split, files are read in sorted path
  order and written to `out_dir/ndws_{split}.npy` (X) and `out_dir/ndws_{split}_labels.npy` (y); returns
  `{split: (x_path, y_path)}`.
- `fit_clip_stats(X_train, lo_pct=0.1, hi_pct=99.9, skip=("th", "PrevFireMask")) -> dict[str, dict[str, float]]`:
  per channel name not in `skip`: `lo`, `hi` (percentiles over finite values), then `mean`, `std` of the clipped
  values. `normalize(X, stats) -> float32` clips each such channel to `[lo, hi]` and returns `(x - mean) / std`
  (`std == 0` -> divide by 1); channels without stats (wind direction `th` and the fire mask) are returned unchanged.

### DATA-06 WSTS GeoTIFF reader + 12-channel NDWS mapper with per-channel shift statistics
**Must (proposal: real weather data + VIIRS satellite).** Module `data/wsts.py`.

`[CANDIDATE]` constants (T13 verifies band descriptions and units): `WSTS_TO_NDWS`: dict mapping each NDWS channel name
except `"population"` to a WSTS band description (e.g. `"th" -> "Wind direction"`, `"PrevFireMask" -> "Active fire"`);
`UNIT_CONVERSIONS`: dict channel -> `(scale, offset)` applied as `v * scale + offset` after resampling (identity for
channels not listed). Fixed rules: `WIND_DIR_CHANNEL = "th"`, `FIRE_CHANNEL = "PrevFireMask"`.

- `read_wsts_day(path) -> tuple[dict[str, np.ndarray], float]`: reads a GeoTIFF; returns `{band description: float32
  array}` (descriptions stripped, matched case-insensitively later) and the source pixel size in metres (from the
  transform). A band without a description raises `ValueError`.
- `map_to_ndws(bands, src_res_m, population=None, target_res_m=1000.0) -> np.ndarray[float32, (12, H', W')]` with
  `H' = round(H * src_res_m / target_res_m)` (same for W). Channel order = `NDWS_INPUT_FEATURES`. Resampling over each
  target cell's footprint:
  - continuous channels: area-weighted mean, NaN source pixels ignored (all-NaN -> NaN);
  - wind direction: circular mean of unit vectors, result in degrees `[0, 360)` (350 and 10 average to ~0, not 180; a float32
    result of exactly 360.0 from rounding is tolerated);
  - previous fire mask: 1.0 if any source pixel overlapping the cell is > 0, else 0.0.
  `population` must have shape `(H', W')` (else `ValueError`); `None` gives an all-NaN population channel. A required
  band missing from `bands` raises `KeyError`.
- `channel_shift_stats(wsts, ndws, threshold=0.2, quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame`: inputs are
  `(N, 12, H, W)` and `(M, 12, h, w)`. One row per NDWS channel name (index, in order); columns `ks` (two-sample
  Kolmogorov-Smirnov statistic on the flattened finite values), `wsts_q{p}` and `ndws_q{p}` for each quantile
  (`p` = `int(round(q * 100))`, two digits, e.g. `wsts_q05`), and `shifted` (`ks > threshold`). A channel with no finite
  values on either side gets `ks = NaN` and `shifted = True` (listed as a caveat).

### DATA-07 Weather observations: Storm Events, SPC tornado tracks, AQS PM2.5, HMS smoke polygons
**Must (proposal: real weather data).** Modules `data/stormevents.py`, `data/spc.py`, `data/aqs.py`, `data/hms.py`.

- `stormevents.read_storm_events(paths) -> pd.DataFrame` (one path or a sequence; `.csv` or `.csv.gz`; rows
  concatenated in path order, index reset). Output columns: `event_id`, `episode_id`, `event_type` (stripped),
  `state_fips`, `cz_type`, `cz_fips`, `zone_id` (`nwszones.zone_id(state_fips, cz_fips)` when `cz_type == "Z"` and `state_fips` is in
  `STATE_FIPS_TO_POSTAL`; otherwise missing (None/NaN), e.g. county rows, marine zones, territories, so a real national
  file never raises), `begin`, `end` (UTC), `begin_lat`, `begin_lon`, `end_lat`, `end_lon` (NaN when missing), `narrative`
  (`EVENT_NARRATIVE` and `EPISODE_NARRATIVE` joined by one space, NaN treated as empty, stripped), `tor_width_m`
  (`TOR_WIDTH` yards x 0.9144) and `tor_length_m` (`TOR_LENGTH` miles x 1609.344), NaN when missing.
  Times: `BEGIN_YEARMONTH` (YYYYMM), `BEGIN_DAY`, `BEGIN_TIME` (HHMM integer, e.g. `5` = 00:05, `1530` = 15:30), same for
  END, in the local standard time given by `CZ_TIMEZONE` such as `"CST-6"`: UTC = local - offset hours (CST-6 -> +6 h).
  A timezone without a trailing offset is looked up in `{"EST": -5, "CST": -6, "MST": -7, "PST": -8, "AKST": -9,
  "HST": -10}`; anything else raises `ValueError`. The output feeds ICE-01 (`zone_id`, `event_type`, `begin`, `end`,
  `narrative`) and the flood labels.
- `stormevents.filter_event_types(df, types) -> pd.DataFrame`: rows whose `event_type` equals one of `types`
  (case-insensitive, stripped), order kept, index reset.
- `spc.read_spc_tornadoes(path) -> pd.DataFrame` `[CANDIDATE columns: om, yr, mo, dy, date, time, tz, st, mag, slat,
  slon, elat, elon, len, wid, sg]`: columns `om`, `begin` (UTC from `date` + `time`; `tz == 3` is CST = UTC-6 all year,
  `tz == 9` is GMT, any other `tz` raises `ValueError`), `st`, `mag`, `slat`, `slon`, `elat`, `elon` (end point
  `0, 0` replaced by the start point), `len_m` (miles x 1609.344), `wid_m` (yards x 0.9144). Rows with `slat == 0` are
  dropped; if an `sg` column exists only `sg == 1` rows (whole tracks) are kept. Index reset.
- `spc.swath_polygons(df, crs="EPSG:5070", min_width_m=50.0) -> geopandas.GeoDataFrame`: one polygon per row: the
  segment start -> end projected from EPSG:4326 to `crs`, buffered by `max(wid_m, min_width_m) / 2` with round caps
  (a zero-length track becomes a disc). Columns of `df` are kept.
- `spc.rasterize_mask(geoms, x, y) -> np.ndarray[bool, (len(y), len(x))]`: True where the cell centre `(x[j], y[i])`
  lies inside or on the boundary of any geometry (same rule as SMOKE-02); empty `geoms` gives all False.
- `aqs.read_aqs_hourly(path) -> pd.DataFrame`: `path` is an EPA `hourly_88101_YYYY.zip` (containing one CSV) or the CSV
  itself. Keeps `Parameter Code == 88101`. Columns: `site_id` = `f"{state:0>2}-{county:0>3}-{site:0>4}"` from
  `State Code`, `County Code`, `Site Num` (as zero-padded strings), `time` (UTC from `Date GMT` + `Time GMT`), `pm25`
  (`Sample Measurement`, negative values clipped to 0), `lat`, `lon`. Several POCs at the same `(site_id, time)` are
  averaged into one row. Sorted by `(site_id, time)`, index reset. Output feeds SMOKE-01.
- `hms.read_hms(path, crs="EPSG:5070") -> list[tuple[shapely.Polygon, str]]`: reads an HMS smoke shapefile (field
  `Density`), reprojects to `crs`, explodes MultiPolygons into Polygons, and normalizes density to `"Light"`,
  `"Medium"` or `"Heavy"` (case-insensitive, stripped); any other value raises `ValueError`. The list is the input of
  SMOKE-02 `rasterize_hms`. `hms.hms_date(name) -> date` parses the `YYYYMMDD` in a file name such as
  `hms_smoke20230607.zip` (no 8-digit date -> `ValueError`).

### DATA-08 SEVIR IR107 60 GB rule
**Must (pipeline).** Module `data/sevir.py`.

- `parse_sevir_filename(name) -> tuple[str, int, date, date]`: `[CANDIDATE pattern]`
  `SEVIR_{TYPE}_STORMEVENTS_{YYYY}_{MMDD}_{MMDD}.h5` -> `(type lowercased, year, start date, end date)`; anything else
  raises `ValueError`.
- `apply_ir107_cap(files: Sequence[tuple[str, int]], cap_bytes=60 * GB) -> CapResult` (`kept: list[str]`,
  `dropped: list[str]`, `total_bytes: int`, `over_cap: bool`). If the total size is `<= cap_bytes`, nothing is dropped.
  Otherwise files of year 2018 whose end date is on or before June 30 are dropped one at a time in ascending start-date
  order until the total is `<= cap_bytes`. Files of 2019 (val/test) and 2018 files ending after June are never dropped.
  `kept` keeps the input order; `total_bytes` is the kept total; `over_cap` is True if still above the cap.

### DATA-09 MRMS FLASH event-window hour selection
**Must (pipeline).** Module `data/mrms.py`. `flash_hours(report_times, rainy_hours, pad_h=6, random_frac=0.2, seed=0,
max_hours=None) -> list[datetime]`:
- Event hours = the union, over all reports, of the hour-floored report time plus every whole hour within `+-pad_h`
  (inclusive): `2 * pad_h + 1` hours per isolated report.
- Random hours: `k = round(random_frac * len(event_hours))` hours drawn without replacement with
  `np.random.default_rng(seed).choice` from the sorted rainy hours that are not event hours (`k` capped at the number
  available). Same inputs and seed -> same output.
- Cap: if `max_hours` is given and the total exceeds it, random hours are dropped first (the latest-drawn first); if event hours alone exceed it,
  only the first `max_hours` event hours in time order are kept.
- Returns the selected hours sorted ascending, without duplicates. Naive datetimes raise `ValueError`.

### DATA-10 Manifest parsing and validation
**Must (pipeline).** Module `data/manifest.py`.

- `ACCESS = ("s3_unsigned", "http", "kaggle", "herbie_byte_range")`, `TARGETS = ("hdd", "ssd", "stream")`.
- `Source` (frozen dataclass): `id`, `desc`, `url`, `access`, `approx_bytes: int`, `license`, `target`,
  `sha256: str | None`; property `is_single_file` = url ends neither with `/` nor contains `{`.
- `Manifest` (frozen dataclass): `version: int`, `sources: tuple[Source, ...]`; `get(id) -> Source` (`KeyError` if
  absent); `by_target(target) -> tuple[Source, ...]` in file order.
- `parse_manifest(data: Mapping) -> Manifest` and `load_manifest(path) -> Manifest` (YAML). `ManifestError(ValueError)`
  naming the offending source id (and key) is raised when: `version != 1`; `sources` missing, not a list or empty; a
  source lacks any of `id, desc, url, access, approx_bytes, license, target, sha256`; an id does not match
  `^[a-z0-9_]+$` or is duplicated; `access` or `target` is not a known value; `approx_bytes` is not a non-negative int
  (bool rejected); `sha256` is not `"TBD"`, null or 64 hex characters (or, for `target: stream` sources only, `"n/a"`:
  nothing is stored, so nothing is hashed); a `target: stream` source is not
  `herbie_byte_range` or vice versa; a multi-file source (directory or brace URL) has a non-TBD `sha256` (multi-file
  hashes are per file, in the fetch state, DATA-12). `"TBD"`, null and (stream) `"n/a"` all give `sha256 = None` ("not yet verified",
  not an error). A 64-hex value is stored lowercased. Unknown extra keys are ignored.
- The committed `data/manifest.yaml` parses without error (schema only; T1 changes sizes, hashes and entries).

### DATA-11 Resumable download
**Must (pipeline).** Module `data/fetch.py`.

- `Transport` protocol: `get(url: str, start: int = 0) -> RangeResponse`, where `RangeResponse` has `status: int`
  (206 = range honoured, 200 = whole body from byte 0) and `chunks: Iterable[bytes]`. The default transports (HTTP via
  `requests`, S3 via unsigned `boto3`) are not unit-tested.
- `download(url, dest, transport, expected_sha256=None, expected_size=None) -> DownloadResult` (`path`, `sha256`,
  `bytes_downloaded`, `skipped: bool`):
  1. If `dest` exists: with `expected_sha256` given, a matching hash returns `skipped=True` without calling the
     transport and a mismatch raises `ChecksumError` (the file is left untouched); without it, `dest` is trusted
     (only step 5 creates it) and returns `skipped=True` if `expected_size` is None or equals its size, else raises
     `DownloadError`.
  2. Partial data lives in `dest.name + ".part"` next to `dest` (parent dirs created). `offset` = its size (0 if
     absent). If `expected_size` is given and `offset > expected_size`, the part file is truncated to 0. If
     `offset == expected_size`, no request is made.
  3. Otherwise `transport.get(url, start=offset)` is called once. Status 206: chunks are appended. Status 200 (server
     ignored the range): the part file is truncated and rewritten from byte 0. Any other status raises
     `DownloadError` (part file kept). If chunk iteration raises, the bytes already received stay in the part file
     and the exception propagates (the next call resumes from there).
  4. If `expected_size` is given and the final part size differs, `DownloadError` is raised (part kept).
  5. The sha256 of the part file is computed. A mismatch with `expected_sha256` renames the part file to
     `dest.name + ".corrupt"` (replacing an old one) and raises `ChecksumError`. Otherwise the part file is atomically
     renamed to `dest` (`os.replace`).
  `bytes_downloaded` counts only bytes received in this call. `DownloadError` and `ChecksumError` subclass `OSError`
  (not `RuntimeError`).

### DATA-12 sha256 verification and fetch state
**Must (pipeline).** Module `data/fetch.py`.

- `sha256_file(path, chunk_size=1 << 20) -> str`: lowercase hex, equal to `hashlib.sha256(bytes).hexdigest()`.
- `FetchState(path)`: JSON record `{source_id: {relpath: {"sha256": str, "bytes": int}}}`. A missing file is an empty
  state; invalid JSON raises `ValueError`. `record(source_id, relpath, sha256, nbytes)` updates and persists
  immediately (write to a temporary file, then `os.replace`); `get(source_id, relpath) -> dict | None`. A new
  `FetchState` on the same path sees earlier records.
- Layout: files of source `s` live under `raw_root / s.id /` (single-file sources: `raw_root / s.id / basename(url)`);
  relpaths in the state are relative to `raw_root / s.id` with `/` separators.
- `verify_source(source, raw_root, state) -> VerifyReport` (`ok`, `mismatched`, `missing`, `unrecorded`: sorted lists
  of relpaths): every recorded file is re-hashed (`ok` / `mismatched`, or `missing` if absent); files present under the
  source folder but not recorded (ignoring `*.part` and `*.corrupt`) are `unrecorded`. For a single-file source with
  a manifest `sha256`, the manifest hash is the reference for that file (whether or not it is recorded), and that
  file is never reported as `unrecorded`.
- `fetch_source(source, raw_root, transport, state, lister=None) -> list[DownloadResult]`: a `target: stream` source
  raises `ValueError` (streamed sources are never stored raw). A single-file source downloads `source.url` to
  `raw_root / s.id / basename(url)` with `expected_sha256 = source.sha256` and records its sha256 and size in `state`.
  Multi-file sources use `lister` (DATA-13 planning) and record every file; `kaggle` access needs the Kaggle token and
  is exercised only in T1.

### DATA-13 Remote listing: brace expansion and skip-existing planning
**Must (pipeline).** Module `data/fetch.py`.

- `expand_braces(url) -> list[str]`: every `{a,b,...}` group (non-nested) is expanded; several groups give the cartesian
  product, left group varying slowest, alternatives in written order. No braces -> `[url]`. Unbalanced or nested braces
  raise `ValueError`. Example: `s3://sevir/data/vil/{2018,2019}/` -> the 2018 and 2019 prefixes.
- `plan_downloads(remote, prefix, dest_dir) -> list[PlannedFile]` (`key`, `size`, `dest`): `remote` is an iterable of
  `(key, size)`; every key must start with `prefix` (else `ValueError`); `dest = dest_dir / key[len(prefix):]` (leading
  `/` stripped). A file whose `dest` already exists with exactly `size` bytes is skipped; one with a different size is
  planned again. Input order kept. `planned_bytes(plan) -> int` sums the sizes.

### DATA-14 Disk budget check
**Must (pipeline).** Module `data/disk.py` (`outrun disk` calls it before every big job).

- `DiskBudget` (frozen dataclass): `hdd_cap_bytes: int = 375 * GB`, `ssd_cap_bytes: int = 225 * GB` (DESIGN 7.3 margin
  rule). `load_disk_budget(config_dir=None) -> DiskBudget` uses a `disk:` mapping (`hdd_cap_gb`, `ssd_cap_gb`) from
  `configs/disk.yaml` if that file exists, else the defaults.
- `planned_bytes(manifest) -> dict[str, int]`: sum of `approx_bytes` per target, keys exactly `"hdd"`, `"ssd"`,
  `"stream"` (0 when none).
- `check_budget(planned, budget, used=None) -> list[str]`: human-readable violations, empty when within budget. HDD
  check: `planned["hdd"] + used.get("hdd", 0) > hdd_cap_bytes`; SSD likewise (stream bytes never count). Each
  violation string names the disk (`"hdd"`/`"ssd"`) and the overshoot in GB.
- `used_bytes(path) -> int`: total size of regular files under `path`, recursively (symlinks not followed; missing path
  -> 0).
- `check_free_space(path, need_bytes, disk_usage=shutil.disk_usage, reserve_bytes=0) -> int`: returns free bytes; raises
  `DiskSpaceError` (subclass of `OSError`) when `free - reserve_bytes < need_bytes`.
- `cut_plan(hdd_bytes, cap_bytes, cuts: Sequence[tuple[str, int]]) -> list[str]`: applies the named cuts (DESIGN 7.3
  order: IR107 2018 H1, WSTS, TorNet 2016) in order, each saving its bytes, stopping as soon as
  `hdd_bytes - saved <= cap_bytes`; returns the names applied (empty if already within the cap). If all cuts are not
  enough, raises `DiskSpaceError`.

### DATA-15 Paths from the environment (`outrun.config`)
**Must (pipeline).** Module `config.py`.

- `REPO_ROOT: Path` = the repository root (the folder containing `pyproject.toml`, two levels above `src/outrun`).
- `Paths` (frozen dataclass): `root`, `raw`, `proc`, `configs`, `splits`, `scenarios`, `models`, `logs` (all absolute
  `Path`).
- `get_paths(env: Mapping[str, str] | None = None) -> Paths` (`os.environ` when None): `raw = OUTRUN_RAW` or
  `/Volumes/OutrunRaw`; `proc = OUTRUN_PROC` or `REPO_ROOT / "data" / "proc"`; `configs = OUTRUN_CONFIG_DIR` or
  `REPO_ROOT / "configs"`; `splits`, `scenarios`, `models`, `logs` = `REPO_ROOT / name`. Environment values are
  `~`-expanded and resolved to absolute paths; an empty string counts as unset. No directory is created.

### DATA-16 YAML config loading, seeds and device (`outrun.config`)
**Must (pipeline).** Module `config.py`.

- `load_config(name, config_dir=None) -> dict`: reads `<config_dir or get_paths().configs>/<name>.yaml` (`name` may
  include the `.yaml` suffix). Missing file raises `FileNotFoundError`; a top level that is not a mapping raises
  `ValueError`. Each call returns a fresh object (mutating it does not affect the next call). The committed
  `configs/domains.yaml`, `hazards.yaml` and `planning.yaml` load.
- `seed_everything(seed: int) -> None` seeds `random`, NumPy's global generator and, if importable, torch; two runs
  with the same seed draw identical numbers.
- `get_device(env=None) -> str`: `OUTRUN_DEVICE` (`"cpu"` or `"mps"`) is returned as given; any other non-empty value
  raises `ValueError`; unset -> `"mps"` if torch reports MPS available, else `"cpu"`.

### DATA-17 NWS forecast zones
**Must (pipeline, ice labels).** Module `data/nwszones.py`.

- `STATE_FIPS_TO_POSTAL`: dict for the 50 states + DC (e.g. 40 -> `"OK"`, 48 -> `"TX"`, 6 -> `"CA"`).
- `zone_id(state_fips, cz_fips) -> str` = `f"{postal}Z{cz_fips:03d}"` (40, 25 -> `"OKZ025"`); unknown state FIPS raises
  `ValueError`.
- `zones_from_frame(gdf) -> GeoDataFrame`: from the NWS public-zones attributes `STATE` (postal) and `ZONE` (3-digit
  string) adds `zone_id = STATE + "Z" + ZONE` (ZONE zero-padded to 3); duplicate zone ids are dissolved into one
  (multi)polygon.
- `zone_masks(gdf, x, y) -> dict[str, np.ndarray[bool]]`: per `zone_id`, the `(len(y), len(x))` mask of cell centres
  inside or on the boundary of its geometry.

### DATA-18 DEM derivatives: slope and TPI
**Must (pipeline, flood statics).** Module `data/dem.py`.

- `slope_deg(elev, cell_m) -> float32`: `degrees(arctan(hypot(dz/dx, dz/dy)))` with `np.gradient` (spacing `cell_m`,
  central differences inside, one-sided at the edges). NaN propagates.
- `tpi(elev, radius_cells) -> float32`: `elev - mean of the (2r+1) x (2r+1) window` centred on each cell, where the mean
  uses only in-bounds, finite cells. `radius_cells < 1` raises `ValueError`.

### DATA-19 Census population and facility tables
**Must (pipeline, shelters/evacuation inputs).** Modules `data/census.py`, `data/facilities.py`.

- `census.block_group_population(table, pop_col="P1_001N") -> pd.DataFrame`: columns `geoid` (12-character string,
  zero-padded) and `population` (int64). Missing columns raise `KeyError`; NaN or negative population, or duplicate
  GEOIDs, raise `ValueError`.
- `facilities.FACILITY_KINDS = ("shelter", "hospital", "fire_station", "school")` (the same kind names as
  `optimize.shelters.FacilityKind`, SHEL-02, so fire stations from this pipeline are recognised as fallback sites).
  `facilities.normalize_facilities(df, kind) -> pd.DataFrame` with columns `facility_id` (`f"{kind}-{i}"`, i = 0..n-1
  in output order), `kind`, `name` (string, `""` if missing), `lon`, `lat`, `capacity` (float, NaN when absent or
  unknown). Coordinates come from `lon`/`lat` or `longitude`/`latitude` columns (case-insensitive); rows with missing
  coordinates are dropped. Unknown `kind` raises `ValueError`; no coordinate columns raises `KeyError`.
