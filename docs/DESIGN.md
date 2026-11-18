# Outrun: Final Design Document

CMSC-5713 Final Project (Oklahoma Christian University). Student: Milton Adina. Builder: Claude Code (Milton makes the decisions).
Written Tue Nov 17 2026. Due Fri Dec 4 2026, 1:40 pm (17 days).

## Guiding rules

1. **Every item in the approved proposal ships end-to-end.** Only depth may shrink (section 7). Nothing in the proposal is optional.
2. **Baseline first, deep learning second, same interface.** Each hazard first ships an honest baseline layer behind its `HazardAdapter`. Its deep-learning model then replaces that layer. The app shows all six layers at all times.
3. **Start long jobs early.** All bulk downloads start within 48 hours. Router, app and tests run on *synthetic* layers by Nov 19-20 and do not depend on any real preprocessed data.
4. **Test data is used once.**
   - Splits are frozen only after the case-study events and the TorNet split rule are verified, on the same day and before any training (T7, Nov 20).
   - Thresholds, calibration and early stopping use validation data only.
   - All test metrics are computed in one task (T24, Dec 1) and logged with the git sha.
5. **Unverified facts are tagged [CANDIDATE].** This covers event dates, catalog membership, file sizes, product names and data publication. Each tag names its verification task and a fallback.
6. **Commit rules.** At least one real-dated commit every day from Nov 17 to Dec 3. Commit messages read `T<n>: ...`. No backdating, and no Claude co-author or attribution trailer.

---

## 1. Architecture overview

### 1.1 Diagram

```
                  RAW (external HDD /Volumes/OutrunRaw)              STREAMED (never stored raw)
 SEVIR VIL+IR107 h5 | TorNet nc | NDWS tfrecord | WSTS tif | MRMS QPE+FLASH grib2 | HRRR byte-range grib2
 AQS csv | HMS shp | Storm Events/SPC csv | ETOPO/3DEP tif | OSM pbf | HIFLD parquet | Census
        |                                                                         |
        v                                                                         v
  outrun.data.*  fetch (manifest.yaml: url, bytes, sha256, license) -> readers -> crops
        |  preprocess -> PROCESSED (SSD): uint8/fp16 .npy memmaps, zarr, parquet
        |  splits/*.json + splits/quarantine.json (frozen T7, hashed, tested)
        v
 +--------------------------- HAZARD LAYER: 6 plug-ins, one interface ---------------------------+
 | storm U-Net(VIL+IR) | tornado CNN | flood U-Net | wildfire U-Net | ice MIL-CNN | smoke CNN+MLP |
 |  each: baseline() / predict() / observed()  ->  RiskLayer (calibrated P in [0,1])            |
 +-----------------------------------------------|-----------------------------------------------+
                                                 v
        RiskStack (xarray/zarr): (hazard, slot[15 min x 12 = 3 h], y, x) on GRID = EPSG:5070, 1 km
        + HazardPolicy per hazard (block_at, speed_factor, exposure_weight, horizon rule)
                 |                                                  |
                 | EdgeRiskSampler (edge x hazard x slot)           | the same arrays -> PNG overlays
                 v                                                  v  ("what you see is what routes")
 OSM road graph (CSR) -> CostModel(profile) -> clock tau_e(t) >= free-flow; cost c_e(t) >= tau_e(t); inf if blocked
                 |
   +-------------+------------------+----------------------------+----------------------------+
   v             v                  v                            v                            v
 TD multi-label  Shelter            Whole-city evacuation        Emergency vehicles           Shelter-in-
 A* (safe route) min-cost flow      SA over (path, wave) + BPR;  reverse TD search +          place rule
                 (OR-Tools)         MSA-UE / shortest / safest   Hungarian dispatch
   +-------------+------------------+----------------------------+----------------------------+
                                    v
       scenarios/<id>/ : risk.zarr, overlays/*.png, results/*.geojson, metrics.json (precomputed)
                                    v
       FastAPI (/api/*, SSE)  <-->  vanilla MapLibre GL JS + local PMTiles (+ vendored deck.gl UMD)
                                    v
       outrun export-static -> docs/ -> GitHub Pages (recruiter demo, no backend)
```

### 1.2 Key abstraction: `HazardAdapter -> RiskLayer + HazardPolicy -> CostModel`

```python
# src/outrun/core/risk.py
class Hazard(Enum): STORM, TORNADO, FLOOD, WILDFIRE, ICE, SMOKE

@dataclass(frozen=True)
class RiskLayer:
    hazard: Hazard
    prob: xr.DataArray                 # (slot, y, x) float16, CALIBRATED P(hazardous) on GRID
    edge_overrides: dict[int, np.ndarray] | None   # per-edge P(slot) for point features (crossings, bridges)
    kind: Literal["forecast", "baseline", "observed"]
    issued: datetime
    valid_until: datetime              # end of the model's real forecast horizon
    source: str                        # "storm.unet_vil_ir@<sha>", "storm.pysteps", "observed.sevir"
    held_out: bool                     # case/test data never seen in training (provenance badge in UI)

@dataclass(frozen=True)
class HazardPolicy:                    # src/outrun/core/policy.py, values in configs/hazards.yaml
    block_at: dict[Profile, float | None]      # P >= block_at -> edge impassable for that profile
    speed_factor: Callable[[float], float]     # in [1/3, 1]; clock slows (ice, heavy storm, flood approach)
    exposure_weight: dict[Profile, float]      # w_h^p >= 0 for the cost multiplier
    beyond_horizon: Literal["hold", "decay", "drop"]; decay_tau_min: float | None

class HazardAdapter(Protocol):         # src/outrun/hazards/<h>/adapter.py
    hazard: Hazard; policy: HazardPolicy
    def baseline(self, scn: Scenario) -> RiskLayer: ...
    def predict(self, scn: Scenario) -> RiskLayer: ...
    def observed(self, scn: Scenario) -> RiskLayer | None: ...   # ground truth for routing evaluation
```

**Properties this gives:**

- **The router never changes when a hazard is added.** It sees only `RiskStack` + `HazardPolicy`.
- **One switch selects the risk source:** `--risk-source model|baseline|observed|none|current-only`. The routing evaluation is built on it: plan on a forecast, score on observed data.
- **What you see is what routes.** The PNG overlays and the edge costs are rendered from the same `RiskStack` arrays. Test `RISK-07` asserts that overlay pixel values equal the sampled values.
- **Point hazards that a 1 km grid would blur** go through `edge_overrides`. These are low-water crossings (flood) and bridges (ice).

**Beyond-horizon rule.** The routing window is t0 to t0+3 h (12 slots of 15 min). Every layer must say what it does after its `valid_until`:

| Hazard | Native horizon | Beyond horizon | Reason |
|---|---|---|---|
| Storm | 60 min | decay, P(t) = P(t_h) * exp(-(t - t_h)/30 min); 0 after 120 min | Extrapolated cells lose skill fast; holding them would over-penalise. |
| Tornado | 30 min | drop to 0, but `blocked` cells keep blocking for 15 min after the swath passes (debris) | This is a detection projection, not a forecast. Damage lingers. |
| Flood | 6 h | hold (covers the window) | |
| Wildfire | next day | hold (daily step, linearly grown hourly between day 0 and day 1 masks) | |
| Ice | 3 h steps | hold the last step | Ice persists. |
| Smoke | +6 h | hold | |

Test `HORIZON-01..03` checks decay, drop and hold on synthetic layers.

### 1.3 Proposal coverage matrix (the graded 40 Functionality points)

This table also lives in `specs/00-overview.md`. `tests/unit/test_traceability.py::test_proposal_matrix` parses it and asserts two things: every proposal phrase has at least one requirement ID, and every ID has a test marked with `@pytest.mark.req`.

| Proposal phrase | Module | Tasks | Req IDs | Shipping evidence |
|---|---|---|---|---|
| deep learning | `hazards/*/model.py` (6 PyTorch nets) | T8-T18 | STORM-04, TOR-04, FLOOD-04, FIRE-04, ICE-04, SMOKE-04 | 6 checkpoints + test metrics vs baselines (T24) |
| real weather data | HRRR (ice, smoke, tornado-motion fallback), MRMS QPE (flood), GRIDMET inside NDWS/WSTS (fire) | T3, T13-T18 | DATA-03..07 | manifest + decoded fixtures |
| real radar data | SEVIR VIL (NEXRAD mosaic), TorNet (NEXRAD Level-II chips), MRMS (radar QPE) | T8-T12, T14 | DATA-01, 02, 04 | |
| real **satellite** data | (1) SEVIR GOES-16 **IR107** as a storm U-Net input channel, with a VIL-only ablation; (2) MODIS fire masks + VIIRS NDVI inside NDWS, and VIIRS bands in WSTS (fire); (3) NOAA **HMS** satellite smoke-density polygons as a smoke-model input, with an ablation without HMS | T8, T9, T13, T18 | STORM-02, FIRE-02, SMOKE-02 | ablation tables; named in report and slides |
| forecast severe storms | `hazards/storm` 0-60 min nowcast | T8, T9 | STORM-* | |
| tornadoes | `hazards/tornado` detector + projected swath | T10, T12 | TOR-* | |
| flash floods | `hazards/flood` 0-6 h cells -> road crossings | T14, T17 | FLOOD-* | |
| wildfire spread | `hazards/wildfire` next-day spread | T13, T20 | FIRE-* | |
| icy roads | `hazards/ice` MIL-CNN + bridge prior | T15, T16 | ICE-* | |
| smoke | `hazards/smoke` PM2.5 +6 h | T18 | SMOKE-* | |
| A* search | `routing/astar.py` (own time-dependent multi-label A*) | T5 | ASTAR-* | optimality tests vs Dijkstra and brute force |
| optimization | `optimize/shelters.py` (min-cost flow), `optimize/evacuation.py` (simulated annealing) | T11, T19 | SHEL-*, EVAC-* | |
| real road maps | `roads/graph.py` (OSM via OSMnx) | T4 | ROAD-* | |
| safe evacuation routes | `routing/planner.py` | T5, T21 | PLAN-* | |
| shelter assignments | `optimize/shelters.py` | T11 | SHEL-* | |
| whole-city evacuations | `optimize/evacuation.py` | T19 | EVAC-* | |
| emergency vehicle routes | `routing/emergency.py` | T21 | EMS-* | |
| interactive map app | `app/` | T6, T23 | APP-* | live demo, Pages build, backup video |
| "any US city" (answer to the professor) | `outrun route --place`, app city box | T5, T23 | ROAD-06 | acceptance test on Tulsa OK + one non-OK city |

---

## 2. Repo layout, modules, tech stack

### 2.1 Layout

```
outrun/
  README.md CLAUDE.md TASKS.md NOTES.md REPORT.md DEMO.md LICENSE(MIT) DATA_LICENSES.md Makefile
  pyproject.toml  .github/workflows/tests.yml  .pre-commit-config.yaml (ruff)
  .gitignore: data/ models/ logs/ scenarios/**/*.zarr *.mp4
  configs/  hazards.yaml (policies, weights)  domains.yaml (bboxes, GRID)  planning.yaml (shelter rate, occupancy)
  data/manifest.yaml                      # committed: every source, url, bytes, sha256, license, target disk
  splits/   sevir.json tornet.json flood.json ice.json smoke.json ndws.json wsts.json quarantine.json hashes.json
  specs/    00-overview 01-data 02-grid-risk 03-roads 04-astar 05-shelters 06-evacuation 07-emergency
            08-storm 09-tornado 10-flood 11-wildfire 12-ice 13-smoke 14-app 15-evaluation 16-acceptance
  src/outrun/  (modules below)
  tests/ unit/ acceptance/ e2e/ fixtures/ (30-node graphs, 32x32 risk grids, 3 fake SEVIR events, 1 GRIB msg)
  scenarios/ reports/{metrics.md, metrics/*.json, figures/} slides/ docs/ (Pages build)
```

### 2.2 Modules (one line each)

| Module | Responsibility |
|---|---|
| `cli.py`, `__main__.py` | `outrun fetch|prep|train|eval|scenario|route|serve|export-static|disk` |
| `config.py` | Paths (RAW on HDD, PROC on SSD), seeds, device (mps/cpu), loads `configs/*.yaml` |
| `core/grid.py` | GRID (EPSG:5070, 1 km), bboxes, reprojection of SEVIR LCC, HRRR LCC, MRMS lat/lon and polar data |
| `core/risk.py` | `RiskLayer`, `RiskStack`, zarr IO, overlay PNG renderer (same arrays the router uses) |
| `core/policy.py` | `HazardPolicy`, beyond-horizon hold/decay/drop |
| `core/scenario.py` | `Scenario` (id, t0, bbox, city, case metadata, held-out flags) |
| `core/synthetic.py` | Moving Gaussian hazards for tests and the skeleton |
| `core/calibrate.py` | Isotonic/temperature calibration, reliability bins, Brier, ECE |
| `data/fetch.py` | Resumable S3 (boto3 unsigned) / Zenodo / Kaggle / HTTP downloads from the manifest, sha256, `aria2c -c -x8` |
| `data/sevir.py` | CATALOG filter, VIL+IR107 h5 to 2 km uint8 memmaps, per-event projection metadata |
| `data/tornet.py` | catalog.csv, NetCDF chips to fp16 memmaps, official split per the verified rule |
| `data/ndws.py` | TFRecord to npy via the pure-Python `tfrecord` package |
| `data/wsts.py` | WSTS GeoTIFF reader + documented 12-channel NDWS mapper + mapper statistics |
| `data/mrms.py` | QPE 01H Pass2 + FLASH products (cfgrib), crop, accumulations |
| `data/hrrr.py` | Herbie `.idx` byte-range field fetch, in-memory crop, f01 rule, MASSDEN unit fix |
| `data/aqs.py`, `hms.py`, `stormevents.py`, `spc.py`, `nwszones.py` | Tables and rasterization for labels and inputs |
| `data/dem.py` | ETOPO 2022 30" (training statics), 3DEP 10 m (OKC crossings), slope/TPI |
| `data/census.py`, `facilities.py` | 2020 block-group population; NSS shelters, hospitals, fire/EMS, schools (source.coop HIFLD Parquet, with OSM fallback) |
| `hazards/base.py` | Adapter protocol, registry, shared MPS training loop (timed epoch, checkpoint/resume, early stop on val) |
| `hazards/<h>/{dataset,model,baselines,adapter}.py` | One package per hazard (storm, tornado, flood, wildfire, ice, smoke) |
| `roads/graph.py` | OSMnx build/cache (bbox or place), speeds, capacities, CSR `.npz` export |
| `roads/features.py` | Bridges, low-water-crossing proxy (OSM ford + road-by-waterway without bridge), facility snapping |
| `routing/cost.py` | `CostModel` profiles, `EdgeRiskSampler`, clock time vs generalized cost |
| `routing/astar.py` | Own time-dependent multi-label A*, multi-goal heuristic, optional ALT; Dijkstra oracle |
| `routing/planner.py` | Safe-route API, `avoid-current-only` baseline, shelter-in-place rule |
| `routing/emergency.py` | Reverse TD search, Hungarian dispatch, discover-and-reroute baseline, hospital leg |
| `optimize/shelters.py` | OR-Tools min-cost flow + nearest/greedy baselines |
| `optimize/evacuation.py` | Zones, k diverse paths, wave-BPR evaluator, simulated annealing, MSA user equilibrium |
| `eval/metrics.py`, `eval/bootstrap.py` | CSI/FSS/AUC/AUPRC/Brier/ECE/RMSE/exposure; day-block bootstrap CIs |
| `eval/truth.py` | Observed-hazard `RiskLayer`s (SEVIR future VIL, SPC tracks, Storm Events, VIIRS/WSTS masks, zones, AQS) |
| `eval/routing_eval.py`, `eval/casestudy.py`, `eval/report.py` | Planners (a)/(a')/(b)/(c)/(d), case studies, `reports/metrics.md` + figures |
| `app/server.py`, `app/sse.py`, `app/static/` | FastAPI API + SSE; `index.html`, `app.js`, `style.css`, `vendor/` (MapLibre, pmtiles.js, deck.gl UMD) |

### 2.3 Tech stack

| Choice | Why this over the alternative |
|---|---|
| Python 3.12, pip + `pyproject.toml` (lock via `pip-compile`) | Same toolchain as phishguard. Torch wheels are stable on 3.12, and uv/poetry add nothing graded. |
| PyTorch 2.x on MPS, plain training loop | Native M5 Max GPU, and a short loop is easy for Milton to explain. JAX-Metal is experimental, TF-macos is fragile, and Lightning hides the loop. |
| 2D U-Nets/CNNs with time as channels | Every op is MPS-native and U-Net is the standard SEVIR baseline. ConvLSTM/3D convs are slow or patchy on MPS, and transformers (Earthformer) need more training time than we have. |
| pySTEPS (Lucas-Kanade + semi-Lagrangian) storm baseline, OpenCV Farneback as fallback | pySTEPS is the credible operational-style extrapolation. Persistence alone is too weak to prove anything. |
| OSMnx (build only) + own A* on numpy CSR (numba if needed); networkx as test oracle | Works for any US city. A* is the graded algorithm and needs time-dependent risk costs. OSRM/GraphHopper/igraph cannot take time-varying hazard costs. |
| OR-Tools `SimpleMinCostFlow` | Exact capacitated assignment in milliseconds. Hungarian is 1:1 only, and a MILP is overkill. |
| `scipy.optimize.linear_sum_assignment` | Exact Hungarian method for multi-incident dispatch. |
| Own simulated annealing + numpy BPR | A transparent custom objective with incremental updates. A GA needs more tuning, and MATSim/SUMO setup alone takes days. |
| Data formats: xarray + zarr (risk stacks, HRRR crops), uint8/fp16 `.npy` memmaps (training tensors), Parquet/GeoParquet (tables, vectors); raw HDF5/NetCDF/GRIB2/TFRecord/GeoTIFF read once only | Zero-copy random access for DataLoaders and labelled time dims. HDF5 reads from many workers are slow and lock-prone, GeoTIFF has no time dim, and CSV is huge. |
| Herbie + cfgrib/eccodes (pygrib fallback) | `.idx` byte-range subsetting keeps HRRR to the needed messages. |
| `tfrecord` pure-Python reader (fallback: `tensorflow-cpu` in a throwaway venv, used once) | Avoids TensorFlow in the main env. |
| rasterio, pyproj, geopandas, shapely 2 | Standard geospatial IO. |
| FastAPI + uvicorn + SSE | Typed, async, proven in phishguard. SSE streams the SA objective curve. Flask lacks typing, and Streamlit cannot do custom map interaction. |
| Vanilla MapLibre GL JS 4 (vendored, no build step) + Protomaps PMTiles basemap + optional vendored deck.gl UMD (TripsLayer only) | WebGL overlays and animation work fully offline with zero build chain. React/Vite would add a toolchain and CI jobs without adding marks. Leaflet/folium cannot animate flows, and Mapbox needs a token. |
| pytest + `@pytest.mark.req` + traceability/proposal-matrix tests; Playwright (Python) e2e; GitHub Actions on CPU torch | Phishguard conventions. CI uses synthetic fixtures only. The e2e offline test runs locally and in a separate CI job. |

---

## 3. Hazards

**Shared definitions**

- **Domains.** OK_BBOX is lon -103.1..-94.4, lat 33.6..37.1, plus a 50 km buffer. The demo metro OKC_BBOX is lon -98.10..-97.10, lat 35.10..35.80 (El Reno, Yukon, OKC, Edmond, Moore, Norman, Midwest City). GRID is EPSG:5070 at 1 km.
- **Training region.** Storm, tornado, flood, wildfire and smoke train on CONUS data. Ice trains on the US ice belt box (32-42N, 103-88W), a CONUS subset chosen for HRRR volume; the report states this.
- **Quarantine rule (replaces "forcing case days into test").**
  - `splits/quarantine.json` lists each case-study date ±3 days. Those dates are removed from **train and val** of every model.
  - Official test splits are **left unmodified**, so the published SEVIR/TorNet/NDWS comparisons stay valid.
  - Case days are reported as a separate "case hold-out" set.
  - Test `SPLIT-07` enforces this in every dataset builder.
- **Split-overlap tests.** `SPLIT-01..06` (one per hazard) assert no day/event/zone-window overlap between train, val and test, plus a gap day at each time boundary. Test IDs are hashed into `splits/hashes.json`.
- **Calibration.** Every probabilistic layer is calibrated on val and reports Brier score and a reliability diagram on test.
- **Seeds.** One seed per main model is required. Three seeds or ensembles are a lever up.

### 3.1 Severe storms: VIL + IR107 nowcast (SEVIR)

- **Data.**
  - `CATALOG.csv` (34 MB).
  - VIL STORMEVENTS files for 2018 and 2019 only: about 65 GB, [CANDIDATE size; T1 runs `aws s3 ls --no-sign-request --recursive --summarize s3://sevir/data/vil/2018/` and `/2019/`].
  - IR107 files for the same years: [CANDIDATE] about 30-40 GB.
  - **IR107 rule:** if the 2018-19 IR107 total is over 60 GB, drop the 2018 Jan-Jun file(s) until the total is 60 GB or less, keeping all 2019 files (val/test). The VIL-only ablation then trains on the identical event subset.
  - T1 also counts the 2018-19 event ids that have both `vil` and `ir107` in CATALOG [CANDIDATE: expected the large majority]. Events missing IR are dropped from both arms.
  - 2017 and RANDOMEVENTS are skipped. Files are taken whole and go to the HDD.
- **Split.**
  - Official cut: test = 2019-06-01 and later (untouched).
  - Train = 2018-01 to 2019-03-31. Val = 2019-04-01 to 2019-05-31. Both minus quarantine, with a 1-day gap at each boundary.
- **Preprocessing.**
  - Common grid is 2 km, 192x192. VIL is 384 at 1 km, **average-pooled** 2x2 (not max-pooled, which would inflate CSI). IR107 is native 192 at 2 km.
  - Both are stored uint8. IR brightness temperature 180-320 K is mapped linearly to 0-255 (0.55 K steps).
  - Memmap shape is (N, 49, 2, 192, 192), about 3.6 MB/event: roughly 13k events is about 47 GB SSD [estimate].
  - Per-event projection metadata (`proj`, corner lat/lons) is kept for reprojection to GRID.
  - Windows: 13 input frames (t-60..t) give 12 outputs (t+5..t+60). Offsets {0, 12, 24}, so 3 windows per event.
- **Model.**
  - U-Net, 4 levels, base 32 (about 8M parameters).
  - Input: 26 channels (13 VIL + 13 IR). The ablation uses 13 VIL only.
  - Heads: (a) 12-frame VIL regression with SEVIR-style intensity-weighted MSE+MAE; (b) 12x3 exceedance logits for VIL ≥ 74 / 133 / 181 (BCE).
  - Calibration: isotonic per lead on val.
- **Training budget.** About 40k windows, batch 16, fp32 (fp16 autocast only if the timed epoch shows no NaN), 20 epochs, about 4-6 h overnight [estimate; T8 times one epoch first]. The ablation runs the next night.
- **Evaluation.**
  - Metrics: CSI at {16, 74, 133, 160, 181, 219} per lead, FSS (5/15 km) at 133, MSE, Brier/reliability for P(VIL ≥ 133).
  - Baselines: persistence and pySTEPS.
  - Target (not a requirement): beat pySTEPS on CSI-74/133 averaged over 60 min. Per-lead curves are shown because U-Nets often lose to advection at high thresholds and late leads.
  - The VIL+IR vs VIL-only delta is the satellite-value number.
- **Pitfalls to check.**
  - Blur inflates CSI-16 and kills CSI-181/219: report all thresholds plus sample frames.
  - Clear-air windows: also report on windows with max VIL ≥ 74.
  - Same-day leakage: split by calendar day.
  - Edge inflow: also score the inner 128x128 crop.
  - Grid optimism: also report CSI at 1 km after bilinear upsampling against native 1 km truth.
  - Projection: unit test that Storm Events hail points for one event overlay high VIL.
  - The Oklahoma subset is reported separately.
- **Risk layer.**
  - r_storm = calibrated P(VIL ≥ 133) per 5-min frame, max over the frames in each 15-min slot, reprojected to GRID.
  - Policy: `block_at=None`, `speed_factor = 1 - 0.3p`, civilian w = 2, beyond horizon = decay.
  - The pySTEPS motion field is exported for the tornado swath.
  - `observed()` = true future VIL ≥ 133.
  - Rule baseline layer (skeleton) = persistence of the last-frame VIL ≥ 133 mask.

### 3.2 Tornadoes: radar detection (TorNet) + projected swath

- **Data.**
  - TorNet `catalog.csv` plus the 2019, 2015 and 2016 yearly tarballs (about 51 GB per the brief) [CANDIDATE exact sizes; T1 lists the Zenodo record files].
  - Download order: 2019 first. More years are a lever.
  - Tarballs go to the HDD. They are extracted one year at a time to the SSD, converted, and the extracted copy is deleted.
  - Also the SPC tornado CSV and Storm Events.
- **Split.**
  - The official split rule is [CANDIDATE: Julian-day based per the brief]. **T1 reads the split code in `mit-ll/tornet` and catalog.csv** and records the exact rule in NOTES.md. Fallback if it differs: implement exactly what the code does.
  - The rule is applied to the 3 downloaded years. Results are labelled "official split restricted to 2015/16/19".
  - Val = 10% of train Julian days (grouped). Quarantined dates are removed from train/val.
- **Preprocessing.**
  - 6 variables (DBZ, VEL, KDP, RHOHV, ZDR, WIDTH) x 2 tilts, last frame = 12 channels, plus last-minus-first difference = 12 more.
  - Fill values become per-variable constants, with a missing mask channel added.
  - CoordConv range/azimuth channels are added.
  - Chip shape [CANDIDATE 120 az x 240 range; T10 confirms].
  - Normalized with train statistics, stored fp16. About 30 GB [estimate].
- **Model.** PyTorch port of the TorNet baseline CNN (VGG blocks 32-256, CoordConv, global max-pool, MLP head), about 1.5M parameters, focal loss (gamma = 2). Isotonic calibration on val, because the catalog base rate is not the real one.
- **Training budget.** Batch 128, 15 epochs, about 3-4 h overnight [estimate; timed epoch]. The range-ablation retrain runs the following night.
- **Evaluation.**
  - Headline: **ROC-AUC tornadic vs warned-null negatives only** (the honest hard case).
  - Also: overall ROC-AUC/AUPRC, AUC per range bin (0-50, 50-100, 100+ km), Brier/reliability, CSI/POD/FAR at the val threshold.
  - Baselines: (a) HistGradientBoosting on hand features: max DBZ, max gate-to-gate VEL difference (shear proxy), min RHOHV where DBZ > 30 (debris proxy), max ZDR/KDP/WIDTH, range; (b) published CNN AUC 0.874 as a reference only (different subset); (c) rule: ΔV ≥ 40 m/s at lowest tilt.
- **Pitfalls to check.**
  - Range shortcut: range-bin AUC, an ablation without the range coordinate channel, and a range-only logistic model must be weak.
  - Missing-data fill rate per class.
  - Imbalance (6.8% positive).
  - EF-stratified recall.
  - Oklahoma subset reported.
- **Risk layer.**
  - For the case day, the detector runs on **TorNet chips for that day near KTLX** (catalog has lat/lon/time). No raw Level-II pipeline is needed. Live KTLX Level-II via Py-ART is a lever up.
  - Swath: P at the chip centre, advected 30 min along a motion vector. The kernel is an anisotropic Gaussian, sigma 4 km across track, widening to 8 km at 30 min.
  - **Motion fallback order:** (1) pySTEPS motion from SEVIR VIL at that location if a SEVIR event covers it; (2) HRRR 2019 analysis 700/500 hPa mean wind for that hour, byte-range fetch of 4 messages, taken at 75% speed and 30 degrees to the right (a documented right-mover approximation); (3) 15 m/s from 240 degrees. The SPC track is never used as input, because it is the truth used for scoring.
  - Policy: civilian w = 20, `block_at` = 0.5 for slots within 20 min, beyond horizon = drop (blocked cells persist 15 min).
  - `observed()` = SPC tracks buffered by reported width.
  - Rule baseline = the ΔV rule.

### 3.3 Flash floods: MRMS QPE + FLASH + terrain to road crossings

- **Data.**
  - MRMS `MultiSensor_QPE_01H_Pass2`, Apr-Sep 2021-2025 CONUS: 5 x 183 d x 24 = 21,960 files x 0.45 MB, about 10 GB.
  - MRMS FLASH QPE-to-FFG ratio (1 h, 3 h) and QPE ARI products [CANDIDATE product names, e.g. `FLASH_QPE_FFG01H`/`FLASH_QPE_ARI01H`; **T3 lists `s3://noaa-mrms-pds/CONUS/`** and measures file size]. Fetched only in event windows: ±6 h around CONUS flash-flood report hours plus 20% random rainy hours, capped at 20 GB.
  - Fallback if FLASH is absent in the bucket: the ARI-style rule (3 h ≥ 50 mm or 6 h ≥ 75 mm) becomes the operational-style baseline. This is documented.
  - Training statics: ETOPO 2022 30" CONUS crop [CANDIDATE < 1 GB] resampled to MRMS 0.01 degree for elevation, slope and TPI. The 1-arcsec CONUS 3DEP mosaic is not used.
  - OKC road downscaling: 6 USGS 3DEP 10 m tiles (about 2.6 GB).
  - Labels: NCEI Storm Events "Flash Flood" 2021-2025. [CANDIDATE: 2025 complete; T1 checks the NCEI file list.]
- **Split.** Train 2021-2023, val 2024, test 2025. Fallback if 2025 is incomplete: train 2021-22, val 2023, test 2024.
- **Preprocessing.**
  - Native MRMS 0.01 degree grid, 64x64 patches.
  - 11 channels: QPE 1/3/6/24 h; FFG ratio 1 h/3 h; max ARI; elevation, slope, TPI; a valid mask.
  - Labels: positive = within 5 km of the report begin-to-end segment during [begin - 1 h, end]. A **5-15 km ring is masked as uncertain** (ignored in loss and metrics). Target = flash flood within the next 6 h.
  - **Sampling:** 1/3 positive patches, 1/3 **hard negatives** (QPE 1 h ≥ 25 mm with no report within 25 km ±3 h), 1/3 random wet hours. Dry hours are excluded so "rain vs no rain" is not a shortcut.
  - Windows with missing hours are skipped.
- **Model.** U-Net, 3 levels, base 32 (about 2M parameters), focal loss with the ring masked, isotonic calibration on val.
- **Training budget.** About 40-60k patches, about 1.5-2 h [estimate].
- **Evaluation.**
  - Metrics: AUPRC, Brier/reliability, neighbourhood CSI (10 km, ±1 h), event POD (fraction of reports covered) and false-alarm area.
  - **Baselines:** (1) operational **FLASH QPE/FFG ratio ≥ 1.0**; (2) QPE ARI ≥ 10-yr; (3) logistic regression on the same features.
  - Ablation: model without FLASH inputs, because FFG is both input and baseline.
- **Pitfalls to check.**
  - Population/reporting bias: population is not an input. Report POD and AUPRC by urban/rural (census density terciles), plus a permutation check with a population channel added.
  - Report-time imprecision (±1 h tolerance).
  - Neighbouring reporting cultures.
  - Hard-negative AUC reported separately.
- **Risk layer.**
  - Cell P is reprojected to GRID.
  - `edge_overrides` on low-water-crossing edges (OSM `ford=yes`, or road crossing `waterway=stream|river` without `bridge=*`) get p' = min(1, 1.5p). Crossings in local depressions (3DEP elevation more than 2 m below the 300 m-radius median) get min(1, 2p). Bridges get 0.
  - Policy: civilian `block_at` = 0.4 at crossings ("Turn Around, Don't Drown"), EV `block_at` = 0.7, `speed_factor = 1 - 0.4p`, beyond horizon = hold.
  - `observed()` = Storm Events segments (5 km).
  - Rule baseline = FFG ratio ≥ 1 gives p = 1.

### 3.4 Wildfire spread: NDWS training, WSTS 2021 out-of-time test

- **Data.**
  - **Train:** Kaggle NDWS (`fantineh/next-day-wildfire-spread`), 3.96 GB, official train/eval/test files. Needs Milton's Kaggle token (Decision 1).
  - **Georeferenced eval + demo:** WildfireSpreadTS (Zenodo 8006177). [CANDIDATE layout: likely one about 48 GB archive; T1 lists the record files.] The whole archive is budgeted. Only 2021 is extracted.
  - WSTS 2021 is out-of-time relative to NDWS (2012-2020). WSTS 2018-20 may overlap NDWS samples and is not used for evaluation.
- **Preprocessing.**
  - NDWS goes to `ndws_{split}.npy` (N, 12, 64, 64) plus labels, with -1 = uncertain masked.
  - Clip/normalize with NDWS train percentiles.
  - Flips/rot90 **with wind direction rotated consistently** (test `FIRE-03`).
  - **WSTS channel mapper** (`data/wsts.py`) resamples 375 m to 1 km and maps channels to NDWS's 12:
    - Direct: elevation, wind direction/speed, Tmin/Tmax, specific humidity, precipitation, drought index, NDVI, ERC, previous active-fire mask (VIIRS).
    - Population: from census density, since WSTS lacks it.
    - Validated by per-channel KS distance and quantile plots vs NDWS train, logged in NOTES.md. Channels with large shift are listed as a caveat.
- **Model.** U-Net, 3 levels, base 32 (about 2M parameters), BCE (pos_weight about 3) + Dice, uncertain pixels ignored. Temperature calibration on val.
- **Training budget.** About 1 h.
- **Evaluation.**
  - Metrics: PR-AUC on NDWS test (published best 0.284 is the reference) and **PR-AUC on new-burn pixels** (label 1 where the previous mask is 0), F1, Brier/reliability.
  - **WSTS 2021 out-of-time PR-AUC** (all and new-burn).
  - Baselines: persistence, persistence dilated 1 px downwind, per-pixel logistic regression.
- **Pitfalls to check.**
  - Previous-mask copying (new-burn metric).
  - Shrinking fires.
  - NDWS lacks lat/lon, so nearby-sample leakage is possible inside NDWS splits. WSTS 2021 gives the clean test.
  - VIIRS detection gaps.
  - Per-fire macro average on WSTS.
- **Risk layer.**
  - Demo fire = the largest WSTS 2021 fire near a populated place [CANDIDATE; chosen in T13 from the WSTS fire list; prefer a Southern Plains fire if one exists, else a western fire]. Routing uses that town's OSM roads (shows "any US city").
  - Day 0 observed, day 1 predicted, linear hourly growth.
  - Policy: civilian `block_at` = 0.5, EV 0.7, w = 20, beyond horizon = hold.
  - `observed()` = next-day WSTS VIIRS mask.
  - **Extra, not evaluation evidence:** the Oklahoma replay for [CANDIDATE] Mar 14 2025 central-OK fires, verified from HMS fire points in T7. Inputs: HRRR weather, 3DEP, census, HMS ignition, NDWS train-mean NDVI/ERC. Labelled in the UI as "replay with climatological inputs, domain-shifted", and compared only qualitatively to next-day HMS points.
  - Rule baseline = downwind-dilated persistence.

### 3.5 Icy roads: HRRR f01 + weak zone labels, MIL-CNN

- **Data.**
  - HRRR `wrfsfc` via Herbie byte-range, **f01 only**. Precipitation and accumulated fields are zero or degenerate at f00.
  - 3-hourly cycles (00, 03, ..., 21Z).
  - 9 fields: TMP:surface, TMP:2 m, DPT:2 m, CFRZR, CICEP, CSNOW, CRAIN, PRATE, FRZR.
  - Window: Nov 15-Mar 15 of winters 2020-21 (starting 2020-12-03, after the HRRR v4 switch on Dec 2 2020) through 2024-25. That is 4 x 121 + 103 = about 587 days. All data is HRRR v4, so there is no version mixing; the report states that v3 was avoided by design.
  - **Volume:** 587 days x 8 cycles x 1 lead x 9 fields = 42,264 messages x m MB/message. At m = 1.0 MB [CANDIDATE; T3 measures], about 42 GB is streamed, cropped in memory and discarded.
  - **Cap 80 GB streamed:** if the measured mean m is over 1.9 MB, switch to 6-hourly cycles.
  - Kept: fp16 zarr on the ice-belt box (32-42N, 103-88W) at 3 km, about 370x460 cells, about 0.34 MB/field/time, about 14 GB SSD.
  - Labels: Storm Events "Ice Storm" + "Sleet" + "Freezing Fog", plus "Winter Storm"/"Winter Weather" rows whose narrative matches `freezing rain|freezing drizzle|ice accum|glaze|sleet`, by NWS forecast zone and time. NWS zone shapefile (about 20-50 MB). Winter rows without an ice keyword are **excluded**, not treated as negative.
- **Split (forward in time).** Train 2020-21..2022-23, val 2023-24, test 2024-25 (latest complete winter). [CANDIDATE: Storm Events Jan-Mar 2026 published → 2025-26 added as an extra test winter if so, checked in T1.]
- **Preprocessing.**
  - One bag = (zone, 6-h window).
  - Input: the 3 km cells of a 64x64 crop around the zone, 2 times (t, t-3 h) x 9 fields + elevation + zone-mask = 20 channels.
  - Hard negatives: windows with T2m < 3 C and no event.
- **Model.**
  - U-Net encoder-decoder (about 2M parameters) giving per-cell logits.
  - **MIL head:** zone logit = LogSumExp pooling over in-zone cells, BCE on zone labels.
  - At inference this gives a cell-level icy-road probability map from zone-level weak labels. Isotonic calibration at zone level.
- **Training budget.** About 1 h.
- **Evaluation.**
  - Metrics: zone-window AUPRC, ROC-AUC, Brier skill vs climatology, reliability, CSI at the val threshold.
  - **Baselines:** (1) physics rule: (CFRZR or CICEP = 1, or PRATE > 0 with TMP:surface ≤ 0 C) and TMP:surface ≤ 0 C; (2) zone-month climatology; (3) logistic regression on zone-aggregated features.
- **Pitfalls to check.**
  - Labels describe weather, not road-surface ice. This is stated in the model card.
  - Loose timing.
  - Urban/rural reporting bias (stratify).
  - Seasonality shortcut: no date or lat/lon inputs.
  - "Cold rainy day" shortcut: compare against rule (1) on the same windows.
- **Risk layer.**
  - Cell P reprojected to GRID.
  - `edge_overrides`: `bridge=yes` gets p' = min(1, 1.5p). This is a documented physical prior (bridges freeze first) and is ablated.
  - Policy: `speed_factor = 1 - 0.5p` (at most 2x slower), civilian w = 3, `block_at` None, beyond horizon = hold.
  - `observed()` = labelled zones.
  - Rule baseline = physics rule.

### 3.6 Smoke: PM2.5 +6 h (AQS + HRRR-Smoke + HMS)

- **Data.**
  - EPA AQS `hourly_88101_{2022,2023,2024}.zip` (about 210 MB). [CANDIDATE: 2025 file posted, checked in T1; used as an extra test year if complete.]
  - HRRR-Smoke via byte-range: **6-hourly cycles (00/06/12/18Z), f01 and f06 only**, 4 fields: MASSDEN 8 m, HPBL, UGRD 10 m, VGRD 10 m.
  - **Volume:** 1,096 days x 4 cycles x 2 leads x 4 fields = 35,072 messages x m. At m = 1.0 MB [CANDIDATE; T3], about 35 GB streamed. Cap 60 GB; if exceeded, drop f01.
  - Kept: 9x9 3 km patches at about 1,000 monitors + a 9 km CONUS grid, about 2 GB.
  - MASSDEN unit change on 2021-12-21 (µg/m³ to kg/m³). This is before our window but asserted in code (`SMOKE-03`).
  - NOAA HMS daily smoke polygons 2022-2024 (about 0.5 GB), rasterized to density 0-3. This is the **satellite input**.
- **Split.**
  - Time: train 2022-01..2023-06 (includes the June 2023 Canadian smoke), val 2023-07..2023-12, test 2024.
  - **Plus 20% of monitors held out spatially**, grouped by 200 km blocks.
- **Preprocessing.**
  - One sample per (station, 6-hourly issue time). Inputs: HRRR patch (4 fields x 2 leads = 8 channels), station PM2.5 for the last 6 h, HMS density, sin/cos hour and day of year.
  - Target: log1p(PM2.5 at t+6 h).
- **Model.** CNN branch (2 conv + pool) concatenated with an MLP on tabular features, giving 3 dense layers. Output: quantiles {0.5, 0.9} with pinball loss (about 0.1M parameters).
- **Training budget.** About 30 min.
- **Evaluation.**
  - Metrics: RMSE/MAE overall and **on smoke episodes** (HMS medium/heavy at site), CSI/POD/FAR for exceeding 35.4 µg/m³, q90 coverage, spatial-holdout scores.
  - Baselines: persistence; raw HRRR-Smoke MASSDEN f06 linearly calibrated + 30-day median background; linear regression on the same inputs.
  - Ablation: without HMS.
- **Pitfalls to check.**
  - Persistence dominates clean hours (episode metrics are the headline).
  - MASSDEN is fire smoke only (the model learns the background).
  - Monitor autocorrelation (spatial holdout).
  - Oklahoma rarely has severe smoke (report OK separately; do not oversell).
- **Risk layer.**
  - Gridded: the model is applied on the 3 km grid, with the observation channel = IDW of current monitors.
  - r_smoke = clip((PM - 35.4)/(150.4 - 35.4), 0, 1).
  - Policy: exposure only. Civilian w = 0.5, a "sensitive" profile w = 5, EV w = 0. Shelters with r > 0.5 are deprioritized.
  - `observed()` = AQS IDW.
  - Case: the top Oklahoma AQS PM2.5 day in 2024 [CANDIDATE; T7]. If no 2024 OK day exceeds 35.4, use the top 2024 HMS-heavy episode at any US metro, with routing on that city.
  - Rule baseline = calibrated raw HRRR-Smoke.

---

## 4. Routing and optimization

### 4.1 Road graph

- **OKC graph.** Geofabrik `oklahoma-latest.osm.pbf` (162 MB, ODbL) is clipped with osmium to OKC_BBOX, then loaded with `osmnx.graph_from_xml`, `network_type="drive"`, simplified. Overpass `graph_from_bbox` is the alternative.
  - Size [estimate 100-150k nodes; T4 records].
  - Exported to CSR `.npz` (indptr, indices, length, free-flow time, lanes, capacity, highway, bridge, crossing flags, geometry id).
- **Speeds.** `ox.add_edge_speeds/add_edge_travel_times`, with imputation.
- **Capacity.** 1,800 veh/h/lane for motorway/trunk, 900 for primary/secondary, 600 for others. These are HCM-style defaults, documented.
- **Any city.** `outrun route --place "Tulsa, Oklahoma, USA"` and the app city box build and cache on demand. Acceptance tests use Tulsa plus the WSTS demo-fire town.
- **Facilities.**
  - NSS shelters (with capacity field), hospitals, fire/EMS stations and schools from `source.coop/seerai/hifld` Parquet [CANDIDATE: capacity field present; T11]. Data Rescue Project is the second source and OSM amenities the third.
  - All snapped to the nearest graph node.

### 4.2 Hazard risk to time-varying edge costs

- **Slots.** 15 min, t0 to t0+3 h (12 slots). Layers are piecewise-constant per slot (no interpolation).
- **EdgeRiskSampler.** r[e, h, s] = max over the 1 km cells crossed by the edge geometry (sampled every 100 m), then `edge_overrides` applied. One vectorized pass, cached per scenario. The length-weighted mean is an ablation.
- **Clock time (advances the vehicle clock):**
  `tau_e(t) = (length_e / speed_e) / prod_h speed_factor_h(r[e,h,s(t)]) * BPR_e(t)`. Every factor is ≥ 1 relative to free flow; the clock is capped at 3x free flow.
- **Generalized cost (what A* minimizes):**
  `c_e^p(t) = tau_e(t) * (1 + sum_h w_h^p * r[e,h,s(t)])`, and `c_e^p(t) = +inf` if `r[e,h,s(t)] >= block_at_h^p` for any h.
  Exposure adds cost, not clock time, so travel times stay physically meaningful.
- **Weights** (`configs/hazards.yaml`; a sensitivity sweep at x0.5 and x2 goes in the report):

| Profile | Storm | Tornado | Flood | Fire | Ice | Smoke | Blocking |
|---|---|---|---|---|---|---|---|
| civilian | 2 | 20 | 10 | 20 | 3 | 0.5 | tornado ≥ 0.5, flood crossing ≥ 0.4, fire ≥ 0.5 |
| sensitive | 2 | 20 | 10 | 20 | 3 | 5 | same as civilian |
| ev | 0.5 | 5 | 3 | 5 | 1 | 0 | flood crossing ≥ 0.7, fire ≥ 0.7, tornado ≥ 0.5 |

- **UI slider.** A "safer <-> faster" lambda multiplies all w_h^p (default 1). It also gives the EMS Pareto curve (4.6).

### 4.3 A* search (own implementation, `routing/astar.py`)

- **Problem.** Because the generalized cost is time-dependent and differs from clock time, one label per node is not exact. Arriving later can be cheaper if a hazard has passed.
- **Chosen design: exact multi-label time-dependent A\* with waiting.**
  - A label is (node, arrival clock t, cost g).
  - Waiting at a node is allowed, with wait cost = wait time x (1 + sum_h w_h r at the node's cell). This is ≥ wait time.
  - Label L1 = (t1, g1) **dominates** L2 = (t2, g2) at the same node if t1 ≤ t2 and g1 + W(t1 -> t2) ≤ g2, since L1 can wait and become at least as good as L2. Each node keeps its Pareto set of non-dominated labels.
  - The priority queue is ordered by f = g + h. Search terminates when the goal label with minimal f is popped.
  - When all risks are zero or static, each Pareto set has size 1 and this reduces to ordinary A*.
  - Safety cap: 8 labels per node. A capped run is flagged in the result (approximate) and counted in NOTES.md.
- **Heuristic.** h(n) = haversine(n, goal) / v_max, where v_max = the maximum free-flow edge speed in the graph (about 33.5 m/s).
  - **Admissible:** every edge cost c ≥ tau ≥ free-flow time ≥ great-circle span / v_max, because speed factors, BPR and risk multipliers are all ≥ 1, blocked edges are infinite, and wait costs are ≥ 0. Hence h never overestimates.
  - **Consistent:** by the triangle inequality of great-circle distance.
  - This is why risk may only multiply by ≥ 1, add ≥ 0 or block, and never discount. The argument goes in spec 04 and the report.
- **Multi-goal ("nearest safe shelter").** h = min over goals (KD-tree pruned to the 20 nearest), still admissible.
- **ALT.** 8 farthest-point landmarks on free-flow times are an optional speed-up. They stay admissible because hazards only add cost.
- **Tests.**
  - `ASTAR-01`: equals networkx Dijkstra on 200 random OD pairs (zero and static risk).
  - `ASTAR-02`: fewer expansions than Dijkstra.
  - `ASTAR-03`: never uses blocked edges.
  - `ASTAR-04`: on 30-node synthetic graphs with time-varying risk, **equals a brute-force time-expanded Dijkstra over (node, minute) states using the same risk-weighted cost and waiting**.
  - `ASTAR-05`: returns None when cut off.
  - `ASTAR-06`: under 1 s per metro query [target].
- **Shelter-in-place rule (`routing/planner.py`).** If the tornado swath or fire front reaches the origin cell before the best route exits the hazard footprint, recommend shelter in place, or the nearest hardened facility (hospital or school) within 5 min drive. OKC has no public tornado shelters, and the UI says so.
- **Baselines.** (a) Plain shortest time (risk ignored). (a') `avoid-current-only`: risk frozen at t0, no forecast. This isolates the value of forecasting.

### 4.4 Shelter assignment (capacity-constrained)

- **Demand.** 2020 Census block groups in the evacuation polygon (any hazard p ≥ 0.3, plus 2 km) [estimate about 700 in metro]. Shelter-seekers = population x 10% (`planning.yaml`; Decision 5), at 2.5 persons/vehicle.
- **Supply.**
  - NSS shelters with their capacity.
  - If there are fewer than 20 in scope, add fallback safe destinations: schools at 500 default; fire stations at 50. These are **flagged as assumed** in the UI.
  - Hospitals are for medical needs only.
  - Shelters with fire/tornado/flood r > 0.5, or smoke r > 0.5 for the sensitive profile, are removed.
- **Costs.** One reverse multi-source time-dependent search per shelter over the civilian cost.
- **Algorithm.** OR-Tools `SimpleMinCostFlow`: source -> zones (supply) -> shelters (cost) -> sink (capacity), plus an "unserved" arc at 10x the max cost so the problem is always feasible. Exact by LP integrality.
- **Baselines.** (1) Nearest shelter by straight line, capacity ignored, reporting overflow %. (2) Greedy nearest-with-remaining-capacity in descending-population order.
- **Metrics.** Total and max person-minutes, overflow, unserved, mean exposure.

### 4.5 Whole-city evacuation (simulated annealing with congestion)

- **Zones and demand.** Block groups. Vehicles v_z = population x 0.9 compliance / 2.5, grouped in packets of up to 50 vehicles per zone (lever: cluster to 200 zones).
- **Destinations.** Min-cost-flow shelters, or metro-boundary exits outside the hazard.
- **Decision per packet.** Path k ∈ {0..3}, from 4 diverse risk-aware candidates made by rerunning A* with 1.4x cost on edges already used. Wave w ∈ {0, 1, 2, 3} (30-min waves).
- **Congestion evaluator (shared by every method).** Static BPR per wave, t_e = t0_e (1 + 0.15 (x_e/cap_e)^4). A packet loads the waves that its traversal window overlaps. Loads come from sparse incidence matrix-vector products.
- **Objective.**
  `J = sum_waves sum_e x_e t_e(x_e)  +  lambda_E sum_packets v * Exposure(path, wave)  +  lambda_C * T_clear  +  M * unserved`
  - Exposure is risk-minutes from the time-dependent layers at the times traversed.
  - T_clear = 95th-percentile arrival time.
  - Defaults: lambda_E = 1 veh-h per risk-veh-h, lambda_C = 0.1 x total vehicles, M large.
- **SA.**
  - Moves: re-pick k (75%), re-pick w (20%), swap waves of two adjacent zones (5%).
  - Incremental ΔJ: only the old and new path edges change.
  - Initial temperature T0 is set so that 80% of uphill moves are accepted in a 500-move warm-up.
  - Geometric cooling alpha = 0.9995, 40k iterations [estimate 30-60 s], 5 seeded restarts. Report best and spread.
  - The objective curve streams to the UI over SSE.
- **Baselines (scored by the same evaluator).**
  - (a) All-take-shortest: free flow, wave 0, nearest shelter.
  - (b) All-take-safest: risk-aware A*, wave 0, congestion-blind.
  - (c) **MSA user equilibrium**: method of successive averages, 50 iterations, on the risk-aware cost.
  - (d) SA ablation with lambda_E = 0.
- **Lower bound.** Sum of free-flow risk-aware shortest costs. The SA gap to it is reported.
- **Claims to test.** SA ≤ MSA on J (system optimum vs user equilibrium). SA << (a) on exposure.
- **Depth lever up.** Re-score the final plans in a mesoscopic link-queue simulator with spillback.

### 4.6 Emergency vehicle routing (inbound, fastest-safe)

- **Inbound.** One reverse time-dependent search from the incident over the reversed graph, EV profile, giving ETA from every station at once.
  - EVs travel 1.2x faster than congested traffic, where congestion comes from the active evacuation plan, so EVs avoid outbound jams.
  - The route itself comes from forward TD-A*.
- **Multiple incidents (1-3).** Station x incident ETA matrix, then the **Hungarian algorithm** (`linear_sum_assignment`).
- **Outbound.** Nearest open hospital outside the hazard.
- **Fastest-safe trade-off.** A lambda sweep over EV weights {0, 0.25, 0.5, 1, 2} gives a time-vs-exposure Pareto curve, shown as a figure and in the app.
- **Baseline (simulated realistically).** Nearest station by straight line, plus a hazard-blind shortest path. When it reaches a blocked edge it "discovers" the block and replans from there, so the reported ETA is the **true ETA including the detour**.
- **Metrics.** True ETA, detour minutes, % of routes hitting a block, exposure.

---

## 5. Interactive map app

### Stack

- FastAPI (`outrun serve`) + static `index.html`/`app.js` with vendored MapLibre GL JS 4 and `pmtiles.js`.
- Local Protomaps basemaps: `okc.pmtiles` [estimate 60-150 MB; T6] plus one for the wildfire demo town.
- Risk overlays: RGBA PNG per (hazard, slot, source) as MapLibre image sources.
- Routes, zones and flows: GeoJSON.
- Evacuation animation: vendored deck.gl UMD `TripsLayer` on a MapLibre overlay. Fallback: animated line-color v/c per wave in plain MapLibre.

### API

- `GET /api/scenarios`
- `GET /api/scenarios/{id}/overlay/{hazard}/{slot}.png?source=model|baseline|observed`
- `POST /api/route {origin, dest|"nearest_shelter", profile, lambda, t0}`
- `GET /api/shelters/{id}?method=mcf|nearest|greedy`
- `GET /api/evac/{id}?method=sa|msa|shortest|safest` and `GET /api/evac/{id}/run` (SSE)
- `POST /api/dispatch {incidents[]}`
- `GET /api/city?name=`
- `GET /api/metrics`

### Screens (tabs on one page)

1. **Forecast.**
   - Scenario picker. Each card shows its real date, data sources and held-out badge:
     - El Reno May 25 2019 (storm + tornado + all routing)
     - flood (test year)
     - ice (2024-25)
     - smoke (2024)
     - WSTS 2021 fire
     - OK Mar 2025 replay (labelled)
     - synthetic sandbox
   - Time slider with play button, six hazard toggles.
   - **Forecast | Observed swipe** comparison.
   - Model-card drawer showing metric vs baseline and the reliability plot.
2. **Safe route.**
   - Click an origin (and optionally a destination).
   - Outrun route (cyan) vs shortest (grey dashed), with ETA, exposure (predicted and observed), blocked crossings avoided, A* vs Dijkstra expansions.
   - lambda slider. A "Shelter in place" card when the rule fires.
3. **Shelters.** Zone-to-shelter flow lines, capacity bars, method toggle (MCF / nearest / greedy), metrics strip with overflow in red.
4. **City evacuation.** "Run optimizer" streams the live SA objective curve. Animated vehicles / v-c colouring over 4 waves, method toggle (SA, MSA, shortest, safest), J breakdown, clearance time.
5. **Dispatch.** Drop 1-3 incidents. Shows the Hungarian assignment, the inbound routes, and the baseline truck visibly hitting a flooded crossing and detouring. Pareto mini-chart.
6. **Models and results.** Six models vs baselines with CIs, plus routing gap-closed numbers, rendered from `reports/metrics/*.json`.
7. **Any US city box** (in the header). Builds or loads a graph and routes on it.

### Offline-safe demo mode

- `make demo` sets `OUTRUN_DEMO=1`. All data comes from `scenarios/` bundles precomputed in T22 (risk.zarr, PNGs, cached graph, MCF/SA/MSA/EMS results). Local PMTiles, no CDN.
- Live A*, MCF and SA still run locally (no network needed). If a live call fails, the app falls back to cached results for 30 preset origins.
- `tests/e2e/test_offline.py` (Playwright) loads every tab and asserts **zero non-localhost requests**.
- `tests/unit/test_offline_socket.py` monkeypatches sockets for server-side code.

### Static recruiter demo

`outrun export-static` writes JSON/PNG/GeoJSON for preset clicks into `docs/`, deployed with GitHub Pages and linked from the README.

### Presentation demo flow (6:30, inside the 5-7 min limit)

| Time | Content |
|---|---|
| 0:00-0:40 | Slide: Oklahoma's six hazards, plus the proposal sentence |
| 0:40-1:20 | Slide: architecture. Six models feed one RiskStack, which feeds one router. HazardPolicy and "what you see is what routes". |
| 1:20-2:20 | Live, El Reno 2019: storm nowcast plays; swipe Forecast/Observed; tornado swath appears (held-out badge) |
| 2:20-3:00 | Route from an El Reno home: shortest drives into the swath; Outrun detours (or shows shelter-in-place); the observed SPC track confirms |
| 3:00-3:45 | Shelters MCF vs nearest (overflow drops); Evacuation: run SA live (~30 s curve), animated congestion, KPIs vs all-shortest and MSA |
| 3:45-4:20 | Flood scenario: dispatch 2 incidents; the baseline truck hits a low-water crossing and detours (true ETA) |
| 4:20-4:40 | Quick flips: ice (bridges flagged), smoke, WSTS 2021 fire in its own town (any US city) |
| 4:40-6:00 | Slides: model results vs baselines (wins and losses), routing gap-closed with CIs, satellite ablations |
| 6:00-6:30 | Lessons, limitations, next steps |

### Backup video

- About 6:30, 1080p QuickTime recording of the full flow, **narrated by Milton** from DEMO.md, recorded Dec 3.
- MP4 attached to the GitHub Release (not committed) and posted as an unlisted video. Linked from the README and the slides.
- Queued in a second window during the live demo.

---

## 6. Evaluation plan

### 6.1 Per-model test metrics (computed once, in T24)

95% CIs come from a day-block bootstrap (1,000 resamples). Each model is reported on (i) its **official/unmodified test split** and (ii) the **case hold-out set** separately.

| Hazard | Primary metric | Calibration | Baselines | Robustness / shortcut headline |
|---|---|---|---|---|
| Storm | CSI-74/133 averaged over leads, per-lead curves, FSS, MSE | Brier, reliability | persistence, pySTEPS | VIL+IR vs VIL-only; CSI-181/219; 1 km check; storm-windows subset |
| Tornado | **TOR-vs-warned-null AUC**, overall AUC/AUPRC | Brier, reliability | GBM hand features, ΔV rule; 0.874 reference | AUC by range bin; no-range-channel ablation; missing-rate check |
| Flood | AUPRC, neighbourhood CSI, event POD | Brier, reliability | **FFG ratio ≥ 1**, ARI ≥ 10-yr, LR | hard-negative AUC; urban/rural POD; no-FLASH ablation |
| Wildfire | PR-AUC all + **new-burn**; **WSTS 2021 out-of-time** PR-AUC | Brier, reliability | persistence, downwind dilation, LR; 0.284 reference | new-burn pixels; per-fire macro average; mapper shift table |
| Ice | zone AUPRC, BSS vs climatology | reliability | physics rule, climatology, LR | urban/rural zones; rule vs model on the same windows; bridge-prior ablation |
| Smoke | RMSE/MAE on **smoke episodes**, CSI > 35.4 | q90 coverage | persistence, calibrated HRRR-Smoke, LR | spatial-holdout monitors; no-HMS ablation |

The report states plainly where a deep model does not beat its baseline (likely candidates: ice, flood, smoke at short leads). The proposal promises a deep-learning forecast, not one that beats every baseline. Routing still uses the DL layer, and the baseline layer's routing results are reported side by side.

### 6.2 Routing metrics (plan on forecast, score on observed)

- **Sample.** For each scenario, 500 population-weighted residential origins routed to the nearest safe shelter.
- **Planners:**
  - (a) no hazard
  - (a') avoid-current-only
  - (b) baseline-risk layers
  - (c) DL-risk layers
  - (d) oracle (planned on `observed()`)
- **Metrics:**
  - % of routes intersecting observed hazard (r ≥ 0.5 or blocked)
  - integrated observed exposure (risk-minutes)
  - travel-time ratio vs (a)
  - blocked crossings traversed
  - predicted exposure as well
- **Headline:** "DL planning (c) closes X% of the observed-exposure gap between (a) and (d)", with a paired-bootstrap CI over the 500 OD pairs. Also reported: (c) vs (b), and (c) vs (a').
- **Group metrics:**
  - Shelters: overflow, unserved, max person-minutes.
  - Evacuation: vehicle-hours, exposure, clearance time, max v/c, SA vs MSA/shortest/safest/lambda=0, gap to lower bound, 5 seeds.
  - EMS: true ETA, detour, % blocked, Pareto curve.

### 6.3 Held-out discipline

| Step | When |
|---|---|
| Catalog checks + TorNet split rule verified | T1 / T7 |
| `splits/*.json` + `quarantine.json` frozen and committed, `SPLIT-01..07` green | T7, Nov 20, before any training |
| Model selection, thresholds, calibration | val only |
| Test evaluation, one run per final model | T24, Dec 1, logged with git sha |
| Acceptance tests (`OUTRUN_ACCEPTANCE=1`) | They check that metrics exist and were computed on test, not that baselines were beaten |

No task before T24 has "test metrics" in its done-when. The training tasks log val metrics only.

### 6.4 Case studies (all real dates)

No single event is covered by all six datasets: SEVIR covers 2017-2019, MRMS on AWS starts 2020-10, HRRR-Smoke starts about 2020-12, and WSTS covers 2018-2021. So there is one primary multi-hazard case plus per-hazard cases. The report states this.

- **Primary (storm + tornado + safe routes + shelters + evacuation + EMS): [CANDIDATE] El Reno EF3, evening of May 25 2019 (about 03:30 UTC May 26), Canadian County.**
  - T1/T7 verify: SEVIR CATALOG VIL+IR events whose extent covers El Reno in that window; TorNet catalog tornadic chips within 100 km of KTLX that day; the SPC track row.
  - May 22-29 2019 is quarantined. That period is on SEVIR's train/val side of the 2019-06-01 cut.
  - Fallbacks in order: [CANDIDATE] May 20 2019 (OK high-risk day); any OK SEVIR event in the official test window found by catalog filter.
- **Official-test OK storm event.** One Oklahoma SEVIR event from Jun-Dec 2019 [CANDIDATE; T7 catalog filter], scored with no quarantine involved.
- **Flood.** The OKC-metro (Oklahoma/Cleveland/Canadian County) flash-flood event with the most Storm Events rows in the test year (2025, else 2024) [CANDIDATE; T7].
- **Ice.** The Oklahoma ice event in the 2024-25 test winter with the most ice-keyword zones [CANDIDATE; T7]. It is inside the HRRR window by construction. Fallback: the top ice event that winter in TX/KS/AR, routed on that city.
- **Smoke.** The top Oklahoma AQS day in 2024, with the fallback in 3.6.
- **Wildfire.** The WSTS 2021 demo fire (evaluated), plus the OK Mar 14 2025 replay (illustrative only).

---

## 7. Scope and risk

### 7.1 Required vs depth levers

**Required:** every row of the 1.3 matrix. That means:

- six deep-learning forecasters, each with a baseline, calibration and a RiskLayer
- IR107, NDWS/WSTS and HMS satellite inputs
- time-dependent A*
- min-cost-flow shelters
- SA evacuation with the MSA baseline
- EMS dispatch
- OSM roads and any-city routing
- the map app with offline mode
- the primary case study
- report, slides and backup video

**Depth levers** (applied at checkpoints; none drops an item):

| Area | Lever |
|---|---|
| Storm | 4 km 96x96, 10 epochs; IR ablation on 2019 files only; drop the 1 km check |
| Tornado | 2019 only; last frame only; 8 epochs |
| Flood | Southern Plains patches only; fewer FLASH hours; drop ARI |
| Wildfire | skip the per-fire macro average; replay uses more climatological channels |
| Ice | 6-hourly cycles; 3 winters; 2-layer CNN with MIL |
| Smoke | f06 only; median output only |
| Routing | 200 zones, 10k SA iterations, MSA 20 iterations, 100 OD pairs for replays |
| App | merge Shelters into Evacuation; plain v/c animation instead of TripsLayer; more cached responses |

**Levers up if ahead:** 3 seeds or ensembles with uncertainty bands, more TorNet years (2017/2018/2020), KTLX Level-II live inference, link-queue simulator re-scoring, freeway contraflow, leave-region-out radar check.

**Checkpoints:**

- Nov 19: CLI A* on the real graph.
- Nov 20: map skeleton offline on synthetic layers.
- Nov 24: storm + tornado DL layers live.
- Nov 27: six baseline layers live, plus at least four DL layers. Otherwise apply levers that day.
- Dec 1: feature freeze (end of day).

### 7.2 Top 8 risks

| # | Risk | Mitigation |
|---|---|---|
| 1 | Downloads slow or endpoints gone (Zenodo throttling, Kaggle auth, HIFLD mirror moved, Storm Events/AQS 2025 not posted) | All bulk downloads start Nov 17-18, resumable, with sizes and sha256 in the manifest. Fallbacks: TorNet 2019 only; HIFLD -> Data Rescue -> OSM; Kaggle -> WSTS-only training (loses the 0.284 comparison); year fallbacks per section 3. |
| 2 | HRRR volume or tooling (eccodes on macOS) blows the schedule | T3 decodes one message per product and measures size before bulk. Hard caps (ice 80 GB, smoke 60 GB streamed) with predefined cut rules; pygrib fallback. |
| 3 | IR107 larger than expected, or poor overlap with VIL | T1 `aws s3 ls` + CATALOG overlap count; the 60 GB rule in 3.1; the ablation uses the same events. |
| 4 | MPS bugs or slow epochs | Plain Conv2d; `PYTORCH_ENABLE_MPS_FALLBACK=1`; one timed epoch before every overnight run; checkpoint every epoch with a resume flag; fp32 fallback. |
| 5 | Weak labels make flood/ice/smoke near-random or lose to baselines | Hard negatives, uncertain ring, MIL, honest comparison to operational rules; the layer still ships; the report treats it as a finding. |
| 6 | Leakage or shortcuts inflate metrics | Splits frozen after verification; quarantine; SPLIT tests; gap days; test once; per-hazard shortcut checks (section 6.1). |
| 7 | Time-dependent A* or SA too slow on about 150k nodes | CSR arrays, metro bbox, cached edge risks, numba, optional ALT, label cap with flag, incremental ΔJ. Targets: under 1 s per A* query, under 60 s per SA run. |
| 8 | Live demo failure or a schedule slip around Thanksgiving | Offline mode with a Playwright no-network test, static Pages, narrated backup video, 3 rehearsals. Thanksgiving holds one light task (training already prepared). Freeze Dec 1 leaves Dec 2-4 for docs only. |

Not pursued: Oklahoma Mesonet road and soil temperatures (academic agreement turnaround could exceed 17 days). The report names it as the top ice improvement.

### 7.3 Disk budget (margin rule: HDD at most 375 of 500 GB, SSD at most 225 of the 300 GB budget)

| Item | HDD raw (GB) | SSD processed (GB) |
|---|---|---|
| SEVIR VIL 2018-19 + CATALOG [CANDIDATE] | 65 | 47 (VIL+IR 2 km uint8) |
| SEVIR IR107 2018-19 (capped by rule) | ≤ 60 | (in the row above) |
| TorNet 2015/16/19 | 51 | 30 + 19 transient (one year extracted at a time) |
| NDWS | 4 | 4 |
| WSTS (whole archive) [CANDIDATE] | 48 | 15 (2021 extracted + processed) |
| MRMS QPE Apr-Sep 2021-25 | 10 | 8 (crops + patches, incl. FLASH) |
| MRMS FLASH event windows (cap) | ≤ 20 | (in the row above) |
| ETOPO 30" + 3DEP 6 tiles | 3 | 4 |
| HRRR ice / smoke (streamed, not stored raw) | 0 | 14 + 2 |
| AQS, HMS, Storm Events, SPC, NWS zones | 1.5 | 1 |
| OSM PBF, HIFLD, census, graphs, PMTiles | 1 | 3 |
| Models, scenarios, logs, figures | - | 12 |
| venv and caches | - | 6 |
| **Total (worst case)** | **≈ 264 (53%)** | **≈ 165 peak (55% of budget)** |

If T1 measurements push the HDD over 375 GB, cut in this order: IR107 2018 H1 files, then WSTS (re-download only the 2021 portion if the record allows it), then TorNet 2016. `outrun disk` checks free space before every job.

---

## 8. Calendar: Nov 17 to Dec 4

**Conventions**

- One commit per task: `T<n>: ...`, real dates, pushed the same day, no co-author trailer.
- A dated NOTES.md entry every day.
- `[overnight]` marks a background job started before bed. The next day's task begins by checking its log.
- `[W]` marks a file-disjoint task suitable for a local `worker` run. Claude reviews the diff and runs the tests twice before merging.
- Every "done when" includes: listed spec tests green and a NOTES.md entry. All model metrics before T24 are **validation** metrics.

| T | Date | Task | Spec | Done when |
|---|---|---|---|---|
| T0 | Tue Nov 17 | Scaffold: pyproject, Makefile, CLAUDE.md, TASKS.md, NOTES.md, specs 00-16 with req IDs, stubs with real signatures, red tests, traceability + **proposal-matrix** tests, CI | 00 | CI green on traceability; rest red with NotImplementedError; repo pushed public |
| T1 | Tue Nov 17 | `data/fetch.py` + `manifest.yaml`. Size inventory: SEVIR vil/ir107 2018-19 via `aws s3 ls`, VIL+IR overlap count in CATALOG, TorNet and WSTS Zenodo file lists, Storm Events/AQS 2025 availability. Read the `mit-ll/tornet` split code. [overnight] batch A: CATALOG, SEVIR VIL+IR107 (per rule), TorNet 2019 -> 2015 -> 2016, NDWS, Storm Events 2019-2026, SPC, Geofabrik OK | 01 | Sizes, overlap count, TorNet split rule and data-availability results in NOTES; downloads running |
| T2 | Wed Nov 18 | `core/grid.py`, `risk.py`, `policy.py` (incl. beyond-horizon), `scenario.py`, `synthetic.py`, `calibrate.py` | 02 | GRID/RISK/HORIZON tests green, incl. overlay = sampled-values test |
| T3 | Wed Nov 18 | `data/hrrr.py`, `mrms.py`, `aqs.py`, `hms.py`, `dem.py`: decode one file per product; **measure HRRR MB/message**; list FLASH products. [overnight] batch B: WSTS, MRMS QPE, FLASH event windows, HRRR **ice** stream-crop, AQS, HMS, ETOPO, 3DEP, census, NWS zones, HIFLD | 01 | Fixture decoded per source; recomputed HRRR volumes and FLASH names in NOTES; batch B running within caps |
| T4 | Thu Nov 19 | `roads/graph.py` + `roads/features.py`: OKC graph from PBF, speeds, capacities, bridges, crossings, CSR export | 03 | Node/edge/crossing counts in NOTES; ROAD tests green |
| T5 | Thu Nov 19 | `routing/cost.py` + multi-label TD-A* + Dijkstra oracle + brute-force time-expanded checker + `outrun route --place`. [overnight] HRRR **smoke** stream-crop | 04 | ASTAR-01..06 green; Tulsa route works. **Checkpoint: CLI A* on the real graph** |
| T6 | Fri Nov 20 | App skeleton: FastAPI, vendored MapLibre, local PMTiles, synthetic scenario overlays + live route compare, offline socket test | 14 | Clicking two points shows risk-aware vs shortest route, network off. **Skeleton milestone** |
| T7 | Fri Nov 20 | Case-study verification in SEVIR/TorNet catalogs, SPC, Storm Events, HMS, AQS; write and commit `splits/*.json`, `quarantine.json`, `hashes.json`; SPLIT-01..07. **Before any training** | 15 | Splits frozen; chosen events (with fallbacks used, if any) in NOTES |
| T8 | Sat Nov 21 | Storm: SEVIR VIL+IR preprocessing to memmap; persistence + pySTEPS on val; storm baseline adapter. [overnight] U-Net VIL+IR after a timed epoch | 08 | Baseline layer in app; val CSI of baselines in NOTES; training running |
| T9 | Sun Nov 22 | Storm: val eval, isotonic calibration, DL adapter with decay. [overnight] VIL-only ablation train + TorNet extract/preprocess (CPU, one year at a time) | 08 | DL storm layer in app; val CSI curves in NOTES |
| T10 | Mon Nov 23 | Tornado: dataset, GBM + ΔV baselines, CNN. [overnight] tornado CNN train | 09 | Baseline val AUC and warned-only val AUC logged; training running |
| T11 [W] | Mon Nov 23 | Shelters: facilities + census loaders, verify capacity field, min-cost flow + nearest/greedy | 05 | SHEL tests green (exact optimum on toy graph); metro shelter counts in NOTES |
| T12 | Tue Nov 24 | Tornado: val eval (warned-only, range bins), swath adapter with motion fallback chain. [overnight] no-range-channel ablation | 09 | Swath on the El Reno scenario. **Checkpoint: storm + tornado DL live** |
| T13 [W] | Tue Nov 24 | Wildfire: NDWS conversion, U-Net + baselines, train (about 1 h), val; WSTS channel mapper + shift table; pick WSTS 2021 demo fire | 11 | Val PR-AUC (all + new-burn) logged; mapper KS table in NOTES |
| T14 | Wed Nov 25 | Flood: accumulations, FLASH features, uncertain-ring labels, hard-negative sampler, FFG/ARI/LR baselines on val. [overnight] flood U-Net train | 10 | Baseline val scores logged; training running |
| T15 [W] | Wed Nov 25 | Ice: zone bags from HRRR crops, labels with keyword filter, physics-rule + climatology baselines on val | 12 | Bags built; rule val AUPRC logged |
| T16 | Thu Nov 26 (Thanksgiving, light) | Ice: MIL-CNN train (about 1 h), val eval, bridge prior, adapter | 12 | Ice layer in app; val metrics logged |
| T17 | Fri Nov 27 | Flood: val eval, no-FLASH ablation queued, crossing adapter (3DEP depressions) | 10 | Flooded crossings block in app |
| T18 [W] | Fri Nov 27 | Smoke: station samples, CNN+MLP quantile model, baselines, spatial holdout, grid adapter | 13 | Val metrics logged. **Checkpoint: 6/6 layers live (at least 6 baselines + 4 DL)** |
| T19 | Sat Nov 28 | Evacuation: zones/packets, k-paths, wave-BPR evaluator, SA, MSA, shortest/safest, lambda=0 ablation, SSE progress | 06 | EVAC tests green; SA ≤ MSA on J for El Reno (or explained) |
| T20 | Sat Nov 28 | Wildfire scenario layers: WSTS 2021 demo-fire RiskLayer on that town's graph; OK Mar 2025 replay builder (labelled) | 11 | Both fire scenarios render; any-city graph cached |
| T21 | Sun Nov 29 | EMS: reverse TD search, Hungarian dispatch, discover-and-reroute baseline, lambda Pareto, hospital leg; shelter-in-place rule | 07, 04 | EMS + PLAN tests green |
| T22 | Sun Nov 29 | `outrun scenario build` for all scenarios. [overnight] build bundles | 02, 15 | `scenarios/` complete; demo loads with network off |
| T23 | Mon Nov 30 | App: all tabs, swipe, lambda slider, SA curve, TripsLayer (or fallback), city box, model drawer; Playwright no-network test | 14 | Full demo flow clickable offline; e2e green |
| T24 | Tue Dec 1 | **One-shot test evaluation** of all 6 models (official + case hold-out); routing eval (a/a'/b/c/d) with bootstrap CIs; optimizer metrics; figures; acceptance suite. **Feature freeze at end of day** | 15, 16 | `reports/metrics.md` + figures; git sha logged; acceptance passes or misses explained |
| T25 | Wed Dec 2 | README (GIF, badges, quickstart, results table), model cards, DATA_LICENSES, `export-static` to Pages, CI polish | 14 | Pages URL works; fresh clone `make test` green twice |
| T26 | Wed Dec 2 | REPORT.md: what and why, tech vs alternatives, problems solved or not, lessons, improvements; satellite inputs named; export PDF via pandoc | - | Covers all 5 rubric prompts with figures and CIs |
| T27 | Thu Dec 3 | Slides (about 12: title, problem, proposal, architecture, RiskLayer/Policy, 6 models vs baselines, satellite ablations, routing and optimization, gap-closed, demo, honest results, lessons) as PPTX + PDF in `slides/` | - | Deck committed |
| T28 | Thu Dec 3 | DEMO.md script; 3 timed rehearsals (target 6:00-6:30); record narrated backup video (about 6:30); Release `v1.0-rc` with MP4 | - | Video linked in README and slides; timings logged |
| T29 | Fri Dec 4 (a.m.) | Clean-clone `make demo` dry run offline; tag `v1.0`; submit repo URL + report PDF + slides before 1:40 pm | - | Submitted |

**Overnight jobs**

| Night | Job |
|---|---|
| Nov 17 | Batch A downloads |
| Nov 18 | Batch B + HRRR ice stream |
| Nov 19 | HRRR smoke stream |
| Nov 21 | Storm U-Net |
| Nov 22 | VIL-only ablation + TorNet preprocessing |
| Nov 23 | Tornado CNN |
| Nov 24 | Range ablation |
| Nov 25 | Flood U-Net |
| Nov 29 | Scenario bundles |

**Dependency checks**

- IR107 lands Nov 17, before storm prep on Nov 21.
- HRRR ice lands Nov 18, before ice prep on Nov 25.
- FLASH lands Nov 18, before flood on Nov 25.
- WSTS lands Nov 18-19, before T13 on Nov 24.
- Splits are frozen Nov 20, before the first training run on Nov 21.

**Coverage and buffer.** All 18 days from Nov 17 to Dec 4 have a commit. Buffer: the afternoons of Nov 30 and Dec 1, plus Dec 2-3 are docs-only.

---

## 9. Decisions for Milton (5, each with a recommended default)

1. **Kaggle account + API token for NDWS (needed today).** *Default: yes.* Put `~/.kaggle/kaggle.json` in place on Nov 17 (about 5 min). Without it, wildfire trains on WSTS 2018-20 only and loses the published 0.284 comparison.
2. **Primary case study.** *Default:* [CANDIDATE] El Reno, May 25 2019, held out by quarantine (May 22-29 removed from train/val). Official test splits stay unmodified, and one official-test OK SEVIR event (Jun-Dec 2019) is also reported. Fallback order: May 20 2019, then any OK event in the official test window.
3. **Repo visibility and live demo.** *Default:* public repo from T0. MIT licence for code; data licences listed (CC BY 4.0 TorNet/NDWS/WSTS, ODbL OSM, public-domain NOAA/USGS/EPA). Professor added as a collaborator. Static GitHub Pages demo for recruiters.
4. **Wildfire demo location.** *Default:* headline fire = a real WSTS 2021 fire routed on its own town's roads (evaluated out-of-time, and shows "any US city"). Plus the Oklahoma Mar 14 2025 replay as a clearly labelled, domain-shifted illustration. No invented fires.
5. **Planning assumptions.** *Default:* 10% public-shelter seekers, 0.9 evacuation compliance, 2.5 persons/vehicle, school capacity 500 when unknown. Each is flagged as assumed in the UI, with a ±50% sensitivity row in the report.

---

## 10. Changes from drafts

| # | Change | Source / reason | Where |
|---|---|---|---|
| 1 | Base kept: coverage table, baseline-first adapters, `--risk-source` switch, avoid-current-only baseline, MSA baseline, admissibility rule, shelter-in-place, vanilla MapLibre + PMTiles, small subset downloads, checkpointed calendar | delivery-risk-first (winner) | 1-8 |
| 2 | HRRR volumes recomputed as days x cycles x leads x fields x MB/message, with caps and cut rules. Ice: 3-hourly, f01 only, about 42k messages, cap 80 GB. Smoke: 6-hourly, f01/f06, about 35k messages, cap 60 GB. MB/message measured in T3 first. Rejected f00 (zero accumulations) and hourly/f01-f12 plans | must-fix (both judges) | 3.5, 3.6, T3 |
| 3 | Ice window starts winter 2020-21 after the HRRR v4 switch (all v4); test = latest winter 2024-25; case chosen inside the window. Dropped the Oct 2020 and Feb 2021 cases | must-fix | 3.5, 6.4 |
| 4 | TorNet split rule verified from `mit-ll/tornet` code in T1; TorNet kept to 2019/2015/2016, more years as a lever | must-fix | 3.2, T1 |
| 5 | Case days quarantined (±3 d removed from train/val) instead of forced into test. Official splits unmodified and reported alongside the case hold-out set. One official-test OK SEVIR event added | must-fix (both) | 3, 6.1, 6.4 |
| 6 | Splits frozen in T7, after catalog and split-rule verification, before any training. No test metrics before the single T24 evaluation; earlier per-model test runs removed | must-fix | 6.3, 8 |
| 7 | Wildfire: NDWS training kept (0.284 benchmark); WSTS 2021 added as the georeferenced out-of-time test and demo, whole ~48 GB archive budgeted, file list checked in T1; documented and validated channel mapper; OK replay demoted to a labelled extra | must-fix (both) | 3.4 |
| 8 | Satellite made first-class: IR107 as a required storm input (size and overlap checked in T1, 60 GB rule) with a VIL-only ablation; HMS as smoke input with a no-HMS ablation; NDWS/WSTS MODIS/VIIRS channels for fire; all named in the matrix, report and slides | must-fix (both); demo-first idea | 1.3, 3.1, 3.6 |
| 9 | Flood: operational FLASH QPE/FFG ≥ 1 as the core baseline (product name and size verified in T3, ARI-rule fallback); hard negatives; 5-15 km uncertain ring; no-FLASH ablation; urban/rural stratification | must-fix; rigor-first + demo-first ideas | 3.3 |
| 10 | Flood terrain: ETOPO 30" for CONUS training and 3DEP 10 m only for OKC crossings. Replaced the infeasible "CONUS 1-arcsec 3DEP ~3 GB" | must-fix | 3.3, 7.3 |
| 11 | Tornado headline = tornadic vs warned-null AUC, plus range-bin AUC and a no-range-channel ablation | must-fix | 3.2, 6.1 |
| 12 | Tornado motion fallback chain (pySTEPS -> HRRR steering wind -> 15 m/s from 240 degrees); the SPC track is never used as input | must-fix | 3.2 |
| 13 | Beyond-horizon rules (storm decay, tornado drop with lingering block, others hold), with tests | must-fix | 1.2 |
| 14 | Time-dependent A* made exact: multi-label Pareto search with waiting (one label per node is not exact under time-varying generalized cost); checked against a brute-force time-expanded Dijkstra on the risk-weighted cost; admissibility proof kept; ALT optional | must-fix; rigor-first idea, corrected | 4.3 |
| 15 | Exposure adds cost but not clock time | rigor-first | 4.2 |
| 16 | Evacuation: all methods scored by one wave-BPR evaluator; MSA-UE strong baseline; lambda=0 ablation; lower-bound gap; link-queue simulator demoted to a lever up | must-fix | 4.5 |
| 17 | EMS: discover-and-reroute baseline (true ETA incl. detour), Hungarian multi-incident dispatch, lambda Pareto curve | must-fix; demo-first + rigor-first | 4.6 |
| 18 | HazardPolicy plus "what you see is what routes", with a test | demo-first | 1.2 |
| 19 | Ice MIL head (LogSumExp over zone cells) | demo-first | 3.5 |
| 20 | Smoke spatial monitor holdout and episode headline; quantile output with coverage | demo-first + rigor-first | 3.6 |
| 21 | Front end: vanilla MapLibre (no React/Vite chain), with cheap impact features grafted: lambda slider, SSE SA curve, vendored deck.gl TripsLayer (fallback animation), Forecast/Observed swipe, any-city box, Playwright no-network test | must-fix; demo-first ideas | 5 |
| 22 | Proposal-matrix traceability test and [CANDIDATE] tagging with verification tasks and fallbacks | rigor-first | 1.3, throughout |
| 23 | Gap-closed routing headline with CIs (planners a/a'/b/c/d) | rigor-first | 6.2 |
| 24 | Storm downsampling changed from max-pool to average-pool, with a 1 km CSI check | judge flaw | 3.1 |
| 25 | Storm Events 2025/2026 and AQS 2025 availability checked in T1, with named year fallbacks | must-fix | 3.3, 3.5, 3.6 |
| 26 | Removed as required items: full SEVIR (147 GB), all 10 TorNet years, CONUS 1-arcsec DEM, 3-seed ensembles per model (now levers up) | must-fix | 7.1, 7.3 |
| 27 | Disk budget recomputed with IR107, WSTS and FLASH; worst case 264 GB HDD (53%) and 165 GB SSD peak (55% of budget), with a cut order | must-fix (at least 25% margin) | 7.3 |
| 28 | Calendar: skeleton on synthetic layers (CLI Nov 19, map Nov 20); 6 baselines + 4 DL by Nov 27; light Thanksgiving (training already prepared Nov 25); freeze Dec 1; a commit every day; narrated video about 6:30 (previously 4 min); overnight jobs ordered before their consumers | must-fix | 8 |
| 29 | Ice training domain stated honestly as a CONUS subset (ice belt) for HRRR volume; the others stay CONUS | resolves a conflict with the answer given to the professor | 3 |
| 30 | Emoji markers replaced with `[overnight]`; document self-contained | user rules | 8 |
