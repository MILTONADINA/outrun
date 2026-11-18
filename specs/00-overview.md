# 00: Overview

Outrun: CMSC-5713 final project (Oklahoma Christian University), Milton Adina. Due Fri Dec 4 2026, 1:40 pm.
Full design: `docs/DESIGN.md` (source of truth). This file fixes the shared conventions every other spec follows.

## Goal

Ship every item of the approved proposal end to end: six deep-learning hazard forecasters trained on real
weather, radar and satellite data, each behind one `HazardAdapter` interface with an honest baseline; a shared
`RiskStack` on one 1 km grid; a time-dependent A* router, min-cost-flow shelter assignment, simulated-annealing
city evacuation and emergency-vehicle dispatch on real OSM roads; and an interactive map app that also runs offline.
Only depth may shrink (DESIGN 7.1); nothing in the proposal is optional.

## Approved proposal (exact)

> For my final project I propose Outrun, an AI system that uses deep learning and real weather, radar and satellite data to forecast severe storms, tornadoes, flash floods, wildfire spread, icy roads and smoke, then uses A* search and optimization on real road maps to plan safe evacuation routes, shelter assignments, whole-city evacuations and emergency vehicle routes, demonstrated through an interactive map app.

## Architecture

```
                  RAW ($OUTRUN_RAW, internal SSD)                     STREAMED (never stored raw)
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

Key abstraction (DESIGN 1.2): `HazardAdapter -> RiskLayer + HazardPolicy -> CostModel`. The router sees only the
`RiskStack` and the `HazardPolicy` objects, so adding or swapping a hazard model never changes routing code. One
switch, `--risk-source model|baseline|observed|none|current-only`, selects the risk source; the routing evaluation
plans on a forecast and scores on observed data. Overlay PNGs and edge costs come from the same arrays
("what you see is what routes", `RISK-07`).

## Conventions

### Requirements and IDs
- Each `specs/NN-name.md` defines numbered requirements. A requirement is defined by a level-3 heading that starts
  with its ID: `### STORM-03 <short title>`, followed by precise, testable text. Acceptance criteria in
  `16-acceptance.md` may instead be table rows starting `| **AC-01** |`. Nothing else defines an ID.
- IDs are `<PREFIX>-<NN>` with a two-digit number. Prefixes are fixed per spec file (table below); a spec may only
  define IDs with its own prefixes. Numbers need not be contiguous, but never reuse a retired number.
- Every requirement has at least one test marked `@pytest.mark.req("ID")` (one ID per marker; stack markers for
  several IDs). `tests/unit/test_traceability.py` enforces: every defined ID has a test; no test cites an
  undefined ID or an unknown prefix; and the proposal coverage matrix below is fully covered.
- Tests are written before code and must be passable by a sound implementation of the spec. Unit tests use only
  small synthetic fixtures (tiny graphs, 32x32 grids, fake arrays, files generated in `tmp_path` or stored in
  `tests/fixtures/`), run offline on CPU in seconds, and never touch real data or the network.
- Markers: `req(id)`, `slow` (seconds, still offline), `acceptance` (real data + models,
  `OUTRUN_ACCEPTANCE=1`, `tests/acceptance/`), `e2e` (Playwright, `tests/e2e/`). Default `pytest` runs
  `tests/unit` without acceptance and e2e.

### Code
- `src/outrun/` starts as stubs: real public signatures with type hints, docstrings stating the contract and the
  req IDs, bodies `raise NotImplementedError`. Pure data classes and enums are fully defined.
- **Core interfaces live in `outrun.core` and match DESIGN 1.2 exactly. Import them; never redefine them:**

| Name | Module | Notes |
|---|---|---|
| `Hazard` (Enum: STORM, TORNADO, FLOOD, WILDFIRE, ICE, SMOKE) | `outrun.core.risk` | |
| `RiskLayer` (frozen dataclass) | `outrun.core.risk` | fields exactly as DESIGN 1.2 |
| `RiskStack` | `outrun.core.risk` | (hazard, slot, y, x) on GRID, 12 slots x 15 min, zarr IO, overlay renderer |
| `HazardAdapter` (Protocol: `hazard`, `policy`, `baseline`, `predict`, `observed`) | `outrun.core.risk` | implemented in `hazards/<h>/adapter.py` |
| `Profile` (Enum: CIVILIAN, SENSITIVE, EV) | `outrun.core.policy` | |
| `HazardPolicy` (frozen dataclass) | `outrun.core.policy` | values from `configs/hazards.yaml` |
| `Scenario` | `outrun.core.scenario` | id, t0, bbox, city, case metadata, held-out flags |
| `GRID` | `outrun.core.grid` | EPSG:5070, 1 km |

- Units: metres, seconds for clock time internally (minutes only at the UI/report boundary), probabilities in
  [0, 1], times are timezone-aware UTC `datetime`.
- Config values come from `configs/*.yaml` (loaded by `outrun.config`), never hard-coded duplicates.
- Data paths: RAW and PROC both on the internal SSD (`OUTRUN_RAW`, `OUTRUN_PROC`; code default `/Volumes/OutrunRaw` only if unset).

### Held-out discipline
- Splits frozen in T7 (`splits/*.json`, `quarantine.json`, `hashes.json`) before any training.
- Fit on train, select/calibrate/threshold on val. Test evaluated once in T24. Case-study dates +-3 days are
  removed from train and val of every model (`SPLIT-07`); official test splits stay unmodified.

## Spec files and ID prefixes

Each spec owns the listed modules and test file(s), so area work stays file-disjoint.

| Spec file | ID prefix(es) | Owns modules (under `src/outrun/`) | Unit test file(s) |
|---|---|---|---|
| 01-data.md | DATA | `config.py`, `data/*.py` | `tests/unit/test_data*.py`, `tests/unit/test_config.py` |
| 02-grid-risk.md | GRID, RISK, HORIZON, CAL | `core/*.py`, `hazards/base.py` | `tests/unit/test_grid.py`, `tests/unit/test_risk.py`, `tests/unit/test_policy.py`, `tests/unit/test_scenario.py`, `tests/unit/test_synthetic.py`, `tests/unit/test_calibrate.py`, `tests/unit/test_hazard_base.py` |
| 03-roads.md | ROAD | `roads/*.py` | `tests/unit/test_roads*.py` |
| 04-astar.md | ASTAR, PLAN | `routing/cost.py`, `routing/astar.py`, `routing/planner.py` | `tests/unit/test_astar.py`, `tests/unit/test_cost.py`, `tests/unit/test_planner.py` |
| 05-shelters.md | SHEL | `optimize/shelters.py` | `tests/unit/test_shelters*.py` |
| 06-evacuation.md | EVAC | `optimize/evacuation.py` | `tests/unit/test_evacuation*.py` |
| 07-emergency.md | EMS | `routing/emergency.py` | `tests/unit/test_emergency*.py` |
| 08-storm.md | STORM | `hazards/storm/*.py` | `tests/unit/test_storm*.py` |
| 09-tornado.md | TORN | `hazards/tornado/*.py` | `tests/unit/test_tornado*.py` |
| 10-flood.md | FLOOD | `hazards/flood/*.py` | `tests/unit/test_flood*.py` |
| 11-wildfire.md | FIRE | `hazards/wildfire/*.py` | `tests/unit/test_wildfire*.py` |
| 12-ice.md | ICE | `hazards/ice/*.py` | `tests/unit/test_ice*.py` |
| 13-smoke.md | SMOKE | `hazards/smoke/*.py` | `tests/unit/test_smoke*.py` |
| 14-app.md | APP, CLI | `app/*`, `cli.py` | `tests/unit/test_app*.py`, `tests/unit/test_cli*.py`, `tests/e2e/*` |
| 15-evaluation.md | EVAL, SPLIT | `eval/*.py` (incl. `eval/splits.py`) | `tests/unit/test_metrics.py`, `tests/unit/test_bootstrap.py`, `tests/unit/test_report.py`, `tests/unit/test_routing_eval.py`, `tests/unit/test_splits*.py` |
| 16-acceptance.md | AC | (none) | `tests/acceptance/*` |

`tests/unit/test_traceability.py` belongs to this overview. Shared synthetic helpers live in `tests/fixtures/`
(`graphs/graphfix.py`, `adapters.py`) and are owned by no single spec. `test_traceability.py::test_unit_test_file_ownership`
checks that every other `tests/unit/test_*.py` file matches exactly one row's patterns.

The tornado prefix is `TORN` (DESIGN 1.3 wrote `TOR-*`; `TORN` is the fixed spelling).

## Reserved IDs (meaning fixed by DESIGN)

Area specs must define these IDs with this meaning (wording and detail are up to the area spec).

| ID | Meaning |
|---|---|
| DATA-01 | SEVIR reader: CATALOG filter, VIL (NEXRAD mosaic) + GOES-16 IR107 h5 to 2 km uint8 memmaps with projection metadata (radar + satellite) |
| DATA-02 | TorNet reader: catalog.csv + NetCDF NEXRAD Level-II chips to fp16 arrays, official split rule (radar) |
| DATA-03 | HRRR reader: Herbie `.idx` byte-range field fetch, in-memory crop, f01 rule, MASSDEN unit handling (weather) |
| DATA-04 | MRMS reader: QPE 01H Pass2 + FLASH GRIB2 decode, crop, 1/3/6/24 h accumulations (radar + weather) |
| DATA-05 | NDWS TFRecord reader to npy (GRIDMET weather, MODIS/VIIRS channels) |
| DATA-06 | WSTS GeoTIFF reader + 12-channel NDWS mapper with per-channel shift statistics (weather + VIIRS) |
| DATA-07 | Weather observations: NCEI Storm Events, SPC tornado tracks, EPA AQS PM2.5, NOAA HMS smoke polygons: parse and rasterize/label |
| RISK-07 | Overlay PNG pixel values equal the values the `EdgeRiskSampler`/RiskStack sampling returns ("what you see is what routes") |
| HORIZON-01 | Beyond-horizon **decay** (storm): P(t) = P(t_h) exp(-(t - t_h)/30 min), 0 after 120 min |
| HORIZON-02 | Beyond-horizon **drop** (tornado): 0 after the horizon, blocked cells keep blocking 15 min |
| HORIZON-03 | Beyond-horizon **hold** (flood, wildfire, ice, smoke): last valid step held |
| ROAD-06 | Any US city: graph built and cached from a place name (`outrun route --place`) |
| ASTAR-01 | Equals networkx Dijkstra on random OD pairs (zero and static risk) |
| ASTAR-02 | Fewer expansions than Dijkstra |
| ASTAR-03 | Never uses blocked edges |
| ASTAR-04 | Time-varying risk with waiting: equals a brute-force time-expanded Dijkstra on 30-node synthetic graphs |
| ASTAR-05 | Returns None when the goal is cut off |
| ASTAR-06 | Query time under the target on a synthetic metro-sized graph (DESIGN target 1 s) |
| STORM-02 / FIRE-02 / SMOKE-02 | The satellite input: GOES-16 IR107 channels (+ VIL-only ablation switch) / MODIS-VIIRS channels in NDWS/WSTS / HMS smoke density (+ no-HMS ablation switch) |
| STORM-04 / TORN-04 / FLOOD-04 / FIRE-04 / ICE-04 / SMOKE-04 | The PyTorch deep-learning model of that hazard (architecture, input/output shapes, forward pass) |
| FIRE-03 | Flip/rot90 augmentation rotates the wind-direction channel consistently |
| SMOKE-03 | MASSDEN unit change (2021-12-21, ug/m3 to kg/m3) asserted and converted in code |
| SPLIT-01..SPLIT-06 | One per hazard: no day/event/zone-window overlap between train, val and test, plus a gap day at each time boundary |
| SPLIT-07 | Quarantine: case-study dates +-3 days are absent from train and val of every dataset builder |

## Proposal coverage matrix

Parsed by `tests/unit/test_traceability.py::test_proposal_matrix`: every row must list at least one ID, and every
listed ID must be defined in some spec (and therefore have a test). IDs are written in full; a range is written
`PREFIX-NN..PREFIX-MM` (inclusive, same prefix) and means every ID in it. Area specs must define at least the IDs
listed here (they may define more).

| Proposal phrase | Spec | Requirement IDs |
|---|---|---|
| deep learning | 08, 09, 10, 11, 12, 13 | STORM-04, TORN-04, FLOOD-04, FIRE-04, ICE-04, SMOKE-04 |
| real weather data | 01 | DATA-03..DATA-07 |
| real radar data | 01 | DATA-01, DATA-02, DATA-04 |
| real satellite data | 08, 11, 13 | STORM-02, FIRE-02, SMOKE-02 |
| forecast severe storms | 08 | STORM-01..STORM-08 |
| tornadoes | 09 | TORN-01..TORN-08 |
| flash floods | 10 | FLOOD-01..FLOOD-08 |
| wildfire spread | 11 | FIRE-01..FIRE-08 |
| icy roads | 12 | ICE-01..ICE-08 |
| smoke | 13 | SMOKE-01..SMOKE-08 |
| A* search | 04 | ASTAR-01..ASTAR-06 |
| optimization | 05, 06 | SHEL-01..SHEL-06, EVAC-01..EVAC-08 |
| real road maps | 03 | ROAD-01..ROAD-05 |
| safe evacuation routes | 04 | PLAN-01..PLAN-04 |
| shelter assignments | 05 | SHEL-01..SHEL-06 |
| whole-city evacuations | 06 | EVAC-01..EVAC-08 |
| emergency vehicle routes | 07 | EMS-01..EMS-06 |
| interactive map app | 14 | APP-01..APP-08, APP-16, APP-17 |
| any US city (answer to the professor) | 03 | ROAD-06 |

Specs 02 (GRID/RISK/HORIZON/CAL), 14 (CLI), 15 (EVAL/SPLIT) and 16 (AC) support every row; their IDs are not
listed here but are still traced (every defined ID needs a test).
