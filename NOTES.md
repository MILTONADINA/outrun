# NOTES: running log

Log anything you'd want in the report while it's fresh: problems, fixes, decisions and why, numbers.
At least one dated row every day from Nov 17 to Dec 4.

| Date | Task | What happened / decided | Why / numbers |
|---|---|---|---|
| 2026-11-17 | T0 | Scaffold: pyproject, Makefile, CI, CLAUDE.md, TASKS.md, configs, manifest skeleton, specs 00-16 with req IDs, stubs, red tests, traceability + proposal-matrix tests | Python 3.12 venv; `pip install -e .[dev]` clean, no brew needed (cfgrib uses the bundled eccodes 2.50 wheel). Got torch 2.15.0 (MPS available), numpy 2.5, pandas 3.1, xarray 2026.9, **zarr 3.4** (v3 API), osmnx 2.1, ortools 9.15, geopandas 1.2, rasterio 1.5, herbie 2026.10. Several are a major version newer than DESIGN assumed (zarr 3, pandas 3, osmnx 2): watch for API changes. pySTEPS and OpenCV (DESIGN 2.3 storm baseline) are not yet dependencies; add in T8 if used. Traceability: 7 checks pass; `test_all_spec_files_exist` and `test_proposal_matrix` stay red until specs 01-16 exist (expected) |
| 2026-11-17 | T1 (inventory) | Size inventory, availability checks and TorNet split rule; `data/manifest.yaml` sizes/URLs updated to `[MEASURED Nov 17]`. Storage: no external HDD; RAW root `/Users/milto/OutrunData/raw`, PROC root `/Users/milto/OutrunData/proc`, both on the internal SSD (568 GB free; caps 300 + 150 GB in `configs/disk.yaml`). Details in the T1 section below | Batch A = 89.47 GB (83.3 GiB). Manifest RAW ("hdd") planned total 172.9 GB of 300 GB cap |
| 2026-11-18 | T1 (batch A) | Overnight batch A downloads finished and checked | 82 GB in `$OUTRUN_RAW`: SEVIR IR107 + VIL 2018-19 (31 GB), TorNet 2019/2015/2016 (51.9 GB), NDWS 2.1 GB, OSM Oklahoma 163 MB, catalogs, SPC. S3 ~5 MB/s, Zenodo ~3.2 MB/s. Zenodo dropped the TorNet 2016 stream at 91% (IncompleteRead), then 4 read timeouts; resumed from `.part` each time. All 3 TorNet tarballs match Zenodo md5 and exact byte sizes |
| 2026-11-18 | Decision | TORN-02 split labels: Milton chose the TorNet **catalog `type` column** (official split can't be reproduced from date alone: 35 mixed days). Spec/test amendment for TORN-02 is done in T7, before splits freeze | |
| 2026-11-18 | T2 | Core: `grid.py` (EPSG:5070 1 km grid, windows, nearest regrid via cKDTree with max distance, point aggregation), `risk.py` (RiskLayer/RiskStack, zarr v3 storage, overlays read through the same `get()` as `sample`, current-only source), `policy.py` (beyond-horizon hold/decay/drop), `scenario.py`, `synthetic.py`, `calibrate.py` (isotonic via PAV knots + temperature scaling via bounded Brent on log T), `hazards/base.py` generic registry + plain PyTorch training loop (early stopping, atomic checkpoints, RNG state for exact resume). Fetch hardening: `download(max_retries=...)` retries transient network errors and 429/5xx (incl. botocore stream drops) with capped exponential backoff, counter resets on progress; `fetch_source` defaults to 8 retries. New `http_index_lister` for Apache-style index pages (Storm Events, AQS). Docs: storage wording now internal SSD; DESIGN 7.3 notes PROC worst-case 165 GB > 150 GB cap, plan = process TorNet half a year at a time (≈146 GB peak) | T2 target files: 185 passed, 1 failed (`test_load_default_adapters_registers_all_six` needs the six hazard adapters from T9-T18; expected). Full suite vs clean T1 checkout: +103 passing, 0 new failures; all remaining failures are later-task stubs. New `tests/unit/test_data_fetch_resilience.py` (offline fakes). Built with parallel agents + 4 independent reviewers (spec, integrity/leakage, fetch, docs); 9 review issues, 7 fixed (NaN coords in regrid, fractional nx/ny, scalar/array mix, S3 retry gap, 5xx retry, DESIGN percentages, spec 00 storage lines) |

## T1 inventory, 2026-11-17

Sizes are exact bytes from `aws s3 ls --no-sign-request`, the Zenodo API, the Kaggle API, or HTTP `Content-Length`, all checked on 2026-11-17.

| Source | Files | Bytes | GB | In batch A |
|---|---|---|---|---|
| SEVIR CATALOG.csv | 1 | 33,838,047 | 0.03 | yes (downloaded) |
| SEVIR VIL 2018-19, whole prefixes (incl. RANDOMEVENTS) | 10 | 116,870,247,054 | 116.87 | no |
| SEVIR VIL 2018-19 STORMEVENTS only | 4 | 21,907,289,712 | 21.91 | yes |
| SEVIR IR107 2018-19, whole prefixes (incl. RANDOMEVENTS) | 10 | 48,919,451,010 | 48.92 | no |
| SEVIR IR107 2018-19 STORMEVENTS only | 4 | 11,398,028,094 | 11.40 | yes (below the 60 GB rule, so no file is dropped) |
| TorNet catalog.csv (record 12636522) | 1 | 38,245,403 | 0.04 | yes (downloaded, md5 OK) |
| TorNet 2019 / 2015 / 2016 tarballs (records 12655716 / 12655151 / 12655179) | 3 | 18,219,887,934 + 17,395,624,045 + 16,245,523,516 = 51,861,035,495 | 51.86 | yes |
| NDWS (Kaggle `fantineh/next-day-wildfire-spread`) | 19 tfrecords | 3,955,889,585 | 3.96 | yes |
| Storm Events details 2019-2026 | 8 .csv.gz | 91,829,403 | 0.09 | yes |
| SPC `1950-2025_actual_tornadoes.csv` | 1 | 9,023,445 | 0.01 | yes |
| Geofabrik `oklahoma-latest.osm.pbf` | 1 | 170,661,998 | 0.17 | yes |
| WSTS (Zenodo 8006177, latest version): `WildfireSpreadTS.zip` + documentation PDF | 2 | 48,359,369,821 + 7,841,847 | 48.37 | no (T13) |
| AQS hourly_88101 2022-2025 | 4 | 287,740,544 | 0.29 | no |
| **Batch A total (STORMEVENTS only)** | | **89,465,841,182** | **89.47** | (would be about 222 GB with whole SEVIR prefixes) |

- **SEVIR VIL/IR107 overlap.** Counted on CATALOG `id` for STORMEVENTS rows (`event_id` not null), 2018-2019. 2,591 ids have both `vil` and `ir107` (1,518 in 2018, 1,073 in 2019). 441 ids are VIL-only and 564 are IR107-only. That means 85.5% of the 3,032 VIL storm events also have IR. All 3,155 matching IR107 rows are in STORMEVENTS files. Counted over all ids, including RANDOMEVENTS: 12,905 both and 3,181 VIL-only. Per DESIGN 3.1, the 441 VIL-only events are dropped from both arms.
- **Projected RAW use against the 300 GB cap**, using measured values plus the DESIGN estimates still open:
  - SEVIR: 33.3 GB
  - TorNet: 51.9 GB
  - NDWS: 4.0 GB
  - WSTS: 48.4 GB
  - MRMS QPE + FLASH: 30 GB (estimate, T3)
  - ETOPO + 3DEP: 3.6 GB (estimate)
  - Small sources: about 1.7 GB

  The total is about 173 GB, which is 58% of the cap. No cut from the DESIGN 7.3 cut order is needed.

**TorNet split rule**
- **Repo (mit-ll/tornet @ 92d9d97, 2025-05-27).** The repo has no function that computes the split. `query_catalog(data_root, data_type, years, random_state, catalog=None)` in `tornet/data/loader.py` only filters on `catalog['type']==data_type` and the year, then shuffles. The torch, keras and tf loaders and the tfds builder (`catalog['type']==data_type`, with `data_type` taken from the `train/` or `test/` parent folder) all use the same precomputed labels. The split ships as the `type` column of `catalog.csv` and as the `train/` and `test/` folders. The filename prefix matches `type` in 100% of rows.
- **Paper (Veillette et al., arXiv 2401.16437, section 3.1).** This is the paper's description, not code. Each sample takes the start time `t_e` of its NSED storm *episode*. A sample is train if `J(t_e) mod 20 < 17` and test if it is `>= 17`, where J is the Julian day. Then "any training sample that was within 30 minutes and within 0.25 degrees of latitude/longitude of any test sample" was removed. Totals: 171,666 train and 31,467 test, which matches the catalog exactly.
- **Measured.** The catalog has no episode start time. Using chip `start_time` (UTC, day of year 1..366) as a proxy, `doy % 20 >= 17` mismatches 1,512 of 203,133 rows (0.74%), 333 of them in 2015/16/19. The offset was tried at -12..+12 h and 0 h is best. Exact reproduction from episode start times is NOT VERIFIED.
- **Against the specs:**
  - DATA-02: the `SplitRule(mode="column", column="type")` default is the verified rule. `mode="julian"` (20/17) is only an approximation, so keep the column default.
  - SPLIT-02: compatible. The official label is a per-chip input and the function never recomputes it.
  - TORN-02: **possible spec issue, reported, not edited.** `split_catalog(catalog, is_test, ...)` derives `official` from `is_test(date)`, a date-only predicate. In 2015/16/19, 35 UTC calendar days hold both train and test chips (4,525 rows), so no date predicate can reproduce the official labels on those days. A fix needs Milton's decision: let `split_catalog` take the per-row `type` (or `official_split` output) instead of `is_test(date)`.

**Availability**
- **Storm Events details.** All 8 files 2019-2026 exist (Nov 17 listing): d2019-d2021 and d2023 c20260323, d2022 c20260625, d2024 c20260728, d2025 c20260819, d2026 c20261026. The 2026 file is a partial year by construction. Completeness of 2025 and 2026 is NOT VERIFIED.
- **EPA AQS PM2.5 (88101).** `hourly_88101_2025.zip` (68,558,274 B) and `daily_88101_2025.zip` (8,647,326 B) return HTTP 200, and both are listed on download_files.html. Hourly and daily 2026 files also exist (hourly 16.4 MB). All hourly files show Last-Modified 2026-07-13. 2025 completeness is NOT VERIFIED.
- **SPC.** `wcm/data/1950-2025_actual_tornadoes.csv` (9.0 MB) and `1950-2025_all_tornadoes.csv` (9.2 MB) are available, Last-Modified 2026-04-24. Daily storm-report CSVs (`climo/reports/YYMMDD_rpts_torn.csv`) return 200 for the dates sampled (2025-04-28, 2026-05-01). Coverage of every date is NOT VERIFIED.
- **Geofabrik.** `oklahoma-latest.osm.pbf` is 170,661,998 B (rebuilt daily).
- **Kaggle.** The token exists at `~/.kaggle/access_token`, and the venv kaggle 2.2.5 CLI authenticated: NDWS listing OK.
- **WSTS.** There is one 48.36 GB zip, so a 2021-only download is impossible. The internal layout is NOT VERIFIED (T13).
- **SEVIR licence.** Still `[CANDIDATE]`, NOT VERIFIED.

**fetch.py notes for batch A**
- SEVIR URLs are now exact-file brace URLs, checked with `expand_braces` and `s3_lister`. With the old directory URLs, the fetch would have pulled RANDOMEVENTS too, about 166 GB.
- TorNet uses `https://zenodo.org/records/{12655716,12655151,12655179}/files/`, checked with `zenodo_lister`.
- `storm_events` and `aqs_pm25_hourly` are http directory URLs, and `default_lister` only handles s3 and zenodo. These sources cannot be fetched as written. Fetch them by explicit file names, or add an http lister.
- The two catalogs are already in the RAW root: `sevir/CATALOG.csv` (hardlinked to `sevir_catalog/CATALOG.csv`) and `tornet_catalog/catalog.csv`.
- Batch A needs `OUTRUN_RAW=/Users/milto/OutrunData/raw` and `OUTRUN_PROC=/Users/milto/OutrunData/proc` exported. The `outrun.config` defaults still point to `/Volumes/OutrunRaw` and `data/proc`.
- CLAUDE.md, DESIGN 7.3 and DATA-14's default caps (375/225) are stale. `configs/disk.yaml` overrides them to 300/150.
