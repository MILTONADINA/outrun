# TASKS: build order

Work top to bottom. For each task: read the spec (and the DESIGN section) -> run its tests (red) -> implement ->
tests green -> NOTES.md row -> tick the box -> commit `T<n>: ...`. Full rules in `CLAUDE.md`.

## Progress (tick as each task is committed)
- [x] T0 (Nov 17) Scaffold: specs 00-16, stubs, red tests, traceability + proposal-matrix tests, CI
- [x] T1 (Nov 17) Fetch + manifest, size inventory, TorNet split rule; [overnight] batch A downloads
- [x] T2 (Nov 18) Core: grid, risk, policy (beyond-horizon), scenario, synthetic, calibrate
- [x] T3 (Nov 18) Readers: HRRR, MRMS, AQS, HMS, DEM; measure HRRR MB/message; [overnight] batch B
- [x] T4 (Nov 19) Roads: OKC graph, speeds, capacities, bridges, crossings, CSR export
- [x] T5 (Nov 19) Cost model + multi-label TD-A* + oracles + `outrun route --place` (checkpoint: CLI A*)
- [x] T6 (Nov 20) App skeleton on synthetic layers, offline (skeleton milestone)
- [x] T7 (Nov 20) Case-study verification; freeze splits + quarantine (SPLIT-01..07) before any training
- [x] T8 (Nov 21) Storm: SEVIR preprocessing, persistence + pySTEPS, baseline adapter; [overnight] U-Net
- [x] T9 (Nov 22) Storm: val eval, calibration, DL adapter with decay; [overnight] VIL-only ablation + TorNet prep
- [x] T10 (Nov 23) Tornado: dataset, GBM + dV baselines, CNN; [overnight] train
- [x] T11 (Nov 23) [W] Shelters: facilities + census, min-cost flow + nearest/greedy
- [ ] T12 (Nov 24) Tornado: val eval, swath adapter + motion fallback (checkpoint: storm + tornado DL live)
- [ ] T13 (Nov 24) [W] Wildfire: NDWS U-Net + baselines, WSTS mapper, demo fire
- [ ] T14 (Nov 25) Flood: features, ring labels, hard negatives, baselines; [overnight] U-Net
- [ ] T15 (Nov 25) [W] Ice: zone bags, keyword labels, physics-rule + climatology baselines
- [ ] T16 (Nov 26, Thanksgiving, light) Ice: MIL-CNN train, val eval, bridge prior, adapter
- [ ] T17 (Nov 27) Flood: val eval, no-FLASH ablation, crossing adapter
- [ ] T18 (Nov 27) [W] Smoke: CNN+MLP quantile model, baselines, spatial holdout (checkpoint: 6/6 layers)
- [ ] T19 (Nov 28) Evacuation: wave-BPR evaluator, SA, MSA, shortest/safest, SSE
- [ ] T20 (Nov 28) Wildfire scenarios: WSTS 2021 demo fire + labelled OK Mar 2025 replay
- [ ] T21 (Nov 29) EMS dispatch + shelter-in-place rule
- [ ] T22 (Nov 29) Scenario bundles for all scenarios; [overnight] build
- [ ] T23 (Nov 30) App: all tabs, offline Playwright test
- [ ] T24 (Dec 1) One-shot test evaluation, routing eval, acceptance (feature freeze)
- [ ] T25 (Dec 2) README, model cards, DATA_LICENSES, static Pages export, CI polish
- [ ] T26 (Dec 2) REPORT.md (5 rubric sections) + PDF
- [ ] T27 (Dec 3) Slides (PPTX + PDF)
- [ ] T28 (Dec 3) DEMO.md, rehearsals, narrated backup video, v1.0-rc
- [ ] T29 (Dec 4 a.m.) Clean-clone offline dry run, tag v1.0, submit before 1:40 pm

Starting state after T0: `.venv/bin/python -m pytest` -> traceability checks run; everything else fails with
`NotImplementedError` (tests are written before code).

## Rules
- **One commit per task**, message `T<n>: <summary>`, made on the real date. **No backdating.** At least one
  real-dated commit every day from Nov 17 to Dec 3, pushed the same day.
- **No `Co-Authored-By:` or other Claude/AI attribution trailer** in any commit or PR.
- A dated NOTES.md entry every day.
- Every "done when" includes: the listed spec tests green, no regressions in the full suite, and a NOTES.md entry.
- All model metrics before T24 are **validation** metrics. Test data is evaluated once, in T24, with the git sha.
- Splits are frozen in T7 (SPLIT-01..07 green) **before any training**. Case-study dates +-3 days are quarantined
  from train and val of every model.
- Never edit `tests/` or `specs/` to make code pass: stop, explain, and let Milton decide.
- `[overnight]` = a background job started before bed (with a log in `logs/`); the next day's task begins by checking it.
- `[W]` = file-disjoint task suitable for a local `worker` run. Claude reviews the diff and runs the tests twice
  before merging.
- Checkpoints (DESIGN 7.1): Nov 19 CLI A* on the real graph; Nov 20 map skeleton offline on synthetic layers;
  Nov 24 storm + tornado DL layers live; Nov 27 six baseline layers + at least four DL layers (else apply depth
  levers that day); Dec 1 feature freeze (end of day).

## Task table (copied from DESIGN.md section 8)

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
