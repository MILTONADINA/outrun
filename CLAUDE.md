# Outrun: instructions for Claude Code

AI-course (CMSC-5713) final project, due **Fri Dec 4 2026, 1:40 pm**. Student and decision-maker: Milton Adina.
Six hazard forecasters (storm, tornado, flash flood, wildfire, icy roads, smoke) produce calibrated `RiskLayer`s on one
1 km grid. A time-dependent multi-label A*, min-cost-flow shelter assignment, simulated-annealing city evacuation and
emergency-vehicle dispatch plan on real OSM roads. A FastAPI + MapLibre app shows it all, offline.

`docs/DESIGN.md` is the full design. `specs/00-overview.md` is the short version plus the proposal coverage matrix.

## This repo is spec-driven
- `specs/` is the source of truth: numbered requirements (`ASTAR-03`, `SHEL-02`, ...), one file per area (00-16).
  The ID prefixes per spec file are fixed in `specs/00-overview.md`. Read the relevant spec before writing code.
- `tests/` encodes every requirement with `@pytest.mark.req("ID")`. Tests were written first and are passable by a
  sound implementation of the spec. `tests/unit/test_traceability.py` checks that every spec ID has a test, that no
  test cites an unknown ID, and that every proposal phrase in the coverage matrix maps to defined IDs.
- `src/outrun/` starts as **stubs**: signatures, docstrings and dataclasses are the contract. Replace the
  `raise NotImplementedError` bodies. Don't rename or change public signatures.
- Core interfaces (`Hazard`, `RiskLayer`, `RiskStack`, `HazardPolicy`, `HazardAdapter`, `Scenario`, `GRID`, `Profile`)
  live in `outrun.core.*` and match DESIGN section 1.2. Import them from there; never redefine them.
- `TASKS.md` is the build order (T0-T29) with a progress checklist at the top.

## Workflow for every task
1. Find the first unchecked task in `TASKS.md`. Read its spec section(s) and the matching DESIGN section.
2. Run that task's tests and confirm they fail.
3. Implement in `src/outrun/` only.
4. Run the task's tests until they pass, then the full suite: `.venv/bin/python -m pytest -q` (no regressions).
5. Add a dated row to `NOTES.md`: what you did, problems and fixes, decisions and why, numbers (sizes, counts,
   val metrics, timings).
6. Tick the task in the `TASKS.md` checklist.
7. Commit: **one commit per task**, message `T<n>: <summary>`, made on the real date (no backdating, ever).
   At least one real-dated commit every day from Nov 17 to Dec 3. The commit history is graded.
8. Stop and report: tests passing, what changed, anything Milton should know. Continue only when asked.

## Hard rules
- **Never add `Co-Authored-By:` lines, "Generated with Claude" footers or any other Claude/AI attribution trailer
  to commits or PRs.**
- **Never edit `tests/` or `specs/` to make something pass.** If a test or spec looks wrong, stop, explain why, and
  propose the change. Milton decides.
- **Never train, tune, calibrate, early-stop or pick thresholds on test data or on case-study quarantine dates**
  (`splits/quarantine.json`). Fit on train, select on val. Test metrics are computed **once**, in T24, logged with
  the git sha. No task before T24 reports test metrics.
- Splits (`splits/*.json`) are frozen in T7 before any training and are protected by the SPLIT-* tests.
- Never commit `data/`, `models/`, `logs/`, `scenarios/**/*.zarr`, `*.mp4` or `.venv/` (see `.gitignore`).
  `data/manifest.yaml` is the only committed file under `data/`.
- Facts not yet verified are tagged `[CANDIDATE]` in DESIGN/specs with a verification task and a fallback. Don't
  state them as facts in the report until verified (record the verification in NOTES.md).
- Keep code readable and explainable: Milton must be able to explain every part in the demo and the report.
  Plain PyTorch training loops, no hidden frameworks.

## Environment
- Python 3.12 venv at `.venv` (`/opt/homebrew/bin/python3.12 -m venv .venv`, then `make install`).
- **Always** go through the venv module form: `.venv/bin/python -m pytest`, `.venv/bin/python -m outrun <cmd>`,
  `.venv/bin/python -m pip install ...`. The project path contains a space, so the `.venv/bin/*` console scripts can
  break; shell activation doesn't persist between commands either.
- Device: Apple silicon `mps` for training (CPU in CI). Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for training jobs.
- `cfgrib` needs eccodes (`brew install eccodes`); pygrib is the documented fallback.

## Data and disks
- **RAW** downloads live on the external HDD, default `/Volumes/OutrunRaw`, configurable with `OUTRUN_RAW`
  (see `outrun.config`). Raw files are read once and never modified.
- **PROC** (processed memmaps, zarr, parquet, graphs, scenarios) lives on the internal SSD, default `data/proc`,
  configurable with `OUTRUN_PROC`.
- `data/manifest.yaml` lists every source with URL, approximate bytes, sha256, licence and target disk.
- Disk budget (DESIGN 7.3): HDD at most 375 GB, SSD at most 225 GB. Run `outrun disk` before every big job.
- Tests never touch real data or the network: synthetic fixtures only (tiny graphs, 32x32 grids, fake arrays).

## Long-running commands
- Downloads, preprocessing and training run in the background with a log, for example
  `mkdir -p logs && nohup .venv/bin/python -m outrun train storm > logs/train_storm.log 2>&1 &`,
  then check with `tail -n 20 logs/train_storm.log`. Never poll in a loop.
- Always time one epoch before an overnight run. Checkpoint every epoch with a resume flag.
- `[overnight]` tasks in TASKS.md start before bed; the next day's task begins by checking that log.

## NOTES.md
- Add at least one dated row **every day** (Nov 17 - Dec 4), even on light days.
- Record numbers while fresh: sizes, counts, val metrics, epoch times, fallbacks taken and why.

## Acceptance (after T24)
`OUTRUN_ACCEPTANCE=1 .venv/bin/python -m pytest tests/acceptance -m acceptance`. If a target is missed, don't
loosen it: record the real number and the likely cause in NOTES.md for the report.

## Report and honesty
`REPORT.md` has the five sections the rubric grades. Fill it from NOTES.md and `reports/metrics.md`. Say plainly
where a deep model does not beat its baseline. Specs, tests and much of the code were produced with AI assistance
(Claude), so keep the "Tools used" section accurate.
