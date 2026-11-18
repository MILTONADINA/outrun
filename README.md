# Outrun

**Status: under construction** (CMSC-5713 final project, due Dec 4 2026).

Outrun forecasts six weather hazards (severe storms, tornadoes, flash floods, wildfire spread, icy roads and
smoke) with deep-learning models trained on real weather, radar and satellite data, turns each forecast into a
calibrated risk map on one shared 1 km grid, and then plans on real OpenStreetMap roads: risk-aware evacuation
routes with a time-dependent A* search, capacity-constrained shelter assignments with min-cost flow, whole-city
evacuations with simulated annealing under congestion, and emergency-vehicle dispatch. Everything is shown in an
interactive map app that also runs fully offline.

> For my final project I propose Outrun, an AI system that uses deep learning and real weather, radar and satellite data to forecast severe storms, tornadoes, flash floods, wildfire spread, icy roads and smoke, then uses A* search and optimization on real road maps to plan safe evacuation routes, shelter assignments, whole-city evacuations and emergency vehicle routes, demonstrated through an interactive map app.
>
> (approved project proposal)

## Quickstart

_Placeholder, completed in T25._

```bash
make install        # Python 3.12 venv + pip install -e ".[dev]"
make test           # unit tests (synthetic data only, offline)
make demo           # offline map app on precomputed scenarios
```

## Data location

Downloads (RAW) and processed data (PROC) live outside the repo. On the project Mac both are on the internal SSD:

```bash
export OUTRUN_RAW=/Users/milto/OutrunData/raw     # raw downloads, read once, never modified
export OUTRUN_PROC=/Users/milto/OutrunData/proc   # memmaps, zarr, parquet, graphs
```

Put these lines in `~/.zshrc` (or set them per command). Without them the code falls back to `/Volumes/OutrunRaw`
and `data/proc`. Disk caps are in `configs/disk.yaml`. The NDWS dataset comes from Kaggle and needs an API
token in `~/.kaggle/kaggle.json` (Kaggle -> Settings -> API -> Create New Token, then `chmod 600` the file).

## Docs

- `docs/DESIGN.md`: full design
- `specs/`: numbered, tested requirements (`specs/00-overview.md` has the proposal coverage matrix)
- `TASKS.md`: build order and progress
- `REPORT.md`: final report
- `DATA_LICENSES.md`: data sources and licences

## License

Code: MIT (see `LICENSE`). Data: see `DATA_LICENSES.md`.
