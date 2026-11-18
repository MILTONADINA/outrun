# Outrun: Report

_Skeleton. Filled in T26 from NOTES.md and `reports/metrics.md`. Every number comes from a logged run;
test metrics come only from the single T24 evaluation (git sha recorded)._

## 1. What I built and why

_The proposal, the six hazards, the shared RiskLayer/RiskStack/HazardPolicy design, the planners, the app.
Name the satellite inputs explicitly (GOES-16 IR107 in SEVIR, MODIS/VIIRS in NDWS/WSTS, NOAA HMS smoke)._

## 2. Technologies used and why (vs alternatives)

_PyTorch on MPS, U-Nets/CNNs, own time-dependent A*, OR-Tools min-cost flow, simulated annealing + BPR,
xarray/zarr, FastAPI + vanilla MapLibre + PMTiles. Each with the alternative considered and why it lost._

## 3. Problems I faced and how I solved them (or didn't)

_Data volume and tooling, weak labels, leakage and shortcuts, exact time-dependent search, offline demo.
Include results tables (models vs baselines with CIs, routing gap closed) and say plainly where a deep model
did not beat its baseline._

## 4. What I learned

_..._

## 5. How I would improve it

_..._

## Tools used

_Specs, tests and much of the code were produced with AI assistance (Claude Code; local Qwen workers for some
mechanical tasks). Milton made the decisions and can explain every part._
