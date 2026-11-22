# Outrun: narrated demo run-sheet (target 6:00 to 6:30)

Follow this to record the backup video (about 6:30) and to rehearse three times. Speak the slide notes in
`slides/outrun.pptx` (speaker notes on every slide; the same text is built by `scripts/make_slides.py`). The slide
notes total about 1,050 words, which is about 7:00 at 150 words per minute. To land at 6:00 to 6:30, speak at about
165 words per minute and trim slides 8 and 9 to the first two sentences if a rehearsal runs long. Log each
rehearsal time in NOTES.md.

## Setup (once, offline)
1. `cd` to the repo, `make demo` (same as `OUTRUN_DEMO=1 .venv/bin/python -m outrun serve`), open http://127.0.0.1:8000.
   No data, models or network are needed; `scenarios/` is committed. Turn Wi-Fi off to prove it.
2. Browser at 1280x720 or larger, zoom 100%, bookmarks bar hidden. Slides open in presenter view (`slides/outrun.pptx`
   or `slides/outrun.pdf`). Close notifications.
3. Pick the scenario from the top-left dropdown. The basemap tiles are not drawn in this build; say so once
   (slide 11 note) and point at the overlays.
4. Record screen plus microphone (QuickTime: File > New Screen Recording). Export MP4, then link it in README and
   slide 11 and attach it to the `v1.0-rc` release.

## Run-sheet
| Time | Slide | Say (slide note) | On screen |
|---|---|---|---|
| 0:00 | 1 Title | Proposal in one sentence | Slide only |
| 0:25 | 2 Problem | Six hazards, one question: which road is safe | Slide |
| 0:50 | 3 What I built | Forecast, plan, show | Slide |
| 1:10 | 4 Architecture | Six models, one RiskStack, one router | Slide |
| 1:30 | 5 RiskLayer / HazardPolicy | Cost = time x (1 + weighted risk) | Slide |
| 2:00 | 6 Results table | Four wins, two non-wins, one loss; held-out test, evaluated once | Slide |
| 2:35 | 7 Where DL lost | Tornado AUC 0.761 vs published 0.874; ice; WSTS | Slide |
| 3:00 | 8 Satellite | GOES IR, MODIS/VIIRS, HMS used; ablations show no help | Slide |
| 3:20 | 9 Routing and optimization | A*, min-cost-flow, SA vs MSA | Slide |
| 3:45 | 10 Gap closed | 40% (17.6 to 56.1) on May 20; about zero on ice | Slide |
| 4:10 | 11 Demo | Switch to the app (below, about 1:30) | App |
| 5:40 | 12 Honest results | Wins, non-wins, caveats | Slide |
| 6:05 | 13 Lessons | Baselines matter; a forecast must move a decision | Slide, stop by 6:30 |

## Live app segment (slide 11, about 1:30)
Scenario sources: `scenarios/may20-2019/` (primary), `scenarios/flood-okc-2025/`, `scenarios/synth-okc/` (fallback).
The numbers in the app are a single run (SA 1876 veh-h), not the report's 5-seed mean (1891.8): say so.

1. **Forecast tab**, scenario `may20-2019`. Drag the time slot slider (slots 0 to 11) or press Play. Tick
   "Forecast | Observed swipe" and drag the swipe position. Click "Model card" and close it.
2. **Safe route tab**. Click the map to set the origin (destination "Nearest shelter"), read the route table
   (time, distance, risk exposure: shortest vs safe). Move the lambda slider once.
3. **Shelters tab**. Method min-cost-flow; point at the load bars and the overflow metrics.
4. **Evacuation tab**. Click "Run optimizer" (simulated annealing runs live, the objective curve falls to about
   9,170), then "Animate waves". Compare to MSA in the method dropdown.
5. **Dispatch tab** (switch to `flood-okc-2025`). Click 1 to 3 incidents, "Dispatch units", read ETA vs baseline.
6. **Metrics tab**. One glance at the tables, then back to the slides.

If the live optimizer is slow or errors, switch to `synth-okc` or show the precomputed method (SA is also in the
bundle). `docs/demo.gif` shows the intended path.

## Rehearsal log
| Run | Total | Notes |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |
