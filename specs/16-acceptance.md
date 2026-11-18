# 16: Acceptance criteria (real data, trained models)

Module: none (reads the files `outrun eval` writes in T24). Tests: `tests/acceptance/test_acceptance.py`.
Design: DESIGN 6.1-6.3, 7.2, 1.3 ("any US city" row).

Run after the one-shot test evaluation (T24):

```
OUTRUN_ACCEPTANCE=1 .venv/bin/python -m pytest tests/acceptance -m acceptance
```

Without `OUTRUN_ACCEPTANCE=1` every test is **skipped**. With it, a missing input file (metrics not computed yet,
splits not frozen) is also a skip with a message naming the file and the command that produces it, never a
false failure. Paths: `reports/` and `splits/` at the repo root (override with `OUTRUN_REPORTS` and
`OUTRUN_SPLITS`).

What acceptance checks (DESIGN 6.3): that every model's metrics **exist, were computed on the official test split
and the case hold-out set, carry CIs and a git sha, and were logged once**. It does **not** require a deep model
to beat its baseline: the proposal promises a deep-learning forecast, not one that beats every baseline.
"Beats baseline", the routing exposure reduction and the latency numbers are **Target** rows: their tests
assert the inputs exist and then compare; a miss is reported as `xfail` with the real numbers in the reason
(visible in the `-ra` summary), so it is recorded rather than hidden or loosened. Copy those numbers and the
likely cause into NOTES.md and the report.

Tiers: **Must** = required to ship the proposal (failure = fix before submission). **Target** = honest number
the report states either way.

| ID | Criterion | Target | Tier |
|---|---|---|---|
| **AC-01** | Storm: `reports/metrics/storm.json` exists (EVAL-17 schema) with a `test` split holding the DL model and both baselines (persistence, pySTEPS), primary metric `csi_mean_74_133` with a CI from >= 1,000 day-block resamples, and a non-empty git sha | exists | Must |
| **AC-02** | Tornado: same for `tornado.json` (DL CNN + GBM baseline; primary `auc_tor_vs_warned_null`) | exists | Must |
| **AC-03** | Flood: same for `flood.json` (DL U-Net + FFG-ratio baseline; primary `auprc`) | exists | Must |
| **AC-04** | Wildfire: same for `wildfire.json` (primary `prauc`), plus the WSTS 2021 out-of-time split `test_wsts2021` | exists | Must |
| **AC-05** | Icy roads: same for `ice.json` (primary `zone_auprc`) | exists | Must |
| **AC-06** | Smoke: same for `smoke.json` (primary `rmse_smoke_episodes`, lower is better) | exists | Must |
| **AC-07** | Case hold-out reported separately: at least one hazard metrics file has a `case_holdout` split, and every `case_holdout` split present has a DL and a baseline row with finite primary estimates (CI bounds may be NaN for a short case window) | exists | Must |
| **AC-08** | Satellite ablations reported: `storm.json` has ablation `vil_only` and `smoke.json` has ablation `no_hms` (DESIGN 1.3 satellite row) | exists | Must |
| **AC-09** | Each DL model beats its best baseline on the primary metric of its official test split (per hazard; `reference` rows excluded) | >= best baseline | Target |
| **AC-10** | Routing evaluation of the primary case: a `routing_<id>.json` with `"primary": true`, all five planners (a, a', b, c, d), >= 100 paired OD pairs (500 planned; 100 is the depth lever), and a finite `gap_closed` CI | exists | Must |
| **AC-11** | Routing exposure reduction: DL planning (c) closes a positive share of the observed-exposure gap between (a) and (d) (`gap_closed.estimate > 0`), and the oracle (d) has no more mean observed exposure than (a) | > 0 | Target |
| **AC-12** | A* latency: `latency.json` `astar_metro_query.median_s` on the real OKC metro graph | < 1.0 s | Target |
| **AC-13** | SA evacuation run time: `latency.json` `sa_run.median_s` | < 60 s | Target |
| **AC-14** | Held-out discipline: `splits/hashes.json` verifies against every frozen split (`verify_hashes == []`), and `reports/test_ledger.json` has a `test` entry for every hazard model in the metrics files whose git sha matches that file's sha | exact | Must |
| **AC-16** | Shelter assignments on real roads: the primary case's `group_<id>.json` (EVAL-22, `"primary": true`) has `mcf`, `nearest` and `greedy` shelter metrics, and MCF overflow <= greedy overflow (both 0 when capacity is respected) and MCF unserved <= greedy unserved | exists | Must |
| **AC-17** | Whole-city evacuation on real roads: the same file has `sa`, `msa`, `shortest`, `safest` and `sa_lambda0` with 5 seeds each, a finite SA clearance time on every seed, and `lower_bound <= ` the mean SA `J` (within 1e-6 relative) | exists | Must |
| **AC-18** | Emergency vehicle routes on real roads: the same file has `hungarian` and `reroute_baseline` EMS metrics with finite true-ETA means and a non-empty Pareto curve | exists | Must |
| **AC-19** | Interactive map app: the primary case's scenario bundle `scenarios/<id>/` (id = the primary `routing_<id>.json` scenario) exists with `scenario.json`, at least one overlay PNG and `results/routes_preset.json`; `static/vendor/` holds the APP-17 libraries; and `create_app(scenarios_dir=scenarios, backend=None)` serves `GET /api/scenarios` listing it, that overlay PNG, and `POST /api/route` for its first preset (APP-11), all with `200` | exists | Must |
| **AC-15** | Any US city: `python -m outrun route --place "Tulsa, OK" --origin LON LAT --dest LON LAT --risk-source none` (CLI-05) and the same for one non-Oklahoma city (Lawrence, KS) each exit 0 and print a route (needs OSM access or a cached graph; skipped with a message otherwise) | exit 0 | Must |

Notes:

- AC-01..07 check CIs only on `dl` and `baseline` rows; `reference` rows (published numbers such as TorNet
  0.874 or NDWS 0.284) have no CI. The CI must have finite, ordered bounds; a percentile interval need not contain
  the point estimate, so that is not asserted.
- AC-01..06 accept extra baselines and metrics. The baseline names in the table are the ones DESIGN 6.1 plans;
  the test requires at least one `baseline`-kind model, so a renamed baseline is not a failure.
- AC-09 uses the `test` split only. Val numbers are in NOTES.md from the training tasks and are not re-checked.
- AC-16..18 read `reports/metrics/group_*.json` (exactly one with `"primary": true`); a miss is a Must failure,
  because the proposal promises shelter assignments, whole-city evacuations and emergency vehicle routes on real
  road maps. AC-19 runs the app in-process (`fastapi.testclient`), no browser; the browser check is APP-16 (e2e).
- AC-12 and AC-13 come from `outrun eval` timing 20 A* queries (after 3 warm-up queries) and 5 SA runs with
  `eval.report.time_calls`, on the demo laptop.
