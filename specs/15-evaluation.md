# 15: Evaluation, splits and report

Modules: `src/outrun/eval/metrics.py`, `eval/bootstrap.py`, `eval/splits.py`, `eval/routing_eval.py`,
`eval/casestudy.py`, `eval/report.py` (`eval/truth.py` is a placeholder, see "Open points").
Tests: `tests/unit/test_metrics.py`, `test_bootstrap.py`, `test_splits.py`, `test_routing_eval.py`, `test_report.py`.
Design: DESIGN 3 (shared definitions: quarantine, split-overlap tests), 6.1-6.4.

Purpose: honest numbers. Every hazard model is scored with the same metric code against its baseline, with
day-block bootstrap CIs, on splits that were frozen before training and that provably do not leak. Routes are
planned on a forecast and scored on observed data. The report states plainly where a deep model loses.

This module depends on no other `outrun` area at import time. The router is injected as a callable, and risk
arrives as plain `(n_edges, n_slots)` arrays, so every requirement here is testable on synthetic data.

Tier: **Must** = needed to ship the proposal (graded evidence or held-out discipline). **Should** = needed for
the report's depth. Every requirement below is tested either way.

## Shared conventions (apply to every requirement)

- Binary labels are `{0, 1}`. Label `-1` means "uncertain" (flood ring, NDWS `-1`) and is dropped wherever labels
  are accepted. An optional boolean `mask` of the same shape keeps only `True` cells.
- A ratio whose denominator is 0 returns `float("nan")`, never raises.
- An event is `value >= threshold` (inclusive). Smoke "CSI > 35.4" is computed as `threshold=35.5`.
- Arrays of different shapes raise `ValueError`.
- "Day" is the UTC calendar day of a sample (`datetime.date`). All input datetimes must be timezone-aware;
  naive datetimes raise `ValueError`.
- Seeded functions use `numpy.random.default_rng(seed)` and give identical results for identical inputs and seed.
  Tests assert properties and same-seed determinism, never particular random draws.

## Requirements: metrics (`eval/metrics.py`)

### EVAL-01 Contingency table and CSI (Must)
`contingency(forecast, observed, mask=None) -> Contingency` takes two boolean arrays of one shape and counts
`hits` (F and O), `misses` (not F, O), `false_alarms` (F, not O) and `correct_negatives` over the cells where
`mask` is `True` (all cells if `None`). `Contingency.csi = hits / (hits + misses + false_alarms)`,
`.pod = hits / (hits + misses)`, `.far = false_alarms / (hits + false_alarms)`, each `nan` when its denominator
is 0. `csi(forecast, observed, threshold, mask=None) -> float` binarises both real-valued fields with
`>= threshold` and returns the contingency CSI (so a field with no events in either array gives `nan`).
Shape mismatch raises `ValueError`.

### EVAL-02 CSI averaged over leads (Must)
`csi_over_leads(forecast, observed, thresholds, lead_axis=0) -> dict` for arrays with a lead axis (storm:
`(L, H, W)`, or `(N, L, H, W)` with `lead_axis=1`). For each threshold it pools the contingency table over every
axis except `lead_axis` and returns `{"per_lead": {thr: ndarray (L,)}, "mean": {thr: float}, "mean_all": float}`.
`mean[thr]` is the `nanmean` of the per-lead CSIs (all-`nan` gives `nan`); `mean_all` is the `nanmean` of
`mean` over thresholds. This is the storm headline "CSI-74/133 averaged over leads" (`thresholds=(74, 133)`).

### EVAL-03 Neighbourhood CSI (Should)
`neighbourhood_csi(forecast, observed, threshold, radius) -> float` on 2-D fields. With F and O the binarised
fields (`>= threshold`) and N(i) the square `(2*radius+1)^2` window around cell i clipped to the field:
`hits` = observed event cells with at least one forecast event in N(i); `misses` = observed event cells with
none; `false_alarms` = forecast event cells with no observed event in N(i). Returns
`hits / (hits + misses + false_alarms)` (`nan` if 0). `radius=0` equals `csi`. Negative `radius` raises
`ValueError`. (Flood neighbourhood CSI, DESIGN 6.1.)

### EVAL-04 Fractions skill score (Must)
`fss(forecast, observed, threshold, window) -> float`. Binarise both with `>= threshold`; compute event
fractions with a `window x window` moving mean over the last two axes, zero padding outside the field
(`scipy.ndimage.uniform_filter(..., mode="constant", cval=0)`), so every fraction is (events in window)/window^2.
Then `FSS = 1 - sum((Pf - Po)^2) / (sum(Pf^2) + sum(Po^2))` pooled over all cells (and over leading axes for
3-D input). Denominator 0 (no events anywhere) gives `nan`. `window` must be a positive odd integer, else
`ValueError`. Identical fields give 1.0; disjoint single-cell events with `window=1` give 0.0.

### EVAL-05 ROC AUC and AUPRC (Must)
`roc_auc(labels, scores, mask=None)` and `auprc(labels, scores, mask=None)` drop `-1` labels and masked-out
cells, then equal `sklearn.metrics.roc_auc_score` and `sklearn.metrics.average_precision_score` (step-wise
average precision, not the trapezoid) on what remains, to 1e-12. If only one class remains, `roc_auc` is `nan`;
if no positives remain, `auprc` is `nan`. NaN in `scores` raises `ValueError`. Labels outside `{-1, 0, 1}` raise
`ValueError`. (Tornado AUC/AUPRC, flood AUPRC, wildfire PR-AUC, ice zone AUPRC.)

### EVAL-06 Brier score and Brier skill score (Must)
`brier(labels, probs, mask=None) = mean((p - y)^2)` over kept cells (`-1` labels dropped). Probabilities outside
`[0, 1]` raise `ValueError`. `brier_skill_score(labels, probs, ref_probs=None, mask=None) = 1 - BS / BS_ref`,
where `BS_ref` is the Brier score of `ref_probs` (a scalar or an array) on the same cells; `ref_probs=None`
means climatology, the constant base rate `mean(y)` of the kept labels. `BS_ref == 0` gives `nan`.
(This is the evaluation copy; `core/calibrate.py` (spec 02, CAL-01) owns calibration. Both use the same
formula and agree numerically where both are defined; they differ only at `BS_ref == 0`, where CAL-01 raises and
this function returns `nan` so a report is never aborted.)

### EVAL-07 Reliability curve and ECE (Must)
`reliability_curve(labels, probs, n_bins=10, mask=None) -> ReliabilityCurve` uses equal-width edges
`linspace(0, 1, n_bins + 1)`. Bin b holds `edges[b] <= p < edges[b+1]`, except the last bin, which also holds
`p == 1.0`. Per bin: `counts[b]`, `mean_pred[b] = mean(p)`, `obs_freq[b] = mean(y)`, with `nan` for empty bins.
`ece(labels, probs, n_bins=10, mask=None) = sum_b (counts[b] / N) * |mean_pred[b] - obs_freq[b]|` over non-empty
bins (`nan` if N = 0). `n_bins < 1` raises `ValueError`.

### EVAL-08 RMSE, MAE and quantile coverage (Must)
`rmse(y_true, y_pred, mask=None)`, `mae(...)` and `quantile_coverage(y_true, q_pred, mask=None)` drop every pair
in which either value is NaN (missing monitors) or the mask is `False`. `quantile_coverage` is the fraction of
kept pairs with `y_true <= q_pred` (smoke q90 coverage). No kept pairs gives `nan`.

### EVAL-09 Route exposure scoring (Must)
`score_route(trace, edge_risk, edge_blocked=None, slot_s=900, threshold=0.5, crossing_edges=frozenset(),
node_risk=None) -> RouteScore`. `trace` is a `RouteTrace(edges, t_enter_s, t_exit_s, t_start_s=0.0,
wait_nodes=None)` with every time in seconds after the **scenario t0** (the slot clock), so a route that leaves at
`t_start_s > 0` is scored against the slots it actually drives through; `edge_risk` is `(n_edges, n_slots)` risk
in `[0, 1]`; `edge_blocked` is a boolean array of the same shape (`None` = never); `node_risk` (optional) is
`(n_nodes, n_slots)` risk in `[0, 1]`. Slot k covers `[k*slot_s, (k+1)*slot_s)`; the last slot extends to
infinity.

- Edge i's **occupancy** interval is `[s_i, t_exit_s[i])` with `s_0 = t_start_s` and `s_i = t_exit_s[i-1]`. It
  splits into the **wait** `[s_i, t_enter_s[i])`, spent at the edge's tail node, and the traversal.
- The wait is charged at `node_risk[wait_nodes[i], k]` when both `node_risk` and `trace.wait_nodes` are given,
  otherwise at `edge_risk[e_i, k]`; the traversal is always charged at `edge_risk[e_i, k]`:
  `exposure_risk_min = sum_i sum_k (wait_risk_i[k] * |wait_i ∩ slot k| + edge_risk[e_i, k] * |traversal_i ∩ slot
  k|) / 60` (risk-minutes). So a planner that waits at a safe node is credited for it.
- Edge i's **traversal** interval is `[t_enter_s[i], t_exit_s[i])`; a zero-length traversal touches only the slot
  containing `t_enter_s[i]`. `intersects` is `True` iff some traversal touches a slot where
  `edge_risk >= threshold` or `edge_blocked`.
- `blocked_crossings` counts edges i with `e_i in crossing_edges` that are blocked in some touched slot.
- `travel_time_s = t_exit_s[-1] - t_start_s` (0 for an empty route).
- Validation (`ValueError`): the three tuples (or a given `wait_nodes`) differ in length, `t_enter_s[i] < s_i`,
  `t_exit_s[i] < t_enter_s[i]`, an edge index outside `edge_risk`, or a wait node outside a given `node_risk`.

`route_trace(route, t0_s=0.0) -> RouteTrace` converts the router's `Route` (spec 04: `nodes`, `edges`, `depart_s`,
`waits_s`, `start_s`, `arrive_s`; duck-typed, no import of `outrun.routing`) to times after `t0_s` (callers pass
the default 0, i.e. the scenario clock): `t_start_s = start_s - t0_s`, `t_enter_s[i] = depart_s[i] - t0_s`,
`t_exit_s[i] = depart_s[i+1] - waits_s[i+1] - t0_s`, `t_exit_s[-1] = arrive_s - t0_s` and
`wait_nodes = tuple(nodes[:-1])`.

DESIGN 6.2 metrics "% of routes intersecting observed hazard (r >= 0.5 or blocked)", "integrated observed exposure
(risk-minutes)" and "blocked crossings traversed" come from this function applied to observed arrays; "predicted
exposure" is the same function applied to the planner's forecast arrays.

## Requirements: bootstrap (`eval/bootstrap.py`)

`CI(estimate, low, high, n_resamples, n_blocks)` is a frozen dataclass with `to_dict()` / `from_dict()`.
Intervals are percentile intervals: `low = quantile(stats, alpha/2)`, `high = quantile(stats, 1 - alpha/2)`
(`numpy.quantile`, default linear method) over the finite resample statistics; `n_resamples` reports how many
were finite. If none are finite, `low = high = nan`.

### EVAL-10 Day-block resampling (Must)
`block_resample_indices(days, rng) -> ndarray[int]`: let D be the unique day labels (any hashable/sortable type,
via `numpy.unique`). Draw `len(D)` labels uniformly with replacement and return the concatenated indices of all
samples of each drawn label. Consequences tested: every returned index's day was drawn; within one draw, all
indices of the same day appear the same number of times (whole blocks only); `len(result)` equals the sum of
the drawn blocks' sizes.

### EVAL-11 Day-block bootstrap CI (Must)
`day_block_ci(metric_fn, arrays, days, n_resamples=1000, alpha=0.05, seed=0) -> CI`. `estimate =
metric_fn(*arrays)` on the full data. Each resample applies `block_resample_indices` (from one
`default_rng(seed)`) to every array's first axis and evaluates `metric_fn`. `n_blocks` is the number of unique
days. Errors (`ValueError`): fewer than 2 unique days, `len(days)` differing from any array's first-axis length,
`n_resamples < 1`, `alpha` outside `(0, 1)`. When days are internally homogeneous (for example 10 days of 50
identical samples), the interval must be as wide as a bootstrap over the 10 day values, not over 500 samples.
DESIGN 6.1 uses 1,000 resamples.

### EVAL-12 Paired bootstrap and "gap closed" (Must)
`paired_bootstrap_ci(stat_fn, arrays, n_resamples=1000, alpha=0.05, seed=0) -> CI` resamples one index vector
(uniform with replacement, length n) and applies it to every array; `estimate = stat_fn(*arrays)`;
`n_blocks = n`. Arrays of different first-axis length, or `n < 2`, raise `ValueError`.
`gap_closed(exp_a, exp_c, exp_d) = (mean(a) - mean(c)) / (mean(a) - mean(d))` (`nan` if `mean(a) == mean(d)`);
`gap_closed_ci(...)` is `paired_bootstrap_ci(gap_closed, (exp_a, exp_c, exp_d), ...)`. Resamples where the gap
is 0 are `nan` and excluded. This is the routing headline "DL planning (c) closes X% of the observed-exposure gap
between (a) and (d)" (DESIGN 6.2).

## Requirements: routing evaluation (`eval/routing_eval.py`)

### EVAL-13 Planner set and origin sampling (Must)
`PlannerKind` has exactly five members with values `"a"`, `"a_prime"`, `"b"`, `"c"`, `"d"`, and
`RISK_SOURCE` maps them to the `--risk-source` values `"none"`, `"current-only"`, `"baseline"`, `"model"`,
`"observed"` (no hazard, avoid-current-only, baseline layers, DL layers, oracle). `PLANNER_ORDER` lists them in
that order. `sample_origins(node_ids, weights, n=500, seed=0) -> ndarray` draws `n` node ids with replacement
with probability proportional to `weights` (population-weighted residential origins). Zero-weight nodes are
never drawn. `ValueError` if lengths differ, any weight is negative or NaN, all weights are 0, or `n < 1`.

### EVAL-14 Plan on forecast, score on observed (Must)
`evaluate_planners(origins, plan, observed_risk, observed_blocked=None, predicted_risk=None, crossing_edges=(),
slot_s=900, threshold=0.5, n_resamples=1000, seed=0, observed_node_risk=None) -> RoutingEvalResult` calls
`plan(kind, origin)` exactly once per (planner in `PLANNER_ORDER`, origin) and scores every returned `RouteTrace`
with `score_route` on **`observed_risk` / `observed_blocked`** (and `node_risk=observed_node_risk`, so waits are
charged at the node waited at) only. `predicted_risk[kind]` (optional) is used only for that planner's
`mean_predicted_exposure` (else `None`). Producers (ASTAR-20, so the report and the app agree): the CLI and the
backend pass `observed_risk = observed_model.combined_risk_table()` and `observed_blocked =
observed_model.blocked_table()` and `observed_node_risk = observed_model.combined_node_risk_table(graph.n_nodes)`
of the CostModel built from the observed stack (`build_cost_model`, ASTAR-18), and
their `plan(kind, origin)` returns `replay_trace(route, observed_model)` (PLAN-03), i.e. the planned route
re-driven on the observed clock.

- `plan` returning `None` means unroutable. `PlannerSummary.n_routed` / `n_failed` count over **all** origins.
- The **paired set** is the origins for which all five planners returned a route; `n_paired` is its size.
  Every other summary field is computed over the paired set only: `pct_intersect` (0-100),
  `mean_exposure_risk_min`, `mean_travel_time_ratio` (per-origin `travel_time_s / travel_time_s of (a)`; exactly
  1.0 for (a); an (a) travel time of 0 gives ratio `nan` and is skipped by a `nanmean`), `blocked_crossings`
  (total).
- An empty paired set gives `nan` means, `pct_intersect = nan` and `blocked_crossings = 0`.
- `origins` empty raises `ValueError`.

### EVAL-15 Routing headline comparisons (Must)
On the paired set with per-origin observed exposures `E_k`: `gap_closed = gap_closed_ci(E_a, E_c, E_d)`,
`c_minus_b = paired_bootstrap_ci(mean(E_c - E_b))`, `c_minus_a_prime = paired_bootstrap_ci(mean(E_c - E_a'))`,
all with the given `n_resamples` and `seed`. Fewer than 2 paired origins gives `CI(nan, nan, nan, 0, n_paired)`
for all three. (DESIGN 6.2: headline, plus (c) vs (b) and (c) vs (a').)

## Requirements: report (`eval/report.py`)

### EVAL-16 CI formatting (Should)
`format_ci(ci, digits=3)` accepts a `CI` or its dict form and returns `f"{est:.{d}f} [{low:.{d}f}, {high:.{d}f}]"`,
e.g. `"0.512 [0.480, 0.540]"`. A NaN estimate gives `"n/a"`; a finite estimate with a NaN bound gives
`f"{est:.{d}f} [n/a]"`.

### EVAL-17 Per-model metrics JSON (Must)
`write_model_metrics(out_dir, hazard, results, primary_metric, higher_is_better, git_sha, n_resamples=1000,
ablations=None) -> Path` writes `<out_dir>/<hazard>.json` (creating `out_dir`):

```json
{"schema": 1, "hazard": "storm", "git_sha": "abc123", "created_utc": "2026-12-01T18:00:00Z",
 "primary_metric": "csi_mean_74_133", "higher_is_better": true, "n_resamples": 1000,
 "splits": {"test": {"<model name>": {"kind": "dl", "metrics": {"<metric>": {"estimate": 0.4, "low": 0.38,
   "high": 0.42, "n_resamples": 1000, "n_blocks": 120}}}}, "case_holdout": {"...": "..."}},
 "ablations": {"vil_only": {"<model name>": {"kind": "dl", "metrics": {"...": "..."}}}}}
```

`results` maps split name to a sequence of `ModelMetrics(name, kind, metrics: dict[str, CI])`; `ablations`
(default `{}`) maps ablation name to the same. `load_model_metrics(path)` returns the parsed dict.
Errors (`ValueError`, nothing written): `hazard` not in `HAZARDS`; empty `git_sha`; no `"test"` split; a split
name other than `"val"`, `"test"`, `"case_holdout"` or `"test_<suffix>"` (e.g. `test_wsts2021`); a split
without at least one `kind="dl"` and one `kind="baseline"` model (honest-baseline rule); a model missing
`primary_metric`; duplicate model names in one split. `reference` rows (published numbers) may carry NaN CI
bounds. Primary metric names per hazard are fixed (the writer does not enforce them, but AC-01..06 check
them): storm `csi_mean_74_133`, tornado `auc_tor_vs_warned_null`, flood `auprc`, wildfire `prauc` (and split
`test_wsts2021`), ice `zone_auprc`, smoke `rmse_smoke_episodes` (lower is better).

### EVAL-18 Routing metrics JSON (Must)
`write_routing_metrics(out_dir, scenario_id, result, git_sha, primary=False) -> Path` writes
`<out_dir>/routing_<scenario_id>.json`: `{"schema": 1, "scenario", "primary", "git_sha", "created_utc",
"n_origins", "n_paired", "planners": {"a": {PlannerSummary fields}, "a_prime": ..., "b": ..., "c": ..., "d": ...},
"gap_closed": CI dict, "c_minus_b": CI dict, "c_minus_a_prime": CI dict}`. NaN is written as JSON `NaN` (Python
`json` default) and reads back as NaN. Empty `git_sha` raises `ValueError`.

### EVAL-19 Markdown report with honest verdicts (Must)
`write_metrics_md(metrics_dir, out_path) -> Path` reads every `<hazard>.json` (in `HAZARDS` order),
`routing_*.json` (sorted) and `latency.json` (if present) in `metrics_dir`, and writes Markdown starting with
`# Outrun metrics`:

- Per hazard: `## <hazard>`, a line `git sha: <sha>`, then for each split (`test` first, the rest sorted) a
  table `| model | kind | <metrics, primary first, rest sorted> |` with one row per model and `format_ci` cells
  (`n/a` where a model lacks a metric).
- Under each split table, one verdict line comparing the best `dl` model with the best `baseline` model
  (`reference` rows are never the comparison) on the primary metric, honouring `higher_is_better`:
  `**<split>:** DL model `<dl>` beats the best baseline `<bl>` on <metric>: <x> vs <y>.` when strictly better,
  otherwise `**<split>:** DL model `<dl>` does not beat the best baseline `<bl>` on <metric>: <x> vs <y>.`
  (`x`, `y` are estimates with 3 decimals; ties and NaN DL estimates count as "does not beat").
- Per routing file: `## Routing: <scenario>`, a planner table, and the headline
  `DL planning (c) closes {100*est:.1f}% [{100*low:.1f}%, {100*high:.1f}%] of the observed-exposure gap between
  (a) and (d).` (`closes n/a of the ...` when the estimate is NaN).
- Per group file (`group_*.json`, sorted, EVAL-22): `## Group: <scenario>` and three tables:
  `### Shelters` (`| method | overflow_persons | unserved_persons | max_minutes |`, rows `mcf`, `nearest`, `greedy`
  then any others sorted; counts as integers, minutes with 3 decimals), `### Evacuation` (`| method | seeds | vehicle_hours | exposure_veh_h | clearance_min |
  max_vc | gap_to_lb_pct |`, one row per method in file order, cells = mean over seeds with 3 decimals, `seeds` =
  number of seeds) followed by the line `lower bound: <lb>`, and `### Emergency vehicles` (`| dispatch |
  mean_eta_min | mean_detour_min | pct_hit_block |`, rows `hungarian` then `reroute_baseline`) followed by the
  number of Pareto points.
- Missing `metrics_dir` raises `FileNotFoundError`; a directory with no metrics files still writes the heading.

### EVAL-22 Group metrics JSON: shelters, whole-city evacuation, emergency vehicles (Must; DESIGN 6.2)
`write_group_metrics(out_dir, scenario_id, *, shelters, evacuation, lower_bound, ems, git_sha, seeds,
primary=False) -> Path` writes `<out_dir>/group_<scenario_id>.json`:
`{"schema": 1, "scenario", "primary", "git_sha", "created_utc",
"shelters": {method: SHEL-07 fields},
"evacuation": {"seeds": [...], "lower_bound": float, "methods": {method: [EVAC-10 dict per seed]}},
"ems": {"hungarian": EMS-07 dict, "reroute_baseline": EMS-07 dict, "pareto": [{"lam", "eta_min", "exposure"}]}}`.
- `shelters` maps method -> a `ShelterMetrics` (converted with `dataclasses.asdict`) or a dict with its fields;
  it must contain `mcf`, `nearest` and `greedy` (SHEL-05/06).
- `evacuation` maps method -> a list of EVAC-10 dicts, one per seed in `seeds`; it must contain `sa`, `msa`,
  `shortest`, `safest` and `sa_lambda0` (the `lambda_E = 0` ablation, EVAC-09). Every list has `len(seeds)`
  entries; DESIGN asks for 5 seeds (the AC row checks 5; this writer accepts >= 1).
- `ems` must contain `hungarian` and `reroute_baseline` (EMS-03/05 scored with true ETAs by EMS-07) and `pareto`
  (EMS-06 points, possibly empty).
- `ValueError` for: a missing required method, a seed-count mismatch, empty `seeds`, empty `git_sha`, or a
  non-finite `lower_bound`. NaN metric values are written as JSON `NaN`. `load_group_metrics(path)` reads it back
  (round trip equal, NaN-aware).

### EVAL-20 One-shot test ledger (Must)
`RunLedger(path)` is an append-only JSON list at `path` (default use: `reports/test_ledger.json`), created with
its parent folders on first `record`. `record(model, git_sha, split="test", rerun_reason=None)` appends
`{"model", "split", "git_sha", "utc", "rerun_reason"}` and returns it. A second record for the same
`(model, split)` raises `HeldOutReuseError` unless `rerun_reason` is a non-empty string (then it is appended
with the reason). Empty `git_sha` raises `ValueError`. A new `RunLedger` on the same path sees earlier entries.
`runs(model=None, split=None)` returns entries in record order, filtered. (DESIGN 6.3: test evaluated once in
T24, logged with git sha.)

### EVAL-21 Latency timing (Should)
`time_calls(fn, inputs, warmup=3) -> LatencyStats(n, median_s, p95_s, max_s)` first calls `fn(inputs[i %
len(inputs)])` for `i in range(warmup)` untimed, then times `fn(x)` once for each `x` in `inputs` with
`time.perf_counter`. `n = len(inputs)`; `median_s` / `p95_s` are `numpy.median` / `numpy.percentile(..., 95)`;
`max_s` the maximum. Empty `inputs` or negative `warmup` raise `ValueError`.
`write_latency(out_dir, name, stats, git_sha)` merges `{name: {"n", "median_s", "p95_s", "max_s", "git_sha",
"created_utc"}}` into `<out_dir>/latency.json`, keeping other names. Names used by T24: `astar_metro_query`
(DESIGN target < 1 s) and `sa_run` (target < 60 s).

## Requirements: splits and held-out discipline (`eval/splits.py`, `eval/casestudy.py`)

Data model: `SplitItem(id, day=None, group=None)` (frozen, ordered) and `Split(dataset, train, val, test,
extra={}, meta={})`. Every builder below returns sets as tuples **sorted by id**, puts the input items whose day
is in `quarantine` into `extra["case_holdout"]` (SPLIT-08), and returns a split for which
`validate_split(split, gap_days=1) == []` holds for **any** valid input. Duplicate input ids raise `ValueError`.
Items outside every period of a builder are dropped. **Official test sets are never changed by quarantine or by
gap days**: only train and val lose items. Split files are `splits/<name>.json` for the names in `SPLIT_NAMES`
(`sevir`, `tornet`, `flood`, `ice`, `smoke`, `ndws`, `wsts`) plus `quarantine.json` and `hashes.json`.

"Gap day at a boundary" means: with boundary date B between an earlier and a later period, items on day
`B - 1 day` are dropped from the earlier set, so the closest train/val/test days are at least 2 days apart.

### SPLIT-01 Storm (SEVIR) split (Must)
`sevir_split(events, quarantine=())`, `events` = `(event_id, start datetime UTC)`. Item: `id = event_id`,
`day = start.date()` (UTC), `group = event_id`. Official cut: **test = day >= 2019-06-01** (all of it, untouched).
Val = 2019-04-01..2019-05-30 (2019-05-31 is the gap day). Train = 2018-01-01..2019-03-30 (2019-03-31 is the gap
day). Days before 2018-01-01 are dropped (2017 is skipped). Quarantined days are removed from train and val.
Split by calendar day (no same-day leakage).

### SPLIT-02 Tornado (TorNet) split (Must)
`tornet_split(samples, quarantine=(), val_frac=0.1, seed=0)`, `samples` = `(chip_id, time UTC, official)` with
`official` in `{"train", "test"}` (else `ValueError`). The official assignment is an input: it comes from the
TorNet rule verified in T1 (DATA-02), not from this function. Test = every official-test chip, unchanged. The
train pool = official-train chips whose day is not quarantined and not within 1 day of any test day. Val days =
`max(1, round(val_frac * P))` distinct days drawn without replacement from the P pool days (seeded); if P < 2,
val is empty. Train = pool chips whose day is not a val day and not within 1 day of a val day. Group = None.

### SPLIT-03 Flood split (Must)
`flood_split(samples, quarantine=(), test_year=2025, first_year=2021)`, `samples` = `(sample_id, valid time
UTC)`. Train = years `first_year..test_year-2`, val = year `test_year-1`, test = year `test_year`. Gap days: Dec 31
of the last train year (dropped from train) and Dec 31 of the val year (dropped from val). `test_year - 2 <
first_year` raises `ValueError`. Fallback when 2025 is incomplete: `test_year=2024` (train 2021-22, val 2023).

### SPLIT-04 Wildfire (NDWS + WSTS 2021) split (Must)
`wildfire_split(ndws, wsts_fires, quarantine=())`. `ndws` maps exactly `{"train", "eval", "test"}` (else
`ValueError`) to sample ids from the official NDWS files: train <- `train`, val <- `eval`, test <- `test`, items
`SplitItem("ndws:<id>")` with no day (NDWS has no dates, so "no overlap" here means: no id in two official files,
else `ValueError`). `wsts_fires` = `(fire_id, start date)`; fires starting in 2021 form
`extra["test_wsts2021"]` (`SplitItem("wsts:<fire_id>", day, group=fire_id)`), the out-of-time test; WSTS fires of
other years (they may overlap NDWS 2012-2020) appear in no set. Quarantine can only touch dated train/val items,
so it never changes NDWS or WSTS 2021 sets; quarantined WSTS 2021 fires are still listed in `case_holdout`.

### SPLIT-05 Ice split (Must)
`ice_season(t)` returns the year a winter season starts (season = Jul 1 .. Jun 30): `2024-01-15 -> 2023`,
`2023-12-01 -> 2023`, `2023-06-30 -> 2022`. `ice_split(bags, quarantine=(), train_seasons=(2020, 2021, 2022),
val_season=2023, test_seasons=(2024,))`, `bags` = `(zone_id, window start UTC)`; item `id = group =
f"{zone_id}|{start.isoformat()}"`, `day = start.date()`. Assignment by `ice_season`; bags of other seasons are
dropped. Gap days: Jun 30 at the end of the last train season and of the val season are dropped from the earlier
set. Overlapping season tuples raise `ValueError`. An extra test winter (2025-26) is `test_seasons=(2024, 2025)`.

### SPLIT-06 Smoke split, temporal + spatial monitors (Must)
`smoke_split(samples, monitors, quarantine=(), holdout_frac=0.2, block_m=200_000, seed=0)`, `samples` =
`(monitor_id, issue time UTC)`, `monitors` = `{monitor_id: (x_m, y_m)}` in EPSG:5070 metres (a sampled monitor
missing from `monitors` raises `ValueError`). Item `id = f"{monitor_id}|{issue.isoformat()}"`, day = issue date.
Time: train 2022-01-01..2023-06-29, val 2023-07-01..2023-12-30 (gap days 2023-06-30 and 2023-12-31), test = 2024.
Spatial: the held-out monitors are `spatial_holdout_monitors({m: monitors[m] for m in sampled}, holdout_frac,
block_m, seed)` over the monitors with at least one sample. `spatial_holdout_monitors(monitors, frac=0.2,
block_m=200_000, seed=0) -> frozenset[str]` is the single owned block-holdout algorithm (SMOKE-06 delegates to
it): monitors are grouped into blocks `(floor(x/block_m), floor(y/block_m))`; the blocks, sorted, are reordered
by `numpy.random.default_rng(seed).permutation(n_blocks)`; the held-out blocks are the shortest prefix of that
order holding at least `ceil(frac * n_monitors)` monitors. `frac` outside (0, 1) raises `ValueError`. Every sample of a held-out monitor is removed from
train and val; `test` keeps all monitors; `extra["test_spatial"]` = test samples of held-out monitors.
`meta["holdout_monitors"]` and `meta["holdout_blocks"]` (lists, sorted) record the choice.

### SPLIT-07 Quarantine of case-study days (Must)
`quarantine_dates(case_dates, radius_days=3)` returns every case date +- 3 days inclusive (7 days per case).
`write_quarantine(path, cases, radius_days=3)` writes `{"radius_days": 3, "cases": {id: "YYYY-MM-DD"}, "dates":
[sorted "YYYY-MM-DD"]}`; `load_quarantine(path)` returns the dates as a `frozenset[date]` and raises `ValueError`
if `dates` is not exactly the expansion of `cases`. **Every builder** (SPLIT-01..06) given a quarantine removes
items on quarantined days from train and val, and returns a test set identical to the one it returns with no
quarantine. Hazard dataset builders (`hazards/*/dataset.py`) take their train/val/test ids from these splits
through one owned helper: `split_mask(ids: Sequence[str], split, part) -> np.ndarray` (bool, `len(ids)`) is True
where `ids[i]` is an item id of `split.<part>` (`part` in `"train"`, `"val"`, `"test"` or a key of `split.extra`;
anything else raises `ValueError`). An id in no part (e.g. quarantined, so only in `extra["case_holdout"]`) is
False for train, val and test. The per-hazard split factories (`SevirWindowDataset.from_split` STORM-01,
`FloodPatchDataset.from_split` FLOOD-14, `NdwsDataset.from_split` FIRE-12, `bags_for_split` ICE-02,
`samples_for_split` SMOKE-01; `split_catalog` TORN-02) select their rows with it, and each has a test that a
quarantined id and a test id never reach its train or val dataset.

### SPLIT-08 Case hold-out set (Must)
`case_holdout(items, quarantine)` returns the items whose day is in `quarantine` (undated items never), sorted by
id. Each builder stores `case_holdout(<its in-period input items>, quarantine)` as `extra["case_holdout"]`
(empty tuple with no quarantine). DESIGN 6.1 reports each model on its official test split and on this set
separately.

### SPLIT-09 Split validation (Must)
`validate_split(split, gap_days=1) -> list[str]` (empty = clean) reports:
1. an id in more than one of train/val/test;
2. a non-None group in more than one of train/val/test;
3. two items in different sets among train/val/test whose days differ by `<= gap_days` days (same day included);
4. an `extra` item whose id or group is in train or val, or whose day equals a train or val day (no gap rule for
   extras: the quarantine radius is already a buffer).
Extras may overlap `test` (e.g. `test_spatial` is a subset of `test`). `assert_valid_split` raises
`SplitLeakageError` (a `ValueError`) with the messages when any exist.

### SPLIT-10 Frozen split files and hashes (Must)
`write_split(split, path)` writes canonical JSON (`json.dumps(obj, sort_keys=True, indent=2) + "\n"`, items
sorted by id, days as `"YYYY-MM-DD"` or `null`, extras by name). `read_split(write_split(s, p)) == s` for splits
whose sets are sorted and whose meta is JSON-typed. `split_hash(split)` is the sha256 hex of those bytes.
`write_hashes(splits_dir)` writes `hashes.json` = `{stem: {"file_sha256": ...}}` for every other `*.json` in the
directory, plus `"test_ids_sha256"` (sha256 of the sorted test ids joined by `"\n"`) for split files, and returns
it. `verify_hashes(splits_dir, strict=False)` returns the sorted stems whose file changed, disappeared, or is not
listed (a missing `hashes.json` lists every file); with `strict=True` a non-empty result raises
`SplitHashMismatch`. Unchanged files give `[]`.

### SPLIT-11 Case-study registry (Should)
`CaseStudy(id, hazards, day, city, held_out, primary=False)` (pure data; `held_out` is `"quarantine"` or
`"official_test"`). `case_quarantine(cases)` returns `{id: day}` for quarantine cases only (official-test cases,
such as the Jun-Dec 2019 OK SEVIR event, never enter the quarantine). Duplicate ids or more than one
`primary=True` case raise `ValueError`. Case dates themselves are T7 outputs, never hard-coded in code or tests.

## Open points

- `eval/truth.py` (observed-hazard readers) defines no requirement: `observed()` belongs to each hazard adapter
  (specs 08-13). If shared truth helpers emerge, they get EVAL IDs here.
- Case-study scenario construction (risk.zarr, overlays, results) is `outrun scenario` (spec 14); this spec owns
  only the case registry and the quarantine.
