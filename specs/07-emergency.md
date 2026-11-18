# 07: Emergency vehicle routing (inbound, fastest-safe)

Module: `src/outrun/routing/emergency.py`.
Tests: `tests/unit/test_emergency*.py`. Design: DESIGN 4.6 (source of truth), 4.2 (costs, EV profile), 6.2 (metrics).

Purpose: get emergency vehicles (EVs) to 1-3 incidents fast and safely: ETAs from every station by one reverse
time-dependent search, a Hungarian station-to-incident assignment, the hospital leg, a fastest-vs-safest Pareto sweep,
and a realistic hazard-blind baseline that discovers blocks only when it reaches them.

## Graph convention (shared with 05 and 06)
- A road graph is a `networkx.DiGraph`. Nodes may carry `x`, `y` (metres on GRID). Real-road graphs are built only
  by `routing.cost.to_td_digraph(graph, cost_model)` (ASTAR-19); per-slot congested times from an evacuation plan
  come from `routing.cost.slot_times_from_waves` (the `congested_s` of EMS-02).
- The time weight (attribute named by `weight`, default `"time"`) of an edge is either a scalar (seconds) or a 1-D
  sequence of per-slot seconds (slot `s` covers clock `[s * slot_s, (s + 1) * slot_s)` after t0, `slot_s` = 900 s,
  slots beyond the end use the last value). `inf` (or a missing attribute) means blocked.
- Traversing edge `(u, v)` when leaving `u` at clock `t` takes `tau(t) = weight[min(floor(t / slot_s), n - 1)]`.
  Callers supply FIFO weights (`t + tau(t)` non-decreasing); under FIFO, waiting never helps and earliest arrival is
  exact with label-setting search.
- Optional `"risk"` edge attribute: scalar `>= 0` (profile-weighted hazard sum, DESIGN 4.2); missing = 0.
- These functions take plain graphs on purpose, so they can be tested without the road and A* areas; the planner
  builds such graphs from `CostModel` (04) and the road graph (03).

**Must** (proposal row "emergency vehicle routes"): EMS-01..EMS-06. **Should** (report metrics): EMS-07.

## Requirements

### EMS-01 Time-dependent travel times to one target (`td_travel_times_to`, `inbound_eta`)
`td_travel_times_to(graph, target, sources: Sequence, depart_s=0.0, weight="time", slot_s=None) -> np.ndarray`.
- Entry `i` is the earliest arrival duration (seconds) at `target` for a vehicle leaving `sources[i]` at clock
  `depart_s`, under the graph convention. `inf` if unreachable; 0.0 if `sources[i] == target`.
- With scalar weights it equals networkx Dijkstra lengths (1e-9). With per-slot FIFO weights it equals the best
  simulated arrival over all simple paths.
- The intended method is a search from `target` over the reversed graph (one search for all sources); a forward
  search per source is acceptable if results are identical.
- Unknown nodes raise `KeyError`; `depart_s < 0` raises `ValueError`. `slot_s` defaults to 900.
`inbound_eta(graph, incident, stations, depart_s=0.0, weight="time") -> np.ndarray`: the same as
`td_travel_times_to(graph, incident, stations, depart_s, weight)` (ETA from every station at once).

### EMS-02 EV edge times (`ev_edge_times`)
`ev_edge_times(free_flow_s, congested_s, speedup=None) -> np.ndarray`: elementwise
`max(free_flow_s, congested_s / speedup)` (EVs move `speedup` = `emergency.ev_speedup_vs_congested` (1.2) times faster
than congested traffic, never faster than free flow). Broadcasts; `inf` (blocked) stays `inf`. `speedup <= 0` raises
`ValueError`.

### EMS-03 Hungarian multi-incident dispatch (`dispatch`)
`dispatch(eta: np.ndarray (S, I), max_incidents=None) -> list[int | None]` returns the station index per incident.
- Distinct stations per incident. First maximise the number of incidents assigned to a station with finite ETA,
  then minimise the total ETA of the assigned pairs (exact; `scipy.optimize.linear_sum_assignment`).
- Incidents left without a feasible station get `None` (also when `I > S`).
- `I > max_incidents` (default `emergency.max_incidents`, 3) raises `ValueError`. `I == 0` gives `[]`.

### EMS-04 Hospital leg (`nearest_open_hospital`)
`nearest_open_hospital(graph, incident, hospitals: Sequence, hospital_risk: np.ndarray, depart_s=0.0, threshold=0.5,
weight="time") -> tuple[int, float] | None`.
- Considers only hospitals with `hospital_risk[i] <= threshold` (open, outside the hazard). Returns `(i, eta_s)` for
  the one with the smallest forward time-dependent travel time from `incident` (EMS-01 semantics), ties lowest `i`.
- `None` if no open hospital is reachable. Length mismatch raises `ValueError`.

### EMS-05 Discover-and-reroute baseline (`nearest_station_straight_line`, `discover_and_reroute`)
`nearest_station_straight_line(station_xy: (S, 2), incident_xy: (2,)) -> int`: Euclidean nearest, ties lowest index;
`S == 0` raises `ValueError`.
`discover_and_reroute(graph, source, target, blocked: set[tuple], weight="time") -> RerouteResult`
(`RerouteResult(path, eta_s, planned_eta_s, detour_s, hit_block, n_replans, reached)`), scalar weights:
- Plans a shortest path ignoring hazards (`blocked` edges have normal finite weights in `graph`); `planned_eta_s` is
  its length.
- Drives the plan edge by edge. When the next edge `(u, v)` is in `blocked`, the vehicle (now at `u`) discovers it,
  adds it to its known blocks, and replans a shortest path from `u` to `target` avoiding all known blocks
  (`n_replans += 1`, also when that replan fails). Only edges the vehicle tries to enter are discovered.
- `path` is the node sequence actually driven; `eta_s` = sum of weights of driven edges (the **true** ETA including
  the detour); `detour_s = eta_s - planned_eta_s`; `hit_block = n_replans > 0`.
- If a replan finds no path: `reached=False`, `eta_s = inf`, `detour_s = inf`, `path` ends at the node where it got
  stuck.
- If `target` is unreachable from the start ignoring blocks: `reached=False`, `path=[source]`, `eta_s=inf`,
  `planned_eta_s=inf`.

### EMS-06 Fastest-safe Pareto sweep (`route_weighted`, `pareto_sweep`)
`route_weighted(graph, source, target, lam, time="time", risk="risk") -> ParetoPoint | None`
(`ParetoPoint(lam, time_s, exposure, path)`): the path minimising `sum_e time_e * (1 + lam * risk_e)` (scalar
weights); `time_s = sum time_e`, `exposure = sum time_e * risk_e` (risk-seconds) along that path. `None` if
unreachable. `lam < 0` raises `ValueError`.
`pareto_sweep(graph, source, target, lambdas=None, ...) -> list[ParetoPoint]`: one point per lambda (default
`emergency.pareto_lambdas`), in input order; unreachable gives `[]`. For increasing lambda, `time_s` is non-decreasing
and `exposure` non-increasing (holds for exact weighted-sum optimisation, ties included).

### EMS-07 EMS metrics (`ems_metrics`)
`ems_metrics(results: Sequence[RerouteResult], exposures: Sequence[float] | None = None) -> dict[str, float]` with
`mean_eta_min` and `mean_detour_min` (over reached results), `pct_hit_block` (100 x share of all results with
`hit_block`), `pct_unreached`, `mean_exposure` (mean over **all** results, `nan` if `exposures` is `None`). Empty input raises `ValueError`.
