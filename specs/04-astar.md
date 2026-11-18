# 04: Edge costs, time-dependent A*, safe-route planner

Modules: `src/outrun/routing/cost.py`, `src/outrun/routing/astar.py`, `src/outrun/routing/planner.py`
Tests: `tests/unit/test_cost.py`, `tests/unit/test_astar.py`, `tests/unit/test_planner.py`
(synthetic graphs from `tests/fixtures/graphs/graphfix.py`; no OSM, no real risk data)
Design: DESIGN 4.2 (risk to edge costs), 4.3 (A*), 6.2 (planners), 2.2 (`routing/*` rows).
Proposal phrases: "A* search" (ASTAR-01..06), "safe evacuation routes" (PLAN-01..04).

`Must` = needed for the approved proposal. `Should` = depth or a support requirement that the Must ones rely on.

## Model and notation

- Clock `t` = seconds since the scenario issue time t0 (`t >= 0`). Slots are `slot_s` long (default 900 s);
  `S` slots (12 = 3 h). `slot(t) = min(floor(t / slot_s), S - 1)`: risk is piecewise constant per slot and the last
  slot is held after the window (the beyond-horizon rules of spec 02 are already applied inside the RiskStack).
- Risk arrays: edge risk `r[e, h, s]` (E, H, S) and node risk `q[n, h, s]` (N, H, S) in [0, 1]. When built from
  policies, H = 6 in `outrun.core.risk.Hazard` order (STORM, TORNADO, FLOOD, WILDFIRE, ICE, SMOKE).
- Per profile: weights `w_h >= 0`, block thresholds `b_h` (NaN = never), speed factor
  `sf_h(p) = clip(1 - slope_h * p, 1/3, 1)` (or `HazardPolicy.speed_factor`), "safer <-> faster" `lam >= 0`,
  static congestion multiplier `bpr_e >= 1`, clock cap 3.
- Clock: `tau_e(t) = min(ff_e / prod_h sf_h(r[e,h,slot(t)]) * bpr_e, 3 * ff_e)`.
- Cost: `c_e(t) = tau_e(t) * (1 + lam * sum_h w_h r[e,h,slot(t)])`, and `+inf` if `r[e,h,slot(t)] >= b_h` for any h.
- Wait at node n from t1 to t2: `W_n(t1, t2) = integral (1 + lam * sum_h w_h q[n,h,slot(t)]) dt >= t2 - t1`.
- An edge is evaluated at its **entry** time (no change of slot mid-edge).

**Admissibility argument (DESIGN 4.3).** Every factor is >= 1 relative to free flow and blocking only adds `+inf`,
so `c_e(t) >= tau_e(t) >= ff_e` and waits cost >= 0. With `v* = max_e hav(u_e, v_e) / ff_e`, every edge satisfies
`hav(u, v) / v* <= ff_e`, so `h(n) = hav(n, goal) / v*` obeys `h(u) <= ff_e + h(v) <= c_e(t) + h(v)` (consistent,
by the triangle inequality of great-circle distance) and never overestimates; `h(goal) = 0`. `v*` is at most the
maximum free-flow speed whenever lengths are at least the great-circle span, so this `h` is at least as tight as
DESIGN's `v_max` version. This is why risk may only multiply by >= 1, add >= 0 or block, and never discount.

**Search (DESIGN 4.3).** Labels (node, t, g). `(t1, g1)` dominates `(t2, g2)` at node n iff `t1 <= t2` and
`g1 + W_n(t1, t2) <= g2`. Each node keeps a Pareto set of non-dominated labels (cap `max_labels`, default 8 from
`configs/planning.yaml`; a cap hit marks the result approximate). Queue ordered by `f = g + h`; the search stops
when a goal label is popped. Waiting is allowed at any node (including the origin) on a discrete grid: a label
that reaches node n at time t may leave at t or at any later time `t0_s + k * wait_step_s` (k a positive integer,
default `wait_step_s = 60`) up to the end of the risk window (`n_slots * slot_s`); waits never extend past the window
end. The search is exact with respect to continuous waiting when every clock time and `t0_s` are multiples of
`wait_step_s` and the label cap is not hit.

## Requirements: edge costs (`routing/cost.py`)

### ASTAR-07 Clock time (Should)
`CostModel(free_flow_s, risk, weights, block_at, *, speed_slope=None, node_risk=None, bpr=None, lam=1.0,
slot_s=900, clock_cap=3.0)`. `clock(e, t)` = `tau_e(t)` above (speed factors multiply, BPR multiplies, capped at
`clock_cap * ff_e`); defined even on a blocked edge. `clock_table()` is the (E, S) array of `tau` at each slot and
equals `clock(e, s * slot_s)`. Always `tau >= ff`.

### ASTAR-08 Generalized cost, blocking, policies, validation (Must for ASTAR-03)
- `cost(e, t)` = `c_e(t)`; `blocked(e, t)` is True iff some `r[e,h,slot(t)] >= b_h` (inclusive); then
  `cost = math.inf`. `cost_table()` is (E, S) with inf where blocked; finite entries are >= `clock_table()`.
- `lam` scales every weight (lam = 0: cost = clock) but never removes blocking.
- Where and when blocking applies (`CostModel(..., block_mask=None, block_until_s=None)`, keyword-only):
  `block_mask[h]` is an `(E,)` bool array; hazard `h` then blocks only edges where it is True. `block_until_s[h]`
  (seconds after t0): hazard `h` blocks only in slots `s` with `s * slot_s < block_until_s[h]`. Keys are hazard
  indices in `[0, H)`; another key or a mask of the wrong length raises `ValueError`. Both affect only blocking
  (`blocked`, the inf in `cost`/`cost_table`, `blocked_table`); exposure and speed factors are unchanged.
- `CostModel.from_policies(free_flow_s, risk, policies, profile, *, node_risk=None, lam=1.0, crossing=None,
  block_until_s=None)`: H = len(Hazard) in Hazard order; `w_h = policy.exposure_weight[profile]`,
  `b_h = policy.block_at[profile]` (None -> never), speed factor = `policy.speed_factor` applied elementwise; a
  hazard missing from `policies` has weight 0, never blocks, speed factor 1.
  - **Flood blocks only at crossings** (DESIGN 4.2, FLOOD-10 "applied at crossings"): `crossing` is the `(E,)` bool
    `graph.crossing`; FLOOD gets `block_mask = crossing`. With `crossing=None`, FLOOD never blocks (no crossing is
    known); its exposure weight and speed factor still apply. A non-crossing road in a flood cell is therefore
    slowed and penalised but never walled off.
  - **Tornado blocks only near issue time** (DESIGN 3.2 "`block_at` = 0.5 for slots within 20 min"):
    `block_until_s: Mapping[Hazard, float] | None`; None reads `hazards.<h>.block_within_min * 60` from
    `configs/hazards.yaml` for every hazard that has the key (tornado: 1200 s, so slots 0 and 1 block, slot 2 on
    does not); `{}` means no limit. The HORIZON-02 lingering tornado values (RISK-02) still add exposure; whether
    they block is decided here.
- `ValueError` for: risk not 3-D, risk/free-flow edge-count mismatch, risk or node risk outside [0, 1] or NaN,
  weights/block_at/slopes not of length H, node_risk with wrong H or S, negative weights, slopes or lam,
  bpr < 1, free_flow <= 0, slot_s <= 0, clock_cap < 1.

### ASTAR-09 Slots and risk swapping (Should)
`slot(t)` as defined above; t < 0 -> `ValueError`; cost is constant within a slot; times past the window use the
last slot. `with_risk(risk, node_risk=None)` returns a new `CostModel` with the same parameters (weights, block,
slopes, bpr, lam, slot_s, cap) and new risk arrays; the original is unchanged.

### ASTAR-10 Wait cost (Should)
`wait_cost(n, t1, t2) = W_n(t1, t2)` integrated slot by slot; `t2 == t1` -> 0; `t2 < t1` -> `ValueError`; no node
risk -> `t2 - t1`; `lam` scales the weights.

### ASTAR-11 Admissible, consistent heuristic (Must)
`crowfly_speed_mps(graph)` = `v*` (0.0 if every edge has zero span). `make_heuristic(graph, goals)` returns the
(N,) array `h(n) = min over goals hav(n, goal) / v*` (all zeros if `v* = 0`). Tested: `h(goal) = 0`, `h >= 0`,
`h[src] <= ff + h[dst]` on every edge, multi-goal `h` = elementwise min of single-goal `h`, `v*` <= max speed.

### ASTAR-14 Edge risk sampler: grid to edges (Must; "what you see is what routes", with RISK-07)
`EdgeRiskSampler(graph, x_left, y_top, cell_m, shape, spacing_m=100)`. Grid is north-up:
`col = floor((x - x_left) / cell_m)`, `row = floor((y_top - y) / cell_m)`, inside iff `0 <= row < ny` and
`0 <= col < nx`. Each edge polyline (graph geometry if present, else the straight segment between its nodes, in
`x_m/y_m`) is sampled at arc lengths 0, spacing, 2*spacing, ... and at its end point.
`sample(prob (H, S, ny, nx), edge_overrides=None) -> (E, H, S) float32`: max over the cells of the sample points,
points outside the grid contribute 0; `edge_overrides[h][e]` (length-S array) **replaces** the sampled value
(point features: low-water crossings, bridges). `sample_nodes(prob) -> (N, H, S)`: value of the cell containing
each node (0 outside). Wrong `prob` shape -> `ValueError`. An `edge_overrides` key that is not an `int` in
`[0, H)` (for example a `Hazard` member) raises `ValueError`, so overrides are never silently dropped.

### ASTAR-18 RiskStack to CostModel handoff (Must; the one owned path from the risk layer to the router)
- `EdgeRiskSampler.from_window(graph, window, spacing_m=100)` = `EdgeRiskSampler(graph, window.x0, window.y0,
  window.res, window.shape, spacing_m)` (same samples).
- `sampler.sample_stack(stack) -> (E, 6, S)` = `sample(np.asarray(stack.data, float32), {list(Hazard).index(h): ov
  for h, ov in stack.edge_overrides.items()})`: the `Hazard`-keyed `RiskStack.edge_overrides` (RISK-02) are mapped
  to hazard indices and **replace** the sampled values. An override edge id outside `[0, n_edges)` raises
  `ValueError` (the overrides were keyed on a different graph; FLOOD-10/ICE-08 require the router's graph).
  `sampler.sample_stack_nodes(stack) -> (N, 6, S)` = `sample_nodes(stack.data)`.
- `build_cost_model(graph, stack, profile, policies=None, *, lam=1.0, spacing_m=100) -> CostModel` =
  `CostModel.from_policies(graph.free_flow_s, s.sample_stack(stack), policies or load_policies(), profile,
  node_risk=s.sample_stack_nodes(stack), lam=lam, crossing=graph.crossing)` with `s = EdgeRiskSampler.from_window(
  graph, stack.window, spacing_m)` (default tornado `block_until_s` from config). The CLI, the app backend, the
  routing evaluation and `to_td_digraph` callers build their cost models through it.

### ASTAR-19 RoadGraph + CostModel to the optimizers' DiGraph (Must; shelters, evacuation, EMS)
Specs 05-07 take a `networkx.DiGraph` (the EMS graph convention, 07-emergency.md). It is built only here:
- `to_td_digraph(graph, cost_model) -> nx.DiGraph`: nodes `0..N-1` with attributes `x = graph.x_m[n]`,
  `y = graph.y_m[n]`; one edge per ordered node pair. For parallel CSR edges `u -> v` the one with the smallest
  `free_flow_s` is kept (ties: lowest edge id). Edge attributes (lists of length S, per slot):
  `"time"` = `clock_table()[e]` with `inf` where `blocked_table()[e]`; `"cost"` = `cost_table()[e]`;
  `"risk"` = `weighted_risk_table()[e]` (ASTAR-20); `"eid"` = `e` (the CSR edge id). So blocking, profile
  weights and `lam` reach shelters (SHEL-04), evacuation (EVAC-02) and EMS (EMS-01..06).
- `node_path_to_edges(G, nodes) -> np.ndarray` (int64) = `[G[u][v]["eid"] for consecutive (u, v)]`: converts
  `diverse_paths` node lists into the edge-index arrays `EvacInstance.candidates` needs (unique, because the
  DiGraph has one edge per pair). Fewer than 2 nodes -> empty array; a missing pair -> `KeyError`.
- `slot_times_from_waves(wave_times, wave_s=1800, slot_s=900, n_slots=12) -> np.ndarray`: `wave_times` is the
  `(W, E)` congested time per evacuation wave (EVAC-03: `bpr_time(edge_t0[e], score.loads[j, e], cap_wave[e])`); returns `(E, n_slots)` where slot `s` takes wave
  `min(floor(s * slot_s / wave_s), W - 1)`. This is the `congested_s` input of `ev_edge_times` (EMS-02).

### ASTAR-20 Scoring tables and the one exposure definition (Must)
- `cost_model.combined_risk_table() -> (E, S)` float = `max_h r[e, h, s]` (in [0, 1]);
  `cost_model.blocked_table() -> (E, S)` bool with `blocked_table()[e, s] == blocked(e, s * slot_s)`;
  `cost_model.weighted_risk_table() -> (E, S)` = `sum_h w_h r[e, h, s]` (profile weights, `lam` not applied);
  `cost_model.combined_node_risk_table(n_nodes=None) -> (N, S)` = `max_h node_risk[n, h, s]`, or zeros
  `(n_nodes, S)` when the model has no node risk (the same "None = zeros" as `wait_cost`); `ValueError` when
  node risk is None and `n_nodes` is None, or when `n_nodes` differs from node risk's N.
- Waits are scored at the node they happen at: every scorer below passes `node_risk =
  combined_node_risk_table(...)` together with a trace that carries `wait_nodes`, so waiting at a safe node is
  credited (EVAL-09). Traces use the scenario clock (`route_trace(route)`, `t_start_s = route.start_s`).
- Exposure has **one** definition, EVAL-09 `score_route`, on these tables: planner exposure (PLAN-05), replay
  (PLAN-03), the app's `exposure` fields (APP-03) and the routing evaluation (EVAL-14, which receives
  `observed_risk = observed_model.combined_risk_table()` and `observed_blocked = observed_model.blocked_table()`
  from the CLI/backend) all use it, so the same route on the same field gets the same number everywhere.

## Requirements: search (`routing/astar.py`)

`astar(graph, cost_model, origin, goals, t0_s=0, *, use_heuristic=True, allow_wait=True, max_labels=8,
wait_step_s=60.0) -> Route | None`. `Route` (frozen dataclass): `nodes` (origin..goal), `edges`, `depart_s` (entry time of each
edge), `waits_s` (wait at `nodes[i]` before `edges[i]`; same length as `edges`), `start_s`, `arrive_s`, `cost`,
`expansions` (labels popped and expanded), `approximate`; properties `goal`, `travel_s = arrive_s - start_s`.

### ASTAR-01 Equals Dijkstra with zero and static risk (Must)
On random synthetic graphs (200 OD pairs in total, zero risk) and with static (slot-independent) risk, the route
cost equals networkx `dijkstra_path_length` with weight `ff_e` (zero risk) or `c_e` (static risk), to 1e-9
relative; zero risk gives `travel_s == cost`; `approximate` is False.

### ASTAR-02 Fewer expansions than Dijkstra (Must)
`dijkstra(...)` = `astar(..., use_heuristic=False)` (same cost, same expansion count). On a 21 x 21 lattice query
across the middle row, A* expands at most half as many labels as Dijkstra; over 30 random OD pairs on a random
graph, total A* expansions < total Dijkstra expansions, with equal costs.

### ASTAR-03 Never uses blocked edges (Must)
No edge of a returned route is entered while blocked (`cost_model.blocked(e, depart_s) is False` and its risk is
below the threshold at that slot). With a static wall of blocked edges the cost equals Dijkstra on the graph
without them.

### ASTAR-04 Exact with time-varying risk and waiting (Must)
On 30-node random graphs whose clock times are whole minutes (slope 0) and whose risk, blocking and node risk vary
by slot, the A* cost equals an independent brute-force time-expanded Dijkstra over (node, minute) states using the
same cost, clock and wait cost (1e-9 relative), with `approximate` False and None exactly when the brute force finds
no path. Hand cases: (a) an edge closed for slots 0-1 on the only path -> the route waits; cost = arrive = 1860 s;
with `allow_wait=False` -> None; starting at t0 = 1800 s needs no wait; (b) the Pareto case (a fast arrival that must
wait in a hazardous cell vs a slower safe arrival; 450 s edges, so it is run with `wait_step_s=30`) returns the slow
path at cost 960.

### ASTAR-05 Returns None when the goal is cut off (Must)
All edges into the goal blocked for the whole window, or the goal in another connected component -> None (with or
without the heuristic).

### ASTAR-06 Metro-sized query time (Must, DESIGN target 1 s)
On a 300 x 300 lattice (90,000 nodes, 100 m spacing, zero risk) a cross-city query (260 rows, 230 columns apart)
returns the exact cost in under 1 s (best of 3, after one warm-up call; marked `slow`).

### ASTAR-12 Multi-goal search (Must for "nearest safe shelter")
With several goals, the route ends at one of them and its cost equals the minimum of the single-goal costs
(None iff all are unreachable). `origin` in `goals` -> zero-edge route, cost 0, `arrive_s = start_s = t0_s`.

### ASTAR-13 Label dominance and cap (Should)
`dominates(t1, g1, t2, g2, wait)` is `t1 <= t2 and g1 + wait(t1, t2) <= g2`. The Pareto case of ASTAR-04 needs two
labels at one node; with `max_labels=1` the result is None or has `approximate=True` and cost >= the exact cost.

### ASTAR-15 Brute-force time-expanded checker in code (Should)
`time_expanded_dijkstra(graph, cost_model, origin, goal, t0_s=0, *, step_s=60, horizon_s=None)` is plain Dijkstra
over (node, step) states (wait one step, or traverse an edge with arrival rounded up to the next step); returns
`(cost, arrival)` (ties: earliest arrival) or None. It reproduces the ASTAR-04 hand cases and the independent
fixture reference.

### ASTAR-16 Route fields are consistent (Should)
`len(nodes) = len(edges) + 1`; consecutive edges connect `nodes`; `depart_s[i]` = previous arrival + `waits_s[i]`;
`arrive_s` = last departure + its clock time; `cost` = sum of `cost(e, depart)` + sum of wait costs; `expansions > 0`;
`travel_s = arrive_s - start_s`; `start_s = t0_s`.

### ASTAR-17 Input errors (Should)
`ValueError` for: origin or goal outside `[0, N)`, empty goals, `t0_s < 0`, `max_labels < 1`, `wait_step_s <= 0`,
`cost_model.n_edges != graph.n_edges`.

## Requirements: planner (`routing/planner.py`)

### PLAN-01 Safe-route API (Must)
`plan_route(graph, cost_model, origin, goals, t0_s=0, *, hardened=(), hardened_max_drive_s=300,
sip_hazards=(TORNADO, WILDFIRE) indices) -> PlanResult`. Without a shelter-in-place trigger: kind `"route"` with
the A* route (same cost), `exposure_risk_min = route_exposure(route, cost_model)`, `hazard_arrival_s = None`;
kind `"no_route"` (route None) when no goal is reachable.

### PLAN-02 Baseline planners and the risk-source switch (Must)
`RiskSource` = model | baseline | observed | none | current-only (`--risk-source`) is the `StrEnum` defined once in
`outrun.core.risk` (RISK-08) and imported here and in the app. `transform_risk(risk, source, t0_slot=0)` never
modifies its input: none -> zeros; current-only -> every slot = slot `t0_slot`; model/baseline/observed -> a copy.
`plan_shortest_time` = planner (a): A* with zero risk (pure free-flow time). `plan_avoid_current_only` = planner
(a'): the **one** current-only freeze, done at the CostModel level: `cost_model.with_risk(transform_risk(r,
CURRENT_ONLY, t0_slot=cost_model.slot(t0_s)), transform_risk(q, CURRENT_ONLY, t0_slot=...))`, so a request with
`t0_s` after the scenario issue time freezes the risk of the request's own slot. On the three-road fixture, (a) takes the fast road, (a')
takes the road that is clear now but hazardous later, and the forecast planner waits 60 s and then takes the fast road
(cost 180).

### PLAN-03 Replay: plan on forecast, score on observed (Must for DESIGN 6.2)
`replay(route, cost_model) -> ReplayScore(travel_s, cost, exposure_risk_min, blocked_edges, intersects_hazard)`:
re-drives the route's edges and waits from `start_s` with the scoring model's clock. Blocked edges are still
traversed: counted in `blocked_edges` (blocked at entry), costed as clock x risk multiplier (never inf).
`replay_trace(route, cost_model) -> RouteTrace` is that re-driven trace (EVAL-09 `RouteTrace` on the scenario
clock: `t_start_s = route.start_s`, `wait_nodes = route.nodes[:-1]`). `exposure_risk_min` and `intersects_hazard`
are `score_route(replay_trace(route, cost_model), cost_model.combined_risk_table(), cost_model.blocked_table(),
slot_s=cost_model.slot_s, node_risk=cost_model.combined_node_risk_table(max(route.nodes) + 1))`'s
`exposure_risk_min` and `intersects` (threshold 0.5, ASTAR-20; the `n_nodes` argument only matters when the model
has no node risk). Replaying on the planning model reproduces `travel_s` and `cost`.

### PLAN-04 Shelter-in-place rule (Must)
Thresholds per SIP hazard: the cost model's `block_at` for it, or 0.5 if NaN. Footprint = nodes whose node risk
reaches a threshold in any slot. `hazard_arrival_s` = start of the earliest slot (>= slot(t0)) in which the origin's
node risk reaches a threshold (None if never; out-of-range hazard indices are ignored). If it is not None,
`footprint_exit_s` = arrival time at the first route node outside the footprint (origin counted at departure; inf
if none). Evacuate (kind `"route"`) iff `footprint_exit_s < hazard_arrival_s`. Otherwise pick, among `hardened`
nodes, the one with the smallest A* travel time that is <= `hardened_max_drive_s` (5 min, configs) and arrives no
later than `hazard_arrival_s` (ties: lowest id) -> kind `"hardened_facility"` with `facility` and its route; if
none qualifies -> kind `"shelter_in_place"`, route None. Smoke, storm, flood and ice never trigger the rule.

### PLAN-05 Exposure in risk-minutes (Should)
`route_exposure(route, cost_model)` = `score_route(route_trace(route), cost_model.combined_risk_table(),
cost_model.blocked_table(), slot_s=cost_model.slot_s, node_risk=cost_model.combined_node_risk_table(max(route.nodes)
+ 1)).exposure_risk_min` (EVAL-09, ASTAR-20): scenario-clock slots (a route starting at `start_s > 0` is charged
for the slots it drives through), traversals charged at `max_h r` of the edge (unweighted), waits at `max_h` node
risk of the node waited at (zero when the model has no node risk). Empty route -> 0.
