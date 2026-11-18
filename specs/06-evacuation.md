# 06: Whole-city evacuation (simulated annealing with congestion)

Module: `src/outrun/optimize/evacuation.py`.
Tests: `tests/unit/test_evacuation*.py`. Design: DESIGN 4.5 (source of truth), 4.2 (BPR, slots), 6.2 (metrics).

Purpose: choose, for every packet of evacuating vehicles, one of `k` diverse risk-aware paths and a departure wave so
that total congested travel time, hazard exposure and clearance time are low, then show that this system optimum beats
the user equilibrium (MSA) and the all-shortest / all-safest baselines under **one shared evaluator**.

Shared conventions for this spec:
- Internal units are seconds and vehicles. The objective `J` and its terms are reported in **vehicle-hours**
  (`veh-h`), clearance in hours (`clearance_h`) and minutes only in `evac_metrics`.
- Wave `w` (0-based) departs at clock `w * wave_s` after t0 (`wave_s` = `evacuation.wave_minutes` * 60 = 1800 s).
  Wave intervals are half-open: `[j * wave_s, (j + 1) * wave_s)`.
- Risk slots are 15 min: the slot of clock `t` (seconds after t0) is `min(floor(t / slot_s), n_slots - 1)`
  with `slot_s = 900`.
- Paths inside an `EvacInstance` are 1-D `int` arrays of **edge indices** `0..E-1` in travel order.
- Every parameter that has a value in `configs/planning.yaml` (`demand.*`, `evacuation.*`) defaults to `None`,
  meaning "read it from the config"; tests always pass explicit values.
- Road graphs passed in are `networkx.DiGraph`s following the EMS graph convention (07-emergency.md), built by
  `routing.cost.to_td_digraph` (ASTAR-19). `diverse_paths` node lists become `EvacInstance.candidates` edge-index
  arrays with `routing.cost.node_path_to_edges`; `edge_risk` is the CostModel's `weighted_risk_table()`.

Instance (pure data, frozen dataclass `EvacInstance`):
`edge_t0` (E,) free-flow seconds > 0; `edge_cap` (E,) veh/h > 0; `edge_risk` (E, n_slots) `>= 0` (profile-weighted
hazard sum per slot); `vehicles` (P,) int > 0; `candidates`: list of length P, each a list of edge-index arrays (an
empty list = the packet has no path and is **unserved**); `n_waves` (4), `wave_s` (1800), `slot_s` (900),
`bpr_alpha` (0.15), `bpr_beta` (4), `lambda_E` (1.0), `lambda_clear_per_vehicle` (0.1), `clearance_percentile` (95),
`unserved_penalty` (`M`, default 1e6 veh-h per vehicle), `zones` ((P,) zone index of each packet, used only by the
SA swap move; `None` = packet index). `EvacEvaluator(instance)` resolves `None` fields from the
config and raises `ValueError` on shape mismatches or out-of-range edge indices.

**Must** (proposal rows "optimization" and "whole-city evacuations"): EVAC-01..EVAC-08.
**Should** (report metrics): EVAC-09, EVAC-10.

## Requirements

### EVAC-01 Zones to vehicle packets (`make_packets`)
`make_packets(population: np.ndarray, compliance=None, persons_per_vehicle=None, packet_max=None) -> list[Packet]`.
`Packet(zone: int, vehicles: int)` (frozen dataclass).
- Vehicles of zone `z`: `v_z = floor(population[z] * compliance / persons_per_vehicle + 0.5)` (defaults 0.9, 2.5).
- A zone with `v_z > 0` is split into `n = ceil(v_z / packet_max)` packets (default 50) whose sizes differ by at most
  1 and sum to `v_z`; the larger packets come first. Zones with `v_z == 0` give no packet.
- Output order: by zone index, then packet order.
- `ValueError` for negative/NaN population, `compliance` outside `[0, 1]`, `persons_per_vehicle <= 0` or
  `packet_max < 1`.

### EVAC-02 k diverse candidate paths (`diverse_paths`)
`diverse_paths(graph, source, target, k=None, penalty=None, weight="cost") -> list[list[int]]` (node paths).
- Runs exactly `k` (default 4) rounds. Round 1 is a shortest path under `weight` (a scalar risk-aware cost attribute,
  default name `"cost"`, separate from the EMS time weight; a missing value counts as 1.0, not as blocked). After each round, the working weight of every edge on that round's path is multiplied by `penalty` (default
  1.4); penalties compound across rounds. The input graph is not modified.
- A round whose path equals an earlier one is not added again, so `1 <= len(result) <= k` when `target` is reachable.
- Every returned path starts at `source`, ends at `target`, follows existing edges and has no repeated node.
  The first path has the networkx shortest-path length (ties may pick any shortest path).
- `source == target` gives `[[source]]`. Unreachable `target` gives `[]`. Unknown nodes raise `KeyError`.
  `k < 1` or `penalty < 1` raises `ValueError`.

### EVAC-03 Wave-BPR loading (`bpr_time`, `EvacEvaluator.loads`)
`bpr_time(t0_s, flow, capacity, alpha=None, beta=None) -> np.ndarray`: elementwise
`t0_s * (1 + alpha * (flow / capacity) ** beta)` (defaults 0.15, 4). `capacity <= 0` raises `ValueError`.
`EvacEvaluator(instance).loads(plan) -> np.ndarray (n_load_waves, E)` (vehicles per wave per edge):
- A served packet `p` with path `P = candidates[p][k_p]` and wave `w_p` has free-flow path time
  `T0_p = sum(edge_t0[P])` and traversal window `[w_p * wave_s, w_p * wave_s + T0_p)`.
- It loads **every** wave `j` whose interval intersects the window, i.e. `j = w_p .. max(w_p, ceil((w_p * wave_s +
  T0_p) / wave_s) - 1)`, adding its full `vehicles` to every edge of `P` in each of those waves.
- `n_load_waves = max(n_waves, 1 + largest loaded j)`.
- Per-wave capacity of edge `e` is `edge_cap[e] * wave_s / 3600` (vehicles per wave); congested time of edge `e` in
  wave `j` is `bpr_time(edge_t0[e], loads[j, e], cap_wave[e])`.
- Unserved packets (no candidate paths) load nothing.

### EVAC-04 Objective J (`EvacEvaluator.evaluate`)
`evaluate(plan) -> EvacScore` (frozen dataclass: `J, congestion_vehh, exposure_vehh, clearance_h, unserved_veh,
arrivals_s, loads`). With `t[j, e]` the congested times of EVAC-03:
- `congestion_vehh = sum_j sum_e loads[j, e] * t[j, e] / 3600`.
- Exposure of served packet `p`: walk `P` in order with free-flow times; edge `e` is entered at clock
  `w_p * wave_s + (sum of edge_t0 of the edges before it)`; `Exp_p = sum_e edge_t0[e] * edge_risk[e, slot(enter)]`
  (risk-seconds). `exposure_vehh = sum_p vehicles_p * Exp_p / 3600`.
- Arrival of served packet `p`: `arrivals_s[p] = w_p * wave_s + sum_{e in P} t[w_p, e]` (congested times of its
  departure wave). Unserved packets have `arrivals_s[p] = nan`.
- `clearance_h`: vehicle-weighted percentile `q` (= `clearance_percentile`, 95) of served arrivals: sort arrivals
  ascending and take the first arrival `a` at which cumulative vehicles `>= q / 100 * served vehicles`; divided by
  3600. 0.0 if nothing is served.
- `unserved_veh` = vehicles of packets with no candidate path.
- `J = congestion_vehh + lambda_E * exposure_vehh + lambda_C * clearance_h + M * unserved_veh` with
  `lambda_C = lambda_clear_per_vehicle * (total vehicles of all packets)` and `M = unserved_penalty`.
- `EvacPlan(k: np.ndarray, w: np.ndarray)` (int arrays of length P). Plans with `k[p]` out of range for a served
  packet, or `w` outside `0..n_waves-1`, raise `ValueError` (`k` of unserved packets is ignored).

### EVAC-05 Incremental move evaluation (`EvacEvaluator.delta`)
`delta(plan, p, k, w) -> float` returns `evaluate(plan') .J - evaluate(plan).J` where `plan'` equals `plan` with
packet `p` moved to path `k` and wave `w`; equal within `1e-9 * max(1, |J|)`. It does not modify `plan`. A move to the
packet's current `(k, w)` returns 0.0. (Only the old and new path edges change loads; the implementation should exploit
this, but correctness is what is tested.)

### EVAC-06 Simulated annealing (`initial_temperature`, `simulated_annealing`)
`initial_temperature(uphill_deltas, accept=None) -> float` returns `-mean(uphill_deltas) / ln(accept)` (accept
defaults to `evacuation.sa.warmup_accept`, 0.8); an empty input returns 1.0; `accept` outside `(0, 1)` or a negative
delta raises `ValueError`.
`simulated_annealing(evaluator, init=None, *, iterations=None, alpha=None, warmup_moves=None, warmup_accept=None,
move_probs=None, seed=0, zone_adjacency=None, callback=None, callback_every=100) -> SAResult`
(`SAResult(best_plan, best_J, init_J, history, T0)`; `T0` is the initial temperature actually used).
- Default `init`: every packet `k = 0, w = 0`. T0 from `initial_temperature` over the uphill deltas of `warmup_moves`
  random moves (not applied). Geometric cooling `T <- alpha * T` per iteration. Metropolis acceptance.
- Moves (probabilities `move_probs`, default 0.75 / 0.20 / 0.05): re-pick a packet's path, re-pick its wave, swap the
  waves of two packets whose zones form a pair in `zone_adjacency` (list of zone pairs, zones from `instance.zones`; if `None` or empty, that move's mass
  goes to re-pick wave).
- Guarantees tested: same seed and inputs give an identical result; `best_J == evaluator.evaluate(best_plan).J`
  (1e-9 relative); `best_J <= init_J`, and `best_J < init_J` whenever some single path or wave re-pick of
  `init` strictly lowers J (such a move is accepted when proposed, and the iteration budget makes proposing it
  practically certain); every entry of `best_plan` is valid (EVAC-04); `history` is the list of
  `(iteration, current_J, best_J, temperature)` recorded at each iteration `i` in `1..iterations` with
  `i % callback_every == 0`, where `temperature = T0 * alpha**i` (the temperature after iteration `i`'s cooling
  step); its `best_J` column is non-increasing; `callback(iteration, current_J, best_J, temperature)` is called at
  the same iterations with the same values. These rows feed the app's SA progress events and
  `results/evac_sa_trace.json` (`{"iter", "objective": current_J, "temperature"}`, APP-05) directly, so the app
  never re-implements SA.

### EVAC-07 MSA user equilibrium (`msa_equilibrium`)
`msa_equilibrium(evaluator, iterations=None) -> EvacPlan` (default `evacuation.msa_iterations`, 50).
- Choices per served packet are all `(k, w)` pairs. Shares start as all-or-nothing on the free-flow user cost.
- User cost of a choice: `sum_{e in P_k} t[w, e] + lambda_E * Exp(k, w)` (seconds) under the **fractional** loads of
  the current shares (each choice contributes `share * vehicles`, loading waves as in EVAC-03). No departure-time
  penalty and no externality term (this is what makes it a user equilibrium).
- Iteration `n = 1..iterations`: all-or-nothing target = each packet's minimum user-cost choice (ties: lowest `k`,
  then lowest `w`); `shares <- shares + (target - shares) / (n + 1)`.
- Final plan: each packet's largest-share choice (ties: lowest `k`, then lowest `w`). Deterministic; valid plan.
- `iterations < 1` raises `ValueError`.

### EVAC-08 Baselines and lower bound (`baseline_plan`, `EvacEvaluator.lower_bound`)
`baseline_plan(evaluator, method: Literal["shortest", "safest"]) -> EvacPlan`, every packet in wave 0:
- `"shortest"`: candidate with minimum free-flow time `T0` (ties: lowest `k`) (all-take-shortest, congestion- and
  risk-blind).
- `"safest"`: candidate with minimum `T0 + lambda_E * Exp(k, 0)` (ties: lowest `k`) (risk-aware, congestion-blind).
- Unknown method raises `ValueError`.
`lower_bound() -> float` (veh-h): `sum_p vehicles_p * min_{k, w} (T0_{p,k} + lambda_E * Exp_p(k, w)) / 3600` over
served packets. For every valid plan of an instance with no unserved packet, `evaluate(plan).J >= lower_bound()`.
All baselines, MSA and SA are scored by the same `EvacEvaluator`.

### EVAC-09 Exposure-weight ablation
An instance with `lambda_E = 0` scores `J` without the exposure term, but `EvacScore.exposure_vehh` is still computed
and equals the value from the same plan with `lambda_E = 1`. SA run on the `lambda_E = 0` instance is the "SA
ablation (d)" of DESIGN 4.5.

### EVAC-10 Evacuation metrics (`evac_metrics`)
`evac_metrics(evaluator, plan) -> dict[str, float]` with keys `vehicle_hours` (= `congestion_vehh`),
`exposure_veh_h` (= `exposure_vehh`), `clearance_min` (= `clearance_h * 60`), `max_vc` (max over waves and edges of
`loads / cap_wave`), `unserved_veh`, `J`, `gap_to_lb_pct` (`100 * (J - LB) / LB`, `nan` if `LB == 0`).
