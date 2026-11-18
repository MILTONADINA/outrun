# 05: Shelter assignment (capacitated min-cost flow)

Module: `src/outrun/optimize/shelters.py` (uses `td_travel_times_to` from `src/outrun/routing/emergency.py`, EMS-01).
Tests: `tests/unit/test_shelters*.py`. Design: DESIGN 4.4 (source of truth), 4.1 (facilities), 1.2 (interfaces).

Purpose: send every shelter-seeking person in the evacuation polygon to a safe shelter without exceeding any
shelter's capacity, minimising total travel time, and compare the exact optimum with two naive baselines.

Shared conventions for this spec:
- Zones are Census block groups, indexed `0..Z-1`; shelter sites are indexed `0..S-1` in the order of the site list.
- Travel costs are **seconds** (float, `inf` = unreachable). Minutes appear only in metrics (`*_minutes`).
- Road graphs are `networkx.DiGraph`s following the EMS graph convention (07-emergency.md, "Graph convention"),
  built from the RoadGraph and the profile's CostModel by `routing.cost.to_td_digraph` (ASTAR-19), so blocking,
  profile weights and `lam` reach this module.
- `Hazard` and `Profile` are imported from `outrun.core.risk` and `outrun.core.policy`; never redefined.
- Every parameter that has a value in `configs/planning.yaml` (`demand.*`, `shelters.*`) defaults to `None`,
  meaning "read it from the config"; tests always pass explicit values.
- People counts are integers (`np.int64`). Flows are integers.

**Must** (proposal rows "optimization" and "shelter assignments"): SHEL-01..SHEL-06.
**Should** (report metrics and the optimality claim): SHEL-07, SHEL-08.

## Requirements

### SHEL-01 Shelter-seeker demand per zone (`shelter_demand`)
`shelter_demand(population: np.ndarray, rate: float | None = None) -> np.ndarray`.
- Returns `np.int64` array of the same shape: `floor(population * rate + 0.5)` (round half up) per zone.
- `rate` defaults to `demand.public_shelter_rate` (0.10).
- Raises `ValueError` if any population is negative or NaN, or if `rate` is outside `[0, 1]`.
- Zero population gives zero demand. An empty array gives an empty `int64` array.

### SHEL-02 Shelter supply with assumed fallbacks (`build_shelter_supply`)
`Facility(id: str, kind: Literal["shelter", "school", "fire_station", "hospital"], node: int, x: float, y: float,
capacity: int | None)` (frozen dataclass; `x, y` in metres on GRID). The kind names equal DATA-19's
`FACILITY_KINDS`; `Facility.from_record(record, node, x, y)` builds one from a DATA-19 row (`facility_id`, `kind`,
`capacity`) and maps a NaN (or missing) capacity to `None` and a finite one to `int`; an unknown kind raises
`ValueError`. `ShelterSite(id, kind, node, x, y, capacity: int,
assumed: bool)` (frozen dataclass).
`build_shelter_supply(facilities, min_shelters=None, school_capacity=None, fire_station_capacity=None) -> list[ShelterSite]`.
- Every `kind == "shelter"` facility with a capacity > 0 becomes a site with `assumed=False`, in input order.
  Shelters with `capacity` `None` or `<= 0` are skipped.
- If fewer than `min_shelters` (config `shelters.min_shelters_before_fallback`, 20) shelter sites result, every
  school and then every fire station is appended (input order within each kind), each with `assumed=True`.
  A school's capacity is its own capacity if given and > 0, else `school_capacity` (500); a fire station's is its own
  if given and > 0, else `fire_station_capacity` (50).
- If there are at least `min_shelters` shelter sites, no school or fire station is added.
- Hospitals are never sites (medical needs only).
- Order of the result: shelters, then schools, then fire stations.

### SHEL-03 Remove unsafe shelters (`filter_unsafe_sites`)
`filter_unsafe_sites(sites, site_risk: Mapping[Hazard, np.ndarray], profile: Profile, threshold: float | None = None)
-> list[ShelterSite]`.
- `site_risk[h][i]` is the maximum probability of hazard `h` at site `i`'s cell over the routing window; each array
  has length `len(sites)` (else `ValueError`). Missing hazards count as 0.
- A site is removed if `r > threshold` (strict; `threshold` defaults to `shelters.remove_if_hazard_r_above`, 0.5) for
  `WILDFIRE`, `TORNADO` or `FLOOD`, or for `SMOKE` when `profile is Profile.SENSITIVE`. `STORM` and `ICE` never
  remove a site; `SMOKE` never removes one for `CIVILIAN` or `EV`.
- Kept sites keep their relative order. The input list is not modified.

### SHEL-04 Zone-to-shelter travel-time matrix (`travel_cost_matrix`)
`travel_cost_matrix(graph, zone_nodes: Sequence[int], shelter_nodes: Sequence[int], depart_s: float = 0.0,
weight: str = "time", slot_s: float | None = None) -> np.ndarray`.
- Returns float array `(Z, S)`: entry `[z, s]` is the time-dependent earliest arrival time (seconds after departure)
  from `zone_nodes[z]` departing at clock `depart_s` to `shelter_nodes[s]`, under the EMS-01 semantics. One
  reverse search per shelter (`td_travel_times_to(graph, shelter, zone_nodes, ...)`) is the intended method.
- With constant (scalar) edge weights it equals networkx Dijkstra lengths exactly (to 1e-9).
- Unreachable pairs are `inf`. A zone node equal to the shelter node gives 0.
- A node missing from the graph raises `KeyError`.

### SHEL-05 Exact capacitated assignment by min-cost flow (`solve_min_cost_flow`)
`solve_min_cost_flow(demand: np.ndarray (Z,), capacity: np.ndarray (S,), cost_s: np.ndarray (Z, S),
unserved_factor: float | None = None) -> ShelterAssignment`.
`ShelterAssignment(method: str, flow: np.ndarray (Z, S) int64, unserved: np.ndarray (Z,) int64)` (frozen dataclass).
- Network (OR-Tools `SimpleMinCostFlow`): source -> zone `z` (capacity `demand[z]`, cost 0); zone `z` -> shelter `s`
  for every **finite** `cost_s[z, s]` (unbounded capacity, cost `round(cost_s[z, s])` as an integer); shelter `s` ->
  sink (capacity `capacity[s]`, cost 0); zone `z` -> sink "unserved" arc (capacity `demand[z]`, cost
  `U = ceil(unserved_factor * max(1, max finite cost))`; `unserved_factor` defaults to
  `shelters.unserved_cost_factor`, 10). If no cost is finite, `U = ceil(unserved_factor)`.
  `unserved_cost(cost_s, unserved_factor=None) -> int` returns this `U`.
- All demand is routed (always feasible). The result minimises `sum(flow * round(cost_s)) + U * sum(unserved)`
  exactly (optimal over all integer flows obeying the constraints below). Tests use integer-valued costs.
- Constraints of the result: `flow >= 0`; `flow.sum(axis=1) + unserved == demand`; `flow.sum(axis=0) <= capacity`;
  `flow[z, s] == 0` wherever `cost_s[z, s]` is `inf`.
- `method == "mcf"`. Shape mismatches, negative demand/capacity or negative/NaN costs raise `ValueError`.
- `Z == 0` or `S == 0` is allowed (with `S == 0`, everything is unserved).

### SHEL-06 Baselines: nearest by straight line and greedy (`assign_nearest`, `assign_greedy`)
`assign_nearest(demand, capacity, zone_xy: (Z, 2), site_xy: (S, 2), cost_s) -> ShelterAssignment` (`method="nearest"`):
- Each zone sends all its demand to the site with the smallest Euclidean distance (ties: lowest site index),
  **ignoring capacity and reachability**. `unserved` is all zeros (overflow is reported by SHEL-07 instead).
`assign_greedy(demand, capacity, cost_s) -> ShelterAssignment` (`method="greedy"`):
- Zones are processed in descending demand order (ties: lower zone index first). Each zone fills sites in ascending
  `cost_s[z, :]` order (ties: lowest site index), skipping `inf` costs, taking `min(remaining demand, remaining
  capacity)` at each site. Demand left after all finite-cost sites is `unserved`.
- Result obeys all SHEL-05 constraints (capacity respected).
- `S == 0` (nearest) raises `ValueError`; (greedy) gives everything unserved.

### SHEL-07 Assignment metrics (`assignment_metrics`)
`assignment_metrics(assignment, capacity, cost_s, exposure: np.ndarray (Z, S) | None = None) -> ShelterMetrics`
(frozen dataclass with the fields below).
- `served_persons = flow.sum()`; `unserved_persons = unserved.sum()`.
- `total_person_minutes = sum(flow * cost_s) / 60` over cells with `flow > 0` (if any such cell has `inf` cost, the
  total is `inf`).
- `max_minutes = max(cost_s[z, s]) / 60` over cells with `flow > 0`; 0.0 if no flow.
- `overflow_persons = sum_s max(0, flow[:, s].sum() - capacity[s])`; `overflow_pct = 100 * overflow_persons /
  demand_total` where `demand_total = served + unserved` (0.0 if `demand_total == 0`).
- `mean_exposure = sum(flow * exposure) / served_persons`; `nan` if `exposure` is `None` or nothing is served.

### SHEL-08 Min-cost flow dominates the capacity-respecting baseline
For any instance, the SHEL-05 objective `sum(flow * cost) + U * unserved` of `solve_min_cost_flow` is `<=` the same
objective (same `U`) evaluated on `assign_greedy`'s result. When total capacity reachable by each zone suffices,
`solve_min_cost_flow` has `overflow_persons == 0` while `assign_nearest` may overflow; `solve_min_cost_flow`
never overflows.
