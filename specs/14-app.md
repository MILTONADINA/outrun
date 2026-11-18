# 14: Interactive map app and command-line interface

Prefixes: `APP` (web app, API, offline demo, static export) and `CLI` (command-line interface).
Design: DESIGN 5 (app), DESIGN 2.2 (`cli.py` row), DESIGN 7.3 (disk check).

Modules:

| Module | Contents |
|---|---|
| `src/outrun/app/server.py` | `create_app(...) -> FastAPI`, all `/api/*` routes, static page mount |
| `src/outrun/app/bundle.py` | Scenario bundle layout, `load_scenario_meta`, `list_scenarios`, `scenario_dir`, `read_metrics` |
| `src/outrun/app/models.py` | Pydantic request models `RouteRequest`, `DispatchRequest` (pure data, fully defined) |
| `src/outrun/app/backend.py` | `AppBackend` protocol (live A*, dispatch, SA, city graphs) and `DefaultBackend` (wires the routing/optimize areas) |
| `src/outrun/app/sse.py` | `format_sse` (server-sent events wire format) |
| `src/outrun/app/export.py` | `export_static(...)` for the GitHub Pages recruiter demo |
| `src/outrun/app/static/` | `index.html`, `app.js`, `style.css`, `vendor/` (MapLibre GL JS 4, pmtiles.js, deck.gl UMD; vendored, never a CDN) |
| `src/outrun/cli.py` | `build_parser()`, `main(argv)`, `HANDLERS`, `check_disk`, one `cmd_*` handler per command |

Tests: `tests/unit/test_app.py` (FastAPI `TestClient`, synthetic bundle in `tmp_path`, fake backend; includes
APP-10's offline socket check), `tests/unit/test_cli.py` (parsing, dispatch with monkeypatched handlers) and the
browser-level e2e test `tests/e2e/test_offline.py` (APP-16: Playwright, marked `e2e`, opt-in with
`make e2e`; it is skipped when Playwright or its Chromium is not installed, so collection stays clean).

**Must for the proposal ("interactive map app"):** APP-01..APP-08, APP-16 and APP-17 (listed in the overview
matrix): without APP-16/17 every JSON endpoint could pass while no map is ever drawn. All other IDs in
this file are Must for the project (demo safety, offline mode, CLI) but support every proposal row rather than one.

## Shared definitions

- **Hazard names** in URLs, JSON and CLI arguments are `h.name.lower()` for `h` in `outrun.core.risk.Hazard`:
  `storm, tornado, flood, wildfire, ice, smoke`. **Profile names** are `p.name.lower()` for
  `outrun.core.policy.Profile`: `civilian, sensitive, ev`. These lists are derived from the core enums, never
  duplicated as literals in `src/`.
- **Overlay sources:** `model`, `baseline`, `observed` (the `RiskLayer.kind` values shown as `forecast` are served
  under the name `model`). **Risk sources** for live routing: `model, baseline, observed, none, current-only`.
- **Scenario id:** matches `^[a-z0-9][a-z0-9_-]{0,63}$`. Any request whose scenario id does not match is answered
  `404` without touching the file system (APP-14).
- Coordinates in the API are `[lon, lat]` in degrees (WGS84). Times are ISO 8601 with a UTC offset.
- Every error response is JSON `{"detail": ...}` (FastAPI default). Validation failures are `422`.

### Scenario bundle layout (written by `outrun scenario`, T22; read by the app and `export-static`)

```
<scenarios_dir>/<id>/
  scenario.json                              # metadata, see below
  overlays/<source>/<hazard>/<slot:02d>.png  # RGBA overlay rendered from the RiskStack (RISK-07)
  results/shelters_<method>.json             # method in mcf|nearest|greedy; GeoJSON FeatureCollection + "metrics"
  results/evac_<method>.json                 # method in sa|msa|shortest|safest; GeoJSON FeatureCollection + "metrics"
  results/evac_sa_trace.json                 # list of SA progress dicts {"iter", "objective", "temperature"}
  results/routes_preset.json                 # cached routes for the preset origins (APP-11)
  risk.zarr, graph.npz, ...                  # used by the live backend only; the app never reads them directly
```

`scenario.json` keys (all required unless marked): `id` (str, equals the folder name), `title` (str), `t0`
(ISO 8601 UTC), `city` (str), `bbox` (`[west, south, east, north]` lon/lat, west < east, south < north),
`hazards` (list of hazard names), `held_out` (bool), `synthetic` (bool), `sources` (list of str: data-source
labels for the card), optional `n_slots` (int, default 12) and `slot_minutes` (int, default 15). These mirror
`outrun.core.scenario.Scenario` plus display fields.

## Requirements

### APP-01 Scenario list (`GET /api/scenarios`)
`200` with a JSON list, one entry per **valid** bundle in `scenarios_dir`, sorted by `id`. Each entry has exactly
the keys of `scenario.json` with defaults filled in (`n_slots`, `slot_minutes` always present) plus
`overlay_sources`: the sorted list of overlay sources (`model|baseline|observed`) that exist as folders under
`overlays/`. A folder is skipped (not listed, no error) when `scenario.json` is missing, is not valid JSON, lacks a
required key, has an `id` different from the folder name, has an id that fails the id pattern, or has an invalid
`bbox`. Files (not folders) at the top level are ignored. A missing or empty `scenarios_dir` gives `[]`.
`bundle.list_scenarios(scenarios_dir)` returns the same list.

### APP-02 Overlay PNG (`GET /api/scenarios/{id}/overlay/{hazard}/{slot}.png?source=`)
`source` defaults to `model`. Returns `200`, `Content-Type: image/png`, body byte-identical to
`overlays/<source>/<hazard>/<slot:02d>.png`. Errors: unknown or invalid scenario id -> `404`; hazard not one of
the six names, `source` not one of the three overlay sources, or `slot` not an integer in `[0, n_slots - 1]` ->
`422`; valid request but file absent -> `404`.

### APP-03 Live safe route (`POST /api/route`)
Body: `RouteRequest` = `{"scenario": id, "origin": [lon, lat], "dest": [lon, lat] | "nearest_shelter",
"profile": profile name (default "civilian"), "lambda": float >= 0 or null (null = `lambda_default` from
`configs/hazards.yaml`), "t0": ISO time or null (null = scenario t0), "risk_source": risk source (default
"model")}`. Validation (`422`): lon outside [-180, 180], lat outside [-90, 90], unknown profile or risk source,
negative lambda, missing origin, `dest` neither a pair nor `"nearest_shelter"`, and origin or a coordinate `dest`
outside the scenario `bbox` (detail mentions `bbox`). Unknown scenario -> `404`. On success the server calls
`backend.route(req)` exactly once and returns `200` with the backend dict plus `"source": "live"` and
`"scenario": id`. The backend dict contains at least `outrun` and `shortest` (GeoJSON `Feature`s with
`LineString` geometry, or `null` as below), `eta_min`, `exposure` (`{"predicted", "observed"}`), `blocked_avoided`
(int), `expansions` (`{"astar", "dijkstra"}`) and `shelter_in_place` (bool); the server passes these through
unchanged. When the backend is missing or raises, APP-11 applies.
`DefaultBackend.route` builds every key except `expansions` with one helper,
`backend.route_fields(plan, shortest, planning_model, observed_model, graph) -> dict` (unit-tested without a
backend), from the PLAN-01 `PlanResult`, the planner-(a) `Route` (`plan_shortest_time`, or `None`), the CostModel
the plan used and the observed CostModel (or `None`):
- `shelter_in_place = plan.kind in {"shelter_in_place", "hardened_facility"}`.
- `outrun` = a `LineString` Feature through `(graph.lon[n], graph.lat[n])` of `plan.route.nodes` (for
  `"hardened_facility"` this is the drive to the facility) with properties `{"kind": plan.kind, "facility":
  plan.facility}`; `null` when `plan.route is None` (`"shelter_in_place"`, `"no_route"`). `shortest` is the same
  for the planner-(a) route (`null` if `None`, properties `{"kind": "shortest"}`).
- `eta_min = plan.route.travel_s / 60` (`null` without a route).
- `exposure = {"predicted": route_exposure(plan.route, planning_model), "observed":
  replay(plan.route, observed_model).exposure_risk_min}` (ASTAR-20; `observed` is `null` without an observed model;
  both `null` without a route).
- `blocked_avoided = replay(shortest, planning_model).blocked_edges` (edges of the shortest route that are blocked
  under the planning model, i.e. what the safe route avoided); `0` when `shortest is None`.

### APP-04 Shelter assignment (`GET /api/shelters/{id}?method=`)
`method` in `mcf|nearest|greedy`, default `mcf`. Returns `200` with the JSON content of
`results/shelters_<method>.json` unchanged. Invalid method -> `422`; unknown scenario -> `404`; file absent -> `404`.

### APP-05 City evacuation results and live SA stream
`GET /api/evac/{id}?method=` with `method` in `sa|msa|shortest|safest` (default `sa`) behaves like APP-04 with
`results/evac_<method>.json`.
`GET /api/evac/{id}/run` streams server-sent events (`Content-Type` starts with `text/event-stream`), each built
with `format_sse` (APP-12) with ids 0, 1, 2, ... in order:
- With a backend: the events of `backend.run_evac(id)` in order. That iterator yields `(kind, data)` pairs where
  kind is `"progress"` (data `{"iter", "objective", "temperature"}`) and the last pair is `("done", result)`.
  The server adds `"source": "live"` to the `done` data. If the iterator raises, the server emits one
  `error` event with `{"detail": <message>}` and ends the stream (status stays `200`).
- Without a backend, or if `backend.run_evac` raises before the first event: replay `results/evac_sa_trace.json`
  as `progress` events, then one `done` event with the content of `results/evac_sa.json` plus
  `"source": "cached"`. If either file is absent -> `503` JSON (no stream).
Unknown scenario -> `404`.

### APP-06 Emergency dispatch (`POST /api/dispatch`)
Body: `DispatchRequest` = `{"scenario": id, "incidents": [[lon, lat], ...], "t0": ISO time or null}`. 1 to 3
incidents, else `422`; any incident outside the scenario `bbox` -> `422`; unknown scenario -> `404`. Calls
`backend.dispatch(req)` once and returns its dict plus `"source": "live"`. The backend dict contains at least
`assignment` (list of `{"incident", "unit"}`), `routes` (GeoJSON `FeatureCollection`) and `baseline`
(GeoJSON `FeatureCollection` of the discover-and-reroute baseline). No backend or backend error -> `503`.

### APP-07 Any US city box (`GET /api/city?name=`)
`name` is required, stripped, 1..100 characters after stripping, else `422`. Calls
`backend.city(name, allow_network=not demo)` and returns its dict (at least `name`, `bbox`, `n_nodes`, `n_edges`,
`cached`). `LookupError` from the backend (city not cached and network not allowed, or place not found) -> `404`
with the message as detail. No backend -> `503`.

### APP-08 Model and routing metrics (`GET /api/metrics`)
Returns `200` with `{"metrics": {<file stem>: <parsed JSON>}, "invalid": [<file name>, ...]}` built from
`<reports_dir>/metrics/*.json` (sorted by stem; `invalid` lists files that are not valid JSON, sorted). A missing
folder gives `{"metrics": {}, "invalid": []}`. `bundle.read_metrics(reports_dir)` returns the same dict.

### APP-09 Single page and static assets
`GET /` returns `200` `text/html`, the content of `app/static/index.html`. `GET /static/app.js` and
`GET /static/style.css` return `200`. `index.html` contains an element with `id="map"`, a search box with
`id="city-box"`, and one element with a `data-tab` attribute for each of the screens
`forecast, route, shelters, evac, dispatch, results`. `index.html` references its assets by **relative** paths
(`static/...`, no leading slash) so the same file works under a GitHub Pages sub-path. `index.html`, `app.js` and
`style.css` contain no `http://` or `https://` URL and no protocol-relative `//host` asset reference
(the files under `static/vendor/` are exempt).

### APP-17 Vendored map libraries and the front end actually uses them (unit)
- `app/static/vendor/` contains `maplibre-gl.js`, `maplibre-gl.css` and `pmtiles.js`, each a real library file
  of more than 10 000 bytes (a README or placeholder does not count). `GET /static/vendor/maplibre-gl.js` returns
  `200`.
- `index.html` loads them by relative path: a `<script src="static/vendor/maplibre-gl.js">`, a
  `<link ... href="static/vendor/maplibre-gl.css">`, a `<script src="static/vendor/pmtiles.js">` and
  `<script src="static/app.js">` (attribute order and extra attributes free).
- `app.js` creates the map (`new maplibregl.Map(`), exposes it as `window.outrunMap`, keeps the last drawn safe
  route GeoJSON in `window.outrunState.route`, and calls every API the screens use: it contains the strings
  `/api/scenarios`, `/overlay/`, `/api/route`, `/api/shelters/`, `/api/evac/`, `/api/dispatch`, `/api/city` and
  `/api/metrics`. When no local PMTiles basemap is present it uses a built-in style (a background layer only), so
  the map canvas still renders offline.

### APP-16 Browser e2e: every tab renders a map offline (e2e, `tests/e2e/test_offline.py`)
Marked `@pytest.mark.e2e` (skipped when `playwright` or its Chromium is missing). The test writes a synthetic
bundle (one scenario with bbox `[-97.6, 35.4, -97.4, 35.6]`, 12 storm overlay PNGs, `results/*.json` and a
`routes_preset.json` preset at the bbox centre with dest `"nearest_shelter"`), starts `create_app(...,
backend=None)` with uvicorn on `127.0.0.1` and a free port in a background thread, and drives headless Chromium:
- loading `/` renders a MapLibre canvas (`canvas.maplibregl-canvas` visible) within 15 s and logs no page error;
- clicking each `[data-tab]` element (forecast, route, shelters, evac, dispatch, results) raises no page error;
- on the forecast tab at least one `/overlay/` PNG request completes with status `200`;
- on the route tab, a click at the centre of the map (the map starts fitted to the scenario bbox) issues
  `POST /api/route` answered with `200` (the cached preset, APP-11) and then `window.outrunState.route` is a
  GeoJSON `Feature` with a `LineString` geometry;
- every request the page makes goes to `http://127.0.0.1:<port>/` (or is a `data:`/`blob:` URL): zero
  non-localhost requests.

### APP-10 Offline-safe server
Creating the app and serving every `GET` endpoint of APP-01, 02, 04, 05 (cached stream), 08, 09, 15 and the cached
route of APP-11 succeed while every non-localhost socket connection raises (the `no_network` fixture). Importing
`outrun.app.server` imports no network client library at module level (`osmnx`, `herbie`, `boto3`, `requests` are
not newly added to `sys.modules` by the import).

### APP-11 Cached route fallback (30 preset origins)
When `backend` is `None` or `backend.route` raises, `POST /api/route` (after APP-03 validation) answers from
`results/routes_preset.json`: a list of `{"origin": [lon, lat], "dest": [lon, lat] | "nearest_shelter",
"profile": name, "result": {...}}`. A preset matches when its profile equals the request profile, its origin is
within 250 m of the request origin (haversine, Earth radius 6 371 008.8 m) and its dest is equal (both
`"nearest_shelter"`, or both coordinates within 250 m). Among matches the one with the closest origin wins. The
response is `200` with the preset `result` plus `"source": "cached"`, `"scenario": id` and
`"cached_origin": [lon, lat]`. No match or no file -> `503` with a detail that says no live router and no cached
route are available.

### APP-12 Server-sent events format (`sse.format_sse(event, data, id=None) -> str`)
Returns `"event: <event>\n"`, then `"id: <id>\n"` when `id` is not `None`, then `"data: <json>\n\n"`, where
`<json>` is `json.dumps(data, separators=(",", ":"), sort_keys=True)` (a single line). `event` must match
`^[a-z][a-z_]*$`, else `ValueError`. `id` must be a non-negative int, else `ValueError`.

### APP-13 Static recruiter export (`export.export_static(scenarios_dir, reports_dir, out_dir) -> list[str]`)
Writes, under `out_dir` (created if absent):
- `index.html` (copy of `app/static/index.html`) and `static/**` (copy of `app/static/` without `index.html`);
- `config.json` = `{"mode": "static"}` (tells `app.js` to read files instead of `/api`); `.nojekyll` (empty);
- `data/scenarios.json` = the APP-01 list; `data/metrics.json` = the APP-08 dict;
- for every listed scenario: `data/scenarios/<id>/overlay/<source>/<hazard>/<slot:02d>.png` (byte copies of every
  overlay), and every `results/*.json` file as `data/scenarios/<id>/<name>.json`;
- `.outrun-static.json` = `{"files": [<every path above, relative posix, sorted, excluding itself>]}`.
Returns the same sorted relative path list. Safety: before writing anything, if any target path exists and is not
listed in an existing `.outrun-static.json` in `out_dir`, raise `FileExistsError` and write nothing. Files in
`out_dir` that the export does not target are never modified or deleted, except files listed in the previous
manifest and no longer targeted, which are removed. Running twice gives the same list and byte-identical files.

### APP-14 Path safety
Scenario ids that fail the id pattern (e.g. `..`, `.hidden`, `Bad.Id`, ids with `%2F`) give `404` on every
endpoint that takes an id, and no file outside `scenarios_dir` is ever read or returned. `bundle.scenario_dir(root,
id)` returns `root / id` for a valid id and raises `ValueError` otherwise.

### APP-15 Health and demo mode (`GET /api/health`)
`200` with exactly `{"status": "ok", "demo": bool, "live": bool, "n_scenarios": int}`: `live` is
`backend is not None`, `n_scenarios` is the APP-01 count. `create_app(..., demo=None)` sets demo from the
environment (`OUTRUN_DEMO == "1"`); an explicit `demo=True/False` overrides the environment. `make demo` and
`outrun serve --demo` run with demo on.

### CLI-01 Commands
`build_parser()` returns an `argparse.ArgumentParser` (prog `outrun`) with exactly the sub-commands
`fetch, prep, train, eval, scenario, route, serve, export-static, disk`; the parsed namespace has
`command` = the sub-command name. An unknown command or no command exits with status 2 (`SystemExit(2)`).
`outrun --version` prints `outrun.__version__` and exits 0.

### CLI-02 Options and defaults
Every option not listed is free to add. Defaults (attribute name in brackets):
- `fetch [--source NAME ...] (sources, default [] = all)`, `--manifest (manifest, "data/manifest.yaml")`, `--dry-run (dry_run, False)`.
- `prep DATASET (dataset, str)`, `--limit N (limit, int, None)`.
- `train HAZARD (hazard, one of the six names)`, `--epochs (epochs, int, None = config)`,
  `--device {auto,mps,cpu} (device, "auto")`, `--seed (seed, int, None = config)`, `--resume (resume, False)`.
- `eval --hazard (hazard, six names or "all", default "all")`, `--split {val,test} (split, "val")`,
  `--routing (routing, False)`, `--final (final, False)`.
- `scenario [ID] (scenario_id, None)`, `--all (all, False)`, `--out (out, "scenarios")`.
- `route` see CLI-05; defaults `--profile "civilian"`, `--risk-source "model"` (attribute `risk_source`),
  `--lambda (lam, float, None)`, `--scenario (scenario, None)`, `--out (out, None)`.
- `serve --host (host, "127.0.0.1")`, `--port (port, int, 8000)`, `--demo (demo, False)`,
  `--scenarios (scenarios, "scenarios")`, `--reports (reports, "reports")`.
- `export-static --out (out, "docs")`, `--scenarios (scenarios, "scenarios")`, `--reports (reports, "reports")`.
- `disk --need-gb (need_gb, float, 0.0)`.
Hazard and profile choices come from the core enums (shared definitions); an invalid choice exits with status 2.

### CLI-03 Dispatch and exit codes
`HANDLERS` is a module-level `dict[str, Callable[[argparse.Namespace], int]]` with one entry per command.
`main(argv=None) -> int` parses `argv` (default `sys.argv[1:]`), looks the handler up in `HANDLERS` **at call
time**, calls it once with the namespace and returns its int result. `outrun.__main__.main(argv)` forwards to
`outrun.cli.main` and returns the same code.

### CLI-04 Test-split guard (held-out discipline, DESIGN 6.3)
`outrun eval --split test` without `--final` exits with status 2 (`SystemExit(2)`) from `main` before any handler
is called. With `--final` the `eval` handler is called normally. `--split val` never needs `--final`.

### CLI-05 Route arguments
`route` requires exactly one of `--place NAME` (place, str) or `--bbox W S E N` (bbox, 4 floats), exactly one of
`--dest LON LAT` (dest, 2 floats) or `--nearest-shelter` (nearest_shelter, bool), and `--origin LON LAT`
(origin, 2 floats). `--profile` choices are the profile names, `--risk-source` choices are
`model, baseline, observed, none, current-only`. Violations exit with status 2. In `main`, a bbox with
`W >= E` or `S >= N` also exits with status 2 before the handler is called.

### CLI-06 `serve`
`cmd_serve(args)` calls `outrun.app.server.create_app(scenarios_dir=Path(args.scenarios),
reports_dir=Path(args.reports), demo=args.demo or None)` (`None` lets `OUTRUN_DEMO` decide), then
`uvicorn.run(app, host=args.host, port=args.port)`, both looked up as module attributes at call time, and
returns 0.

### CLI-07 `export-static`
`cmd_export_static(args)` calls `outrun.app.export.export_static(Path(args.scenarios), Path(args.reports),
Path(args.out))` (module attribute, looked up at call time), prints one line with the number of files written,
returns 0. `FileExistsError` is caught: the message is printed to stderr and the exit code is 1.

### CLI-08 Disk check (DESIGN 7.3)
`check_disk(path, need_gb, max_used_gb) -> DiskStatus` reads `shutil.disk_usage(path)` (module attribute) and
returns `DiskStatus(path, total_gb, used_gb, free_gb, need_gb, max_used_gb, ok)` with GB = 1e9 bytes and
`ok = (used_gb + need_gb <= max_used_gb) and (need_gb <= free_gb)`. `need_gb < 0` -> `ValueError`; a missing
path -> `FileNotFoundError`. `cmd_disk(args)` checks the RAW folder (`OUTRUN_RAW`) with the HDD limit and the PROC
folder (`OUTRUN_PROC`) with the SSD limit (limits from config), prints one line per folder, and returns 0 if both
are ok, 1 otherwise.
