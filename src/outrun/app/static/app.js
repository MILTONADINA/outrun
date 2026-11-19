// Outrun front end: plain JavaScript, no build step. Uses only the vendored MapLibre GL JS and pmtiles.js.
// Sections: 1 state + API helpers, 2 map setup, 3 map layers, 4 tabs, 5 each screen, 6 start-up.
"use strict";

// ---------------------------------------------------------------- 1. state and API helpers
const HAZARDS = ["storm", "tornado", "flood", "wildfire", "ice", "smoke"];
const BASE = ".";   // relative, so the page also works under a GitHub Pages sub-path
const $ = (id) => document.getElementById(id);

// window.outrunState is read by the e2e test: route = last drawn risk-aware route (GeoJSON Feature).
const state = window.outrunState = {
  mode: "live",        // "live" (FastAPI) or "static" (exported files, see config.json)
  scenarios: [], scenario: null, tab: "forecast",
  origin: null, dest: null, incidents: [], route: null,
};

function setStatus(msg) { $("status").textContent = msg || ""; }

async function getJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = JSON.stringify((await res.json()).detail); } catch (e) { /* not JSON */ }
    throw new Error(res.status + ": " + detail);
  }
  return res.json();
}

// URL builders: the live API, or the files written by export-static (spec APP-13).
const api = {
  scenarios: () => state.mode === "static" ? BASE + "/data/scenarios.json" : BASE + "/api/scenarios",
  overlay: (hazard, slot, source) => {
    const id = state.scenario.id, nn = String(slot).padStart(2, "0");
    return state.mode === "static"
      ? `${BASE}/data/scenarios/${id}/overlay/${source}/${hazard}/${nn}.png`
      : `${BASE}/api/scenarios/${id}/overlay/${hazard}/${slot}.png?source=${source}`;
  },
  shelters: (method) => state.mode === "static"
    ? `${BASE}/data/scenarios/${state.scenario.id}/shelters_${method}.json`
    : `${BASE}/api/shelters/${state.scenario.id}?method=${method}`,
  evac: (method) => state.mode === "static"
    ? `${BASE}/data/scenarios/${state.scenario.id}/evac_${method}.json`
    : `${BASE}/api/evac/${state.scenario.id}?method=${method}`,
  evacRun: () => `${BASE}/api/evac/${state.scenario.id}/run`,
  metrics: () => state.mode === "static" ? BASE + "/data/metrics.json" : BASE + "/api/metrics",
  city: (name) => `${BASE}/api/city?name=${encodeURIComponent(name)}`,
};

function post(url, body) {
  return getJSON(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

// ---------------------------------------------------------------- 2. map setup
// Basemap: local PMTiles file served by the backend. If the file is absent we fall back to a plain
// background so the map still renders offline.
const TILES_URL = "static/tiles/okc.pmtiles";

function plainStyle() {
  return { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#e9eef2" } }] };
}

function pmtilesStyle() {   // Protomaps basemap layers: land, water, roads, buildings
  const src = "basemap";
  return {
    version: 8,
    sources: { [src]: { type: "vector", url: "pmtiles://" + TILES_URL } },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": "#e9eef2" } },
      { id: "earth", type: "fill", source: src, "source-layer": "earth", paint: { "fill-color": "#f2efe9" } },
      { id: "landuse", type: "fill", source: src, "source-layer": "landuse", paint: { "fill-color": "#d9e6cf" } },
      { id: "water", type: "fill", source: src, "source-layer": "water", paint: { "fill-color": "#a9cce3" } },
      { id: "roads", type: "line", source: src, "source-layer": "roads", paint: { "line-color": "#ffffff", "line-width": 1.2 } },
      { id: "buildings", type: "fill", source: src, "source-layer": "buildings", paint: { "fill-color": "#d8d2c8" } },
    ],
  };
}

async function hasBasemap() {
  try { return (await fetch(TILES_URL, { method: "HEAD" })).ok; } catch (e) { return false; }
}

async function createMap() {
  let style = plainStyle();
  if (await hasBasemap()) {
    maplibregl.addProtocol("pmtiles", new pmtiles.Protocol().tile);
    style = pmtilesStyle();
  }
  const map = new maplibregl.Map({ container: "map", style, center: [-97.5, 35.5], zoom: 9 });
  window.outrunMap = map;
  return map;
}

// ---------------------------------------------------------------- 3. map layers
const EMPTY = { type: "FeatureCollection", features: [] };
let map;

// Add one GeoJSON source plus the layers that draw it.
function addGeo(id, layers) {
  map.addSource(id, { type: "geojson", data: EMPTY });
  for (const layer of layers) map.addLayer({ source: id, ...layer, id: id + "-" + layer.id });
}
const isLine = ["==", ["geometry-type"], "LineString"];
const isPoint = ["==", ["geometry-type"], "Point"];

function addLayers() {
  // Hazard overlay: an image stretched over the scenario bounding box (filled in by loadOverlay).
  map.addSource("overlay", { type: "image", url: transparentPixel(), coordinates: [[0, 1], [1, 1], [1, 0], [0, 0]] });
  map.addLayer({ id: "overlay", type: "raster", source: "overlay", paint: { "raster-opacity": 0.7, "raster-fade-duration": 0 } });

  addGeo("shortest", [{ id: "line", type: "line", paint: { "line-color": "#666", "line-width": 4, "line-dasharray": [2, 2] } }]);
  addGeo("outrun", [{ id: "line", type: "line", paint: { "line-color": "#1971c2", "line-width": 5 } }]);
  addGeo("shelters", [
    { id: "line", type: "line", filter: isLine, paint: { "line-color": "#2f9e44", "line-width": 2 } },
    { id: "pt", type: "circle", filter: isPoint, paint: { "circle-color": "#2f9e44", "circle-radius": 6, "circle-stroke-width": 1, "circle-stroke-color": "#fff" } },
  ]);
  addGeo("evac", [
    { id: "line", type: "line", filter: isLine, paint: { "line-color": "#e8590c", "line-width": 2 } },
    { id: "pt", type: "circle", filter: isPoint, paint: { "circle-color": "#e8590c", "circle-radius": 5 } },
  ]);
  addGeo("dispatch-base", [{ id: "line", type: "line", paint: { "line-color": "#666", "line-width": 3, "line-dasharray": [2, 2] } }]);
  addGeo("dispatch", [{ id: "line", type: "line", paint: { "line-color": "#c2255c", "line-width": 4 } }]);
  addGeo("points", [{ id: "pt", type: "circle", paint: {
    "circle-radius": 8, "circle-stroke-width": 2, "circle-stroke-color": "#fff",
    "circle-color": ["match", ["get", "role"], "origin", "#1971c2", "dest", "#2f9e44", "#c2255c"],
  } }]);
}

function transparentPixel() {   // 1x1 transparent PNG used until the first real overlay is loaded
  return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
}

function setGeo(id, data) { map.getSource(id).setData(data || EMPTY); }

// Which map layers belong to which tab; the others are hidden.
const tabLayers = {
  forecast: ["overlay"], route: ["shortest-line", "outrun-line", "points-pt"],
  shelters: ["shelters-line", "shelters-pt"], evac: ["evac-line", "evac-pt"],
  dispatch: ["dispatch-base-line", "dispatch-line", "points-pt"], results: [],
};

function showTabLayers(tab) {
  const all = new Set(Object.values(tabLayers).flat());
  for (const id of all) map.setLayoutProperty(id, "visibility", tabLayers[tab].includes(id) ? "visible" : "none");
}

function pointsFromState() {   // origin / destination / incident markers
  const pts = [];
  if (state.tab === "route") {
    if (state.origin) pts.push({ role: "origin", c: state.origin });
    if (state.dest) pts.push({ role: "dest", c: state.dest });
  } else if (state.tab === "dispatch") {
    for (const c of state.incidents) pts.push({ role: "incident", c });
  }
  setGeo("points", {
    type: "FeatureCollection",
    features: pts.map((p) => ({ type: "Feature", properties: { role: p.role }, geometry: { type: "Point", coordinates: p.c } })),
  });
}

function fitScenario() {
  const [w, s, e, n] = state.scenario.bbox;
  map.fitBounds([[w, s], [e, n]], { padding: 0, animate: false });
}

// ---------------------------------------------------------------- 4. tabs
function selectTab(tab) {
  state.tab = tab;
  for (const b of document.querySelectorAll("[data-tab]")) b.classList.toggle("active", b.dataset.tab === tab);
  for (const p of document.querySelectorAll("[data-panel]")) p.hidden = p.dataset.panel !== tab;
  setStatus("");
  if (!map || !map.getLayer("overlay")) return;   // layers not added yet (start-up calls selectTab again)
  showTabLayers(tab);
  pointsFromState();
  if (tab === "forecast") loadOverlay();
  if (tab === "shelters") loadShelters();
  if (tab === "evac") loadEvac();
  if (tab === "results") loadMetrics();
}

// ---------------------------------------------------------------- 5a. scenario + forecast overlay
function fillSelect(select, values) {
  select.innerHTML = "";
  for (const v of values) select.add(new Option(v, v));
}

function chooseScenario(id) {
  state.scenario = state.scenarios.find((s) => s.id === id);
  const sc = state.scenario;
  fillSelect($("hazard-select"), sc.hazards);
  fillSelect($("source-select"), sc.overlay_sources || ["model"]);
  $("slot-slider").max = (sc.n_slots || 12) - 1;
  $("slot-slider").value = 0;
  state.origin = state.dest = state.route = null;
  state.incidents = [];
  const [w, s, e, n] = sc.bbox;
  map.getSource("overlay").setCoordinates([[w, n], [e, n], [e, s], [w, s]]);
  fitScenario();
  clearRoute();
  selectTab(state.tab);
}

function loadOverlay() {
  if (!state.scenario) return;
  const slot = Number($("slot-slider").value), minutes = slot * (state.scenario.slot_minutes || 15);
  $("slot-label").textContent = `+${minutes} min (slot ${slot})`;
  map.getSource("overlay").updateImage({
    url: api.overlay($("hazard-select").value, slot, $("source-select").value),
    coordinates: map.getSource("overlay").coordinates,
  });
}

// ---------------------------------------------------------------- 5b. safe route
function clearRoute() {
  setGeo("outrun"); setGeo("shortest");
  $("route-table").hidden = true;
  $("route-note").textContent = "";
  $("route-hint").textContent = "Click the map to set the origin.";
  state.route = null;
}

function lineKm(feature) {   // length of a LineString Feature in km (haversine)
  if (!feature) return null;
  const c = feature.geometry.coordinates, rad = Math.PI / 180;
  let km = 0;
  for (let i = 1; i < c.length; i++) {
    const dLat = (c[i][1] - c[i - 1][1]) * rad, dLon = (c[i][0] - c[i - 1][0]) * rad;
    const a = Math.sin(dLat / 2) ** 2 + Math.cos(c[i - 1][1] * rad) * Math.cos(c[i][1] * rad) * Math.sin(dLon / 2) ** 2;
    km += 6371.0088 * 2 * Math.asin(Math.sqrt(a));
  }
  return km;
}
const fmt = (x, d = 1) => (x === null || x === undefined ? "n/a" : Number(x).toFixed(d));

// Static export has no live router: use the cached preset whose origin is closest (APP-11 does the same on the server).
async function routeFromPresets(origin) {
  const presets = await getJSON(`${BASE}/data/scenarios/${state.scenario.id}/routes_preset.json`);
  const dist = (p) => Math.hypot(p.origin[0] - origin[0], p.origin[1] - origin[1]);
  const best = presets.filter((p) => p.dest === "nearest_shelter").sort((a, b) => dist(a) - dist(b))[0];
  if (!best) throw new Error("no cached route");
  return { ...best.result, source: "cached" };
}

async function requestRoute() {
  const nearest = $("dest-mode").value === "nearest_shelter";
  const body = {
    scenario: state.scenario.id, origin: state.origin, dest: nearest ? "nearest_shelter" : state.dest,
    profile: $("profile-select").value, lambda: null, t0: null, risk_source: "model",
  };
  setStatus("Routing...");
  try {
    const res = state.mode === "static" ? await routeFromPresets(state.origin) : await post(BASE + "/api/route", body);
    drawRoute(res);
    setStatus("");
  } catch (err) {
    setStatus("Route failed: " + err.message);
  }
}

function drawRoute(res) {
  setGeo("outrun", res.outrun);
  setGeo("shortest", res.shortest);
  state.route = res.outrun;   // read by the e2e test
  const pred = res.exposure && res.exposure.predicted, obs = res.exposure && res.exposure.observed;
  const sp = (res.shortest && res.shortest.properties) || {};   // shortest-route stats, if the backend sends them
  $("rt-time-o").textContent = fmt(res.eta_min);
  $("rt-time-s").textContent = fmt(sp.eta_min);
  $("rt-dist-o").textContent = fmt(lineKm(res.outrun), 2);
  $("rt-dist-s").textContent = fmt(lineKm(res.shortest), 2);
  $("rt-risk-o").textContent = fmt(pred, 2) + (obs != null ? ` (obs ${fmt(obs, 2)})` : "");
  $("rt-risk-s").textContent = fmt(sp.exposure, 2);
  $("route-table").hidden = false;
  const notes = [`source: ${res.source}`];
  if (res.shelter_in_place) notes.push("no safe drive: shelter in place");
  if (res.blocked_avoided) notes.push(`avoided ${res.blocked_avoided} blocked road segments`);
  if (res.expansions) notes.push(`A* expansions ${res.expansions.astar} vs Dijkstra ${res.expansions.dijkstra}`);
  $("route-note").textContent = notes.join("; ");
}

function onRouteClick(lngLat) {
  const point = [lngLat.lng, lngLat.lat], clickDest = $("dest-mode").value === "click";
  if (!clickDest || !state.origin || state.dest) {   // first click (or restart): set the origin
    state.origin = point; state.dest = null;
    clearRoute();
    $("route-hint").textContent = clickDest ? "Now click the destination." : "Click again to move the origin.";
    pointsFromState();
    if (!clickDest) requestRoute();
  } else {                                            // second click: set the destination and route
    state.dest = point;
    pointsFromState();
    requestRoute();
  }
}

// ---------------------------------------------------------------- 5c. shelters and evacuation
async function loadShelters() {
  try {
    const data = await getJSON(api.shelters($("shelter-method").value));
    setGeo("shelters", data);
    $("shelter-metrics").textContent = JSON.stringify(data.metrics || {}, null, 2);
    setStatus("");
  } catch (err) { setGeo("shelters"); $("shelter-metrics").textContent = ""; setStatus("Shelters: " + err.message); }
}

async function loadEvac() {
  try {
    const data = await getJSON(api.evac($("evac-method").value));
    setGeo("evac", data);
    $("evac-metrics").textContent = JSON.stringify(data.metrics || {}, null, 2);
    setStatus("");
  } catch (err) { setGeo("evac"); $("evac-metrics").textContent = ""; setStatus("Evacuation: " + err.message); }
}

// Live simulated annealing: server-sent events "progress" then "done" (or "error").
function runEvac() {
  if (state.mode === "static") { setStatus("Live annealing needs the running server."); return; }
  const source = new EventSource(api.evacRun());
  source.addEventListener("progress", (e) => {
    const p = JSON.parse(e.data);
    $("evac-progress").textContent = `iter ${p.iter}  objective ${fmt(p.objective, 3)}  T ${fmt(p.temperature, 3)}`;
  });
  source.addEventListener("done", (e) => {
    const data = JSON.parse(e.data);
    setGeo("evac", data);
    $("evac-metrics").textContent = JSON.stringify(data.metrics || {}, null, 2);
    $("evac-progress").textContent = `done (${data.source})`;
    source.close();
  });
  source.addEventListener("error", (e) => {   // server "error" event carries data; a dropped connection does not
    if (e.data) setStatus("Evacuation: " + JSON.parse(e.data).detail);
    source.close();
  });
}

// ---------------------------------------------------------------- 5d. dispatch and metrics
function onDispatchClick(lngLat) {
  if (state.incidents.length >= 3) state.incidents = [];
  state.incidents.push([lngLat.lng, lngLat.lat]);
  pointsFromState();
}

async function runDispatch() {
  if (state.mode === "static") { setStatus("Dispatch needs the running server."); return; }
  if (!state.incidents.length) { setStatus("Click the map to add an incident first."); return; }
  setStatus("Dispatching...");
  try {
    const res = await post(BASE + "/api/dispatch", { scenario: state.scenario.id, incidents: state.incidents, t0: null });
    setGeo("dispatch", res.routes);
    setGeo("dispatch-base", res.baseline);
    $("dispatch-out").textContent = JSON.stringify(res.assignment, null, 2);
    setStatus("");
  } catch (err) { setStatus("Dispatch: " + err.message); }
}

function clearDispatch() {
  state.incidents = [];
  setGeo("dispatch"); setGeo("dispatch-base");
  $("dispatch-out").textContent = "";
  pointsFromState();
}

async function loadMetrics() {
  try {
    const data = await getJSON(api.metrics());
    const names = Object.keys(data.metrics || {});
    $("metrics-out").textContent = names.length ? JSON.stringify(data.metrics, null, 2) : "No metrics reports found.";
  } catch (err) { setStatus("Metrics: " + err.message); }
}

async function searchCity(name) {
  if (state.mode === "static") { setStatus("City search needs the running server."); return; }
  try {
    const city = await getJSON(api.city(name));
    const [w, s, e, n] = city.bbox;
    map.fitBounds([[w, s], [e, n]], { padding: 20 });
    setStatus(`${city.name}: ${city.n_nodes} nodes, ${city.n_edges} edges${city.cached ? " (cached)" : ""}`);
  } catch (err) { setStatus("City: " + err.message); }
}

// ---------------------------------------------------------------- 6. start-up
async function loadConfig() {   // export-static writes config.json = {"mode": "static"}; the live server has none
  try { state.mode = (await getJSON(BASE + "/config.json")).mode || "live"; } catch (e) { state.mode = "live"; }
}

function wireEvents() {
  $("scenario-select").addEventListener("change", (e) => chooseScenario(e.target.value));
  for (const id of ["hazard-select", "source-select", "slot-slider"]) $(id).addEventListener("input", loadOverlay);
  $("shelter-method").addEventListener("change", loadShelters);
  $("evac-method").addEventListener("change", loadEvac);
  $("evac-run").addEventListener("click", runEvac);
  $("dispatch-go").addEventListener("click", runDispatch);
  $("dispatch-clear").addEventListener("click", clearDispatch);
  $("dest-mode").addEventListener("change", () => { state.dest = null; clearRoute(); pointsFromState(); });
  $("city-box").addEventListener("keydown", (e) => { if (e.key === "Enter" && e.target.value.trim()) searchCity(e.target.value.trim()); });
  map.on("click", (e) => {
    if (!state.scenario) return;
    if (state.tab === "route") onRouteClick(e.lngLat);
    if (state.tab === "dispatch") onDispatchClick(e.lngLat);
  });
}

async function start() {
  // Tab buttons work from the start (selectTab waits for the map layers to exist).
  for (const b of document.querySelectorAll("[data-tab]")) b.addEventListener("click", () => selectTab(b.dataset.tab));
  await loadConfig();
  map = await createMap();
  map.on("error", (e) => console.warn("map:", e.error && e.error.message));
  await new Promise((resolve) => map.on("load", resolve));
  addLayers();
  wireEvents();
  try {
    state.scenarios = await getJSON(api.scenarios());
  } catch (err) { setStatus("Could not load scenarios: " + err.message); return; }
  if (!state.scenarios.length) { setStatus("No scenarios found. Run `outrun scenario` first."); return; }
  fillSelect($("scenario-select"), state.scenarios.map((s) => s.id));
  chooseScenario(state.scenarios[0].id);
}

start();
