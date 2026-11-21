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
  hazard: null, metrics: null, evacTrace: [], waveTimer: null, playTimer: null, swipe: false,
};

function setStatus(msg) { $("status").textContent = msg || ""; }

// Build DOM nodes without innerHTML (all text is set as text).
function el(tag, props, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) { if (k === "class") n.className = v; else n[k] = v; }
  for (const k of kids) n.append(k);
  return n;
}

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
  trace: () => `${BASE}/data/scenarios/${state.scenario.id}/evac_sa_trace.json`,
  dispatchPreset: () => `${BASE}/data/scenarios/${state.scenario.id}/dispatch_preset.json`,
  city: (name) => `${BASE}/api/city?name=${encodeURIComponent(name)}`,
};

function post(url, body) {
  return getJSON(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

// ---------------------------------------------------------------- 2. map setup
// Basemap: local PMTiles file served by the backend. If the file is absent we fall back to a plain
// background so the map still renders offline.
const TILES_URL = "static/tiles/okc.pmtiles";
let basemapOk = null;

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
  if (basemapOk === null) {
    try { basemapOk = (await fetch(TILES_URL, { method: "HEAD" })).ok; } catch (e) { basemapOk = false; }
    if (basemapOk) maplibregl.addProtocol("pmtiles", new pmtiles.Protocol().tile);
  }
  return basemapOk;
}

async function makeMap(container) {
  const style = (await hasBasemap()) ? pmtilesStyle() : plainStyle();
  return new maplibregl.Map({ container, style, center: [-97.5, 35.5], zoom: 9, attributionControl: false });
}

async function createMap() {
  const m = await makeMap("map");
  window.outrunMap = m;
  return m;
}

// ---------------------------------------------------------------- 3. map layers
const EMPTY = { type: "FeatureCollection", features: [] };
let map, map2 = null;

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
    { id: "line", type: "line", filter: isLine, paint: {
      "line-color": "#2f9e44", "line-opacity": 0.7,
      "line-width": ["interpolate", ["linear"], ["coalesce", ["get", "people"], 0], 0, 1, 2000, 5] } },
    { id: "pt", type: "circle", filter: isPoint, paint: {
      "circle-color": ["case", [">", ["coalesce", ["get", "overflow"], 0], 0], "#e03131", "#2f9e44"],
      "circle-radius": ["interpolate", ["linear"], ["coalesce", ["get", "capacity"], 0], 0, 4, 500, 8, 5000, 14],
      "circle-stroke-width": 1, "circle-stroke-color": "#fff" } },
  ]);
  // Evacuation packets: one circle per vehicle packet, coloured by departure wave, sized by vehicles.
  addGeo("evac", [
    { id: "line", type: "line", filter: isLine, paint: { "line-color": "#e8590c", "line-width": 2 } },
    { id: "pt", type: "circle", filter: isPoint, paint: {
      "circle-color": ["match", ["coalesce", ["get", "wave"], 0], 0, "#fab005", 1, "#f76707", 2, "#e03131", "#862e9c"],
      "circle-radius": ["interpolate", ["linear"], ["coalesce", ["get", "vehicles"], 1], 1, 3, 100, 8],
      "circle-opacity": 0.75, "circle-stroke-width": 1, "circle-stroke-color": "#fff" } },
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
function stopTimers() {
  for (const k of ["waveTimer", "playTimer"]) { if (state[k]) clearInterval(state[k]); state[k] = null; }
  $("play-btn").textContent = "Play";
  $("evac-play").textContent = "Animate waves";
  if (map && map.getLayer("evac-pt")) map.setFilter("evac-pt", isPoint);
}

function selectTab(tab) {
  if (tab !== state.tab) stopTimers();
  state.tab = tab;
  for (const b of document.querySelectorAll("[data-tab]")) b.classList.toggle("active", b.dataset.tab === tab);
  for (const p of document.querySelectorAll("[data-panel]")) p.hidden = p.dataset.panel !== tab;
  setStatus("");
  updateSwipe();
  if (!map || !map.getLayer("overlay")) return;   // layers not added yet (start-up calls selectTab again)
  showTabLayers(tab);
  pointsFromState();
  if (tab === "forecast") loadOverlay();
  if (tab === "shelters") loadShelters();
  if (tab === "evac") loadEvac();
  if (tab === "results") loadResults();
}

// ---------------------------------------------------------------- 5a. scenario card + forecast overlay + swipe
function fillSelect(select, values) {
  select.innerHTML = "";
  for (const v of values) select.add(new Option(v, v));
}

function renderScenarioCard() {
  const sc = state.scenario, card = $("scenario-card");
  card.replaceChildren(el("h3", { textContent: sc.title }));
  const badges = el("div", {});
  badges.append(el("span", { class: "badge", textContent: sc.t0.slice(0, 16).replace("T", " ") + " UTC" }));
  if (sc.held_out) badges.append(el("span", { class: "badge held", textContent: "held out" }));
  if (sc.synthetic) badges.append(el("span", { class: "badge synth", textContent: "synthetic" }));
  card.append(badges, el("div", { textContent: sc.city }));
  const det = el("details", {}, el("summary", { textContent: "Data sources" }));
  det.append(el("ul", {}, ...(sc.sources || []).map((t) => el("li", { textContent: t }))));
  card.append(det);
}

function renderHazardChips() {
  const box = $("hazard-chips");
  box.replaceChildren();
  for (const h of state.scenario.hazards) {
    const b = el("button", { type: "button", textContent: h, class: h === state.hazard ? "on" : "" });
    b.dataset.hazard = h;
    b.addEventListener("click", () => { state.hazard = h; renderHazardChips(); loadOverlay(); });
    box.append(b);
  }
}

function chooseScenario(id) {
  stopTimers();
  state.scenario = state.scenarios.find((s) => s.id === id);
  const sc = state.scenario;
  state.hazard = sc.hazards[0];
  renderScenarioCard();
  renderHazardChips();
  fillSelect($("source-select"), sc.overlay_sources || ["model"]);
  if ((sc.overlay_sources || []).includes("model")) $("source-select").value = "model";
  $("slot-slider").max = (sc.n_slots || 12) - 1;
  $("slot-slider").value = 0;
  state.origin = state.dest = state.route = null;
  state.incidents = [];
  state.coords = (() => { const [w, s, e, n] = sc.bbox; return [[w, n], [e, n], [e, s], [w, s]]; })();
  map.getSource("overlay").setCoordinates(state.coords);
  if (map2) map2.getSource("overlay").setCoordinates(state.coords);
  fitScenario();
  clearRoute(); clearDispatch();
  $("swipe-toggle").disabled = !(sc.overlay_sources || []).some((x) => x !== "model");
  if ($("swipe-toggle").disabled && state.swipe) { $("swipe-toggle").checked = false; state.swipe = false; }
  selectTab(state.tab);
}

function overlayUrl(slot, source) { return api.overlay(state.hazard, slot, source); }

function loadOverlay() {
  if (!state.scenario) return;
  const slot = Number($("slot-slider").value), minutes = slot * (state.scenario.slot_minutes || 15);
  $("slot-label").textContent = `+${minutes} min (slot ${slot})`;
  const left = state.swipe ? "model" : $("source-select").value;
  map.getSource("overlay").updateImage({ url: overlayUrl(slot, left), coordinates: state.coords });
  if (state.swipe && map2) {
    const other = swipeRightSource();
    map2.getSource("overlay").updateImage({ url: overlayUrl(slot, other), coordinates: state.coords });
    $("swipe-note").textContent = `Left: model forecast. Right: ${other}.`;
  } else {
    $("swipe-note").textContent = "";
  }
}

function swipeRightSource() {
  const src = state.scenario.overlay_sources || [];
  return src.includes("observed") ? "observed" : (src.find((x) => x !== "model") || "model");
}

function playToggle() {
  if (state.playTimer) { stopTimers(); return; }
  $("play-btn").textContent = "Pause";
  const slider = $("slot-slider");
  state.playTimer = setInterval(() => {
    slider.value = (Number(slider.value) + 1) % (Number(slider.max) + 1);
    loadOverlay();
  }, 700);
}

// Swipe: a second map, clipped to the right of the divider, shows the other source; both maps move together.
async function ensureMap2() {
  if (map2) return;
  map2 = await makeMap("map2");
  await new Promise((resolve) => map2.on("load", resolve));
  map2.addSource("overlay", { type: "image", url: transparentPixel(), coordinates: state.coords });
  map2.addLayer({ id: "overlay", type: "raster", source: "overlay", paint: { "raster-opacity": 0.7, "raster-fade-duration": 0 } });
  const sync = () => map2.jumpTo({ center: map.getCenter(), zoom: map.getZoom(), bearing: map.getBearing(), pitch: map.getPitch() });
  map.on("move", sync);
  sync();
}

function applySwipePosition() {
  const pct = Number($("swipe-slider").value);
  $("map2").style.clipPath = `inset(0 0 0 ${pct}%)`;
  $("swipe-bar").style.left = `calc(${pct}% - 1px)`;
}

function updateSwipe() {   // the swipe widgets only exist on the forecast tab
  const on = state.swipe && state.tab === "forecast";
  for (const id of ["map2", "swipe-bar", "swipe-left", "swipe-right"]) $(id).hidden = !on;
  $("swipe-row").hidden = !state.swipe;
  if (on) { $("swipe-right").textContent = state.scenario ? swipeRightSource() : ""; applySwipePosition(); if (map2) map2.resize(); }
}

async function toggleSwipe(on) {
  state.swipe = on;
  if (on) await ensureMap2();
  updateSwipe();
  loadOverlay();
}

// ---------------------------------------------------------------- 5b. safe route
function clearRoute() {
  setGeo("outrun"); setGeo("shortest");
  $("route-table").hidden = true;
  $("sip-card").hidden = true;
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
  const profile = $("profile-select").value;
  const dist = (p) => Math.hypot(p.origin[0] - origin[0], p.origin[1] - origin[1]);
  const pool = presets.filter((p) => p.dest === "nearest_shelter");
  const best = (pool.filter((p) => p.profile === profile).length ? pool.filter((p) => p.profile === profile) : pool)
    .sort((a, b) => dist(a) - dist(b))[0];
  if (!best) throw new Error("no cached route");
  return { ...best.result, source: "cached" };
}

function currentLambda() { return $("lambda-default").checked ? null : Number($("lambda-slider").value); }

async function requestRoute() {
  if (!state.origin) return;
  const nearest = $("dest-mode").value === "nearest_shelter";
  if (!nearest && !state.dest) return;
  const body = {
    scenario: state.scenario.id, origin: state.origin, dest: nearest ? "nearest_shelter" : state.dest,
    profile: $("profile-select").value, lambda: currentLambda(), t0: null, risk_source: $("risk-source").value,
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
  $("sip-card").hidden = !res.shelter_in_place;
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
  if (res.cached_origin) notes.push("cached preset origin");
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
function kpi(label, value, bad) {
  return el("div", { class: "kpi" + (bad ? " bad" : "") }, el("b", { textContent: value }), label);
}

async function loadShelters() {
  try {
    const data = await getJSON(api.shelters($("shelter-method").value));
    setGeo("shelters", data);
    const m = data.metrics || {};
    $("shelter-strip").replaceChildren(
      kpi("served", fmt(m.served_persons, 0)), kpi("unserved", fmt(m.unserved_persons, 0), m.unserved_persons > 0),
      kpi("person-min", fmt(m.total_person_minutes, 0)), kpi("max min", fmt(m.max_minutes)),
      kpi("overflow %", fmt(m.overflow_pct), m.overflow_pct > 0));
    const sites = data.features.filter((f) => f.geometry.type === "Point").map((f) => f.properties)
      .sort((a, b) => b.capacity - a.capacity).slice(0, 12);
    $("shelter-bars").replaceChildren(...sites.map((p) => {
      const pct = Math.min(100, (100 * p.load) / Math.max(1, p.capacity));
      const fill = el("div", { class: "fill" + (p.overflow > 0 ? " over" : "") });
      fill.style.width = pct + "%";
      return el("div", { class: "bar" },
        `${p.id} (${p.kind}${p.assumed ? ", assumed" : ""}): ${p.load}/${p.capacity}` + (p.overflow > 0 ? `, overflow ${p.overflow}` : ""),
        el("div", { class: "track" }, fill));
    }));
    $("shelter-metrics").textContent = JSON.stringify(m, null, 2);
    setStatus("");
  } catch (err) {
    setGeo("shelters"); $("shelter-metrics").textContent = ""; $("shelter-strip").replaceChildren(); $("shelter-bars").replaceChildren();
    setStatus("Shelters: " + err.message);
  }
}

// SA objective curve on a plain canvas.
function drawCurve(points) {
  const cv = $("sa-curve"), g = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 22;
  g.clearRect(0, 0, W, H);
  g.fillStyle = "#5b6773"; g.font = "10px sans-serif";
  if (!points.length) { g.fillText("objective J vs iteration", pad, H / 2); return; }
  const xs = points.map((p) => p.iter), ys = points.map((p) => p.objective);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1), y0 = Math.min(...ys), y1 = Math.max(...ys, y0 + 1e-9);
  const X = (x) => pad + ((x - x0) / (x1 - x0)) * (W - pad - 6), Y = (y) => H - pad - ((y - y0) / (y1 - y0)) * (H - pad - 8);
  g.strokeStyle = "#adb5bd"; g.strokeRect(pad, 8, W - pad - 6, H - pad - 8);
  g.strokeStyle = "#e8590c"; g.lineWidth = 2; g.beginPath();
  points.forEach((p, i) => (i ? g.lineTo(X(p.iter), Y(p.objective)) : g.moveTo(X(p.iter), Y(p.objective))));
  g.stroke();
  g.fillText(`J ${fmt(y1, 0)}`, 2, 12); g.fillText(fmt(y0, 0), 2, H - pad); g.fillText(`iter ${x1}`, W - 60, H - 6);
}

function showEvac(data) {
  setGeo("evac", data);
  const m = data.metrics || {};
  $("evac-strip").replaceChildren(
    kpi("clearance min", fmt(m.clearance_min)), kpi("vehicle-h", fmt(m.vehicle_hours, 0)),
    kpi("exposure veh-h", fmt(m.exposure_veh_h, 0)), kpi("max v/c", fmt(m.max_vc, 2), m.max_vc > 1),
    kpi("J", fmt(m.J, 0)), kpi("gap to bound %", fmt(m.gap_to_lb_pct, 0)), kpi("unserved veh", fmt(m.unserved_veh, 0), m.unserved_veh > 0));
  $("evac-metrics").textContent = JSON.stringify(m, null, 2);
}

async function loadEvac() {
  stopTimers();
  try {
    showEvac(await getJSON(api.evac($("evac-method").value)));
    setStatus("");
  } catch (err) {
    setGeo("evac"); $("evac-metrics").textContent = ""; $("evac-strip").replaceChildren();
    setStatus("Evacuation: " + err.message);
  }
}

// Wave animation (DESIGN fallback to deck.gl TripsLayer): packets appear wave by wave, filtered in plain MapLibre.
function waveToggle() {
  if (state.waveTimer) { stopTimers(); return; }
  $("evac-play").textContent = "Stop";
  let wave = -1;
  state.waveTimer = setInterval(() => {
    wave = (wave + 1) % 5;
    map.setFilter("evac-pt", wave === 4 ? isPoint : ["all", isPoint, ["<=", ["coalesce", ["get", "wave"], 0], wave]]);
    $("evac-progress").textContent = wave === 4 ? "all waves" : `waves up to ${wave}`;
  }, 900);
}

function onProgress(p, points) {
  points.push(p);
  drawCurve(points);
  $("evac-progress").textContent = `iter ${p.iter}  objective ${fmt(p.objective, 1)}  T ${fmt(p.temperature, 2)}`;
}

// Live simulated annealing: server-sent events "progress" then "done" (or "error"). The static export replays the trace file.
async function runEvac() {
  stopTimers();
  const points = [];
  drawCurve(points);
  if (state.mode === "static") {
    try {
      const trace = await getJSON(api.trace());
      let i = 0;
      const t = setInterval(() => {
        if (i < trace.length) { onProgress(trace[i++], points); return; }
        clearInterval(t);
        $("evac-progress").textContent = "done (cached trace)";
        loadEvac();
      }, 60);
    } catch (err) { setStatus("Evacuation: " + err.message); }
    return;
  }
  const source = new EventSource(api.evacRun());
  source.addEventListener("progress", (e) => onProgress(JSON.parse(e.data), points));
  source.addEventListener("done", (e) => {
    const data = JSON.parse(e.data);
    showEvac(data);
    $("evac-progress").textContent = `done (${data.source})`;
    source.close();
  });
  source.addEventListener("error", (e) => {   // server "error" event carries data; a dropped connection does not
    if (e.data) setStatus("Evacuation: " + JSON.parse(e.data).detail);
    source.close();
  });
}

// ---------------------------------------------------------------- 5d. dispatch
function onDispatchClick(lngLat) {
  if (state.incidents.length >= 3) state.incidents = [];
  state.incidents.push([lngLat.lng, lngLat.lat]);
  pointsFromState();
}

function renderDispatch(res) {
  setGeo("dispatch", res.routes);
  setGeo("dispatch-base", res.baseline);
  const rows = res.comparison || [];
  const head = ["Incident", "Unit", "ETA min", "Baseline ETA", "Detour", "Hit block"];
  const table = el("table", {}, el("thead", {}, el("tr", {}, ...head.map((h) => el("th", { textContent: h })))));
  const body = el("tbody", {});
  for (const r of rows) {
    body.append(el("tr", {}, ...[r.incident + 1, r.unit, fmt(r.eta_min), fmt(r.baseline_eta_min),
      fmt(r.baseline_detour_min), r.baseline_hit_block ? "yes" : "no"].map((v) => el("td", { textContent: String(v) }))));
  }
  table.append(body);
  $("dispatch-table").replaceChildren(rows.length ? table : "");
  const m = res.metrics || {};
  $("dispatch-out").textContent = (res.source ? `source: ${res.source}\n` : "") +
    (m.mean_eta_min != null ? `mean ETA ${fmt(m.mean_eta_min)} min, baseline hit-block ${fmt(m.pct_hit_block, 0)} %, unreached ${fmt(m.pct_unreached, 0)} %\n` : "") +
    JSON.stringify(res.assignment, null, 2);
}

// Static export: nearest cached dispatch preset by its first incident.
async function dispatchFromPresets() {
  const presets = await getJSON(api.dispatchPreset());
  const a = state.incidents[0], d = (p) => Math.hypot(p.incidents[0][0] - a[0], p.incidents[0][1] - a[1]);
  const best = presets.sort((x, y) => d(x) - d(y))[0];
  if (!best) throw new Error("no cached dispatch");
  return { ...best.result, source: "cached" };
}

async function runDispatch() {
  if (!state.incidents.length) { setStatus("Click the map to add an incident first."); return; }
  setStatus("Dispatching...");
  try {
    const res = state.mode === "static"
      ? await dispatchFromPresets()
      : await post(BASE + "/api/dispatch", { scenario: state.scenario.id, incidents: state.incidents, t0: null });
    renderDispatch(res);
    setStatus("");
  } catch (err) { setStatus("Dispatch: " + err.message); }
}

function clearDispatch() {
  state.incidents = [];
  setGeo("dispatch"); setGeo("dispatch-base");
  $("dispatch-out").textContent = "";
  $("dispatch-table").replaceChildren();
  pointsFromState();
}

// ---------------------------------------------------------------- 5e. metrics, results, model card
function flatten(obj, prefix = "", out = []) {   // leaf numbers and strings of a JSON value as [path, value]
  if (obj !== null && typeof obj === "object") {
    for (const [k, v] of Object.entries(obj)) flatten(v, prefix ? prefix + "." + k : k, out);
  } else if (typeof obj === "number" || typeof obj === "string" || typeof obj === "boolean") out.push([prefix, obj]);
  return out;
}

function kvTable(rows) {
  const t = el("table", {});
  for (const [k, v] of rows) t.append(el("tr", {}, el("td", { textContent: k }), el("td", { textContent: typeof v === "number" ? fmt(v, 3) : String(v) })));
  return el("div", { class: "tbl-scroll" }, t);
}

async function fetchMetrics() {
  if (!state.metrics) state.metrics = (await getJSON(api.metrics())).metrics || {};
  return state.metrics;
}

// Scenario results table: one column per method, one row per metric (shelters and evacuation).
async function methodTable(kind, methods) {
  const got = await Promise.allSettled(methods.map((m) => getJSON(kind === "shelters" ? api.shelters(m) : api.evac(m))));
  const cols = methods.filter((_, i) => got[i].status === "fulfilled");
  if (!cols.length) return null;
  const metrics = cols.map((m) => got[methods.indexOf(m)].value.metrics || {});
  const keys = [...new Set(metrics.flatMap((m) => Object.keys(m)))];
  const t = el("table", {}, el("thead", {}, el("tr", {}, el("th", {}), ...cols.map((c) => el("th", { textContent: c })))));
  const tb = el("tbody", {});
  for (const k of keys) tb.append(el("tr", {}, el("td", { textContent: k }), ...metrics.map((m) => el("td", { textContent: fmt(m[k], 2) }))));
  t.append(tb);
  return el("div", {}, el("h3", { textContent: kind === "shelters" ? "Shelter methods" : "Evacuation methods" }), el("div", { class: "tbl-scroll" }, t));
}

async function loadResults() {
  const box = $("metrics-tables");
  box.replaceChildren();
  try {
    const metrics = await fetchMetrics(), names = Object.keys(metrics);
    $("metrics-out").textContent = names.length ? "" : "No model metrics reports found in reports/metrics yet (written by the evaluation step).";
    for (const n of names) box.append(el("h3", { textContent: n }), kvTable(flatten(metrics[n]).slice(0, 60)));
  } catch (err) { setStatus("Metrics: " + err.message); }
  if (!state.scenario) return;
  for (const t of await Promise.all([methodTable("shelters", ["mcf", "nearest", "greedy"]), methodTable("evac", ["sa", "msa", "shortest", "safest"])])) {
    if (t) box.append(t);
  }
}

// Reliability data: any object holding a predicted-probability array and an observed-frequency array.
function findReliability(o) {
  if (o === null || typeof o !== "object") return null;
  const pick = (names) => names.find((n) => Array.isArray(o[n]) && o[n].length > 1 && o[n].every((x) => typeof x === "number"));
  const px = pick(["mean_predicted", "predicted", "confidence", "bin_mean", "p"]);
  const py = pick(["observed_frequency", "observed", "frequency", "accuracy", "freq"]);
  if (px && py) return { x: o[px], y: o[py] };
  for (const v of Object.values(o)) { const r = findReliability(v); if (r) return r; }
  return null;
}

function reliabilityCanvas(r) {
  const cv = el("canvas", { width: 300, height: 220 }), g = cv.getContext("2d"), S = 190, o = 28;
  g.strokeStyle = "#adb5bd"; g.strokeRect(o, 10, S, S);
  g.beginPath(); g.moveTo(o, 10 + S); g.lineTo(o + S, 10); g.stroke();   // perfect calibration
  g.fillStyle = "#1971c2";
  r.x.forEach((x, i) => { g.beginPath(); g.arc(o + x * S, 10 + S - r.y[i] * S, 3, 0, 7); g.fill(); });
  g.fillStyle = "#5b6773"; g.font = "10px sans-serif"; g.fillText("predicted", o + 70, 215); g.fillText("observed", 0, 8);
  return cv;
}

async function openCard() {
  const hz = state.hazard, body = $("drawer-body");
  $("drawer-title").textContent = `Model card: ${hz}`;
  body.replaceChildren(el("p", { class: "hint", textContent: "Loading..." }));
  $("drawer").hidden = false;
  let metrics = {};
  try { metrics = await fetchMetrics(); } catch (err) { /* shown below as "no metrics" */ }
  const stems = Object.keys(metrics).filter((k) => k.toLowerCase().includes(hz));
  const out = [];
  out.push(el("h3", { textContent: "Layers used in this scenario" }),
    el("ul", {}, ...(state.scenario.sources || []).filter((t) => t.toLowerCase().includes(hz)).map((t) => el("li", { textContent: t }))));
  if (!stems.length) out.push(el("p", { class: "hint", textContent: `No metrics file for ${hz} in reports/metrics (metric vs baseline and reliability appear once the evaluation has been run).` }));
  for (const k of stems) {
    out.push(el("h3", { textContent: k + " (model vs baseline)" }), kvTable(flatten(metrics[k]).slice(0, 40)));
    const r = findReliability(metrics[k]);
    if (r) out.push(el("h3", { textContent: "Reliability" }), reliabilityCanvas(r));
  }
  body.replaceChildren(...out);
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
  for (const id of ["source-select", "slot-slider"]) $(id).addEventListener("input", loadOverlay);
  $("play-btn").addEventListener("click", playToggle);
  $("swipe-toggle").addEventListener("change", (e) => toggleSwipe(e.target.checked));
  $("swipe-slider").addEventListener("input", applySwipePosition);
  $("card-btn").addEventListener("click", openCard);
  $("drawer-close").addEventListener("click", () => { $("drawer").hidden = true; });
  $("shelter-method").addEventListener("change", loadShelters);
  $("evac-method").addEventListener("change", loadEvac);
  $("evac-run").addEventListener("click", runEvac);
  $("evac-play").addEventListener("click", waveToggle);
  $("dispatch-go").addEventListener("click", runDispatch);
  $("dispatch-clear").addEventListener("click", clearDispatch);
  $("dest-mode").addEventListener("change", () => { state.dest = null; clearRoute(); pointsFromState(); });
  for (const id of ["profile-select", "risk-source"]) $(id).addEventListener("change", requestRoute);
  $("lambda-slider").addEventListener("input", () => {
    $("lambda-default").checked = false;
    $("lambda-label").textContent = Number($("lambda-slider").value).toFixed(2);
  });
  $("lambda-slider").addEventListener("change", requestRoute);
  $("lambda-default").addEventListener("change", () => {
    $("lambda-label").textContent = $("lambda-default").checked ? "default" : Number($("lambda-slider").value).toFixed(2);
    requestRoute();
  });
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
