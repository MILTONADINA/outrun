# Data sources and licences

Code in this repository is MIT-licensed (`LICENSE`). **No data is committed.** Every dataset is downloaded by
`outrun fetch` from the source listed in `data/manifest.yaml` and keeps its own licence below.
Rows tagged [CANDIDATE] could not be fully confirmed from a licence page and rest on the registry or mirror terms
named in the row; re-check before any redistribution. Trained model weights are not committed (`models/` is
ignored); they are derived from the datasets above and carry the attribution duties below.

**Redistributed in this repo:** `scenarios/` holds derived bundles (risk rasters, overlays, route results) and
shared road graphs built from OpenStreetMap (ODbL 1.0: (c) OpenStreetMap contributors; the graphs are a derived
database and stay under ODbL) plus HIFLD facility lists (public domain, via the source.coop mirror). The static
Pages export copies only overlays and result JSON from `scenarios/`, plus `reports/` metrics.

**Not committed:** raw SEVIR, TorNet, NDWS, WSTS, MRMS, HRRR, AQS and every other download. NDWS needs the user's own
Kaggle API token in `~/.kaggle/kaggle.json`; the token is never stored in this repo.

**Attribution:** SEVIR (Veillette et al., NeurIPS 2020), TorNet (Veillette et al., MIT Lincoln Laboratory),
Next Day Wildfire Spread (Huot et al., 2022), WildfireSpreadTS (Gerard et al., 2023). Licences per the table.

| Dataset | Used for | Source | Licence |
|---|---|---|---|
| SEVIR (VIL + GOES-16 IR107, CATALOG) | Storm nowcast (radar + satellite) | AWS Open Data `s3://sevir` | [CANDIDATE] CC BY 4.0 (per AWS registry entry) |
| TorNet | Tornado detection (NEXRAD Level-II chips) | Zenodo (MIT Lincoln Laboratory) | CC BY 4.0 |
| Next Day Wildfire Spread (NDWS) | Wildfire training (MODIS fire masks, VIIRS NDVI, GRIDMET) | Kaggle `fantineh/next-day-wildfire-spread` | CC BY 4.0 |
| WildfireSpreadTS (WSTS) | Wildfire out-of-time test + demo (VIIRS) | Zenodo 8006177 | CC BY 4.0 |
| MRMS QPE + FLASH | Flash-flood inputs / operational baseline | AWS Open Data `s3://noaa-mrms-pds` | Public domain (NOAA) |
| HRRR / HRRR-Smoke | Ice and smoke inputs, tornado motion fallback | AWS Open Data `s3://noaa-hrrr-bdp-pds` | Public domain (NOAA) |
| NCEI Storm Events | Flood and ice labels, case-study verification | NOAA NCEI | Public domain (NOAA) |
| SPC tornado tracks | Tornado observed truth | NOAA Storm Prediction Center | Public domain (NOAA) |
| NWS forecast zones | Ice label zones | NOAA NWS | Public domain (NOAA) |
| NOAA HMS fire points (Mar 2025 replay) | Wildfire replay: satellite fire detections | NOAA OSPO | Public domain (NOAA) |
| NOAA HMS smoke polygons | Smoke-model satellite input | NOAA OSPO | Public domain (NOAA) |
| EPA AQS hourly PM2.5 (88101) | Smoke labels and observations | EPA AirData | Public domain (US EPA) |
| ETOPO 2022 30" | Flood training terrain statics | NOAA NCEI | Public domain (NOAA) |
| USGS 3DEP 10 m | OKC low-water-crossing depressions | USGS National Map | Public domain (USGS) |
| OpenStreetMap (Geofabrik extract, OSMnx, Overpass) | Road graphs, bridges, fords, amenities | Geofabrik / OSM | ODbL 1.0 (attribution: (c) OpenStreetMap contributors) |
| Protomaps basemap (PMTiles) | Offline map tiles | Protomaps builds of OSM | ODbL 1.0 (OSM data) |
| HIFLD (NSS shelters, hospitals, fire/EMS, schools) | Shelter and station locations | source.coop `seerai/hifld` (mirror) | [CANDIDATE] public domain (US government), per mirror terms |
| US Census 2020 block groups | Evacuation demand | US Census Bureau | Public domain (US government) |
| US Census 2020 PL 94-171 redistricting file, Oklahoma (`census_2020_pl94171_ok`) | Block-group population for evacuation demand | US Census Bureau | Public domain (US government) |
