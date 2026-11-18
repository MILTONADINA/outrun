# Data sources and licences

Code in this repository is MIT-licensed (`LICENSE`). **No data is committed.** Every dataset is downloaded by
`outrun fetch` from the source listed in `data/manifest.yaml` and keeps its own licence below.
Rows tagged [CANDIDATE] are verified in T1/T3 (licence page checked, recorded in NOTES.md).

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
| NOAA HMS smoke polygons | Smoke-model satellite input | NOAA OSPO | Public domain (NOAA) |
| EPA AQS hourly PM2.5 (88101) | Smoke labels and observations | EPA AirData | Public domain (US EPA) |
| ETOPO 2022 30" | Flood training terrain statics | NOAA NCEI | Public domain (NOAA) |
| USGS 3DEP 10 m | OKC low-water-crossing depressions | USGS National Map | Public domain (USGS) |
| OpenStreetMap (Geofabrik extract, OSMnx, Overpass) | Road graphs, bridges, fords, amenities | Geofabrik / OSM | ODbL 1.0 (attribution: (c) OpenStreetMap contributors) |
| Protomaps basemap (PMTiles) | Offline map tiles | Protomaps builds of OSM | ODbL 1.0 (OSM data) |
| HIFLD (NSS shelters, hospitals, fire/EMS, schools) | Shelter and station locations | source.coop `seerai/hifld` (mirror) | [CANDIDATE] public domain (US government), per mirror terms |
| US Census 2020 block groups | Evacuation demand | US Census Bureau | Public domain (US government) |
