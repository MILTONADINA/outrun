# Model cards

One card per hazard model. All numbers are copied from `reports/metrics.md` (one-shot test evaluation, T24,
git sha 72d9715-dirty). Intervals are 95% bootstrap CIs. The tests ran once; nothing was tuned on them.

| Hazard | Card | Headline (test) | Beats best baseline? |
|---|---|---|---|
| Storm | [storm.md](storm.md) | CSI74/133 0.533 vs pySTEPS 0.467 | Yes |
| Tornado | [tornado.md](tornado.md) | TOR-vs-warned AUC 0.761 vs raw CNN 0.762 | No (ties its own uncalibrated twin) |
| Flood | [flood.md](flood.md) | AUPRC 0.224 vs logistic 0.144 | Yes |
| Wildfire | [wildfire.md](wildfire.md) | NDWS PR-AUC 0.357 vs 0.153; WSTS 2021 0.372 vs 0.456 | Yes on NDWS, no on WSTS |
| Ice | [ice.md](ice.md) | Zone AUPRC 0.801 vs logistic 0.804 | No |
| Smoke | [smoke.md](smoke.md) | Episode RMSE 10.21 vs 12.20 | Yes on RMSE; persistence has the better CSI |

Intended use for all six: a course demonstration of forecast-then-plan. None is an operational warning product.
All risk layers are probabilities on one shared 1 km grid (EPSG:5070) and are consumed by the planners as
`RiskLayer`s. Data sources and licences: `DATA_LICENSES.md`.
