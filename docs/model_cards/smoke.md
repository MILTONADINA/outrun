# Smoke PM2.5 (`smoke.cnn_mlp`)

- **Task:** PM2.5 (ug/m3) forecast with quantiles at EPA monitor stations, spread to the grid.
- **Model:** CNN + MLP quantile model trained with pinball loss (`models/smoke`); no-HMS twin `models/smoke_nohms`.
- **Data:** HRRR-Smoke (NOAA), NOAA HMS smoke polygons, EPA AQS hourly PM2.5.
- **Baselines:** persistence, calibrated HRRR, linear regression.

| Split | Model | RMSE (smoke episodes) | CSI at 35.4 |
|---|---|---|---|
| test | CNN+MLP | 10.209 [8.775, 11.951] | 0.114 [0.095, 0.132] |
| test | calibrated HRRR | 12.198 [10.679, 13.913] | 0.053 |
| test | persistence | 12.420 [10.818, 14.547] | 0.182 [0.163, 0.200] |
| spatial hold-out | CNN+MLP | 12.114 [9.807, 14.544] | 0.113 |
| spatial hold-out | calibrated HRRR | 13.670 [11.536, 16.009] | 0.035 |

**Honest result:** the model wins on RMSE and MAE, but **persistence has the better CSI** at the 35.4 threshold
(0.182 vs 0.114, 0.188 vs 0.113 on the spatial hold-out); the model has fewer false alarms (FAR 0.363 vs 0.692) but
lower POD. The nominal 90% quantile interval covers 0.911 overall and 0.901 on smoke episodes.

**Ablation:** HMS smoke polygons add nothing (RMSE 10.283 [8.624, 12.356] without them).

**Demo caveat:** for the OKC April 2024 case the model forecasts at most 20.7 ug/m3, below threshold, so the smoke
risk layers are empty and all five planners score 0.
