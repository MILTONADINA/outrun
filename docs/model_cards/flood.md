# Flash flood (`flood.unet`)

- **Task:** probability of flash-flood occurrence per 1 km cell for the next window.
- **Model:** U-Net on MRMS QPE, FLASH (FFG ratio, ARI) and terrain statics, focal loss with the uncertain ring
  masked, calibrated on validation (`models/flood`); an uncalibrated twin (`flood.unet_raw`) and a no-FLASH ablation
  (`models/flood_noflash`).
- **Data:** MRMS and FLASH (NOAA, public domain), ETOPO 2022, NCEI Storm Events labels.
- **Baselines:** FFG ratio >= 1, QPE ARI >= 10 yr, ARI rule, logistic regression.

| Model (test) | AUPRC | AUC vs hard negative | Brier |
|---|---|---|---|
| U-Net (calibrated) | 0.224 [0.191, 0.255] | 0.841 [0.821, 0.856] | 0.038 [0.033, 0.044] |
| U-Net raw | 0.226 [0.193, 0.258] | 0.841 [0.821, 0.856] | 0.048 |
| logistic regression | 0.144 [0.119, 0.167] | 0.680 [0.641, 0.713] | 0.040 |
| FFG ratio >= 1 | 0.065 [0.054, 0.075] | 0.532 | 0.046 |

**Ablation:** FLASH adds nothing. Without it AUPRC is 0.207 [0.175, 0.236] (calibrated) and 0.210 [0.177, 0.239]
(raw), within the interval of the full model.

**Limits:** event POD at p50 is low (0.096); pixel CSI 0.035. Good at ranking, weak as a hard alarm.
