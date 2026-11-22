# Wildfire spread (`wildfire.unet`)

- **Task:** next-day fire-mask probability per cell.
- **Model:** U-Net, BCE + Dice with uncertain pixels ignored, calibrated (`models/wildfire`).
- **Data:** training on NDWS (Kaggle, CC BY 4.0); out-of-time test on WildfireSpreadTS 2021 (Zenodo, CC BY 4.0).
- **Baselines:** persistence, downwind persistence, pixel logistic regression; NDWS published PR-AUC 0.284.

| Split | Model | PR-AUC |
|---|---|---|
| NDWS test | U-Net | 0.357 [0.340, 0.373] |
| NDWS test | pixel logistic | 0.153 [0.139, 0.168] |
| NDWS test | NDWS published | 0.284 |
| WSTS 2021 | U-Net | 0.372 [0.323, 0.410] |
| WSTS 2021 | pixel logistic | 0.456 [0.416, 0.488] |
| WSTS 2021 | persistence | 0.433 [0.392, 0.464] |

**Honest result:** the model wins on NDWS but **loses on the out-of-time WSTS 2021 test** to both persistence and
the pixel logistic baseline (it is better only on newly burned area: 0.060 vs 0.003). The NDWS bootstrap
resamples per sample and is optimistic; treat its intervals as too narrow.

**Demo caveat:** the Oklahoma March 2025 scenario is a labelled replay.
