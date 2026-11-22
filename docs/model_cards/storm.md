# Storm nowcast (`storm.unet_vil_ir`)

- **Task:** radar (VIL) nowcast as exceedance probabilities at VIL levels 74, 133, 181 (plus 16, 160, 219 for
  reporting), per 1 km cell.
- **Model:** U-Net on SEVIR VIL, optionally with GOES-16 IR107 (`models/storm`); the VIL-only twin is `models/storm_vil_only`.
- **Data:** SEVIR (AWS Open Data, CC BY 4.0, licence flagged as a candidate; see `DATA_LICENSES.md`).
- **Baselines:** persistence, pySTEPS extrapolation.

| Split | Model | CSI mean 74/133 |
|---|---|---|
| test | U-Net | 0.533 [0.520, 0.544] |
| test | pySTEPS | 0.467 [0.455, 0.477] |
| test | persistence | 0.352 [0.339, 0.364] |
| case hold-out | U-Net | 0.563 [0.536, 0.587] |
| case hold-out | pySTEPS | 0.479 [0.466, 0.489] |

**Ablation:** IR adds nothing. The VIL-only model scores 0.536 [0.522, 0.548] on test and 0.570 on the case
hold-out, equal to or above the VIL+IR model.

**Limits:** SEVIR covers 2018-2019 storm events only; skill at VIL 219 is low (CSI 0.105). Not tested outside the
SEVIR footprint.
