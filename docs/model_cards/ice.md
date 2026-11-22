# Icy roads (`ice.milcnn`)

- **Task:** probability of an ice event per NWS forecast zone and 6-hour window.
- **Model:** multiple-instance CNN: U-Net cell logits, LogSumExp zone pooling, zone-level isotonic calibration
  (`models/ice`).
- **Data:** HRRR (NOAA, public domain), NCEI Storm Events labels, NWS zones.
- **Baselines:** physics rule, logistic regression, climatology.

| Model (test) | Zone AUPRC | Brier | ROC AUC |
|---|---|---|---|
| MIL-CNN | 0.801 [0.622, 0.858] | 0.038 [0.020, 0.058] | 0.963 |
| MIL-CNN calibrated | 0.786 [0.599, 0.846] | 0.039 | 0.961 |
| logistic | 0.804 [0.587, 0.876] | 0.036 [0.020, 0.053] | 0.964 |
| physics rule | 0.591 [0.374, 0.712] | 0.050 | 0.869 |
| climatology | 0.157 [0.064, 0.256] | 0.099 | 0.700 |

**Honest result:** the deep model (0.801) does not beat logistic regression (0.804); the difference is -0.003
[-0.045, 0.077]. It does beat the physics rule by 0.210 [0.119, 0.326]. Negatives are subsampled 1-in-20, so AUPRC
and Brier describe that inflated universe, not operational rates.

**Routing:** on the Feb 2025 OKC ice case, risk-aware planning (c) closes -1.3% [-5.2%, -0.3%] of the exposure gap,
a hair worse than the shortest route.
