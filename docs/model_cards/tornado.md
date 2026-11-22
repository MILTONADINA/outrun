# Tornado detection (`tornado.cnn`)

- **Task:** tornado probability per radar chip (TorNet), turned into a swath risk layer.
- **Model:** TorNet-style CNN with focal loss, isotonic calibration, operating point frozen on validation
  (`models/tornado`).
- **Data:** TorNet (Zenodo, MIT Lincoln Laboratory, CC BY 4.0), split by the catalog `type` column.
- **Baselines:** uncalibrated CNN (`tornado.cnn_raw`), gradient boosting, range-only, delta-V rule; the published
  TorNet AUC 0.874 is a reference only.

| Split | Model | AUC TOR vs warned null | AUPRC | CSI |
|---|---|---|---|---|
| test | CNN (calibrated) | 0.761 [0.709, 0.810] | 0.396 [0.285, 0.493] | 0.272 [0.206, 0.332] |
| test | CNN raw | 0.762 [0.709, 0.811] | 0.423 [0.305, 0.523] | 0.208 [0.142, 0.261] |
| test | GBM | 0.648 [0.591, 0.698] | 0.156 [0.111, 0.211] | 0.008 |
| test | published TorNet | 0.874 | n/a | n/a |
| case hold-out | CNN | 0.799 [0.688, 0.870] | 0.702 [0.608, 0.828] | 0.438 [0.368, 0.506] |

**Honest result:** calibrated is about equal to the raw CNN on ranking (0.761 vs 0.762). Calibration helps
CSI and Brier, not discrimination. The published AUC is not reached. Reliability is worse on the case hold-out
(ECE 0.157).

**Demo caveat:** the tornado demo never blocks a road; the swath risk raises route cost but no edge is closed. The
May 20 2019 tornado layer over the OKC box is empty (max p 0.000), not investigated.
