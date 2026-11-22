# Outrun metrics

## storm

git sha: 72d9715-dirty

### test

| model | kind | csi_mean_74_133 | brier_133 | csi_133 | csi_16 | csi_160 | csi_181 | csi_219 | csi_74 | ece_133 | fss_133_w3 | fss_133_w7 | mse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| storm.unet_vil_ir | dl | 0.533 [0.520, 0.544] | 0.021 [0.019, 0.022] | 0.438 [0.426, 0.449] | 0.698 [0.684, 0.712] | 0.375 [0.361, 0.388] | 0.332 [0.314, 0.347] | 0.105 [0.096, 0.114] | 0.627 [0.612, 0.641] | 0.002 [0.001, 0.002] | 0.672 [0.662, 0.682] | 0.736 [0.727, 0.744] | 732.218 [687.847, 776.268] |
| persistence | baseline | 0.352 [0.339, 0.364] | n/a | 0.235 [0.224, 0.246] | 0.600 [0.582, 0.616] | 0.153 [0.145, 0.162] | 0.116 [0.107, 0.124] | 0.063 [0.057, 0.070] | 0.470 [0.453, 0.485] | n/a | 0.443 [0.427, 0.460] | 0.538 [0.520, 0.556] | 1189.329 [1100.398, 1272.514] |
| pysteps_extrapolation | baseline | 0.467 [0.455, 0.477] | n/a | 0.342 [0.332, 0.352] | 0.690 [0.677, 0.702] | 0.256 [0.245, 0.267] | 0.208 [0.197, 0.220] | 0.125 [0.113, 0.136] | 0.591 [0.577, 0.604] | n/a | 0.586 [0.575, 0.596] | 0.683 [0.672, 0.692] | 735.716 [682.870, 787.235] |

**test:** DL model `storm.unet_vil_ir` beats the best baseline `pysteps_extrapolation` on csi_mean_74_133: 0.533 vs 0.467.

### case_holdout

| model | kind | csi_mean_74_133 | brier_133 | csi_133 | csi_16 | csi_160 | csi_181 | csi_219 | csi_74 | ece_133 | fss_133_w3 | fss_133_w7 | mse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| storm.unet_vil_ir | dl | 0.563 [0.536, 0.587] | 0.029 [0.023, 0.035] | 0.451 [0.427, 0.469] | 0.740 [0.703, 0.776] | 0.384 [0.365, 0.401] | 0.341 [0.328, 0.356] | 0.111 [0.097, 0.124] | 0.676 [0.640, 0.707] | 0.002 [0.001, 0.004] | 0.676 [0.652, 0.694] | 0.733 [0.709, 0.751] | 836.962 [765.902, 915.237] |
| persistence | baseline | 0.366 [0.333, 0.395] | n/a | 0.221 [0.202, 0.237] | 0.646 [0.595, 0.696] | 0.135 [0.117, 0.154] | 0.097 [0.081, 0.113] | 0.047 [0.037, 0.056] | 0.512 [0.458, 0.559] | n/a | 0.420 [0.390, 0.446] | 0.507 [0.474, 0.538] | 1487.583 [1335.966, 1654.477] |
| pysteps_extrapolation | baseline | 0.479 [0.466, 0.489] | n/a | 0.338 [0.323, 0.358] | 0.709 [0.687, 0.732] | 0.253 [0.238, 0.274] | 0.207 [0.192, 0.225] | 0.126 [0.113, 0.143] | 0.619 [0.598, 0.636] | n/a | 0.566 [0.545, 0.595] | 0.650 [0.624, 0.683] | 963.773 [804.398, 1127.410] |

**case_holdout:** DL model `storm.unet_vil_ir` beats the best baseline `pysteps_extrapolation` on csi_mean_74_133: 0.563 vs 0.479.

### ablations

- vil_only: `storm.unet_vil_only` csi_mean_74_133 0.536 [0.522, 0.548]
- vil_only@case_holdout: `storm.unet_vil_only` csi_mean_74_133 0.570 [0.541, 0.594]

## tornado

git sha: 72d9715-dirty

### test

| model | kind | auc_tor_vs_warned_null | auc | auc_range_0-50 | auc_range_100+ | auc_range_50-100 | auprc | brier | csi | ece | far | pod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tornado.cnn | dl | 0.761 [0.709, 0.810] | 0.861 [0.825, 0.889] | 0.884 [n/a] | 0.816 [n/a] | 0.872 [n/a] | 0.396 [0.285, 0.493] | 0.051 [n/a] | 0.272 [0.206, 0.332] | 0.015 [n/a] | 0.649 [n/a] | 0.550 [n/a] |
| tornado.cnn_raw | baseline | 0.762 [0.709, 0.811] | 0.862 [0.826, 0.891] | 0.886 [n/a] | 0.818 [n/a] | 0.873 [n/a] | 0.423 [0.305, 0.523] | 0.075 [n/a] | 0.208 [0.142, 0.261] | n/a | 0.402 [n/a] | 0.241 [n/a] |
| tornado.gbm | baseline | 0.648 [0.591, 0.698] | 0.733 [0.693, 0.769] | 0.729 [n/a] | 0.696 [n/a] | 0.707 [n/a] | 0.156 [0.111, 0.211] | 0.062 [n/a] | 0.008 [0.003, 0.013] | n/a | 0.806 [n/a] | 0.008 [n/a] |
| tornado.range_only | baseline | 0.553 [0.463, 0.641] | 0.629 [0.564, 0.686] | 0.569 [n/a] | 0.622 [n/a] | 0.495 [n/a] | 0.103 [0.066, 0.155] | 0.063 [n/a] | 0.000 [0.000, 0.000] | n/a | n/a | 0.000 [n/a] |
| tornado.delta_v_rule | baseline | 0.533 [0.496, 0.572] | 0.533 [0.504, 0.560] | 0.521 [n/a] | 0.506 [n/a] | 0.542 [n/a] | 0.073 [0.051, 0.099] | 0.614 [n/a] | 0.073 [0.051, 0.098] | n/a | 0.925 [n/a] | 0.704 [n/a] |
| tornet_published | reference | 0.874 [n/a] | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**test:** DL model `tornado.cnn` does not beat the best baseline `tornado.cnn_raw` on auc_tor_vs_warned_null: 0.761 vs 0.762.

### case_holdout

| model | kind | auc_tor_vs_warned_null | auc | auc_range_0-50 | auc_range_100+ | auc_range_50-100 | auprc | brier | csi | ece | far | pod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tornado.cnn | dl | 0.799 [0.688, 0.870] | 0.857 [0.798, 0.902] | 0.886 [n/a] | 0.834 [n/a] | 0.842 [n/a] | 0.702 [0.608, 0.828] | 0.176 [n/a] | 0.438 [0.368, 0.506] | 0.157 [n/a] | 0.329 [n/a] | 0.558 [n/a] |
| tornado.cnn_raw | baseline | 0.800 [0.689, 0.872] | 0.858 [0.799, 0.904] | 0.885 [n/a] | 0.835 [n/a] | 0.843 [n/a] | 0.721 [0.629, 0.844] | 0.149 [n/a] | 0.272 [0.169, 0.354] | n/a | 0.171 [n/a] | 0.288 [n/a] |
| tornado.gbm | baseline | 0.633 [0.545, 0.700] | 0.726 [0.645, 0.789] | 0.706 [n/a] | 0.652 [n/a] | 0.688 [n/a] | 0.500 [0.335, 0.675] | 0.212 [n/a] | 0.018 [0.008, 0.032] | n/a | 0.400 [n/a] | 0.018 [n/a] |
| tornado.range_only | baseline | 0.665 [0.565, 0.754] | 0.701 [0.619, 0.774] | 0.581 [n/a] | 0.572 [n/a] | 0.526 [n/a] | 0.500 [0.352, 0.628] | 0.235 [n/a] | 0.000 [0.000, 0.000] | n/a | n/a | 0.000 [n/a] |
| tornado.delta_v_rule | baseline | 0.527 [0.470, 0.576] | 0.537 [0.485, 0.587] | 0.526 [n/a] | 0.552 [n/a] | 0.509 [n/a] | 0.297 [0.187, 0.445] | 0.555 [n/a] | 0.274 [0.177, 0.399] | n/a | 0.697 [n/a] | 0.746 [n/a] |
| tornet_published | reference | 0.874 [n/a] | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**case_holdout:** DL model `tornado.cnn` does not beat the best baseline `tornado.cnn_raw` on auc_tor_vs_warned_null: 0.799 vs 0.800.

## flood

git sha: 72d9715-dirty

### test

| model | kind | auprc | auc_vs_hard_negative | brier | csi_pixel_p50 | ece | event_pod_p50 | far_pixel_p50 | hard_neg_alarm_rate_p50 | neighbourhood_csi_r2_p50 | pod_pixel_p50 | roc_auc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flood.unet | dl | 0.224 [0.191, 0.255] | 0.841 [0.821, 0.856] | 0.038 [0.033, 0.044] | 0.035 [0.025, 0.045] | 0.001 [0.001, 0.005] | 0.096 [0.077, 0.114] | 0.350 [0.236, 0.454] | 0.006 [0.003, 0.008] | 0.052 [0.038, 0.065] | 0.036 [0.026, 0.046] | 0.806 [0.791, 0.818] |
| flood.unet_raw | dl | 0.226 [0.193, 0.258] | 0.841 [0.821, 0.856] | 0.048 [0.044, 0.053] | 0.023 [0.015, 0.031] | 0.100 [0.096, 0.104] | 0.066 [0.050, 0.080] | 0.276 [0.175, 0.382] | 0.003 [0.001, 0.005] | 0.034 [0.023, 0.045] | 0.023 [0.015, 0.031] | 0.806 [0.791, 0.818] |
| ffg_ratio_ge_1 | baseline | 0.065 [0.054, 0.075] | 0.532 [0.524, 0.540] | 0.046 [0.039, 0.054] | 0.060 [0.047, 0.073] | n/a | 0.245 [0.213, 0.271] | 0.641 [0.549, 0.705] | 0.159 [0.145, 0.173] | 0.110 [0.093, 0.127] | 0.067 [0.051, 0.083] | 0.531 [0.524, 0.538] |
| qpe_ari_ge_10yr | baseline | 0.061 [0.050, 0.070] | 0.519 [0.514, 0.527] | 0.044 [0.038, 0.051] | 0.040 [0.029, 0.053] | n/a | 0.191 [0.169, 0.215] | 0.559 [0.496, 0.630] | 0.156 [0.143, 0.169] | 0.079 [0.062, 0.098] | 0.042 [0.030, 0.056] | 0.520 [0.514, 0.527] |
| ari_rule_3h50_6h75 | baseline | 0.068 [0.056, 0.079] | 0.540 [0.530, 0.550] | 0.048 [0.042, 0.055] | 0.074 [0.059, 0.090] | n/a | 0.235 [0.209, 0.258] | 0.677 [0.618, 0.732] | 0.222 [0.206, 0.239] | 0.110 [0.092, 0.129] | 0.087 [0.068, 0.108] | 0.539 [0.531, 0.550] |
| logistic_regression | baseline | 0.144 [0.119, 0.167] | 0.680 [0.641, 0.713] | 0.040 [0.034, 0.047] | 0.026 [0.017, 0.036] | 0.005 [0.001, 0.011] | 0.137 [0.113, 0.158] | 0.567 [0.397, 0.680] | 0.054 [0.047, 0.062] | 0.055 [0.041, 0.068] | 0.027 [0.018, 0.037] | 0.690 [0.660, 0.714] |

**test:** DL model `flood.unet_raw` beats the best baseline `logistic_regression` on auprc: 0.226 vs 0.144.

### ablations

- no_flash: `flood.unet_noflash` auprc 0.207 [0.175, 0.236]
- no_flash: `flood.unet_noflash_raw` auprc 0.210 [0.177, 0.239]

## wildfire

git sha: 72d9715-dirty

### test

| model | kind | prauc | brier | ece | prauc_new_burn |
|---|---|---|---|---|---|
| wildfire.unet | dl | 0.357 [0.340, 0.373] | 0.010 [0.009, 0.011] | 0.003 [0.002, 0.003] | 0.260 [0.246, 0.274] |
| persistence | baseline | 0.107 [0.097, 0.117] | 0.015 [0.014, 0.016] | 0.015 [0.014, 0.016] | 0.009 [0.009, 0.010] |
| downwind_persistence | baseline | 0.114 [0.104, 0.124] | 0.017 [0.016, 0.019] | 0.017 [0.016, 0.019] | 0.026 [0.024, 0.028] |
| pixel_logistic | baseline | 0.153 [0.139, 0.168] | 0.011 [0.010, 0.012] | 0.001 [0.000, 0.002] | 0.020 [0.018, 0.022] |
| ndws_published | reference | 0.284 [n/a] | n/a | n/a | n/a |

**test:** DL model `wildfire.unet` beats the best baseline `pixel_logistic` on prauc: 0.357 vs 0.153.

### test_wsts2021

| model | kind | prauc | brier | ece | prauc_new_burn | prauc_per_fire_macro |
|---|---|---|---|---|---|---|
| wildfire.unet | dl | 0.372 [0.323, 0.410] | 0.005 [0.004, 0.006] | 0.011 [0.010, 0.012] | 0.060 [0.051, 0.067] | 0.205 [0.178, 0.233] |
| persistence | baseline | 0.433 [0.392, 0.464] | 0.004 [0.003, 0.004] | 0.004 [0.003, 0.004] | 0.002 [0.001, 0.002] | 0.234 [0.207, 0.263] |
| downwind_persistence | baseline | 0.391 [0.356, 0.417] | 0.005 [0.004, 0.006] | 0.005 [0.004, 0.006] | 0.039 [0.035, 0.042] | 0.216 [0.191, 0.242] |
| pixel_logistic | baseline | 0.456 [0.416, 0.488] | 0.004 [0.003, 0.004] | 0.012 [0.011, 0.012] | 0.003 [0.002, 0.004] | 0.252 [0.225, 0.283] |

**test_wsts2021:** DL model `wildfire.unet` does not beat the best baseline `pixel_logistic` on prauc: 0.372 vs 0.456.

### ablations

- no_calibration: `wildfire.unet_uncalibrated` prauc 0.372 [0.323, 0.410]

## ice

git sha: 72d9715-dirty

### test

| model | kind | zone_auprc | auprc_minus_logistic | auprc_minus_physics_rule | brier | bss_vs_climatology | csi | ece | roc_auc |
|---|---|---|---|---|---|---|---|---|---|
| ice.milcnn | dl | 0.801 [0.622, 0.858] | -0.003 [-0.045, 0.077] | 0.210 [0.119, 0.326] | 0.038 [0.020, 0.058] | 0.581 [0.482, 0.638] | 0.570 [0.401, 0.642] | 0.011 [0.008, 0.028] | 0.963 [0.949, 0.979] |
| ice.milcnn_calibrated | dl | 0.786 [0.599, 0.846] | n/a | n/a | 0.039 [0.018, 0.062] | 0.568 [0.498, 0.633] | 0.505 [0.365, 0.566] | 0.015 [0.006, 0.040] | 0.961 [0.947, 0.978] |
| physics_rule | baseline | 0.591 [0.374, 0.712] | n/a | n/a | 0.050 [0.030, 0.070] | 0.456 [0.225, 0.602] | 0.605 [0.422, 0.707] | 0.050 [0.030, 0.070] | 0.869 [0.814, 0.908] |
| logistic | baseline | 0.804 [0.587, 0.876] | n/a | n/a | 0.036 [0.020, 0.053] | 0.600 [0.469, 0.677] | 0.606 [0.398, 0.699] | 0.010 [0.008, 0.026] | 0.964 [0.937, 0.976] |
| climatology | baseline | 0.157 [0.064, 0.256] | n/a | n/a | 0.099 [0.056, 0.145] | -0.090 [-0.283, -0.002] | 0.178 [0.070, 0.276] | 0.075 [0.060, 0.108] | 0.700 [0.656, 0.749] |

**test:** DL model `ice.milcnn` does not beat the best baseline `logistic` on zone_auprc: 0.801 vs 0.804.

## smoke

git sha: 72d9715-dirty

### test

| model | kind | rmse_smoke_episodes | csi_35_4 | far_35_4 | mae | mae_smoke_episodes | pod_35_4 | q90_coverage | q90_coverage_episodes | q90_coverage_shifted | rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke.cnn_mlp | dl | 10.209 [8.775, 11.951] | 0.114 [0.095, 0.132] | 0.363 [0.322, 0.406] | 2.574 [2.521, 2.636] | 4.010 [3.788, 4.275] | 0.122 [0.100, 0.142] | 0.911 [0.909, 0.913] | 0.901 [0.893, 0.908] | 0.906 [0.904, 0.908] | 5.603 [5.007, 6.399] |
| persistence | baseline | 12.420 [10.818, 14.547] | 0.182 [0.163, 0.200] | 0.692 [0.667, 0.719] | 3.229 [3.154, 3.318] | 4.821 [4.540, 5.150] | 0.308 [0.280, 0.334] | n/a | n/a | n/a | 6.984 [6.227, 7.999] |
| hrrr_calibrated | baseline | 12.198 [10.679, 13.913] | 0.053 [0.040, 0.069] | 0.573 [0.516, 0.630] | 3.642 [3.568, 3.713] | 5.455 [5.118, 5.814] | 0.057 [0.043, 0.076] | n/a | n/a | n/a | 6.737 [6.109, 7.540] |
| linear_regression | baseline | 26.276 [12.541, 42.895] | 0.101 [0.078, 0.127] | 0.532 [0.480, 0.583] | 5.541 [5.448, 5.638] | 7.309 [6.973, 7.712] | 0.114 [0.087, 0.145] | n/a | n/a | n/a | 9.642 [7.981, 12.239] |

**test:** DL model `smoke.cnn_mlp` beats the best baseline `hrrr_calibrated` on rmse_smoke_episodes: 10.209 vs 12.198.

### test_spatial

| model | kind | rmse_smoke_episodes | csi_35_4 | far_35_4 | mae | mae_smoke_episodes | pod_35_4 | q90_coverage | q90_coverage_episodes | q90_coverage_shifted | rmse |
|---|---|---|---|---|---|---|---|---|---|---|---|
| smoke.cnn_mlp | dl | 12.114 [9.807, 14.544] | 0.113 [0.086, 0.140] | 0.356 [0.286, 0.436] | 2.710 [2.648, 2.777] | 4.549 [4.177, 4.943] | 0.120 [0.091, 0.150] | 0.904 [0.902, 0.907] | 0.890 [0.879, 0.899] | 0.898 [0.896, 0.901] | 5.883 [5.274, 6.583] |
| persistence | baseline | 14.799 [12.024, 17.737] | 0.188 [0.164, 0.214] | 0.685 [0.648, 0.719] | 3.332 [3.248, 3.425] | 5.536 [5.065, 6.055] | 0.317 [0.281, 0.354] | n/a | n/a | n/a | 6.999 [6.382, 7.761] |
| hrrr_calibrated | baseline | 13.670 [11.536, 16.009] | 0.035 [0.023, 0.049] | 0.649 [0.579, 0.721] | 3.805 [3.728, 3.884] | 6.128 [5.664, 6.623] | 0.038 [0.024, 0.053] | n/a | n/a | n/a | 6.835 [6.272, 7.489] |
| linear_regression | baseline | 14.654 [12.624, 16.881] | 0.133 [0.096, 0.171] | 0.453 [0.393, 0.531] | 5.890 [5.788, 5.999] | 7.822 [7.424, 8.284] | 0.149 [0.105, 0.195] | n/a | n/a | n/a | 8.583 [8.160, 9.115] |

**test_spatial:** DL model `smoke.cnn_mlp` beats the best baseline `hrrr_calibrated` on rmse_smoke_episodes: 12.114 vs 13.670.

### ablations

- no_hms: `smoke.cnn_mlp_no_hms` rmse_smoke_episodes 10.283 [8.624, 12.356]
- no_hms: `smoke.cnn_mlp_no_hms[test_spatial]` rmse_smoke_episodes 11.922 [9.652, 14.301]

## Routing: elreno-2019

git sha: 72d9715-dirty

origins: 100, paired: 100

| planner | n_routed | n_failed | pct_intersect | mean_exposure_risk_min | mean_travel_time_ratio | blocked_crossings | mean_predicted_exposure |
|---|---|---|---|---|---|---|---|
| a | 100 | 0 | 65.000 | 3.146 | 1.000 | 0 | 0.000 |
| a_prime | 100 | 0 | 65.000 | 2.920 | 0.999 | 0 | 1.553 |
| b | 100 | 0 | 65.000 | 3.133 | 1.057 | 0 | 1.005 |
| c | 100 | 0 | 65.000 | 2.920 | 0.999 | 0 | 1.553 |
| d | 100 | 0 | 47.000 | 1.793 | 1.106 | 0 | 1.832 |

DL planning (c) closes 16.7% [4.9%, 31.0%] of the observed-exposure gap between (a) and (d).

c minus b (mean observed exposure, negative = DL better): -0.213 [-0.476, 0.008]

c minus a': 0.000 [0.000, 0.000]

## Routing: flood-okc-2025

git sha: 72d9715-dirty

origins: 100, paired: 95

| planner | n_routed | n_failed | pct_intersect | mean_exposure_risk_min | mean_travel_time_ratio | blocked_crossings | mean_predicted_exposure |
|---|---|---|---|---|---|---|---|
| a | 100 | 0 | 15.789 | 1.156 | 1.000 | 11 | 0.000 |
| a_prime | 95 | 5 | 12.632 | 0.798 | 1.034 | 4 | 0.547 |
| b | 100 | 0 | 14.737 | 0.965 | 1.027 | 10 | 0.055 |
| c | 95 | 5 | 12.632 | 0.798 | 1.034 | 4 | 0.547 |
| d | 98 | 2 | 12.632 | 0.758 | 1.052 | 0 | 0.758 |

DL planning (c) closes 90.0% [32.8%, 121.8%] of the observed-exposure gap between (a) and (d).

c minus b (mean observed exposure, negative = DL better): -0.167 [-0.578, 0.228]

c minus a': 0.000 [0.000, 0.000]

## Routing: ice-okc-2025

git sha: 72d9715-dirty

origins: 100, paired: 100

| planner | n_routed | n_failed | pct_intersect | mean_exposure_risk_min | mean_travel_time_ratio | blocked_crossings | mean_predicted_exposure |
|---|---|---|---|---|---|---|---|
| a | 100 | 0 | 83.000 | 7.850 | 1.000 | 0 | 0.000 |
| a_prime | 100 | 0 | 83.000 | 7.856 | 1.001 | 0 | 0.140 |
| b | 100 | 0 | 83.000 | 8.250 | 1.049 | 0 | 1.676 |
| c | 100 | 0 | 83.000 | 7.856 | 1.001 | 0 | 0.140 |
| d | 100 | 0 | 81.000 | 7.362 | 1.005 | 0 | 7.362 |

DL planning (c) closes -1.3% [-5.2%, -0.3%] of the observed-exposure gap between (a) and (d).

c minus b (mean observed exposure, negative = DL better): -0.394 [-0.782, -0.091]

c minus a': 0.000 [0.000, 0.000]

## Routing: may20-2019

git sha: 72d9715-dirty

origins: 500, paired: 500

| planner | n_routed | n_failed | pct_intersect | mean_exposure_risk_min | mean_travel_time_ratio | blocked_crossings | mean_predicted_exposure |
|---|---|---|---|---|---|---|---|
| a | 500 | 0 | 94.000 | 5.903 | 1.000 | 0 | 0.000 |
| a_prime | 500 | 0 | 93.200 | 5.689 | 1.002 | 0 | 5.236 |
| b | 500 | 0 | 93.800 | 5.969 | 1.037 | 0 | 4.853 |
| c | 500 | 0 | 93.200 | 5.681 | 1.001 | 0 | 5.242 |
| d | 500 | 0 | 91.000 | 5.349 | 1.007 | 0 | 5.370 |

DL planning (c) closes 40.0% [17.6%, 56.1%] of the observed-exposure gap between (a) and (d).

c minus b (mean observed exposure, negative = DL better): -0.288 [-0.447, -0.137]

c minus a': -0.008 [-0.023, 0.000]

## Routing: smoke-okc-2024

git sha: 72d9715-dirty

origins: 100, paired: 100

| planner | n_routed | n_failed | pct_intersect | mean_exposure_risk_min | mean_travel_time_ratio | blocked_crossings | mean_predicted_exposure |
|---|---|---|---|---|---|---|---|
| a | 100 | 0 | 0.000 | 0.000 | 1.000 | 0 | 0.000 |
| a_prime | 100 | 0 | 0.000 | 0.000 | 1.000 | 0 | 0.000 |
| b | 100 | 0 | 0.000 | 0.000 | 1.000 | 0 | 0.000 |
| c | 100 | 0 | 0.000 | 0.000 | 1.000 | 0 | 0.000 |
| d | 100 | 0 | 0.000 | 0.000 | 1.000 | 0 | 0.000 |

DL planning (c) closes n/a of the observed-exposure gap between (a) and (d).

c minus b (mean observed exposure, negative = DL better): 0.000 [0.000, 0.000]

c minus a': 0.000 [0.000, 0.000]

## Group: may20-2019

git sha: 72d9715-dirty

### Shelters

| method | overflow_persons | unserved_persons | max_minutes |
|---|---|---|---|
| mcf | 0 | 112461 | 28.470 |
| nearest | 114983 | 0 | 25.917 |
| greedy | 0 | 112461 | 75.184 |

### Evacuation

| method | seeds | vehicle_hours | exposure_veh_h | clearance_min | max_vc | gap_to_lb_pct |
|---|---|---|---|---|---|---|
| shortest | 5 | 9416.533 | 3344.773 | 32.469 | 7.820 | 595.393 |
| safest | 5 | 9416.854 | 3344.770 | 32.469 | 7.820 | 595.406 |
| msa | 5 | 2962.027 | 814.051 | 94.772 | 4.467 | 543.152 |
| sa | 5 | 1891.798 | 2779.210 | 33.298 | 3.218 | 264.511 |
| sa_lambda0 | 5 | 2272.982 | 3155.064 | 27.071 | 3.897 | 263.890 |

lower bound: 2413.98

### Emergency vehicles

| dispatch | mean_eta_min | mean_detour_min | pct_hit_block |
|---|---|---|---|
| hungarian | 4.149 | 0.000 | 0.000 |
| reroute_baseline | 3.968 | 0.000 | 0.000 |

Pareto points: 15

## Latency

| name | n | median_s | p95_s | max_s | git sha |
|---|---|---|---|---|---|
| sa_run | 5 | 4.941 | 5.186 | 5.219 | 72d9715-dirty |
| astar_metro_query | 20 | 0.215 | 0.325 | 0.344 | 72d9715-dirty |

