# Phase 3 Notes

## Design Decisions

SOH is measured as discharge capacity divided by nominal capacity, clipped to `[0, 1.05]`. For this phase, capacity-derived inputs are excluded because they are the target. The model is intentionally asked to infer SOH from voltage shape, dQ/dV shape, protocol, and chemistry probabilities without seeing `feat_capacity_ratio`, `discharge_capacity_ah`, or `charge_capacity_ah`.

The primary architecture is per-chemistry SOH modeling with a probability-weighted ensemble. At inference, chemistry probabilities weight the per-chemistry SOH distributions and the ensemble variance uses the law of total variance, so chemistry uncertainty propagates into SOH uncertainty.

Both GPR and XGBoost-Quantile are trained. GPR is the principled uncertainty model; XGBoost-Quantile is the scalable deployment baseline.

## Headline Metrics

| chemistry | scope | model | rmse | mae | r2 | picp90 | mean_width | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LCO | per_chemistry | gpr | 0.0693 | 0.0221 | 0.8888 | 0.8583 | 0.0849 | 0.0608 |
| LFP | per_chemistry | gpr | 0.0137 | 0.0079 | 0.9129 | 0.8530 | 0.0293 | 0.0511 |
| all | ensemble | gpr | 0.0353 | 0.0112 | 0.9307 | 0.8873 | 0.0445 | 0.0220 |
| all | shared | gpr | 0.0554 | 0.0163 | 0.8287 | 0.8755 | 0.0529 | 0.0555 |
| LCO | per_chemistry | xgb | 0.0376 | 0.0268 | 0.9672 | 0.8542 | 0.2222 | 0.0758 |
| LFP | per_chemistry | xgb | 0.0109 | 0.0075 | 0.9448 | 0.8409 | 0.0330 | 0.0591 |
| all | ensemble | xgb | 0.0268 | 0.0124 | 0.9598 | 0.9269 | 0.0752 | 0.0442 |
| all | shared | xgb | 0.0383 | 0.0151 | 0.9181 | 0.8590 | 0.0753 | 0.0546 |

## Calibration Commentary

Average PICP@90 is 0.869, so the current SOH interval behavior is reasonably calibrated: average PICP@90 is inside the 0.85-0.95 target band.

## Disagreement Case

Found in `ensemble_gpr_predictions.csv`: cell `calce_cs2_33` cycle 865 measured SOH 0.049, predicted 0.779, 90% interval [0.322, 1.050].

## Caveats

The same Phase 2.5 limitations carry forward: only LCO and LFP are present, each chemistry comes from one source, and c-rate/protocol signatures remain confounded with source. GPR uses stratified subsampling capped at 300 training cycles per fold to keep cubic-time fitting practical.

## Phase 4 Handoff

Phase 4 needs `(soh_mean, soh_lower_90, soh_upper_90, chemistry_probs)`. The `SOHEnsemble.predict` method emits `(mean, lower, upper, std)`, and the chemistry classifier provides the probability vector.
