# Phase 2 Notes

## Dataset Asymmetry

Phase 1.5 data is asymmetric: CALCE contributes LCO cycles down to very low SOH, while
MIT/Stanford contributes LFP cycles only down to roughly 80% SOH. Training chemistry on
the full table would let the model learn dataset degradation depth instead of chemistry.

Phase 2 therefore trains the chemistry classifier only inside the matched SOH window
`[0.80, 1.00]`. The raw matched window contains:

| Dataset | Chemistry | Cells | Cycles | Min SOH | Max SOH |
| --- | --- | ---: | ---: | ---: | ---: |
| calce | LCO | 12 | 7427 | 0.800191 | 0.999828 |
| mit | LFP | 41 | 34298 | 0.800051 | 0.998485 |

For model fitting and evaluation, the training script applies a deterministic cap of 30
SOH-stratified cycles per cell. This prevents cells with longer tests from dominating the
cycle-level rows. The capped training table is:

| Dataset | Chemistry | Cells | Cycles | Min SOH | Max SOH |
| --- | --- | ---: | ---: | ---: | ---: |
| calce | LCO | 12 | 360 | 0.800191 | 0.999828 |
| mit | LFP | 41 | 1230 | 0.800051 | 0.998485 |

## Metrics

| Split | Status | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| Leave-one-cell-out, matched SOH | ok | 0.9988 | 0.9993 | 0.9980 | 1.0000 |
| Leave-one-dataset-out, matched SOH | untrainable | n/a | n/a | n/a | n/a |
| Leave-one-cell-out, full SOH baseline | ok | 0.9993 | 0.9996 | 0.9990 | 1.0000 |

Leave-one-dataset-out is untrainable with the current data, not merely low scoring:
holding out CALCE leaves only LFP in training, and holding out MIT leaves only LCO in
training. This will become meaningful once Sandia or another multi-chemistry/multi-source
dataset lands.

The full-SOH baseline is slightly higher than the matched-SOH model, but both are near
ceiling under leave-one-cell-out. That means the current LCO/LFP separation is easy even
after SOH matching, but this is not yet proof of cross-source chemistry generalization.

## Top Features

| Feature | Importance | Interpretation |
| --- | ---: | --- |
| `feat_voltage_mean` | 0.2759 | Captures the broad voltage level difference between LCO and LFP discharge curves. |
| `feat_voltage_std` | 0.2283 | Measures curve spread; LFP and LCO have visibly different discharge-shape variance. |
| `feat_plateau_flatness` | 0.2128 | Physics-aligned LFP plateau descriptor; lower middle-window variance indicates a flatter plateau. |
| `feat_voltage_at_20pct_capacity` | 0.1434 | Early-discharge voltage location, useful for separating flat vs. sloped chemistries. |
| `feat_voltage_at_80pct_capacity` | 0.1136 | Late-discharge voltage location, complementary to the 20% and 50% voltage features. |

`feat_capacity_ratio` is excluded from chemistry features to prevent SOH leakage.
`feat_energy_efficiency` is also excluded for Phase 2 because MIT rows do not contain that
field while CALCE rows do, so it acted as dataset provenance rather than chemistry.

## Artifacts

- `models/chemistry_xgb.joblib`
- `models/chemistry_metadata.json`
- `reports/phase2_chemistry/feature_importance.png`
- `reports/phase2_chemistry/leave_one_cell_out_*`
- `reports/phase2_chemistry/without_soh_window_*`
- `notebooks/02_chemistry_classifier.ipynb`

## Caveats

The independent sample size is still only 53 cells across 2 chemistries. Dataset and
chemistry are fully confounded: CALCE is LCO and MIT is LFP. The model cannot yet claim
manufacturer-independent chemistry generalization, and leave-one-dataset-out cannot train
until at least one dataset contributes multiple chemistries or one chemistry appears in
multiple datasets.

The dQ/dV features were not among the top features in this run. The voltage curves alone
separate current LCO/LFP data strongly, but Phase 3/4 should revisit IC features after
Sandia adds NMC/NCA and better source diversity.

## Phase 3 Handoff

Phase 3 can consume the saved chemistry classifier and metadata. The SOH regressor should
reintroduce `feat_capacity_ratio`, remain chemistry-aware, and report uncertainty. It
should not treat the near-ceiling Phase 2 leave-one-cell-out score as evidence that
cross-dataset generalization is solved.

## Phase 2.5: Voltage-Leakage Audit

The Phase 2 top features were dominated by absolute-voltage statistics, which is a
fragile shortcut because LFP and LCO occupy different nominal voltage windows. I added a
second voltage-normalized shape feature set, retrained the production classifier on it,
and kept the original absolute-voltage model as `models/chemistry_xgb_voltage_axis.joblib`
for comparison. The active production artifact `models/chemistry_xgb.joblib` now uses
`SHAPE_FEATURES`.

| Feature set | Matched-SOH balanced accuracy | Acc @ shift +0.2 V | Acc @ shift -0.2 V |
| --- | ---: | ---: | ---: |
| Absolute voltage (Phase 2) | 0.9993 | 0.9993 | 0.8709 |
| Shape (Phase 2.5) | 0.9993 | 0.9993 | 0.9993 |

Voltage-shift probe details are saved in
`reports/phase2_chemistry/voltage_robustness.json`, with the plot at
`reports/phase2_chemistry/voltage_robustness.png`. The absolute-voltage baseline did not
collapse symmetrically: it was robust to positive shifts but degraded sharply at `-0.2 V`.
The shape model held steady across `[-0.2, -0.1, 0.0, +0.1, +0.2] V`.

New top 5 features for the shape model:

| Feature | Importance | Interpretation |
| --- | ---: | --- |
| `feat_c_rate_charge` | 0.2751 | Charging protocol signal; useful but potentially source-confounded. |
| `feat_shape_curve_kurtosis` | 0.2558 | Normalized voltage curve sharpness/tailedness. |
| `feat_shape_plateau_flatness_normalized` | 0.2527 | Voltage plateau flatness after removing absolute voltage scale. |
| `feat_shape_v_norm_at_75pct_capacity` | 0.2018 | Late-discharge normalized voltage position. |
| `feat_c_rate_discharge` | 0.0092 | Discharge protocol signal; small but still a possible source artifact. |

What we still cannot prove: this is still a two-chemistry, two-source dataset where CALCE
is LCO and MIT is LFP. The voltage-shift probe is a partial substitute for deployment
shift testing, not a replacement for cross-source chemistry validation. The fact that
C-rate appears in the shape model's top features is another honest warning that cycling
protocol may still be confounded with chemistry.

Interview framing: this audit is a deliberate methodology choice. The point is not that
Phase 2 had a software bug; the point is that a near-perfect metric deserved an adversarial
leakage test. Preserving the voltage-axis baseline makes the robustness story testable
instead of rhetorical.

## Confound Chain: What the Audits Actually Revealed

The first confound was SOH-asymmetry leakage. CALCE cycles LCO cells down to roughly 5%
SOH, while MIT/Stanford stops its LFP cells around 80% SOH. A naive EOL-cycle classifier
can therefore clear 99% accuracy by learning degradation depth instead of chemistry. The
mitigation was the matched-SOH window `[0.80, 1.00]`, which forces both LCO and LFP rows
into the same health range before chemistry training or evaluation.

The second confound was energy-efficiency provenance leakage, caught during Phase 2
implementation. CALCE provides `energy_efficiency`, while MIT does not. If that feature is
included with a simple imputation policy, null-presence becomes a dataset identifier. The
mitigation was an explicit exclusion from chemistry features. That was not a modeling
optimization; it was removing a field that told the model where the row came from.

The third confound was voltage-range leakage, addressed in Phase 2.5. LFP and LCO have
largely non-overlapping discharge voltage windows, roughly 2.5-3.4 V for LFP and
3.0-4.2 V for LCO in this data. Absolute-voltage features achieved better than 99%
balanced accuracy by reading the voltage window. The mitigation was a voltage-normalized
shape feature set plus a voltage-shift robustness probe at `-0.2`, `-0.1`, `0.0`, `+0.1`,
and `+0.2 V`. The result was useful: the shape model held 99.93% balanced accuracy under
the shift probe, while the absolute-voltage model degraded to 87.09% at `-0.2 V`.

The fourth confound is c-rate and protocol leakage, identified in Phase 2.5 and not yet
mitigated. After removing the easy voltage-axis signal, the shape model leaned on
`feat_c_rate_charge` with importance `0.2751`. CALCE used standard 0.5C-1C cycling, while
MIT's Severson study varied charge rate as the experimental variable. C-rate is therefore
a strong proxy for dataset, and dataset is currently a perfect proxy for chemistry.

The structural conclusion is simple: with only one LCO source and one LFP source, dataset
identity is mathematically inseparable from chemistry identity. Every cheap proxy for
"which dataset" is also a cheap proxy for "which chemistry." Removing c-rate would not
magically solve the problem; it would push the model toward the next source-specific
signature, such as cycle-count distribution, temperature distribution, internal resistance
characteristics, or another protocol artifact.

The chemistry classifier is therefore valid for cells drawn from CALCE-like or MIT-like
operating conditions, and provisional outside that regime. Production deployment requires
either cross-chemistry data from a single lab, which decouples chemistry from protocol and
manufacturer, or deliberate domain-randomization across many sources per chemistry.

## What Unblocks Real Generalization

The highest-value acquisition is the Sandia National Labs cycling dataset. It covers four
chemistries from one lab and one protocol family, which directly attacks the current
dataset-equals-chemistry confound. Sandia is the single most important addition and is
currently email-pending.

The second-best path is adding HNEI or HUST data, especially additional LFP and NMC
sources. Those datasets would let us ask a more realistic question: given a new source for
a chemistry already seen during training, does the model still recognize the chemistry
instead of the lab protocol?

The gold-standard near-term dataset mix is Sandia plus HUST. Together they would provide
multi-source coverage per chemistry and make leave-one-source-out evaluation a real test
instead of an untrainable split. Until at least Sandia lands, leave-one-dataset-out cannot
be meaningfully run, and any chemistry-classification accuracy should be qualified as:
"validated under matched-SOH and voltage-shift conditions; cross-source chemistry
generalization remains unproven."
