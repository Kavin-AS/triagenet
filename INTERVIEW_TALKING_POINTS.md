# Interview Talking Points

## Problem Framing

TriageNet is a chemistry-agnostic triage system for retired lithium-ion batteries. The
constraint is the important part: a recycler should not need full cycling history or a
known label from the pack. The system takes one characterization cycle from an incoming
cell, predicts chemistry with calibrated probabilities, then feeds that into SOH
estimation and an economic decision about second-life use versus direct recycling. That
single-cycle, uncertainty-aware framing matters at a recycling intake because the operator
usually sees mixed chemistries, incomplete provenance, and high downside from confident
wrong decisions.

## Methodology Story

The first impressive-looking result was not the result I trusted. A naive classifier could
score above 99% because the two available datasets were structurally different: CALCE LCO
cells were cycled to deep degradation, while MIT LFP cells stopped around 80% SOH. That
made SOH a chemistry proxy. The fix was to train and evaluate inside a matched SOH window
of `[0.80, 1.00]`, so the model had to work where both datasets had coverage. During that
same pass, I also removed energy efficiency from the chemistry feature set because MIT
lacked the field and CALCE had it. Null-presence is not chemistry.

The next audit looked at voltage range. The Phase 2 top features were mostly absolute
voltage statistics, and LFP and LCO occupy different nominal voltage windows. That is a
real battery-chemistry fact, but it is also a fragile deployment shortcut: a shifted
cycler, BMS-clamped second-life cell, or offset measurement can break a model that reads
absolute voltage windows. I kept the original voltage-axis model as a baseline, built a
voltage-normalized shape model, and ran a `+-0.2 V` test-time voltage-shift probe. The
shape model held 99.93% balanced accuracy across the shift probe; the voltage-axis
baseline degraded to 87.09% under `-0.2 V`. The point is not to brag about 99%; the point
is that each high score was treated as a prompt to look for the next confound.

## What's Honestly Unproven

After fixing voltage-range leakage, the model leaned on c-rate. That is another source
confound: CALCE used standard cycling, while the MIT Severson study varied charge rate as
the experimental variable. With one LCO source and one LFP source, dataset identity is
mathematically inseparable from chemistry identity. Removing c-rate would probably just
move the model to another source proxy. The current classifier is valid for CALCE-like and
MIT-like operating conditions, but cross-source chemistry generalization remains
unproven. The cleanest unblock is Sandia, because it gives multiple chemistries from one
lab and protocol family. HNEI or HUST would add additional source diversity and make
leave-one-source-out evaluation meaningful.

## Roadmap

Phase 3 builds the chemistry-aware SOH regressor with calibrated uncertainty, using the
chemistry probabilities from Phase 2 rather than assuming the chemistry label is known.
Phase 4 turns chemistry and SOH into a techno-economic triage decision using commodity
prices, recovery rates, and uncertainty-aware expected value. Phase 5 wraps the pipeline
in a FastAPI service and React dashboard so a recycler can upload a cycle file and inspect
the verdict, probabilities, SOH interval, and economic rationale.

## Phase 5 Demo Results

The final demo surface is a FastAPI backend plus a React dashboard. The dashboard has two
paths: upload a CSV/Parquet slice of cycles, or pick one of the curated real sample cycles.
The backend returns a single verdict JSON with chemistry probabilities, SOH mean and 90%
interval, economics, value of information, and the human-readable rationale.

The headline business result from Phase 4 is visible in the UI and README: risk-aware
routing captured `$1.47/cell` more expected value than the naive 80% SOH threshold on the
held-out evaluation slice, an 8.41% uplift. The required demo moment is `calce_cs2_33`
cycle 865. It has a high SOH mean but a wide interval crossing the 80% threshold, so the
system recommends `needs_more_characterization` instead of pretending the point estimate
is enough.

## Questions I Expect

**How do I know the chemistry classifier is real and not dataset leakage?**  
I do not claim cross-source chemistry generalization yet. I audited SOH, voltage-range,
energy-field, and c-rate confounds; Sandia is the data that would make this test real.

**Why keep a high-accuracy classifier if c-rate is still a confound?**  
Because it is useful inside CALCE/MIT-like conditions, and the audit makes its deployment
boundary explicit instead of hiding it.

**Why per-chemistry SOH models instead of one shared model?**  
SOH degradation shape differs by chemistry. The ensemble lets chemistry uncertainty
propagate into SOH uncertainty through mixture variance.

**Why exclude capacity from the SOH regressor?**  
Capacity ratio is the target. Including it would answer a trivial measurement question,
not the useful question of whether cycle shape agrees with measured capacity.

**Why XGBoost-Quantile if GPR is the headline uncertainty model?**  
GPR is principled and interview-defensible, but XGBoost-Quantile scales better. The API
uses the saved ensemble artifact without retraining.

**Why not deep learning on raw curves?**  
The independent cell count is small and source-confounded. Hand-built physics-aware
features are easier to audit and harder to overfit silently.

**How did you calibrate uncertainty?**  
Phase 3 reports PICP@90, interval width, and ECE. The ensemble XGB interval covered
92.69% of held-out points at nominal 90%.

**What would you do with a year and a budget?**  
Acquire Sandia and additional multi-source datasets, replace economics constants with
Bridge Green process data, and validate decisions at module/pack scale.

**What happens when commodity prices move?**  
The valuation layer is explicit. Prices are inputs, not baked into the ML model, so the
same predictions can be revalued under new market conditions.

**What would production deployment look like?**  
Containerized FastAPI inference, dashboard, model registry, internal commodity/yield
feeds, and audit logs for every route decision.
