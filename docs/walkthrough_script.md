# TriageNet Walkthrough Script

## 0:00-0:30 The Problem

Bridge Green-style recycling intake has to decide what to do with retired lithium-ion cells before
spending much time on them. A cell can be routed to second-life use, direct recycling, or more
testing. TriageNet is built around that intake decision: one characterization cycle in, a business
verdict out.

## 0:30-1:30 The Methodology

The pipeline has three stages. First, a chemistry classifier estimates LFP versus LCO from
single-cycle shape features. Second, a chemistry-aware SOH regressor predicts state of health with
a 90% interval. Third, the triage layer converts those uncertain predictions into expected dollar
value and value of information.

The important part is the audit trail. The first classifier looked almost perfect, which was a red
flag rather than a victory. I found SOH-asymmetry leakage, then voltage-range leakage, then c-rate
protocol leakage. The current result is useful under matched conditions, but I am explicit that true
cross-source chemistry generalization needs Sandia or another multi-chemistry source.

## 1:30-2:30 The Dashboard Demo

Open the dashboard and choose a sample cell. The right side shows the verdict first, then chemistry
probabilities, SOH with the 90% interval, and the economics. For a high-SOH LFP sample, the system
routes to second life. For a deeply degraded LCO sample, it routes to recycle.

Now choose CALCE CS2_33 cycle 865. This is the demo moment. The mean SOH looks healthy, but the
interval crosses the 80% threshold. The risk-aware rule does not pretend to know more than it does:
it recommends one more characterization cycle.

## 2:30-3:30 The Economics

The economics card compares recycle expected value against second-life expected value. It also shows
the value of information. In Phase 4, risk-aware routing recovered $1.47 per evaluated cell versus
the naive SOH threshold, an 8.41% uplift on the held-out slice.

This is the operational story: uncertainty is not just an error bar. It changes what the operator
does next.

## 3:30-4:30 What's Next

The first data acquisition priority is Sandia because it has multiple chemistries from one lab and
one protocol family. After that, I would replace the literature economics with Bridge Green's
internal yields and pack-level costs, then containerize the API and dashboard for deployment.
