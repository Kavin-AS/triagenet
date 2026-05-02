# Phase 4 Notes

## Decision Rules

Rule 1 is the naive SOH mean threshold at 0.80. Rule 2 uses the lower 90% SOH bound. Rule 3 is the production risk-aware expected-value rule and may return `needs_more_characterization` when value-of-information exceeds the EV gap.

## Economics Assumptions

Commodity prices use a documented late-April-2026 fallback snapshot in `triagenet.economics.prices`; this is marked `TODO(stub)` and should be replaced by Bridge Green's internal price feed. Recovery rates follow Velazquez-Martinez 2019 and Harper 2019 review values. Metal masses follow IEA/BNEF-style chemistry averages. Second-life cycle assumptions follow Martinez-Laserna 2018 and use a conservative $0.05/kWh grid-service credit.

## VOI Math

VOI is approximated as `E[max(EV_recycle, EV_second_life) after one more cycle] - max(current EVs)`, with the post-data SOH posterior modeled as the current normal posterior with standard deviation divided by `sqrt(2)`. This is approximate but makes uncertainty actionable in dollars.

## Headline Metrics

```json
{
  "rules": {
    "naive": {
      "decisions": {
        "second_life": 221,
        "direct_recycle": 44
      },
      "mean_expected_value_usd": 17.479988150490605,
      "total_expected_value_usd": 4632.19685988001
    },
    "conservative": {
      "decisions": {
        "second_life": 201,
        "direct_recycle": 64
      },
      "mean_expected_value_usd": 16.95643352678521,
      "total_expected_value_usd": 4493.454884598081
    },
    "risk_aware": {
      "decisions": {
        "second_life": 203,
        "direct_recycle": 57,
        "needs_more_characterization": 5
      },
      "mean_expected_value_usd": 18.94935942803496,
      "total_expected_value_usd": 5021.580248429264
    }
  },
  "money_on_table_rule3_minus_rule1_total_usd": 389.38338854925405,
  "money_on_table_rule3_minus_rule1_per_cell_usd": 1.4693712775443548,
  "money_on_table_uplift_pct": 8.406019871947764,
  "voi_summary": {
    "mean": 0.024099236695632336,
    "p50": 0.022186113367048677,
    "p90": 0.03054617720638575,
    "max": 0.3036404630950251
  },
  "needs_more_characterization_count": 5,
  "cycle_865_verdict": {
    "decision": "needs_more_characterization",
    "expected_value_usd": 22.16697919425219,
    "second_life_value_usd": {
      "mean": 22.16697919425219,
      "p10": 0.9980003442684743,
      "p90": 37.735000000000014
    },
    "recycle_value_usd": {
      "mean": -0.20461783147143062,
      "p10": -0.6014175276703914,
      "p90": -0.3994607241269645
    },
    "value_of_info_one_more_cycle_usd": 0.3036404630950251,
    "rationale": "Likely LFP (96%), SOH 0.94 [0.76, 1.05]. Second-life EV $22.17 vs recycle EV $-0.20. VOI for one more cycle $0.30.",
    "confidence": "low",
    "cell_id": "calce_cs2_33",
    "cycle_index": 865,
    "chemistry_probs": {
      "LCO": 0.0402496854464213,
      "LFP": 0.9597503145535787,
      "NMC": 0.0,
      "NCA": 0.0
    },
    "soh_mean": 0.9395984601639462,
    "soh_lower_90": 0.7555188557354773,
    "soh_upper_90": 1.05,
    "soh_std": 0.11191245312790363,
    "cell_kwh": 0.004070000000000001,
    "rule": "risk_aware"
  },
  "n_evaluated": 265,
  "sensitivity": {
    "lithium_price_pm50": "Recycle values for LFP move, but most decisions are SOH-driven.",
    "lcoe_credit_0p04_to_0p10": "Higher second-life credit increases second-life routing.",
    "second_life_cost_5_to_15": "Higher regrading cost pushes borderline cells to recycle/more characterization.",
    "baseline_decisions": {
      "second_life": 203,
      "direct_recycle": 57,
      "needs_more_characterization": 5
    }
  }
}
```

## Caveats

Recovery rates and composition tables are literature averages, not Bridge Green yields. Second-life processing costs and the lab-cell economic scale factor are demo assumptions. The VOI variance-halving assumption should be replaced with empirical repeat-cycle data.

## Phase 5 Handoff

The API should serve `TriagePipeline.predict()` results directly.
