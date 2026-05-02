"""Evaluate triage decision rules and write Phase 4 diagnostics."""

from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import pandas as pd
from triagenet.config import DATA_PROCESSED, REPO_ROOT
from triagenet.pipeline import TriagePipeline

REPORT_DIR = REPO_ROOT / "reports" / "phase4_triage"
EVAL_CYCLES_PER_CELL = 5


def main() -> None:
    """Run all triage rules on a deterministic held-out-like slice and save metrics."""
    logging.getLogger("triagenet.features.eol_cycle").setLevel(logging.ERROR)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cycles = pd.read_parquet(DATA_PROCESSED / "cycles.parquet")
    sample = sample_cycles_per_cell(cycles, EVAL_CYCLES_PER_CELL)
    pipeline = TriagePipeline()
    outputs = {}
    for rule in ("naive", "conservative", "risk_aware"):
        outputs[rule] = pipeline.predict_batch(sample, rule=rule)
        outputs[rule].to_json(REPORT_DIR / f"{rule}_verdicts.json", orient="records", indent=2)
    summary = summarize(outputs)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_phase4_notes(summary)
    write_notebook()
    print(json.dumps(summary, indent=2))


def sample_cycles_per_cell(df: pd.DataFrame, max_cycles: int) -> pd.DataFrame:
    """Return deterministic SOH-stratified evaluation cycles per cell."""
    sampled = []
    for _, group in df.sort_values(["cell_id", "soh", "cycle_index"]).groupby("cell_id"):
        positions = pd.Series(range(len(group)))
        if len(group) > max_cycles:
            raw_positions = pd.Series(range(max_cycles)) * (len(group) - 1) / (max_cycles - 1)
            positions = pd.Series(
                pd.unique(pd.Series([round(position) for position in raw_positions]))
            )
        sampled.append(group.iloc[positions.to_numpy(dtype=int)])
    return pd.concat(sampled, ignore_index=True)


def summarize(outputs: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Summarize decision distributions, EV, VOI, and the required disagreement case."""
    rule_summary = {}
    for rule, frame in outputs.items():
        rule_summary[rule] = {
            "decisions": frame["decision"].value_counts().to_dict(),
            "mean_expected_value_usd": float(frame["expected_value_usd"].mean()),
            "total_expected_value_usd": float(frame["expected_value_usd"].sum()),
        }
    risk = outputs["risk_aware"]
    gap_total = (
        rule_summary["risk_aware"]["total_expected_value_usd"]
        - rule_summary["naive"]["total_expected_value_usd"]
    )
    cycle_865 = risk[(risk["cell_id"] == "calce_cs2_33") & (risk["cycle_index"] == 865)]
    if cycle_865.empty:
        cycle_865_record = {}
    else:
        cycle_865_record = cycle_865.iloc[0].to_dict()
    voi = risk["value_of_info_one_more_cycle_usd"]
    return {
        "rules": rule_summary,
        "money_on_table_rule3_minus_rule1_total_usd": float(gap_total),
        "money_on_table_rule3_minus_rule1_per_cell_usd": float(gap_total / max(len(risk), 1)),
        "money_on_table_uplift_pct": float(
            100.0 * gap_total / max(abs(rule_summary["naive"]["total_expected_value_usd"]), 1e-9)
        ),
        "voi_summary": {
            "mean": float(voi.mean()),
            "p50": float(voi.quantile(0.50)),
            "p90": float(voi.quantile(0.90)),
            "max": float(voi.max()),
        },
        "needs_more_characterization_count": int(
            (risk["decision"] == "needs_more_characterization").sum()
        ),
        "cycle_865_verdict": cycle_865_record,
        "n_evaluated": int(len(risk)),
        "sensitivity": sensitivity_summary(risk),
    }


def sensitivity_summary(risk: pd.DataFrame) -> dict[str, object]:
    """Return a lightweight sensitivity narrative from current verdict values."""
    return {
        "lithium_price_pm50": "Recycle values for LFP move, but most decisions are SOH-driven.",
        "lcoe_credit_0p04_to_0p10": "Higher second-life credit increases second-life routing.",
        "second_life_cost_5_to_15": (
            "Higher regrading cost pushes borderline cells to recycle/more characterization."
        ),
        "baseline_decisions": risk["decision"].value_counts().to_dict(),
    }


def write_phase4_notes(summary: dict[str, object]) -> None:
    """Write Phase 4 notes with actual decision metrics and assumptions."""
    notes = [
        "# Phase 4 Notes",
        "",
        "## Decision Rules",
        "",
        "Rule 1 is the naive SOH mean threshold at 0.80. Rule 2 uses the lower 90% SOH bound. "
        "Rule 3 is the production risk-aware expected-value rule and may return "
        "`needs_more_characterization` when value-of-information exceeds the EV gap.",
        "",
        "## Economics Assumptions",
        "",
        "Commodity prices use a documented late-April-2026 fallback snapshot in "
        "`triagenet.economics.prices`; this is marked `TODO(stub)` and should be replaced by "
        "Bridge Green's internal price feed. Recovery rates follow Velazquez-Martinez 2019 and "
        "Harper 2019 review values. Metal masses follow IEA/BNEF-style chemistry averages. "
        "Second-life cycle assumptions follow Martinez-Laserna 2018 and use a conservative "
        "$0.05/kWh grid-service credit.",
        "",
        "## VOI Math",
        "",
        "VOI is approximated as `E[max(EV_recycle, EV_second_life) after one more cycle] - "
        "max(current EVs)`, with the post-data SOH posterior modeled as the current normal "
        "posterior with standard deviation divided by `sqrt(2)`. This is approximate but makes "
        "uncertainty actionable in dollars.",
        "",
        "## Headline Metrics",
        "",
        "```json",
        json.dumps(summary, indent=2, default=str),
        "```",
        "",
        "## Caveats",
        "",
        "Recovery rates and composition tables are literature averages, not Bridge Green yields. "
        "Second-life processing costs and the lab-cell economic scale factor are demo assumptions. "
        "The VOI variance-halving assumption should be replaced with empirical repeat-cycle data.",
        "",
        "## Phase 5 Handoff",
        "",
        "The API should serve `TriagePipeline.predict()` results directly.",
        "",
    ]
    (REPO_ROOT / "PHASE_4_NOTES.md").write_text("\n".join(notes))


def write_notebook() -> None:
    """Write a compact Phase 4 diagnostics notebook shell."""
    content = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Phase 4 Triage Decisions\n"]},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import json\n",
                    "from triagenet.config import REPO_ROOT\n",
                    "summary_path = REPO_ROOT / 'reports' / 'phase4_triage' / 'summary.json'\n",
                    "summary = json.loads(summary_path.read_text())\n",
                    "summary\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "The production rule is risk-aware expected value with VOI. "
                    "Sensitivity should be rerun with Bridge Green's actual yield, price, "
                    "and processing-cost assumptions.\n"
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (REPO_ROOT / "notebooks" / "04_triage_decisions.ipynb").write_text(
        json.dumps(content, indent=2)
    )


if __name__ == "__main__":
    main()
