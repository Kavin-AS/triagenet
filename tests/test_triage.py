"""Tests for the Phase 4 triage engine."""

from __future__ import annotations

import pandas as pd
from triagenet.config import DATA_PROCESSED
from triagenet.models.triage import TriageEngine
from triagenet.pipeline import TriagePipeline


def test_high_confidence_lfp_routes_second_life() -> None:
    engine = TriageEngine()
    args = ({"LFP": 1.0}, 0.95, 0.92, 0.98, 0.02, 0.0036)
    assert engine.decide(*args, rule="naive")["decision"] == "second_life"
    assert engine.decide(*args, rule="conservative")["decision"] == "second_life"
    assert engine.decide(*args, rule="risk_aware")["decision"] == "second_life"


def test_high_confidence_lco_routes_recycle() -> None:
    engine = TriageEngine()
    args = ({"LCO": 1.0}, 0.10, 0.07, 0.13, 0.02, 0.0040)
    assert engine.decide(*args, rule="naive")["decision"] == "direct_recycle"
    assert engine.decide(*args, rule="conservative")["decision"] == "direct_recycle"
    assert engine.decide(*args, rule="risk_aware")["decision"] == "direct_recycle"


def test_uncertain_boundary_requests_more_characterization() -> None:
    engine = TriageEngine()
    verdict = engine.decide({"LFP": 0.5, "LCO": 0.5}, 0.80, 0.45, 1.05, 0.20, 0.0040)
    assert verdict["decision"] == "needs_more_characterization"
    assert verdict["value_of_info_one_more_cycle_usd"] >= 0


def test_pipeline_predict_real_cycle_has_required_fields() -> None:
    row = pd.read_parquet(DATA_PROCESSED / "cycles.parquet").iloc[0]
    verdict = TriagePipeline().predict(row)
    required = {
        "decision",
        "expected_value_usd",
        "second_life_value_usd",
        "recycle_value_usd",
        "value_of_info_one_more_cycle_usd",
        "rationale",
        "confidence",
    }
    assert required.issubset(verdict)
