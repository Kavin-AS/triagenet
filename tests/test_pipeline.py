"""End-to-end tests for TriagePipeline."""

from __future__ import annotations

import pandas as pd
from triagenet.config import DATA_PROCESSED
from triagenet.pipeline import TriagePipeline


def test_pipeline_predicts_five_real_rows() -> None:
    """Pipeline returns complete verdicts on real processed cycles."""
    rows = pd.read_parquet(DATA_PROCESSED / "cycles.parquet").head(5)
    pipeline = TriagePipeline()
    verdicts = pipeline.predict_batch(rows)
    assert len(verdicts) == 5
    assert verdicts["decision"].notna().all()
    assert verdicts["expected_value_usd"].notna().all()


def test_pipeline_reproducible_for_same_input() -> None:
    """Seeded Monte Carlo economics makes repeated verdicts deterministic."""
    row = pd.read_parquet(DATA_PROCESSED / "cycles.parquet").iloc[0]
    pipeline = TriagePipeline(random_state=42)
    first = pipeline.predict(row)
    second = pipeline.predict(row)
    assert first["decision"] == second["decision"]
    assert first["expected_value_usd"] == second["expected_value_usd"]
