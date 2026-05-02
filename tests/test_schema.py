"""Tests for the UnifiedCycle schema validator."""

from __future__ import annotations

import pandas as pd
import pytest
from triagenet.io.unified_schema import SchemaError, validate_unified_cycle


def good_frame() -> pd.DataFrame:
    """Return a small valid UnifiedCycle DataFrame for schema tests."""
    curve = [1.0] * 100
    return pd.DataFrame(
        {
            "cell_id": pd.Series(["cell_a", "cell_a"], dtype="string"),
            "dataset": pd.Series(["sandia", "sandia"], dtype="string"),
            "chemistry": pd.Series(["LFP", "LFP"], dtype="string"),
            "manufacturer": pd.Series(["A123", "A123"], dtype="string"),
            "nominal_capacity_ah": [1.1, 1.1],
            "cycle_index": pd.Series([1, 2], dtype="int64"),
            "is_eol_cycle": [False, True],
            "discharge_capacity_ah": [1.0, 0.9],
            "charge_capacity_ah": [1.01, 0.91],
            "coulombic_efficiency": [1.0 / 1.01, 0.9 / 0.91],
            "energy_efficiency": [None, None],
            "soh": [0.91, 0.82],
            "temperature_c_mean": [25.0, 25.0],
            "c_rate_charge": [0.5, 0.5],
            "c_rate_discharge": [1.0, 1.0],
            "voltage_curve": [curve, curve],
            "current_curve": [curve, curve],
            "time_curve_s": [curve, curve],
            "dq_dv_curve": [None, None],
        }
    )


def test_validate_unified_cycle_accepts_good_frame() -> None:
    """A complete valid frame passes without raising."""
    validate_unified_cycle(good_frame())


@pytest.mark.parametrize(
    "mutator",
    [
        lambda df: df.drop(columns=["cell_id"]),
        lambda df: df.assign(cycle_index=pd.Series([1.0, 2.0])),
        lambda df: df.assign(soh=[0.9, 1.2]),
        lambda df: df.assign(is_eol_cycle=[True, True]),
        lambda df: df.assign(voltage_curve=[[1.0], [1.0] * 100]),
    ],
)
def test_validate_unified_cycle_rejects_bad_frames(mutator) -> None:
    """Missing columns, wrong dtypes, invalid invariants, and bad curves fail."""
    with pytest.raises(SchemaError):
        validate_unified_cycle(mutator(good_frame()))
