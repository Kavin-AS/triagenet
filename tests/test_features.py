"""Tests for single-cycle feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from triagenet.config import DATA_PROCESSED
from triagenet.features.eol_cycle import (
    ABSOLUTE_VOLTAGE_FEATURES,
    ALL_FEATURES,
    CHEMISTRY_FEATURES,
    SHAPE_FEATURES,
    extract_features,
)
from triagenet.features.ic_curve import compute_ic_curve, find_dqdv_peaks


def _cycle(voltage: np.ndarray, chemistry: str = "LFP") -> dict[str, object]:
    current = -np.ones(100)
    time_s = np.linspace(0, 3600, 100)
    return {
        "cell_id": f"test_{chemistry.lower()}",
        "dataset": "synthetic",
        "chemistry": chemistry,
        "manufacturer": "synthetic",
        "nominal_capacity_ah": 1.1,
        "cycle_index": 1,
        "is_eol_cycle": True,
        "discharge_capacity_ah": 1.0,
        "charge_capacity_ah": 1.01,
        "coulombic_efficiency": 1.0 / 1.01,
        "energy_efficiency": None,
        "soh": 1.0 / 1.1,
        "temperature_c_mean": None,
        "c_rate_charge": None,
        "c_rate_discharge": 1.0,
        "voltage_curve": voltage.tolist(),
        "current_curve": current.tolist(),
        "time_curve_s": time_s.tolist(),
        "dq_dv_curve": None,
    }


def test_extract_features_distinguishes_flat_lfp_from_sloped_lco() -> None:
    """LFP-like flat plateau has lower plateau flatness than sloped LCO-like voltage."""
    x = np.linspace(0, 1, 100)
    lfp_voltage = 3.3 + 0.03 * np.tanh((0.5 - x) * 16)
    lco_voltage = 4.2 - 1.2 * x
    lfp = extract_features(_cycle(lfp_voltage, "LFP"))
    lco = extract_features(_cycle(lco_voltage, "LCO"))
    assert set(ALL_FEATURES).issubset(lfp)
    assert "feat_capacity_ratio" not in CHEMISTRY_FEATURES
    assert "feat_energy_efficiency" not in CHEMISTRY_FEATURES
    assert CHEMISTRY_FEATURES == SHAPE_FEATURES
    assert "feat_voltage_mean" in ABSOLUTE_VOLTAGE_FEATURES
    assert "feat_voltage_mean" not in SHAPE_FEATURES
    assert lfp["feat_plateau_flatness"] < lco["feat_plateau_flatness"]
    assert not any(np.isnan(value) for value in lfp.values())


def test_shape_features_are_voltage_shift_invariant() -> None:
    """Normalized shape features must not move when a voltage offset is added."""
    x = np.linspace(0, 1, 100)
    voltage = 3.35 - 0.08 * x - 0.65 / (1 + np.exp(-(x - 0.82) * 35))
    base = extract_features(_cycle(voltage, "LFP"))
    shifted = extract_features(_cycle(voltage + 0.5, "LFP"))
    for name in SHAPE_FEATURES:
        if name.startswith("feat_shape_"):
            assert np.isclose(base[name], shifted[name], rtol=1e-2, atol=1e-6), name


def test_shape_features_are_voltage_scale_invariant() -> None:
    """Active normalized shape features must be stable under multiplicative voltage scaling."""
    x = np.linspace(0, 1, 100)
    voltage = 4.2 - 1.1 * x + 0.05 * np.sin(2 * np.pi * x)
    base = extract_features(_cycle(voltage, "LCO"))
    scaled = extract_features(_cycle(voltage * 1.1, "LCO"))
    assert "feat_shape_voltage_range" not in SHAPE_FEATURES
    for name in SHAPE_FEATURES:
        if name.startswith("feat_shape_"):
            assert np.isclose(base[name], scaled[name], rtol=1e-2, atol=1e-6), name


def test_ic_curve_peak_finder_identifies_lfp_like_peak_voltage() -> None:
    """A synthetic LFP-like plateau produces a dominant IC peak near 3.3 V."""
    time_s = np.linspace(0, 3600, 100)
    current = -np.ones(100)
    x = np.linspace(-1, 1, 100)
    voltage = 3.3 - 0.5 * np.power(x, 3)
    grid, curve = compute_ic_curve(voltage, current, time_s)
    peaks = find_dqdv_peaks(curve, grid)
    assert peaks
    assert abs(peaks[0][0] - 3.3) < 0.15


def test_featurize_real_processed_slice_has_no_nans() -> None:
    """A real cycles.parquet slice produces finite Phase 2 features."""
    df = pd.read_parquet(DATA_PROCESSED / "cycles.parquet").head(20)
    rows = [extract_features(row) for _, row in df.iterrows()]
    features = pd.DataFrame(rows)
    assert list(features.columns) == list(ALL_FEATURES)
    assert np.isfinite(features.to_numpy()).all()
