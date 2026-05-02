"""Single-cycle feature extraction for TriageNet EOL-style chemistry classification."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, savgol_filter
from triagenet.features.ic_curve import compute_ic_curve, find_dqdv_peaks
from triagenet.features.voltage_relax import extract_relaxation_features

LOGGER = logging.getLogger(__name__)

ALL_FEATURES = (
    "feat_capacity_ratio",
    "feat_coulombic_efficiency",
    "feat_energy_efficiency",
    "feat_charge_to_discharge_ratio",
    "feat_voltage_mean",
    "feat_voltage_std",
    "feat_voltage_at_50pct_capacity",
    "feat_voltage_at_80pct_capacity",
    "feat_voltage_at_20pct_capacity",
    "feat_plateau_flatness",
    "feat_initial_voltage_drop",
    "feat_final_voltage",
    "feat_dqdv_peak_voltage_1",
    "feat_dqdv_peak_magnitude_1",
    "feat_dqdv_peak_voltage_2",
    "feat_dqdv_peak_magnitude_2",
    "feat_dqdv_n_significant_peaks",
    "feat_c_rate_charge",
    "feat_c_rate_discharge",
    "feat_temperature_mean",
    "feat_voltage_range",
    "feat_shape_v_norm_at_25pct_capacity",
    "feat_shape_v_norm_at_50pct_capacity",
    "feat_shape_v_norm_at_75pct_capacity",
    "feat_shape_voltage_range",
    "feat_shape_plateau_flatness_normalized",
    "feat_shape_dqdv_peak_v_norm_1",
    "feat_shape_dqdv_peak_v_norm_2",
    "feat_shape_dqdv_peak_separation_norm",
    "feat_shape_curve_skewness",
    "feat_shape_curve_kurtosis",
    "feat_shape_curve_inflection_position",
    "feat_shape_avg_slope_first_half",
    "feat_dqdv_peak_width",
    "feat_dqdv_peak_area",
    "feat_voltage_curve_smoothness",
    "feat_internal_resistance_proxy",
    "feat_low_voltage_tail_fraction",
)

# Original Phase 2 chemistry features, kept as the voltage-axis leakage baseline. It excludes
# feat_capacity_ratio for SOH leakage and feat_energy_efficiency because MIT lacks the field.
ABSOLUTE_VOLTAGE_FEATURES = tuple(
    feature
    for feature in ALL_FEATURES[:21]
    if feature not in {"feat_capacity_ratio", "feat_energy_efficiency"}
)
# Active Phase 2.5 production chemistry features. feat_shape_voltage_range is computed for
# audit visibility, but excluded here because it is not invariant under voltage scaling.
SHAPE_FEATURES = (
    "feat_shape_v_norm_at_25pct_capacity",
    "feat_shape_v_norm_at_50pct_capacity",
    "feat_shape_v_norm_at_75pct_capacity",
    "feat_shape_plateau_flatness_normalized",
    "feat_shape_dqdv_peak_v_norm_1",
    "feat_shape_dqdv_peak_v_norm_2",
    "feat_shape_dqdv_peak_separation_norm",
    "feat_shape_curve_skewness",
    "feat_shape_curve_kurtosis",
    "feat_shape_curve_inflection_position",
    "feat_shape_avg_slope_first_half",
    "feat_coulombic_efficiency",
    "feat_charge_to_discharge_ratio",
    "feat_dqdv_peak_magnitude_1",
    "feat_dqdv_peak_magnitude_2",
    "feat_dqdv_n_significant_peaks",
    "feat_c_rate_charge",
    "feat_c_rate_discharge",
)
CHEMISTRY_FEATURES = SHAPE_FEATURES
# Capacity features excluded: they ARE the target. Including them defeats the purpose of
# shape-based SOH inference.
SOH_DEGRADATION_FEATURES = (
    "feat_dqdv_peak_width",
    "feat_dqdv_peak_area",
    "feat_voltage_curve_smoothness",
    "feat_internal_resistance_proxy",
    "feat_low_voltage_tail_fraction",
)
SOH_FEATURES = tuple(SHAPE_FEATURES) + SOH_DEGRADATION_FEATURES
LOW_VOLTAGE_TAIL_THRESHOLDS = {"LFP": 3.0, "LCO": 3.5, "NMC": 3.4, "NCA": 3.4}

IMPUTATION_DEFAULTS = {
    # Missing energy efficiency is neutral-filled near an ideal round trip, instead of zero,
    # because MIT lacks this field and a zero default would encode dataset provenance.
    "feat_energy_efficiency": 1.0,
    "feat_c_rate_charge": 0.0,
    "feat_c_rate_discharge": 0.0,
    "feat_temperature_mean": 25.0,
    "feat_dqdv_peak_voltage_1": 0.0,
    "feat_dqdv_peak_magnitude_1": 0.0,
    "feat_dqdv_peak_voltage_2": 0.0,
    "feat_dqdv_peak_magnitude_2": 0.0,
    "feat_dqdv_n_significant_peaks": 0.0,
    "feat_shape_dqdv_peak_v_norm_1": 0.0,
    "feat_shape_dqdv_peak_v_norm_2": 0.0,
    "feat_shape_dqdv_peak_separation_norm": 0.0,
    "feat_dqdv_peak_width": 0.0,
    "feat_dqdv_peak_area": 0.0,
}


def extract_features(cycle: Mapping[str, object] | pd.Series) -> dict[str, float]:
    """Extract the stable 21-feature single-cycle vector from one UnifiedCycle row."""
    features = _extract_features(cycle)
    imputed = _imputed_feature_names(features)
    if imputed:
        row = cycle.to_dict() if isinstance(cycle, pd.Series) else dict(cycle)
        LOGGER.warning(
            "Imputed non-finite feature values for %s cycle %s: %s",
            row.get("cell_id", "unknown"),
            row.get("cycle_index", "unknown"),
            imputed,
        )
    return _impute_and_order(features)


def featurize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Map ``extract_features`` over a UnifiedCycle frame and keep row identifiers."""
    records = []
    imputation_counts: dict[str, int] = {}
    for _, row in df.iterrows():
        record = {
            "cell_id": row["cell_id"],
            "cycle_index": int(row["cycle_index"]),
            "dataset": row["dataset"],
            "chemistry": row["chemistry"],
            "soh": float(row["soh"]),
        }
        raw_features = _extract_features(row)
        for feature_name in _imputed_feature_names(raw_features):
            imputation_counts[feature_name] = imputation_counts.get(feature_name, 0) + 1
        record.update(_impute_and_order(raw_features))
        records.append(record)
    if imputation_counts:
        LOGGER.warning("Imputed feature values during featurization: %s", imputation_counts)
    return pd.DataFrame.from_records(records)


def _extract_features(cycle: Mapping[str, object] | pd.Series) -> dict[str, float]:
    row = cycle.to_dict() if isinstance(cycle, pd.Series) else dict(cycle)
    voltage = _array(row["voltage_curve"])
    current = _array(row["current_curve"])
    time_s = _array(row["time_curve_s"])
    discharge_capacity = _finite_float(row.get("discharge_capacity_ah"), 0.0)
    charge_capacity = _finite_float(row.get("charge_capacity_ah"), discharge_capacity)
    nominal_capacity = max(_finite_float(row.get("nominal_capacity_ah"), 1.0), 1e-9)

    # Capacity, efficiency, and curve-shape features mirror battery ML diagnostics used by
    # Severson 2019 for early prediction and Roman 2021-style voltage/IC shape descriptors.
    features = {
        "feat_capacity_ratio": discharge_capacity / nominal_capacity,
        "feat_coulombic_efficiency": _finite_float(row.get("coulombic_efficiency"), 0.0),
        "feat_energy_efficiency": _finite_float(row.get("energy_efficiency"), np.nan),
        "feat_charge_to_discharge_ratio": charge_capacity / max(discharge_capacity, 1e-9),
        "feat_voltage_mean": float(np.mean(voltage)),
        "feat_voltage_std": float(np.std(voltage)),
        "feat_voltage_at_50pct_capacity": float(voltage[50]),
        "feat_voltage_at_80pct_capacity": float(voltage[80]),
        "feat_voltage_at_20pct_capacity": float(voltage[20]),
        # LFP's olivine voltage plateau is much flatter near 3.2-3.3 V than layered oxides
        # such as LCO/NMC/NCA; this follows the chemistry contrast summarized by Schmuch 2018.
        "feat_plateau_flatness": float(np.std(voltage[20:80])),
        "feat_initial_voltage_drop": float(voltage[0] - voltage[min(5, voltage.size - 1)]),
        "feat_final_voltage": float(voltage[-1]),
        "feat_c_rate_charge": _finite_float(row.get("c_rate_charge"), np.nan),
        "feat_c_rate_discharge": _finite_float(row.get("c_rate_discharge"), np.nan),
        "feat_temperature_mean": _finite_float(row.get("temperature_c_mean"), np.nan),
        "feat_voltage_range": float(np.max(voltage) - np.min(voltage)),
    }
    features.update(_dqdv_features(row, voltage, current, time_s))
    features.update(_shape_features(voltage, current, time_s))
    features.update(_soh_degradation_features(row, voltage, current, time_s, features))
    features.update(extract_relaxation_features(voltage, current, time_s))
    return features


def _shape_features(
    voltage: np.ndarray, current: np.ndarray, time_s: np.ndarray
) -> dict[str, float]:
    # Savitzky-Golay smoothing follows the same numerical-denoising choice as dQ/dV extraction.
    smoothed_voltage = savgol_filter(voltage, window_length=11, polyorder=2, mode="interp")
    v_min = float(np.min(smoothed_voltage))
    v_max = float(np.max(smoothed_voltage))
    v_range = max(v_max - v_min, 1e-9)
    v_norm = (smoothed_voltage - v_min) / v_range
    t_norm = _normalized_axis(time_s)
    normalized_peaks = _normalized_ic_peaks(v_norm, current, time_s)
    peak_1_norm = normalized_peaks[0][0] if normalized_peaks else np.nan
    peak_2_norm = normalized_peaks[1][0] if len(normalized_peaks) > 1 else np.nan
    if np.isfinite(peak_1_norm) and np.isfinite(peak_2_norm):
        peak_separation = abs(peak_2_norm - peak_1_norm)
    else:
        peak_separation = np.nan

    centered = v_norm - float(np.mean(v_norm))
    std = max(float(np.std(v_norm)), 1e-12)
    first_half = max(v_norm.size // 2, 2)
    slope = np.gradient(v_norm, t_norm)
    curvature = np.gradient(slope, t_norm)

    return {
        "feat_shape_v_norm_at_25pct_capacity": float(v_norm[25]),
        "feat_shape_v_norm_at_50pct_capacity": float(v_norm[50]),
        "feat_shape_v_norm_at_75pct_capacity": float(v_norm[75]),
        "feat_shape_voltage_range": float(v_range),
        "feat_shape_plateau_flatness_normalized": float(np.std(v_norm[20:80])),
        "feat_shape_dqdv_peak_v_norm_1": float(peak_1_norm),
        "feat_shape_dqdv_peak_v_norm_2": float(peak_2_norm),
        "feat_shape_dqdv_peak_separation_norm": float(peak_separation),
        "feat_shape_curve_skewness": float(np.mean((centered / std) ** 3)),
        "feat_shape_curve_kurtosis": float(np.mean((centered / std) ** 4)),
        "feat_shape_curve_inflection_position": float(t_norm[int(np.argmax(np.abs(curvature)))]),
        "feat_shape_avg_slope_first_half": float(np.mean(slope[:first_half])),
    }


def _dqdv_features(
    row: dict[str, object], voltage: np.ndarray, current: np.ndarray, time_s: np.ndarray
) -> dict[str, float]:
    dq_dv = row.get("dq_dv_curve")
    if isinstance(dq_dv, list) and len(dq_dv) == voltage.size:
        ic_curve = np.abs(np.asarray(dq_dv, dtype=float))
        voltage_grid = voltage
    else:
        voltage_grid, ic_curve = compute_ic_curve(voltage, current, time_s)
    peaks = find_dqdv_peaks(ic_curve, voltage_grid)
    significant = 0
    if peaks:
        threshold = 0.30 * peaks[0][1]
        significant = sum(1 for _, magnitude in peaks if magnitude >= threshold)
    peak_1 = peaks[0] if peaks else (np.nan, np.nan)
    peak_2 = peaks[1] if len(peaks) > 1 else (np.nan, np.nan)
    return {
        "feat_dqdv_peak_voltage_1": peak_1[0],
        "feat_dqdv_peak_magnitude_1": peak_1[1],
        "feat_dqdv_peak_voltage_2": peak_2[0],
        "feat_dqdv_peak_magnitude_2": peak_2[1],
        "feat_dqdv_n_significant_peaks": float(significant),
    }


def _soh_degradation_features(
    row: dict[str, object],
    voltage: np.ndarray,
    current: np.ndarray,
    time_s: np.ndarray,
    features: dict[str, float],
) -> dict[str, float]:
    voltage_grid, ic_curve = compute_ic_curve(voltage, current, time_s)
    peak_width, peak_area = _largest_peak_width_area(voltage_grid, ic_curve)
    smoothed_voltage = savgol_filter(voltage, window_length=11, polyorder=2, mode="interp")
    v_norm = (smoothed_voltage - float(np.min(smoothed_voltage))) / max(
        float(np.max(smoothed_voltage) - np.min(smoothed_voltage)), 1e-9
    )
    t_norm = _normalized_axis(time_s)
    second_derivative = np.gradient(np.gradient(v_norm, t_norm), t_norm)
    c_rate = abs(_finite_float(row.get("c_rate_discharge"), 0.0))
    threshold = LOW_VOLTAGE_TAIL_THRESHOLDS.get(str(row.get("chemistry", "")).upper(), 3.0)
    return {
        "feat_dqdv_peak_width": peak_width,
        "feat_dqdv_peak_area": peak_area,
        "feat_voltage_curve_smoothness": float(np.sum(np.abs(second_derivative))),
        "feat_internal_resistance_proxy": float(
            abs(features["feat_initial_voltage_drop"]) / max(c_rate, 1e-6)
        ),
        "feat_low_voltage_tail_fraction": _low_voltage_tail_fraction(voltage, threshold),
    }


def _imputed_feature_names(features: dict[str, float]) -> list[str]:
    imputed = []
    for name in ALL_FEATURES:
        raw_value = features.get(name)
        if name in features and raw_value is not None:
            try:
                was_finite = np.isfinite(float(raw_value))
            except (TypeError, ValueError):
                was_finite = False
            if not was_finite:
                imputed.append(name)
        elif name in features:
            imputed.append(name)
    return imputed


def _impute_and_order(features: dict[str, float]) -> dict[str, float]:
    ordered = {}
    for name in ALL_FEATURES:
        default = IMPUTATION_DEFAULTS.get(name, 0.0)
        value = _finite_float(features.get(name), default)
        ordered[name] = float(value)
    return ordered


def _array(value: object) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.size != 100:
        raise ValueError("UnifiedCycle curve features must be 100 points")
    if not np.isfinite(values).all():
        raise ValueError("UnifiedCycle curve contains non-finite values")
    return values


def _normalized_axis(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - float(np.min(values))
    span = max(float(np.max(shifted)), 1e-9)
    return shifted / span


def _normalized_ic_peaks(
    v_norm: np.ndarray, current: np.ndarray, time_s: np.ndarray
) -> list[tuple[float, float]]:
    dt = np.gradient(time_s)
    charge_ah = np.cumsum(np.abs(current) * np.maximum(dt, 0.0)) / 3600.0
    order = np.argsort(v_norm)
    v_sorted = v_norm[order]
    q_sorted = charge_ah[order]
    unique_v, unique_idx = np.unique(v_sorted, return_index=True)
    if unique_v.size < 11:
        return []
    q_unique = q_sorted[unique_idx]
    grid = np.linspace(0.0, 1.0, 100)
    q_grid = np.interp(grid, unique_v, q_unique)
    ic_curve = np.abs(np.gradient(q_grid, grid))
    ic_curve = savgol_filter(ic_curve, window_length=11, polyorder=2, mode="interp")
    max_value = float(np.max(ic_curve))
    if max_value <= 0 or not np.isfinite(max_value):
        return []
    peak_indices, _ = find_peaks(ic_curve, prominence=0.05 * max_value)
    if peak_indices.size == 0:
        peak_indices = np.array([int(np.argmax(ic_curve))])
    ranked = sorted(
        ((float(grid[index]), float(ic_curve[index])) for index in peak_indices),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked


def _largest_peak_width_area(voltage_grid: np.ndarray, ic_curve: np.ndarray) -> tuple[float, float]:
    finite_curve = np.nan_to_num(np.abs(ic_curve), nan=0.0, posinf=0.0, neginf=0.0)
    if finite_curve.size < 3 or float(np.max(finite_curve)) <= 0:
        return 0.0, 0.0
    peak_indices, _ = find_peaks(finite_curve, prominence=0.05 * float(np.max(finite_curve)))
    if peak_indices.size == 0:
        peak_indices = np.array([int(np.argmax(finite_curve))])
    peak_index = int(peak_indices[np.argmax(finite_curve[peak_indices])])
    widths = peak_widths(finite_curve, np.array([peak_index]), rel_height=0.5)
    left_ip = float(widths[2][0])
    right_ip = float(widths[3][0])
    voltage_step = float(np.mean(np.diff(voltage_grid))) if voltage_grid.size > 1 else 0.0
    width_v = max(0.0, (right_ip - left_ip) * abs(voltage_step))
    left_index = max(0, int(np.floor(left_ip)))
    right_index = min(finite_curve.size - 1, int(np.ceil(right_ip)))
    if right_index <= left_index:
        area = 0.0
    else:
        area = float(
            np.trapz(
                finite_curve[left_index : right_index + 1],
                voltage_grid[left_index : right_index + 1],
            )
        )
    return float(width_v), abs(area)


def _low_voltage_tail_fraction(voltage: np.ndarray, threshold_v: float) -> float:
    return float(np.mean(np.asarray(voltage, dtype=float) < threshold_v))


def _finite_float(value: object, default: float) -> float:
    if value is None:
        return float(default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(numeric):
        return float(default)
    return numeric
