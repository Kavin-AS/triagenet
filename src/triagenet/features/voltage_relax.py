"""Voltage relaxation feature extraction for optional later SOH diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def extract_relaxation_features(
    voltage: np.ndarray, current: np.ndarray, time_s: np.ndarray
) -> dict[str, float]:
    """Extract end-of-cycle near-zero-current voltage relaxation features when present."""
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    if voltage.size < 8 or current.size != voltage.size or time_s.size != voltage.size:
        return {}
    near_zero = np.abs(current) <= max(0.02, 0.02 * float(np.nanmax(np.abs(current))))
    if not near_zero[-1]:
        return {}
    start = len(near_zero) - 1
    while start > 0 and near_zero[start - 1]:
        start -= 1
    if len(near_zero) - start < 6:
        return {}
    relax_voltage = voltage[start:]
    relax_time = time_s[start:] - time_s[start]
    if np.nanmax(relax_time) <= 0:
        return {}
    try:
        params, _ = curve_fit(_exp_decay, relax_time, relax_voltage, maxfev=2000)
    except (RuntimeError, ValueError):
        return {}
    v_inf, amplitude, tau = params
    if tau <= 0 or not np.isfinite(tau):
        return {}
    return {
        "feat_relaxation_delta_v": float(relax_voltage[0] - relax_voltage[-1]),
        "feat_relaxation_tau_s": float(tau),
        "feat_relaxation_v_inf": float(v_inf),
        "feat_relaxation_amplitude": float(amplitude),
    }


def _exp_decay(time_s: np.ndarray, v_inf: float, amplitude: float, tau_s: float) -> np.ndarray:
    return v_inf + amplitude * np.exp(-time_s / tau_s)
