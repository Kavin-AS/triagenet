"""Incremental-capacity curve utilities for single-cycle battery features."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, savgol_filter

VOLTAGE_GRID = np.linspace(2.0, 4.4, 100)
SG_WINDOW = 11
SG_POLYORDER = 2
PEAK_PROMINENCE_FRACTION = 0.05


def compute_ic_curve(
    voltage: np.ndarray, current: np.ndarray, time_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute dQ/dV on a fixed voltage grid from one discharge cycle.

    The fixed 2.0-4.4 V grid covers the common Li-ion operating range across LFP, LCO,
    NMC, and NCA cells. Savitzky-Golay smoothing uses window=11/polyorder=2, a conservative
    local polynomial smoother used broadly for incremental-capacity analysis in battery
    diagnostics, including Severson-style early-cycle feature extraction.
    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    valid = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(time_s)
    voltage = voltage[valid]
    current = current[valid]
    time_s = time_s[valid]
    if voltage.size < 5 or current.size != voltage.size or time_s.size != voltage.size:
        return VOLTAGE_GRID.copy(), np.zeros_like(VOLTAGE_GRID)

    dt_hours = np.diff(time_s, prepend=time_s[0]) / 3600.0
    if dt_hours.size > 1 and dt_hours[0] <= 0:
        dt_hours[0] = dt_hours[1]
    dq = np.abs(current) * np.maximum(dt_hours, 0.0)
    capacity = np.cumsum(dq)
    if float(np.nanmax(capacity) - np.nanmin(capacity)) <= 0:
        return VOLTAGE_GRID.copy(), np.zeros_like(VOLTAGE_GRID)

    order = np.argsort(voltage)
    sorted_voltage = voltage[order]
    sorted_capacity = capacity[order]
    unique_voltage, unique_indices = np.unique(sorted_voltage, return_index=True)
    unique_capacity = sorted_capacity[unique_indices]
    if unique_voltage.size < 5:
        return VOLTAGE_GRID.copy(), np.zeros_like(VOLTAGE_GRID)

    interpolated_capacity = np.interp(
        VOLTAGE_GRID,
        unique_voltage,
        unique_capacity,
        left=np.nan,
        right=np.nan,
    )
    finite = np.isfinite(interpolated_capacity)
    if finite.sum() < 5:
        return VOLTAGE_GRID.copy(), np.zeros_like(VOLTAGE_GRID)
    interpolated_capacity = np.interp(
        VOLTAGE_GRID,
        VOLTAGE_GRID[finite],
        interpolated_capacity[finite],
    )
    window = min(SG_WINDOW, interpolated_capacity.size - (1 - interpolated_capacity.size % 2))
    if window >= 5:
        if window % 2 == 0:
            window -= 1
        interpolated_capacity = savgol_filter(
            interpolated_capacity, window_length=window, polyorder=SG_POLYORDER
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        ic_curve = np.gradient(interpolated_capacity, VOLTAGE_GRID)
    ic_curve = np.nan_to_num(ic_curve, nan=0.0, posinf=0.0, neginf=0.0)
    return VOLTAGE_GRID.copy(), np.abs(ic_curve)


def find_dqdv_peaks(ic_curve: np.ndarray, voltage_grid: np.ndarray) -> list[tuple[float, float]]:
    """Return dQ/dV peaks ranked by magnitude as ``(voltage, magnitude)`` pairs."""
    curve = np.asarray(ic_curve, dtype=float)
    grid = np.asarray(voltage_grid, dtype=float)
    if curve.size == 0 or grid.size != curve.size or not np.isfinite(curve).any():
        return []
    curve = np.nan_to_num(np.abs(curve), nan=0.0, posinf=0.0, neginf=0.0)
    max_value = float(np.max(curve))
    if max_value <= 0:
        return []
    # A 5% prominence threshold follows common IC-curve practice: keep chemistry-scale peaks
    # while ignoring numerical ripple from smoothing and interpolation.
    peak_indices, properties = find_peaks(curve, prominence=max_value * PEAK_PROMINENCE_FRACTION)
    if peak_indices.size == 0:
        peak_indices = np.asarray([int(np.argmax(curve))])
    peaks = [(float(grid[index]), float(curve[index])) for index in peak_indices]
    return sorted(peaks, key=lambda item: item[1], reverse=True)
