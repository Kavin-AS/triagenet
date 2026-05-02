"""Loader for Sandia/Battery Archive SNL exported CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from triagenet.config import DATA_RAW, DEFAULT_NOMINAL_CAPACITY_AH
from triagenet.io.loader_utils import (
    clipped_soh,
    compute_dq_dv,
    downsample,
    finalize_frame,
    first_existing,
    infer_chemistry,
    normalize_columns,
    warn_skip,
)
from triagenet.io.unified_schema import validate_unified_cycle


def load(raw_dir: Path | None = None) -> pd.DataFrame:
    """Parse Sandia CSV exports into the UnifiedCycle schema."""
    root = raw_dir or DATA_RAW / "sandia"
    rows: list[dict[str, object]] = []
    cycle_files = sorted(root.rglob("*cycle_data.csv"))
    standalone_files = sorted(path for path in root.rglob("*.csv") if path not in cycle_files)

    for cycle_path in cycle_files:
        cell_id = cycle_path.name.removesuffix("_cycle_data.csv")
        timeseries_path = cycle_path.with_name(f"{cell_id}_timeseries.csv")
        if not timeseries_path.exists():
            raise FileNotFoundError(f"Missing Sandia timeseries file for {cycle_path}")
        rows.extend(_parse_battery_archive_pair(cycle_path, timeseries_path, cell_id))

    for csv_path in standalone_files:
        if csv_path.name.endswith("_timeseries.csv"):
            continue
        rows.extend(_parse_standalone_csv(csv_path))

    df = finalize_frame(rows)
    if not df.empty:
        validate_unified_cycle(df)
    return df


def _parse_battery_archive_pair(
    cycle_path: Path, timeseries_path: Path, source_cell_id: str
) -> list[dict[str, object]]:
    cycle_df = normalize_columns(pd.read_csv(cycle_path))
    ts_df = normalize_columns(pd.read_csv(timeseries_path))
    cycle_col = first_existing(cycle_df, ["cycle_index", "cycle", "cycle_number"])
    ts_cycle_col = first_existing(ts_df, ["cycle_index", "cycle", "cycle_number"])
    rows = []
    for _, cycle in cycle_df.sort_values(cycle_col).iterrows():
        cycle_index = int(cycle[cycle_col])
        trace = ts_df[ts_df[ts_cycle_col] == cycle_index]
        rows.extend(_build_rows_from_trace(cycle_path, source_cell_id, cycle_index, cycle, trace))
    return rows


def _parse_standalone_csv(path: Path) -> list[dict[str, object]]:
    df = normalize_columns(pd.read_csv(path))
    cell_col = first_existing(df, ["cell_id", "cell"])
    cycle_col = first_existing(df, ["cycle_index", "cycle", "cycle_number"])
    rows = []
    for (cell_id, cycle_index), trace in df.groupby([cell_col, cycle_col], sort=True):
        summary = trace.iloc[-1]
        rows.extend(_build_rows_from_trace(path, str(cell_id), int(cycle_index), summary, trace))
    return rows


def _build_rows_from_trace(
    path: Path,
    source_cell_id: str,
    cycle_index: int,
    summary: pd.Series,
    trace: pd.DataFrame,
) -> list[dict[str, object]]:
    if trace.empty:
        warn_skip(path, cycle_index, "no timeseries rows")
        return []
    try:
        voltage_col = first_existing(trace, ["voltage_v", "voltage", "v"])
        current_col = first_existing(trace, ["current_a", "current", "i"])
        time_col = first_existing(trace, ["time_s", "test_time_s", "time", "time_sec"])
        discharge_col = _optional_column(summary, ["discharge_capacity_ah", "discharge_capacity"])
        charge_col = _optional_column(summary, ["charge_capacity_ah", "charge_capacity"])
        capacity_col = _optional_column(trace.iloc[-1], ["capacity_ah", "discharge_capacity_ah"])
    except KeyError as exc:
        warn_skip(path, cycle_index, str(exc))
        return []

    voltage = trace[voltage_col].astype(float).to_numpy()
    current = trace[current_col].astype(float).to_numpy()
    time_s = trace[time_col].astype(float).to_numpy()
    discharge_capacity = float(summary[discharge_col]) if discharge_col else 0.0
    if discharge_capacity <= 0 and capacity_col:
        discharge_capacity = float(abs(trace[capacity_col].astype(float).max()))
    charge_capacity = float(summary[charge_col]) if charge_col else discharge_capacity / 0.995
    if (
        discharge_capacity <= 0
        or charge_capacity <= 0
        or np.isnan(voltage).any()
        or np.isnan(current).any()
    ):
        warn_skip(path, cycle_index, "non-positive capacity or NaN curve")
        return []

    chemistry = infer_chemistry(f"{source_cell_id} {path}")
    nominal = _nominal_capacity(chemistry, summary)
    cell_id = f"sandia_{source_cell_id}".lower()
    return [
        {
            "cell_id": cell_id,
            "dataset": "sandia",
            "chemistry": chemistry,
            "manufacturer": str(summary.get("manufacturer", "unknown") or "unknown"),
            "nominal_capacity_ah": nominal,
            "cycle_index": cycle_index,
            "is_eol_cycle": False,
            "discharge_capacity_ah": discharge_capacity,
            "charge_capacity_ah": charge_capacity,
            "coulombic_efficiency": discharge_capacity / charge_capacity,
            "energy_efficiency": _optional_float(summary, ["energy_efficiency"]),
            "soh": clipped_soh(discharge_capacity, nominal),
            "temperature_c_mean": _optional_float(summary, ["temperature_c_mean", "temperature_c"]),
            "c_rate_charge": _optional_float(summary, ["c_rate_charge"]),
            "c_rate_discharge": _optional_float(summary, ["c_rate_discharge"]),
            "voltage_curve": downsample(voltage),
            "current_curve": downsample(current),
            "time_curve_s": downsample(time_s),
            "dq_dv_curve": compute_dq_dv(voltage, np.linspace(0, discharge_capacity, voltage.size)),
        }
    ]


def _nominal_capacity(chemistry: str, row: pd.Series) -> float:
    value = _optional_float(row, ["nominal_capacity_ah", "rated_capacity_ah"])
    if value:
        return value
    return DEFAULT_NOMINAL_CAPACITY_AH[chemistry]["18650"]


def _optional_column(row: pd.Series, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in row.index:
            return candidate
    return None


def _optional_float(row: pd.Series, candidates: list[str]) -> float | None:
    column = _optional_column(row, candidates)
    if column is None or pd.isna(row[column]):
        return None
    return float(row[column])
