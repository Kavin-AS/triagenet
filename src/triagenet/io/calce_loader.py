"""Loader for CALCE CS2/CX2 battery files."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from triagenet.config import DATA_RAW
from triagenet.io.loader_utils import (
    clipped_soh,
    compute_dq_dv,
    downsample,
    finalize_frame,
    first_existing,
    normalize_columns,
    warn_skip,
)
from triagenet.io.unified_schema import validate_unified_cycle


def load(raw_dir: Path | None = None) -> pd.DataFrame:
    """Parse CALCE CSV, Excel, TXT, and ZIP archives into the UnifiedCycle schema."""
    root = raw_dir or DATA_RAW / "calce"
    rows: list[dict[str, object]] = []
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            rows.extend(_parse_frame(path, pd.read_csv(path)))
        elif suffix in {".xls", ".xlsx"}:
            rows.extend(_parse_excel(path))
        elif suffix == ".txt":
            rows.extend(_parse_frame(path, pd.read_csv(path, sep=None, engine="python")))
        elif suffix == ".zip":
            rows.extend(_parse_zip(path))
    rows = _reindex_real_calce_cycles(rows)
    df = finalize_frame(rows)
    if not df.empty:
        validate_unified_cycle(df)
    return df


def _parse_zip(path: Path) -> list[dict[str, object]]:
    rows = []
    with ZipFile(path) as archive:
        xlsx_names = sorted(name for name in archive.namelist() if name.lower().endswith(".xlsx"))
        for name in xlsx_names:
            with archive.open(name) as handle:
                # Real CALCE downloads are zip files containing dated Arbin .xlsx workbooks.
                workbook = pd.ExcelFile(BytesIO(handle.read()))
            rows.extend(_parse_excel_workbook(path, workbook, source_name=name))
    return rows


def _parse_excel(path: Path) -> list[dict[str, object]]:
    try:
        workbook = pd.ExcelFile(path)
    except ImportError as exc:
        raise ImportError(
            f"Reading CALCE Excel files requires pandas Excel optional engines: {path}"
        ) from exc
    return _parse_excel_workbook(path, workbook, source_name=path.name)


def _parse_excel_workbook(
    path: Path, workbook: pd.ExcelFile, source_name: str
) -> list[dict[str, object]]:
    rows = []
    data_sheets = [
        sheet for sheet in workbook.sheet_names if sheet.lower().startswith(("channel", "record"))
    ]
    for sheet_name in data_sheets:
        sheet = pd.read_excel(workbook, sheet_name=sheet_name)
        rows.extend(_parse_frame(path, sheet, sheet_name=sheet_name, source_name=source_name))
    return rows


def _parse_frame(
    path: Path,
    df: pd.DataFrame,
    sheet_name: str | None = None,
    source_name: str | None = None,
) -> list[dict[str, object]]:
    frame = normalize_columns(df).dropna(how="all")
    if frame.empty:
        return []
    cell_col = _maybe_column(frame, ["cell_id", "cell"]) or None
    cycle_col = first_existing(frame, ["cycle_index", "cycle", "cycle_number"])
    date_col = _maybe_column(frame, ["date_time"])
    if date_col:
        frame = frame.assign(_parsed_date_time=pd.to_datetime(frame[date_col], errors="coerce"))
        frame = frame.sort_values("_parsed_date_time", kind="mergesort")

    rows = []
    group_cols = [cycle_col] if cell_col is None else [cell_col, cycle_col]
    for key, trace in frame.groupby(group_cols, sort=False):
        cell_name = key[0] if cell_col is not None else _cell_from_path(path, source_name)
        cycle_index = key[-1] if isinstance(key, tuple) else key
        rows.extend(
            _build_row(path, str(cell_name), int(cycle_index), trace, sheet_name, source_name)
        )
    return rows


def _build_row(
    path: Path,
    source_cell_id: str,
    cycle_index: int,
    trace: pd.DataFrame,
    sheet_name: str | None,
    source_name: str | None = None,
) -> list[dict[str, object]]:
    try:
        voltage_col = first_existing(trace, ["voltage_v", "voltage", "v"])
        current_col = first_existing(trace, ["current_a", "current", "i"])
        time_col = first_existing(trace, ["time_s", "test_time_s", "time", "time_sec"])
    except KeyError as exc:
        warn_skip(path, cycle_index, str(exc))
        return []

    current = trace[current_col].astype(float).to_numpy()
    time_s = trace[time_col].astype(float).to_numpy()
    discharge = trace[current < -1e-6]
    charge = trace[current > 1e-6]
    dt_hours = _delta_hours(time_s)
    discharge_capacity = _capacity_from_columns(
        trace, ["discharge_capacity_ah", "capacity_ah"], current < -1e-6, dt_hours
    )
    charge_capacity = _capacity_from_columns(
        trace, ["charge_capacity_ah"], current > 1e-6, dt_hours
    )
    if charge_capacity <= 0:
        charge_capacity = discharge_capacity / 0.995 if discharge_capacity > 0 else 0.0
    if discharge.empty:
        discharge = trace
    if (
        discharge_capacity <= 0.05
        or charge_capacity <= 0
        or trace[[voltage_col, current_col]].isna().any().any()
    ):
        warn_skip(path, cycle_index, "non-positive/junk capacity or NaN curve")
        return []

    nominal = _nominal_capacity(source_cell_id)
    cell_id = f"calce_{source_cell_id}".lower().replace("-", "_")
    temp_col = _maybe_column(trace, ["temperature_c", "temperature"])
    discharge_time = discharge[time_col].astype(float) - float(discharge[time_col].iloc[0])
    return [
        {
            "cell_id": cell_id,
            "dataset": "calce",
            "chemistry": "LCO",
            "manufacturer": "CALCE",
            "nominal_capacity_ah": nominal,
            "cycle_index": cycle_index,
            "is_eol_cycle": False,
            "discharge_capacity_ah": discharge_capacity,
            "charge_capacity_ah": charge_capacity,
            "coulombic_efficiency": discharge_capacity / charge_capacity,
            "energy_efficiency": _energy_efficiency(trace),
            "soh": clipped_soh(discharge_capacity, nominal),
            "temperature_c_mean": float(trace[temp_col].mean()) if temp_col else None,
            "c_rate_charge": _mean_c_rate(charge[current_col], nominal),
            "c_rate_discharge": _mean_c_rate(discharge[current_col], nominal),
            "voltage_curve": downsample(discharge[voltage_col]),
            "current_curve": downsample(discharge[current_col]),
            "time_curve_s": downsample(discharge_time),
            "dq_dv_curve": compute_dq_dv(
                discharge[voltage_col], np.linspace(0, discharge_capacity, len(discharge))
            ),
            "_cycle_start": _cycle_start(trace),
            "_source_name": source_name or sheet_name or path.name,
        }
    ]


def _capacity_from_columns(
    trace: pd.DataFrame, candidates: list[str], current_mask: np.ndarray, dt_hours: np.ndarray
) -> float:
    column = _maybe_column(trace, candidates)
    if column:
        values = pd.to_numeric(trace[column], errors="coerce")
        if values.notna().any():
            return float(abs(values.max() - values.min()))
    current_col = first_existing(trace, ["current_a", "current", "i"])
    current = trace[current_col].astype(float).to_numpy()
    return abs(float(np.sum(current[current_mask] * dt_hours[current_mask])))


def _energy_efficiency(trace: pd.DataFrame) -> float | None:
    charge_col = _maybe_column(trace, ["charge_energy_wh"])
    discharge_col = _maybe_column(trace, ["discharge_energy_wh"])
    if not charge_col or not discharge_col:
        return None
    charge = pd.to_numeric(trace[charge_col], errors="coerce")
    discharge = pd.to_numeric(trace[discharge_col], errors="coerce")
    charge_delta = float(charge.max() - charge.min())
    discharge_delta = float(discharge.max() - discharge.min())
    if charge_delta <= 0 or discharge_delta <= 0:
        return None
    return discharge_delta / charge_delta


def _cycle_start(trace: pd.DataFrame) -> pd.Timestamp:
    if "_parsed_date_time" in trace.columns and trace["_parsed_date_time"].notna().any():
        return pd.Timestamp(trace["_parsed_date_time"].min())
    time_col = _maybe_column(trace, ["time_s", "test_time_s", "time", "time_sec"])
    if time_col:
        return pd.Timestamp("1970-01-01") + pd.to_timedelta(float(trace[time_col].min()), unit="s")
    return pd.Timestamp("1970-01-01")


def _reindex_real_calce_cycles(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows or "_cycle_start" not in rows[0]:
        return rows
    frame = pd.DataFrame(rows).sort_values(["cell_id", "_cycle_start", "_source_name"])
    cleaned: list[dict[str, object]] = []
    for _, group in frame.groupby("cell_id", sort=False):
        for new_index, (_, row) in enumerate(group.iterrows(), start=1):
            record = row.to_dict()
            record["cycle_index"] = new_index
            record.pop("_cycle_start", None)
            record.pop("_source_name", None)
            cleaned.append(record)
    return cleaned


def _nominal_capacity(source_cell_id: str) -> float:
    # CALCE battery-data page lists CS2 capacity as 1100 mAh and CX2 as 1350 mAh.
    # Source: https://calce.umd.edu/battery-data
    return 1.35 if "CX2" in source_cell_id.upper() else 1.10


def _delta_hours(time_s: np.ndarray) -> np.ndarray:
    if time_s.size == 1:
        return np.asarray([0.0])
    deltas = np.diff(time_s, prepend=time_s[0])
    if deltas[0] == 0 and deltas.size > 1:
        deltas[0] = deltas[1]
    return np.maximum(deltas, 0.0) / 3600.0


def _mean_c_rate(values: pd.Series, nominal_capacity_ah: float) -> float | None:
    if values.empty:
        return None
    return float(values.abs().mean() / nominal_capacity_ah)


def _maybe_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _cell_from_path(path: Path, source_name: str | None = None) -> str:
    parts = [*(source_name or "").split("/"), *path.parts]
    for part in reversed(parts):
        upper = part.upper().replace("-", "_")
        if upper.startswith(("CS2_", "CX2_")):
            return upper.split(".")[0]
    return path.stem
