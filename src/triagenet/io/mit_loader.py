"""Loader for MIT/Stanford Severson fast-charge LFP data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat

from triagenet.config import DATA_RAW, DEFAULT_NOMINAL_CAPACITY_AH
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

LOGGER = logging.getLogger(__name__)

# Published Severson/Braatz processing notebooks remove these cells before analysis.
# Source: https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation
# Load Data.ipynb deletes b1c8/b1c10/b1c12/b1c13/b1c22, moves b2c7/b2c8/b2c9/b2c15/b2c16
# into Batch 1 continuations, and deletes noisy Batch 3 channels b3c37/b3c2/b3c23/b3c32/b3c42/b3c43.
MIT_KNOWN_BAD_CELLS: set[tuple[str, int]] = {
    ("b1", 8),
    ("b1", 10),
    ("b1", 12),
    ("b1", 13),
    ("b1", 22),
    ("b2", 7),
    ("b2", 8),
    ("b2", 9),
    ("b2", 15),
    ("b2", 16),
    ("b3", 2),
    ("b3", 23),
    ("b3", 32),
    ("b3", 37),
    ("b3", 42),
    ("b3", 43),
}

MATLAB_V5_HEADER = b"MATLAB 5.0 MAT-file"
HDF5_MAGIC = b"\x89HDF"


def load(raw_dir: Path | None = None) -> pd.DataFrame:
    """Parse MIT/Stanford CSV or MAT files into the UnifiedCycle schema."""
    root = raw_dir or DATA_RAW / "mit_stanford"
    rows: list[dict[str, object]] = []
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        if path.suffix.lower() == ".csv":
            rows.extend(_parse_csv(path))
        elif path.suffix.lower() == ".mat":
            rows.extend(_parse_mat(path))
    df = finalize_frame(rows)
    if not df.empty:
        validate_unified_cycle(df)
    return df


def _parse_csv(path: Path) -> list[dict[str, object]]:
    df = normalize_columns(pd.read_csv(path))
    cell_col = first_existing(df, ["cell_id", "cell"])
    cycle_col = first_existing(df, ["cycle_index", "cycle", "cycle_number"])
    rows = []
    for (cell_id, cycle_index), trace in df.groupby([cell_col, cycle_col], sort=True):
        rows.extend(_build_row(path, str(cell_id), int(cycle_index), trace))
    return rows


def _parse_mat(path: Path) -> list[dict[str, object]]:
    if _is_matlab_v73(path):
        cells = _load_v73(path)
        return _cells_to_rows(path, cells)
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    rows = []
    if {"cell_id", "cycle_index", "time_s", "voltage_v", "current_a"}.issubset(mat):
        frame = pd.DataFrame(
            {
                "cell_id": np.atleast_1d(mat["cell_id"]).astype(str),
                "cycle_index": np.atleast_1d(mat["cycle_index"]).astype(int),
                "time_s": np.atleast_1d(mat["time_s"]).astype(float),
                "voltage_v": np.atleast_1d(mat["voltage_v"]).astype(float),
                "current_a": np.atleast_1d(mat["current_a"]).astype(float),
            }
        )
        return _parse_csv_like(path, frame)
    if "batch" in mat:
        return _parse_batch_array(path, mat["batch"])
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        rows.extend(_parse_possible_cell_struct(path, key, value))
    if not rows:
        raise ValueError(f"Could not find recognized Severson cycle arrays in {path}")
    return rows


def _is_matlab_v73(path: Path) -> bool:
    with path.open("rb") as handle:
        header = handle.read(512)
    if header.startswith(MATLAB_V5_HEADER):
        return False
    return header.startswith(HDF5_MAGIC) or b"MATLAB 7.3" in header or h5py.is_hdf5(path)


def _load_v73(path: Path) -> list[dict[str, object]]:
    with h5py.File(path, "r") as h5:
        if "batch" not in h5:
            raise ValueError(f"MAT v7.3 file has no top-level 'batch' group: {path}")
        batch = h5["batch"]
        batch_name = _batch_name(path)
        num_cells = int(batch["summary"].shape[0])
        cells = []
        for cell_index in range(num_cells):
            if (batch_name, cell_index) in MIT_KNOWN_BAD_CELLS:
                cycle_count = _cell_cycle_count(h5, batch, cell_index)
                LOGGER.warning(
                    "Skipping known-bad MIT cell %s "
                    "(index %s, %s cycles) per Severson reference repo",
                    f"{batch_name}c{cell_index}",
                    cell_index,
                    cycle_count,
                )
                continue
            cells.append(_read_v73_cell(h5, batch, batch_name, cell_index))
    return cells


def _read_v73_cell(
    h5: h5py.File, batch: h5py.Group, batch_name: str, cell_index: int
) -> dict[str, object]:
    summary = h5[batch["summary"][cell_index, 0]]
    cycles_group = h5[batch["cycles"][cell_index, 0]]
    summary_qd = _dataset_array(summary["QDischarge"])
    summary_qc = _dataset_array(summary["QCharge"])
    summary_ir = _dataset_array(summary["IR"])
    summary_tavg = _dataset_array(summary["Tavg"])
    policy = _read_hdf5_string(h5, batch["policy_readable"][cell_index, 0])
    cycle_count = int(cycles_group["V"].shape[0])
    cycles = []
    for cycle_position in range(cycle_count):
        voltage = _deref_array(h5, cycles_group["V"][cycle_position, 0])
        current = _deref_array(h5, cycles_group["I"][cycle_position, 0])
        time_s = _normalize_time_seconds(_deref_array(h5, cycles_group["t"][cycle_position, 0]))
        qd_curve = _deref_array(h5, cycles_group["Qd"][cycle_position, 0])
        qc_curve = _deref_array(h5, cycles_group["Qc"][cycle_position, 0])
        temperature = _deref_array(h5, cycles_group["T"][cycle_position, 0])
        if voltage.size < 3 or current.size != voltage.size or time_s.size != voltage.size:
            continue
        cycles.append(
            {
                "cycle_index": cycle_position + 1,
                "voltage": voltage,
                "current": current,
                "time_s": time_s,
                "discharge_capacity_ah": _capacity_for_cycle(qd_curve, summary_qd, cycle_position),
                "charge_capacity_ah": _capacity_for_cycle(qc_curve, summary_qc, cycle_position),
                "temperature_mean_c": _indexed_or_mean(summary_tavg, temperature, cycle_position),
                "internal_resistance_ohm": _indexed_or_none(summary_ir, cycle_position),
                "dq_dv_curve": _read_optional_dq_dv(h5, cycles_group, cycle_position),
            }
        )
    return {
        "cell_id": f"mit_{batch_name}c{cell_index}",
        "cycles": cycles,
        "nominal_capacity_ah": DEFAULT_NOMINAL_CAPACITY_AH["LFP"]["A123_APR18650M1A"],
        "manufacturer": "A123 Systems",
        "chemistry": "LFP",
        "policy": policy,
    }


def _cells_to_rows(path: Path, cells: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for cell in cells:
        kept_index = 0
        for cycle in cell["cycles"]:
            if cycle["discharge_capacity_ah"] <= 0.05:
                warn_skip(path, cycle["cycle_index"], "non-positive/junk discharge capacity")
                continue
            kept_index += 1
            cycle = {**cycle, "cycle_index": kept_index}
            rows.extend(_build_row_from_v73_cell(cell, cycle))
    return rows


def _build_row_from_v73_cell(
    cell: dict[str, object], cycle: dict[str, object]
) -> list[dict[str, object]]:
    nominal = float(cell["nominal_capacity_ah"])
    discharge_capacity = float(cycle["discharge_capacity_ah"])
    charge_capacity = float(cycle["charge_capacity_ah"])
    if charge_capacity <= 0:
        charge_capacity = discharge_capacity / 0.995
    current = np.asarray(cycle["current"], dtype=float)
    voltage = np.asarray(cycle["voltage"], dtype=float)
    time_s = np.asarray(cycle["time_s"], dtype=float)
    discharge_mask = current < -1e-9
    discharge_current = current[current < -1e-9]
    charge_current = current[current > 1e-9]
    curve_voltage = voltage[discharge_mask] if discharge_mask.any() else voltage
    curve_current = current[discharge_mask] if discharge_mask.any() else current
    curve_time = time_s[discharge_mask] if discharge_mask.any() else time_s
    curve_time = curve_time - curve_time[0]
    dq_dv = cycle.get("dq_dv_curve")
    return [
        {
            "cell_id": str(cell["cell_id"]),
            "dataset": "mit",
            "chemistry": str(cell["chemistry"]),
            "manufacturer": str(cell["manufacturer"]),
            "nominal_capacity_ah": nominal,
            "cycle_index": int(cycle["cycle_index"]),
            "is_eol_cycle": False,
            "discharge_capacity_ah": discharge_capacity,
            "charge_capacity_ah": charge_capacity,
            "coulombic_efficiency": discharge_capacity / charge_capacity,
            "energy_efficiency": None,
            "soh": clipped_soh(discharge_capacity, nominal),
            "temperature_c_mean": cycle["temperature_mean_c"],
            "c_rate_charge": (
                float(np.mean(np.abs(charge_current)) / nominal) if charge_current.size else None
            ),
            "c_rate_discharge": (
                float(np.mean(np.abs(discharge_current)) / nominal)
                if discharge_current.size
                else None
            ),
            "voltage_curve": downsample(curve_voltage),
            "current_curve": downsample(curve_current),
            "time_curve_s": downsample(curve_time),
            "dq_dv_curve": downsample(dq_dv) if dq_dv is not None and len(dq_dv) >= 3 else None,
        }
    ]


def _parse_csv_like(path: Path, frame: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for (cell_id, cycle_index), trace in frame.groupby(["cell_id", "cycle_index"], sort=True):
        rows.extend(_build_row(path, str(cell_id), int(cycle_index), trace))
    return rows


def _parse_batch_array(path: Path, batch: Any) -> list[dict[str, object]]:
    rows = []
    for index, cell in enumerate(np.atleast_1d(batch), start=1):
        policy = _scalar_string(getattr(cell, "policy_readable", "")) or f"cell{index:03d}"
        cell_id = f"batch_{index:03d}_{policy}".replace(" ", "_").replace("-", "_")
        rows.extend(_parse_possible_cell_struct(path, cell_id, cell))
    return rows


def _parse_possible_cell_struct(path: Path, cell_id: str, value: Any) -> list[dict[str, object]]:
    cycle_life = getattr(value, "cycle_life", None)
    cycles = getattr(value, "cycles", None)
    if cycles is None:
        return []
    rows = []
    for index, cycle in enumerate(np.atleast_1d(cycles), start=1):
        # Real Severson MATR files store each cycle as batch[i].cycles[j].V/I/t/Qd.
        time_s = np.asarray(getattr(cycle, "t", []), dtype=float).ravel()
        voltage = np.asarray(getattr(cycle, "V", []), dtype=float).ravel()
        current = np.asarray(getattr(cycle, "I", []), dtype=float).ravel()
        qd = np.asarray(getattr(cycle, "Qd", []), dtype=float).ravel()
        if time_s.size and voltage.size and current.size:
            trace = pd.DataFrame(
                {
                    "time_s": time_s,
                    "voltage_v": voltage,
                    "current_a": current,
                    "cell_id": cell_id,
                    "discharge_capacity_ah": qd if qd.size == time_s.size else np.nan,
                }
            )
            rows.extend(_build_row(path, cell_id, index, trace, cycle_life=cycle_life))
    return rows


def _build_row(
    path: Path, source_cell_id: str, cycle_index: int, trace: pd.DataFrame, cycle_life: Any = None
) -> list[dict[str, object]]:
    try:
        voltage_col = first_existing(trace, ["voltage_v", "voltage", "v"])
        current_col = first_existing(trace, ["current_a", "current", "i"])
        time_col = first_existing(trace, ["time_s", "test_time_s", "time", "t"])
    except KeyError as exc:
        warn_skip(path, cycle_index, str(exc))
        return []
    voltage = trace[voltage_col].astype(float).to_numpy()
    current = trace[current_col].astype(float).to_numpy()
    time_s = trace[time_col].astype(float).to_numpy()
    discharge = trace[current < 0]
    capacity_col = _maybe_column(trace, ["discharge_capacity_ah", "qd"])
    if discharge.empty and capacity_col is not None:
        trace = trace.assign(**{current_col: -trace[current_col].abs()})
        current = trace[current_col].astype(float).to_numpy()
        discharge = trace
    if discharge.empty:
        warn_skip(path, cycle_index, "no discharge segment")
        return []
    dt_hours = _delta_hours(time_s)
    discharge_capacity = abs(float(np.sum(current[current < 0] * dt_hours[current < 0])))
    if capacity_col is not None:
        qd = pd.to_numeric(trace[capacity_col], errors="coerce")
        if qd.notna().any():
            discharge_capacity = float(abs(qd.max() - qd.min()))
    charge_capacity = abs(float(np.sum(current[current > 0] * dt_hours[current > 0])))
    if discharge_capacity <= 0:
        warn_skip(path, cycle_index, "non-positive discharge capacity")
        return []
    if charge_capacity <= 0:
        charge_capacity = discharge_capacity / 0.995
    if np.isnan(voltage).any() or np.isnan(current).any():
        warn_skip(path, cycle_index, "NaN curve")
        return []
    nominal = DEFAULT_NOMINAL_CAPACITY_AH["LFP"]["A123_APR18650M1A"]
    cell_id = f"mit_{source_cell_id}".lower()
    return [
        {
            "cell_id": cell_id,
            "dataset": "mit",
            "chemistry": "LFP",
            "manufacturer": "A123",
            "nominal_capacity_ah": nominal,
            "cycle_index": cycle_index,
            "is_eol_cycle": False,
            "discharge_capacity_ah": discharge_capacity,
            "charge_capacity_ah": charge_capacity,
            "coulombic_efficiency": discharge_capacity / charge_capacity,
            "energy_efficiency": None,
            "soh": clipped_soh(discharge_capacity, nominal),
            "temperature_c_mean": None,
            "c_rate_charge": _mean_c_rate(trace.loc[current > 0, current_col], nominal),
            "c_rate_discharge": _mean_c_rate(discharge[current_col], nominal),
            "voltage_curve": downsample(discharge[voltage_col]),
            "current_curve": downsample(discharge[current_col]),
            "time_curve_s": downsample(discharge[time_col]),
            "dq_dv_curve": compute_dq_dv(
                discharge[voltage_col], np.linspace(0, discharge_capacity, len(discharge))
            ),
        }
    ]


def _batch_name(path: Path) -> str:
    name = path.name
    if "2017-05-12" in name:
        return "b1"
    if "2017-06-30" in name:
        return "b2"
    if "2018-04-12" in name:
        return "b3"
    if "2019-01-24" in name:
        return "b4"
    return path.stem.replace("_", "-")[:12]


def _cell_cycle_count(h5: h5py.File, batch: h5py.Group, cell_index: int) -> int:
    return int(h5[batch["cycles"][cell_index, 0]]["V"].shape[0])


def _dataset_array(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset[()], dtype=float).ravel()


def _deref_array(h5: h5py.File, reference: h5py.Reference) -> np.ndarray:
    return np.asarray(h5[reference][()], dtype=float).ravel()


def _read_hdf5_string(h5: h5py.File, reference: h5py.Reference) -> str:
    values = np.asarray(h5[reference][()]).ravel()
    if values.dtype.kind in {"u", "i"}:
        return "".join(chr(int(value)) for value in values if int(value) != 0)
    return "".join(str(value) for value in values)


def _normalize_time_seconds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    if values.size and np.nanmax(values) < 200:
        return values * 60.0
    return values


def _capacity_for_cycle(
    curve: np.ndarray, summary_values: np.ndarray, cycle_position: int
) -> float:
    if curve.size and np.isfinite(curve).any():
        return float(np.nanmax(curve) - np.nanmin(curve))
    return _indexed_or_none(summary_values, cycle_position) or 0.0


def _indexed_or_mean(
    summary_values: np.ndarray, curve_values: np.ndarray, cycle_position: int
) -> float | None:
    indexed = _indexed_or_none(summary_values, cycle_position)
    if indexed is not None:
        return indexed
    finite = curve_values[np.isfinite(curve_values)]
    if finite.size:
        return float(np.mean(finite))
    return None


def _indexed_or_none(values: np.ndarray, cycle_position: int) -> float | None:
    if cycle_position >= values.size:
        return None
    value = float(values[cycle_position])
    if not np.isfinite(value):
        return None
    return value


def _read_optional_dq_dv(
    h5: h5py.File, cycles_group: h5py.Group, cycle_position: int
) -> np.ndarray | None:
    field = "discharge_dQdV"
    if field not in cycles_group:
        return None
    values = _deref_array(h5, cycles_group[field][cycle_position, 0])
    if values.size < 3 or not np.isfinite(values).all():
        return None
    return values


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


def _scalar_string(value: Any) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return ""
    return str(array.item() if array.size == 1 else array.ravel()[0])
