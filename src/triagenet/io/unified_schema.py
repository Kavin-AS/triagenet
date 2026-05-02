"""Canonical cycle-level schema for all TriageNet raw battery datasets.

Each row represents one full charge/discharge cycle from one cell after parsing from a
source-specific raw format. Units and provenance:

- cell_id: globally unique source-derived cell identifier.
- dataset: source dataset label, one of sandia, calce, mit.
- chemistry: cathode chemistry label, one of LFP, NMC, NCA, LCO, inferred from source metadata.
- manufacturer: source metadata manufacturer or "unknown" when unavailable.
- nominal_capacity_ah: rated cell capacity in ampere-hours from source metadata or config default.
- cycle_index: 1-based cycle number, monotonic within a cell.
- is_eol_cycle: true only for the last retained full cycle for a cell.
- discharge_capacity_ah / charge_capacity_ah: cycle capacities in ampere-hours.
- coulombic_efficiency: discharge_capacity_ah divided by charge_capacity_ah.
- energy_efficiency: discharge energy divided by charge energy when source energy is available.
- soh: discharge_capacity_ah / nominal_capacity_ah, clipped by loaders to [0, 1.05].
- temperature_c_mean: mean cell or ambient temperature in degrees Celsius when present.
- c_rate_charge / c_rate_discharge: absolute current divided by nominal capacity where inferable.
- voltage_curve/current_curve/time_curve_s: discharge traces linearly downsampled to 100 points.
- dq_dv_curve: incremental-capacity trace downsampled to 100 points when dense enough.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
import pyarrow as pa

from triagenet.config import CHEMISTRIES, CURVE_LENGTH


class SchemaError(ValueError):
    """Raised when a DataFrame does not satisfy the UnifiedCycle schema."""


@dataclass(frozen=True)
class UnifiedCycle:
    """Documented dataclass representation of the canonical cycle-level row."""

    cell_id: str
    dataset: str
    chemistry: str
    manufacturer: str
    nominal_capacity_ah: float
    cycle_index: int
    is_eol_cycle: bool
    discharge_capacity_ah: float
    charge_capacity_ah: float
    coulombic_efficiency: float
    energy_efficiency: float | None
    soh: float
    temperature_c_mean: float | None
    c_rate_charge: float | None
    c_rate_discharge: float | None
    voltage_curve: list[float]
    current_curve: list[float]
    time_curve_s: list[float]
    dq_dv_curve: list[float] | None

    datasets: ClassVar[set[str]] = {"sandia", "calce", "mit"}


UNIFIED_CYCLE_SCHEMA = pa.schema(
    [
        pa.field("cell_id", pa.string(), nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("chemistry", pa.string(), nullable=False),
        pa.field("manufacturer", pa.string(), nullable=False),
        pa.field("nominal_capacity_ah", pa.float64(), nullable=False),
        pa.field("cycle_index", pa.int64(), nullable=False),
        pa.field("is_eol_cycle", pa.bool_(), nullable=False),
        pa.field("discharge_capacity_ah", pa.float64(), nullable=False),
        pa.field("charge_capacity_ah", pa.float64(), nullable=False),
        pa.field("coulombic_efficiency", pa.float64(), nullable=False),
        pa.field("energy_efficiency", pa.float64(), nullable=True),
        pa.field("soh", pa.float64(), nullable=False),
        pa.field("temperature_c_mean", pa.float64(), nullable=True),
        pa.field("c_rate_charge", pa.float64(), nullable=True),
        pa.field("c_rate_discharge", pa.float64(), nullable=True),
        pa.field("voltage_curve", pa.list_(pa.float64(), CURVE_LENGTH), nullable=False),
        pa.field("current_curve", pa.list_(pa.float64(), CURVE_LENGTH), nullable=False),
        pa.field("time_curve_s", pa.list_(pa.float64(), CURVE_LENGTH), nullable=False),
        pa.field("dq_dv_curve", pa.list_(pa.float64(), CURVE_LENGTH), nullable=True),
    ]
)

REQUIRED_COLUMNS = [field.name for field in UNIFIED_CYCLE_SCHEMA]


def validate_unified_cycle(df: pd.DataFrame) -> None:
    """Validate that a DataFrame conforms to the UnifiedCycle schema and invariants."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")
    if df.empty:
        raise SchemaError("UnifiedCycle DataFrame is empty")

    _validate_string_column(df, "cell_id")
    _validate_string_column(df, "dataset", allowed=UnifiedCycle.datasets)
    _validate_string_column(df, "chemistry", allowed=set(CHEMISTRIES))
    _validate_string_column(df, "manufacturer")
    _validate_bool_column(df, "is_eol_cycle")

    for column in (
        "nominal_capacity_ah",
        "discharge_capacity_ah",
        "charge_capacity_ah",
        "coulombic_efficiency",
        "soh",
    ):
        _validate_numeric_column(df, column, nullable=False)

    for column in (
        "energy_efficiency",
        "temperature_c_mean",
        "c_rate_charge",
        "c_rate_discharge",
    ):
        _validate_numeric_column(df, column, nullable=True)

    if not pd.api.types.is_integer_dtype(df["cycle_index"]):
        raise SchemaError("cycle_index must be an integer dtype")
    if (df["cycle_index"] < 1).any():
        raise SchemaError("cycle_index must be 1-based and positive")
    if (df["soh"] < 0).any() or (df["soh"] > 1.05).any():
        raise SchemaError("soh must be in [0, 1.05]")
    if (df["nominal_capacity_ah"] <= 0).any():
        raise SchemaError("nominal_capacity_ah must be positive")
    if (df["discharge_capacity_ah"] <= 0).any() or (df["charge_capacity_ah"] <= 0).any():
        raise SchemaError("cycle capacities must be positive")

    for cell_id, group in df.sort_values(["cell_id", "cycle_index"]).groupby("cell_id"):
        cycle_indices = group["cycle_index"].to_numpy()
        if not np.all(np.diff(cycle_indices) > 0):
            raise SchemaError(f"cycle_index must be strictly increasing for {cell_id}")
        eol_count = int(group["is_eol_cycle"].sum())
        if eol_count != 1:
            raise SchemaError(f"{cell_id} must have exactly one EOL cycle, found {eol_count}")
        max_cycle = group["cycle_index"].max()
        eol_cycle = group.loc[group["is_eol_cycle"], "cycle_index"].iloc[0]
        if int(eol_cycle) != int(max_cycle):
            raise SchemaError(f"EOL cycle must be the last retained cycle for {cell_id}")

    for column in ("voltage_curve", "current_curve", "time_curve_s"):
        _validate_curve_column(df, column, nullable=False)
    _validate_curve_column(df, "dq_dv_curve", nullable=True)


def _validate_string_column(df: pd.DataFrame, column: str, allowed: set[str] | None = None) -> None:
    if not pd.api.types.is_string_dtype(df[column]):
        raise SchemaError(f"{column} must be a string dtype")
    if df[column].isna().any() or (df[column].str.len() == 0).any():
        raise SchemaError(f"{column} cannot contain null or empty values")
    if allowed is not None:
        invalid = set(df[column]) - allowed
        if invalid:
            raise SchemaError(f"{column} has invalid values: {sorted(invalid)}")


def _validate_bool_column(df: pd.DataFrame, column: str) -> None:
    if not pd.api.types.is_bool_dtype(df[column]):
        raise SchemaError(f"{column} must be a boolean dtype")


def _validate_numeric_column(df: pd.DataFrame, column: str, nullable: bool) -> None:
    series = df[column]
    if nullable and series.isna().all():
        return
    numeric = pd.to_numeric(series, errors="coerce")
    if not pd.api.types.is_numeric_dtype(series) and numeric[series.notna()].isna().any():
        raise SchemaError(f"{column} must be numeric")
    if not nullable and series.isna().any():
        raise SchemaError(f"{column} cannot contain null values")
    values = numeric.dropna().to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise SchemaError(f"{column} contains non-finite values")


def _validate_curve_column(df: pd.DataFrame, column: str, nullable: bool) -> None:
    for index, value in df[column].items():
        if value is None and nullable:
            continue
        if value is None:
            raise SchemaError(f"{column} cannot be null at row {index}")
        if not isinstance(value, list) or len(value) != CURVE_LENGTH:
            raise SchemaError(f"{column} must contain {CURVE_LENGTH}-point lists")
        values = np.asarray(value, dtype=float)
        if not np.isfinite(values).all():
            raise SchemaError(f"{column} contains non-finite values at row {index}")
