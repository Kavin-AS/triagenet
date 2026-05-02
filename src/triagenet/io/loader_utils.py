"""Shared parsing utilities for raw battery dataset loaders."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from triagenet.config import CURVE_LENGTH

LOGGER = logging.getLogger(__name__)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase snake-case columns for permissive raw parsing."""
    renamed = {
        column: re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        for column in df.columns
    }
    return df.rename(columns=renamed)


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    """Return the first candidate column present in a normalized DataFrame."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise KeyError(f"None of the expected columns exist: {list(candidates)}")


def infer_chemistry(text: str) -> str:
    """Infer a supported chemistry label from a file path or metadata string."""
    upper = text.upper()
    for chemistry in ("LFP", "NMC", "NCA", "LCO"):
        if chemistry in upper:
            return chemistry
    if "LICOO2" in upper or "LI_CO_O2" in upper:
        return "LCO"
    raise ValueError(f"Could not infer chemistry from {text}")


def downsample(values: Iterable[float], length: int = CURVE_LENGTH) -> list[float]:
    """Linearly downsample a numeric sequence to a fixed-length list."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("curve contains no finite values")
    if array.size == 1:
        return [float(array[0])] * length
    source_x = np.linspace(0.0, 1.0, array.size)
    target_x = np.linspace(0.0, 1.0, length)
    return np.interp(target_x, source_x, array).astype(float).tolist()


def compute_dq_dv(voltage: Iterable[float], capacity_ah: Iterable[float]) -> list[float] | None:
    """Compute a 100-point incremental-capacity curve when voltage data is dense enough."""
    voltage_array = np.asarray(list(voltage), dtype=float)
    capacity_array = np.asarray(list(capacity_ah), dtype=float)
    if voltage_array.size < CURVE_LENGTH or capacity_array.size != voltage_array.size:
        return None
    if not np.isfinite(voltage_array).all() or not np.isfinite(capacity_array).all():
        return None
    dv = np.gradient(voltage_array)
    dq = np.gradient(capacity_array)
    with np.errstate(divide="ignore", invalid="ignore"):
        dq_dv = dq / dv
    dq_dv = np.nan_to_num(dq_dv, nan=0.0, posinf=0.0, neginf=0.0)
    return downsample(dq_dv)


def finalize_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Return deterministic UnifiedCycle-like rows with EOL flags set per cell."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(["cell_id", "cycle_index"]).reset_index(drop=True)
    df["is_eol_cycle"] = False
    max_indices = df.groupby("cell_id")["cycle_index"].transform("max")
    df.loc[df["cycle_index"] == max_indices, "is_eol_cycle"] = True
    df["cycle_index"] = df["cycle_index"].astype("int64")
    return df


def warn_skip(path: Path, cycle_index: object, reason: str) -> None:
    """Log an explicit warning for a skipped malformed cycle."""
    LOGGER.warning("Skipping %s cycle %s: %s", path, cycle_index, reason)


def clipped_soh(discharge_capacity_ah: float, nominal_capacity_ah: float) -> float:
    """Return discharge capacity divided by nominal capacity clipped to the schema range."""
    return float(np.clip(discharge_capacity_ah / nominal_capacity_ah, 0.0, 1.05))
