"""Tests for Phase 1 dataset loaders using tiny synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

from triagenet.io import calce_loader, mit_loader, sandia_loader
from triagenet.io.unified_schema import validate_unified_cycle

FIXTURES = Path(__file__).parent / "fixtures"


def test_sandia_loader_returns_unified_schema() -> None:
    """Sandia paired cycle/timeseries CSV fixtures parse into valid unified rows."""
    df = sandia_loader.load(FIXTURES / "sandia")
    validate_unified_cycle(df)
    assert len(df) == 5
    assert df["is_eol_cycle"].sum() == 1
    assert df["chemistry"].unique().tolist() == ["LFP"]


def test_calce_loader_returns_unified_schema_and_skips_malformed() -> None:
    """CALCE synthetic fixtures parse valid cycles and log-skip malformed rows."""
    df = calce_loader.load(FIXTURES / "calce")
    validate_unified_cycle(df)
    assert len(df) == 5
    assert df["dataset"].unique().tolist() == ["calce"]
    assert df["cell_id"].nunique() == 1
    assert df.groupby("cell_id")["is_eol_cycle"].sum().eq(1).all()


def test_calce_real_minimal_fixture_is_real_arbin_shape() -> None:
    """The minimized real CALCE workbook keeps the Arbin channel columns."""
    df = calce_loader.load(FIXTURES / "calce_real_minimal.xlsx")
    real = df[df["cell_id"] == "calce_calce_real_minimal"]
    validate_unified_cycle(real)
    assert real["cycle_index"].is_monotonic_increasing
    assert real["discharge_capacity_ah"].max() > 1.0


def test_mit_loader_returns_unified_schema() -> None:
    """MIT/Stanford fixture parses into valid LFP unified rows."""
    df = mit_loader.load(FIXTURES / "mit")
    validate_unified_cycle(df)
    assert len(df) == 5
    assert df["manufacturer"].unique().tolist() == ["A123"]


def test_mit_v73_loader_returns_unified_schema() -> None:
    """MIT/Stanford MATLAB v7.3 HDF5 fixture parses into valid LFP unified rows."""
    df = mit_loader.load(FIXTURES / "mit_v73_minimal.mat")
    validate_unified_cycle(df)
    assert len(df) == 10
    assert df["cell_id"].unique().tolist() == ["mit_mit-v73-minic0"]
    assert df["chemistry"].unique().tolist() == ["LFP"]
    assert df["is_eol_cycle"].sum() == 1
