"""Tests for matched-SOH chemistry dataset construction and grouped splits."""

from __future__ import annotations

import pandas as pd
from triagenet.models.data_split import build_chemistry_dataset, leave_one_cell_out_split


def _split_frame() -> pd.DataFrame:
    rows = []
    for chemistry, dataset in [("LFP", "mit"), ("LCO", "calce")]:
        for cell in range(3):
            for cycle in range(4):
                rows.append(
                    {
                        "cell_id": f"{dataset}_{cell}",
                        "dataset": dataset,
                        "chemistry": chemistry,
                        "cycle_index": cycle + 1,
                        "soh": 0.75 + 0.05 * cycle,
                    }
                )
    return pd.DataFrame(rows)


def test_build_chemistry_dataset_filters_soh_before_splitting() -> None:
    """The matched SOH window is applied before any chemistry split."""
    df = build_chemistry_dataset(_split_frame(), soh_window=(0.80, 0.90))
    assert df["soh"].between(0.80, 0.90).all()
    assert set(df["chemistry"]) == {"LCO", "LFP"}


def test_leave_one_cell_out_split_has_no_cell_leakage() -> None:
    """No fold may place cycles from the same cell in train and test."""
    df = build_chemistry_dataset(_split_frame(), soh_window=(0.80, 0.90))
    for train_idx, test_idx in leave_one_cell_out_split(df):
        train_cells = set(df.iloc[train_idx]["cell_id"])
        test_cells = set(df.iloc[test_idx]["cell_id"])
        assert train_cells.isdisjoint(test_cells)
