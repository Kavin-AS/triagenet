"""Matched-SOH data construction and leakage-safe chemistry splitters."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def build_chemistry_dataset(
    cycles_df: pd.DataFrame, soh_window: tuple[float, float] = (0.80, 1.00)
) -> pd.DataFrame:
    """Return cycles inside the matched SOH window used by the chemistry classifier.

    The 0.80-1.00 default is the intersection where CALCE LCO and MIT LFP both have
    coverage. Filtering before feature extraction prevents the model from learning the
    dataset asymmetry "low SOH implies LCO" instead of chemistry-specific curve shape.
    """
    low, high = soh_window
    if low >= high:
        raise ValueError("soh_window must be ordered as (low, high)")
    filtered = cycles_df[cycles_df["soh"].between(low, high, inclusive="both")].copy()
    if filtered.empty:
        raise ValueError(f"No cycles found inside SOH window {soh_window}")
    return filtered.sort_values(["dataset", "chemistry", "cell_id", "cycle_index"]).reset_index(
        drop=True
    )


def leave_one_cell_out_split(df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield GroupKFold train/test indices with no cell appearing in both sides."""
    groups = df["cell_id"].to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(5, unique_groups.size)
    if n_splits < 2:
        raise ValueError("Need at least two cells for grouped splitting")
    splitter = GroupKFold(n_splits=n_splits)
    yield from splitter.split(df, df["chemistry"], groups=groups)


def leave_one_dataset_out_split(df: pd.DataFrame) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield leave-one-dataset-out indices.

    With only CALCE=LCO and MIT=LFP this split is intentionally reported as untrainable by
    the training script because each training fold contains one class. It becomes a true
    manufacturer/dataset generalization test once Sandia or another multi-chemistry dataset lands.
    """
    datasets = sorted(df["dataset"].unique())
    indices = np.arange(len(df))
    for dataset in datasets:
        test_mask = df["dataset"].to_numpy() == dataset
        yield indices[~test_mask], indices[test_mask]


def class_balance_by_fold(
    df: pd.DataFrame, splits: Iterator[tuple[np.ndarray, np.ndarray]]
) -> list[dict[str, object]]:
    """Summarize class balance for each train/test fold."""
    balances = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        balances.append(
            {
                "fold": fold,
                "train": df.iloc[train_idx]["chemistry"].value_counts().to_dict(),
                "test": df.iloc[test_idx]["chemistry"].value_counts().to_dict(),
            }
        )
    return balances
