"""Probe chemistry-classifier robustness to additive voltage offsets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from train_chemistry import CYCLES_PER_CELL, SOH_WINDOW, sample_cycles_per_cell
from triagenet.config import DATA_PROCESSED, REPO_ROOT
from triagenet.features.eol_cycle import (
    ABSOLUTE_VOLTAGE_FEATURES,
    SHAPE_FEATURES,
    featurize_dataframe,
)
from triagenet.models.chemistry import ChemistryClassifier
from triagenet.models.data_split import build_chemistry_dataset, leave_one_cell_out_split

REPORT_DIR = REPO_ROOT / "reports" / "phase2_chemistry"
SHIFTS = (-0.2, -0.1, 0.0, 0.1, 0.2)


def main() -> None:
    """Run leave-one-cell-out voltage-shift robustness probes and persist results."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cycles = pd.read_parquet(DATA_PROCESSED / "cycles.parquet")
    matched = sample_cycles_per_cell(build_chemistry_dataset(cycles, SOH_WINDOW), CYCLES_PER_CELL)
    base_features = featurize_dataframe(matched)

    results = {
        "shape": _probe_feature_set(matched, base_features, list(SHAPE_FEATURES)),
        "absolute_voltage": _probe_feature_set(
            matched, base_features, list(ABSOLUTE_VOLTAGE_FEATURES)
        ),
    }
    output = {
        "soh_window": SOH_WINDOW,
        "cycles_per_cell_cap": CYCLES_PER_CELL,
        "shifts_v": list(SHIFTS),
        "results": results,
    }
    (REPORT_DIR / "voltage_robustness.json").write_text(json.dumps(output, indent=2))
    _plot_results(results, REPORT_DIR / "voltage_robustness.png")
    print(json.dumps(output, indent=2))


def _probe_feature_set(
    cycles: pd.DataFrame, base_features: pd.DataFrame, feature_names: list[str]
) -> dict[str, float]:
    scores = {}
    splits = list(leave_one_cell_out_split(base_features))
    for shift in SHIFTS:
        shifted_features = featurize_dataframe(_shift_voltage(cycles, shift))
        fold_scores = []
        for train_idx, test_idx in splits:
            y_train = base_features.iloc[train_idx]["chemistry"]
            y_test = base_features.iloc[test_idx]["chemistry"]
            clf = ChemistryClassifier(feature_names=feature_names).fit(
                base_features.iloc[train_idx][feature_names],
                y_train,
                base_features.iloc[train_idx]["cell_id"],
            )
            predicted = clf.predict(shifted_features.iloc[test_idx][feature_names])
            fold_scores.append(float(balanced_accuracy_score(y_test, predicted)))
        scores[f"{shift:+.1f}"] = float(np.mean(fold_scores))
    return scores


def _shift_voltage(cycles: pd.DataFrame, shift_v: float) -> pd.DataFrame:
    shifted = cycles.copy()
    shifted["voltage_curve"] = shifted["voltage_curve"].map(
        lambda curve: (np.asarray(curve, dtype=float) + shift_v).tolist()
    )
    return shifted


def _plot_results(results: dict[str, dict[str, float]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    shifts = np.asarray(SHIFTS, dtype=float)
    for label, values in results.items():
        ax.plot(shifts, [values[f"{shift:+.1f}"] for shift in SHIFTS], marker="o", label=label)
    ax.set_xlabel("Voltage shift applied to held-out test cycles (V)")
    ax.set_ylabel("Mean leave-one-cell-out balanced accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.set_title("Voltage-shift robustness")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
