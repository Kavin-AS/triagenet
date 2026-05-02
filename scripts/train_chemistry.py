"""Train and evaluate the Phase 2 chemistry classifier."""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from triagenet.config import DATA_PROCESSED, MODELS_DIR, REPO_ROOT
from triagenet.features.eol_cycle import (
    ABSOLUTE_VOLTAGE_FEATURES,
    SHAPE_FEATURES,
    featurize_dataframe,
)
from triagenet.models.chemistry import ChemistryClassifier
from triagenet.models.data_split import (
    build_chemistry_dataset,
    leave_one_cell_out_split,
    leave_one_dataset_out_split,
)

SOH_WINDOW = (0.80, 1.00)
REPORT_DIR = REPO_ROOT / "reports" / "phase2_chemistry"
CYCLES_PER_CELL = 30
FEATURE_SETS = {
    "shape": tuple(SHAPE_FEATURES),
    "voltage": tuple(ABSOLUTE_VOLTAGE_FEATURES),
}


def main() -> None:
    """Train the chemistry classifier and write model, metadata, and diagnostics."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default="shape")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cycles_path = DATA_PROCESSED / "cycles.parquet"
    cycles = pd.read_parquet(cycles_path)
    matched_cycles = build_chemistry_dataset(cycles, SOH_WINDOW)
    full_cycles = cycles.copy()
    matched_model_cycles = sample_cycles_per_cell(matched_cycles, CYCLES_PER_CELL)
    full_model_cycles = sample_cycles_per_cell(full_cycles, CYCLES_PER_CELL)

    matched_features = featurize_dataframe(matched_model_cycles)
    full_features = featurize_dataframe(full_model_cycles)
    active_feature_names = list(FEATURE_SETS[args.feature_set])
    active_result = train_feature_set(
        feature_set_name=args.feature_set,
        feature_names=active_feature_names,
        matched_features=matched_features,
        full_features=full_features,
        model_path=MODELS_DIR / "chemistry_xgb.joblib",
        plot_prefix="" if args.feature_set == "shape" else f"{args.feature_set}_",
    )

    baseline_result = None
    if args.feature_set == "shape":
        baseline_result = train_feature_set(
            feature_set_name="voltage",
            feature_names=list(ABSOLUTE_VOLTAGE_FEATURES),
            matched_features=matched_features,
            full_features=full_features,
            model_path=MODELS_DIR / "chemistry_xgb_voltage_axis.joblib",
            plot_prefix="voltage_axis_",
        )

    metadata = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "active_feature_set": args.feature_set,
        "soh_window": SOH_WINDOW,
        "cycles_per_cell_cap": CYCLES_PER_CELL,
        "cycle_cap_reason": (
            "Deterministic per-cell SOH-stratified cap prevents long-lived cells from "
            "dominating correlated cycle-level training rows."
        ),
        "feature_names": active_feature_names,
        "active_feature_names": active_feature_names,
        "deprecated_voltage_axis_feature_names": list(ABSOLUTE_VOLTAGE_FEATURES),
        "class_labels": active_result["model"].classes_.tolist(),
        "training_cell_ids": sorted(matched_features["cell_id"].unique().tolist()),
        "metrics": {
            "active": active_result["metrics"],
            "shape": active_result["metrics"] if args.feature_set == "shape" else None,
            "voltage_axis": (
                baseline_result["metrics"]
                if baseline_result is not None
                else active_result["metrics"] if args.feature_set == "voltage" else None
            ),
        },
        "dataset_hashes": {"cycles_parquet_sha256": sha256(cycles_path)},
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
        "feature_importances": active_result["feature_importance"],
        "voltage_axis_feature_importances": (
            baseline_result["feature_importance"] if baseline_result is not None else None
        ),
    }
    (MODELS_DIR / "chemistry_metadata.json").write_text(json.dumps(metadata, indent=2))

    print_run_summary(args.feature_set, active_result)
    if baseline_result is not None:
        print_run_summary("voltage", baseline_result)


def train_feature_set(
    feature_set_name: str,
    feature_names: list[str],
    matched_features: pd.DataFrame,
    full_features: pd.DataFrame,
    model_path: Path,
    plot_prefix: str,
) -> dict[str, Any]:
    """Train/evaluate one chemistry feature set and persist its model plus diagnostics."""
    X_matched = matched_features[feature_names]
    y_matched = matched_features["chemistry"]
    X_full = full_features[feature_names]
    y_full = full_features["chemistry"]
    output_stem = f"{feature_set_name}_"

    loco = evaluate_splits(
        f"{output_stem}leave_one_cell_out",
        matched_features,
        X_matched,
        y_matched,
        list(leave_one_cell_out_split(matched_features)),
        feature_names,
    )
    lodo = evaluate_splits(
        f"{output_stem}leave_one_dataset_out",
        matched_features,
        X_matched,
        y_matched,
        list(leave_one_dataset_out_split(matched_features)),
        feature_names,
    )
    naive = evaluate_splits(
        f"{output_stem}without_soh_window_leave_one_cell_out",
        full_features,
        X_full,
        y_full,
        list(leave_one_cell_out_split(full_features)),
        feature_names,
    )

    final_model = ChemistryClassifier(feature_names=feature_names).fit(
        X_matched, y_matched, matched_features["cell_id"]
    )
    final_model.save(model_path)
    feature_importance = final_model.feature_importances_
    plot_feature_importance(
        feature_importance, REPORT_DIR / f"{plot_prefix or ''}feature_importance.png"
    )

    pd.DataFrame(loco["predictions"]).to_csv(
        REPORT_DIR / f"{output_stem}leave_one_cell_predictions.csv", index=False
    )
    pd.DataFrame(naive["predictions"]).to_csv(
        REPORT_DIR / f"{output_stem}without_soh_window_predictions.csv", index=False
    )
    if lodo["predictions"]:
        pd.DataFrame(lodo["predictions"]).to_csv(
            REPORT_DIR / f"{output_stem}leave_one_dataset_predictions.csv", index=False
        )
    return {
        "model": final_model,
        "feature_importance": feature_importance,
        "metrics": {
            "leave_one_cell_out": loco["summary"],
            "leave_one_dataset_out": lodo["summary"],
            "without_soh_window_leave_one_cell_out": naive["summary"],
        },
    }


def print_run_summary(feature_set_name: str, result: dict[str, Any]) -> None:
    """Print a compact training summary for one feature-set result."""
    metrics = result["metrics"]
    print(f"\n=== Feature set: {feature_set_name} ===")
    print_metrics("Matched SOH leave-one-cell-out", metrics["leave_one_cell_out"])
    print_metrics("Leave-one-dataset-out", metrics["leave_one_dataset_out"])
    print_metrics(
        "Without SOH window leave-one-cell-out", metrics["without_soh_window_leave_one_cell_out"]
    )
    print("Top features:")
    for name, value in sorted(
        result["feature_importance"].items(), key=lambda item: item[1], reverse=True
    )[:5]:
        print(f"  {name}: {value:.4f}")


def sample_cycles_per_cell(df: pd.DataFrame, max_cycles: int) -> pd.DataFrame:
    """Return a deterministic SOH-stratified cycle cap per cell for model training rows."""
    sampled = []
    for _, group in df.sort_values(["cell_id", "soh", "cycle_index"]).groupby("cell_id"):
        if len(group) <= max_cycles:
            sampled.append(group)
            continue
        positions = np.linspace(0, len(group) - 1, max_cycles).round().astype(int)
        sampled.append(group.iloc[np.unique(positions)])
    return pd.concat(sampled, ignore_index=True)


def evaluate_splits(
    name: str,
    feature_frame: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    feature_names: list[str],
) -> dict[str, Any]:
    """Fit/evaluate a classifier across predefined folds and save diagnostics."""
    labels = sorted(y.unique().tolist())
    predictions: list[dict[str, Any]] = []
    fold_metrics = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        if y_train.nunique() < 2:
            fold_metrics.append(
                {
                    "fold": fold,
                    "status": "untrainable_single_class_train",
                    "train_classes": sorted(y_train.unique().tolist()),
                    "test_classes": sorted(y_test.unique().tolist()),
                }
            )
            continue
        clf = ChemistryClassifier(feature_names=feature_names).fit(
            X.iloc[train_idx], y_train, feature_frame.iloc[train_idx]["cell_id"]
        )
        pred = clf.predict(X.iloc[test_idx])
        proba = clf.predict_proba(X.iloc[test_idx])
        fold_metrics.append(_metrics_for_fold(fold, y_test.to_numpy(), pred, proba, clf.classes_))
        _plot_confusion(
            y_test.to_numpy(),
            pred,
            labels,
            REPORT_DIR / f"{name}_fold{fold}_confusion.png",
        )
        positive_class = clf.classes_[-1]
        positive_index = list(clf.classes_).index(positive_class)
        for local_i, row_index in enumerate(test_idx):
            record = feature_frame.iloc[row_index][
                ["cell_id", "cycle_index", "dataset", "chemistry", "soh"]
            ].to_dict()
            record.update(
                {
                    "fold": fold,
                    "split": name,
                    "predicted_chemistry": pred[local_i],
                    "prob_positive": float(proba[local_i, positive_index]),
                    "positive_class": positive_class,
                }
            )
            for class_index, class_label in enumerate(clf.classes_):
                record[f"prob_{class_label}"] = float(proba[local_i, class_index])
            predictions.append(record)
    summary = _summarize_metrics(fold_metrics)
    if predictions:
        plot_calibration(
            pd.DataFrame(predictions),
            REPORT_DIR / f"{name}_calibration.png",
        )
    return {"folds": fold_metrics, "summary": summary, "predictions": predictions}


def _metrics_for_fold(
    fold: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
) -> dict[str, Any]:
    metrics = {
        "fold": fold,
        "status": "ok",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if len(classes) == 2 and len(np.unique(y_true)) == 2:
        positive = classes[-1]
        positive_index = list(classes).index(positive)
        metrics["roc_auc"] = float(roc_auc_score(y_true == positive, proba[:, positive_index]))
    else:
        metrics["roc_auc"] = None
    return metrics


def _summarize_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [fold for fold in folds if fold.get("status") == "ok"]
    if not ok:
        return {"status": "untrainable", "folds": folds}
    summary = {"status": "ok", "n_folds": len(ok)}
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc"):
        values = [fold[key] for fold in ok if fold.get(key) is not None]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
        summary[f"std_{key}"] = float(np.std(values)) if values else None
    summary["folds"] = folds
    return summary


def _plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_path: Path
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_calibration(predictions: pd.DataFrame, output_path: Path) -> None:
    """Save a binary calibration plot from held-out prediction records."""
    if predictions.empty:
        return
    positive = predictions["positive_class"].iloc[0]
    y_true = (predictions["chemistry"] == positive).astype(float)
    bins = pd.cut(predictions["prob_positive"], bins=np.linspace(0, 1, 11), include_lowest=True)
    grouped = pd.DataFrame({"prob": predictions["prob_positive"], "truth": y_true}).groupby(
        bins, observed=False
    )
    observed = grouped["truth"].mean()
    predicted = grouped["prob"].mean()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(predicted, observed, marker="o")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Chemistry probability calibration")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(importances: dict[str, float], output_path: Path) -> None:
    """Save a global feature importance bar chart."""
    ordered = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    names = [name for name, _ in ordered]
    values = [value for _, value in ordered]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names[::-1], values[::-1])
    ax.set_xlabel("Mean XGBoost importance")
    ax.set_title("Chemistry classifier feature importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_metrics(title: str, summary: dict[str, Any]) -> None:
    """Print a compact metric summary."""
    print(f"\n{title}")
    if summary.get("status") != "ok":
        print(json.dumps(summary, indent=2))
        return
    for key in ("mean_accuracy", "mean_balanced_accuracy", "mean_macro_f1", "mean_roc_auc"):
        print(f"{key}: {summary.get(key)}")


def sha256(path: Path) -> str:
    """Return the SHA256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
