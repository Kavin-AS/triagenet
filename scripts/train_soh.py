"""Train Phase 3 chemistry-aware SOH regressors with prediction intervals."""

from __future__ import annotations

import hashlib
import json
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from triagenet.config import DATA_PROCESSED, MODELS_DIR, REPO_ROOT
from triagenet.features.eol_cycle import CHEMISTRY_FEATURES, SOH_FEATURES, featurize_dataframe
from triagenet.models.chemistry import ChemistryClassifier
from triagenet.models.soh import GPRSOHModel, SOHEnsemble, XGBQuantileSOHModel
from triagenet.models.soh_calibration import (
    expected_calibration_error,
    mean_prediction_interval_width,
    prediction_interval_coverage_probability,
    reliability_diagram,
    sharpness_diagram,
)

REPORT_DIR = REPO_ROOT / "reports" / "phase3_soh"
ALPHA = 0.10
N_SPLITS = 5
GPR_MAX_TRAIN_SAMPLES = 300
TRAIN_CYCLES_PER_CELL = 40
CAPACITY_LEAKAGE_COLUMNS = {"feat_capacity_ratio", "discharge_capacity_ah", "charge_capacity_ah"}


def main() -> None:
    """Train SOH models, evaluate grouped calibration, and persist artifacts."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gpr", "xgb", "both"], default="both")
    parser.add_argument(
        "--strategy", choices=["per_chemistry", "shared", "ensemble"], default="ensemble"
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cycles_path = DATA_PROCESSED / "cycles.parquet"
    cycles = pd.read_parquet(cycles_path)
    model_cycles = sample_cycles_per_cell(cycles, TRAIN_CYCLES_PER_CELL)
    features = featurize_dataframe(model_cycles)
    validate_soh_feature_list()

    model_kinds = ["gpr", "xgb"] if args.model == "both" else [args.model]
    all_results: list[dict[str, Any]] = []
    saved_models: dict[str, str] = {}

    for model_kind in model_kinds:
        if args.strategy in {"per_chemistry", "ensemble"}:
            per_models = train_final_per_chemistry_models(features, model_kind)
            saved_models.update(save_per_chemistry_models(per_models, model_kind))
            ensemble = SOHEnsemble(
                per_models, chemistry_classifier=_load_optional_chemistry_model()
            )
            ensemble_path = MODELS_DIR / f"soh_{model_kind}_ensemble.joblib"
            ensemble.save(ensemble_path)
            saved_models[f"{model_kind}_ensemble"] = str(ensemble_path)
            if model_kind == "gpr":
                ensemble.save(MODELS_DIR / "soh_gpr.joblib")
                ensemble.save(MODELS_DIR / "soh_ensemble.joblib")
                saved_models["soh_gpr"] = str(MODELS_DIR / "soh_gpr.joblib")
                saved_models["soh_ensemble"] = str(MODELS_DIR / "soh_ensemble.joblib")
            if model_kind == "xgb":
                ensemble.save(MODELS_DIR / "soh_xgb_quantile.joblib")
                saved_models["soh_xgb_quantile"] = str(MODELS_DIR / "soh_xgb_quantile.joblib")
            all_results.extend(evaluate_per_chemistry(features, model_kind))
            all_results.extend(evaluate_ensemble(features, model_kind))
        if args.strategy == "shared":
            shared = train_final_shared_model(features, model_kind)
            path = MODELS_DIR / f"soh_{model_kind}_shared.joblib"
            shared.save(path)
            saved_models[f"{model_kind}_shared"] = str(path)
            all_results.extend(evaluate_shared(features, model_kind))
        elif args.strategy == "ensemble":
            shared = train_final_shared_model(features, model_kind)
            path = MODELS_DIR / f"soh_{model_kind}_shared.joblib"
            shared.save(path)
            saved_models[f"{model_kind}_shared"] = str(path)
            all_results.extend(evaluate_shared(features, model_kind))

    metrics = pd.DataFrame(all_results)
    metrics.to_csv(REPORT_DIR / "metrics.csv", index=False)
    write_metadata(cycles_path, features, metrics, saved_models)
    write_phase3_notes(metrics)
    write_notebook()
    print(
        metrics[["scope", "chemistry", "model", "rmse", "mae", "r2", "picp90", "mean_width", "ece"]]
    )


def validate_soh_feature_list() -> None:
    """Raise if any direct capacity leakage feature enters the SOH feature set."""
    leakage = CAPACITY_LEAKAGE_COLUMNS.intersection(SOH_FEATURES)
    if leakage:
        raise ValueError(
            f"SOH_FEATURES contains direct capacity leakage columns: {sorted(leakage)}"
        )


def sample_cycles_per_cell(df: pd.DataFrame, max_cycles: int) -> pd.DataFrame:
    """Return deterministic SOH-stratified cycle rows per cell across the full SOH range."""
    sampled = []
    for _, group in df.sort_values(["cell_id", "soh", "cycle_index"]).groupby("cell_id"):
        if len(group) <= max_cycles:
            sampled.append(group)
            continue
        positions = np.linspace(0, len(group) - 1, max_cycles).round().astype(int)
        sampled.append(group.iloc[np.unique(positions)])
    return pd.concat(sampled, ignore_index=True)


def train_final_per_chemistry_models(
    features: pd.DataFrame, model_kind: str
) -> dict[str, GPRSOHModel | XGBQuantileSOHModel]:
    """Train one final SOH model per chemistry on all sampled cycles."""
    models = {}
    for chemistry, group in features.groupby("chemistry"):
        model = make_model(model_kind, list(SOH_FEATURES))
        model.fit(group[list(SOH_FEATURES)], group["soh"])
        models[str(chemistry)] = model
    return models


def save_per_chemistry_models(
    models: dict[str, GPRSOHModel | XGBQuantileSOHModel], model_kind: str
) -> dict[str, str]:
    """Persist each per-chemistry SOH model and return artifact paths."""
    paths = {}
    for chemistry, model in models.items():
        path = MODELS_DIR / f"soh_{model_kind}_{chemistry.lower()}.joblib"
        model.save(path)
        paths[f"{model_kind}_{chemistry.lower()}"] = str(path)
    return paths


def train_final_shared_model(
    features: pd.DataFrame, model_kind: str
) -> GPRSOHModel | XGBQuantileSOHModel:
    """Train a final shared baseline model using chemistry probabilities as extra features."""
    shared = add_oracle_chemistry_probabilities(features)
    names = list(SOH_FEATURES) + ["prob_LCO", "prob_LFP"]
    model = make_model(model_kind, names)
    model.fit(shared[names], shared["soh"])
    return model


def evaluate_per_chemistry(features: pd.DataFrame, model_kind: str) -> list[dict[str, Any]]:
    """Evaluate per-chemistry SOH models with grouped cell folds."""
    rows = []
    for chemistry, group in features.groupby("chemistry"):
        predictions = []
        for train_idx, test_idx in group_splits(group):
            train = group.iloc[train_idx]
            test = group.iloc[test_idx]
            model = make_model(model_kind, list(SOH_FEATURES)).fit(
                train[list(SOH_FEATURES)], train["soh"]
            )
            predictions.append(predict_frame(model, test, chemistry=str(chemistry)))
        pred_df = pd.concat(predictions, ignore_index=True)
        rows.append(
            score_predictions(
                pred_df, scope="per_chemistry", model=model_kind, chemistry=str(chemistry)
            )
        )
        save_prediction_plots(pred_df, f"per_chemistry_{model_kind}_{str(chemistry).lower()}")
    return rows


def evaluate_shared(features: pd.DataFrame, model_kind: str) -> list[dict[str, Any]]:
    """Evaluate a shared SOH baseline with oracle chemistry probabilities as inputs."""
    shared = add_oracle_chemistry_probabilities(features)
    names = list(SOH_FEATURES) + ["prob_LCO", "prob_LFP"]
    predictions = []
    for train_idx, test_idx in group_splits(shared):
        train = shared.iloc[train_idx]
        test = shared.iloc[test_idx]
        model = make_model(model_kind, names).fit(train[names], train["soh"])
        predictions.append(predict_frame(model, test, chemistry="all"))
    pred_df = pd.concat(predictions, ignore_index=True)
    save_prediction_plots(pred_df, f"shared_{model_kind}")
    return [score_predictions(pred_df, scope="shared", model=model_kind, chemistry="all")]


def evaluate_ensemble(features: pd.DataFrame, model_kind: str) -> list[dict[str, Any]]:
    """Evaluate per-chemistry models mixed by fold-local chemistry probabilities."""
    predictions = []
    for train_idx, test_idx in group_splits(features):
        train = features.iloc[train_idx]
        test = features.iloc[test_idx]
        models = {}
        for chemistry, group in train.groupby("chemistry"):
            models[str(chemistry)] = make_model(model_kind, list(SOH_FEATURES)).fit(
                group[list(SOH_FEATURES)], group["soh"]
            )
        chemistry_probs = fold_chemistry_probabilities(train, test)
        ensemble = SOHEnsemble(models)
        mean, lower, upper, std = ensemble.predict(test[list(SOH_FEATURES)], chemistry_probs)
        predictions.append(base_prediction_frame(test, "all", mean, lower, upper, std))
    pred_df = pd.concat(predictions, ignore_index=True)
    save_prediction_plots(pred_df, f"ensemble_{model_kind}")
    return [score_predictions(pred_df, scope="ensemble", model=model_kind, chemistry="all")]


def fold_chemistry_probabilities(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Return fold-local chemistry probabilities without training on held-out cells."""
    clf = ChemistryClassifier(feature_names=list(CHEMISTRY_FEATURES)).fit(
        train[list(CHEMISTRY_FEATURES)], train["chemistry"], train["cell_id"]
    )
    proba = clf.predict_proba(test[list(CHEMISTRY_FEATURES)])
    return pd.DataFrame(proba, columns=clf.classes_, index=test.index)


def add_oracle_chemistry_probabilities(features: pd.DataFrame) -> pd.DataFrame:
    """Add one-hot chemistry probabilities for the shared-model baseline."""
    shared = features.copy()
    for chemistry in ("LCO", "LFP"):
        shared[f"prob_{chemistry}"] = (shared["chemistry"] == chemistry).astype(float)
    return shared


def group_splits(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return GroupKFold splits by cell, using five folds as a practical GPR trade-off."""
    groups = df["cell_id"].to_numpy()
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Need at least two cells for SOH grouped evaluation")
    return list(GroupKFold(n_splits=n_splits).split(df, groups=groups))


def make_model(model_kind: str, feature_names: list[str]) -> GPRSOHModel | XGBQuantileSOHModel:
    """Construct one SOH model with the fixed Phase 3 hyperparameters."""
    if model_kind == "gpr":
        return GPRSOHModel(feature_names, max_train_samples=GPR_MAX_TRAIN_SAMPLES)
    if model_kind == "xgb":
        return XGBQuantileSOHModel(feature_names)
    raise ValueError(f"Unknown SOH model kind: {model_kind}")


def predict_frame(
    model: GPRSOHModel | XGBQuantileSOHModel, test: pd.DataFrame, chemistry: str
) -> pd.DataFrame:
    """Return a standard held-out prediction frame for one model and fold."""
    mean, lower, upper = model.predict_interval(test[model.feature_names], alpha=ALPHA)
    std = (upper - lower) / (2.0 * 1.6448536269514722)
    return base_prediction_frame(test, chemistry, mean, lower, upper, std)


def base_prediction_frame(
    test: pd.DataFrame,
    chemistry: str,
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    std: np.ndarray,
) -> pd.DataFrame:
    """Build a standard prediction dataframe with identifiers and SOH intervals."""
    return pd.DataFrame(
        {
            "cell_id": test["cell_id"].to_numpy(),
            "cycle_index": test["cycle_index"].to_numpy(),
            "chemistry": test["chemistry"].to_numpy(),
            "model_chemistry": chemistry,
            "soh_true": test["soh"].to_numpy(dtype=float),
            "soh_mean": mean,
            "soh_lower_90": lower,
            "soh_upper_90": upper,
            "soh_std": std,
        }
    )


def score_predictions(
    predictions: pd.DataFrame, scope: str, model: str, chemistry: str
) -> dict[str, Any]:
    """Compute point and interval metrics for one prediction dataframe."""
    y_true = predictions["soh_true"].to_numpy()
    mean = predictions["soh_mean"].to_numpy()
    lower = predictions["soh_lower_90"].to_numpy()
    upper = predictions["soh_upper_90"].to_numpy()
    return {
        "scope": scope,
        "chemistry": chemistry,
        "model": model,
        "rmse": float(np.sqrt(mean_squared_error(y_true, mean))),
        "mae": float(mean_absolute_error(y_true, mean)),
        "r2": float(r2_score(y_true, mean)),
        "picp90": prediction_interval_coverage_probability(y_true, lower, upper),
        "mean_width": mean_prediction_interval_width(lower, upper),
        "ece": expected_calibration_error(y_true, (mean, lower, upper), ALPHA),
        "n": int(len(predictions)),
    }


def save_prediction_plots(predictions: pd.DataFrame, stem: str) -> None:
    """Save reliability, sharpness, and interval scatter plots for held-out predictions."""
    y_true = predictions["soh_true"].to_numpy()
    mean = predictions["soh_mean"].to_numpy()
    lower = predictions["soh_lower_90"].to_numpy()
    upper = predictions["soh_upper_90"].to_numpy()
    std = predictions["soh_std"].to_numpy()
    predictions.to_csv(REPORT_DIR / f"{stem}_predictions.csv", index=False)
    reliability_diagram(y_true, mean, std, REPORT_DIR / f"{stem}_reliability.png")
    sharpness_diagram(std, REPORT_DIR / f"{stem}_sharpness.png")
    order = np.argsort(y_true)
    selected = order[:: max(1, len(order) // 200)]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.errorbar(
        y_true[selected],
        mean[selected],
        yerr=[mean[selected] - lower[selected], upper[selected] - mean[selected]],
        fmt="o",
        alpha=0.45,
        markersize=2,
    )
    ax.plot([0, 1.05], [0, 1.05], "--", color="gray")
    ax.set_xlabel("Measured SOH")
    ax.set_ylabel("Predicted SOH with 90% interval")
    ax.set_title(stem)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / f"{stem}_scatter_interval.png", dpi=160)
    plt.close(fig)


def _load_optional_chemistry_model() -> object | None:
    path = MODELS_DIR / "chemistry_xgb.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def write_metadata(
    cycles_path: Path,
    features: pd.DataFrame,
    metrics: pd.DataFrame,
    saved_models: dict[str, str],
) -> None:
    """Write Phase 3 model metadata with metrics and reproducibility details."""
    metadata = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "target": "soh",
        "methodology": "Predict SOH from shape features without direct capacity inputs.",
        "feature_names": list(SOH_FEATURES),
        "excluded_capacity_features": sorted(CAPACITY_LEAKAGE_COLUMNS),
        "training_cell_ids": sorted(features["cell_id"].unique().tolist()),
        "metrics": metrics.to_dict(orient="records"),
        "artifacts": saved_models,
        "dataset_hashes": {"cycles_parquet_sha256": sha256(cycles_path)},
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    (MODELS_DIR / "soh_metadata.json").write_text(json.dumps(metadata, indent=2))


def write_phase3_notes(metrics: pd.DataFrame) -> None:
    """Write concise Phase 3 notes with actual metrics and caveats."""
    table = metrics[
        ["chemistry", "scope", "model", "rmse", "mae", "r2", "picp90", "mean_width", "ece"]
    ].copy()
    disagreement = find_disagreement_case()
    notes = [
        "# Phase 3 Notes",
        "",
        "## Design Decisions",
        "",
        (
            "SOH is measured as discharge capacity divided by nominal capacity, clipped to "
            "`[0, 1.05]`. For this phase, capacity-derived inputs are excluded because they "
            "are the target. The model is intentionally asked to infer SOH from voltage "
            "shape, dQ/dV shape, protocol, and chemistry probabilities without seeing "
            "`feat_capacity_ratio`, `discharge_capacity_ah`, or `charge_capacity_ah`."
        ),
        "",
        (
            "The primary architecture is per-chemistry SOH modeling with a "
            "probability-weighted ensemble. At inference, chemistry probabilities weight "
            "the per-chemistry SOH distributions and the ensemble variance uses the law "
            "of total variance, so chemistry uncertainty propagates into SOH uncertainty."
        ),
        "",
        "Both GPR and XGBoost-Quantile are trained. GPR is the principled uncertainty model; "
        "XGBoost-Quantile is the scalable deployment baseline.",
        "",
        "## Headline Metrics",
        "",
        markdown_table(table),
        "",
        "## Calibration Commentary",
        "",
        calibration_commentary(metrics),
        "",
        "## Disagreement Case",
        "",
        disagreement,
        "",
        "## Caveats",
        "",
        (
            "The same Phase 2.5 limitations carry forward: only LCO and LFP are present, "
            "each chemistry comes from one source, and c-rate/protocol signatures remain "
            f"confounded with source. GPR uses stratified subsampling capped at "
            f"{GPR_MAX_TRAIN_SAMPLES} training cycles per fold to keep cubic-time fitting "
            "practical."
        ),
        "",
        "## Phase 4 Handoff",
        "",
        "Phase 4 needs `(soh_mean, soh_lower_90, soh_upper_90, chemistry_probs)`. "
        "The `SOHEnsemble.predict` method emits `(mean, lower, upper, std)`, and the chemistry "
        "classifier provides the probability vector.",
        "",
    ]
    (REPO_ROOT / "PHASE_3_NOTES.md").write_text("\n".join(notes))


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small dataframe as a markdown table without optional tabulate dependency."""
    formatted = frame.copy()
    for column in ("rmse", "mae", "r2", "picp90", "mean_width", "ece"):
        formatted[column] = formatted[column].map(lambda value: f"{float(value):.4f}")
    headers = list(formatted.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def calibration_commentary(metrics: pd.DataFrame) -> str:
    """Summarize interval calibration from actual metrics."""
    picp = metrics["picp90"].mean()
    if picp < 0.85:
        verdict = "over-confident: 90% intervals contain too little held-out truth."
    elif picp > 0.95:
        verdict = "under-confident: intervals are wider than needed."
    else:
        verdict = "reasonably calibrated: average PICP@90 is inside the 0.85-0.95 target band."
    return f"Average PICP@90 is {picp:.3f}, so the current SOH interval behavior is {verdict}"


def find_disagreement_case() -> str:
    """Find a held-out prediction differing from measured SOH by more than five points."""
    candidates = sorted(REPORT_DIR.glob("*_predictions.csv"))
    for path in candidates:
        frame = pd.read_csv(path)
        frame["abs_error"] = (frame["soh_mean"] - frame["soh_true"]).abs()
        large = frame[frame["abs_error"] > 0.05].sort_values("abs_error", ascending=False)
        if not large.empty:
            row = large.iloc[0]
            return (
                f"Found in `{path.name}`: cell `{row.cell_id}` cycle {int(row.cycle_index)} "
                f"measured SOH {row.soh_true:.3f}, predicted {row.soh_mean:.3f}, "
                f"90% interval [{row.soh_lower_90:.3f}, {row.soh_upper_90:.3f}]."
            )
    return "No held-out prediction differed from measured SOH by more than five percentage points."


def write_notebook() -> None:
    """Write a lightweight Phase 3 diagnostics notebook that loads generated artifacts."""
    content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 3 SOH Regressor Diagnostics\n",
                    "\n",
                    "This notebook inspects the SOH models trained by `scripts/train_soh.py`.",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "from triagenet.config import REPO_ROOT, DATA_PROCESSED\n",
                    "report_dir = REPO_ROOT / 'reports' / 'phase3_soh'\n",
                    "metrics = pd.read_csv(report_dir / 'metrics.csv')\n",
                    "display(metrics)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "pred = pd.read_csv(report_dir / 'ensemble_gpr_predictions.csv')\n",
                    "display(\n",
                    "    pred.assign(abs_error=(pred.soh_mean - pred.soh_true).abs())\n",
                    "    .sort_values('abs_error', ascending=False)\n",
                    "    .head()\n",
                    ")\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "for image in sorted(report_dir.glob('*_scatter_interval.png'))[:6]:\n",
                    "    plt.figure(figsize=(6, 4))\n",
                    "    plt.imshow(plt.imread(image))\n",
                    "    plt.title(image.name)\n",
                    "    plt.axis('off')\n",
                    "    plt.show()\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## What I observed\n",
                    "\n",
                    (
                        "The model predicts SOH without direct capacity features, so the "
                        "useful diagnostic is calibration: whether the 90% intervals contain "
                        "roughly 90% of held-out measured SOH. Any disagreement cases above "
                        "five percentage points should be inspected as possible "
                        "partial-discharge, temperature, or protocol artifacts."
                    ),
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (REPO_ROOT / "notebooks" / "03_soh_regressor.ipynb").write_text(json.dumps(content, indent=2))


def sha256(path: Path) -> str:
    """Return SHA256 hash for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
