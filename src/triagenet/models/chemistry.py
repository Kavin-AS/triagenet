"""Chemistry classifier wrapper with calibrated XGBoost probabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


class ChemistryClassifier:
    """Calibrated chemistry classifier for single-cycle TriageNet features."""

    def __init__(self, feature_names: list[str], random_state: int = 42) -> None:
        self.feature_names = feature_names
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.model: CalibratedClassifierCV | None = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray, cell_ids: pd.Series | None = None
    ) -> Self:
        """Fit the calibrated XGBoost pipeline; ``cell_ids`` is accepted for audit logging."""
        del cell_ids
        y_encoded = self.label_encoder.fit_transform(np.asarray(y))
        classes, counts = np.unique(y_encoded, return_counts=True)
        if classes.size < 2:
            raise ValueError("ChemistryClassifier requires at least two classes")
        scale_pos_weight = 1.0
        if classes.size == 2:
            negative_count = counts[classes == 0][0]
            positive_count = counts[classes == 1][0]
            scale_pos_weight = float(negative_count / max(positive_count, 1))
        self.model = CalibratedClassifierCV(
            estimator=_pipeline(self.random_state, scale_pos_weight),
            method="isotonic",
            cv=3,
        )
        self.model.fit(X[self.feature_names], y_encoded)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted chemistry labels."""
        model = self._require_model()
        encoded = model.predict(X[self.feature_names])
        return self.label_encoder.inverse_transform(encoded.astype(int))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated class probabilities in ``classes_`` order."""
        model = self._require_model()
        return model.predict_proba(X[self.feature_names])

    @property
    def classes_(self) -> np.ndarray:
        """Return class labels in probability column order."""
        return self.label_encoder.classes_

    @property
    def feature_importances_(self) -> dict[str, float]:
        """Return mean XGBoost feature importances named by feature."""
        model = self._require_model()
        importances = []
        for calibrated in model.calibrated_classifiers_:
            estimator = calibrated.estimator
            clf = estimator.named_steps["clf"]
            importances.append(np.asarray(clf.feature_importances_, dtype=float))
        if not importances:
            return dict.fromkeys(self.feature_names, 0.0)
        mean_importance = np.mean(np.vstack(importances), axis=0)
        return {
            name: float(value)
            for name, value in zip(self.feature_names, mean_importance, strict=True)
        }

    def top_features(
        self, X_row: pd.Series | pd.DataFrame, k: int = 5
    ) -> list[tuple[str, float, float]]:
        """Return a simple local ranking using global importance times feature value."""
        row = X_row.iloc[0] if isinstance(X_row, pd.DataFrame) else X_row
        importances = self.feature_importances_
        ranked = []
        for name in self.feature_names:
            value = float(row[name])
            ranked.append((name, value, value * importances.get(name, 0.0)))
        return sorted(ranked, key=lambda item: abs(item[2]), reverse=True)[:k]

    def save(self, path: Path) -> None:
        """Persist this classifier with joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a persisted classifier."""
        return joblib.load(path)

    def _require_model(self) -> CalibratedClassifierCV:
        if self.model is None:
            raise RuntimeError("ChemistryClassifier is not fitted")
        return self.model


def _pipeline(random_state: int, scale_pos_weight: float) -> Pipeline:
    # Conservative XGBoost settings: shallow trees, shrinkage, subsampling, and L2 regularization
    # reduce overfit risk on the current small number of independent cells.
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=4,
                    eval_metric="logloss",
                ),
            ),
        ]
    )
