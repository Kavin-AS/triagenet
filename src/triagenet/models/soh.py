"""SOH regression models with calibrated uncertainty for single-cycle features."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class GPRSOHModel:
    """Gaussian-process SOH regressor that always returns mean and uncertainty."""

    def __init__(
        self,
        feature_names: list[str],
        random_state: int = 42,
        max_train_samples: int = 800,
        n_restarts_optimizer: int = 2,
    ) -> None:
        self.feature_names = feature_names
        self.random_state = random_state
        self.max_train_samples = max_train_samples
        self.n_restarts_optimizer = n_restarts_optimizer
        self.model = _gpr_pipeline(random_state, n_restarts_optimizer)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> Self:
        """Fit the GPR, stratifying any subsample across SOH deciles for coverage."""
        X_fit, y_fit = _stratified_soh_sample(
            X[self.feature_names],
            np.asarray(y, dtype=float),
            self.max_train_samples,
            self.random_state,
        )
        self.model.fit(X_fit, y_fit)
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return predictive SOH mean and standard deviation."""
        mean, std = self.model.predict(X[self.feature_names], return_std=True)
        return np.clip(mean, 0.0, 1.05), np.maximum(std, 1e-6)

    def predict_interval(
        self, X: pd.DataFrame, alpha: float = 0.10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return Gaussian prediction intervals; default is the central 90% interval."""
        mean, std = self.predict(X)
        z_score = float(norm.ppf(1.0 - alpha / 2.0))
        lower = np.clip(mean - z_score * std, 0.0, 1.05)
        upper = np.clip(mean + z_score * std, 0.0, 1.05)
        return mean, lower, upper

    def save(self, path: Path) -> None:
        """Persist this GPR model with joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a persisted GPR SOH model."""
        return joblib.load(path)


class XGBQuantileSOHModel:
    """XGBoost quantile SOH regressor with post-hoc quantile crossing repair."""

    def __init__(
        self,
        feature_names: list[str],
        random_state: int = 42,
        quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95),
    ) -> None:
        self.feature_names = feature_names
        self.random_state = random_state
        self.quantiles = quantiles
        self.models = {
            quantile: _xgb_quantile_pipeline(random_state, quantile) for quantile in quantiles
        }
        self.interval_adjustment_ = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> Self:
        """Fit independent quantile regressors with a held-out conformal interval adjustment."""
        target = np.asarray(y, dtype=float)
        if len(target) >= 80:
            bins = pd.qcut(target, q=min(5, len(np.unique(target))), duplicates="drop")
            train_idx, cal_idx = train_test_split(
                np.arange(len(target)),
                test_size=0.2,
                random_state=self.random_state,
                stratify=bins,
            )
        else:
            train_idx = np.arange(len(target))
            cal_idx = np.arange(len(target))
        for model in self.models.values():
            model.fit(X.iloc[train_idx][self.feature_names], target[train_idx])
        _, lower, upper = self.predict(X.iloc[cal_idx], apply_adjustment=False)
        conformity = np.maximum.reduce(
            [lower - target[cal_idx], target[cal_idx] - upper, np.zeros(len(cal_idx))]
        )
        # Split-conformal expansion repairs over-confident independent quantile models
        # without changing median predictions.
        self.interval_adjustment_ = float(np.quantile(conformity, 0.90))
        return self

    def predict(
        self, X: pd.DataFrame, apply_adjustment: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return median, lower 90%, and upper 90% SOH predictions."""
        predictions = np.vstack(
            [self.models[quantile].predict(X[self.feature_names]) for quantile in self.quantiles]
        ).T
        # XGBoost quantile models are independent, so quantile crossing can occur. Sorting is
        # the standard post-hoc repair for lower <= median <= upper consistency.
        repaired = np.sort(predictions, axis=1)
        lower, median, upper = repaired[:, 0], repaired[:, 1], repaired[:, 2]
        if apply_adjustment:
            lower = lower - self.interval_adjustment_
            upper = upper + self.interval_adjustment_
        return np.clip(median, 0.0, 1.05), np.clip(lower, 0.0, 1.05), np.clip(upper, 0.0, 1.05)

    def predict_interval(
        self, X: pd.DataFrame, alpha: float = 0.10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the trained 90% quantile interval; other alpha values are unsupported."""
        if abs(alpha - 0.10) > 1e-9:
            raise ValueError("XGBQuantileSOHModel currently supports only alpha=0.10")
        return self.predict(X)

    def save(self, path: Path) -> None:
        """Persist this quantile model bundle with joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a persisted quantile SOH model."""
        return joblib.load(path)


class SOHEnsemble:
    """Probability-weighted per-chemistry SOH ensemble using total-variance math.

    For chemistry probabilities p(c), per-chemistry means m_c, and variances v_c, the
    mixture mean is sum_c p(c)m_c and the mixture variance is
    sum_c p(c)(v_c + m_c^2) - mean^2. This is the law of total variance for a Gaussian
    mixture approximation, and it propagates chemistry uncertainty into SOH uncertainty.
    """

    def __init__(
        self,
        models_by_chemistry: dict[str, GPRSOHModel | XGBQuantileSOHModel],
        chemistry_classifier: object | None = None,
    ) -> None:
        self.models_by_chemistry = models_by_chemistry
        self.chemistry_classifier = chemistry_classifier

    def predict(
        self, X: pd.DataFrame, chemistry_probs: pd.DataFrame, alpha: float = 0.10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return mixture SOH mean, lower interval, upper interval, and standard deviation."""
        means = []
        variances = []
        weights = []
        z_score = float(norm.ppf(1.0 - alpha / 2.0))
        for chemistry, model in self.models_by_chemistry.items():
            if chemistry not in chemistry_probs:
                continue
            mean, lower, upper = model.predict_interval(X, alpha=alpha)
            std = np.maximum((upper - lower) / (2.0 * z_score), 1e-6)
            means.append(mean)
            variances.append(std**2)
            weights.append(chemistry_probs[chemistry].to_numpy(dtype=float))
        if not means:
            raise ValueError("No ensemble chemistry probabilities matched available SOH models")
        mean_matrix = np.vstack(means)
        variance_matrix = np.vstack(variances)
        weight_matrix = np.vstack(weights)
        weight_matrix = weight_matrix / np.maximum(weight_matrix.sum(axis=0, keepdims=True), 1e-12)
        mixture_mean = np.sum(weight_matrix * mean_matrix, axis=0)
        second_moment = np.sum(weight_matrix * (variance_matrix + mean_matrix**2), axis=0)
        mixture_std = np.sqrt(np.maximum(second_moment - mixture_mean**2, 1e-12))
        lower = np.clip(mixture_mean - z_score * mixture_std, 0.0, 1.05)
        upper = np.clip(mixture_mean + z_score * mixture_std, 0.0, 1.05)
        return np.clip(mixture_mean, 0.0, 1.05), lower, upper, mixture_std

    def save(self, path: Path) -> None:
        """Persist this SOH ensemble bundle with joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a persisted SOH ensemble."""
        return joblib.load(path)


def _gpr_pipeline(random_state: int, n_restarts_optimizer: int) -> Pipeline:
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "gpr",
                GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    n_restarts_optimizer=n_restarts_optimizer,
                    normalize_y=True,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _xgb_quantile_pipeline(random_state: int, quantile: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "xgb",
                XGBRegressor(
                    objective="reg:quantileerror",
                    quantile_alpha=quantile,
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=random_state,
                    n_jobs=4,
                ),
            ),
        ]
    )


def _stratified_soh_sample(
    X: pd.DataFrame, y: np.ndarray, max_samples: int, random_state: int
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(y) <= max_samples:
        return X, y
    rng = np.random.default_rng(random_state)
    bins = pd.qcut(y, q=min(10, len(np.unique(y))), duplicates="drop")
    sampled_indices = []
    per_bin = max(1, int(np.ceil(max_samples / len(pd.unique(bins)))))
    for _, group_indices in pd.Series(np.arange(len(y))).groupby(bins, observed=False):
        chosen = rng.choice(
            group_indices.to_numpy(),
            size=min(per_bin, len(group_indices)),
            replace=False,
        )
        sampled_indices.extend(chosen.tolist())
    if len(sampled_indices) > max_samples:
        sampled_indices = rng.choice(sampled_indices, size=max_samples, replace=False).tolist()
    sampled_indices = np.asarray(sorted(sampled_indices), dtype=int)
    return X.iloc[sampled_indices], y[sampled_indices]
