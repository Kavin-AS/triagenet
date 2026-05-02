"""Tests for Phase 3 SOH models, calibration, and leakage guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
from triagenet.features.eol_cycle import SOH_FEATURES
from triagenet.models.soh import GPRSOHModel, SOHEnsemble, XGBQuantileSOHModel
from triagenet.models.soh_calibration import prediction_interval_coverage_probability


def _linear_data(n: int = 80) -> tuple[pd.DataFrame, np.ndarray]:
    x = np.linspace(0, 1, n)
    frame = pd.DataFrame({"feat_x": x, "feat_extra": np.sin(x)})
    y = 0.2 + 0.6 * x
    return frame, y


def test_gpr_and_xgb_recover_synthetic_linear_soh() -> None:
    """Both SOH regressors recover a simple leakage-free synthetic target."""
    X, y = _linear_data()
    train = np.arange(0, 80, 2)
    test = np.arange(1, 80, 2)
    gpr = GPRSOHModel(["feat_x", "feat_extra"], max_train_samples=60, n_restarts_optimizer=0)
    gpr.fit(X.iloc[train], y[train])
    gpr_mean, _, _ = gpr.predict_interval(X.iloc[test])
    assert float(np.sqrt(np.mean((gpr_mean - y[test]) ** 2))) < 0.01

    xgb = XGBQuantileSOHModel(["feat_x", "feat_extra"])
    xgb.fit(X.iloc[train], y[train])
    xgb_mean, _, _ = xgb.predict_interval(X.iloc[test])
    assert float(np.sqrt(np.mean((xgb_mean - y[test]) ** 2))) < 0.08


def test_calibration_coverage_with_known_noise_interval() -> None:
    """A synthetic Gaussian 90% interval has empirical coverage near 90%."""
    rng = np.random.default_rng(42)
    y_true = rng.normal(0.5, 0.1, 10_000)
    lower = 0.5 - 1.6448536269514722 * 0.1
    upper = 0.5 + 1.6448536269514722 * 0.1
    picp = prediction_interval_coverage_probability(y_true, lower, upper)
    assert abs(picp - 0.90) <= 0.05


def test_soh_model_round_trip_predictions_match(tmp_path) -> None:
    """Saved and loaded SOH models produce identical predictions."""
    X, y = _linear_data()
    model = GPRSOHModel(["feat_x", "feat_extra"], max_train_samples=80, n_restarts_optimizer=0)
    model.fit(X, y)
    path = tmp_path / "gpr.joblib"
    model.save(path)
    loaded = GPRSOHModel.load(path)
    before = model.predict_interval(X.iloc[:5])[0]
    after = loaded.predict_interval(X.iloc[:5])[0]
    assert np.allclose(before, after)


class _ConstantSOHModel:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def predict_interval(self, X: pd.DataFrame, alpha: float = 0.10):
        del alpha
        mean = np.full(len(X), self.mean)
        lower = mean - 1.6448536269514722 * self.std
        upper = mean + 1.6448536269514722 * self.std
        return mean, lower, upper


def test_ensemble_mixture_math_matches_hand_calculation() -> None:
    """SOHEnsemble uses law-of-total-variance mixture math."""
    X = pd.DataFrame({"feat": [0.0]})
    probs = pd.DataFrame({"LFP": [0.25], "LCO": [0.75]})
    ensemble = SOHEnsemble({"LFP": _ConstantSOHModel(0.8, 0.1), "LCO": _ConstantSOHModel(0.4, 0.2)})
    mean, _, _, std = ensemble.predict(X, probs)
    expected_mean = 0.25 * 0.8 + 0.75 * 0.4
    expected_var = 0.25 * (0.1**2 + 0.8**2) + 0.75 * (0.2**2 + 0.4**2) - expected_mean**2
    assert np.isclose(mean[0], expected_mean)
    assert np.isclose(std[0], np.sqrt(expected_var))


def test_soh_feature_list_excludes_capacity_leakage() -> None:
    """SOH features must not contain direct capacity-derived target leakage."""
    forbidden = {"feat_capacity_ratio", "discharge_capacity_ah", "charge_capacity_ah"}
    assert not forbidden.intersection(SOH_FEATURES)
