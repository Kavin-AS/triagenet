"""Tests for the ChemistryClassifier wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from triagenet.models.chemistry import ChemistryClassifier


def test_chemistry_classifier_fit_predict_proba_and_importances() -> None:
    """The wrapper fits, predicts labels, returns probabilities, and names importances."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "feat_a": np.r_[rng.normal(0, 0.1, 60), rng.normal(3, 0.1, 60)],
            "feat_b": np.r_[rng.normal(1, 0.1, 60), rng.normal(1, 0.1, 60)],
        }
    )
    y = pd.Series(["LCO"] * 60 + ["LFP"] * 60)
    cells = pd.Series([f"cell_{i}" for i in range(120)])
    clf = ChemistryClassifier(feature_names=["feat_a", "feat_b"]).fit(X, y, cells)
    pred = clf.predict(X)
    proba = clf.predict_proba(X)
    importances = clf.feature_importances_
    assert set(pred) == {"LCO", "LFP"}
    assert proba.shape == (120, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(importances) == {"feat_a", "feat_b"}
