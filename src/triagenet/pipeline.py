"""End-to-end TriageNet prediction pipeline."""

from __future__ import annotations

import logging
from typing import Any

import joblib
import pandas as pd

from triagenet.config import DEFAULT_NOMINAL_VOLTAGE_V, MODELS_DIR
from triagenet.economics.prices import get_spot_prices
from triagenet.features.eol_cycle import CHEMISTRY_FEATURES, SOH_FEATURES, extract_features
from triagenet.models.triage import TriageEngine

LOGGER = logging.getLogger(__name__)


class TriagePipeline:
    """Load trained models and produce a full triage verdict from one cycle."""

    def __init__(self, verbose: bool = False, random_state: int = 42) -> None:
        self.verbose = verbose
        self.random_state = random_state
        self.chemistry_model = joblib.load(MODELS_DIR / "chemistry_xgb.joblib")
        self.soh_ensemble = joblib.load(MODELS_DIR / "soh_ensemble.joblib")
        self.engine = TriageEngine(random_state=random_state)
        self.prices = get_spot_prices()

    def predict(
        self, cycle: dict[str, Any] | pd.Series, rule: str = "risk_aware"
    ) -> dict[str, Any]:
        """Run featurization, chemistry, SOH, and economics for one UnifiedCycle row."""
        row = cycle.to_dict() if isinstance(cycle, pd.Series) else dict(cycle)
        features = extract_features(row)
        frame = pd.DataFrame([features])
        chemistry_proba = self.chemistry_model.predict_proba(frame[list(CHEMISTRY_FEATURES)])[0]
        chemistry_probs = {
            str(label): float(prob)
            for label, prob in zip(self.chemistry_model.classes_, chemistry_proba, strict=True)
        }
        for chemistry in ("LFP", "LCO", "NMC", "NCA"):
            chemistry_probs.setdefault(chemistry, 0.0)
        probs_df = pd.DataFrame([chemistry_probs])
        soh_mean, soh_lower, soh_upper, soh_std = self.soh_ensemble.predict(
            frame[list(SOH_FEATURES)], probs_df
        )
        cell_kwh = _cell_kwh(row)
        if self.verbose:
            LOGGER.info("chemistry_probs=%s", chemistry_probs)
            LOGGER.info(
                "soh mean/lower/upper/std=%s/%s/%s/%s",
                soh_mean[0],
                soh_lower[0],
                soh_upper[0],
                soh_std[0],
            )
        verdict = self.engine.decide(
            chemistry_probs=chemistry_probs,
            soh_mean=float(soh_mean[0]),
            soh_lower_90=float(soh_lower[0]),
            soh_upper_90=float(soh_upper[0]),
            soh_std=float(soh_std[0]),
            cell_kwh=cell_kwh,
            prices=self.prices,
            rule=rule,
        )
        verdict.update(
            {
                "cell_id": row.get("cell_id"),
                "cycle_index": row.get("cycle_index"),
                "chemistry_probs": chemistry_probs,
                "soh_mean": float(soh_mean[0]),
                "soh_lower_90": float(soh_lower[0]),
                "soh_upper_90": float(soh_upper[0]),
                "soh_std": float(soh_std[0]),
                "cell_kwh": cell_kwh,
                "rule": rule,
            }
        )
        return verdict

    def predict_batch(self, cycles_df: pd.DataFrame, rule: str = "risk_aware") -> pd.DataFrame:
        """Run the triage pipeline over a dataframe of UnifiedCycle rows."""
        return pd.DataFrame([self.predict(row, rule=rule) for _, row in cycles_df.iterrows()])


def _cell_kwh(row: dict[str, Any]) -> float:
    chemistry = str(row.get("chemistry", "LFP")).upper()
    nominal_capacity = float(row.get("nominal_capacity_ah", 1.1))
    nominal_voltage = DEFAULT_NOMINAL_VOLTAGE_V.get(chemistry, 3.6)
    return nominal_capacity * nominal_voltage / 1000.0
