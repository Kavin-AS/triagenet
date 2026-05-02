"""Risk-aware techno-economic triage decision engine."""

from __future__ import annotations

from typing import Any

from triagenet.economics.prices import get_spot_prices
from triagenet.economics.valuation import (
    compute_recycle_value,
    compute_second_life_value,
    compute_value_of_information,
)


class TriageEngine:
    """Convert chemistry and SOH uncertainty into a business routing verdict."""

    def __init__(self, characterization_cost: float = 2.0, random_state: int = 42) -> None:
        self.characterization_cost = characterization_cost
        self.random_state = random_state

    def decide(
        self,
        chemistry_probs: dict[str, float],
        soh_mean: float,
        soh_lower_90: float,
        soh_upper_90: float,
        soh_std: float,
        cell_kwh: float,
        prices: dict[str, float] | None = None,
        rule: str = "risk_aware",
    ) -> dict[str, Any]:
        """Return a triage verdict JSON for one cell under the selected decision rule."""
        price_map = prices or get_spot_prices()
        if rule == "naive":
            return self._decide_naive(chemistry_probs, soh_mean, soh_std, cell_kwh, price_map)
        if rule == "conservative":
            return self._decide_conservative(
                chemistry_probs, soh_mean, soh_lower_90, soh_std, cell_kwh, price_map
            )
        if rule == "risk_aware":
            return self._decide_risk_aware(
                chemistry_probs,
                soh_mean,
                soh_lower_90,
                soh_upper_90,
                soh_std,
                cell_kwh,
                price_map,
            )
        raise ValueError(f"Unknown triage rule: {rule}")

    def _decide_naive(
        self,
        chemistry_probs: dict[str, float],
        soh_mean: float,
        soh_std: float,
        cell_kwh: float,
        prices: dict[str, float],
    ) -> dict[str, Any]:
        values = self._values(chemistry_probs, soh_mean, soh_std, cell_kwh, prices)
        decision = "second_life" if soh_mean >= 0.80 else "direct_recycle"
        return self._verdict(decision, values, 0.0, "medium", "Naive SOH mean threshold rule.")

    def _decide_conservative(
        self,
        chemistry_probs: dict[str, float],
        soh_mean: float,
        soh_lower_90: float,
        soh_std: float,
        cell_kwh: float,
        prices: dict[str, float],
    ) -> dict[str, Any]:
        values = self._values(chemistry_probs, soh_mean, soh_std, cell_kwh, prices)
        decision = "second_life" if soh_lower_90 >= 0.80 else "direct_recycle"
        return self._verdict(decision, values, 0.0, "medium", "Conservative lower-bound rule.")

    def _decide_risk_aware(
        self,
        chemistry_probs: dict[str, float],
        soh_mean: float,
        soh_lower_90: float,
        soh_upper_90: float,
        soh_std: float,
        cell_kwh: float,
        prices: dict[str, float],
    ) -> dict[str, Any]:
        values = self._values(chemistry_probs, soh_mean, soh_std, cell_kwh, prices)
        second_mean = values["second_life_value_usd"]["mean"]
        recycle_mean = values["recycle_value_usd"]["mean"]
        gap = abs(second_mean - recycle_mean)
        voi = compute_value_of_information(
            chemistry_probs,
            soh_mean,
            soh_std,
            cell_kwh,
            prices,
            characterization_cost=self.characterization_cost,
            random_state=self.random_state,
        )
        threshold_crossed = soh_lower_90 < 0.80 < soh_upper_90
        if threshold_crossed and soh_std > 0.08:
            decision = "needs_more_characterization"
            confidence = "low"
        elif gap < voi and voi > self.characterization_cost:
            decision = "needs_more_characterization"
            confidence = "low"
        else:
            decision = "second_life" if second_mean >= recycle_mean else "direct_recycle"
            confidence = "high" if gap > 2.0 * max(voi, 1e-9) else "medium"
            if voi < self.characterization_cost:
                confidence = "high"
        rationale = (
            f"{self._top_chemistry(chemistry_probs)}, SOH {soh_mean:.2f} "
            f"[{soh_lower_90:.2f}, {soh_upper_90:.2f}]. "
            f"Second-life EV ${second_mean:.2f} vs recycle EV ${recycle_mean:.2f}. "
            f"VOI for one more cycle ${voi:.2f}."
        )
        return self._verdict(decision, values, voi, confidence, rationale)

    def _values(
        self,
        chemistry_probs: dict[str, float],
        soh_mean: float,
        soh_std: float,
        cell_kwh: float,
        prices: dict[str, float],
    ) -> dict[str, Any]:
        recycle = compute_recycle_value(
            chemistry_probs, cell_kwh, prices, random_state=self.random_state
        )
        second = compute_second_life_value(
            chemistry_probs, soh_mean, soh_std, cell_kwh, prices, random_state=self.random_state
        )
        return {"second_life_value_usd": second, "recycle_value_usd": recycle}

    def _verdict(
        self,
        decision: str,
        values: dict[str, Any],
        voi: float,
        confidence: str,
        rationale: str,
    ) -> dict[str, Any]:
        expected = (
            values["second_life_value_usd"]["mean"]
            if decision == "second_life"
            else (
                values["recycle_value_usd"]["mean"]
                if decision == "direct_recycle"
                else max(
                    values["second_life_value_usd"]["mean"], values["recycle_value_usd"]["mean"]
                )
            )
        )
        return {
            "decision": decision,
            "expected_value_usd": float(expected),
            "second_life_value_usd": _value_block(values["second_life_value_usd"]),
            "recycle_value_usd": _value_block(values["recycle_value_usd"]),
            "value_of_info_one_more_cycle_usd": float(voi),
            "rationale": rationale,
            "confidence": confidence,
        }

    @staticmethod
    def _top_chemistry(chemistry_probs: dict[str, float]) -> str:
        chemistry, probability = max(chemistry_probs.items(), key=lambda item: item[1])
        return f"Likely {chemistry} ({probability:.0%})"


def _value_block(value: dict[str, float]) -> dict[str, float]:
    return {"mean": float(value["mean"]), "p10": float(value["p10"]), "p90": float(value["p90"])}
