"""Typed API schemas for the TriageNet demo service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ChemistryName = Literal["LFP", "LCO", "NMC", "NCA"]
DecisionName = Literal["second_life", "direct_recycle", "needs_more_characterization"]
ConfidenceName = Literal["high", "medium", "low"]


class CyclePayload(BaseModel):
    """A single raw characterization cycle."""

    model_config = ConfigDict(extra="forbid")

    cell_id: str
    cycle_index: int
    nominal_capacity_ah: float
    discharge_capacity_ah: float
    charge_capacity_ah: float
    coulombic_efficiency: float
    voltage_curve: list[float]
    current_curve: list[float]
    time_curve_s: list[float]
    temperature_c_mean: float | None = None
    c_rate_charge: float | None = None
    c_rate_discharge: float | None = None
    known_chemistry: ChemistryName | None = None
    known_soh: float | None = None
    description: str | None = None

    @field_validator("voltage_curve", "current_curve", "time_curve_s")
    @classmethod
    def validate_curve_length(cls, value: list[float]) -> list[float]:
        """Reject malformed cycle curves before they hit the featurizer."""
        if len(value) != 100:
            raise ValueError("cycle curves must contain exactly 100 points")
        return value


class ChemistryBlock(BaseModel):
    """Chemistry classification response block."""

    probabilities: dict[ChemistryName, float]
    predicted: ChemistryName


class SOHBlock(BaseModel):
    """State-of-health prediction response block."""

    mean: float
    lower_90: float
    upper_90: float
    std: float


class ValueBlock(BaseModel):
    """Expected-value distribution for one route."""

    mean: float
    p10: float
    p90: float


class EconomicsBlock(BaseModel):
    """Economic valuation response block."""

    expected_value_usd: float
    second_life_value_usd: ValueBlock
    recycle_value_usd: ValueBlock
    value_of_info_one_more_cycle_usd: float


class TopFeature(BaseModel):
    """A feature contribution shown in the dashboard."""

    name: str
    value: float
    importance: float


class Verdict(BaseModel):
    """Output of the full TriageNet pipeline."""

    cell_id: str
    cycle_index: int
    chemistry: ChemistryBlock
    soh: SOHBlock
    decision: DecisionName
    confidence: ConfidenceName
    economics: EconomicsBlock
    rationale: str
    top_features: list[TopFeature]
    runtime_ms: int


class HealthResponse(BaseModel):
    """Backend liveness and artifact status."""

    status: Literal["ok"]
    model_versions: dict[str, str]
    prices_age_hours: float


class PricesResponse(BaseModel):
    """Commodity price snapshot used by the triage layer."""

    prices: dict[str, float]
    prices_usd_per_kg: dict[str, float]
    as_of: str
    is_live: bool


class MetricsResponse(BaseModel):
    """Training/evaluation metrics surfaced to the dashboard."""

    chemistry: dict[str, object] = Field(default_factory=dict)
    soh: dict[str, object] = Field(default_factory=dict)
    triage: dict[str, object] = Field(default_factory=dict)
