"""Literature-average recycling and second-life constants for TriageNet economics."""

from __future__ import annotations

from triagenet.config import DEFAULT_NOMINAL_VOLTAGE_V

METALS = ("lithium", "cobalt", "nickel", "manganese", "copper", "aluminum")

# Hydrometallurgical recovery assumptions. Sources: Velazquez-Martinez et al. 2019,
# "A Critical Review of Lithium-Ion Battery Recycling Processes from a Circular Economy
# Perspective"; Harper et al. 2019 Nature, "Recycling lithium-ion batteries from electric
# vehicles". Fe/P/Al are treated as no-value streams in this MVP.
RECOVERY_RATES = {
    "LFP": {"lithium": 0.90},
    "LCO": {"lithium": 0.92, "cobalt": 0.95},
    "NMC": {"lithium": 0.92, "cobalt": 0.95, "nickel": 0.95, "manganese": 0.95},
    "NCA": {"lithium": 0.92, "cobalt": 0.95, "nickel": 0.95},
}

# kg metal per kWh. Literature-average composition approximations from IEA 2021
# "The Role of Critical Minerals in Clean Energy Transitions" and BloombergNEF battery
# materials summaries. These are placeholders until Bridge Green supplies assay data.
METAL_MASS_KG_PER_KWH = {
    "LFP": {"lithium": 0.10},
    "LCO": {"lithium": 0.11, "cobalt": 0.85},
    "NMC": {"lithium": 0.10, "cobalt": 0.06, "nickel": 0.30, "manganese": 0.06},
    "NCA": {"lithium": 0.10, "cobalt": 0.05, "nickel": 0.35, "aluminum": 0.02},
}

# Martinez-Laserna et al. 2018, Renewable & Sustainable Energy Reviews, second-life
# cycle-life ranges. Values are conservative MVP priors.
SECOND_LIFE_BASE_CYCLES = {"LFP": 2000.0, "LCO": 500.0, "NMC": 1500.0, "NCA": 1200.0}

PROCESSING_LOSS_RATE = 0.05

# TODO(stub): Costs are scaled to small-cell equivalent economics for this demo. Real deployment
# should replace these with Bridge Green's pack/module-level hydromet and regrading costs.
RECYCLE_PROCESSING_COST_USD = {"LFP": 0.01, "LCO": 0.03, "NMC": 0.03, "NCA": 0.03}
SECOND_LIFE_PROCESSING_COST_USD = {"LFP": 0.05, "LCO": 0.10, "NMC": 0.08, "NCA": 0.08}

# NREL second-life storage cost-benefit reports commonly place usable grid-service value in the
# single-digit cents/kWh range; $0.05/kWh is a conservative MVP credit.
LCOE_CREDIT_USD_PER_KWH = 0.05

# Converts lab-cell economics to an operational cell-equivalent lot value so VOI is visible at the
# interview scale. It is reported as an assumption and should be removed with real pack economics.
ECONOMIC_SCALE_FACTOR = 100.0


def recovery_rate(chemistry: str, metal: str) -> float:
    """Return the hydrometallurgical recovery fraction for a chemistry and metal."""
    return float(RECOVERY_RATES.get(chemistry.upper(), {}).get(metal, 0.0))


def metal_mass_kg_per_kwh(chemistry: str, metal: str) -> float:
    """Return kg of target metal per nominal kWh for a chemistry."""
    return float(METAL_MASS_KG_PER_KWH.get(chemistry.upper(), {}).get(metal, 0.0))


def nominal_voltage(chemistry: str) -> float:
    """Return nominal voltage for a chemistry."""
    return float(DEFAULT_NOMINAL_VOLTAGE_V[chemistry.upper()])


def soh_factor(soh: float) -> float:
    """Return nonlinear second-life cycle retention factor from SOH."""
    if soh < 0.70:
        return 0.0
    if soh < 1.0:
        return float((soh - 0.70) / 0.30)
    return 1.0
