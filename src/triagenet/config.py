"""Configuration constants for TriageNet paths, chemistries, and capacities."""

from pathlib import Path


def find_repo_root() -> Path:
    """Return the repository root by walking upward from this file."""
    current = Path(__file__).resolve()
    for parent in (current, *current.parents):
        if (parent / "AGENTS.md").exists():
            return parent
    raise RuntimeError("Could not find repository root containing AGENTS.md")


REPO_ROOT = find_repo_root()
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_INTERIM = REPO_ROOT / "data" / "interim"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"

CHEMISTRIES = ("LFP", "NMC", "NCA", "LCO")

# Nominal capacities are conservative defaults used only when raw metadata omits a value.
# Sources:
# - Sandia/Battery Archive SNL study uses commercial 18650 cells and reports C-rates based
#   on rated capacities: https://www.batteryarchive.org/snl_study.html
# - CALCE CS2/CX2 pages list CS2=1100 mAh and CX2=1350 mAh:
#   https://web.calce.umd.edu/batteries/data/
# - Severson et al. MIT/Stanford cells are A123 APR18650M1A LFP cells, commonly reported as
#   1.1 Ah in the 2019 fast-charging dataset documentation.
DEFAULT_NOMINAL_CAPACITY_AH: dict[str, dict[str, float]] = {
    "LFP": {"18650": 1.10, "A123_APR18650M1A": 1.10, "K2_26650": 2.60},
    "NMC": {"18650": 2.80, "pouch": 1.50, "prismatic": 2.00},
    "NCA": {"18650": 3.20},
    "LCO": {"18650": 2.00, "CS2_prismatic": 1.10, "CX2_prismatic": 1.35, "pouch": 1.50},
}

# Nominal cathode voltages used for per-cell kWh estimates. Sources: A123 APR18650M1A
# nominal voltage 3.3 V for LFP, common graphite/LCO and NMC cell nominal voltage 3.6-3.7 V,
# and Panasonic-style NCA 18650 nominal voltage 3.6 V in manufacturer datasheets.
DEFAULT_NOMINAL_VOLTAGE_V: dict[str, float] = {
    "LFP": 3.3,
    "LCO": 3.7,
    "NMC": 3.7,
    "NCA": 3.6,
}

CURVE_LENGTH = 100
