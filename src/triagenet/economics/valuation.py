"""Economic valuation functions for recycle, second-life, and information value."""

from __future__ import annotations

import numpy as np
from triagenet.config import CHEMISTRIES
from triagenet.economics.prices import get_price_uncertainty
from triagenet.economics.recovery import (
    ECONOMIC_SCALE_FACTOR,
    LCOE_CREDIT_USD_PER_KWH,
    METALS,
    PROCESSING_LOSS_RATE,
    RECYCLE_PROCESSING_COST_USD,
    SECOND_LIFE_BASE_CYCLES,
    SECOND_LIFE_PROCESSING_COST_USD,
    metal_mass_kg_per_kwh,
    recovery_rate,
    soh_factor,
)


def compute_recycle_value(
    chemistry_probs: dict[str, float],
    cell_kwh: float,
    prices: dict[str, float],
    random_state: int = 42,
) -> dict[str, float]:
    """Return recycle value moments from chemistry mixture and price uncertainty."""
    rng = np.random.default_rng(random_state)
    values = []
    chemistries = _chemistry_labels(chemistry_probs)
    probabilities = _probabilities(chemistry_probs, chemistries)
    price_cv = get_price_uncertainty()
    for _ in range(1000):
        chemistry = str(rng.choice(chemistries, p=probabilities))
        sampled_prices = _sample_prices(prices, price_cv, rng)
        values.append(_recycle_value_given_chemistry(chemistry, cell_kwh, sampled_prices))
    return _summary(values)


def compute_second_life_value(
    chemistry_probs: dict[str, float],
    soh_mean: float,
    soh_std: float,
    cell_kwh: float,
    prices: dict[str, float],
    n_samples: int = 1000,
    random_state: int = 42,
) -> dict[str, float]:
    """Return Monte Carlo second-life value over chemistry and SOH uncertainty."""
    del prices
    rng = np.random.default_rng(random_state)
    chemistries = _chemistry_labels(chemistry_probs)
    probabilities = _probabilities(chemistry_probs, chemistries)
    values = []
    for _ in range(n_samples):
        chemistry = str(rng.choice(chemistries, p=probabilities))
        soh = float(np.clip(rng.normal(soh_mean, max(soh_std, 1e-6)), 0.0, 1.05))
        values.append(_second_life_value_given_chemistry(chemistry, soh, cell_kwh))
    return _summary(values)


def compute_value_of_information(
    chemistry_probs: dict[str, float],
    soh_mean: float,
    soh_std: float,
    cell_kwh: float,
    prices: dict[str, float],
    characterization_cost: float = 2.0,
    random_state: int = 42,
) -> float:
    """Approximate value of one more characterization cycle by halving SOH variance."""
    del characterization_cost
    current_recycle = compute_recycle_value(chemistry_probs, cell_kwh, prices, random_state)
    current_second = compute_second_life_value(
        chemistry_probs, soh_mean, soh_std, cell_kwh, prices, random_state=random_state
    )
    current_best = max(current_recycle["mean"], current_second["mean"])
    rng = np.random.default_rng(random_state)
    posterior_std = max(soh_std / np.sqrt(2.0), 1e-6)
    future_best = []
    for _ in range(200):
        posterior_mean = float(np.clip(rng.normal(soh_mean, posterior_std), 0.0, 1.05))
        second = compute_second_life_value(
            chemistry_probs,
            posterior_mean,
            posterior_std,
            cell_kwh,
            prices,
            n_samples=250,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        future_best.append(max(current_recycle["mean"], second["mean"]))
    return max(0.0, float(np.mean(future_best) - current_best))


def _recycle_value_given_chemistry(
    chemistry: str, cell_kwh: float, prices: dict[str, float]
) -> float:
    gross = 0.0
    for metal in METALS:
        gross += (
            recovery_rate(chemistry, metal)
            * metal_mass_kg_per_kwh(chemistry, metal)
            * cell_kwh
            * prices.get(metal, 0.0)
            * (1.0 - PROCESSING_LOSS_RATE)
        )
    return ECONOMIC_SCALE_FACTOR * (gross - RECYCLE_PROCESSING_COST_USD.get(chemistry, 0.03))


def _second_life_value_given_chemistry(chemistry: str, soh: float, cell_kwh: float) -> float:
    cycles = SECOND_LIFE_BASE_CYCLES.get(chemistry, 1000.0) * soh_factor(soh)
    throughput = cell_kwh * soh * cycles
    revenue = throughput * LCOE_CREDIT_USD_PER_KWH
    cost = SECOND_LIFE_PROCESSING_COST_USD.get(chemistry, 0.08)
    return ECONOMIC_SCALE_FACTOR * (revenue - cost)


def _sample_prices(
    prices: dict[str, float], price_cv: dict[str, float], rng: np.random.Generator
) -> dict[str, float]:
    sampled = {}
    for metal, price in prices.items():
        std = abs(price) * price_cv.get(metal, 0.10)
        sampled[metal] = max(0.0, float(rng.normal(price, std)))
    return sampled


def _chemistry_labels(chemistry_probs: dict[str, float]) -> list[str]:
    labels = [chemistry for chemistry in CHEMISTRIES if chemistry_probs.get(chemistry, 0.0) > 0]
    return labels or ["LFP"]


def _probabilities(chemistry_probs: dict[str, float], labels: list[str]) -> np.ndarray:
    probs = np.asarray([chemistry_probs.get(label, 0.0) for label in labels], dtype=float)
    if probs.sum() <= 0:
        probs = np.ones(len(labels), dtype=float)
    return probs / probs.sum()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }
