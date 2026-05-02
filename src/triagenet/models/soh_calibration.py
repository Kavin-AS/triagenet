"""Calibration metrics and plots for SOH prediction intervals."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def prediction_interval_coverage_probability(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Return the fraction of true SOH values inside the predicted interval."""
    truth = np.asarray(y_true, dtype=float)
    return float(np.mean((truth >= lower) & (truth <= upper)))


def mean_prediction_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Return the mean width of predicted SOH intervals."""
    return float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))


def expected_calibration_error(
    y_true: np.ndarray,
    predicted_intervals: tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray],
    alpha: float,
    n_bins: int = 10,
) -> float:
    """Compute weighted coverage error after binning predictions by predicted uncertainty."""
    if len(predicted_intervals) == 3:
        mean, lower, upper = predicted_intervals
        z_score = float(norm.ppf(1.0 - alpha / 2.0))
        std = np.maximum((np.asarray(upper) - np.asarray(lower)) / (2.0 * z_score), 1e-12)
    else:
        lower, upper = predicted_intervals
        mean = (np.asarray(lower) + np.asarray(upper)) / 2.0
        std = np.maximum(np.asarray(upper) - np.asarray(lower), 1e-12)
    del mean
    truth = np.asarray(y_true, dtype=float)
    covered = (truth >= lower) & (truth <= upper)
    quantiles = np.quantile(std, np.linspace(0, 1, n_bins + 1))
    quantiles = np.unique(quantiles)
    if quantiles.size < 2:
        return abs(float(np.mean(covered)) - (1.0 - alpha))
    total = 0.0
    for low, high in zip(quantiles[:-1], quantiles[1:], strict=True):
        mask = (std >= low) & (std <= high if high == quantiles[-1] else std < high)
        if not np.any(mask):
            continue
        total += (np.sum(mask) / len(std)) * abs(float(np.mean(covered[mask])) - (1.0 - alpha))
    return float(total)


def reliability_diagram(
    y_true: np.ndarray, mean: np.ndarray, std: np.ndarray, save_path: Path
) -> None:
    """Save predicted Gaussian quantile versus empirical quantile reliability plot."""
    truth = np.asarray(y_true, dtype=float)
    predicted = np.asarray(mean, dtype=float)
    sigma = np.maximum(np.asarray(std, dtype=float), 1e-12)
    nominal = np.linspace(0.1, 0.9, 9)
    empirical = []
    for quantile in nominal:
        threshold = predicted + norm.ppf(quantile) * sigma
        empirical.append(float(np.mean(truth <= threshold)))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(nominal, empirical, marker="o")
    ax.set_xlabel("Predicted quantile")
    ax.set_ylabel("Empirical quantile")
    ax.set_title("SOH interval reliability")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def sharpness_diagram(std: np.ndarray, save_path: Path) -> None:
    """Save a histogram of predictive standard deviations."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(np.asarray(std, dtype=float), bins=20)
    ax.set_xlabel("Predicted SOH std")
    ax.set_ylabel("Count")
    ax.set_title("SOH interval sharpness")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
