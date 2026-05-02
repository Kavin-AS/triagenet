"""Tests for Phase 4 economics functions."""

from __future__ import annotations

import json

from triagenet.economics.prices import get_spot_prices
from triagenet.economics.valuation import (
    compute_recycle_value,
    compute_second_life_value,
    compute_value_of_information,
)


def test_lfp_healthy_second_life_beats_recycle() -> None:
    prices = get_spot_prices(refresh=True)
    probs = {"LFP": 1.0}
    recycle = compute_recycle_value(probs, 0.0036, prices)
    second = compute_second_life_value(probs, 1.0, 0.01, 0.0036, prices)
    assert second["mean"] > recycle["mean"]


def test_lco_dead_recycle_beats_second_life() -> None:
    prices = get_spot_prices()
    probs = {"LCO": 1.0}
    recycle = compute_recycle_value(probs, 0.0040, prices)
    second = compute_second_life_value(probs, 0.05, 0.01, 0.0040, prices)
    assert recycle["mean"] > second["mean"]


def test_voi_nonnegative_and_shrinks_with_certainty() -> None:
    prices = get_spot_prices()
    probs = {"LFP": 1.0}
    uncertain = compute_value_of_information(probs, 0.80, 0.20, 0.0036, prices)
    certain = compute_value_of_information(probs, 0.80, 1e-6, 0.0036, prices)
    assert uncertain >= 0
    assert certain <= uncertain
    assert certain < 1e-3


def test_voi_largest_near_boundary() -> None:
    prices = get_spot_prices()
    probs = {"LFP": 1.0}
    near = compute_value_of_information(probs, 0.72, 0.20, 0.0036, prices)
    far = compute_value_of_information(probs, 0.98, 0.02, 0.0036, prices)
    assert near >= far


def test_price_cache_corruption_falls_back(tmp_path, monkeypatch) -> None:
    from triagenet.economics import prices as price_module

    cache = tmp_path / "prices_cache.json"
    cache.write_text("{broken")
    monkeypatch.setattr(price_module, "CACHE_PATH", cache)
    prices = price_module.get_spot_prices()
    assert prices["lithium"] > 0
    payload = json.loads(cache.read_text())
    assert "prices_usd_per_kg" in payload
