"""Commodity price access with a documented snapshot fallback."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta

from triagenet.config import DATA_PROCESSED

LOGGER = logging.getLogger(__name__)
CACHE_PATH = DATA_PROCESSED / "prices_cache.json"

# TODO(stub): Snapshot fallback used when no authenticated live commodity API is configured.
# Approximate late-April-2026 USD/kg levels from public market pages such as LME delayed
# non-ferrous prices, Trading Economics commodity summaries, and Benchmark/industry lithium
# carbonate commentary. Replace with Bridge Green's internal price feed in production.
FALLBACK_PRICES_USD_PER_KG = {
    "lithium": 14.0,
    "cobalt": 32.0,
    "nickel": 18.0,
    "manganese": 2.0,
    "copper": 10.0,
    "aluminum": 3.0,
}

PRICE_CV = {
    "lithium": 0.15,
    "cobalt": 0.12,
    "nickel": 0.10,
    "manganese": 0.08,
    "copper": 0.08,
    "aluminum": 0.08,
}


def get_spot_prices(refresh: bool = False) -> dict[str, float]:
    """Return USD/kg spot prices from cache or a loudly labeled snapshot fallback."""
    if not refresh:
        cached = _read_cache()
        if cached is not None:
            return cached
    LOGGER.warning("Using documented fallback commodity price snapshot; no live API configured.")
    _write_cache(FALLBACK_PRICES_USD_PER_KG, source="fallback_snapshot_late_april_2026")
    return dict(FALLBACK_PRICES_USD_PER_KG)


def get_price_uncertainty() -> dict[str, float]:
    """Return coefficient-of-variation assumptions for commodity-price Monte Carlo."""
    return dict(PRICE_CV)


def _read_cache() -> dict[str, float] | None:
    if not CACHE_PATH.exists():
        return None
    try:
        payload = json.loads(CACHE_PATH.read_text())
        timestamp = datetime.fromisoformat(payload["timestamp_utc"])
        if datetime.now(UTC) - timestamp > timedelta(hours=24):
            return None
        prices = payload["prices_usd_per_kg"]
        return {metal: float(price) for metal, price in prices.items()}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        LOGGER.warning("Ignoring corrupt commodity price cache at %s: %s", CACHE_PATH, exc)
        return None


def _write_cache(prices: dict[str, float], source: str) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source": source,
        "prices_usd_per_kg": prices,
    }
    CACHE_PATH.write_text(json.dumps(payload, indent=2))


def main() -> None:
    """CLI helper for refreshing and printing cached commodity prices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Force refresh/fallback cache write")
    args = parser.parse_args()
    print(json.dumps(get_spot_prices(refresh=args.refresh), indent=2))


if __name__ == "__main__":
    main()
