"""End-to-end API smoke tests for the required demo path."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


def test_cycle_865_routes_to_more_characterization() -> None:
    with TestClient(app) as client:
        samples = client.get("/sample-cycles").json()
        cycle_865 = next(
            sample
            for sample in samples
            if sample["cell_id"] == "calce_cs2_33" and sample["cycle_index"] == 865
        )
        response = client.post("/predict", json=cycle_865)
    assert response.status_code == 200
    verdict = response.json()
    assert verdict["decision"] == "needs_more_characterization"
    assert verdict["confidence"] == "low"
    assert verdict["economics"]["value_of_info_one_more_cycle_usd"] > 0
