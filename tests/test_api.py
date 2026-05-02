"""FastAPI route tests for the Phase 5 serving surface."""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from api.main import app


def test_health_and_metrics_endpoints() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        metrics = client.get("/metrics")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert "chemistry" in health.json()["model_versions"]
    assert metrics.status_code == 200
    assert "triage" in metrics.json()


def test_predict_sample_cycle_happy_path() -> None:
    with TestClient(app) as client:
        samples = client.get("/sample-cycles").json()
        response = client.post("/predict", json=samples[0])
    assert response.status_code == 200
    verdict = response.json()
    assert verdict["cell_id"] == samples[0]["cell_id"]
    assert verdict["decision"] in {
        "second_life",
        "direct_recycle",
        "needs_more_characterization",
    }
    assert len(verdict["top_features"]) > 0


def test_sample_cycles_are_diverse_and_described() -> None:
    with TestClient(app) as client:
        samples = client.get("/sample-cycles").json()
    assert len(samples) == 12
    assert samples[0]["cell_id"] == "calce_cs2_33"
    assert samples[0]["cycle_index"] == 865
    assert all(sample.get("description") for sample in samples)
    assert {sample["known_chemistry"] for sample in samples} == {"LCO", "LFP"}
    assert max(sample["cycle_index"] for sample in samples) > 800
    assert min(sample["known_soh"] for sample in samples) < 0.10
    assert samples[9]["cell_id"] == "calce_cs2_37"
    assert samples[9]["cycle_index"] == 98


def test_predict_rejects_bad_curve_length() -> None:
    with TestClient(app) as client:
        payload = client.get("/sample-cycles").json()[0]
        payload["voltage_curve"] = payload["voltage_curve"][:-1]
        response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_batch_and_file_upload() -> None:
    with TestClient(app) as client:
        samples = client.get("/sample-cycles").json()[:2]
        batch = client.post("/predict-batch", json=samples)
        csv_response = client.post(
            "/predict-file",
            files={
                "file": (
                    "cycles.csv",
                    _sample_csv(samples[:1]),
                    "text/csv",
                )
            },
        )
    assert batch.status_code == 200
    assert len(batch.json()) == 2
    assert csv_response.status_code == 200
    assert len(csv_response.json()) == 1


def test_prices_endpoint() -> None:
    started = time.perf_counter()
    with TestClient(app) as client:
        response = client.get("/prices")
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert elapsed < 3.0
    payload = response.json()
    assert "lithium" in payload["prices"]
    assert "lithium" in payload["prices_usd_per_kg"]
    assert isinstance(payload["is_live"], bool)


def test_required_sample_verdicts() -> None:
    with TestClient(app) as client:
        samples = client.get("/sample-cycles").json()
        cycle_865 = client.post("/predict", json=samples[0]).json()
        fresh_lfp = client.post("/predict", json=samples[2]).json()
        near_dead_lco = client.post("/predict", json=samples[9]).json()
    assert cycle_865["decision"] == "needs_more_characterization"
    assert fresh_lfp["decision"] == "second_life"
    assert fresh_lfp["confidence"] == "high"
    assert near_dead_lco["decision"] == "direct_recycle"
    assert near_dead_lco["confidence"] == "high"


def _sample_csv(samples: list[dict[str, object]]) -> str:
    sample = samples[0]
    fields = [
        "cell_id",
        "cycle_index",
        "nominal_capacity_ah",
        "discharge_capacity_ah",
        "charge_capacity_ah",
        "coulombic_efficiency",
        "voltage_curve",
        "current_curve",
        "time_curve_s",
        "temperature_c_mean",
        "c_rate_charge",
        "c_rate_discharge",
        "known_chemistry",
        "known_soh",
    ]
    values = []
    for field in fields:
        value = sample.get(field)
        if isinstance(value, list):
            values.append(json.dumps(value))
        else:
            values.append("" if value is None else str(value))
    escaped = [f'"{value.replace(chr(34), chr(34) + chr(34))}"' for value in values]
    return ",".join(fields) + "\n" + ",".join(escaped) + "\n"
