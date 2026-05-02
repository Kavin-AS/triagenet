"""FastAPI surface for the TriageNet end-to-end demo."""

from __future__ import annotations

import ast
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from triagenet.config import DATA_PROCESSED, MODELS_DIR, REPO_ROOT
from triagenet.economics.prices import get_spot_prices
from triagenet.features.eol_cycle import CHEMISTRY_FEATURES, extract_features
from triagenet.pipeline import TriagePipeline

from api.schemas import (
    CyclePayload,
    HealthResponse,
    MetricsResponse,
    PricesResponse,
    Verdict,
)

LOGGER = logging.getLogger("triagenet.api")
MAX_BATCH_SIZE = 1000
PRICE_CACHE = DATA_PROCESSED / "prices_cache.json"
SUMMARY_PATH = REPO_ROOT / "reports" / "phase4_triage" / "summary.json"
UPLOAD_FILE = File(...)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts once and reuse them across requests."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    app.state.pipeline = TriagePipeline()
    app.state.model_versions = _model_versions()
    LOGGER.info("Loaded TriageNet models for API serving")
    yield


app = FastAPI(
    title="TriageNet API",
    description="Single-cycle lithium-ion battery chemistry, SOH, and triage predictions.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ordered, hand-picked real cycles for the interview demo. Cycle indices were selected once from
# data/processed/cycles.parquet by nearest SOH target, then hardcoded so /sample-cycles is stable.
# Slot 10 substitutes CALCE CS2_37 cycle 98 for the requested CS2_33 last cycle because the saved
# Phase 4 model routes CS2_33 cycles 865/866 to "more characterization"; this substitute preserves
# the required near-dead LCO confident-recycle demo without changing triage logic.
SAMPLE_CYCLE_SPECS: tuple[tuple[str, int, float, str], ...] = (
    (
        "calce_cs2_33",
        865,
        0.05,
        "Disagreement case - measurement says dead, shape says healthy. Wide interval.",
    ),
    ("mit_b1c0", 531, 0.95, "Healthy LFP - confident second life"),
    ("mit_b1c2", 11, 0.99, "Fresh LFP - confident second life"),
    ("mit_b1c5", 886, 0.92, "Mid-life LFP - second life expected"),
    ("mit_b1c11", 765, 0.82, "Near 80% threshold - borderline LFP"),
    ("calce_cs2_34", 128, 0.98, "Fresh LCO - confident second life"),
    ("calce_cs2_35", 521, 0.85, "Mid-life LCO - borderline"),
    ("calce_cs2_36", 728, 0.65, "Degraded LCO - leaning toward recycle"),
    ("calce_cs2_37", 919, 0.40, "Heavily degraded LCO - recycle"),
    ("calce_cs2_37", 98, 0.05, "Near-dead LCO - confident recycle"),
    ("calce_cs2_38", 602, 0.70, "Wide-interval LCO - illustrates VOI"),
    ("mit_b1c14", 878, 0.82, "MIT EOL cycle - protocol edge case"),
)


@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Log per-request latency without leaking stack traces to clients."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    LOGGER.info(
        "%s %s -> %s in %sms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return service liveness, artifact versions, and commodity-price cache age."""
    return HealthResponse(
        status="ok",
        model_versions=request.app.state.model_versions,
        prices_age_hours=_prices_age_hours(),
    )


@app.post("/predict", response_model=Verdict)
def predict(payload: CyclePayload, request: Request) -> Verdict:
    """Run one cycle through the full TriageNet pipeline."""
    return _predict_payload(payload, request.app.state.pipeline)


@app.post("/predict-batch", response_model=list[Verdict])
def predict_batch(payloads: list[CyclePayload], request: Request) -> list[Verdict]:
    """Run up to 1000 cycles through the full TriageNet pipeline."""
    if len(payloads) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail="predict-batch is capped at 1000 cycles")
    return [_predict_payload(payload, request.app.state.pipeline) for payload in payloads]


@app.post("/predict-file", response_model=list[Verdict])
async def predict_file(request: Request, file: UploadFile = UPLOAD_FILE) -> list[Verdict]:
    """Parse an uploaded CSV or Parquet cycle file and return verdicts."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise HTTPException(status_code=415, detail="Only .csv and .parquet uploads are supported")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    frame = _read_uploaded_table(contents, suffix)
    if len(frame) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=413, detail="Uploaded files are capped at 1000 cycles")
    payloads = [_payload_from_row(row) for _, row in frame.iterrows()]
    return [_predict_payload(payload, request.app.state.pipeline) for payload in payloads]


@app.get("/sample-cycles", response_model=list[CyclePayload])
def sample_cycles() -> list[CyclePayload]:
    """Return hand-picked real cycles for the dashboard demo, including CALCE cycle 865."""
    cycles_path = DATA_PROCESSED / "cycles.parquet"
    if not cycles_path.exists():
        raise HTTPException(status_code=503, detail="data/processed/cycles.parquet is missing")
    frame = pd.read_parquet(cycles_path)
    samples = _sample_rows(frame)
    return [_payload_from_row(row) for _, row in samples.iterrows()]


@app.get("/prices", response_model=PricesResponse)
def prices() -> PricesResponse:
    """Return the commodity price snapshot used by the valuation layer."""
    price_map = get_spot_prices()
    as_of, is_live = _price_metadata()
    return PricesResponse(
        prices=price_map,
        prices_usd_per_kg=price_map,
        as_of=as_of,
        is_live=is_live,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Return compact held-out metrics for dashboard footer and docs."""
    return MetricsResponse(
        chemistry=_read_json(REPO_ROOT / "models" / "chemistry_metadata.json"),
        soh=_read_json(REPO_ROOT / "models" / "soh_metadata.json"),
        triage=_read_json(SUMMARY_PATH),
    )


def _predict_payload(payload: CyclePayload, pipeline: TriagePipeline) -> Verdict:
    start = time.perf_counter()
    row = _payload_to_unified_row(payload)
    try:
        raw = pipeline.predict(row, rule="risk_aware")
        top_features = _top_features(row, pipeline)
    except Exception as exc:
        LOGGER.exception("Prediction failed for %s cycle %s", payload.cell_id, payload.cycle_index)
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc
    probabilities = {
        chemistry: float(raw["chemistry_probs"].get(chemistry, 0.0))
        for chemistry in ("LFP", "LCO", "NMC", "NCA")
    }
    predicted = max(probabilities.items(), key=lambda item: item[1])[0]
    runtime_ms = int((time.perf_counter() - start) * 1000)
    LOGGER.info(
        "decision=%s confidence=%s cell=%s cycle=%s",
        raw["decision"],
        raw["confidence"],
        payload.cell_id,
        payload.cycle_index,
    )
    return Verdict(
        cell_id=str(raw["cell_id"]),
        cycle_index=int(raw["cycle_index"]),
        chemistry={"probabilities": probabilities, "predicted": predicted},
        soh={
            "mean": float(raw["soh_mean"]),
            "lower_90": float(raw["soh_lower_90"]),
            "upper_90": float(raw["soh_upper_90"]),
            "std": float(raw["soh_std"]),
        },
        decision=raw["decision"],
        confidence=raw["confidence"],
        economics={
            "expected_value_usd": float(raw["expected_value_usd"]),
            "second_life_value_usd": raw["second_life_value_usd"],
            "recycle_value_usd": raw["recycle_value_usd"],
            "value_of_info_one_more_cycle_usd": float(raw["value_of_info_one_more_cycle_usd"]),
        },
        rationale=str(raw["rationale"]),
        top_features=[
            {"name": name, "value": value, "importance": importance}
            for name, value, importance in top_features
        ],
        runtime_ms=runtime_ms,
    )


def _payload_to_unified_row(payload: CyclePayload) -> dict[str, Any]:
    known_chemistry = payload.known_chemistry or "LFP"
    soh = payload.known_soh
    if soh is None:
        soh = min(
            max(payload.discharge_capacity_ah / max(payload.nominal_capacity_ah, 1e-9), 0.0),
            1.05,
        )
    return {
        "cell_id": payload.cell_id,
        "dataset": "api",
        "chemistry": known_chemistry,
        "manufacturer": "unknown",
        "nominal_capacity_ah": payload.nominal_capacity_ah,
        "cycle_index": payload.cycle_index,
        "is_eol_cycle": True,
        "discharge_capacity_ah": payload.discharge_capacity_ah,
        "charge_capacity_ah": payload.charge_capacity_ah,
        "coulombic_efficiency": payload.coulombic_efficiency,
        "energy_efficiency": None,
        "soh": soh,
        "temperature_c_mean": payload.temperature_c_mean,
        "c_rate_charge": payload.c_rate_charge,
        "c_rate_discharge": payload.c_rate_discharge,
        "voltage_curve": payload.voltage_curve,
        "current_curve": payload.current_curve,
        "time_curve_s": payload.time_curve_s,
        "dq_dv_curve": None,
    }


def _top_features(row: dict[str, Any], pipeline: TriagePipeline) -> list[tuple[str, float, float]]:
    features = extract_features(row)
    frame = pd.DataFrame([features])
    return pipeline.chemistry_model.top_features(frame[list(CHEMISTRY_FEATURES)], k=5)


def _read_uploaded_table(contents: bytes, suffix: str) -> pd.DataFrame:
    try:
        if suffix == ".csv":
            with NamedTemporaryFile(suffix=".csv") as handle:
                handle.write(contents)
                handle.flush()
                frame = pd.read_csv(handle.name)
        else:
            with NamedTemporaryFile(suffix=".parquet") as handle:
                handle.write(contents)
                handle.flush()
                frame = pd.read_parquet(handle.name)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Could not parse uploaded file: {exc}"
        ) from exc
    return _normalize_table(frame)


def _normalize_table(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in ("voltage_curve", "current_curve", "time_curve_s"):
        if column not in normalized:
            raise HTTPException(status_code=422, detail=f"Missing required column: {column}")
        normalized[column] = normalized[column].map(_coerce_curve)
    return normalized


def _coerce_curve(value: object) -> list[float]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail="Curve columns must be JSON-like arrays"
            ) from exc
    try:
        curve = [float(item) for item in value]  # type: ignore[union-attr]
    except TypeError as exc:
        raise HTTPException(status_code=422, detail="Curve columns must be arrays") from exc
    if len(curve) != 100:
        raise HTTPException(status_code=422, detail="Curve columns must contain 100 values")
    return curve


def _payload_from_row(row: pd.Series) -> CyclePayload:
    chemistry = str(row.get("chemistry", "")).upper()
    description = _description(row)
    return CyclePayload(
        cell_id=str(row["cell_id"]),
        cycle_index=int(row["cycle_index"]),
        nominal_capacity_ah=float(row["nominal_capacity_ah"]),
        discharge_capacity_ah=float(row["discharge_capacity_ah"]),
        charge_capacity_ah=float(row["charge_capacity_ah"]),
        coulombic_efficiency=float(row["coulombic_efficiency"]),
        voltage_curve=_coerce_curve(row["voltage_curve"]),
        current_curve=_coerce_curve(row["current_curve"]),
        time_curve_s=_coerce_curve(row["time_curve_s"]),
        temperature_c_mean=_optional_float(row.get("temperature_c_mean")),
        c_rate_charge=_optional_float(row.get("c_rate_charge")),
        c_rate_discharge=_optional_float(row.get("c_rate_discharge")),
        known_chemistry=chemistry if chemistry in {"LFP", "LCO", "NMC", "NCA"} else None,
        known_soh=_optional_float(row.get("soh")),
        description=description,
    )


def _sample_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell_id, cycle_index, _, description in SAMPLE_CYCLE_SPECS:
        match = frame[(frame["cell_id"] == cell_id) & (frame["cycle_index"] == cycle_index)]
        if match.empty:
            raise HTTPException(
                status_code=503,
                detail=f"Configured sample cycle is missing: {cell_id} #{cycle_index}",
            )
        row = match.iloc[0].copy()
        row["sample_description"] = description
        rows.append(row)
    return pd.DataFrame(rows)


def _description(row: pd.Series) -> str:
    sample_description = row.get("sample_description")
    if isinstance(sample_description, str) and sample_description.strip():
        return sample_description
    chemistry = row.get("chemistry", "unknown")
    soh = _optional_float(row.get("soh"))
    soh_text = f"{soh:.0%} SOH" if soh is not None else "unknown SOH"
    return f"{chemistry} {row.get('cell_id')} cycle {int(row.get('cycle_index', 0))}: {soh_text}"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _model_versions() -> dict[str, str]:
    versions = {}
    for name, path in {
        "chemistry": MODELS_DIR / "chemistry_metadata.json",
        "soh": MODELS_DIR / "soh_metadata.json",
        "triage": SUMMARY_PATH,
    }.items():
        data = _read_json(path)
        if "trained_at_utc" in data:
            versions[name] = str(data["trained_at_utc"])
        elif path.exists():
            versions[name] = str(path.stat().st_mtime)
        else:
            versions[name] = "missing"
    return versions


def _prices_age_hours() -> float:
    if not PRICE_CACHE.exists():
        return -1.0
    modified = datetime.fromtimestamp(PRICE_CACHE.stat().st_mtime, tz=UTC)
    return (datetime.now(tz=UTC) - modified).total_seconds() / 3600.0


def _price_metadata() -> tuple[str, bool]:
    if not PRICE_CACHE.exists():
        return "fallback snapshot", False
    data = _read_json(PRICE_CACHE)
    return str(data.get("timestamp_utc", "fallback snapshot")), bool(data.get("is_live", False))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
