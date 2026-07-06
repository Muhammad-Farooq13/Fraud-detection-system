"""
FastAPI service for real-time fraud detection inference.

Loads the trained model artifact produced by `src/models/train.py` and
exposes a `/predict` endpoint that scores a single transaction, plus
`/health` and `/model-info` for operational monitoring.

Run locally:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Run via Docker:
    docker build -t fraud-api . && docker run -p 8000:8000 fraud-api
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path("models_store")
MODEL_PATH = ARTIFACT_DIR / "fraud_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
CATEGORY_RATES_PATH = ARTIFACT_DIR / "category_rates.json"

_state: dict = {"model": None, "metadata": None, "category_rates": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_artifacts()
    except RuntimeError as exc:
        # Fail loudly in logs but let the app come up so /health reports "not_ready"
        logger.error(str(exc))
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card transaction fraud scoring service.",
    version="1.0.0",
    lifespan=lifespan,
)


class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_category: str = Field(
        ..., description="Merchant category, e.g. 'grocery', 'crypto_exchange'"
    )
    is_night_transaction: Literal[0, 1] = Field(
        ..., description="1 if transaction occurred 00:00-05:59"
    )
    geo_velocity_km: float = Field(
        ..., ge=0, description="Distance from customer's last known location (km)"
    )
    is_foreign_country: Literal[0, 1]
    card_present: Literal[0, 1]
    customer_avg_amount: float = Field(
        ..., gt=0, description="Customer's historical average transaction amount"
    )
    txn_count_last_hour: int = Field(..., ge=0)
    customer_txn_seq: int = Field(
        0, ge=0, description="Number of prior transactions for this customer"
    )
    seconds_since_last_txn: float = Field(999_999, ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "amount": 1450.00,
                "merchant_category": "electronics",
                "is_night_transaction": 1,
                "geo_velocity_km": 320.5,
                "is_foreign_country": 1,
                "card_present": 0,
                "customer_avg_amount": 62.30,
                "txn_count_last_hour": 3,
                "customer_txn_seq": 45,
                "seconds_since_last_txn": 180.0,
            }
        }
    )


class PredictionResponse(BaseModel):
    is_fraud_prediction: int
    fraud_probability: float
    threshold_used: float
    risk_level: str
    latency_ms: float


def _load_artifacts() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run `python -m src.models.train` first."
        )
    _state["model"] = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _state["metadata"] = json.load(f)
    with open(CATEGORY_RATES_PATH, "r", encoding="utf-8") as f:
        _state["category_rates"] = json.load(f)
    logger.info(
        "Loaded model=%s threshold=%.4f",
        _state["metadata"]["best_model"],
        _state["metadata"]["threshold"],
    )


def _risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "critical"
    if prob >= 0.4:
        return "high"
    if prob >= 0.15:
        return "medium"
    return "low"


def _build_feature_row(txn: TransactionRequest) -> pd.DataFrame:
    category_rates = _state["category_rates"]
    global_rate = sum(category_rates.values()) / len(category_rates)
    merchant_category_fraud_rate = category_rates.get(txn.merchant_category, global_rate)

    amount_deviation_ratio = txn.amount / max(txn.customer_avg_amount, 1.0)
    amount_vs_customer_rolling_avg = txn.amount / max(txn.customer_avg_amount, 1.0)

    row = {
        "amount": txn.amount,
        "is_night_transaction": txn.is_night_transaction,
        "geo_velocity_km": txn.geo_velocity_km,
        "is_foreign_country": txn.is_foreign_country,
        "card_present": txn.card_present,
        "amount_deviation_ratio": amount_deviation_ratio,
        "txn_count_last_hour": txn.txn_count_last_hour,
        "customer_txn_seq": txn.customer_txn_seq,
        "customer_rolling_avg_amount": txn.customer_avg_amount,
        "amount_vs_customer_rolling_avg": amount_vs_customer_rolling_avg,
        "seconds_since_last_txn": txn.seconds_since_last_txn,
        "merchant_category_fraud_rate": merchant_category_fraud_rate,
    }
    feature_order = _state["metadata"]["feature_order"]
    return pd.DataFrame([row])[feature_order]


@app.get("/health")
def health() -> dict:
    ready = _state["model"] is not None
    return {"status": "ok" if ready else "not_ready", "model_loaded": ready}


@app.get("/model-info")
def model_info() -> dict:
    if _state["metadata"] is None:
        raise HTTPException(status_code=503, detail="Model metadata not loaded")
    meta = _state["metadata"]
    return {
        "best_model": meta["best_model"],
        "decision_threshold": meta["threshold"],
        "feature_order": meta["feature_order"],
        "test_metrics": meta["results"][meta["best_model"]]["test"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: TransactionRequest) -> PredictionResponse:
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    start = time.perf_counter()
    features = _build_feature_row(txn)
    proba = float(_state["model"].predict_proba(features)[:, 1][0])
    threshold = _state["metadata"]["threshold"]
    prediction = int(proba >= threshold)
    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    return PredictionResponse(
        is_fraud_prediction=prediction,
        fraud_probability=round(proba, 4),
        threshold_used=round(threshold, 4),
        risk_level=_risk_level(proba),
        latency_ms=latency_ms,
    )
