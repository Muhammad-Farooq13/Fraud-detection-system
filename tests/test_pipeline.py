"""Unit and integration tests for the fraud detection platform."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generate_data import GenerationConfig, generate_transactions
from src.features.build_features import (
    add_customer_rolling_features,
    add_merchant_category_risk,
)


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    config = GenerationConfig(n_transactions=5000, n_customers=300, fraud_rate=0.02, random_seed=7)
    return generate_transactions(config)


class TestDataGeneration:
    def test_row_count(self, sample_df):
        assert len(sample_df) == 5000

    def test_no_missing_values(self, sample_df):
        assert sample_df.isna().sum().sum() == 0

    def test_fraud_rate_close_to_target(self, sample_df):
        actual_rate = sample_df["is_fraud"].mean()
        assert abs(actual_rate - 0.02) < 0.002

    def test_unique_transaction_ids(self, sample_df):
        assert sample_df["transaction_id"].is_unique

    def test_amount_is_positive(self, sample_df):
        assert (sample_df["amount"] > 0).all()

    def test_labels_are_binary(self, sample_df):
        assert set(sample_df["is_fraud"].unique()).issubset({0, 1})

    def test_reproducible_with_same_seed(self):
        cfg = GenerationConfig(n_transactions=1000, random_seed=123)
        df1 = generate_transactions(cfg)
        df2 = generate_transactions(cfg)
        pd.testing.assert_frame_equal(df1, df2)


class TestFeatureEngineering:
    def test_rolling_avg_uses_only_past_transactions(self, sample_df):
        """The N-th transaction's rolling avg must not depend on the N-th amount itself."""
        df = add_customer_rolling_features(sample_df)
        # Pick a customer with multiple transactions
        cust_id = df["customer_id"].value_counts().index[0]
        cust_txns = df[df["customer_id"] == cust_id].sort_values("timestamp").reset_index(drop=True)
        assert len(cust_txns) >= 2

        # Manually recompute rolling avg for the 2nd transaction: should equal amount[0]
        expected_second_avg = cust_txns.loc[0, "amount"]
        actual_second_avg = cust_txns.loc[1, "customer_rolling_avg_amount"]
        assert abs(expected_second_avg - actual_second_avg) < 1e-6

    def test_first_transaction_has_no_negative_seconds_since_last(self, sample_df):
        df = add_customer_rolling_features(sample_df)
        assert (df["seconds_since_last_txn"] >= 0).all()

    def test_customer_txn_seq_starts_at_zero(self, sample_df):
        df = add_customer_rolling_features(sample_df)
        first_txns = df.sort_values("timestamp").groupby("customer_id").first()
        assert (first_txns["customer_txn_seq"] == 0).all()

    def test_category_encoding_no_leakage_on_unseen_category(self, sample_df):
        df = add_customer_rolling_features(sample_df)
        train_df, category_rates = add_merchant_category_risk(df, fit=True)
        # Simulate an unseen category at inference time
        fake_row = df.iloc[[0]].copy()
        fake_row["merchant_category"] = "totally_new_category"
        transformed, _ = add_merchant_category_risk(
            fake_row, fit=False, category_rates=category_rates
        )
        global_rate = np.mean(list(category_rates.values()))
        assert abs(transformed["merchant_category_fraud_rate"].iloc[0] - global_rate) < 1e-9

    def test_fit_false_requires_category_rates(self, sample_df):
        with pytest.raises(ValueError):
            add_merchant_category_risk(sample_df, fit=False, category_rates=None)


class TestAPI:
    @pytest.fixture(scope="class")
    @classmethod
    def client(cls):
        pytest.importorskip("fastapi")
        from pathlib import Path

        if not Path("models_store/fraud_model.joblib").exists():
            pytest.skip("Model artifact not trained yet; run src.models.train first")
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as c:
            yield c

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True

    def test_predict_high_risk_transaction(self, client):
        payload = {
            "amount": 5000.0,
            "merchant_category": "crypto_exchange",
            "is_night_transaction": 1,
            "geo_velocity_km": 500.0,
            "is_foreign_country": 1,
            "card_present": 0,
            "customer_avg_amount": 40.0,
            "txn_count_last_hour": 6,
            "customer_txn_seq": 10,
            "seconds_since_last_txn": 30.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["risk_level"] in {"low", "medium", "high", "critical"}

    def test_predict_rejects_invalid_amount(self, client):
        payload = {
            "amount": -10.0,
            "merchant_category": "grocery",
            "is_night_transaction": 0,
            "geo_velocity_km": 1.0,
            "is_foreign_country": 0,
            "card_present": 1,
            "customer_avg_amount": 20.0,
            "txn_count_last_hour": 0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_model_info_endpoint(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200
        assert "test_metrics" in response.json()
