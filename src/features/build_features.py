"""
Feature engineering for the fraud detection model.

Builds on the EDA findings (night-time risk, foreign-country risk, merchant
category risk, amount deviation) to construct a model-ready feature matrix,
including customer-level rolling aggregates that must be computed carefully
to avoid label leakage (all rolling stats use only *past* transactions for
each customer, ordered by timestamp).

Usage:
    python -m src.features.build_features --input data/raw/transactions.csv \
        --output data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = ["merchant_category"]
NUMERIC_FEATURES = [
    "amount",
    "is_night_transaction",
    "geo_velocity_km",
    "is_foreign_country",
    "card_present",
    "amount_deviation_ratio",
    "txn_count_last_hour",
    "customer_txn_seq",
    "customer_rolling_avg_amount",
    "amount_vs_customer_rolling_avg",
    "seconds_since_last_txn",
    "merchant_category_fraud_rate",
]
TARGET = "is_fraud"


def add_customer_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe, per-customer rolling features computed in timestamp order."""
    df = df.sort_values(["customer_id", "timestamp"]).copy()

    df["customer_txn_seq"] = df.groupby("customer_id").cumcount()

    # Expanding (cumulative, shifted by 1) mean amount per customer -> only past info
    grp = df.groupby("customer_id")["amount"]
    df["customer_rolling_avg_amount"] = grp.transform(lambda s: s.shift(1).expanding().mean())
    # First transaction per customer has no history -> fall back to global median
    global_median = df["amount"].median()
    df["customer_rolling_avg_amount"] = df["customer_rolling_avg_amount"].fillna(global_median)

    df["amount_vs_customer_rolling_avg"] = df["amount"] / df["customer_rolling_avg_amount"].replace(
        0, np.nan
    )
    df["amount_vs_customer_rolling_avg"] = df["amount_vs_customer_rolling_avg"].fillna(1.0)

    df["seconds_since_last_txn"] = df.groupby("customer_id")["timestamp"].diff().dt.total_seconds()
    # No prior transaction -> treat as a long gap (not suspicious by itself)
    df["seconds_since_last_txn"] = df["seconds_since_last_txn"].fillna(999_999)

    return df.sort_values("timestamp").reset_index(drop=True)


def add_merchant_category_risk(
    df: pd.DataFrame, fit: bool, category_rates: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Encode merchant category by historical fraud rate.

    IMPORTANT: rates must be learned on the TRAIN split only and reused
    (passed in via `category_rates`) when transforming validation/test data,
    otherwise this leaks target information across the split boundary.
    """
    if fit:
        category_rates = df.groupby("merchant_category")[TARGET].mean().to_dict()
        global_rate = df[TARGET].mean()
    else:
        if category_rates is None:
            raise ValueError("category_rates must be provided when fit=False")
        global_rate = np.mean(list(category_rates.values()))

    df = df.copy()
    df["merchant_category_fraud_rate"] = (
        df["merchant_category"].map(category_rates).fillna(global_rate)
    )
    return df, category_rates


def build_feature_frame(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    logger.info("Loaded raw data: %d rows", len(df))
    df = add_customer_rolling_features(df)
    logger.info("Added customer rolling features")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-ready features")
    parser.add_argument("--input", type=str, default="data/raw/transactions.csv")
    parser.add_argument("--output", type=str, default="data/processed/features.parquet")
    args = parser.parse_args()

    df = build_feature_frame(args.input)

    # NOTE: merchant_category_fraud_rate is fit inside the training split logic
    # (src/models/train.py) to avoid leakage; here we just persist the base
    # feature frame with rolling features, keeping merchant_category as a
    # raw categorical column for downstream fitting.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(
        "Saved feature frame to %s (%d rows, %d cols)",
        output_path,
        len(df),
        df.shape[1],
    )


if __name__ == "__main__":
    main()
