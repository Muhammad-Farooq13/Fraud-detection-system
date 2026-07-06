"""
Train and evaluate fraud detection models.

Handles:
  - Time-based train/val/test split (no shuffling — fraud detection is a
    temporal problem; shuffling would leak future information into training)
  - Leakage-safe merchant-category target encoding (fit on train only)
  - Class imbalance via class_weight (no naive oversampling that would
    duplicate the tiny fraud class into val/test)
  - Model comparison: Logistic Regression baseline vs. Gradient Boosting
  - Metrics appropriate for extreme imbalance: PR-AUC, ROC-AUC, recall,
    precision, F1 at a tuned threshold
  - MLflow experiment tracking
  - Persisted model + encoder artifacts for the serving API

Usage:
    python -m src.models.train --features data/processed/features.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import (
    NUMERIC_FEATURES,
    TARGET,
    add_merchant_category_risk,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path("models_store")


def time_based_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """Split chronologically: train on the past, validate/test on the future."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    logger.info(
        "Time-based split -> train=%d (%.1f%% fraud) val=%d (%.1f%% fraud) test=%d (%.1f%% fraud)",
        len(train_df),
        train_df[TARGET].mean() * 100,
        len(val_df),
        val_df[TARGET].mean() * 100,
        len(test_df),
        test_df[TARGET].mean() * 100,
    )
    return train_df, val_df, test_df


def prepare_splits(features_path: str):
    df = pd.read_parquet(features_path)
    train_df, val_df, test_df = time_based_split(df)

    # Fit merchant-category fraud-rate encoding on TRAIN ONLY, then apply to all splits
    train_df, category_rates = add_merchant_category_risk(train_df, fit=True)
    val_df, _ = add_merchant_category_risk(val_df, fit=False, category_rates=category_rates)
    test_df, _ = add_merchant_category_risk(test_df, fit=False, category_rates=category_rates)

    X_train, y_train = train_df[NUMERIC_FEATURES], train_df[TARGET]
    X_val, y_val = val_df[NUMERIC_FEATURES], val_df[TARGET]
    X_test, y_test = test_df[NUMERIC_FEATURES], test_df[TARGET]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), category_rates


def build_models() -> dict:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        max_depth=3,
                        learning_rate=0.08,
                        subsample=0.8,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def find_best_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """Pick the probability threshold maximizing F1 on the validation set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
    best_idx = int(np.nanargmax(f1s[:-1])) if len(thresholds) > 0 else 0
    return float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5, float(f1s[best_idx])


def evaluate(model, X, y, threshold: float) -> dict:
    y_scores = model.predict_proba(X)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, y_scores)), 4),
        "pr_auc": round(float(average_precision_score(y, y_scores)), 4),
        "precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "threshold": round(float(threshold), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("--features", type=str, default="data/processed/features.parquet")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="sqlite:///mlruns.db")
    parser.add_argument("--experiment-name", type=str, default="fraud-detection")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), category_rates = prepare_splits(
        args.features
    )

    models = build_models()
    results = {}
    best_model_name, best_pr_auc, best_model_obj, best_threshold = None, -1.0, None, 0.5

    for name, pipeline in models.items():
        with mlflow.start_run(run_name=name):
            logger.info("Training %s ...", name)
            pipeline.fit(X_train, y_train)

            val_scores = pipeline.predict_proba(X_val)[:, 1]
            threshold, val_f1 = find_best_threshold(y_val.values, val_scores)

            val_metrics = evaluate(pipeline, X_val, y_val, threshold)
            test_metrics = evaluate(pipeline, X_test, y_test, threshold)

            mlflow.log_params({"model": name, "threshold": threshold})
            mlflow.log_metrics(
                {f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, float)}
            )
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, float)}
            )
            mlflow.sklearn.log_model(pipeline, name)

            results[name] = {"val": val_metrics, "test": test_metrics}
            logger.info(
                "%s -> val_pr_auc=%.4f test_pr_auc=%.4f test_recall=%.4f test_precision=%.4f",
                name,
                val_metrics["pr_auc"],
                test_metrics["pr_auc"],
                test_metrics["recall"],
                test_metrics["precision"],
            )

            if val_metrics["pr_auc"] > best_pr_auc:
                best_pr_auc = val_metrics["pr_auc"]
                best_model_name = name
                best_model_obj = pipeline
                best_threshold = threshold

    logger.info("Best model selected: %s (val PR-AUC=%.4f)", best_model_name, best_pr_auc)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model_obj, ARTIFACT_DIR / "fraud_model.joblib")
    with open(ARTIFACT_DIR / "category_rates.json", "w", encoding="utf-8") as f:
        json.dump(category_rates, f, indent=2)
    with open(ARTIFACT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "threshold": best_threshold,
                "feature_order": NUMERIC_FEATURES,
                "results": results,
            },
            f,
            indent=2,
        )

    logger.info("Saved model artifacts to %s", ARTIFACT_DIR)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
