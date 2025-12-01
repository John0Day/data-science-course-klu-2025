#!/usr/bin/env python3
"""
Baseline classifier for flight delays.

Trains a simple model (logistic regression or random forest) to predict whether
an arrival will be delayed (arr_delay >= 15 minutes). Includes common metrics:
accuracy, precision, recall, F1, confusion matrix, and a short class report.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = Path("rawdata/flight_data_2024.csv")
SAMPLE_PATH = Path("rawdata/flight_data_2024_sample.csv")
ARRIVAL_DELAY_THRESHOLD = 15

USECOLS = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "fl_date",
    "op_unique_carrier",
    "op_carrier_fl_num",
    "origin",
    "dest",
    "crs_dep_time",
    "arr_delay",
    "cancelled",
    "diverted",
    "distance",
]

DTYPES = {
    "year": "int16",
    "month": "int8",
    "day_of_month": "int8",
    "day_of_week": "int8",
    "op_unique_carrier": "category",
    "op_carrier_fl_num": "float32",
    "origin": "category",
    "dest": "category",
    "crs_dep_time": "Int32",
    "arr_delay": "float32",
    "cancelled": "int8",
    "diverted": "int8",
    "distance": "float32",
}

FEATURE_COLS = [
    "month",
    "day_of_week",
    "dep_hour",
    "distance",
    "op_unique_carrier",
    "origin",
    "dest",
]

NUMERIC_FEATURES = ["month", "day_of_week", "dep_hour", "distance"]
CATEGORICAL_FEATURES = ["op_unique_carrier", "origin", "dest"]


def load_data(use_sample: bool, nrows: int | None) -> pd.DataFrame:
    """
    Load data, drop cancellations/diversions, create binary target is_delayed.
    """
    path = SAMPLE_PATH if use_sample else DATA_PATH
    df = pd.read_csv(
        path,
        usecols=USECOLS,
        dtype=DTYPES,
        nrows=nrows,
        low_memory=False,
    )

    df = df[(df["cancelled"] == 0) & (df["diverted"] == 0)]
    df["arr_delay"] = pd.to_numeric(df["arr_delay"], errors="coerce")
    df = df[df["arr_delay"].notna()]
    df["is_delayed"] = df["arr_delay"] >= ARRIVAL_DELAY_THRESHOLD

    df["dep_hour"] = pd.to_numeric(
        df["crs_dep_time"] // 100, errors="coerce"
    ).fillna(-1)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    df = df.dropna(subset=FEATURE_COLS + ["is_delayed"])
    return df


def build_pipeline(model_type: str, random_state: int) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    # scikit-learn 1.2+ uses sparse_output, older versions use sparse
    try:
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=True
        )
    except TypeError:
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse=True
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
    else:
        model = LogisticRegression(
            max_iter=200,
            n_jobs=-1,
            class_weight="balanced",
            solver="lbfgs",
        )

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_and_evaluate(
    df: pd.DataFrame,
    model_type: str,
    test_size: float,
    random_state: int,
) -> Tuple[Pipeline, dict]:
    X = df[FEATURE_COLS]
    y = df["is_delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline = build_pipeline(model_type, random_state)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, digits=3
        ),
    }
    return pipeline, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a baseline classifier to predict flight arrival delays."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use the smaller sample file instead of the full dataset.",
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Limit number of rows read (useful for quick experiments).",
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "rf"],
        default="logreg",
        help="Model type: logistic regression (logreg) or random forest (rf).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splits and models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(use_sample=args.sample, nrows=args.nrows)

    print(
        f"Loaded {len(df):,} rows after cleaning. Positive rate (delayed): "
        f"{df['is_delayed'].mean() * 100:.2f}%"
    )
    model, metrics = train_and_evaluate(
        df,
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\nMetrics:")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])
    print("\nClassification report:")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
