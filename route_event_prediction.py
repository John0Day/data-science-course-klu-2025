#!/usr/bin/env python3
"""
Route/event-level delay prediction for 2024 flight data.

Predicts whether a flight will be delayed (arr_delay >= 15) using route,
schedule, carrier, and event-day indicators (from combined_schedules_with_airports.csv).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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

from event_utils import load_event_lookup

DATA_PATH = Path(__file__).resolve().parent / "rawdata" / "flight_data_2024.csv"
SAMPLE_PATH = Path(__file__).resolve().parent / "rawdata" / "flight_data_2024_sample.csv"
EVENTS_PATH = Path(__file__).resolve().parent / "rawdata" / "combined_schedules_with_airports.csv"

ARRIVAL_DELAY_THRESHOLD = 15

CATEGORICAL_FEATURES = [
    "origin",
    "dest",
    "op_unique_carrier",
    "event_name",
]
NUMERIC_FEATURES = [
    "month",
    "day_of_week",
    "dep_hour",
    "distance",
    "event_flag",
]


def load_data(use_sample: bool, nrows: int | None, events_lookup: dict) -> pd.DataFrame:
    path = SAMPLE_PATH if use_sample else DATA_PATH
    df = pd.read_csv(path, low_memory=False, nrows=nrows)

    # Basic cleaning
    df = df[(df["cancelled"] == 0) & (df["diverted"] == 0)]
    df["arr_delay"] = pd.to_numeric(df["arr_delay"], errors="coerce")
    df = df[df["arr_delay"].notna()]
    df["is_delayed"] = df["arr_delay"] >= ARRIVAL_DELAY_THRESHOLD

    df["dep_hour"] = pd.to_numeric(df["crs_dep_time"] // 100, errors="coerce")
    df["dep_hour"] = df["dep_hour"].fillna(-1)

    df["event_name"] = "none"
    if events_lookup:
        origin_keys = pd.Series(
            list(zip(pd.to_datetime(df["fl_date"]).dt.date, df["origin"])), index=df.index
        )
        df["event_name"] = origin_keys.map(events_lookup).fillna("none")
        no_event_mask = df["event_name"] == "none"
        if no_event_mask.any():
            dest_keys = pd.Series(
                list(zip(pd.to_datetime(df.loc[no_event_mask, "fl_date"]).dt.date, df.loc[no_event_mask, "dest"])),
                index=df.index[no_event_mask],
            )
            df.loc[no_event_mask, "event_name"] = dest_keys.map(events_lookup).fillna("none")
    df["event_flag"] = (df["event_name"] != "none").astype(int)

    # Ensure numeric dtype
    for col in ["month", "day_of_week", "distance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["is_delayed"])
    return df


def build_pipeline(model_type: str, random_state: int) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    # scikit-learn 1.2+ uses sparse_output; older versions use sparse
    try:
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=True
        )
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=True)

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
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    else:
        model = LogisticRegression(
            max_iter=300,
            n_jobs=-1,
            class_weight="balanced",
            solver="lbfgs",
        )

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def train_and_eval(
    df: pd.DataFrame,
    model_type: str,
    test_size: float,
    random_state: int,
) -> Tuple[Pipeline, pd.DataFrame, pd.Series, pd.Series]:
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y = df["is_delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
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
        "classification_report": classification_report(y_test, y_pred, digits=3),
    }

    print("\nMetrics:")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k.capitalize()}: {metrics[k]:.3f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(metrics["confusion_matrix"])
    print("\nClassification report:")
    print(metrics["classification_report"])

    return pipeline, X_test, y_test, y_pred


def route_risk_summary(
    df_test: pd.DataFrame,
    y_test: pd.Series,
    pipeline: Pipeline,
    top_n: int,
    min_flights: int,
) -> None:
    proba = pipeline.predict_proba(df_test)[:, 1]
    df_routes = df_test.copy()
    df_routes["is_delayed"] = y_test.values
    df_routes["p_delay"] = proba
    df_routes["route"] = df_routes["origin"] + "-" + df_routes["dest"]

    agg = df_routes.groupby("route").agg(
        flights=("route", "count"),
        delayed=("is_delayed", "sum"),
        p_mean=("p_delay", "mean"),
    )
    agg["delay_rate"] = agg["delayed"] / agg["flights"]
    agg = agg[agg["flights"] >= min_flights]
    agg = agg.sort_values("p_mean", ascending=False)

    print(f"\nTop {top_n} routes by predicted delay probability (min {min_flights} flights in test):")
    print(agg.head(top_n)[["flights", "delayed", "delay_rate", "p_mean"]].to_string())

    # Event vs non-event comparison on test set
    if "event_flag" in df_test.columns:
        event_df = df_routes.groupby("event_name").agg(
            flights=("is_delayed", "count"),
            delayed=("is_delayed", "sum"),
            p_mean=("p_delay", "mean"),
        )
        event_df["delay_rate"] = event_df["delayed"] / event_df["flights"]
        event_df = event_df.sort_values("p_mean", ascending=False)
        print("\nPredicted delay probability by event name (test set):")
        print(event_df.head(20).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a route/event-level delay classifier."
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
        help="Model type: logistic regression or random forest.",
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
        help="Random seed.",
    )
    parser.add_argument(
        "--top-routes",
        type=int,
        default=10,
        help="How many routes to display in the route risk summary.",
    )
    parser.add_argument(
        "--min-route-flights",
        type=int,
        default=50,
        help="Minimum flights per route (in test) to include in the route summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events_lookup = load_event_lookup(EVENTS_PATH)
    df = load_data(use_sample=args.sample, nrows=args.nrows, events_lookup=events_lookup)

    print(
        f"Loaded {len(df):,} rows after cleaning. Positive rate (delayed): "
        f"{df['is_delayed'].mean() * 100:.2f}%"
    )
    pipeline, X_test, y_test, y_pred = train_and_eval(
        df,
        model_type=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    route_risk_summary(
        X_test,
        y_test,
        pipeline,
        top_n=args.top_routes,
        min_flights=args.min_route_flights,
    )


if __name__ == "__main__":
    main()
