"""
End-to-end modeling script for 2024 flight delay data.

This version extends the original Modeling_Pipeline with **result visualizations**:
Classification:
- ROC curves + AUC (saved as PNG when --diagnostics)
- Confusion matrices (LogReg, RandomForest, LightGBM @0.5; plus LGBM thresholds 0.3/0.5/0.7)
- Profit / Expected Value curve vs threshold (LightGBM)
- LightGBM global feature importance plot (bar chart)

LightGBM diagnostics (when --diagnostics):
- Learning curve (F1 vs training size)
- Complexity curve (F1 vs max_depth)

Regression:
- Actual vs Predicted scatter
- Residual distribution plot

Clustering:
- Cluster scatter (distance vs dep_hour), colored by cluster (on subsample)
- Elbow method plot (SSE/inertia vs K) (on subsample)
- Silhouette score vs K plot (on subsample)

All plots are saved to ./plots/ (created if missing).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb  # pip install lightgbm

# Optional plotting for diagnostics/visualizations
try:  # pragma: no cover
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths and data options
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent):
    csv_dir = candidate / "CSV Files"
    if csv_dir.exists():
        PROJECT_ROOT = candidate  # kept for possible future use
        DATA_DIR = csv_dir
        break
else:  # pragma: no cover
    raise FileNotFoundError("Could not locate 'CSV Files' directory (expected 'CSV Files').")

DATA_PATH = DATA_DIR / "flight_data_2024.csv"
SAMPLE_PATH = DATA_DIR / "flight_data_2024_sample.csv"

PLOTS_DIR = Path("plots")

ARRIVAL_DELAY_THRESHOLD = 15  # minutes for classification target
DEFAULT_CHUNK_SIZE = 250_000  # rows per chunk when streaming

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------
CATEGORICAL_FEATURES = [
    "op_unique_carrier",
    "origin",
    "origin_state_nm",
    "dest",
    "dest_state_nm",
]

RAW_NUMERIC_COLS = [
    "year",
    "month",
    "day_of_month",
    "day_of_week",
    "crs_dep_time",
    "crs_arr_time",
    "crs_elapsed_time",
    "distance",
]

ENGINEERED_NUMERIC_COLS = [
    "dep_hour",
    "day_of_year",
    "is_weekend",
    "dep_hour_sin",
    "dep_hour_cos",
]

NUMERIC_FEATURES = RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS

CLUSTER_FEATURES = ["distance", "crs_elapsed_time", "crs_dep_time", "dep_hour"]

DTYPES = {
    "year": "int16",
    "month": "int8",
    "day_of_month": "int8",
    "day_of_week": "int8",
    "crs_dep_time": "int16",
    "crs_arr_time": "int16",
    "crs_elapsed_time": "float32",
    "distance": "float32",
    "arr_delay": "float32",
    "op_unique_carrier": "category",
    "origin": "category",
    "origin_state_nm": "category",
    "dest": "category",
    "dest_state_nm": "category",
}

USECOLS = CATEGORICAL_FEATURES + RAW_NUMERIC_COLS + ["arr_delay", "fl_date"]

# ---------------------------------------------------------------------------
# Expected-value cost defaults (can be overridden via CLI)
# ---------------------------------------------------------------------------
DEFAULT_COST_TP = 100.0
DEFAULT_COST_FP = -10.0
DEFAULT_COST_FN = -200.0
DEFAULT_COST_TN = 0.0


# ---------------------------------------------------------------------------
# Utilities: plotting helpers
# ---------------------------------------------------------------------------
def ensure_plots_dir() -> None:
    if MATPLOTLIB_AVAILABLE:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    """Save current figure to plots/NAME.png with sane defaults."""
    if not MATPLOTLIB_AVAILABLE:
        return
    ensure_plots_dir()
    out = PLOTS_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    print(f"Saved plot to '{out.as_posix()}'.")


def plot_confusion_matrix_png(cm: np.ndarray, title: str, filename: str) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # annotate
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha="center", va="center")
    plt.xticks([0, 1], ["On-time", "Delayed"])
    plt.yticks([0, 1], ["On-time", "Delayed"])
    savefig(filename)


def plot_feature_importance_png(model: lgb.LGBMClassifier, feature_names: List[str], top_n: int = 20) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    importances = model.feature_importances_
    series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, 6))
    plt.barh(series.index[::-1], series.values[::-1])
    plt.xlabel("Importance")
    plt.title(f"LightGBM Global Feature Importance (Top {top_n})")
    savefig("feature_importance_lightgbm.png")


def plot_profit_curve_png(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_tp: float,
    cost_fp: float,
    cost_fn: float,
    cost_tn: float,
    mark_thresholds: List[float] = [0.3, 0.5, 0.7],
    filename: str = "profit_curve_lightgbm.png",
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    thresholds = np.linspace(0.0, 1.0, 101)
    evs = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        evs.append(compute_expected_value(y_true, y_pred, cost_tp, cost_fp, cost_fn, cost_tn))
    plt.figure()
    plt.plot(thresholds, evs)
    for thr in mark_thresholds:
        idx = int(round(thr * 100))
        plt.scatter([thresholds[idx]], [evs[idx]])
        plt.text(thresholds[idx], evs[idx], f" thr={thr:.1f}", va="bottom")
    best_idx = int(np.argmax(evs))
    plt.scatter([thresholds[best_idx]], [evs[best_idx]])
    plt.text(thresholds[best_idx], evs[best_idx], " best", va="bottom")
    plt.xlabel("Probability threshold")
    plt.ylabel("Expected value per flight")
    plt.title("Profit / Expected Value Curve (LightGBM)")
    plt.grid(True, alpha=0.3)
    savefig(filename)


def plot_regression_scatter_png(y_true: np.ndarray, y_pred: np.ndarray, filename: str) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.figure()
    plt.scatter(y_true, y_pred, s=6, alpha=0.3)
    # 45-degree reference line
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Actual arrival delay (min)")
    plt.ylabel("Predicted arrival delay (min)")
    plt.title("Actual vs Predicted (Regression)")
    plt.grid(True, alpha=0.2)
    savefig(filename)


def plot_residuals_png(residuals: np.ndarray, filename: str) -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.figure()
    plt.hist(residuals, bins=60)
    plt.xlabel("Residual (actual - predicted) minutes")
    plt.ylabel("Count")
    plt.title("Residual Distribution (Regression)")
    plt.grid(True, alpha=0.2)
    savefig(filename)


def plot_elbow_and_silhouette_png(X_scaled: np.ndarray, k_range: List[int]) -> None:
    """Compute and plot elbow (inertia) and silhouette scores for a range of k."""
    if not MATPLOTLIB_AVAILABLE:
        return
    inertias = []
    silhouettes = []
    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10_000, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        # silhouette can be expensive; kmeans on scaled subsample makes it manageable
        silhouettes.append(silhouette_score(X_scaled, labels))

    plt.figure()
    plt.plot(k_range, inertias, marker="o")
    plt.xlabel("K (number of clusters)")
    plt.ylabel("SSE / Inertia")
    plt.title("Elbow Method (MiniBatchKMeans)")
    plt.grid(True, alpha=0.3)
    savefig("kmeans_elbow.png")

    plt.figure()
    plt.plot(k_range, silhouettes, marker="o")
    plt.xlabel("K (number of clusters)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Scores (MiniBatchKMeans)")
    plt.grid(True, alpha=0.3)
    savefig("kmeans_silhouette.png")


def plot_cluster_scatter_png(df: pd.DataFrame, labels: np.ndarray, filename: str) -> None:
    """2D scatter for clustering (distance vs dep_hour), colored by cluster id."""
    if not MATPLOTLIB_AVAILABLE:
        return
    if "distance" not in df.columns or "dep_hour" not in df.columns:
        return
    plt.figure()
    plt.scatter(df["distance"].values, df["dep_hour"].values, c=labels, s=6, alpha=0.35)
    plt.xlabel("Distance")
    plt.ylabel("Departure hour")
    plt.title("Clusters (distance vs dep_hour)")
    plt.grid(True, alpha=0.2)
    savefig(filename)


# ---------------------------------------------------------------------------
# Data loading and feature preparation
# ---------------------------------------------------------------------------
def read_in_chunks(
    use_sample: bool = False,
    nrows: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterable[pd.DataFrame]:
    """Yield pandas DataFrames in manageable chunks to stream a large CSV."""
    path = SAMPLE_PATH if use_sample else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    reader = pd.read_csv(
        path,
        usecols=USECOLS,
        dtype=DTYPES,
        na_values=["", "NA", "NaN"],
        chunksize=chunk_size,
        nrows=nrows,
        low_memory=False,
    )
    if isinstance(reader, pd.DataFrame):
        yield reader
    else:
        yield from reader


def prepare_features(chunks: Iterable[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Engineer features, clean data, impute missing values, concat into one DataFrame."""
    processed_chunks: list[pd.DataFrame] = []
    for chunk in chunks:
        if "fl_date" in chunk.columns:
            chunk["fl_date"] = pd.to_datetime(chunk["fl_date"], errors="coerce")

        chunk["dep_hour"] = (chunk["crs_dep_time"] // 100).astype("int8")

        if "fl_date" in chunk.columns:
            chunk["day_of_year"] = chunk["fl_date"].dt.dayofyear.astype("Int16")
        else:
            chunk["day_of_year"] = pd.NA

        chunk["is_weekend"] = chunk["day_of_week"].isin([6, 7]).astype("Int8")

        hour = chunk["dep_hour"].astype("float32")
        chunk["dep_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        chunk["dep_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        for col in RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("float32")
                chunk[col] = chunk[col].fillna(chunk[col].median())

        for col in CATEGORICAL_FEATURES:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("string").fillna("Unknown")

        chunk_clean = chunk.dropna(subset=["arr_delay"]).copy()
        processed_chunks.append(chunk_clean)

    df_clean = pd.concat(processed_chunks, ignore_index=True)

    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype("category")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df_clean[feature_cols]
    return df_clean, X


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_classification_metrics(name: str, y_true, y_pred) -> None:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"{name:15s} Acc: {acc:5.3f} Prec: {prec:5.3f} Rec: {rec:5.3f} F1: {f1:5.3f}")


def compute_expected_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_tp: float,
    cost_fp: float,
    cost_fn: float,
    cost_tn: float,
) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    total = tp * cost_tp + fp * cost_fp + fn * cost_fn + tn * cost_tn
    return total / len(y_true)


def subsample(
    X: pd.DataFrame,
    y: pd.Series,
    max_samples: int = 200_000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    n = len(X)
    if n <= max_samples:
        return X, y
    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=max_samples, replace=False)
    return X.iloc[idx], y.iloc[idx]


def run_learning_curve_lightgbm(X: pd.DataFrame, y: pd.Series) -> None:
    print("\n--- LightGBM Learning Curve (F1 vs training size) ---")
    X_sub, y_sub = subsample(X, y, max_samples=200_000, random_state=42)
    estimator = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1,
    )
    train_sizes_frac = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_sub,
        y_sub,
        train_sizes=train_sizes_frac,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        fit_params={"categorical_feature": CATEGORICAL_FEATURES},
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    print("Train size | F1_train | F1_val")
    for n, t, v in zip(train_sizes, train_mean, val_mean):
        print(f"{int(n):9d} | {t:8.3f} | {v:6.3f}")

    if MATPLOTLIB_AVAILABLE:
        plt.figure()
        plt.plot(train_sizes, train_mean, marker="o", label="Train F1")
        plt.plot(train_sizes, val_mean, marker="s", label="CV F1")
        plt.xlabel("Training set size")
        plt.ylabel("F1 score")
        plt.title("LightGBM Learning Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        savefig("learning_curve_lightgbm.png")
    else:
        print("matplotlib not available; skipping learning curve plot.")


def run_complexity_curve_lightgbm(X: pd.DataFrame, y: pd.Series) -> None:
    print("\n--- LightGBM Complexity Curve (F1 vs max_depth) ---")
    X_sub, y_sub = subsample(X, y, max_samples=300_000, random_state=123)
    X_train, X_val, y_train, y_val = train_test_split(
        X_sub, y_sub, test_size=0.2, random_state=123, stratify=y_sub
    )

    depths = [3, 5, 7, 9, 11]
    print("max_depth | F1_train | F1_val")
    train_scores = []
    val_scores = []
    for d in depths:
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=d,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        f1_train = f1_score(y_train, y_train_pred, zero_division=0)
        f1_val = f1_score(y_val, y_val_pred, zero_division=0)
        train_scores.append(f1_train)
        val_scores.append(f1_val)
        print(f"{d:9d} | {f1_train:8.3f} | {f1_val:6.3f}")

    if MATPLOTLIB_AVAILABLE:
        plt.figure()
        plt.plot(depths, train_scores, marker="o", label="Train F1")
        plt.plot(depths, val_scores, marker="s", label="Val F1")
        plt.xlabel("max_depth")
        plt.ylabel("F1 score")
        plt.title("LightGBM Complexity Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        savefig("complexity_curve_lightgbm.png")
    else:
        print("matplotlib not available; skipping complexity curve plot.")


def print_lightgbm_feature_importance(model: lgb.LGBMClassifier, feature_names, top_n: int = 20) -> None:
    importances = model.feature_importances_
    series = pd.Series(importances, index=list(feature_names)).sort_values(ascending=False)
    print(f"\nTop {top_n} LightGBM feature importances:")
    print(series.head(top_n).to_string())


def tune_lightgbm_hyperparams(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    print("\n--- Hyperparameter search for LightGBM (F1, subsampled) ---")
    X_sub, y_sub = subsample(X_train, y_train, max_samples=250_000, random_state=2025)
    base = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    param_grid = {
        "max_depth": [5, 8],
        "num_leaves": [31, 63],
        "min_child_samples": [20, 50],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }
    grid = GridSearchCV(estimator=base, param_grid=param_grid, scoring="f1", cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_sub, y_sub, categorical_feature=CATEGORICAL_FEATURES)
    print(f"Best LightGBM params (subsampled CV): {grid.best_params_}")
    print(f"Best CV F1: {grid.best_score_:.3f}")
    return grid.best_params_


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def run_classification(
    df_clean: pd.DataFrame,
    X: pd.DataFrame,
    diagnostics: bool = False,
    ev_costs: Optional[Tuple[float, float, float, float]] = None,
    plots: bool = False,
) -> None:
    print("\n=== Classification: Predicting 15+ minute arrival delay ===")
    Y_class = (df_clean["arr_delay"] >= ARRIVAL_DELAY_THRESHOLD).astype(int)

    pos_rate = Y_class.mean()
    print(f"Positive rate (delayed >= {ARRIVAL_DELAY_THRESHOLD} min): {pos_rate:.3f} ({pos_rate*100:.1f}%)")
    print(f"Naive baseline (always on-time) accuracy: {1 - pos_rate:.3f} ({(1-pos_rate)*100:.1f}%)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y_class, test_size=0.2, random_state=42, stratify=Y_class
    )

    # Numeric-only view
    X_train_num = X_train.select_dtypes(include=[np.number])
    X_test_num = X_test.select_dtypes(include=[np.number])

    # Standardize numeric features for linear models
    scaler_clf = StandardScaler()
    X_train_num_scaled = scaler_clf.fit_transform(X_train_num)
    X_test_num_scaled = scaler_clf.transform(X_test_num)

    # Logistic Regression
    log_reg = LogisticRegression(solver="liblinear", max_iter=200, random_state=42, class_weight="balanced")
    log_reg.fit(X_train_num_scaled, y_train)
    y_pred_log = log_reg.predict(X_test_num_scaled)
    print_classification_metrics("LogReg", y_test, y_pred_log)
    y_proba_log = log_reg.predict_proba(X_test_num_scaled)[:, 1]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_train_num, y_train)
    y_pred_rf = rf.predict(X_test_num)
    print_classification_metrics("RandomForest", y_test, y_pred_rf)
    y_proba_rf = rf.predict_proba(X_test_num)[:, 1]

    # LightGBM (tuned)
    best_params = tune_lightgbm_hyperparams(X_train, y_train)
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
        **best_params,
    )
    lgb_clf.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)

    y_proba_lgb = lgb_clf.predict_proba(X_test)[:, 1]
    y_pred_lgb_05 = (y_proba_lgb >= 0.5).astype(int)
    print_classification_metrics("LightGBM@0.5", y_test, y_pred_lgb_05)

    # ROC-AUC scores
    auc_log = roc_auc_score(y_test, y_proba_log)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    auc_lgb = roc_auc_score(y_test, y_proba_lgb)
    print("\nROC-AUC scores (test set):")
    print(f" LogReg AUC: {auc_log:5.3f}")
    print(f" RandomForest AUC: {auc_rf:5.3f}")
    print(f" LightGBM AUC: {auc_lgb:5.3f}")

    # Expected-value costs
    if ev_costs is not None:
        cost_tp, cost_fp, cost_fn, cost_tn = ev_costs
    else:
        cost_tp, cost_fp, cost_fn, cost_tn = (DEFAULT_COST_TP, DEFAULT_COST_FP, DEFAULT_COST_FN, DEFAULT_COST_TN)

    print(f"\nExpected-value costs used: TP={cost_tp}, FP={cost_fp}, FN={cost_fn}, TN={cost_tn}")

    # Threshold sweep on LightGBM probabilities for metrics + EV
    for thr in [0.3, 0.5, 0.7]:
        y_pred_thr = (y_proba_lgb >= thr).astype(int)
        print_classification_metrics(f"LGBM@thr={thr:.1f}", y_test, y_pred_thr)
        ev = compute_expected_value(y_test, y_pred_thr, cost_tp, cost_fp, cost_fn, cost_tn)
        print(f"{'':15s} Expected value per flight: {ev:7.2f}")

    # Feature importance text + plot
    print_lightgbm_feature_importance(lgb_clf, X_train.columns, top_n=20)

    # ------------------ Visualizations ------------------
    if plots and MATPLOTLIB_AVAILABLE:
        # Confusion matrices
        cm_log = confusion_matrix(y_test, y_pred_log)
        cm_rf = confusion_matrix(y_test, y_pred_rf)
        cm_lgb = confusion_matrix(y_test, y_pred_lgb_05)

        plot_confusion_matrix_png(cm_log, "Confusion Matrix - LogReg", "cm_logreg.png")
        plot_confusion_matrix_png(cm_rf, "Confusion Matrix - RandomForest", "cm_random_forest.png")
        plot_confusion_matrix_png(cm_lgb, "Confusion Matrix - LightGBM @ 0.5", "cm_lightgbm_thr_0_5.png")

        for thr in [0.3, 0.7]:
            y_pred_thr = (y_proba_lgb >= thr).astype(int)
            cm_thr = confusion_matrix(y_test, y_pred_thr)
            plot_confusion_matrix_png(cm_thr, f"Confusion Matrix - LightGBM @ {thr:.1f}", f"cm_lightgbm_thr_{thr:.1f}.png".replace(".", "_"))

        # Profit curve (EV vs threshold) on a manageable subsample of the test set
        y_test_sub = np.asarray(y_test)
        y_proba_sub = np.asarray(y_proba_lgb)
        if len(y_test_sub) > 250_000:
            rng = np.random.RandomState(7)
            idx = rng.choice(len(y_test_sub), size=250_000, replace=False)
            y_test_sub = y_test_sub[idx]
            y_proba_sub = y_proba_sub[idx]
        plot_profit_curve_png(y_test_sub, y_proba_sub, cost_tp, cost_fp, cost_fn, cost_tn)

        # Feature importance plot
        plot_feature_importance_png(lgb_clf, list(X_train.columns), top_n=20)

    # Advanced diagnostics (ROC plot + learning/complexity curves)
    if diagnostics:
        if MATPLOTLIB_AVAILABLE:
            fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
            fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
            fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_proba_lgb)

            plt.figure()
            plt.plot(fpr_log, tpr_log, label=f"LogReg (AUC={auc_log:.3f})")
            plt.plot(fpr_rf, tpr_rf, label=f"RandomForest (AUC={auc_rf:.3f})")
            plt.plot(fpr_lgb, tpr_lgb, label=f"LightGBM (AUC={auc_lgb:.3f})")
            plt.plot([0, 1], [0, 1], "k--", label="Random guess")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves - Delay Classification")
            plt.legend()
            plt.grid(True, alpha=0.3)
            savefig("roc_curves_classification.png")
        else:
            print("matplotlib not available; skipping ROC curve plot.")

        run_learning_curve_lightgbm(X_train, y_train)
        run_complexity_curve_lightgbm(X_train, y_train)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
def run_regression(df_clean: pd.DataFrame, X: pd.DataFrame, plots: bool = False) -> None:
    print("\n=== Regression: Predicting arrival delay (minutes) ===")
    Y_reg = df_clean["arr_delay"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y_reg, test_size=0.2, random_state=42)

    X_train_num = X_train.select_dtypes(include=[np.number])
    X_test_num = X_test.select_dtypes(include=[np.number])

    scaler_reg = StandardScaler()
    X_train_num_scaled = scaler_reg.fit_transform(X_train_num)
    X_test_num_scaled = scaler_reg.transform(X_test_num)

    baseline_pred = np.full_like(y_test, y_train.mean(), dtype=float)
    rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_pred))
    mae_baseline = mean_absolute_error(y_test, baseline_pred)
    print(f"Baseline (mean delay) - RMSE: {rmse_baseline:.2f}, MAE: {mae_baseline:.2f}")

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_num_scaled, y_train)
    y_pred_lin = lin_reg.predict(X_test_num_scaled)
    rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
    mae_lin = mean_absolute_error(y_test, y_pred_lin)
    print(f"Linear Regression - RMSE: {rmse_lin:.2f}, MAE: {mae_lin:.2f}")

    gb_reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    gb_reg.fit(X_train_num, y_train)
    y_pred_gb = gb_reg.predict(X_test_num)
    rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    print(f"Gradient Boosting - RMSE: {rmse_gb:.2f}, MAE: {mae_gb:.2f}")

    lgb_reg = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_reg.fit(X_train, y_train, categorical_feature=CATEGORICAL_FEATURES)
    y_pred_lgb = lgb_reg.predict(X_test)
    rmse_lgb = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
    print(f"LightGBM Regressor - RMSE: {rmse_lgb:.2f}, MAE: {mae_lgb:.2f}")

    if plots and MATPLOTLIB_AVAILABLE:
        # Scatter/residuals for the best regressor (LightGBM) by default
        y_true = np.asarray(y_test)
        y_pred = np.asarray(y_pred_lgb)
        if len(y_true) > 200_000:
            rng = np.random.RandomState(13)
            idx = rng.choice(len(y_true), size=200_000, replace=False)
            y_true = y_true[idx]
            y_pred = y_pred[idx]

        plot_regression_scatter_png(y_true, y_pred, "regression_actual_vs_pred_lightgbm.png")
        residuals = y_true - y_pred
        plot_residuals_png(residuals, "regression_residuals_lightgbm.png")


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def run_clustering(df_clean: pd.DataFrame, plots: bool = False) -> None:
    print("\n=== Clustering: MiniBatchKMeans on distance/time features ===")
    features = [c for c in CLUSTER_FEATURES if c in df_clean.columns]
    if len(features) < 3:
        print("Skipping clustering (missing required features).")
        return

    X_cluster = df_clean[features].dropna()
    if X_cluster.empty:
        print("Skipping clustering (no rows after dropping NA).")
        return

    # Subsample for speed (clustering + metrics + plots)
    if len(X_cluster) > 300_000:
        rng = np.random.RandomState(21)
        idx = rng.choice(len(X_cluster), size=300_000, replace=False)
        X_cluster_sub = X_cluster.iloc[idx]
        df_for_join = df_clean.loc[X_cluster_sub.index]
    else:
        X_cluster_sub = X_cluster
        df_for_join = df_clean.loc[X_cluster_sub.index]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster_sub)

    k = 5  # default; elbow/silhouette can justify in plots
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10_000, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    print(f"Assigned to {k} clusters. Cluster sizes:")
    print(pd.Series(labels).value_counts().sort_index().to_string())

    cluster_profile = pd.DataFrame(X_cluster_sub).assign(cluster=labels).groupby("cluster").mean()
    print("\nCluster profiles (average values):")
    print(cluster_profile.to_string())

    delay_rate = ((df_for_join["arr_delay"] >= ARRIVAL_DELAY_THRESHOLD).astype(int)).groupby(labels).mean()
    avg_delay = df_for_join.groupby(labels)["arr_delay"].mean()
    delay_stats = pd.DataFrame({"delay_rate_ge_15min": delay_rate, "avg_arr_delay_min": avg_delay}).sort_values(
        "delay_rate_ge_15min", ascending=False
    )
    print("\nDelay stats by cluster (sorted by delay rate):")
    print(delay_stats.to_string())

    if plots and MATPLOTLIB_AVAILABLE:
        # Cluster scatter
        plot_cluster_scatter_png(X_cluster_sub, labels, "clusters_scatter.png")

        # Elbow + silhouette across K
        k_range = list(range(2, 11))
        plot_elbow_and_silhouette_png(X_scaled, k_range)


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate models on 2024 flight delay data.")
    parser.add_argument("--sample", action="store_true", help="Use the smaller sample file for a quick run.")
    parser.add_argument("--nrows", type=int, default=None, help="Only read the first N rows (for smoke testing).")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size for streaming the CSV.")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run advanced diagnostics (ROC curve plot, learning curve, complexity curve).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save core result visualizations (confusion matrices, profit curve, feature importance plot, regression plots, clustering plots).",
    )
    parser.add_argument(
        "--ev-costs",
        type=float,
        nargs=4,
        metavar=("C_TP", "C_FP", "C_FN", "C_TN"),
        help="Custom costs for expected-value calculation per case: TP FP FN TN. Defaults used if not provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plots and not MATPLOTLIB_AVAILABLE:
        print("WARNING: matplotlib not available; cannot save plots. Install matplotlib to enable --plots.")
    if args.plots and MATPLOTLIB_AVAILABLE:
        ensure_plots_dir()

    chunks = read_in_chunks(use_sample=args.sample, nrows=args.nrows, chunk_size=args.chunk_size)
    df_clean, X = prepare_features(chunks)

    print(f"Total rows after cleaning: {len(df_clean):,}")

    run_classification(
        df_clean,
        X,
        diagnostics=args.diagnostics,
        ev_costs=tuple(args.ev_costs) if args.ev_costs is not None else None,
        plots=args.plots,
    )
    run_regression(df_clean, X, plots=args.plots)
    run_clustering(df_clean, plots=args.plots)


if __name__ == "__main__":
    main()
