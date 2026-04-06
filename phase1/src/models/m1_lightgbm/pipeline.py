"""
M1 LightGBM — Pipeline (Detection + Forecast)
===============================================
LightGBM with RF-selected engineered features.
Should significantly outperform M0 baseline.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    cohen_kappa_score, precision_recall_curve, auc,
    log_loss, brier_score_loss, roc_auc_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import os
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 42
SCALE_POS_WEIGHT = 5.0

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": SCALE_POS_WEIGHT,
    "verbose": -1,
    "seed": SEED,
    "n_jobs": -1,
    "min_child_samples": 100,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50


def load_data(split: str, task: str):
    """Load pre-prepared feature data."""
    X = pd.read_parquet(RESULTS_DIR / f"{split}_{task}_X.parquet")
    y = pd.read_parquet(RESULTS_DIR / f"{split}_{task}_y.parquet")
    target_col = "target" if task == "detect" else "target_forecast"
    return X, y[target_col].values


def train_lgb(X_train, y_train, X_val, y_val, task_name: str):
    """Train LightGBM with early stopping."""
    print(f"\n{'─'*50}")
    print(f"Training LightGBM — {task_name}")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"  Features: {list(X_train.columns)}")

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(100),
    ]

    start_time = time.time()
    model = lgb.train(
        LGB_PARAMS, train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_data],
        callbacks=callbacks,
    )
    train_time = time.time() - start_time
    print(f"  Training time: {train_time:.1f}s, Best iteration: {model.best_iteration}")

    return model, train_time


def evaluate_detection(model, X_test, y_test, results_dir: Path, prefix: str = "detect"):
    """Evaluate detection model."""
    print(f"\n{'─'*50}")
    print(f"Evaluating Detection Model")

    start = time.time()
    y_prob = model.predict(X_test)
    inference_time_ms = (time.time() - start) / len(X_test) * 1000
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, target_names=["Unoccupied", "Occupied"],
                                   output_dict=True)
    print(classification_report(y_test, y_pred, target_names=["Unoccupied", "Occupied"]))

    metrics = {
        "recall_occupied": report["Occupied"]["recall"],
        "precision_unoccupied": report["Unoccupied"]["precision"],
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "cohen_kappa": cohen_kappa_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "inference_time_ms_per_sample": inference_time_ms,
    }

    print(f"  Key Metrics:")
    print(f"    Recall (Occupied):      {metrics['recall_occupied']:.4f}")
    print(f"    Precision (Unoccupied): {metrics['precision_unoccupied']:.4f}")
    print(f"    Macro F1:               {metrics['macro_f1']:.4f}")
    print(f"    Cohen's Kappa:          {metrics['cohen_kappa']:.4f}")
    print(f"    ROC-AUC:                {metrics['roc_auc']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Unoccupied", "Occupied"],
                yticklabels=["Unoccupied", "Occupied"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"M1 LightGBM — Detection Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance (LightGBM gain)
    importance = model.feature_importance(importance_type="gain")
    feat_names = model.feature_name()
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#E74C3C")
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("M1 LightGBM — Detection Feature Importance (LGB Gain)")
    plt.tight_layout()
    plt.savefig(results_dir / f"{prefix}_feature_importance_lgb.png", dpi=150)
    plt.close()

    return metrics


def evaluate_forecast(model, X_test, y_test, results_dir: Path, prefix: str = "forecast"):
    """Evaluate forecast model with probabilistic metrics."""
    print(f"\n{'─'*50}")
    print(f"Evaluating Forecast Model (+30 min)")

    start = time.time()
    y_prob = model.predict(X_test)
    inference_time_ms = (time.time() - start) / len(X_test) * 1000
    y_pred = (y_prob >= 0.5).astype(int)

    valid = ~np.isnan(y_test)
    y_test_v = y_test[valid].astype(int)
    y_prob_v = y_prob[valid]
    y_pred_v = y_pred[valid]

    brier = brier_score_loss(y_test_v, y_prob_v)
    logloss = log_loss(y_test_v, y_prob_v)
    precision, recall, _ = precision_recall_curve(y_test_v, y_prob_v)
    pr_auc = auc(recall, precision)

    report = classification_report(y_test_v, y_pred_v,
                                   target_names=["Unoccupied", "Occupied"],
                                   output_dict=True)
    print(classification_report(y_test_v, y_pred_v,
                                target_names=["Unoccupied", "Occupied"]))

    metrics = {
        "brier_score": brier,
        "log_loss": logloss,
        "pr_auc": pr_auc,
        "recall_occupied": report["Occupied"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "cohen_kappa": cohen_kappa_score(y_test_v, y_pred_v),
        "inference_time_ms_per_sample": inference_time_ms,
    }

    print(f"  Forecast Metrics:")
    print(f"    Brier Score:  {metrics['brier_score']:.4f}")
    print(f"    Log Loss:     {metrics['log_loss']:.4f}")
    print(f"    PR-AUC:       {metrics['pr_auc']:.4f}")
    print(f"    Recall (Occ): {metrics['recall_occupied']:.4f}")
    print(f"    Macro F1:     {metrics['macro_f1']:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test_v, y_pred_v)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=["Unoccupied", "Occupied"],
                yticklabels=["Unoccupied", "Occupied"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"M1 LightGBM — Forecast (+30min) Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    return metrics


def main():
    print("=" * 60)
    print("M1 LightGBM — Pipeline (RF-Selected Features)")
    print("=" * 60)

    # Load feature selection metadata
    with open(RESULTS_DIR / "feature_selection.json") as f:
        meta = json.load(f)
    print(f"  Detection features ({meta['n_detect']}): {meta['detect_features']}")
    print(f"  Forecast features ({meta['n_forecast']}): {meta['forecast_features']}")

    # ── Detection ──────────────────────────────────────────────
    X_train_d, y_train_d = load_data("train", "detect")
    X_val_d, y_val_d = load_data("val", "detect")
    X_test_d, y_test_d = load_data("test", "detect")

    model_detect, train_time_d = train_lgb(
        X_train_d, y_train_d, X_val_d, y_val_d, "Detection"
    )
    detect_metrics = evaluate_detection(model_detect, X_test_d, y_test_d, RESULTS_DIR)
    detect_metrics["train_time_s"] = train_time_d
    model_detect.save_model(str(RESULTS_DIR / "model_detect.txt"))

    # ── Forecast ───────────────────────────────────────────────
    X_train_f, y_train_f = load_data("train", "forecast")
    X_val_f, y_val_f = load_data("val", "forecast")
    X_test_f, y_test_f = load_data("test", "forecast")

    model_forecast, train_time_f = train_lgb(
        X_train_f, y_train_f, X_val_f, y_val_f, "Forecast (+30min)"
    )
    forecast_metrics = evaluate_forecast(model_forecast, X_test_f, y_test_f, RESULTS_DIR)
    forecast_metrics["train_time_s"] = train_time_f
    model_forecast.save_model(str(RESULTS_DIR / "model_forecast.txt"))

    # ── Save all metrics ───────────────────────────────────────
    detect_size = os.path.getsize(RESULTS_DIR / "model_detect.txt") / 1024 / 1024
    forecast_size = os.path.getsize(RESULTS_DIR / "model_forecast.txt") / 1024 / 1024

    all_metrics = {
        "model": "M1_lightgbm_rf",
        "detection": detect_metrics,
        "detection_model_size_MB": detect_size,
        "forecast": forecast_metrics,
        "forecast_model_size_MB": forecast_size,
        "features_detect": meta["detect_features"],
        "features_forecast": meta["forecast_features"],
        "n_features_detect": meta["n_detect"],
        "n_features_forecast": meta["n_forecast"],
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"M1 LightGBM COMPLETE")
    print(f"  Detection: F1={detect_metrics['macro_f1']:.4f}, "
          f"Recall={detect_metrics['recall_occupied']:.4f}")
    print(f"  Forecast:  Brier={forecast_metrics['brier_score']:.4f}, "
          f"PR-AUC={forecast_metrics['pr_auc']:.4f}")
    print(f"  Models saved to: {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
