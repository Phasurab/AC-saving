"""
M0 Baseline — Pipeline (Detection + Forecast)
===============================================
LightGBM with raw sensor features only.
Establishes the accuracy floor for all other models to beat.
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

# Asymmetric cost: FN (miss occupied) = 5x worse than FP (waste energy)
# With balanced classes (~50/50), scale_pos_weight handles the asymmetry
SCALE_POS_WEIGHT = 5.0

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "scale_pos_weight": SCALE_POS_WEIGHT,
    "verbose": -1,
    "seed": SEED,
    "n_jobs": -1,
}
NUM_BOOST_ROUND = 500
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
    print(f"  Pos weight: {SCALE_POS_WEIGHT}x (asymmetric cost)")

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
    """Evaluate detection model with full metrics suite."""
    print(f"\n{'─'*50}")
    print(f"Evaluating Detection Model")

    # Predict
    start = time.time()
    y_prob = model.predict(X_test)
    inference_time_ms = (time.time() - start) / len(X_test) * 1000

    y_pred = (y_prob >= 0.5).astype(int)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=["Unoccupied", "Occupied"],
                                   output_dict=True)
    print(classification_report(y_test, y_pred, target_names=["Unoccupied", "Occupied"]))

    # Key metrics
    metrics = {
        "recall_occupied": report["Occupied"]["recall"],
        "precision_unoccupied": report["Unoccupied"]["precision"],
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "cohen_kappa": cohen_kappa_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "inference_time_ms_per_sample": inference_time_ms,
    }

    print(f"\n  Key Metrics:")
    print(f"    Recall (Occupied):      {metrics['recall_occupied']:.4f} (target ≥ 0.98)")
    print(f"    Precision (Unoccupied): {metrics['precision_unoccupied']:.4f} (target ≥ 0.90)")
    print(f"    Macro F1:               {metrics['macro_f1']:.4f} (target ≥ 0.90)")
    print(f"    Cohen's Kappa:          {metrics['cohen_kappa']:.4f} (target ≥ 0.80)")
    print(f"    ROC-AUC:                {metrics['roc_auc']:.4f}")
    print(f"    Inference:              {metrics['inference_time_ms_per_sample']:.4f} ms/sample")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Unoccupied", "Occupied"],
                yticklabels=["Unoccupied", "Occupied"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"M0 Baseline — Detection Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_names = model.feature_name()
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(imp_df["feature"], imp_df["importance"], color="#4A90D9")
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("M0 Baseline — Detection Feature Importance")
    plt.tight_layout()
    plt.savefig(results_dir / f"{prefix}_feature_importance.png", dpi=150)
    plt.close()

    return metrics


def evaluate_forecast(model, X_test, y_test, results_dir: Path, prefix: str = "forecast"):
    """Evaluate forecast model with probabilistic metrics."""
    print(f"\n{'─'*50}")
    print(f"Evaluating Forecast Model (+30 min)")

    # Predict probabilities
    start = time.time()
    y_prob = model.predict(X_test)
    inference_time_ms = (time.time() - start) / len(X_test) * 1000

    y_pred = (y_prob >= 0.5).astype(int)

    # Valid indices (no NaN in test target)
    valid = ~np.isnan(y_test)
    y_test_v = y_test[valid].astype(int)
    y_prob_v = y_prob[valid]
    y_pred_v = y_pred[valid]

    # Core forecast metrics
    brier = brier_score_loss(y_test_v, y_prob_v)
    logloss = log_loss(y_test_v, y_prob_v)
    precision, recall, _ = precision_recall_curve(y_test_v, y_prob_v)
    pr_auc = auc(recall, precision)

    # Classification metrics for forecast
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

    print(f"\n  Forecast Metrics:")
    print(f"    Brier Score:  {metrics['brier_score']:.4f} (target ≤ 0.15)")
    print(f"    Log Loss:     {metrics['log_loss']:.4f} (target ≤ 0.40)")
    print(f"    PR-AUC:       {metrics['pr_auc']:.4f} (target ≥ 0.85)")
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
    ax.set_title(f"M0 Baseline — Forecast (+30min) Confusion Matrix")
    plt.tight_layout()
    plt.savefig(results_dir / f"{prefix}_confusion_matrix.png", dpi=150)
    plt.close()

    return metrics


def main():
    print("=" * 60)
    print("M0 BASELINE — LightGBM Pipeline (Raw Sensors Only)")
    print("=" * 60)

    # ── Detection ──────────────────────────────────────────────
    X_train_d, y_train_d = load_data("train", "detect")
    X_val_d, y_val_d = load_data("val", "detect")
    X_test_d, y_test_d = load_data("test", "detect")

    model_detect, train_time_d = train_lgb(
        X_train_d, y_train_d, X_val_d, y_val_d, "Detection"
    )
    detect_metrics = evaluate_detection(model_detect, X_test_d, y_test_d, RESULTS_DIR)
    detect_metrics["train_time_s"] = train_time_d

    # Save detection model
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

    # Save forecast model
    model_forecast.save_model(str(RESULTS_DIR / "model_forecast.txt"))

    # ── Save all metrics ───────────────────────────────────────
    # Model size
    detect_size = os.path.getsize(RESULTS_DIR / "model_detect.txt") / 1024 / 1024
    forecast_size = os.path.getsize(RESULTS_DIR / "model_forecast.txt") / 1024 / 1024

    all_metrics = {
        "model": "M0_baseline",
        "detection": detect_metrics,
        "detection_model_size_MB": detect_size,
        "forecast": forecast_metrics,
        "forecast_model_size_MB": forecast_size,
        "features_detect": list(X_train_d.columns),
        "features_forecast": list(X_train_f.columns),
        "n_features_detect": len(X_train_d.columns),
        "n_features_forecast": len(X_train_f.columns),
    }

    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"M0 BASELINE COMPLETE")
    print(f"  Detection model size: {detect_size:.2f} MB")
    print(f"  Forecast model size:  {forecast_size:.2f} MB")
    print(f"  Results saved to: {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
